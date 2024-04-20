#include "../include/cmt/cmt.h"
#include "../include/cmt/command_buf.h"
#include "../include/cmt/command_enc_compute.h"
#include "../include/cmt/command_queue.h"
#include "../include/cmt/compute/compute-pipeline.h"
#include "../include/cmt/device.h"
#include "../include/cmt/error_handling.h"
#include "../include/cmt/kernels/library.h"
#include "../include/cmt/memory/buffer.h"
#include "../include/cmt/rendering/pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *matrixAdditionShader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matrix_addition(const device float* A [[buffer(0)]],\n"
    "                            const device float* B [[buffer(1)]],\n"
    "                            device float* C [[buffer(2)]],\n"
    "                            uint id [[thread_position_in_grid]]) {\n"
    "   C[id] = A[id] + B[id];\n"
    "}";

typedef struct {
  int rows;
  int cols;
  float **data;
} Matrix;

Matrix createMatrix(int rows, int cols) {
    Matrix mat = {rows, cols, malloc(rows * sizeof(float*))};
    if (!mat.data) {
        printf("Failed to allocate memory for data pointers.\n");
        return mat;
    }

    for (int i = 0; i < rows; i++) {
        mat.data[i] = malloc(cols * sizeof(float));
        if (!mat.data[i]) {
            printf("Failed to allocate memory for row %d.\n", i);
            for (int j = 0; j < i; j++) {
                free(mat.data[j]);
            }
            free(mat.data);
            mat.data = NULL;
            return mat;
        }
    }
    return mat;
}

void freeMatrix(Matrix mat) {
  for (int i = 0; i < mat.rows; i++) {
    free(mat.data[i]);
  }

  free(mat.data);
}

int main() {
  MtDevice *device = mtCreateSystemDefaultDevice();
  if (!device) {
    printf("Failed to create Metal device\n");
    return -1;
  }

  MtCommandQueue *cmdQueue = mtNewCommandQueue(device);
  if (!cmdQueue) {
    printf("Failed to create command queue\n");
    return -1;
  }

  NsError *error = NULL;
  MtLibrary *lib = mtNewLibraryWithSource(device, (char *)matrixAdditionShader, NULL, &error);
  if (!lib) {
    printf("Failed to create library\n");
    if (error) {
      printNSError(error);
    }
    return -1;
  }

  MtFunction *addFunc = mtNewFunctionWithName(lib, "matrix_addition");
  if (!addFunc) {
    printf("Failed to retrieve function from library\n");
    return -1;
  }

  MtComputePipelineState *pipelineState = mtNewComputePipelineStateWithFunction(device, addFunc, error);
  if (!pipelineState) {
    printf("Failed to create compute pipeline state\n");
    if (error) {
      printNSError(error);
    }
    return -1;
  }

  Matrix mat1 = createMatrix(2, 2);
  Matrix mat2 = createMatrix(2, 2);
  Matrix result = createMatrix(2, 2);

  if (!mat1.data || !mat2.data || !result.data) {
    printf("Failed to allocate matrices\n");
    return -1;
  }

  mat1.data[0][0] = 1.0f; mat1.data[0][1] = 2.0f;
  mat2.data[0][0] = 5.0f; mat2.data[0][1] = 6.0f;

  MtBuffer *bufferA = mtDeviceNewBufferWithLength(device, sizeof(float) * 4, MtResourceStorageModeShared);
  MtBuffer *bufferB = mtDeviceNewBufferWithLength(device, sizeof(float) * 4, MtResourceStorageModeShared);
  MtBuffer *bufferC = mtDeviceNewBufferWithLength(device, sizeof(float) * 4, MtResourceStorageModeShared);

  if (!bufferA || !bufferB || !bufferC) {
    printf("Failed to create buffers\n");
    return -1;
  }
  
  memcpy(mtBufferContents(bufferA), mat1.data[0], sizeof(float) * 4);
  memcpy(mtBufferContents(bufferB), mat2.data[0], sizeof(float) * 4);

  MtCommandBuffer *cmdBuffer = mtCommandQueueCommandBuffer(cmdQueue);
  MtCommandEncoder *computeEncoder = mtCommandBufferComputeEncoder(cmdBuffer);

  mtSetComputePipelineState(computeEncoder, pipelineState);
  mtSetBuffer(computeEncoder, bufferA, 0, 0);
  mtSetBuffer(computeEncoder, bufferB, 0, 1);
  mtSetBuffer(computeEncoder, bufferC, 0, 2);

  mtDispatchThreads(computeEncoder, (MtSize){2, 2, 1}, (MtSize){1, 1, 1});

  mtEndEncoding(computeEncoder);
  mtCommitCommandBuffer(cmdBuffer);
  mtWaitUntilCompleted(cmdBuffer);

  memcpy(result.data[0], mtBufferContents(bufferC), sizeof(float) * 4);

  printf("Result Matrix:\n");
  for (int i = 0; i < result.rows; i++) {
    for (int j = 0; j < result.cols; j++) {
      printf("%f ", result.data[i][j]);
    }
    printf("\n");
  }

  freeMatrix(mat1);
  freeMatrix(mat2);
  freeMatrix(result);
  mtReleaseBuffer(bufferA);
  mtReleaseBuffer(bufferB);
  mtReleaseBuffer(bufferC);
  mtReleaseCommandBuffer(cmdBuffer);
  mtReleaseCommandEncoder(computeEncoder);
  mtReleaseComputePipelineState(pipelineState);
  mtReleaseFunction(addFunc);
  mtReleaseLibrary(lib);
  mtReleaseCommandQueue(cmdQueue);
  mtReleaseDevice(device);

  return 0;
}
