#include "../include/cmt/cmt.h"
#include "../include/cmt/command_buf.h"
#include "../include/cmt/command_enc_compute.h"
#include "../include/cmt/command_queue.h"
#include "../include/cmt/compute/compute-pipeline.h"
#include "../include/cmt/device.h"
#include "../include/cmt/kernels/library.h"
#include "../include/cmt/memory/buffer.h"
#include "../include/cmt/rendering/pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *matrixAdditionShader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matrix_addition(const device float* A [[buffer(0)]]),\n"
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
  Matrix mat;
  mat.rows = rows;
  mat.cols = cols;
  mat.data = malloc(rows * sizeof(float *));

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

  // Create a command queue
  MtCommandQueue *cmdQueue = mtNewCommandQueue(device);
  if (!cmdQueue) {
    printf("Failed to create command queue\n");
    return -1;
  }

  NsError *error = NULL;

  MtLibrary *lib =
      mtNewLibraryWithSource(device, matrixAdditionShader, NULL, &error);
  if (!lib) {
    printf("Failed to create library from source\n");
    return -1;
  }

  MtFunction *addFunc = mtNewFunctionWithName(lib, "matrix_addition");
  if (!addFunc) {
    printf("Failed to retrieve function from library\n");
    return -1;
  }

  MtComputePipelineState *pipelineState =
      mtNewComputePipelineStateWithFunction(device, addFunc, &error);
  if (!pipelineState) {
    printf("Failed to create compute pipeline state\n");
    return -1;
  }

  Matrix mat1 = createMatrix(2, 2);
  Matrix mat2 = createMatrix(2, 2);
  Matrix result = createMatrix(2, 2);

  // Example data
  *mat1.data[0] = 1.0f;
  *mat1.data[1] = 2.0f;
  *mat1.data[2] = 3.0f;
  *mat1.data[3] = 4.0f;

  *mat2.data[0] = 5.0f;
  *mat2.data[1] = 6.0f;
  *mat2.data[2] = 7.0f;
  *mat2.data[3] = 8.0f;

  MtBuffer *bufferA = mtDeviceNewBufferWithLength(device, sizeof(float) * 4,
                                                  MtResourceStorageModeShared);
  MtBuffer *bufferB = mtDeviceNewBufferWithLength(device, sizeof(float) * 4,
                                                  MtResourceStorageModeShared);
  MtBuffer *bufferC = mtDeviceNewBufferWithLength(device, sizeof(float) * 4,
                                                  MtResourceStorageModeShared);

  memcpy(mtBufferContents(bufferA), mat1.data, sizeof(float) * 4);
  memcpy(mtBufferContents(bufferB), mat2.data, sizeof(float) * 4);

  MtCommandBuffer *commandBuffer = mtNewCommandBuffer(cmdQueue);
  MtComputeCommandEncoder *encoder = mtNewComputeCommandEncoder(commandBuffer);
  mtComputeCommandEncoderSetComputePipelineState(encoder, pipelineState);
  mtComputeCommandEncoderSetBufferOffsetAtIndex(encoder, bufferA, 0, 0);
  mtComputeCommandEncoderSetBufferOffsetAtIndex(encoder, bufferB, 1, 0);
  mtComputeCommandEncoderSetBufferOffsetAtIndex(encoder, bufferC, 2, 0);

  mtComputeCommandEncoderDispatchThreadgroups_threadsPerThreadgroup(
      encoder, (MtSize){.width = 4, .height = 1, .depth = 1},
      (MtSize){.width = 1, .height = 1, .depth = 1});

  mtComputeCommandEncoderEndEncoding(encoder);
  mtCommandBufferCommit(commandBuffer);
  mtCommandBufferWaitUntilCompleted(commandBuffer);

  memcpy(result.data, mtBufferContents(bufferC), sizeof(float) * 4);

  // Print result
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%f ", result.data[i * 2 + j]);
    }
    printf("\n");
  }

  freeMatrix(mat1);
  freeMatrix(mat2);
  freeMatrix(result);
  mtRelease(bufferA);
  mtRelease(bufferB);
  mtRelease(bufferC);
  mtRelease(commandBuffer);
  mtRelease(pipelineState);
  mtRelease(addFunc);
  mtRelease(lib);
  mtRelease(cmdQueue);
  mtRelease(device);

  return 0;
}