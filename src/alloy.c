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
#include <time.h>

#define CHECK_ERROR(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "Error: %s\n", message);                                 \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;

typedef enum { MATRIX_OP_ADD, MATRIX_OP_MULTIPLY } MatrixOperation;

Matrix createMatrix(size_t rows, size_t cols);
void freeMatrix(Matrix *mat);
void fillMatrixRandom(Matrix *mat);
void printMatrix(const Matrix *mat);
MtBuffer *createBuffer(MtDevice *device, size_t size,
                       MtResourceOptions options);
MtLibrary *createLibraryFromFile(MtDevice *device, const char *filename);
int performMatrixOperation(MtDevice *device, MtCommandQueue *cmdQueue,
                           Matrix *a, Matrix *b, Matrix *result,
                           MatrixOperation op);

const char *matrixAdditionShader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matrix_addition(const device float* A [[buffer(0)]],\n"
    "                            const device float* B [[buffer(1)]],\n"
    "                            device float* C [[buffer(2)]],\n"
    "                            uint2 id [[thread_position_in_grid]],\n"
    "                            uint2 gridSize [[threads_per_grid]]) {\n"
    "   if (id.x < gridSize.x && id.y < gridSize.y) {\n"
    "       unsigned int index = id.y * gridSize.x + id.x;\n"
    "       C[index] = A[index] + B[index];\n"
    "   }\n"
    "}";

const char *matrixMultiplicationShader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matrix_multiplication(const device float* A [[buffer(0)]],\n"
    "                                  const device float* B [[buffer(1)]],\n"
    "                                  device float* C [[buffer(2)]],\n"
    "                                  constant uint& M [[buffer(3)]],\n"
    "                                  constant uint& N [[buffer(4)]],\n"
    "                                  constant uint& K [[buffer(5)]],\n"
    "                                  uint2 id [[thread_position_in_grid]]) "
    "{\n"
    "   if (id.x >= N || id.y >= M) return;\n"
    "   float sum = 0.0f;\n"
    "   for (uint i = 0; i < K; ++i) {\n"
    "       sum += A[id.y * K + i] * B[i * N + id.x];\n"
    "   }\n"
    "   C[id.y * N + id.x] = sum;\n"
    "}";

int main() {
  MtDevice *device = NULL;
  MtCommandQueue *cmdQueue = NULL;
  Matrix a, b, result;
  int status = -1;

  srand(time(NULL));

  device = mtCreateSystemDefaultDevice();
  CHECK_ERROR(device, "Failed to create Metal device");

  cmdQueue = mtNewCommandQueue(device);
  CHECK_ERROR(cmdQueue, "Failed to create command queue");

  // test matrix addition
  a = createMatrix(1024, 1024);
  b = createMatrix(1024, 1024);
  result = createMatrix(1024, 1024);
  CHECK_ERROR(a.data && b.data && result.data, "Failed to create matrices");

  fillMatrixRandom(&a);
  fillMatrixRandom(&b);

  printf("Performing matrix addition...\n");
  status =
      performMatrixOperation(device, cmdQueue, &a, &b, &result, MATRIX_OP_ADD);
  CHECK_ERROR(status == 0, "Matrix addition failed");

  // test matrix multiplication
  freeMatrix(&b);
  freeMatrix(&result);
  b = createMatrix(1024, 512);
  result = createMatrix(1024, 512);
  CHECK_ERROR(b.data && result.data,
              "Failed to create matrices for multiplication");

  fillMatrixRandom(&b);

  printf("Performing matrix multiplication...\n");
  status = performMatrixOperation(device, cmdQueue, &a, &b, &result,
                                  MATRIX_OP_MULTIPLY);
  CHECK_ERROR(status == 0, "Matrix multiplication failed");

  printf("Operations completed successfully.\n");
  status = 0;

cleanup:
  freeMatrix(&a);
  freeMatrix(&b);
  freeMatrix(&result);
  if (cmdQueue)
    mtReleaseCommandQueue(cmdQueue);
  if (device)
    mtReleaseDevice(device);

  return status;
}

Matrix createMatrix(size_t rows, size_t cols) {
  Matrix mat = {
      .rows = rows, .cols = cols, .data = malloc(rows * cols * sizeof(float))};
  if (!mat.data) {
    fprintf(stderr, "Failed to allocate memory for matrix.\n");
  }
  return mat;
}

void freeMatrix(Matrix *mat) {
  if (mat && mat->data) {
    free(mat->data);
    mat->data = NULL;
  }
}

void fillMatrixRandom(Matrix *mat) {
  for (size_t i = 0; i < mat->rows * mat->cols; ++i) {
    mat->data[i] = (float)rand() / RAND_MAX;
  }
}

void printMatrix(const Matrix *mat) {
  for (size_t i = 0; i < mat->rows; ++i) {
    for (size_t j = 0; j < mat->cols; ++j) {
      printf("%f ", mat->data[i * mat->cols + j]);
    }
    printf("\n");
  }
}

MtBuffer *createBuffer(MtDevice *device, size_t size,
                       MtResourceOptions options) {
  return mtDeviceNewBufferWithLength(device, size, options);
}

MtLibrary *createLibraryFromFile(MtDevice *device, const char *filename) {
  // In a real implementation, read the shader from a file
  // For this example, we'll use the hardcoded shader strings
  NsError *error = NULL;
  MtLibrary *lib = NULL;

  if (strcmp(filename, "addition.metal") == 0) {
    lib = mtNewLibraryWithSource(device, (char *)matrixAdditionShader, NULL,
                                 &error);
  } else if (strcmp(filename, "multiplication.metal") == 0) {
    lib = mtNewLibraryWithSource(device, (char *)matrixMultiplicationShader,
                                 NULL, &error);
  }

  if (!lib && error) {
    printNSError(error);
  }

  return lib;
}

int performMatrixOperation(MtDevice *device, MtCommandQueue *cmdQueue,
                           Matrix *a, Matrix *b, Matrix *result,
                           MatrixOperation op) {
  MtLibrary *lib = NULL;
  MtFunction *func = NULL;
  MtComputePipelineState *pipelineState = NULL;
  MtCommandBuffer *cmdBuffer = NULL;
  MtCommandEncoder *computeEncoder = NULL;
  MtBuffer *bufferA = NULL, *bufferB = NULL, *bufferC = NULL;
  NsError *error = NULL;
  int status = -1;

  const char *shaderFile =
      (op == MATRIX_OP_ADD) ? "addition.metal" : "multiplication.metal";
  const char *funcName =
      (op == MATRIX_OP_ADD) ? "matrix_addition" : "matrix_multiplication";

  lib = createLibraryFromFile(device, shaderFile);
  CHECK_ERROR(lib, "Failed to create library");

  func = mtNewFunctionWithName(lib, funcName);
  CHECK_ERROR(func, "Failed to create function");

  pipelineState = mtNewComputePipelineStateWithFunction(device, func, error);
  CHECK_ERROR(pipelineState, "Failed to create compute pipeline state");

  size_t bufferSize = a->rows * a->cols * sizeof(float);
  bufferA = createBuffer(device, bufferSize, MtResourceStorageModeShared);
  bufferB = createBuffer(device, bufferSize, MtResourceStorageModeShared);
  bufferC = createBuffer(device, bufferSize, MtResourceStorageModeShared);
  CHECK_ERROR(bufferA && bufferB && bufferC, "Failed to create buffers");

  memcpy(mtBufferContents(bufferA), a->data, bufferSize);
  memcpy(mtBufferContents(bufferB), b->data, bufferSize);

  cmdBuffer = mtCommandQueueCommandBuffer(cmdQueue);
  computeEncoder = mtCommandBufferComputeEncoder(cmdBuffer);
  CHECK_ERROR(cmdBuffer && computeEncoder,
              "Failed to create command buffer or encoder");

  mtSetComputePipelineState(computeEncoder, pipelineState);
  mtSetBuffer(computeEncoder, bufferA, 0, 0);
  mtSetBuffer(computeEncoder, bufferB, 0, 1);
  mtSetBuffer(computeEncoder, bufferC, 0, 2);

  if (op == MATRIX_OP_MULTIPLY) {
    MtBuffer *bufferM =
        createBuffer(device, sizeof(uint32_t), MtResourceStorageModeShared);
    MtBuffer *bufferN =
        createBuffer(device, sizeof(uint32_t), MtResourceStorageModeShared);
    MtBuffer *bufferK =
        createBuffer(device, sizeof(uint32_t), MtResourceStorageModeShared);
    CHECK_ERROR(bufferM && bufferN && bufferK,
                "Failed to create dimension buffers");

    uint32_t M = a->rows, N = b->cols, K = a->cols;
    memcpy(mtBufferContents(bufferM), &M, sizeof(uint32_t));
    memcpy(mtBufferContents(bufferN), &N, sizeof(uint32_t));
    memcpy(mtBufferContents(bufferK), &K, sizeof(uint32_t));

    mtSetBuffer(computeEncoder, bufferM, 0, 3);
    mtSetBuffer(computeEncoder, bufferN, 0, 4);
    mtSetBuffer(computeEncoder, bufferK, 0, 5);
  }

  MtSize gridSize = {a->cols, a->rows, 1};
  MtSize threadGroupSize = {16, 16, 1}; // Adjust based on your GPU capabilities
  mtDispatchThreads(computeEncoder, gridSize, threadGroupSize);

  mtEndEncoding(computeEncoder);
  mtCommitCommandBuffer(cmdBuffer);
  mtWaitUntilCompleted(cmdBuffer);

  memcpy(result->data, mtBufferContents(bufferC), bufferSize);

  status = 0;

cleanup:
  if (bufferA)
    mtReleaseBuffer(bufferA);
  if (bufferB)
    mtReleaseBuffer(bufferB);
  if (bufferC)
    mtReleaseBuffer(bufferC);
  if (cmdBuffer)
    mtReleaseCommandBuffer(cmdBuffer);
  if (computeEncoder)
    mtReleaseCommandEncoder(computeEncoder);
  if (pipelineState)
    mtReleaseComputePipelineState(pipelineState);
  if (func)
    mtReleaseFunction(func);
  if (lib)
    mtReleaseLibrary(lib);

  return status;
}
