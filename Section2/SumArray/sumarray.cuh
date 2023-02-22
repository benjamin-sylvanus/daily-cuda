#include "../../common.h"
#include "../../cuda_common.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void sum_array_gpu(int *a, int *b, int *c, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}

void sum_array_cpu(int *a, int *b, int *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

int sumarrayexample() {
  int size = 10000;
  int block_size = 128;
  int NO_BYTES = size * sizeof(int);
  // create host pointers
  int * hostA;
  int * hostB;
  int * hostResults;
  int * hostC;

  // allocate memory for host pointers
  hostA = (int *)malloc(NO_BYTES);
  hostB = (int *)malloc(NO_BYTES);
  hostResults = (int *)malloc(NO_BYTES);
  hostC = (int *)malloc(NO_BYTES);

  // init host pointers
  time_t t;

  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    hostA[i] = (int)(rand() & 0xFF);
  }
  for (int i = 0; i < size; i++) {
    hostB[i] = (int)(rand() & 0xFF);
  }

  // cpu sum array
  sum_array_cpu(hostA, hostB, hostC, size);

  // device pointers
  int *deviceA;
  int * deviceB;
  int * deviceC;
  cudaMalloc((int **)&deviceA, NO_BYTES);
  cudaMalloc((int **)&deviceB, NO_BYTES);
  cudaMalloc((int **)&deviceC, NO_BYTES);

  // memory transfer host -> device
  cudaMemcpy(deviceA, hostA, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, NO_BYTES, cudaMemcpyHostToDevice);

  // kernel launch parameters
  dim3 block(block_size);
  dim3 grid((size / block.x) + 1);

  sum_array_gpu<<<grid, block>>>(deviceA, deviceB, deviceC, size);

  // Wait for results
  cudaDeviceSynchronize();

  // Copy memory back to host
  cudaMemcpy(hostResults, deviceC, NO_BYTES, cudaMemcpyDeviceToHost);

  // array comparision
  compare_arrays(hostC, hostResults,size);

  // free device memory
  cudaFree(deviceC);
  cudaFree(deviceB);
  cudaFree(deviceA);

  // free host memory
  free(hostResults);
  free(hostB);
  free(hostA);

  return EXIT_SUCCESS;
}
