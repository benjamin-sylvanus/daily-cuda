#include "common.h"
#include "cuda_common.cuh"
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

int main() {
  int size = 10000;
  int block_size = 128;

  cudaError error;

  int NO_BYTES = size * sizeof(int);
  // create host pointers
  int *hostA, hostB, hostResults, hostC;

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
  clock_t cpu_start, cpu_end;

  cpu_start = clock();
  sum_array_cpu(hostA, hostB, hostC, size);
  cpu_end = clock();





  // device pointers
  int *deviceA, deviceB, deviceC;
  gpuErrchk(cudaMalloc((int **)&deviceA, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&deviceB, NO_BYTES));
  gpuErrchk(cudaMalloc((int **)&deviceC, NO_BYTES));

  // memory transfer host -> device
  clock_t host_to_device_start, host_to_device_end;
  host_to_device_start = clock();
  cudaMemcpy(deviceA, hostA, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, NO_BYTES, cudaMemcpyHostToDevice);
  host_to_device_end = clock();

  clock_t gpu_start,gpu_end;

  // kernel launch parameters
  dim3 block(block_size);
  dim3 grid((size / block.x) + 1);

  gpu_start = clock();
  sum_array_gpu<<<grid, block>>>(deviceA, deviceB, deviceC, size);

  // Wait for results
  cudaDeviceSynchronize();
  gpu_end = clock();

  // Copy memory back to host
  clock_t device_to_host_start, device_to_host_end;
  device_to_host_start=clock();
  cudaMemcpy(hostResults, deviceC, NO_BYTES, cudaMemcpyDeviceToHost);
  device_to_host_end = clock();

  // array comparision
  compare_arrays(hostC, hostResults);

  printf("Sum Array CPU Execution Time : %4.6f \n",(double)((double)(cpu_end-cpu_start)/CLOCKS_PER_SECOND));
  printf("Sum Array GPU Execution Time : %4.6f \n",(double)((double)(gpu_end-gpu_start)/CLOCKS_PER_SECOND));
  printf("Host -> Device Memory Transfer Time : %4.6f \n",(double)((double)(host_to_device_end-host_to_device_start)/CLOCKS_PER_SECOND));
  printf("Device -> Host Memory Transfer Time : %4.6f \n",(double)((double)(device_to_host_end-device_to_host_start)/CLOCKS_PER_SECOND));
  printf("Sum Array GPU Total Execution Time : %4.6f \n",(double)((double)(device_to_host_end-host_to_device_start)/CLOCKS_PER_SECOND));
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