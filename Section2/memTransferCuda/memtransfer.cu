#include "cstring"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"

__global__ void mem_trs_test(int *input) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("tid:%d , gid:%d , value:%d \n ", threadIdx.x, gid, input[gid]);
}

__global__ void mem_trs_test2(int *input, int size) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    printf("tid : %d , gid : %d , value : %d \n ", threadIdx.x, gid,
           input[gid]);
}

int main() {
  int size = 150;
  int byte_size = size * sizeof(int);

  int *h_input;
  h_input = (int *)malloc(byte_size);

  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++) {
    h_input[i] = (int)(rand() & 0xff);
  }

  int *d_input;
  cudaMalloc((void **)&d_input, byte_size);

  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

  // init block and grid
  dim3 block(32);
  dim3 grid(5);

  mem_trs_test2<<<grid, block>>>(d_input, size);

  // Wait for gpu to finish
  cudaDeviceSynchronize();

  // Free device memory
  cuda_Free(d_input);

  // Free host memory
  free(h_input);

  cudaDeviceReset();
  return 0;
}