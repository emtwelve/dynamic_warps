/* This test is a proof of concept that our idea works */

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "lib/CycleTimer.h"


#define N 10000000

// If we modify the program
#define NO_OPT 0

extern float toBW(int bytes, float sec);

/// GPU CODE FOR THE REGULAR RUN ///
////////////////////////////////////

// Everything with (arg1 = 0)  = return 0;
__device__ int
helper_func(int arg1, int arg2)
{
  int retval = 0;
  for (int i = 0; i < 100; i++) {
    retval += arg2;
    retval *= 2;
    if (i%5)
      retval /= 2;
  }
  retval *= arg1;
  return retval;
}

__global__ void
inner_kernel(int* input1, int* input2, int* output)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N)
    output[idx] = helper_func(input1[idx], input2[idx]);
}

////////////////////////////////////
////////////////////////////////////


/// GPU CODE FOR THE OPTIMIZED RUN ///
//////////////////////////////////////

// Everything with (arg1 = 0)  = return 0;

__device__ int
opt_one() {
  return 0;
}

__device__ int
opt_two(int arg1, int arg2) {
  int retval = 0;
  for (int i = 0; i < 100; i++) {
    retval += arg2;
    retval *= 2;
    if (i%5)
      retval /= 2;
  }
  retval *= arg1;
  return retval;
}

__device__ int
helper_func_opt(int arg1, int arg2)
{
  if (arg1 == 0) {
    return opt_one();
  }
  return opt_two(arg1, arg2);
}

__global__ void
inner_kernel_opt(int* input1, int* input2, int* output, int* warp_remap)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int ridx = warp_remap[idx];

  if (idx < N)
    output[idx] = helper_func_opt(input1[ridx], input2[ridx]);
}

//////////////////////////////////////
//////////////////////////////////////


void test(int* input_1, int* input_2, int* warp_map) {

  int totalBytes = sizeof(int) * N;

  // Number of blocks and threads per block
  const int threadsPerBlock = 512;
  const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  int* device_in_one;
  int* device_in_two;
  int* device_out;
  int* device_warp_map;

  cudaMalloc((void**) &device_in_one, N * sizeof(int));
  cudaMalloc((void**) &device_in_two, N * sizeof(int));
  cudaMalloc((void**) &device_out, N * sizeof(int));
  cudaMalloc((void**) &device_warp_map, N * sizeof(int));

  // Copy over my generated array
  cudaMemcpy(device_in_one, input_1, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_in_two, input_2, N * sizeof(int), cudaMemcpyHostToDevice);
  // Copy over my generated warp map
  cudaMemcpy(device_warp_map, warp_map, N * sizeof(int), cudaMemcpyHostToDevice);

  double kernelStartTime = CycleTimer::currentSeconds();

  // Run the Kernel on the GPU
  if (NO_OPT) {
    inner_kernel<<<blocks, threadsPerBlock>>>(device_in_one, device_in_two, device_out);
  } else {
    inner_kernel_opt<<<blocks, threadsPerBlock>>>(device_in_one, device_in_two, device_out, device_warp_map);
  }
  cudaThreadSynchronize();

  double kernelEndTime = CycleTimer::currentSeconds();

  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "WARNING: A CUDA ERROR OCCURED=%d, %s\n", errCode, cudaGetErrorString(errCode));
  }

  double kernelDuration = kernelEndTime - kernelStartTime;
  printf("Kernel time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));

  cudaFree(device_warp_map); cudaFree(device_in_one); cudaFree(device_in_two); cudaFree(device_out);
}