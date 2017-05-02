/* Norman Ponte; Joey Fernau
 * annotation generation test
 */

#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../../lib/CycleTimer.h"

void mainCuda(int N, float* resultarray);

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

__device__ int test ( bool x , int y , int z ) {
  printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 2);
  int result = 0;
  if (x) {
    printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 3);
    for (int i = 0; i < 10000; i++)
      result += y - z;
  } else {
    printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 4);
    for (int i = 0; i < 10000; i++)
      result += y - z;
  }
  printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 5);
  return result;
}

__global__ void
test_kernel(int N, float* result) {
    printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 0);
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
       printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 1);
       result[index] = test(index % 2 == 0, index % 13, index % 7);
    }
    printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, 6);
}

int main(int argc, char** argv)
{
  int N = 64;

  float* resultarray = new float[N];

  printf("tid,bb\n");
  mainCuda(N, resultarray);
  return 0;
}

void
mainCuda(int N, float* resultarray) {

    int totalBytes = sizeof(float) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_result;
    cudaMalloc((void **) &device_result, totalBytes);

    // start timing after allocation of device memory.
    //double startTime = CycleTimer::currentSeconds();

    double kernelStartTime = CycleTimer::currentSeconds();
    test_kernel<<<blocks, threadsPerBlock>>>(N, device_result);
    cudaThreadSynchronize();
    //double kernelEndTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
                errCode, cudaGetErrorString(errCode));
    }
    //double kernelDuration = kernelEndTime - kernelStartTime;

    //printf("Kernel time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
    //double overallDuration = endTime - startTime;
    //printf("Overall time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    /*
    std::cout << "{ ";
    for (int i = 0; i < N; i++) {
        std::cout << resultarray[i] << ", ";
    } std::cout << " }" << std::endl;
    */

    cudaFree(device_result);
}

