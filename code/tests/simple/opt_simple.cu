
int host_remap[64] = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63};
__device__ volatile int device_remap[64];
int *remap() {
    //int *device_remap;
    cudaMemcpyToSymbol(device_remap, host_remap, 64 * sizeof(int));
    //return device_remap;
    return NULL;
}
/* Norman Ponte; Joey Fernau
 * annotation generation test
 */

#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../../lib/CycleTimer.h"

void mainCuda(int N, float* resultArray);

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

__device__ int test ( bool x , int y , int z ) {
  int result = 0;
  if (x) {
    for (int i = 0; i < 10000; i++)
      result += y - z;
  } else {
    for (int i = 0; i < 10000; i++)
      result += y - z;
  }
  return result;
}

__global__ void
test_kernel(int N, float* result) {
    // compute overall ridx from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

volatile int ridx = device_remap[index];

    if (ridx < N) {
       result[ridx] = test(ridx % 2 == 0, ridx % 13, ridx % 7);
    }
}

int main(int argc, char** argv) {

remap();
  int N = 64;

  float* resultarray = new float[N];

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
    cudaMalloc((void **) &device_result, N * sizeof(float));
    cudaMemset((void **) &device_result, 0, N * sizeof(float));
    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //cudaMemcpy(device_x, xarray, N * sizeof(float),
    //           cudaMemcpyHostToDevice);

    double kernelStartTime = CycleTimer::currentSeconds();
    test_kernel<<<blocks, threadsPerBlock>>>(N, device_result);
    cudaThreadSynchronize();
    double kernelEndTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
                errCode, cudaGetErrorString(errCode));
    }
    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("Kernel time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));
    double overallDuration = endTime - startTime;
    printf("Overall time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    std::cout << "{ ";
    for (int i = 0; i < N; i++) {
        std::cout << resultarray[i] << ", ";
    } std::cout << " }" << std::endl;

    cudaFree(device_result);
}



