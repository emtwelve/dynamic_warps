/* Norman Ponte; Joey Fernau
 * annotation generation test
 */

#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../../lib/CycleTimer.h"

extern float toBW(int bytes, float sec);

__device__ int test_x_1 (  int y , int z  ) { 
	bool x = 1;
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
__device__ int test_x_0 (  int y , int z  ) { 
	bool x = 0;
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
__device__ int branch_test ( bool x , int y , int z ) {
	switch (x) {
		case 1:
			return test_x_1 ( y , z ) ;
		case 0:
			return test_x_0 ( y , z ) ;
	}
	int *asdffdsa12344321 = NULL;
	return (int) *asdffdsa12344321;
}

__global__ void
test_kernel(int N, float* result) {
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
       result[index] = branch_test(index % 2 == 0, index % 13, index % 7);
    }
}

void
mainCuda(int N, float* resultarray) {

    int totalBytes = sizeof(float) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_result;
    cudaMalloc((void **) &device_result, N * sizeof(float));

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


