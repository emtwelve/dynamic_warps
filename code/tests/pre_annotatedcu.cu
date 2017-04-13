/* Norman Ponte; Joey Fernau
 * warp divergence test
 */

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../../lib/CycleTimer.h"

extern float toBW(int bytes, float sec);

__device__ int
test_dev(int x, int y) {
  return x + y;
}

__global__ void
test_kernel(int N, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = test_dev(index, -index+1);
}

void
mainCuda(int N, float* resultarray) {

    int totalBytes = sizeof(float) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
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

    cudaFree(device_result);
}


