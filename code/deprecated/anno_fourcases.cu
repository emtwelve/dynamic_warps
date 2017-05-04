/* Norman Ponte; Joey Fernau
 * annotation generation test
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <getopt.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "../../lib/CycleTimer.h"

#define BBLOG(bbid) printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, bbid)

extern float toBW(int bytes, float sec);

__device__ int test ( int x , int y , int z ) {
  BBLOG(2);
  int result = 0;
  if (x == 0) {
    BBLOG(3);
    for (int i = 0; i < 1000000; i++)
      result += y - z;
  } else if (x == 1) {
    BBLOG(4);
    for (int i = 0; i < 1000000; i++)
      result += y + z;
  } else if (x == 2) {
    BBLOG(5);
    for (int i = 0; i < 1000000; i++)
      result += y * z;
  } else {
    BBLOG(6);
    for (int i = 0; i < 1000000; i++)
      result += y / z;
  }
  BBLOG(7);
  return result;
}

__global__ void
test_kernel(int N, float* result) {
    BBLOG(0);
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
       BBLOG(1);
       result[index] = test(index % 4, index % 13, index % 7);
    }
    BBLOG(8);
}

void
mainCuda(int N, float* resultarray) {


    // compute number of blocks and threads per block
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_result;

    cudaMalloc((void **) &device_result, N * sizeof(float));

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //cudaMemcpy(device_x, xarray, N * sizeof(float),
    //           cudaMemcpyHostToDevice);

    test_kernel<<<blocks, threadsPerBlock>>>(N, device_result);
    cudaThreadSynchronize();

    cudaMemcpy(resultarray, device_result, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
                errCode, cudaGetErrorString(errCode));
    }

    cudaFree(device_result);
}

// return GB/s
float toBW(int bytes, float sec) {
   return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void mainCuda(int N, float* result);

int main(int argc, char** argv)
{
    printf("tid,bb\n");
    int N = std::atoi(argv[1]);

    float* resultarray = new float[N];

    mainCuda(N, resultarray);

    return 0;
}


