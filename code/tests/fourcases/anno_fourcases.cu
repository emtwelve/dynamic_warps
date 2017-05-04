#define BBLOG(bbid) printf("%d,%d\n", blockIdx.x * blockDim.x + threadIdx.x, bbid)
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

extern float toBW(int bytes, float sec);

__device__ int test ( int x , int y , int z ) {
  int result = 0;
  if (x == 0) { BBLOG(0);

   for (int i = 0; i < 1000000; i++) int loopycntr1 = 0;

   for (int i = 0; i < 1000000; i++) {
      result += y - z;
    loopycntr1++;
}
 LOOPLOG(1, loopycntr1);
{ BBLOG(2);

    for (int i = 0; i < 1000000; i++) int loopycntr3 = 0;

    for (int i = 0; i < 1000000; i++) {
      result += y + z;
    loopycntr3++;
}
 LOOPLOG(3, loopycntr3);
{ BBLOG(4);

    for (int i = 0; i < 1000000; i++) { BBLOG(5);

      result += y * z;
    }
  } else { BBLOG(6);

    for (int i = 0; i < 1000000; i++) { BBLOG(7);

      result += y / z;
    }
  }
  return result;
}

__global__ void test_kernel(int N, float* result) {
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) 