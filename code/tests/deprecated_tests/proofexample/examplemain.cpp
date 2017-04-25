#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#define N 10000000

void test(int* input1, int* input2, int* warpmap);
int* generateInput(int which);
int* generateWarpMap();

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

int main(int argc, char** argv)
{
  test(generateInput(0), generateInput(1), generateWarpMap());
}

// Helper Functions for hardcoding arrays
int* generateInput(int which) {
  int* retarr = (int*)malloc(sizeof(int)*N);
  for (int i = 0; i < N; i++) {
    if ((i % 2) == which) {
      retarr[i] = 1;
    }
  }
  return retarr;
}

int* generateWarpMap() {
  int* warp_map = (int*)malloc(sizeof(int)*N);
  int cur_index = 0;
  for (int i = 0; i < N; i++) {
    if (i == (N/2)) cur_index = 1;
    warp_map[i] = cur_index;
    cur_index += 2;
  }
  return warp_map;
}
