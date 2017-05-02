#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>


// return GB/s
float toBW(int bytes, float sec) {
   return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

void mainCuda(int N, float* result);

int main(int argc, char** argv)
{

    int N = 64;

    float* resultarray = new float[N];

    mainCuda(N, resultarray);

    return 0;
}

