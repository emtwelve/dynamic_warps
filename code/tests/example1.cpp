
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

int main(int argc, char** argv)
{

    int N = 20 * 1000 * 1000;

    float* resultarray = new float[N];

    no_warp_div(resultarray);

    return 0;
}


