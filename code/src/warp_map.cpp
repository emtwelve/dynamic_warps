#include <stdio.h>
#include "csv.h"

void create_warp_map(int** input_array, int* result_array, int N, int num_block);

int* warp_map(std::string filename, int num_block, int N) {

    // Input into the map
    int input_array[N][num_block];
    memset(input_array, 0, sizeof(int)*N*num_block);
    int* result_array = (int*)malloc(sizeof(int)*N);

    // Use the log file to populate the input vector
    io::CSVReader<2> in(filename);
    in.read_header(io::ignore_extra_column, "tid", "bb");
    int tid, bb;
    while(in.read_row(tid,bb)){
        input_array[tid][bb]++;
    }

    /*
    DEBUG
    // Print out the input array
    printf("Printing out the input_array\n");
    for (int i = 0; i < N; i++) {
        printf("input_array[%d] = [%d,%d,%d,%d]\n", i,
                input_array[i][0], input_array[i][1], input_array[i][2], input_array[i][3]);
    }
    printf("Done printing out the input_array\n");
    */

    create_warp_map((int**)input_array, result_array, N, num_block);

    /*
    DEBUG
    // Print out the result array
    printf("START OF RESULT_ARRAY\n");
    for (int i = 0; i < N; i++) {
        printf("result_array[%d] = %d\n", i, result_array[i]);
    }
    printf("END OF THE RESULT_ARRAY\n");
    */
    return result_array;
}

int main(int argc, char** argv)
{
    int N = std::stoi(argv[3]);
    int* output = warp_map(argv[1],std::stoi(argv[2]),N);

    // Print out the list
    int i;
    for (i = 0; i < (N-1); i++) {
        printf("%d,", output[i]);
    }
    printf("%d", output[i]);

    return 0;
}
