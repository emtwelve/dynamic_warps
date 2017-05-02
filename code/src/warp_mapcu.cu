#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define WARP_SIZE 4

/////// CREATE THE ARGUMENT MATCHING ARRAY ///////
//////////////////////////////////////////////////

/* Find how many arguments I share with all the other tid */

__global__ void
find_likeness(int* input, int* likeness, int tid, int num_block, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        for (int i = 0; i < num_block; i++) {
            if (input[tid*num_block+i] == input[idx*num_block+i]) {
                likeness[idx]++;
            }
        }
    }
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////


//////////// MAIN ALGORITHM /////////////
/////////////////////////////////////////

void create_warp_map(int** input_array, int* result_array, int N, int num_block)
{
    int totalBytes = sizeof(int) * N;

    // compute number of Blocks and Threads per block
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // DEVICE MEMORY
    int* device_likeness;    // This is for the likeness of the threads
    cudaMalloc((void**) &device_likeness, totalBytes);
    // 2-D array of the input values
    int* device_input_array;
    cudaMalloc((void**)&device_input_array, num_block*totalBytes);
    cudaMemcpy(device_input_array, input_array, num_block*totalBytes, cudaMemcpyHostToDevice);

    // Host Memory
    std::vector<bool> visited(N);      // This is for if a node is already assigned a warp
    visited.clear();
    int num_mapped = 0; // How many have been mapped
    int cur_index = 0;  // First tid in the current warp

    // Initiliaze our first tid in our first warp
    visited[cur_index] = true;
    result_array[num_mapped] = cur_index;
    num_mapped++;

    // Until I have filled all of my output array
    while (num_mapped < N) {

        cudaMemset(device_likeness, 0, totalBytes);
        // Find argument matches based on an index
        find_likeness<<<blocks, threadsPerBlock>>>(device_input_array, device_likeness, cur_index, num_block, N);
        cudaDeviceSynchronize();

        // Sort the index thrust array based on the value thrust array
        thrust::device_ptr<int> index = thrust::device_malloc<int>(N);
        thrust::sequence(index, index + N);

        thrust::device_ptr<int> ptr_likeness(device_likeness);
        thrust::sort_by_key(ptr_likeness, ptr_likeness + N, index, thrust::greater<int>());

        /*
        FOR DEBUGGING
        thrust::host_vector<int> idx(index, index + N);
        thrust::host_vector<int> like(ptr_likeness, ptr_likeness + N);

        for (int i = 0; i < N; i++) {
            printf("%d %d\n", idx[i], like[i]);
        }
        */

        // Loop through the sorted list of arg match and pick the best 31
        for (int i = 0; i < N; i++) {
            // After sorting grab the 32 who arent visited yet
            int pos_index = index[i];
            // If its not already in a warp then put it in this warp
            if (visited[pos_index] == false) {
                visited[pos_index] = true;
                result_array[num_mapped] = pos_index;
                num_mapped++;
            }

            // If this warp has been filled move on
            if ((num_mapped % WARP_SIZE) == 0) {
                break;
            }
        }

        // Start a new warp
        for (int i = 0; i < N; i++) {
            // Find the first index with visited == false
            if (visited[i] == false) {
                cur_index = i;
                visited[cur_index] = true;
                result_array[num_mapped] = cur_index;
                num_mapped++;
                break;
            }
        }
    }
}

/////////////////////////////////////////
/////////////////////////////////////////
