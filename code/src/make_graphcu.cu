#define WARP_SIZE 32

/////// CREATE THE ARGUMENT MATCHING ARRAY ///////
//////////////////////////////////////////////////

/* Find how many arguments I share with all the other tid */

__global__ void
find_grouping(int** input, int* arg_match, int tid, int num_arg, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int match = 0;
    if (idx < N) {
        for (int i = 0; i < num_arg; i++) {
            if (input[tid][i] == input[idx][i]) {
                match++
            }
        }
        arg_match[idx] = match;
    }
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////

//////////// MAIN ALGORITHM /////////////
/////////////////////////////////////////

void create_Graph(int** input_array, int* result_array, int N, int num_arg)
{
    int totalBytes = sizeof(int) * N;

    // compute number of Blocks and Threads per block
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Device Memory
    int* device_arg_match;    // This is for the number of argument matches
    cudaMalloc((void**) &device_arg_match, N * sizeof(int));
    // 2-D array of the input values
    int** device_input_array;
    cudaMalloc((void**)&device_input_array, N*num_arg*sizeof(int));
    cudaMemcpy(device_input_array, input_array, N*num_arg*sizeof(int), cudaMemcpyHostToDevice);
    // Initialize the Thrust arrays
    thrust::device_vector<int> index(N);


    // Host Memory
    int* result_array; // This is the final mapping of index to warp
    bool* visited;      // This is for if a node is already assigned a warp
    int num_mapped = 0; // How many have been mapped
    int cur_index = 0;  // First tid in the current warp

    // Initiliaze our first tid in our first warp
    visited[cur_index] = true;
    result_array[num_mapped] = cur_index;
    num_mapped++;

    // Until I have filled all of my output array
    while (num_mapped < N) {

        // Find argument matches based on an index
        find_arg_match<<blocks, threadsPerBlock>>(input_array, device_arg_match, cur_index, N);
        cudaThreadSynchronize();


        // Sort the index thrust array based on the value thrust array
        thrust::sequence(index.begin(), index.end());
        thrust::device_ptr<int> ptr_arg_match = thrust::device_pointer_cast(arg_match);
        thrust::sort_by_key(ptr_arg_match, ptr_arg_match + N, index);

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
            if (num_mapped % WARP_SIZE) {
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
            }
        }
    }
}

/////////////////////////////////////////
/////////////////////////////////////////
