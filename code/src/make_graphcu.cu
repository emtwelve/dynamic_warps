/////// CREATE THE GRAPH ///////
////////////////////////////////

__device__ int
find_total(int** input, int my_index, int other_index, int n_arg) {
    int match = 0;
    for (int i = 0; i < num_arg; i++) {
        if (input[my_index] == input[other_index][arg_num]) {
            match++;
        }
    }
    return match;
}

__global__ void
graph_creator(int** input, int** graph, int n_arg, int N) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        for (int i = 0; i < N; i++) {
            graph[index][i] = find_total(input, index, i, num_arg);
        }
    }
}

////////////////////////////////
////////////////////////////////

/////// OUTPUT THE GRAPH /////////
//////////////////////////////////

__global__ void
graph_to_out_kernel(int** graph, int* device_out, int N, int n_arg) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        for (int i = 0; i < N; i++) {
            graph[index][i] = find_total(input, index, i, num_arg);

}

//////////////////////////////////
//////////////////////////////////


void create_Graph(int** input_array, int* result_array, int N, int num_arg)
{
    int totalBytes = sizeof(int) * N;

    // compute number of Blocks and Threads per block
    const int threadsPerBlock = 32;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int** graph;
    for (int
    cudaMalloc((void**) & device_graph, N * (int**)


    // Generate my Graph of the edges and weight
    graph_creation_kernel<<blocks, threadsPerBlock>>(device_input, device_graph, N, n_arg);
    cudaThreadSynchronize();

    // Create the output array
    graph_to_out_kernel<<blocks, threadsPerBlock>>(device_graph, device_out, N, n_arg);
