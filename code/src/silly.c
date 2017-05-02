

int *host_remap = ...;
int *remap() {
    int *device_remap;
    cudaMemcpy(device_remap, host_remap, ... * sizeof(int),
                       cudaMemcpyHostToDevice);
    return device_remap;
}

int main() {
   int *device_remap = remap();



}
