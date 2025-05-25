#include <stdio.h>
#include <cuda_runtime.h>


__global__ void printThreadIdsKernel() {
    int globalThreadIds = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Block ID: %d, Thread ID in Block: %d (Global Thread ID: %d)\n", blockIdx.x, threadIdx.x, globalThreadIds);
}


int main() {
    int numBlocks = 2;
    int threadsPerBlock = 4;

    printThreadIdsKernel<<<numBlocks, threadsPerBlock>>>();

    // Wait for all calcualtion of GPU finished
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    printf("\n");
    return 0;
}