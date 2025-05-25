#include <stdio.h>
#include <cuda_runtime.h>


__global__ void sharedMemoryTestKernel() {
    // Every block has their own `shareVal` instance
    __shared__ int sharedVal;


    // The first thread of block will assign the blockIdx.x into `sharedVal`
    if (threadIdx.x == 0) {
        sharedVal = blockIdx.x * 100;
    }

    // Sync all threads of the block
    __syncthreads();

    printf("Block ID: %d, Thread ID: %d -- sees sharedVal = %d\n", blockIdx.x, threadIdx.x, sharedVal);
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sharedVal = 777;
    }

    __syncthreads();

    if (blockIdx.x == 1) {
        printf(
            "After block 0 modification attemp: Block ID: %d, Thread ID: %d -- sees sharedVal = %d\n",
            blockIdx.x,
            threadIdx.x,
            sharedVal
        );
    }
}


int main() {
    int numBlocks = 2;
    int threadsPerBlock = 4;

    sharedMemoryTestKernel<<<numBlocks, threadsPerBlock>>>();

    // Wait for all calcualtion of GPU finished
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    printf("\n");
    return 0;
}