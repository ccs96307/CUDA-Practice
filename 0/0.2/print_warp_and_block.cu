#include <stdio.h>
#include <cuda_runtime.h>

#define N 33
#define WARP_SIZE 32


__global__ void printInfoKernel() {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        printf("\n\n=================\nBlock ID: %d\nThread Id: %d\nGlobal Thread ID: %d\nWarp ID: %d\nIs it the first thread of warp? %s",
            blockIdx.x,
            threadIdx.x,
            globalIdx,
            threadIdx.x / WARP_SIZE,
            threadIdx.x % WARP_SIZE == 0 ? "true" : "false"
        );
    }

    __syncthreads();
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main() {
    dim3 block(32);
    dim3 grid((N + block.x - 1) / block.x);
    printInfoKernel<<<grid, block>>>();

    checkCudaError(cudaGetLastError(), "Kernel Launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");

    return 0;
}