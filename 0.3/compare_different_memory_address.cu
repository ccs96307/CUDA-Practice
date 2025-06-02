#include <stdio.h>
#include <cuda_runtime.h>

#define N 320000000
#define DATA_SIZE (N * sizeof(float))


__global__ void alignedKernel(float *d_A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = d_A[idx];
        d_A[idx] = val * 2.0f;
    }
}


__global__ void stridedKernel(float *d_B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = d_B[idx * 2 % N];
        d_B[idx] = val * 2.0f;
    }
}


__global__ void randomKernel(float *d_C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int randIdx = (idx * 2 + 13) % N;
        float val = d_C[randIdx];
        d_C[randIdx] = val * 2.0f;
    }
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


void benchmark(void (*kernel)(float*), float *d_out, const char *name) {
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate &start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate &stop");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 block(blockSize);
    dim3 grid(gridSize);

    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");
    kernel<<<grid, block>>>(d_out);
    checkCudaError(cudaGetLastError(), "Kernel Launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Show time
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("%s time: %f sec\n", name, ms / 1000);

    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");    
}


int main() {
    // Init
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    for (int i=0; i<N; ++i) {
        h_A[i] = 1.0 * (i + 1);
        h_B[i] = 1.0 * (i + 1);
        h_C[i] = 1.0 * (i + 1);        
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, DATA_SIZE), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, DATA_SIZE), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, DATA_SIZE), "cudaMalloc d_C");

    // Memory copy
    checkCudaError(cudaMemcpy(d_A, h_A, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");
    checkCudaError(cudaMemcpy(d_C, h_C, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_C to d_C");

    // Benchmark
    benchmark(alignedKernel, d_A, "alignedKernel");
    benchmark(stridedKernel, d_B, "stridedKernel");
    benchmark(randomKernel, d_C, "randomKernel");

    // Free the memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");

    return 0;
}
