#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


#define N 32
#define ITERATIONS 1000


__global__ void withoutPaddingKernel(float *out) {
    __shared__ float tile[N][N];  // No padding
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    for (int i=0; i<ITERATIONS; ++i) {
        tile[tx][ty] = tx * 1.0f;
        __syncthreads();
        sum += tile[ty][tx]; // Column-wise access (bad)
        __syncthreads();
    }
    out[tx * N + ty] = sum;
}


__global__ void withPaddingKernel(float *out) {
    __shared__ float tile[N][N+1];  // Padding by 1
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    for (int i=0; i<ITERATIONS; ++i) {
        tile[tx][ty] = tx * 1.0f;
        __syncthreads();
        sum += tile[ty][tx]; // Column-wise access (good)
        __syncthreads();
    }
    out[tx * N + ty] = sum;
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


void benchmark(void (*kernel)(float*), float* d_out, const char* name) {
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    dim3 block(N, N);
    dim3 grid(1, 1);

    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");
    kernel<<<grid, block>>>(d_out);
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");

    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime: stop - start and assign to `ms`");
    std::cout << name << " time: " << ms << " ms\n";

    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
}


int main() {
    float *d_out;
    checkCudaError(cudaMalloc(&d_out, sizeof(float) * N * N), "cudaMalloc d_out");

    benchmark(withPaddingKernel, d_out, "With Padding");
    benchmark(withoutPaddingKernel, d_out, "Without Padding");
    benchmark(withPaddingKernel, d_out, "With Padding");

    checkCudaError(cudaFree(d_out), "cudaFree d_out");

    return 0;
}