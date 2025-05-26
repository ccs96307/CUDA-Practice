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


void benchmark(void (*kernel)(float*), float* d_out, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(N, N);
    dim3 grid(1, 1);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_out);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << name << " time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    float *d_out;
    cudaMalloc(&d_out, sizeof(float) * N * N);

    benchmark(withoutPaddingKernel, d_out, "Without Padding");
    benchmark(withPaddingKernel, d_out, "With Padding");
    
    cudaFree(d_out);

    return 0;
}