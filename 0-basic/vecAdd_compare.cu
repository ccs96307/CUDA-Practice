#include <iostream>
#include <chrono>
#include <unistd.h>
#include <cuda_runtime.h>

#define N 51200000  // Vector size


// CUDA Kernel (runs on GPU)
__global__ void vecAdd(float* A, float* B, float* C) {
    // int i = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


int main() {
    // Allocate host memory
    // float h_A[N], h_B[N], h_C[N];
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i=0; i<N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Count time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    vecAdd<<<1, N>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    // Sync to make sure kernel is done
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_C[i] = h_A[i] + h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}