#include <iostream>
#include <cuda_runtime.h>


#define N 4096


#define TIME_KERNEL(start, stop, kernel_call, ms_out) \
    cudaEventRecord(start); \
    kernel_call; \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&ms_out, start, stop); \
    std::cout << #kernel_call << " time: " << ms_out << " ms" << std::endl;


// Constant memory
__constant__ float const_mem[N];


// Register
__global__ void test_register(float *out) {
    float val = threadIdx.x;
    for (int i=0; i<1000; ++i) {
        val *= 1.00001f;
        out[threadIdx.x] = val;
    }
}


// Shared Memory test
__global__ void test_shared(float *out) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = threadIdx.x;
    __syncthreads();

    for (int i=0; i<1000; ++i) {
        out[tid] = smem[tid];
    }
}


// Global Memory test
__global__ void test_global(float *in, float *out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float val = in[i];

    for (int j=0; j<1000; ++j) {
        val += 1.0f;
    }

    out[i] = val;
}


int main() {
    // Init
    float ms_register, ms_shared, ms_global;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int size = N * sizeof(float);

    float *d_A = NULL;
    float *d_B = NULL;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    TIME_KERNEL(start, stop, (test_register<<<1, 256>>>(d_B)), ms_register);
    TIME_KERNEL(start, stop, (test_shared<<<1, 256>>>(d_B)), ms_shared);
    TIME_KERNEL(start, stop, (test_global<<<N / 256, 256>>>(d_A, d_B)), ms_global);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}