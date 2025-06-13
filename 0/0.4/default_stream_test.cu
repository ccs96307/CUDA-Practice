#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000000
#define THREADS 256
#define DATA_SIZE (N * sizeof(float))



__global__ void addKernel(float *a, float *b, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Increase many operations
        float temp_a = a[idx];
        float temp_b = b[idx];

        for (int i=0; i<1000; ++i) {
            temp_a = temp_a * 0.9999f + temp_b * 0.0001f;
        }

        out[idx] = temp_a;
    }
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Init host data
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_out = new float[N];

    for (int i=0; i<N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Init device data
    float *d_a, *d_b, *d_out;
    checkCudaError(cudaMalloc((void**)&d_a, DATA_SIZE), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, DATA_SIZE), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_out, DATA_SIZE), "cudaMalloc d_out");

    // Init Event
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_a, h_a, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_a => d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_b => d_b");
    checkCudaError(cudaMemcpy(d_out, h_out, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_out => d_out");

    // Launch Kernel
    dim3 block(THREADS);
    dim3 grid((N + block.x - 1) / block.x);
    addKernel<<<grid, block>>>(d_a, d_b, d_out);

    checkCudaError(cudaGetLastError(), "addKerenl launched");

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print the time
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("GPU Kernel Spend Time: %f ms\n", ms);

    // Copy result from device to host
    checkCudaError(cudaMemcpyAsync(h_out, d_out, DATA_SIZE, cudaMemcpyDeviceToHost), "cudaMemcpyAsync d_out to host");

    // Destory
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_a), "cudaFree d_a");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
    checkCudaError(cudaFree(d_out), "cudaFree d_out");
    
    return 0;
}