#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define DATA_SIZE (N * sizeof(int))


__global__ void squareKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = data[idx] * data[idx];
    }
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main() {
    // Init
    int *h_data = new int[N];
    for (int i=0; i<N; ++i) {
        h_data[i] = i;
    }

    // Allocate device data
    int *d_data;
    checkCudaError(cudaMalloc((void**)&d_data, DATA_SIZE), "cudaMalloc d_data");

    // Copy (host to device)
    checkCudaError(cudaMemcpy(d_data, h_data, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy h_data into d_data");

    // Launch kernel
    dim3 block(32);
    dim3 grid((N + block.x - 1) / block.x);
    squareKernel<<<grid, block>>>(d_data);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Sync to make sure kernel is done
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSychronize");

    // Copy (device to host)
    checkCudaError(cudaMemcpy(h_data, d_data, DATA_SIZE, cudaMemcpyDeviceToHost), "cudaMemcpy d_data into h_data");
    
    // Print the first 10 data
    for (int i=0; i<10; ++i) {
        printf("Results: %d^2 = %d\n", i, h_data[i]);
    }

    // Release
    delete[] h_data;
    checkCudaError(cudaFree(d_data), "cudaFree d_data");

    return 0;
}




