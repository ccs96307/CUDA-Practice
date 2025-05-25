#include <stdio.h>
#include <cuda_runtime.h>


#define N 10000


__global__ void square(float *A) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        A[i] = A[i] * A[i];
    }
}


int main() {
    cudaError_t err = cudaSuccess;

    // Init
    size_t size = N * sizeof(float);

    // Allocate the host input vector
    float *h_A = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i=0; i<N; ++i) {
        h_A[i] = rand();
    }

    // Allocate the device input vectors
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy into GPU
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 40 * 256 = 10240
    square<<<blocksPerGrid, threadsPerBlock>>>(d_A);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Sync to make sure kernel is done
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print
    for (int i=0; i<N; ++i) {
        printf("h_A[%d] = %.4f\n", i, h_A[i]);
    }

    // Free device memory
    free(h_A);
    cudaFree(d_A);

    return 0;
}