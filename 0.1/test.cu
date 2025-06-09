#include <stdio.h>

// For CUDA runtime routines
#include <cuda_runtime.h>


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}


int main(void) {
    // Print the vector length to be used, and compute its size
    int numElements = 100;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector
    float *h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i=0; i<numElements; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate the device input vectors
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
        
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy into GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy into Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print
    for (int i=0; i<numElements; ++i) {
        printf("h_C[%d] = %f\n", i, h_C[i]);
    }

    printf("Test PASSED\n");

    // Free
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}