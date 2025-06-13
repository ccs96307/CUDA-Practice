#include <stdio.h>
#include <cuda_runtime.h>

#define N 32
#define DATA_SIZE (N * sizeof(float))


__global__ void computeKernel(float *d_B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        d_B[idx] += idx;
    }
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main() {
    // Allocate host memory
    float* h_A = new float[N];
    for (int i=0; i<N; ++i) {
        h_A[i] = 1.0;
    }

    // Allocate device memroy
    float *d_A;
    checkCudaError(cudaMalloc((void**)&d_A, DATA_SIZE), "cudaMalloc d_A");

    // Use cudaStream_t to startup the kernel
    cudaStream_t stream1;
    checkCudaError(cudaStreamCreate(&stream1), "cudaStreamCreate stream1");

    // Copy data from host to device
    checkCudaError(cudaMemcpyAsync(d_A, h_A, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpyAsync d_A to stream1");
 
    // Count time
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Launch kernel (stream1)
    checkCudaError(cudaEventRecord(start, stream1), "cudaEventRecord start");

    computeKernel<<<1, N, 0, stream1>>>(d_A);
    checkCudaError(cudaGetLastError(), "computeKernel launch");
    printf("computeKernel launched in stream1\n");

    checkCudaError(cudaEventRecord(stop, stream1), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime stream1");
    printf("GPU kernel (stream1) time: %f ms\n", milliseconds);

    // Copy result from device to host
    checkCudaError(cudaMemcpyAsync(h_A, d_A, DATA_SIZE, cudaMemcpyDeviceToHost), "cudaMemcpyAsync d_A to host from stream1");

    // Sync to make sure kernel is done
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSychronize");

    for (int i=0; i<N; ++i) {
        printf("h_A[%d] = %f\n", i, h_A[i]);
    }

    // Destroy
    delete[] h_A;
    checkCudaError(cudaStreamDestroy(stream1), "cudaStreamDestroy stream1");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_A), "cudaFree d_A");

    return 0;
}