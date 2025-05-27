#include <stdio.h>
#include <cuda_runtime.h>

#define N 32
#define DATA_SIZE (N * sizeof(float))

__global__ void streamKernel1(float *d_A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        d_A[idx] += idx;
    }
}


__global__ void streamKernel2(float *d_B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        d_B[idx] += idx + 1;
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
    float* h_B = new float[N];
    for (int i=0; i<N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 1.0;
    }

    // Allocate device memroy
    float *d_A, *d_B;
    checkCudaError(cudaMalloc((void**)&d_A, DATA_SIZE), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, DATA_SIZE), "cudaMalloc d_B");

    // Use cudaStream_t to startup two kernel in the same time
    cudaStream_t stream1, stream2;
    checkCudaError(cudaStreamCreate(&stream1), "cudaStreamCreate stream1");
    checkCudaError(cudaStreamCreate(&stream2), "cudaStreamCreate stream2");

    printf("Streams created: stream1=%p, stream2=%p\n", (void*)stream1, (void*)stream2);

    // Copy data from host to device
    checkCudaError(cudaMemcpyAsync(d_A, h_A, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpyAsync d_A to stream1");
    checkCudaError(cudaMemcpyAsync(d_B, h_B, DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpyAsync d_B to stream2");
 
    // Count time
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Launch kernel (stream1)
    checkCudaError(cudaEventRecord(start, stream1), "cudaEventRecord start");

    streamKernel1<<<1, N, 0, stream1>>>(d_A);
    checkCudaError(cudaGetLastError(), "streamKernel1 launch");
    printf("streamKernel1 launched in stream1\n");

    checkCudaError(cudaEventRecord(stop, stream1), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime stream1");
    printf("GPU kernel (stream2) time: %f ms\n", milliseconds);

    // Count time
    // cudaEvent_t start, stop;
    // checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    // checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Launch kernel (stream2)
    checkCudaError(cudaEventRecord(start, stream2), "cudaEventRecord start");

    streamKernel2<<<1, N, 0, stream2>>>(d_B);
    checkCudaError(cudaGetLastError(), "streamKernel2 launch");
    printf("streamKernel2 launched in stream2\n");

    checkCudaError(cudaEventRecord(stop, stream2), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime stream2");
    printf("GPU kernel (stream2) time: %f ms\n", milliseconds);

    // Copy result from device to host
    checkCudaError(cudaMemcpyAsync(h_A, d_A, DATA_SIZE, cudaMemcpyDeviceToHost), "cudaMemcpyAsync d_A to host from stream1");
    checkCudaError(cudaMemcpyAsync(h_B, d_B, DATA_SIZE, cudaMemcpyDeviceToHost), "cudaMemcpyAsync d_B to host from stream2");

    // Sync to make sure kernel is done
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSychronize");

    for (int i=0; i<N; ++i) {
        printf("h_A[%d] = %f\n", i, h_A[i]);
    }

    for (int i=0; i<N; ++i) {
        printf("h_B[%d] = %f\n", i, h_B[i]);
    }

    // Destroy streams
    checkCudaError(cudaStreamDestroy(stream1), "cudaStreamDestroy stream1");
    checkCudaError(cudaStreamDestroy(stream2), "cudaStreamDestroy stream2");
    

    delete[] h_A;
    delete[] h_B;
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");

    return 0;
}