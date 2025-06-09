#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000000
#define NUM_THREADS 256
#define NUM_STREAMS 4

// Check N can be divisible
#if N % NUM_STREAMS != 0
#error N must be divisible by NUM_STREAMS
#endif

#define CHUNK_SIZE (N / NUM_STREAMS)
#define CHUNK_BYTES (CHUNK_SIZE * sizeof(float))
#define TOTAL_DATA_SIZE (N * sizeof(float))


__global__ void addKernel(float *a, float *b, float *out, int chunkSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < chunkSize) {
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
    // Init host data (try to use fixed memory)
    float *h_a, *h_b, *h_out;
    checkCudaError(cudaMallocHost((void**)&h_a, TOTAL_DATA_SIZE), "cudaMallocHost h_a");
    checkCudaError(cudaMallocHost((void**)&h_b, TOTAL_DATA_SIZE), "cudaMallocHost h_b");
    checkCudaError(cudaMallocHost((void**)&h_out, TOTAL_DATA_SIZE), "cudaMallocHost h_out");

    for (int i=0; i<N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    // Init device data
    float *d_a, *d_b, *d_out;
    checkCudaError(cudaMalloc((void**)&d_a, TOTAL_DATA_SIZE), "cudaMalloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, TOTAL_DATA_SIZE), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_out, TOTAL_DATA_SIZE), "cudaMalloc d_out");

    // Create Streams and Events
    cudaStream_t streams[NUM_STREAMS];
    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
    }
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Assign tasks to different streams
    for (int i=0; i<NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;

        // Copy data from host to device
        checkCudaError(cudaMemcpyAsync(d_a + offset, h_a + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpy h_a => d_a");
        checkCudaError(cudaMemcpyAsync(d_b + offset, h_b + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpy h_b => d_b");

        // Launch Kernel
        dim3 block(NUM_THREADS);
        dim3 grid((CHUNK_SIZE + block.x - 1) / block.x);
        addKernel<<<grid, block, 0, streams[i]>>>(
            d_a + offset,
            d_b + offset,
            d_out + offset,
            CHUNK_SIZE
        );

        checkCudaError(cudaGetLastError(), "addKerenl launched");

        // Copy result from device to host
        checkCudaError(cudaMemcpyAsync(h_out + offset, d_out + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, streams[i]), "cudaMemcpyAsync d_out to host");
    }

    // Sync all operations
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for timing");

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print the time
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("GPU Kernel Spend Time with %d streams: %f ms\n", NUM_STREAMS, ms);

    // Destory
    checkCudaError(cudaFreeHost(h_a), "cudaFree h_a");
    checkCudaError(cudaFreeHost(h_b), "cudaFree h_b");
    checkCudaError(cudaFreeHost(h_out), "cudaFree h_out");

    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamDestroy(streams[i]), "cudaStreamDestroy streams");
    }
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_a), "cudaFree d_a");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
    checkCudaError(cudaFree(d_out), "cudaFree d_out");
    
    return 0;
}