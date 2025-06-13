#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000000
#define NUM_THREADS 256
#define NUM_STREAMS 4

#if N % NUM_STREAMS != 0
#error N must be divisible by NUM_STREAMS
#endif

#define CHUNK_SIZE (N / NUM_STREAMS)
#define CHUNK_BYTES (CHUNK_SIZE * sizeof(float))
#define TOTAL_DATA_SIZE (N * sizeof(float))


// The compute formula: y = scale * relu(x + b)
__global__ void add_relu_scaleKernel(const float* x, const float *b, float scale, float* y, int chunkSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < chunkSize) {
        y[idx] = scale * (fmaxf(0.0f, x[idx] + b[idx]));
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
    float scale = 2.0f;
    float *h_x, *h_b, *h_y;
    checkCudaError(cudaMallocHost((void**)&h_x, TOTAL_DATA_SIZE), "cudaMallocHost h_x");
    checkCudaError(cudaMallocHost((void**)&h_b, TOTAL_DATA_SIZE), "cudaMallocHost h_b");
    checkCudaError(cudaMallocHost((void**)&h_y, TOTAL_DATA_SIZE), "cudaMallocHost h_y");    

    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    // Init device data
    float *d_x, *d_b, *d_y;
    checkCudaError(cudaMalloc((void**)&d_x, TOTAL_DATA_SIZE), "cudaMalloc d_x");
    checkCudaError(cudaMalloc((void**)&d_b, TOTAL_DATA_SIZE), "cudaMalloc d_b");
    checkCudaError(cudaMalloc((void**)&d_y, TOTAL_DATA_SIZE), "cudaMalloc d_y");

    // Create Streams and Events
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
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
        checkCudaError(cudaMemcpyAsync(d_x + offset, h_x + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpy h_x => d_x");
        checkCudaError(cudaMemcpyAsync(d_b + offset, h_b + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpy h_b => d_b");

        // Init block and grid
        dim3 block(NUM_THREADS);
        dim3 grid((CHUNK_SIZE + block.x - 1) / block.x);

        // Launch Add Kernel
        add_relu_scaleKernel<<<grid, block, 0, streams[i]>>>(
            d_x + offset,
            d_b + offset,
            scale,
            d_y + offset,
            CHUNK_SIZE
        );
        checkCudaError(cudaGetLastError(), "add_relu_scaleKernel launched");
        checkCudaError(cudaMemcpyAsync(h_y + offset, d_y + offset, CHUNK_BYTES, cudaMemcpyDeviceToHost, streams[i]), "cudaMemcpy d_y => h_y");
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
    checkCudaError(cudaFreeHost(h_x), "cudaFree h_x");
    checkCudaError(cudaFreeHost(h_b), "cudaFree h_b");
    checkCudaError(cudaFreeHost(h_y), "cudaFree h_y");

    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamDestroy(streams[i]), "cudaStreamDestroy streams");
    }
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_x), "cudaFree d_x");
    checkCudaError(cudaFree(d_b), "cudaFree d_b");
    checkCudaError(cudaFree(d_y), "cudaFree d_y");

    return 0;
}