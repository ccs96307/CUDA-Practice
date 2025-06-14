#include <stdio.h>
#include <random>
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

#define ITERS 100


__global__ void divergeKernel(const float* x1, float* y1, int chunkSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < chunkSize) {
        float x = x1[idx];
        float result;
        
        if (x > 0.5f) {
            result = x;
            for (int i=0; i<ITERS; ++i) {
                result = sinf(result) * 0.5f + cosf(result) * 0.5f;
            }
        }
        else {
            result = x;
        }

        y1[idx] = result;
    }
}


__global__ void nonDivergeKernel(const float* x2, float* y2, int chunkSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < chunkSize) {
        float x = x2[idx];

        bool condition = (x > 0.5f);

        float heavy_result = x;
        for (int i=0; i<ITERS; ++i) {
            heavy_result = sinf(heavy_result) * 0.5f + cosf(heavy_result) * 0.5f;
        }

        float simple_result = x;

        y2[idx] = condition ? heavy_result : simple_result;
    }
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


float getRand(std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    return dist(gen);
}


int main() {
    // Init random number generator and distribution
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    // Init host data (try to use fixed memory)
    float *h_x1, *h_x2, *h_y1, *h_y2;
    checkCudaError(cudaMallocHost((void**)&h_x1, TOTAL_DATA_SIZE), "cudaMallocHost h_x1");
    checkCudaError(cudaMallocHost((void**)&h_x2, TOTAL_DATA_SIZE), "cudaMallocHost h_x2");
    checkCudaError(cudaMallocHost((void**)&h_y1, TOTAL_DATA_SIZE), "cudaMallocHost h_y1");
    checkCudaError(cudaMallocHost((void**)&h_y2, TOTAL_DATA_SIZE), "cudaMallocHost h_y2");

    for (int i = 0; i < N; ++i) {
        h_x1[i] = getRand(gen, dist);
        h_x2[i] = h_x1[i];
    }

    // Init device data
    float *d_x1, *d_x2, *d_y1, *d_y2;
    checkCudaError(cudaMalloc((void**)&d_x1, TOTAL_DATA_SIZE), "cudaMalloc d_x1");
    checkCudaError(cudaMalloc((void**)&d_x2, TOTAL_DATA_SIZE), "cudaMalloc d_x2");
    checkCudaError(cudaMalloc((void**)&d_y1, TOTAL_DATA_SIZE), "cudaMalloc d_y1");
    checkCudaError(cudaMalloc((void**)&d_y2, TOTAL_DATA_SIZE), "cudaMalloc d_y2");

    // Copy data from host to device
    checkCudaError(cudaMemcpyAsync(
        d_x1,
        h_x1, 
        CHUNK_BYTES * NUM_STREAMS,
        cudaMemcpyHostToDevice
    ), "cudaMemcpy h_x1 => d_x1");
    checkCudaError(cudaMemcpyAsync(
        d_x2,
        h_x2, 
        CHUNK_BYTES * NUM_STREAMS,
        cudaMemcpyHostToDevice
    ), "cudaMemcpy h_x2 => d_x2");

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

        // Init block and grid
        dim3 block(NUM_THREADS);
        dim3 grid((CHUNK_SIZE + block.x - 1) / block.x);

        // Launch Add Kernel
        divergeKernel<<<grid, block, 0, streams[i]>>>(
            d_x1 + offset,
            d_y1 + offset,
            CHUNK_SIZE
        );
        checkCudaError(cudaGetLastError(), "divergeKernel launched");
    }

    // Sync all operations
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for timing");

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print the time
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("Divergence Kernel Spend Time with %d streams: %f ms\n", NUM_STREAMS, ms);

    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Assign tasks to different streams
    for (int i=0; i<NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;

        // Init block and grid
        dim3 block(NUM_THREADS);
        dim3 grid((CHUNK_SIZE + block.x - 1) / block.x);

        // Launch Add Kernel
        nonDivergeKernel<<<grid, block, 0, streams[i]>>>(
            d_x2 + offset,
            d_y2 + offset,
            CHUNK_SIZE
        );
        checkCudaError(cudaGetLastError(), "nonDivergeKernel launched");
    }

    // Sync all operations
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for timing");

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print the time
    ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("Non-Divergence Kernel Spend Time with %d streams: %f ms\n", NUM_STREAMS, ms);

    // Destory
    checkCudaError(cudaFreeHost(h_x1), "cudaFree h_x1");
    checkCudaError(cudaFreeHost(h_x2), "cudaFree h_x2");
    checkCudaError(cudaFreeHost(h_y1), "cudaFree h_y1");
    checkCudaError(cudaFreeHost(h_y2), "cudaFree h_y2");

    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamDestroy(streams[i]), "cudaStreamDestroy streams");
    }
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_x1), "cudaFree d_x1");
    checkCudaError(cudaFree(d_x2), "cudaFree d_x2");
    checkCudaError(cudaFree(d_y1), "cudaFree d_y1");
    checkCudaError(cudaFree(d_y2), "cudaFree d_y2");

    return 0;
}