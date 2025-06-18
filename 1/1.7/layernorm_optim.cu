#include <stdio.h>
#include <random>
#include<cuda_runtime.h>

#define NUM_THREADS 256
#define NUM_STREAMS 2
#define B 16384
#define H 4096

#if (B * H) % NUM_STREAMS != 0
#error (B * H) must be divisible by NUM_STREAMS
#endif

#define CHUNK_ROWS (B / NUM_STREAMS)
#define CHUNK_SIZE (CHUNK_ROWS * H)
#define CHUNK_BYTES (CHUNK_SIZE * sizeof(float))

// x_i = (x1, x2, ..., xn)
// x_i = (xi - mean) / sqrt(var^2 + eps)
// y_i = gamma * x_i + beta
__global__ void LayerNormKernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ gamma, const float* __restrict__ beta, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float accum = 0.0f;
    float mean = 0.0f;
    float var = 0.0f;
    
    if (idx >= B / NUM_STREAMS) {
        return;
    }
    
    // Calculate `mean`
    for (int i=0; i<H; ++i) {
        accum += input[idx * H + i];
    }
    mean = accum / H;
    accum = 0;

    // Calculate `var^2`
    for (int i=0; i<H; ++i) {
        float diff = input[idx * H + i] - mean;
        accum += diff * diff;
    }
    var = accum / H;

    // Calculate `x_i`
    for (int i=0; i<H; ++i) {
        output[idx * H + i] = gamma[i] * ((input[idx * H + i] - mean) / sqrtf(var + epsilon)) + beta[i];
    }
}


static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


float getRand(std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    return dist(gen);
}


int main() {
    // Init
    int sizeMat = B * H * sizeof(float);
    int sizeArr = H * sizeof(float);

    float epsilon = 1e-10;
    float *h_input, *h_output, *h_gamma, *h_beta;

    // Malloc host data
    checkCudaError(cudaMallocHost((void**)&h_input, sizeMat), "cudaMallocHost h_input");
    checkCudaError(cudaMallocHost((void**)&h_output, sizeMat), "cudaMallocHost h_output");
    checkCudaError(cudaMallocHost((void**)&h_gamma, sizeArr), "cudaMallocHost h_gamma");
    checkCudaError(cudaMallocHost((void**)&h_beta, sizeArr), "cudaMallocHost h_beta");
    
    // Random initialization
    // Init random number generator and distribution
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (int i=0; i<H; ++i) {
        h_gamma[i] = getRand(gen, dist);
        h_beta[i] = getRand(gen, dist);

        for (int j=0; j<B; ++j) {
            h_input[j * H + i] = getRand(gen, dist);
        }
    }

    // Init device data
    float *d_input, *d_output, *d_gamma, *d_beta;
    checkCudaError(cudaMalloc((void**)&d_input, sizeMat), "cudaMalloc d_input");
    checkCudaError(cudaMalloc((void**)&d_output, sizeMat), "cudaMalloc d_output");
    checkCudaError(cudaMalloc((void**)&d_gamma, sizeArr), "cudaMalloc d_gamma");
    checkCudaError(cudaMalloc((void**)&d_beta, sizeArr), "cudaMalloc d_beta");

    // Create Streams and Events
    cudaStream_t streams[NUM_STREAMS];
    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
    }
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");  

    // Copy gamma and beta
    checkCudaError(cudaMemcpy(d_gamma, h_gamma, sizeArr, cudaMemcpyHostToDevice), "cudaMemcpy h_gamma => d_gamma");
    checkCudaError(cudaMemcpy(d_beta, h_beta, sizeArr, cudaMemcpyHostToDevice), "cudaMemcpy h_beta => d_beta");

    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Assign tasks to different streams
    for (int i=0; i<NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;

        // Copy data from host to device
        checkCudaError(cudaMemcpyAsync(d_input + offset, h_input + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpy d_input => h_input");

        // Launch LayerNorm Kernel
        dim3 block(NUM_THREADS);
        dim3 grid((CHUNK_ROWS + block.x - 1) / block.x);
        LayerNormKernel<<<grid, block, 0, streams[i]>>>(d_input + offset, d_output + offset, d_gamma, d_beta, epsilon);
        checkCudaError(cudaGetLastError(), "LayerNormKernel launched");
    }
    
    // Sync all operations
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for timing");

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print timing
    float ms = 0.0f;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("Original Kernel Spend Time: %f ms\n", ms);

    // Destroy
    checkCudaError(cudaFreeHost(h_input), "cudaFree h_input");
    checkCudaError(cudaFreeHost(h_output), "cudaFree h_output");
    checkCudaError(cudaFreeHost(h_gamma), "cudaFree h_gamma");
    checkCudaError(cudaFreeHost(h_beta), "cudaFree h_beta");

    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamDestroy(streams[i]), "cudaStreamDestroy streams");
    }
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_input), "cudaFree d_input");
    checkCudaError(cudaFree(d_output), "cudaFree d_output");
    checkCudaError(cudaFree(d_gamma), "cudaFree d_gamma");
    checkCudaError(cudaFree(d_beta), "cudaFree d_beta");

    return 0;
}