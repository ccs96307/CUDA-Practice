#include<iostream>
#include<random>
#include<cuda_runtime.h>

#define N 32
#define C 4
#define H 256
#define W 512

#define TILE_H 8
#define TILE_W 32


__launch_bounds__(256)
__global__ void nchw_to_nhwc_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output) {
    int n = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= H || w >= W) {
        return;
    }

    #pragma unroll
    for (int c=0; c<C; ++c) {
        int inputIdx = ((n * C + c) * H + h) * W + w;
        int outputIdx = ((n * H + h) * W + w) * C + c;

        output[outputIdx] = input[inputIdx];
    }
}


static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


float genRand(std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    return dist(gen);
}


int main() {
    // Init
    int dataNum = N * C * H * W;
    int dataSize = dataNum * sizeof(float);
    float *h_input, *h_output;

    // Malloc host memory
    checkCudaError(cudaMallocHost((void**)&h_input, dataSize), "cudaMallocHost h_input");
    checkCudaError(cudaMallocHost((void**)&h_output, dataSize), "cudaMallocHost h_output");

    // Random initialization
    // Init random number generator and distribution
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (int i=0; i<dataNum; ++i) {
        h_input[i] = float(genRand(gen, dist));
    }

    // Init device data
    float *d_input, *d_output;
    checkCudaError(cudaMalloc((void**)&d_input, dataSize), "cudaMalloc d_input");
    checkCudaError(cudaMalloc((void**)&d_output, dataSize), "cudaMalloc d_output");

    // Create streams and Events
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate stream");

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Copy data
    checkCudaError(cudaMemcpy(d_input, h_input, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy h_input => d_input");
    
    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");
    

    // Launch kernel
    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W,
              (H + TILE_H - 1) / TILE_H,
              N);
    nchw_to_nhwc_kernel<<<grid, block>>>(d_input, d_output);

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print timing
    float ms = 0.0f;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("Optimal Kernel Spend Time: %f ms\n", ms);
    
    // Copy data
    checkCudaError(cudaMemcpy(h_output, d_output, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_output => h_output");
    
    // Destroy
    checkCudaError(cudaFreeHost(h_input), "cudaFreeHost h_input");
    checkCudaError(cudaFreeHost(h_output), "cudaFreeHost h_output");
    checkCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy stream");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_input), "cudaFree d_input");
    checkCudaError(cudaFree(d_output), "cudaFree d_output");
    return 0;
}