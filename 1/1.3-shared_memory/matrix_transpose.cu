#include <iostream>
#include <cuda_runtime.h>

#define NUM_THREADS 16


__global__ void transposeKernel(float *in, float *out, int width, int height, int num_threads) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int inputIdx = height * x + y;

    if (inputIdx < width * height) {
        int transposeIdx = width * y + x;
        out[transposeIdx] = in[inputIdx];
    }
}


static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main() {
    int width = 81920;
    int height = 81920;
    int size = width * height * sizeof(float);

    float *h_in, *h_out;
    float *d_in, *d_out;

    // Malloc host memory
    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);

    // Init input data
    for (int i=0; i<width * height; ++i) {
        h_in[i] = (float)i;
    }

    // Malloc device memory
    checkCudaError(cudaMalloc((void**)&d_in, size), "cudaMalloc d_in");
    checkCudaError(cudaMalloc((void**)&d_out, size), "cudaMalloc d_out");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "cudaMemcpy from h_in in d_in");

    // Create Events
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Set block and grid
    dim3 blockSize(NUM_THREADS, NUM_THREADS);

    int gridX = (width + blockSize.x - 1) / blockSize.x;
    int gridY = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(gridX, gridY);

    // Launch CUDA Kernel
    transposeKernel<<<gridSize, blockSize>>>(d_in, d_out, width, height, NUM_THREADS);
    checkCudaError(cudaGetLastError(), "transposeKernel launched");
    
    // Sync all operations
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for timing");

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print the time
    float ms = 0;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    std::cout << "transposeKernel Spend Time: " << ms << " ms" << std::endl;

    // Copy data from device to host
    checkCudaError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_out => h_out");

    // std::cout << "Original matrix (top-left corner):" << std::endl;
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         std::cout << h_in[i * width + j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::cout << "\nTransposed matrix (top-left corner):" << std::endl;
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         std::cout << h_out[i * height + j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}