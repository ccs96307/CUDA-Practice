#include <stdio.h>
#include <random>
#include <cuda_runtime.h>
#include <math.h>

#define N 20000000
#define NUM_THREADS 256
#define ITERS 200 // 增加迭代次數讓效果更明顯

#define TOTAL_DATA_SIZE (N * sizeof(float))
static void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ===================================================================
// KERNEL 1: 純粹的分支發散
// if/else 中有不同的、耗時的迴圈
// ===================================================================
__global__ void true_divergeKernel(const float* in, float* out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float result = 0.0f;

        if (x > 0.0f) {
            // 路徑 A: 迴圈加 1.0
            for (int i = 0; i < ITERS; ++i) {
                result += 1.0f;
            }
        } else {
            // 路徑 B: 迴圈加 2.0
            for (int i = 0; i < ITERS; ++i) {
                result += 2.0f;
            }
        }
        out[idx] = result + x;
    }
}

// ===================================================================
// KERNEL 2: 真正無分支的優化
// 在迴圈外決定好參數，迴圈內部指令完全統一
// ===================================================================
__global__ void true_nonDivergeKernel(const float* in, float* out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        
        // 關鍵：在迴圈開始前，用一個廉價的條件移動決定好增量
        float increment = (x > 0.0f) ? 1.0f : 2.0f;

        float result = 0.0f;
        
        // 這個迴圈對於Warp中的所有執行緒，指令都是100%相同的
        for (int i = 0; i < ITERS; ++i) {
            result += increment;
        }

        out[idx] = result + x;
    }
}

// Main function
int main() {
    printf("N = %d, ITERS = %d\n", N, ITERS);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    float *h_x, *h_y;
    checkCudaError(cudaMallocHost((void**)&h_x, TOTAL_DATA_SIZE), "cudaMallocHost h_x");
    checkCudaError(cudaMallocHost((void**)&h_y, TOTAL_DATA_SIZE), "cudaMallocHost h_y");

    for (int i = 0; i < N; ++i) { h_x[i] = dist(gen); }

    float *d_x, *d_y;
    checkCudaError(cudaMalloc((void**)&d_x, TOTAL_DATA_SIZE), "cudaMalloc d_x");
    checkCudaError(cudaMalloc((void**)&d_y, TOTAL_DATA_SIZE), "cudaMalloc d_y");
    checkCudaError(cudaMemcpy(d_x, h_x, TOTAL_DATA_SIZE, cudaMemcpyHostToDevice), "cudaMemcpy");

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    dim3 block(NUM_THREADS);
    dim3 grid((N + NUM_THREADS - 1) / NUM_THREADS);
    float ms;

    // --- 測試 Divergence Kernel ---
    checkCudaError(cudaEventRecord(start), "start");
    true_divergeKernel<<<grid, block>>>(d_x, d_y, N);
    checkCudaError(cudaEventRecord(stop), "stop");
    checkCudaError(cudaEventSynchronize(stop), "sync");
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    printf("True Divergence Kernel Time:   %f ms\n", ms);

    // --- 測試 Non-Divergence Kernel ---
    checkCudaError(cudaEventRecord(start), "start");
    true_nonDivergeKernel<<<grid, block>>>(d_x, d_y, N);
    checkCudaError(cudaEventRecord(stop), "stop");
    checkCudaError(cudaEventSynchronize(stop), "sync");
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    printf("True Non-Divergence Kernel Time: %f ms\n", ms);

    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    cudaFreeHost(h_x); cudaFreeHost(h_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}