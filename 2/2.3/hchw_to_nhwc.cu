#include <iostream>
#include <cuda_runtime.h>

#define N 32
#define C 4
#define H 256
#define W 512

#define TILE_H 8
#define TILE_W 32


__launch_bounds__(256)
__global__ void nchw_to_nhwc_kernel(const float* __restrict__ input, 
                         float* __restrict__ output, 
                         int N,
                         int C,
                         int H,
                         int W) {
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
        fprintf()
    }
}


int main() {
    // Init

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W,
              (H + TILE_H - 1) / TILE_H,
              N);

    nchw_to_nhwc_kernel<<<grid, block>>>(input, output, N, C, H, W);

    return 0;
}