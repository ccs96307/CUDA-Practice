#include <iostream>
#include <cuda_runtime.h>

#define M 1024
#define N 1024


__global__ void matAddOne(float *mat) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        mat[row * N + col] += 1;
    }
}


int main() {
    // Init
    float *h_mat = new float[M * N];
    int total = M * N;
    for (int i=0; i<total; ++i) {
        h_mat[i] = i - 1;
    }

    // Allocate device memory
    float *d_mat;
    cudaMalloc((void**)&d_mat, M * N * sizeof(float));

    cudaMemcpy(d_mat, h_mat, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (M + 31) / 32);

    matAddOne<<<gridDim, blockDim>>>(d_mat);

    cudaMemcpy(h_mat, d_mat, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<M; ++i) {
        std::cout << "Idx: " << i << " | Value: " << h_mat[i] << std::endl;
    }

    // Free
    cudaFree(d_mat);
    delete[] h_mat;

    return 0;
}