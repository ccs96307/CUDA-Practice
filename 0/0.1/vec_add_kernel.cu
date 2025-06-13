// vec_add_kernel.cu
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be a CUDA-contiguous tensor")
#define THREADS 256


__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void fusedOpKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = A[i];
        float y = B[i];
        C[i] = sinf(x * y) + log1pf(x + y);
    }
}

void vecAdd(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int N = A.size(0);
    int blocks = (N + THREADS - 1) / THREADS;

    vecAddKernel<<<blocks, THREADS>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}


void fusedOp(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");

    int N = A.size(0);
    const int blocks = (N + THREADS - 1) / THREADS;

    fusedOpKernel<<<blocks, THREADS>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
}