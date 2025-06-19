import torch
from torch.utils.cpp_extension import load_inline
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="The given code was built .* with PyTorch CXX11 ABI flag set")


cpp_src = """
void add(const torch::Tensor& x, const torch::Tensor& y, torch::Tensor& out);
"""

cuda_src = """#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void addKernel(const float* x, const float* y, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        out[idx] = x[idx] + y[idx];
    }
}

void add(const torch::Tensor& x, const torch::Tensor& y, torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.numel() == y.numel() && x.numel() == out.numel(), "All tensors must have the same number of elements");
    
    int N = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    const float* y_ptr = y.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    addKernel<<<blocks, threads>>>(x_ptr, y_ptr, out_ptr, N);

    C10_CUDA_CHECK(cudaGetLastError());
}
"""


ext = load_inline(
    name="addKernel",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["add"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
    build_directory="/tmp/"
)


# Testing
N = 1024
x = torch.randn(N, device="cuda:0")
y = torch.randn(N, device="cuda:0")
out = torch.empty(N, device="cuda:0")

ext.add(x, y, out)


print("Output from custom CUDA kernel:")
print(out[:10])

print("\nOutput from PyTorch's built-in addition:")
pytorch_out = x + y
print(pytorch_out[:10])

assert torch.allclose(out, pytorch_out), "Mismatch between CUDA and PyTorch results!"
print("\n✅ Results match!")