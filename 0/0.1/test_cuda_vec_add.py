from torch.utils.cpp_extension import load

cuda_ops = load(
    name="vec_ops",
    sources=[
        "vec_add_bind.cpp",
        "vec_add_kernel.cu",
    ],
    verbose=True,
)

import torch
import time

# CUDA
N = 51_200_000
a = torch.arange(N, dtype=torch.float32, device="cuda")
b = 2 * a
c = torch.empty_like(a)

start_time = time.time()
cuda_ops.fused_op(a, b, c)
torch.cuda.synchronize()
print(c[:5])
print(f"CUDA Spent: {time.time() - start_time:.4f} seconds")

# CPU
a_cpu = a.cpu()
b_cpu = b.cpu()
start_time = time.time()
c_cpu = torch.sin(a_cpu * b_cpu) + torch.log1p(a_cpu + b_cpu)
print(c_cpu[:5])
print(f"CPU Spent:  {time.time() - start_time:.4f} seconds")