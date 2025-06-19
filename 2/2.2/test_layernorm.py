import torch
import custom_layernorm_cuda
import torch.nn.functional as F
import time

# Configuration
B = 16384
H = 4096
eps = 1e-10

# Input tensors
x = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
gamma = torch.randn(H, dtype=torch.bfloat16, device="cuda")
beta = torch.randn(H, dtype=torch.bfloat16, device="cuda")

# Output from custom CUDA kernel
out_custom = torch.empty_like(x)
custom_layernorm_cuda.LayerNorm(x, out_custom, gamma, beta, eps)

# Reference output using PyTorch's native computation (manual implementation)
x_fp32 = x.to(torch.float32)
gamma_fp32 = gamma.to(torch.float32)
beta_fp32 = beta.to(torch.float32)

mean = x_fp32.mean(dim=-1, keepdim=True)
var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
normed = (x_fp32 - mean) / torch.sqrt(var + eps)
out_ref = normed * gamma_fp32 + beta_fp32
out_ref = out_ref.to(torch.bfloat16)

# Compare results
abs_diff = (out_ref - out_custom).abs()
max_diff = abs_diff.max().item()
print(f"[Max Error] Absolute difference: {max_diff:.5f}")

# Warm-up for custom kernel
for _ in range(10):
    custom_layernorm_cuda.LayerNorm(x, out_custom, gamma, beta, eps)

# Benchmark custom CUDA kernel
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    custom_layernorm_cuda.LayerNorm(x, out_custom, gamma, beta, eps)
torch.cuda.synchronize()
custom_time = time.time() - start

# Setup native PyTorch LayerNorm
layernorm_ref = torch.nn.LayerNorm(H, eps=eps).to("cuda").bfloat16()
layernorm_ref.weight.data.copy_(gamma)
layernorm_ref.bias.data.copy_(beta)

# Warm-up for PyTorch LayerNorm
for _ in range(10):
    _ = layernorm_ref(x)

# Benchmark PyTorch LayerNorm
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = layernorm_ref(x)
torch.cuda.synchronize()
torch_time = time.time() - start

print(f"[Execution Time] Custom kernel: {custom_time:.4f}s")
print(f"[Execution Time] PyTorch LayerNorm: {torch_time:.4f}s")
