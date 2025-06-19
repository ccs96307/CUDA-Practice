#include <torch/extension.h>
#include <cuda_bf16.h>


void launch_layernorm_cuda(
    const __nv_bfloat16* input_ptr,
    __nv_bfloat16* output_ptr,
    const __nv_bfloat16* gamma_ptr,
    const __nv_bfloat16* beta_ptr,
    float epsilon,
    int B,
    int H
);


void launch_layernorm(
    at::Tensor input,
    at::Tensor output,
    at::Tensor gamma,
    at::Tensor beta,
    float epsilon
) {
    int B = input.size(0);
    int H = input.size(1);

    __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    const __nv_bfloat16* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());
    const __nv_bfloat16* gamma_ptr = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>());
    const __nv_bfloat16* beta_ptr = reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>());

    launch_layernorm_cuda(input_ptr, output_ptr, gamma_ptr, beta_ptr, epsilon, B, H);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "LayerNorm",
        &launch_layernorm,
        "Optimized LayerNorm with bfloat16 (CUDA)"
    );
}
