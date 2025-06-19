#include<cuda_runtime.h>
#include<cuda_bf16.h>

#define NUM_THREADS 256


// x_i = (x1, x2, ..., xn)
// x_i = (xi - mean) / sqrt(var^2 + eps)
// y_i = gamma * x_i + beta
__launch_bounds__(256)
__global__ void customLayerNormKernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    float epsilon,
    int B,
    int H
) {
    // One block process one row
    int rowIdx = blockIdx.x;
    int tId = threadIdx.x;
    int blockSize = blockDim.x;

    // Shared Memory
    extern __shared__ __nv_bfloat16 shared_row[];
    for (int i=tId; i<H; i+=blockDim.x) {
        shared_row[i] = input[rowIdx * H + i];
    }

    // Check shared memory is loaded
    __syncthreads();

    // Declare a shared memory array to save partial_accum
    __shared__ float partial_accum[NUM_THREADS];
    __shared__ float partial_accum_sq[NUM_THREADS];

    float local_accum = 0.0f;
    float local_accum_sq = 0.0f;
    for (int i=tId; i<H; i+=blockSize) {
        float val = __bfloat162float(shared_row[i]);
        local_accum += val;
        local_accum_sq += val * val;
    }
    partial_accum[tId] = local_accum;
    partial_accum_sq[tId] = local_accum_sq;
    __syncthreads();

    // This will happen in order
    // offset = 16: threads 0-15 will get the sum of data from threads 16-31
    // offset = 8: threads 0-7 will get data from threads 8-15
    // ...
    // Finally, offset = 1: threads 0 will get data from threads 1
    for (int offset=blockSize/2; offset>0; offset>>=1) {
        if (tId < offset) {
            partial_accum[tId] += partial_accum[tId + offset];
            partial_accum_sq[tId] += partial_accum_sq[tId + offset];
        }
        __syncthreads();
    }

    // Calculate `mean` and `var`
    __shared__ float mean, var;
    if (tId == 0) {
        float accum = partial_accum[0];
        float accum_sq = partial_accum_sq[0];
        mean = accum / H;
        var = (accum_sq / H) - (mean * mean);
    }
    __syncthreads();

    // Calculate `x_i`
    float inv_std = rsqrtf(var + epsilon);
    for (int i=tId; i<H; i+=blockSize) {
        float val = __bfloat162float(shared_row[i]);
        float norm = (val - mean) * inv_std;
        float gamma_val = __bfloat162float(gamma[i]);
        float beta_val = __bfloat162float(beta[i]);

        output[rowIdx * H + i] = __float2bfloat16(gamma_val * norm + beta_val);
    }
}


void launch_layernorm_cuda(
    const __nv_bfloat16* input_ptr,
    __nv_bfloat16* output_ptr,
    const __nv_bfloat16* gamma_ptr,
    const __nv_bfloat16* beta_ptr,
    float epsilon,
    int B,
    int H
) {
    dim3 block(NUM_THREADS);
    dim3 grid(B);

    size_t shared_mem_bytes = H * sizeof(__nv_bfloat16);

    customLayerNormKernel<<<grid, block, shared_mem_bytes>>>(
        input_ptr, output_ptr, gamma_ptr, beta_ptr, epsilon, B, H
    );
}