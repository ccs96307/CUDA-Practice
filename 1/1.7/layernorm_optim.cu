#include <stdio.h>
#include <random>
#include<cuda_runtime.h>
#include<cuda_bf16.h>

#define NUM_THREADS 256
#define NUM_STREAMS 4
#define B 16384
#define H 4096

#if (B * H) % NUM_STREAMS != 0
#error (B * H) must be divisible by NUM_STREAMS
#endif

#define CHUNK_ROWS (B / NUM_STREAMS)
#define CHUNK_SIZE (CHUNK_ROWS * H)
#define CHUNK_BYTES (CHUNK_SIZE * sizeof(__nv_bfloat16))
// #define CHUNK_BYTES (CHUNK_SIZE * sizeof(float))


// x_i = (x1, x2, ..., xn)
// x_i = (xi - mean) / sqrt(var^2 + eps)
// y_i = gamma * x_i + beta
__launch_bounds__(256)
__global__ void LayerNormKernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, const __nv_bfloat16* __restrict__ gamma, const __nv_bfloat16* __restrict__ beta, float epsilon) {
    // One block process one row
    int rowIdx = blockIdx.x;
    int tId = threadIdx.x;
    int blockSize = blockDim.x;

    // Shared Memory
    __shared__ __nv_bfloat16 shared_row[H];
    for (int i=tId; i<H; i+=blockDim.x) {
        shared_row[i] = input[rowIdx * H + i];
    }

    // Check shared memory is loaded
    __syncthreads();

    // Declare a shared memory array to save partial_sums
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


static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


float getRand(std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    return dist(gen);
}


int main() {
    // Init
    int sizeMat = B * H * sizeof(__nv_bfloat16);
    int sizeArr = H * sizeof(__nv_bfloat16);

    float epsilon = 1e-10;
    __nv_bfloat16 *h_input, *h_output, *h_gamma, *h_beta;

    // Malloc host data
    checkCudaError(cudaMallocHost((void**)&h_input, sizeMat), "cudaMallocHost h_input");
    checkCudaError(cudaMallocHost((void**)&h_output, sizeMat), "cudaMallocHost h_output");
    checkCudaError(cudaMallocHost((void**)&h_gamma, sizeArr), "cudaMallocHost h_gamma");
    checkCudaError(cudaMallocHost((void**)&h_beta, sizeArr), "cudaMallocHost h_beta");
    
    // Random initialization
    // Init random number generator and distribution
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (int i=0; i<H; ++i) {
        h_gamma[i] = __float2bfloat16_rn(getRand(gen, dist));
        h_beta[i] = __float2bfloat16_rn(getRand(gen, dist));

        for (int j=0; j<B; ++j) {
            h_input[j * H + i] = __float2bfloat16_rn(getRand(gen, dist));
        }
    }

    // Init device data
    __nv_bfloat16 *d_input, *d_output, *d_gamma, *d_beta;
    checkCudaError(cudaMalloc((void**)&d_input, sizeMat), "cudaMalloc d_input");
    checkCudaError(cudaMalloc((void**)&d_output, sizeMat), "cudaMalloc d_output");
    checkCudaError(cudaMalloc((void**)&d_gamma, sizeArr), "cudaMalloc d_gamma");
    checkCudaError(cudaMalloc((void**)&d_beta, sizeArr), "cudaMalloc d_beta");

    // Create Streams and Events
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t doneEvents[NUM_STREAMS];
    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        checkCudaError(cudaEventCreate(&doneEvents[i]), "cudaEventCreate");
    }
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");


    // Copy gamma and beta
    checkCudaError(cudaMemcpy(d_gamma, h_gamma, sizeArr, cudaMemcpyHostToDevice), "cudaMemcpy h_gamma => d_gamma");
    checkCudaError(cudaMemcpy(d_beta, h_beta, sizeArr, cudaMemcpyHostToDevice), "cudaMemcpy h_beta => d_beta");

    // Start to record
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

    // Assign tasks to different streams
    for (int i=0; i<NUM_STREAMS; ++i) {
        int offset = i * CHUNK_SIZE;

        // Copy data from host to device
        checkCudaError(cudaMemcpyAsync(d_input + offset, h_input + offset, CHUNK_BYTES, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpy d_input => h_input");

        // Launch LayerNorm Kernel
        dim3 block(NUM_THREADS);
        dim3 grid(CHUNK_ROWS);
        LayerNormKernel<<<grid, block, 0, streams[i]>>>(d_input + offset, d_output + offset, d_gamma, d_beta, epsilon);
        checkCudaError(cudaGetLastError(), "LayerNormKernel launched");

        // Record the doneEvent of stream
        checkCudaError(cudaEventRecord(doneEvents[i], streams[i]), "cudaEventRecord doneEvent");
    }
    
    // Sync all operations
    for (int i = 0; i < NUM_STREAMS; ++i) {
        checkCudaError(cudaEventSynchronize(doneEvents[i]), "cudaEventSynchronize doneEvent");
    }

    // Stop to record
    checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    // Print timing
    float ms = 0.0f;
    checkCudaError(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("Optimal Kernel Spend Time: %f ms\n", ms);

    // Destroy
    checkCudaError(cudaFreeHost(h_input), "cudaFree h_input");
    checkCudaError(cudaFreeHost(h_output), "cudaFree h_output");
    checkCudaError(cudaFreeHost(h_gamma), "cudaFree h_gamma");
    checkCudaError(cudaFreeHost(h_beta), "cudaFree h_beta");

    for (int i=0; i<NUM_STREAMS; ++i) {
        checkCudaError(cudaStreamDestroy(streams[i]), "cudaStreamDestroy streams");
        checkCudaError(cudaEventDestroy(doneEvents[i]), "cudaEventDestroy events");        
    }
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");
    checkCudaError(cudaFree(d_input), "cudaFree d_input");
    checkCudaError(cudaFree(d_output), "cudaFree d_output");
    checkCudaError(cudaFree(d_gamma), "cudaFree d_gamma");
    checkCudaError(cudaFree(d_beta), "cudaFree d_beta");

    return 0;
}