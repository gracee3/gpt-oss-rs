// Half-precision RMSNorm kernel using __half and __half2 for throughput.
// Accumulation is done in f32 for numerical stability, but reads/writes
// are f16, halving memory bandwidth requirements.
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(hidden_size, 1024), 1, 1)
//   Shared memory: blockDim.x * sizeof(float)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C"
__global__ void rms_norm_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __half* x = input + token_idx * hidden_size;
    __half* y = output + token_idx * hidden_size;

    extern __shared__ float sdata[];

    // Pass 1: sum of squares in f32 for precision
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(x[i]);
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    // Parallel reduction
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    // Pass 2: normalize and scale, write f16
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(x[i]) * __half2float(weight[i]) * rms_scale;
        y[i] = __float2half(val);
    }
}

// BF16 policy RMSNorm variant for GPT-OSS compatibility work.
// Reads half storage through BF16 semantics, uses f32 pairwise/tree reduction,
// applies BF16-rounded weights, and rounds the output through BF16 before
// storing back into the existing half pipeline.
extern "C"
__global__ void rms_norm_f16_bf16_policy_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const float* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __half* x = input + token_idx * hidden_size;
    __half* y = output + token_idx * hidden_size;

    extern __shared__ float sdata[];
    float* src = sdata;
    float* dst = sdata + hidden_size;

    for (int i = tid; i < hidden_size; i += stride) {
        float val = __bfloat162float(__float2bfloat16(__half2float(x[i])));
        src[i] = val * val;
    }
    __syncthreads();

    int active = hidden_size;
    while (active > 1) {
        int pairs = active >> 1;
        int next_active = pairs + (active & 1);
        for (int pair = tid; pair < pairs; pair += stride) {
            dst[pair] = src[2 * pair] + src[2 * pair + 1];
        }
        if ((active & 1) && tid == 0) {
            dst[pairs] = src[active - 1];
        }
        __syncthreads();
        float* tmp = src;
        src = dst;
        dst = tmp;
        active = next_active;
        __syncthreads();
    }

    float rms_scale = 1.0f / sqrtf(src[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float x_bf16 = __bfloat162float(__float2bfloat16(__half2float(x[i])));
        float w_bf16 = __bfloat162float(__float2bfloat16(weight[i]));
        float val = (x_bf16 * rms_scale) * w_bf16;
        y[i] = __float2half(__bfloat162float(__float2bfloat16(val)));
    }
}

// Fused residual add + RMS norm, f16 variant.
extern "C"
__global__ void fused_residual_rmsnorm_f16_kernel(
    __half* __restrict__ output,
    __half* __restrict__ residual,
    const __half* __restrict__ input,
    const __half* __restrict__ add,
    const __half* __restrict__ weight,
    float eps,
    int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int row_offset = token_idx * hidden_size;

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(input[row_offset + i]) + __half2float(add[row_offset + i]);
        residual[row_offset + i] = __float2half(val);
        local_ss += val * val;
    }
    sdata[tid] = local_ss;
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_scale = rsqrtf(sdata[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float val = __half2float(residual[row_offset + i]) * __half2float(weight[i]) * rms_scale;
        output[row_offset + i] = __float2half(val);
    }
}
