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

// Layer-0 attention RMSNorm compatibility path: emulate official BF16
// input/weight/output semantics while storing the BF16-valued output in a
// half buffer for the existing f16 pipeline.
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

extern "C"
__global__ void rms_norm_f16_bf16_policy_debug_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ debug,
    int target_token,
    int target_lane,
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

    if (token_idx == target_token && tid == 0) {
        debug[0] = src[0];
    }

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

    float sum_sq = src[0];
    float mean_square = sum_sq / (float)hidden_size;
    float mean_square_plus_epsilon = mean_square + eps;
    float rms_scale = 1.0f / sqrtf(mean_square_plus_epsilon);

    if (token_idx == target_token && tid == 0) {
        debug[1] = sum_sq;
        debug[2] = mean_square;
        debug[3] = mean_square_plus_epsilon;
        debug[4] = rms_scale;
    }

    for (int i = tid; i < hidden_size; i += stride) {
        float raw_x = __half2float(x[i]);
        float x_bf16 = __bfloat162float(__float2bfloat16(raw_x));
        float raw_w = weight[i];
        float w_bf16 = __bfloat162float(__float2bfloat16(raw_w));
        float x_times_scale = x_bf16 * rms_scale;
        float x_times_weight = x_bf16 * w_bf16;
        float val = x_times_scale * w_bf16;
        float output_bf16 = __bfloat162float(__float2bfloat16(val));
        __half stored = __float2half(output_bf16);
        y[i] = stored;

        if (token_idx == target_token && i == target_lane) {
            debug[5] = raw_x;
            debug[6] = x_bf16;
            debug[7] = raw_w;
            debug[8] = w_bf16;
            debug[9] = x_times_scale;
            debug[10] = x_times_weight;
            debug[11] = val;
            debug[12] = val;
            debug[13] = output_bf16;
            debug[14] = __half2float(stored);
        }
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
