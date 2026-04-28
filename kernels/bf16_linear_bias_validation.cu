// Validation-only BF16 biased linear kernel.
//
// This kernel is intentionally correctness-first and is only launched by the
// qkv_projection_policy_compare bench harness. It is not production routing and
// does not replace cuBLAS projection paths.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" __global__ void bf16_linear_bias_validation_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int m,
    int n,
    int k,
    int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) {
        return;
    }

    int token = idx / n;
    int feature = idx - token * n;

    float sum = 0.0f;
    const __nv_bfloat16* input_row = input + token * k;
    const __nv_bfloat16* weight_row = weight + feature * k;
    for (int h = 0; h < k; ++h) {
        sum += __bfloat162float(input_row[h]) * __bfloat162float(weight_row[h]);
    }

    // mode 0: f32 accumulation + f32 bias add + BF16 output.
    // mode 1: BF16-round pre-bias, then f32 bias add + BF16 output.
    if (mode == 1) {
        sum = __bfloat162float(__float2bfloat16(sum));
    }
    float out = sum + __bfloat162float(bias[feature]);
    output[idx] = __float2bfloat16(out);
}
