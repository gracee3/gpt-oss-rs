namespace {

__device__ __forceinline__ float mxfp4_value(unsigned int code) {
    switch (code & 0x0fU) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        case 7: return 6.0f;
        case 8: return -0.0f;
        case 9: return -0.5f;
        case 10: return -1.0f;
        case 11: return -1.5f;
        case 12: return -2.0f;
        case 13: return -3.0f;
        case 14: return -4.0f;
        default: return -6.0f;
    }
}

__device__ __forceinline__ float e8m0_value(unsigned char code) {
    if (code == 0) {
        return __uint_as_float(0x00400000U);
    }
    if (code == 0xffU) {
        return __uint_as_float(0x7fc00000U);
    }
    return __uint_as_float(static_cast<unsigned int>(code) << 23);
}

// Match half 2.7.1's scalar f32<->BF16 conversion exactly. CUDA's native
// conversion canonicalizes every NaN to 0x7fff, while the CPU authority keeps
// the f32 sign/high payload bits and sets the quiet bit.
__device__ __forceinline__ unsigned short cpu_f32_to_bf16_bits(float value) {
    const unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<unsigned short>((bits >> 16) | 0x0040U);
    }
    constexpr unsigned int round_bit = 0x00008000U;
    unsigned int rounded = bits >> 16;
    if ((bits & round_bit) != 0U && (bits & (3U * round_bit - 1U)) != 0U) {
        ++rounded;
    }
    return static_cast<unsigned short>(rounded);
}

__device__ __forceinline__ float cpu_bf16_bits_to_f32(unsigned short bits) {
    if ((bits & 0x7fffU) > 0x7f80U) {
        bits = static_cast<unsigned short>(bits | 0x0040U);
    }
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

__device__ __forceinline__ float bf16_round(float value) {
    return cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(value));
}

__device__ __forceinline__ float quiet_nan(float value) {
    return __uint_as_float(__float_as_uint(value) | 0x00400000U);
}

// Match the scalar x86 f32 operations used by the CPU authority for NaN
// creation and operand-order propagation. NVIDIA arithmetic otherwise
// canonicalizes these cases to 0x7fffffff before the BF16 boundary.
__device__ __forceinline__ float cpu_f32_mul(float left, float right) {
    if (isnan(left)) {
        return quiet_nan(left);
    }
    if (isnan(right)) {
        return quiet_nan(right);
    }
    if ((isinf(left) && right == 0.0f) || (left == 0.0f && isinf(right))) {
        return __uint_as_float(0xffc00000U);
    }
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float cpu_f32_add(float left, float right) {
    if (isnan(left)) {
        return quiet_nan(left);
    }
    if (isnan(right)) {
        return quiet_nan(right);
    }
    if (isinf(left) && isinf(right) && signbit(left) != signbit(right)) {
        return __uint_as_float(0xffc00000U);
    }
    return __fadd_rn(left, right);
}

}  // namespace

// One thread owns one output row. Native packed bytes are indexed as
// [row][input_block][16 packed E2M1 pairs]. Each thread preserves the CPU
// authority's 16 independent FP32 lanes and its final lane-sum order.
extern "C" __global__ void gpt_oss_selected_mxfp4_bf16_gemv_kernel(
    const unsigned short* input,
    const unsigned char* blocks,
    const unsigned char* scales,
    const unsigned short* bias,
    unsigned short* output,
    int output_rows,
    int input_blocks
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output_rows || input_blocks <= 0) {
        return;
    }

    float lanes[16] = {0.0f};
    const int row_block_base = row * input_blocks;

    for (int block = 0; block < input_blocks; ++block) {
        const int block_index = row_block_base + block;
        const float scale = e8m0_value(scales[block_index]);
        const unsigned char* packed = blocks + block_index * 16;
        const unsigned short* input_block = input + block * 32;
#pragma unroll
        for (int pair = 0; pair < 16; ++pair) {
            const unsigned char byte = packed[pair];
            const int low_index = pair * 2;
            const int high_index = low_index + 1;
            const float low_weight = bf16_round(cpu_f32_mul(mxfp4_value(byte), scale));
            const float high_weight = bf16_round(cpu_f32_mul(mxfp4_value(byte >> 4), scale));
            const float low_product =
                cpu_f32_mul(low_weight, cpu_bf16_bits_to_f32(input_block[low_index]));
            const float high_product =
                cpu_f32_mul(high_weight, cpu_bf16_bits_to_f32(input_block[high_index]));
            lanes[low_index & 15] = cpu_f32_add(lanes[low_index & 15], low_product);
            lanes[high_index & 15] = cpu_f32_add(lanes[high_index & 15], high_product);
        }
    }

    float lane_sum = 0.0f;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
        lane_sum = cpu_f32_add(lane_sum, lanes[lane]);
    }
    const float total = cpu_f32_add(cpu_bf16_bits_to_f32(bias[row]), lane_sum);
    output[row] = cpu_f32_to_bf16_bits(total);
}

// GPT-OSS SwiGLU for the interleaved [gate, up] projection. Every tensor
// operation rounds back to BF16 exactly where the CPU semantic path does.
extern "C" __global__ void gpt_oss_selected_swiglu_bf16_kernel(
    const unsigned short* gate_up,
    unsigned short* output,
    unsigned short* scaled_gate_trace,
    unsigned short* sigmoid_trace,
    unsigned short* glu_trace,
    unsigned short* linear_trace,
    int intermediate,
    float alpha,
    float limit
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= intermediate) {
        return;
    }

    float gate = cpu_bf16_bits_to_f32(gate_up[index * 2]);
    float up = cpu_bf16_bits_to_f32(gate_up[index * 2 + 1]);
    // Rust f32::min selects the numeric limit for a NaN gate, while f32::clamp
    // preserves a NaN up value. CUDA fminf/fmaxf select numeric operands in
    // both cases, so only the clamp needs explicit NaN propagation.
    gate = fminf(gate, limit);
    up = isnan(up) ? up : fminf(fmaxf(up, -limit), limit);
    const float scaled_gate = bf16_round(cpu_f32_mul(gate, alpha));
    const float exponential = expf(-scaled_gate);
    const float sigmoid = bf16_round(__fdiv_rn(1.0f, cpu_f32_add(1.0f, exponential)));
    const float glu = bf16_round(cpu_f32_mul(gate, sigmoid));
    const float linear = bf16_round(cpu_f32_add(up, 1.0f));
    const float result = bf16_round(cpu_f32_mul(glu, linear));
    scaled_gate_trace[index] = cpu_f32_to_bf16_bits(scaled_gate);
    sigmoid_trace[index] = cpu_f32_to_bf16_bits(sigmoid);
    glu_trace[index] = cpu_f32_to_bf16_bits(glu);
    linear_trace[index] = cpu_f32_to_bf16_bits(linear);
    output[index] = cpu_f32_to_bf16_bits(result);
}
