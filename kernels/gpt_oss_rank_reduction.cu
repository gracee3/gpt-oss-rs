namespace {

// These conversions and arithmetic helpers intentionally match the scalar
// CPU authority and the selected-expert kernel. Native CUDA BF16 conversion
// canonicalizes NaNs and therefore is not an exact boundary for this project.
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

__device__ __forceinline__ float quiet_nan(float value) {
    return __uint_as_float(__float_as_uint(value) | 0x00400000U);
}

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

// Canonical arena layout is [row][route_rank][hidden]. One thread owns one
// output element and performs ranks 0,1,2,3 serially. Atomics, tree reduction,
// reassociation, and completion-order accumulation are impossible here.
extern "C" __global__ void gpt_oss_rank_order_reduce_bf16_kernel(
    const unsigned short* contributions,
    const unsigned short* weights,
    unsigned short* output,
    unsigned int* weighted_trace,
    unsigned int* accumulator_trace,
    int rows,
    int hidden
) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * hidden;
    if (linear >= total || rows <= 0 || hidden <= 0) {
        return;
    }

    const int row = linear / hidden;
    const int column = linear - row * hidden;
    float accumulator = 0.0f;

#pragma unroll
    for (int rank = 0; rank < 4; ++rank) {
        const int route = row * 4 + rank;
        const int contribution_index = route * hidden + column;
        const float value = cpu_bf16_bits_to_f32(contributions[contribution_index]);
        const float weight = cpu_bf16_bits_to_f32(weights[route]);
        const float weighted = cpu_f32_mul(value, weight);
        accumulator = cpu_f32_add(accumulator, weighted);
        weighted_trace[contribution_index] = __float_as_uint(weighted);
        accumulator_trace[contribution_index] = __float_as_uint(accumulator);
    }

    output[linear] = cpu_f32_to_bf16_bits(accumulator);
}
