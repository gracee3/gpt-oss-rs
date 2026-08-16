namespace {

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

struct GptOssRouteWireV1 {
    unsigned int source_row;
    unsigned int activation_slot;
    unsigned short expert_id;
    unsigned short weight_bf16_bits;
    unsigned char route_rank;
    unsigned char reserved[3];
};

static_assert(sizeof(GptOssRouteWireV1) == 16, "route wire v1 must remain 16 bytes");

// One thread owns one (source row, expert) logit. The 16 lane accumulation,
// lane fold, and bias placement match gpt-oss-cpu-kernels::scalar_bf16_dot
// followed by CpuModel::project_bf16's `destination += bias[row]`.
extern "C" __global__ void gpt_oss_router_bf16_projection_kernel(
    const unsigned short* input,
    const unsigned short* weights,
    const unsigned short* bias,
    unsigned short* logits,
    int rows,
    int experts,
    int hidden
) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_count = rows * experts;
    if (output_index >= output_count || hidden <= 0) {
        return;
    }
    const int source_row = output_index / experts;
    const int expert = output_index - source_row * experts;
    const unsigned short* input_row = input + source_row * hidden;
    const unsigned short* weight_row = weights + expert * hidden;

    float lanes[16] = {0.0f};
    for (int column = 0; column < hidden; ++column) {
        const float product = cpu_f32_mul(
            cpu_bf16_bits_to_f32(weight_row[column]),
            cpu_bf16_bits_to_f32(input_row[column]));
        const int lane = column & 15;
        lanes[lane] = cpu_f32_add(lanes[lane], product);
    }
    float lane_sum = 0.0f;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
        lane_sum = cpu_f32_add(lane_sum, lanes[lane]);
    }
    const float total = cpu_f32_add(lane_sum, cpu_bf16_bits_to_f32(bias[expert]));
    logits[output_index] = cpu_f32_to_bf16_bits(total);
}

// One thread owns all routing semantics for one source row. Selected IDs,
// rank order, and BF16 weights are GPU-authored. The host may copy these
// records for dispatch/evidence but never reconstructs them from logits.
extern "C" __global__ void gpt_oss_router_stable_top4_kernel(
    const unsigned short* logits,
    unsigned char* route_record_bytes,
    unsigned int* status,
    int rows,
    int experts
) {
    const int source_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_row >= rows) {
        return;
    }
    if (!((experts == 32) || (experts == 128))) {
        status[source_row] = 2U;
        return;
    }

    const float negative_infinity = __uint_as_float(0xff800000U);
    float top_values[4] = {
        negative_infinity, negative_infinity, negative_infinity, negative_infinity};
    int top_ids[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    const unsigned short* row_logits = logits + source_row * experts;
    for (int expert = 0; expert < experts; ++expert) {
        const float value = cpu_bf16_bits_to_f32(row_logits[expert]);
        if (!isfinite(value)) {
            status[source_row] = 1U;
            return;
        }
        int position = 4;
#pragma unroll
        for (int rank = 0; rank < 4; ++rank) {
            if (value > top_values[rank] ||
                (value == top_values[rank] && expert < top_ids[rank])) {
                position = rank;
                break;
            }
        }
        if (position < 4) {
            for (int rank = 3; rank > position; --rank) {
                top_values[rank] = top_values[rank - 1];
                top_ids[rank] = top_ids[rank - 1];
            }
            top_values[position] = value;
            top_ids[position] = expert;
        }
    }

    float exponentials[4];
    float denominator = 0.0f;
#pragma unroll
    for (int rank = 0; rank < 4; ++rank) {
        exponentials[rank] = expf(__fsub_rn(top_values[rank], top_values[0]));
        denominator = cpu_f32_add(denominator, exponentials[rank]);
    }
    GptOssRouteWireV1* route_records =
        reinterpret_cast<GptOssRouteWireV1*>(route_record_bytes);
    const int route_base = source_row * 4;
#pragma unroll
    for (int rank = 0; rank < 4; ++rank) {
        const float weight = __fdiv_rn(exponentials[rank], denominator);
        GptOssRouteWireV1& route = route_records[route_base + rank];
        route.source_row = static_cast<unsigned int>(source_row);
        route.activation_slot = static_cast<unsigned int>(source_row);
        route.expert_id = static_cast<unsigned short>(top_ids[rank]);
        route.weight_bf16_bits = cpu_f32_to_bf16_bits(weight);
        route.route_rank = static_cast<unsigned char>(rank);
        route.reserved[0] = 0U;
        route.reserved[1] = 0U;
        route.reserved[2] = 0U;
    }
    status[source_row] = 0U;
}
