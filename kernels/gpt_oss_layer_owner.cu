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

extern "C" __global__ void gpt_oss_layer_embedding_kernel(
    const unsigned short* embeddings,
    unsigned short* hidden,
    int token_id,
    int hidden_size
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < hidden_size) {
        hidden[column] = embeddings[token_id * hidden_size + column];
    }
}

// One thread preserves the CPU authority's fixed 16-lane square reduction,
// lane fold, inverse RMS, multiply order, and BF16 output boundary.
extern "C" __global__ void gpt_oss_layer_rms_norm_kernel(
    const unsigned short* input,
    const unsigned short* weight,
    unsigned short* output,
    int hidden_size,
    float epsilon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || hidden_size <= 0) {
        return;
    }
    float lanes[16] = {0.0f};
    for (int column = 0; column < hidden_size; ++column) {
        const float value = cpu_bf16_bits_to_f32(input[column]);
        lanes[column & 15] = cpu_f32_add(
            lanes[column & 15], cpu_f32_mul(value, value));
    }
    float sum = 0.0f;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
        sum = cpu_f32_add(sum, lanes[lane]);
    }
    const float mean = __fdiv_rn(sum, static_cast<float>(hidden_size));
    const float inverse_rms = __fdiv_rn(1.0f, sqrtf(cpu_f32_add(mean, epsilon)));
    for (int column = 0; column < hidden_size; ++column) {
        const float scaled = cpu_f32_mul(
            cpu_f32_mul(cpu_bf16_bits_to_f32(input[column]), inverse_rms),
            cpu_bf16_bits_to_f32(weight[column]));
        output[column] = cpu_f32_to_bf16_bits(scaled);
    }
}

// One thread owns one output row. This is deliberately the same serial
// 16-lane arithmetic contract as the exact H4 router projection.
extern "C" __global__ void gpt_oss_layer_bf16_projection_kernel(
    const unsigned short* input,
    const unsigned short* weights,
    const unsigned short* bias,
    unsigned short* output,
    int rows,
    int columns
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || columns <= 0) {
        return;
    }
    const unsigned short* weight_row = weights + row * columns;
    float lanes[16] = {0.0f};
    for (int column = 0; column < columns; ++column) {
        const float product = cpu_f32_mul(
            cpu_bf16_bits_to_f32(weight_row[column]),
            cpu_bf16_bits_to_f32(input[column]));
        lanes[column & 15] = cpu_f32_add(lanes[column & 15], product);
    }
    float sum = 0.0f;
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
        sum = cpu_f32_add(sum, lanes[lane]);
    }
    output[row] = cpu_f32_to_bf16_bits(
        cpu_f32_add(sum, cpu_bf16_bits_to_f32(bias[row])));
}

extern "C" __global__ void gpt_oss_layer_rope_kernel(
    unsigned short* values,
    const unsigned short* cosine,
    const unsigned short* sine,
    int heads,
    int head_dim
) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = head_dim / 2;
    const int pair_count = heads * half;
    if (pair >= pair_count) {
        return;
    }
    const int head = pair / half;
    const int index = pair - head * half;
    const int left_index = head * head_dim + index;
    const int right_index = left_index + half;
    const float left = cpu_bf16_bits_to_f32(values[left_index]);
    const float right = cpu_bf16_bits_to_f32(values[right_index]);
    const float cos_value = cpu_bf16_bits_to_f32(cosine[index]);
    const float sin_value = cpu_bf16_bits_to_f32(sine[index]);
    const float left_cosine = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(
        cpu_f32_mul(left, cos_value)));
    const float right_sine = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(
        cpu_f32_mul(right, sin_value)));
    const float right_cosine = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(
        cpu_f32_mul(right, cos_value)));
    const float left_sine = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(
        cpu_f32_mul(left, sin_value)));
    values[left_index] = cpu_f32_to_bf16_bits(
        cpu_f32_add(left_cosine, -right_sine));
    values[right_index] = cpu_f32_to_bf16_bits(
        cpu_f32_add(right_cosine, left_sine));
}

extern "C" __global__ void gpt_oss_layer_append_kv_kernel(
    const unsigned short* key,
    const unsigned short* value,
    unsigned short* keys,
    unsigned short* values,
    int token,
    int width
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < width) {
        keys[token * width + column] = key[column];
        values[token * width + column] = value[column];
    }
}

// One thread owns one attention head. Score and value reductions are serial
// in token/dimension order, matching CPU attention_one_staged. The bounded
// decode proof supports no more than 128 visible rows.
extern "C" __global__ void gpt_oss_layer_attention_kernel(
    const unsigned short* query,
    const unsigned short* keys,
    const unsigned short* values,
    const unsigned short* sinks,
    unsigned short* output,
    int visible_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= num_heads || visible_tokens <= 0 || visible_tokens > 128) {
        return;
    }
    const int groups = num_heads / num_kv_heads;
    const int kv_head = head / groups;
    const float scale = __fdiv_rn(1.0f, sqrtf(static_cast<float>(head_dim)));
    float scores[128];
    float maximum = cpu_bf16_bits_to_f32(sinks[head]);
    for (int token = 0; token < visible_tokens; ++token) {
        float dot = 0.0f;
        const unsigned short* key = keys + token * num_kv_heads * head_dim
            + kv_head * head_dim;
        const unsigned short* q = query + head * head_dim;
        for (int column = 0; column < head_dim; ++column) {
            dot = cpu_f32_add(dot, cpu_f32_mul(
                cpu_bf16_bits_to_f32(q[column]),
                cpu_bf16_bits_to_f32(key[column])));
        }
        dot = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(dot));
        scores[token] = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(
            cpu_f32_mul(dot, scale)));
        maximum = fmaxf(maximum, scores[token]);
    }
    float denominator = expf(__fsub_rn(
        cpu_bf16_bits_to_f32(sinks[head]), maximum));
    for (int token = 0; token < visible_tokens; ++token) {
        denominator = cpu_f32_add(
            denominator, expf(__fsub_rn(scores[token], maximum)));
    }
    for (int column = 0; column < head_dim; ++column) {
        float accumulated = 0.0f;
        for (int token = 0; token < visible_tokens; ++token) {
            const float probability = cpu_bf16_bits_to_f32(cpu_f32_to_bf16_bits(
                __fdiv_rn(expf(__fsub_rn(scores[token], maximum)), denominator)));
            const int value_index = token * num_kv_heads * head_dim
                + kv_head * head_dim + column;
            accumulated = cpu_f32_add(accumulated, cpu_f32_mul(
                probability, cpu_bf16_bits_to_f32(values[value_index])));
        }
        output[head * head_dim + column] = cpu_f32_to_bf16_bits(accumulated);
    }
}

extern "C" __global__ void gpt_oss_layer_residual_kernel(
    const unsigned short* residual,
    const unsigned short* update,
    unsigned short* output,
    int values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < values) {
        output[index] = cpu_f32_to_bf16_bits(cpu_f32_add(
            cpu_bf16_bits_to_f32(residual[index]),
            cpu_bf16_bits_to_f32(update[index])));
    }
}
