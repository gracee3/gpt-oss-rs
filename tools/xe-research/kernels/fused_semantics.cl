// SPDX-License-Identifier: Apache-2.0
// Research-only fused-expert semantic boundary probe. This source is not part
// of the production Xe projection artifact or its ABI/cache identity.
#pragma OPENCL FP_CONTRACT OFF

inline ushort xe_bf16_bits_rne(const float value) {
    const uint bits = as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort)((bits >> 16) | 0x0040U);
    }
    if ((bits & 0x00008000U) != 0U && (bits & 0x00017fffU) != 0U) {
        return (ushort)((bits >> 16) + 1U);
    }
    return (ushort)(bits >> 16);
}

inline float xe_bf16_to_float(const ushort bits) {
    return as_float(((uint)bits) << 16);
}

inline float xe_bf16_roundtrip(const float value) {
    return xe_bf16_to_float(xe_bf16_bits_rne(value));
}

__kernel void xe_sigmoid_native_bf16(__global ushort *output) {
    const uint bits = (uint)get_global_id(0);
    const float scaled_gate = xe_bf16_to_float((ushort)bits);
    const float sigmoid = 1.0f / (1.0f + exp(-scaled_gate));
    output[bits] = xe_bf16_bits_rne(sigmoid);
}

__kernel void xe_fused_prepare_semantics(
    __global const float *gate_up,
    __global const ushort *sigmoid_lut,
    __global ushort *gate_up_bf16,
    __global ushort *native_swiglu_bf16,
    __global ushort *exact_swiglu_bf16,
    __global char *primary,
    __global char *residual,
    __global float *primary_scale,
    __global float *residual_scale,
    const uint intermediate,
    const uint count) {
    const uint block = (uint)get_global_id(0);
    const uint base = block * 32U;
    if (base >= count) {
        return;
    }

    float activated[32];
    float primary_max = 0.0f;
    for (uint lane = 0; lane < 32U; ++lane) {
        const uint index = base + lane;
        if (index >= count) {
            activated[lane] = 0.0f;
            continue;
        }
        const uint row = index / intermediate;
        const uint column = index - row * intermediate;
        const uint pair = (row * intermediate + column) * 2U;
        const ushort gate_bits = xe_bf16_bits_rne(gate_up[pair]);
        const ushort up_bits = xe_bf16_bits_rne(gate_up[pair + 1U]);
        gate_up_bf16[pair] = gate_bits;
        gate_up_bf16[pair + 1U] = up_bits;

        const float gate = fmin(xe_bf16_to_float(gate_bits), 7.0f);
        const float up = clamp(xe_bf16_to_float(up_bits), -7.0f, 7.0f);
        const ushort scaled_bits = xe_bf16_bits_rne(gate * 1.702f);
        const float scaled_gate = xe_bf16_to_float(scaled_bits);

        const float native_sigmoid = xe_bf16_roundtrip(
            1.0f / (1.0f + exp(-scaled_gate)));
        const float native_glu = xe_bf16_roundtrip(gate * native_sigmoid);
        const float linear = xe_bf16_roundtrip(up + 1.0f);
        const ushort native_output = xe_bf16_bits_rne(native_glu * linear);
        native_swiglu_bf16[index] = native_output;

        const float exact_sigmoid = xe_bf16_to_float(sigmoid_lut[scaled_bits]);
        const float exact_glu = xe_bf16_roundtrip(gate * exact_sigmoid);
        const ushort exact_output = xe_bf16_bits_rne(exact_glu * linear);
        exact_swiglu_bf16[index] = exact_output;
        activated[lane] = xe_bf16_to_float(exact_output);
        primary_max = fmax(primary_max, fabs(activated[lane]));
    }

    const float p_scale = primary_max / 127.0f;
    const float p_inverse = p_scale == 0.0f ? 0.0f : 1.0f / p_scale;
    float residual_values[32];
    float residual_max = 0.0f;
    for (uint lane = 0; lane < 32U; ++lane) {
        const uint index = base + lane;
        if (index >= count) {
            continue;
        }
        const float rounded = clamp(round(activated[lane] * p_inverse), -127.0f, 127.0f);
        const char quantized = convert_char_rtz(rounded);
        primary[index] = quantized;
        const float remainder = activated[lane] - convert_float(quantized) * p_scale;
        residual_values[lane] = remainder;
        residual_max = fmax(residual_max, fabs(remainder));
    }

    const float r_scale = residual_max / 127.0f;
    const float r_inverse = r_scale == 0.0f ? 0.0f : 1.0f / r_scale;
    for (uint lane = 0; lane < 32U; ++lane) {
        const uint index = base + lane;
        if (index >= count) {
            continue;
        }
        const float rounded = clamp(round(residual_values[lane] * r_inverse), -127.0f, 127.0f);
        residual[index] = convert_char_rtz(rounded);
    }
    primary_scale[block] = p_scale;
    residual_scale[block] = r_scale;
}
