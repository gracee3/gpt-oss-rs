// SPDX-License-Identifier: Apache-2.0
// Research-only exact MXFP4 kernels. Canonical checkpoint bytes use adjacent
// values packed low nibble first. Integer LUT entries are exactly twice E2M1.
#pragma OPENCL FP_CONTRACT OFF

constant char XE_E2M1_X2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

inline float xe_e8m0(const uchar scale) {
    if (scale == (uchar)0) {
        return as_float((uint)0x00400000);
    }
    if (scale == (uchar)0xff) {
        return as_float((uint)0x7fc00000);
    }
    return as_float(((uint)scale) << 23);
}

inline int2 xe_block_dots(
    __global const uchar *packed,
    __global const char *primary,
    __global const char *residual) {
    int primary_sum = 0;
    int residual_sum = 0;
    for (uint byte_index = 0; byte_index < 16; ++byte_index) {
        const uchar byte = packed[byte_index];
        const char low = XE_E2M1_X2[byte & (uchar)0x0f];
        const char high = XE_E2M1_X2[byte >> 4];
        const uint value_index = byte_index * 2;
        primary_sum += (int)low * (int)primary[value_index];
        primary_sum += (int)high * (int)primary[value_index + 1];
        residual_sum += (int)low * (int)residual[value_index];
        residual_sum += (int)high * (int)residual[value_index + 1];
    }
    return (int2)(primary_sum, residual_sum);
}

__kernel void mxfp4_exact_blocks(
    __global const uchar *packed,
    __global const uchar *weight_scales,
    __global const char *primary,
    __global const char *residual,
    __global const float *primary_scales,
    __global const float *residual_scales,
    __global int *primary_integer,
    __global int *residual_integer,
    __global float *q8_output,
    __global float *residual_q8_output,
    const uint count) {
    const size_t index = get_global_id(0);
    if (index >= count) {
        return;
    }
    const int2 dots = xe_block_dots(
        packed + index * 16, primary + index * 32, residual + index * 32);
    primary_integer[index] = dots.x;
    residual_integer[index] = dots.y;
    const float weight_scale = 0.5f * xe_e8m0(weight_scales[index]);
    const float primary_value = (float)dots.x * weight_scale * primary_scales[index];
    q8_output[index] = primary_value;
    residual_q8_output[index] =
        primary_value + (float)dots.y * weight_scale * residual_scales[index];
}

__kernel void mxfp4_project_scalar(
    __global const uchar *packed,
    __global const uchar *weight_scales,
    __global const char *primary,
    __global const char *residual,
    __global const float *primary_scales,
    __global const float *residual_scales,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const size_t linear = get_global_id(0);
    const size_t output_count = (size_t)rows * (size_t)columns;
    if (linear >= output_count) {
        return;
    }
    const uint row = (uint)(linear / columns);
    const uint column = (uint)(linear - (size_t)row * columns);
    float total = bias[column];
    for (uint block = 0; block < blocks; ++block) {
        const size_t weight_record = (size_t)column * blocks + block;
        const size_t activation_record = (size_t)row * blocks + block;
        const int2 dots = xe_block_dots(
            packed + weight_record * 16,
            primary + activation_record * 32,
            residual + activation_record * 32);
        const float weight_scale = 0.5f * xe_e8m0(weight_scales[weight_record]);
        total += (float)dots.x * weight_scale * primary_scales[activation_record];
        total += (float)dots.y * weight_scale * residual_scales[activation_record];
    }
    output[linear] = total;
}

#if defined(XE_ENABLE_DP4A) && defined(__opencl_c_integer_dot_product_input_4x8bit)
inline int2 xe_block_dots_dp4a(
    __global const uchar *packed,
    __global const char *primary,
    __global const char *residual) {
    int primary_sum = 0;
    int residual_sum = 0;
    for (uint byte_index = 0; byte_index < 16; byte_index += 2) {
        const uchar first = packed[byte_index];
        const uchar second = packed[byte_index + 1];
        const char4 weights = (char4)(
            XE_E2M1_X2[first & (uchar)0x0f],
            XE_E2M1_X2[first >> 4],
            XE_E2M1_X2[second & (uchar)0x0f],
            XE_E2M1_X2[second >> 4]);
        const char4 p = vload4(byte_index / 2, primary);
        const char4 r = vload4(byte_index / 2, residual);
        primary_sum += dot(weights, p);
        residual_sum += dot(weights, r);
    }
    return (int2)(primary_sum, residual_sum);
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void mxfp4_project_dp4a_sg8(
    __global const uchar *packed,
    __global const uchar *weight_scales,
    __global const char *primary,
    __global const char *residual,
    __global const float *primary_scales,
    __global const float *residual_scales,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const size_t linear = get_global_id(0);
    if (linear >= (size_t)rows * (size_t)columns) return;
    const uint row = (uint)(linear / columns);
    const uint column = (uint)(linear - (size_t)row * columns);
    float total = bias[column];
    for (uint block = 0; block < blocks; ++block) {
        const size_t w = (size_t)column * blocks + block;
        const size_t a = (size_t)row * blocks + block;
        const int2 dots = xe_block_dots_dp4a(
            packed + w * 16, primary + a * 32, residual + a * 32);
        const float ws = 0.5f * xe_e8m0(weight_scales[w]);
        total += (float)dots.x * ws * primary_scales[a];
        total += (float)dots.y * ws * residual_scales[a];
    }
    output[linear] = total;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mxfp4_project_dp4a_sg16(
    __global const uchar *packed,
    __global const uchar *weight_scales,
    __global const char *primary,
    __global const char *residual,
    __global const float *primary_scales,
    __global const float *residual_scales,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const size_t linear = get_global_id(0);
    if (linear >= (size_t)rows * (size_t)columns) return;
    const uint row = (uint)(linear / columns);
    const uint column = (uint)(linear - (size_t)row * columns);
    float total = bias[column];
    for (uint block = 0; block < blocks; ++block) {
        const size_t w = (size_t)column * blocks + block;
        const size_t a = (size_t)row * blocks + block;
        const int2 dots = xe_block_dots_dp4a(
            packed + w * 16, primary + a * 32, residual + a * 32);
        const float ws = 0.5f * xe_e8m0(weight_scales[w]);
        total += (float)dots.x * ws * primary_scales[a];
        total += (float)dots.y * ws * residual_scales[a];
    }
    output[linear] = total;
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void mxfp4_project_dp4a_sg32(
    __global const uchar *packed,
    __global const uchar *weight_scales,
    __global const char *primary,
    __global const char *residual,
    __global const float *primary_scales,
    __global const float *residual_scales,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const size_t linear = get_global_id(0);
    if (linear >= (size_t)rows * (size_t)columns) return;
    const uint row = (uint)(linear / columns);
    const uint column = (uint)(linear - (size_t)row * columns);
    float total = bias[column];
    for (uint block = 0; block < blocks; ++block) {
        const size_t w = (size_t)column * blocks + block;
        const size_t a = (size_t)row * blocks + block;
        const int2 dots = xe_block_dots_dp4a(
            packed + w * 16, primary + a * 32, residual + a * 32);
        const float ws = 0.5f * xe_e8m0(weight_scales[w]);
        total += (float)dots.x * ws * primary_scales[a];
        total += (float)dots.y * ws * residual_scales[a];
    }
    output[linear] = total;
}
#endif
