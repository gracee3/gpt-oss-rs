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

// ABI v2: one compact Xe-derived weight representation replaces the canonical
// packed/scales pair on device. A subgroup lane reads one output column; every
// plane access is contiguous across the exact 32-lane subgroup.
#if defined(XE_ENABLE_DP4A) && defined(__opencl_c_integer_dot_product_input_4x8bit)
#define XE_V2_TILE 32u
#define XE_V2_PLANES 17u
#define XE_V2_ACTIVATION_BYTES 72u

inline size_t xe_v2_weight_base(const uint tile, const uint block, const uint blocks) {
    return ((size_t)tile * blocks + block) * XE_V2_PLANES * XE_V2_TILE;
}

inline char4 xe_v2_weights4(
    __global const uchar *weights,
    const size_t base,
    const uint lane,
    const uint byte_index) {
    const uchar first = weights[base + (byte_index + 1u) * XE_V2_TILE + lane];
    const uchar second = weights[base + (byte_index + 2u) * XE_V2_TILE + lane];
    return (char4)(
        XE_E2M1_X2[first & (uchar)0x0f],
        XE_E2M1_X2[first >> 4],
        XE_E2M1_X2[second & (uchar)0x0f],
        XE_E2M1_X2[second >> 4]);
}

inline int2 xe_v2_block_dots(
    __global const uchar *weights,
    const size_t weight_base,
    const uint lane,
    __global const uchar *activation) {
    __global const char *primary = (__global const char *)activation;
    __global const char *residual = (__global const char *)(activation + 32u);
    int primary_sum = 0;
    int residual_sum = 0;
    for (uint byte_index = 0; byte_index < 16u; byte_index += 2u) {
        const char4 decoded = xe_v2_weights4(weights, weight_base, lane, byte_index);
        primary_sum += dot(decoded, vload4(byte_index / 2u, primary));
        residual_sum += dot(decoded, vload4(byte_index / 2u, residual));
    }
    return (int2)(primary_sum, residual_sum);
}

inline float2 xe_v2_activation_scales(__global const uchar *activation) {
    return vload2(0, (__global const float *)(activation + 64u));
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void mxfp4_tile32_m1_v2(
    __global const uchar *weights,
    __global const uchar *activations,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const uint column = (uint)get_global_id(0);
    const uint row = (uint)get_global_id(1);
    if (column >= columns || row >= rows) return;
    const uint tile = column >> 5;
    const uint lane = column & 31u;
    float total = bias[column];
    for (uint block = 0; block < blocks; ++block) {
        const size_t weight_base = xe_v2_weight_base(tile, block, blocks);
        __global const uchar *activation =
            activations + ((size_t)row * blocks + block) * XE_V2_ACTIVATION_BYTES;
        const int2 dots = xe_v2_block_dots(weights, weight_base, lane, activation);
        const float2 scales = xe_v2_activation_scales(activation);
        const float weight_scale =
            0.5f * xe_e8m0(weights[weight_base + lane]);
        total += (float)dots.x * weight_scale * scales.x;
        total += (float)dots.y * weight_scale * scales.y;
    }
    output[(size_t)row * columns + column] = total;
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void mxfp4_tile32_m2_v2(
    __global const uchar *weights,
    __global const uchar *activations,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const uint column = (uint)get_global_id(0);
    const uint row = (uint)get_global_id(1) * 2u;
    if (column >= columns || row + 1u >= rows) return;
    const uint tile = column >> 5;
    const uint lane = column & 31u;
    float2 totals = (float2)(bias[column]);
    for (uint block = 0; block < blocks; ++block) {
        const size_t weight_base = xe_v2_weight_base(tile, block, blocks);
        const float weight_scale =
            0.5f * xe_e8m0(weights[weight_base + lane]);
        __global const uchar *a0 =
            activations + ((size_t)row * blocks + block) * XE_V2_ACTIVATION_BYTES;
        __global const uchar *a1 = a0 + (size_t)blocks * XE_V2_ACTIVATION_BYTES;
        __global const char *p0 = (__global const char *)a0;
        __global const char *r0 = (__global const char *)(a0 + 32u);
        __global const char *p1 = (__global const char *)a1;
        __global const char *r1 = (__global const char *)(a1 + 32u);
        int2 primary_dots = (int2)(0);
        int2 residual_dots = (int2)(0);
        for (uint byte_index = 0; byte_index < 16u; byte_index += 2u) {
            const char4 decoded = xe_v2_weights4(weights, weight_base, lane, byte_index);
            const uint vector_index = byte_index / 2u;
            primary_dots += (int2)(
                dot(decoded, vload4(vector_index, p0)),
                dot(decoded, vload4(vector_index, p1)));
            residual_dots += (int2)(
                dot(decoded, vload4(vector_index, r0)),
                dot(decoded, vload4(vector_index, r1)));
        }
        const float2 s0 = xe_v2_activation_scales(a0);
        const float2 s1 = xe_v2_activation_scales(a1);
        totals += convert_float2(primary_dots) * weight_scale * (float2)(s0.x, s1.x);
        totals += convert_float2(residual_dots) * weight_scale * (float2)(s0.y, s1.y);
    }
    output[(size_t)row * columns + column] = totals.x;
    output[(size_t)(row + 1u) * columns + column] = totals.y;
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void mxfp4_tile32_m4_v2(
    __global const uchar *weights,
    __global const uchar *activations,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const uint column = (uint)get_global_id(0);
    const uint row = (uint)get_global_id(1) * 4u;
    if (column >= columns || row + 3u >= rows) return;
    const uint tile = column >> 5;
    const uint lane = column & 31u;
    float4 totals = (float4)(bias[column]);
    for (uint block = 0; block < blocks; ++block) {
        const size_t weight_base = xe_v2_weight_base(tile, block, blocks);
        const float weight_scale =
            0.5f * xe_e8m0(weights[weight_base + lane]);
        __global const uchar *a0 =
            activations + ((size_t)row * blocks + block) * XE_V2_ACTIVATION_BYTES;
        __global const uchar *a1 = a0 + (size_t)blocks * XE_V2_ACTIVATION_BYTES;
        __global const uchar *a2 = a1 + (size_t)blocks * XE_V2_ACTIVATION_BYTES;
        __global const uchar *a3 = a2 + (size_t)blocks * XE_V2_ACTIVATION_BYTES;
        __global const char *p0 = (__global const char *)a0;
        __global const char *p1 = (__global const char *)a1;
        __global const char *p2 = (__global const char *)a2;
        __global const char *p3 = (__global const char *)a3;
        __global const char *r0 = (__global const char *)(a0 + 32u);
        __global const char *r1 = (__global const char *)(a1 + 32u);
        __global const char *r2 = (__global const char *)(a2 + 32u);
        __global const char *r3 = (__global const char *)(a3 + 32u);
        int4 primary_dots = (int4)(0);
        int4 residual_dots = (int4)(0);
        for (uint byte_index = 0; byte_index < 16u; byte_index += 2u) {
            const char4 decoded = xe_v2_weights4(weights, weight_base, lane, byte_index);
            const uint vector_index = byte_index / 2u;
            primary_dots += (int4)(
                dot(decoded, vload4(vector_index, p0)),
                dot(decoded, vload4(vector_index, p1)),
                dot(decoded, vload4(vector_index, p2)),
                dot(decoded, vload4(vector_index, p3)));
            residual_dots += (int4)(
                dot(decoded, vload4(vector_index, r0)),
                dot(decoded, vload4(vector_index, r1)),
                dot(decoded, vload4(vector_index, r2)),
                dot(decoded, vload4(vector_index, r3)));
        }
        const float2 s0 = xe_v2_activation_scales(a0);
        const float2 s1 = xe_v2_activation_scales(a1);
        const float2 s2 = xe_v2_activation_scales(a2);
        const float2 s3 = xe_v2_activation_scales(a3);
        totals += convert_float4(primary_dots) * weight_scale *
            (float4)(s0.x, s1.x, s2.x, s3.x);
        totals += convert_float4(residual_dots) * weight_scale *
            (float4)(s0.y, s1.y, s2.y, s3.y);
    }
    output[(size_t)row * columns + column] = totals.x;
    output[(size_t)(row + 1u) * columns + column] = totals.y;
    output[(size_t)(row + 2u) * columns + column] = totals.z;
    output[(size_t)(row + 3u) * columns + column] = totals.w;
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void mxfp4_splitk_terms_v2(
    __global const uchar *weights,
    __global const uchar *activations,
    __global float *terms,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const uint column = (uint)get_global_id(0);
    const uint block = (uint)get_global_id(1);
    const uint row = (uint)get_global_id(2);
    if (column >= columns || block >= blocks || row >= rows) return;
    const uint tile = column >> 5;
    const uint lane = column & 31u;
    const size_t weight_base = xe_v2_weight_base(tile, block, blocks);
    __global const uchar *activation =
        activations + ((size_t)row * blocks + block) * XE_V2_ACTIVATION_BYTES;
    const int2 dots = xe_v2_block_dots(weights, weight_base, lane, activation);
    const float2 scales = xe_v2_activation_scales(activation);
    const float weight_scale = 0.5f * xe_e8m0(weights[weight_base + lane]);
    const size_t term = (((size_t)row * columns + column) * blocks + block) * 2u;
    terms[term] = (float)dots.x * weight_scale * scales.x;
    terms[term + 1u] = (float)dots.y * weight_scale * scales.y;
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void mxfp4_splitk_reduce_v2(
    __global const float *terms,
    __global const float *bias,
    __global float *output,
    const uint rows,
    const uint columns,
    const uint blocks) {
    const uint column = (uint)get_global_id(0);
    const uint row = (uint)get_global_id(1);
    if (column >= columns || row >= rows) return;
    float total = bias[column];
    size_t term = ((size_t)row * columns + column) * blocks * 2u;
    for (uint block = 0; block < blocks; ++block) {
        total += terms[term++];
        total += terms[term++];
    }
    output[(size_t)row * columns + column] = total;
}
#endif

__kernel void xe_bandwidth_coalesced(
    __global const uchar *input,
    __global uint *checksums,
    const uint bytes,
    const uint passes) {
    const uint worker = (uint)get_global_id(0);
    const uint workers = (uint)get_global_size(0);
    uint sum = 0;
    for (uint pass = 0; pass < passes; ++pass) {
        for (uint index = worker; index < bytes; index += workers) {
            sum += input[index];
        }
    }
    checksums[worker] = sum;
}

__kernel void xe_bandwidth_strided(
    __global const uchar *input,
    __global uint *checksums,
    const uint bytes,
    const uint passes) {
    const uint worker = (uint)get_global_id(0);
    const uint workers = (uint)get_global_size(0);
    const uint chunk = (bytes + workers - 1u) / workers;
    const uint start = worker * chunk;
    const uint end = min(start + chunk, bytes);
    uint sum = 0;
    for (uint pass = 0; pass < passes; ++pass) {
        for (uint index = start; index < end; ++index) {
            sum += input[index];
        }
    }
    checksums[worker] = sum;
}
