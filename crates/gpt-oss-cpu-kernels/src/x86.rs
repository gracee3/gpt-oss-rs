//! Narrowly audited x86-64 SIMD implementations.
//!
//! MXFP4 unpacking and integer accumulation are adapted from the organization
//! of llama.cpp `ggml/src/ggml-cpu/arch/x86/quants.c` at revision
//! 030ebb558a5820b444a8f836ed5cdd46c9b4bd7a (MIT). SafeTensors nibble order
//! follows mistral.rs `mistralrs-quant/src/mxfp4/mod.rs` at revision
//! 8010b6a0578e416120b590ed72fd46ed5f24ee85 (MIT).

use std::arch::x86_64::*;

use half::bf16;

use crate::{Mxfp4Block, Q8Block};

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn bf16_dot_avx2(left: &[bf16], right: &[bf16]) -> f32 {
    let mut accumulator_low = _mm256_setzero_ps();
    let mut accumulator_high = _mm256_setzero_ps();
    let mut index = 0;
    while index + 16 <= left.len() {
        // SAFETY: the loop bounds guarantee sixteen BF16 values per slice.
        let left_low = unsafe { _mm_loadu_si128(left.as_ptr().add(index).cast()) };
        let right_low = unsafe { _mm_loadu_si128(right.as_ptr().add(index).cast()) };
        let left_high = unsafe { _mm_loadu_si128(left.as_ptr().add(index + 8).cast()) };
        let right_high = unsafe { _mm_loadu_si128(right.as_ptr().add(index + 8).cast()) };
        let left_low = _mm256_slli_epi32(_mm256_cvtepu16_epi32(left_low), 16);
        let right_low = _mm256_slli_epi32(_mm256_cvtepu16_epi32(right_low), 16);
        let left_high = _mm256_slli_epi32(_mm256_cvtepu16_epi32(left_high), 16);
        let right_high = _mm256_slli_epi32(_mm256_cvtepu16_epi32(right_high), 16);
        accumulator_low = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_castsi256_ps(left_low),
                _mm256_castsi256_ps(right_low),
            ),
            accumulator_low,
        );
        accumulator_high = _mm256_add_ps(
            _mm256_mul_ps(
                _mm256_castsi256_ps(left_high),
                _mm256_castsi256_ps(right_high),
            ),
            accumulator_high,
        );
        index += 16;
    }
    let mut lanes = [0.0_f32; 16];
    // SAFETY: each half of `lanes` has room for one 256-bit vector.
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator_low);
        _mm256_storeu_ps(lanes.as_mut_ptr().add(8), accumulator_high);
    }
    while index < left.len() {
        let lane = index % lanes.len();
        lanes[lane] += left[index].to_f32() * right[index].to_f32();
        index += 1;
    }
    lanes.into_iter().sum()
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(super) unsafe fn bf16_dot_avx512(left: &[bf16], right: &[bf16]) -> f32 {
    let mut accumulator = _mm512_setzero_ps();
    let mut index = 0;
    while index + 16 <= left.len() {
        // SAFETY: the loop bounds guarantee sixteen BF16 values per slice.
        let left16 = unsafe { _mm256_loadu_si256(left.as_ptr().add(index).cast()) };
        let right16 = unsafe { _mm256_loadu_si256(right.as_ptr().add(index).cast()) };
        let left32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(left16), 16);
        let right32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(right16), 16);
        accumulator = _mm512_add_ps(
            _mm512_mul_ps(_mm512_castsi512_ps(left32), _mm512_castsi512_ps(right32)),
            accumulator,
        );
        index += 16;
    }
    let mut lanes = [0.0_f32; 16];
    // SAFETY: `lanes` has room for one 512-bit vector.
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    while index < left.len() {
        let lane = index % lanes.len();
        lanes[lane] += left[index].to_f32() * right[index].to_f32();
        index += 1;
    }
    lanes.into_iter().sum()
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn max_abs_avx2(values: &[f32]) -> f32 {
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut maximum = _mm256_setzero_ps();
    let mut index = 0;
    while index + 8 <= values.len() {
        // SAFETY: the loop bounds guarantee eight values.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(index)) };
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign_mask, value));
        index += 8;
    }
    let mut lanes = [0.0_f32; 8];
    // SAFETY: `lanes` has room for one 256-bit vector.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), maximum) };
    let mut result = lanes.into_iter().fold(0.0_f32, f32::max);
    while index < values.len() {
        result = result.max(values[index].abs());
        index += 1;
    }
    result
}

#[target_feature(enable = "avx512f,avx512dq")]
pub(super) unsafe fn max_abs_avx512(values: &[f32]) -> f32 {
    let sign_mask = _mm512_set1_ps(-0.0);
    let mut maximum = _mm512_setzero_ps();
    let mut index = 0;
    while index + 16 <= values.len() {
        // SAFETY: the loop bounds guarantee sixteen values.
        let value = unsafe { _mm512_loadu_ps(values.as_ptr().add(index)) };
        maximum = _mm512_max_ps(maximum, _mm512_andnot_ps(sign_mask, value));
        index += 16;
    }
    let mut lanes = [0.0_f32; 16];
    // SAFETY: `lanes` has room for one 512-bit vector.
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), maximum) };
    let mut result = lanes.into_iter().fold(0.0_f32, f32::max);
    while index < values.len() {
        result = result.max(values[index].abs());
        index += 1;
    }
    result
}

#[target_feature(enable = "avx2")]
unsafe fn unpack_mxfp4_avx2(weight: &Mxfp4Block) -> __m256i {
    const LUT: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    // SAFETY: both fixed arrays contain at least sixteen bytes.
    let packed = unsafe { _mm_loadu_si128(weight.packed.as_ptr().cast()) };
    let lut = unsafe { _mm_loadu_si128(LUT.as_ptr().cast()) };
    let mask = _mm_set1_epi8(0x0f);
    let low = _mm_shuffle_epi8(lut, _mm_and_si128(packed, mask));
    let high = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    let first = _mm_unpacklo_epi8(low, high);
    let second = _mm_unpackhi_epi8(low, high);
    _mm256_inserti128_si256(_mm256_castsi128_si256(first), second, 1)
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mxfp4_q8_dot_avx2(weight: &Mxfp4Block, activation: &Q8Block) -> i32 {
    // SAFETY: caller verified AVX2 availability.
    let weights = unsafe { unpack_mxfp4_avx2(weight) };
    // SAFETY: the fixed activation array contains 32 bytes.
    let activations = unsafe { _mm256_loadu_si256(activation.values.as_ptr().cast()) };
    let weight_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(weights));
    let weight_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weights, 1));
    let activation_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(activations));
    let activation_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(activations, 1));
    let pair_low = _mm256_madd_epi16(weight_low, activation_low);
    let pair_high = _mm256_madd_epi16(weight_high, activation_high);
    let sums = _mm256_add_epi32(pair_low, pair_high);
    let mut lanes = [0_i32; 8];
    // SAFETY: `lanes` has room for one 256-bit vector.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    lanes.into_iter().sum()
}

#[target_feature(enable = "avx2,avx512vl,avx512vnni")]
pub(super) unsafe fn mxfp4_q8_dot_avx512_vnni(weight: &Mxfp4Block, activation: &Q8Block) -> i32 {
    // VNNI's byte dot instruction accepts unsigned weights and signed
    // activations. Shift the small signed E2M1 integers by 12, then subtract
    // 12 times each activation sum. This keeps the entire dot exact in i32
    // while avoiding two 8-to-16-bit expansions.
    // SAFETY: caller verified AVX2, AVX-512 VL, and VNNI availability.
    let signed_weights = unsafe { unpack_mxfp4_avx2(weight) };
    // SAFETY: the fixed activation array contains 32 bytes.
    let activations = unsafe { _mm256_loadu_si256(activation.values.as_ptr().cast()) };
    let shifted_weights = _mm256_add_epi8(signed_weights, _mm256_set1_epi8(12));
    let shifted_dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), shifted_weights, activations);

    let pair_sums = _mm256_maddubs_epi16(_mm256_set1_epi8(1), activations);
    let activation_sums = _mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1));
    let sums = _mm256_sub_epi32(
        shifted_dot,
        _mm256_mullo_epi32(activation_sums, _mm256_set1_epi32(12)),
    );
    let mut lanes = [0_i32; 8];
    // SAFETY: `lanes` has room for one 256-bit vector.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    lanes.into_iter().sum()
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn sum_squares_avx2(values: &[f32]) -> f32 {
    let mut accumulator_low = _mm256_setzero_ps();
    let mut accumulator_high = _mm256_setzero_ps();
    let mut index = 0;
    while index + 16 <= values.len() {
        // SAFETY: the loop bounds guarantee sixteen values.
        let low = unsafe { _mm256_loadu_ps(values.as_ptr().add(index)) };
        let high = unsafe { _mm256_loadu_ps(values.as_ptr().add(index + 8)) };
        accumulator_low = _mm256_add_ps(_mm256_mul_ps(low, low), accumulator_low);
        accumulator_high = _mm256_add_ps(_mm256_mul_ps(high, high), accumulator_high);
        index += 16;
    }
    let mut lanes = [0.0_f32; 16];
    // SAFETY: each half of `lanes` has room for one 256-bit vector.
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator_low);
        _mm256_storeu_ps(lanes.as_mut_ptr().add(8), accumulator_high);
    }
    while index < values.len() {
        let lane = index % lanes.len();
        lanes[lane] += values[index] * values[index];
        index += 1;
    }
    lanes.into_iter().sum()
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn sum_squares_avx512(values: &[f32]) -> f32 {
    let mut accumulator = _mm512_setzero_ps();
    let mut index = 0;
    while index + 16 <= values.len() {
        // SAFETY: the loop bounds guarantee sixteen values.
        let value = unsafe { _mm512_loadu_ps(values.as_ptr().add(index)) };
        accumulator = _mm512_add_ps(_mm512_mul_ps(value, value), accumulator);
        index += 16;
    }
    let mut lanes = [0.0_f32; 16];
    // SAFETY: `lanes` has room for one 512-bit vector.
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    while index < values.len() {
        let lane = index % lanes.len();
        lanes[lane] += values[index] * values[index];
        index += 1;
    }
    lanes.into_iter().sum()
}
