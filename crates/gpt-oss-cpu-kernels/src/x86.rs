//! Narrowly audited x86-64 SIMD implementations.
//!
//! MXFP4 unpacking and integer accumulation are adapted from the organization
//! of llama.cpp `ggml/src/ggml-cpu/arch/x86/quants.c` at revision
//! 030ebb558a5820b444a8f836ed5cdd46c9b4bd7a (MIT). SafeTensors nibble order
//! follows mistral.rs `mistralrs-quant/src/mxfp4/mod.rs` at revision
//! 8010b6a0578e416120b590ed72fd46ed5f24ee85 (MIT). The x8 row-interleaved
//! packing organization was also cross-checked against ik_llama.cpp's IQK
//! row-interleaved kernels at revision
//! 26ceed9d4091a1696cf50e2ed87e5767d5811d81 (MIT). This is an independent
//! Rust implementation of those algorithmic ideas.

use std::arch::x86_64::*;

use half::bf16;

use crate::{
    e8m0_scale, Mxfp4ActivationMatrix, Mxfp4Block, Mxfp4MatrixView, Q8Block, ResidualQ8Block,
    QUANT_BLOCK_SIZE,
};

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
    // SAFETY: caller verified AVX2 availability.
    unsafe { mxfp4_q8_dot_unpacked_avx2(weights, activation) }
}

#[target_feature(enable = "avx2")]
unsafe fn mxfp4_q8_dot_unpacked_avx2(weights: __m256i, activation: &Q8Block) -> i32 {
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

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mxfp4_residual_q8_dot_avx2(
    weight: &Mxfp4Block,
    activation: &ResidualQ8Block,
) -> [i32; 2] {
    // SAFETY: caller verified AVX2 availability. The unpacked vector is reused
    // for both integer dots.
    let weights = unsafe { unpack_mxfp4_avx2(weight) };
    [
        unsafe { mxfp4_q8_dot_unpacked_avx2(weights, &activation.primary) },
        unsafe { mxfp4_q8_dot_unpacked_avx2(weights, &activation.residual) },
    ]
}

#[target_feature(enable = "avx2")]
unsafe fn repeated_i8x8(values: *const i8) -> __m256i {
    // SAFETY: callers pass a pointer to at least eight activation bytes.
    let values = unsafe { values.cast::<i64>().read_unaligned() };
    _mm256_set1_epi64x(values)
}

#[target_feature(enable = "avx2")]
fn dot_four_rows_eight(weights: __m256i, activations: __m256i) -> [i32; 4] {
    // VPMADDUBSW accepts unsigned bytes first and signed bytes second. Q8 is
    // represented as abs(activation), while VPSIGNB transfers its sign to the
    // small signed doubled-E2M1 weight. Each eight-byte segment is one row.
    let absolute_activations = _mm256_abs_epi8(activations);
    let signed_weights = _mm256_sign_epi8(weights, activations);
    let pairs = _mm256_maddubs_epi16(absolute_activations, signed_weights);
    let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let mut partial = [0_i32; 8];
    // SAFETY: `partial` has room for one 256-bit vector.
    unsafe { _mm256_storeu_si256(partial.as_mut_ptr().cast(), quads) };
    std::array::from_fn(|row| partial[row * 2] + partial[row * 2 + 1])
}

struct X8ChunkActivations {
    primary_low: __m256i,
    primary_high: __m256i,
    residual_low: Option<__m256i>,
    residual_high: Option<__m256i>,
}

#[target_feature(enable = "avx2")]
unsafe fn accumulate_x8_chunk(
    packed: *const u8,
    activations: X8ChunkActivations,
    row_start: usize,
    primary: &mut [i32; 8],
    residual: &mut [i32; 8],
) {
    const LUT: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    // SAFETY: callers pass 32 packed bytes and the fixed LUT has 16 bytes.
    let packed = unsafe { _mm256_loadu_si256(packed.cast()) };
    let lut = _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(LUT.as_ptr().cast()) });
    let mask = _mm256_set1_epi8(0x0f);
    let low_weights = _mm256_shuffle_epi8(lut, _mm256_and_si256(packed, mask));
    let high_weights =
        _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask));

    let low_dot = dot_four_rows_eight(low_weights, activations.primary_low);
    let high_dot = dot_four_rows_eight(high_weights, activations.primary_high);
    for row in 0..4 {
        primary[row_start + row] += low_dot[row] + high_dot[row];
    }
    if let (Some(residual_low), Some(residual_high)) =
        (activations.residual_low, activations.residual_high)
    {
        let low_dot = dot_four_rows_eight(low_weights, residual_low);
        let high_dot = dot_four_rows_eight(high_weights, residual_high);
        for row in 0..4 {
            residual[row_start + row] += low_dot[row] + high_dot[row];
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mxfp4_x8_block_dots(
    packed: &[u8; 136],
    primary_values: &[i8; 32],
    residual_values: Option<&[i8; 32]>,
) -> (__m256i, __m256i) {
    let mut primary = [0_i32; 8];
    let mut residual = [0_i32; 8];
    for chunk in 0..2 {
        // SAFETY: each chunk points at an eight-byte half-row.
        let primary_low = unsafe { repeated_i8x8(primary_values.as_ptr().add(chunk * 8)) };
        let primary_high = unsafe { repeated_i8x8(primary_values.as_ptr().add(16 + chunk * 8)) };
        let residual_low = residual_values.map(|values| {
            // SAFETY: same fixed 32-byte activation bound as the primary.
            unsafe { repeated_i8x8(values.as_ptr().add(chunk * 8)) }
        });
        let residual_high = residual_values.map(|values| {
            // SAFETY: same fixed 32-byte activation bound as the primary.
            unsafe { repeated_i8x8(values.as_ptr().add(16 + chunk * 8)) }
        });
        for row_group in 0..2 {
            let offset = 8 + chunk * 64 + row_group * 32;
            // SAFETY: the x8 record has 128 packed bytes after its scales; each
            // iteration consumes one in-bounds 32-byte four-row segment.
            unsafe {
                accumulate_x8_chunk(
                    packed.as_ptr().add(offset),
                    X8ChunkActivations {
                        primary_low,
                        primary_high,
                        residual_low,
                        residual_high,
                    },
                    row_group * 4,
                    &mut primary,
                    &mut residual,
                )
            };
        }
    }
    // SAFETY: both arrays have room for one 256-bit vector.
    (
        unsafe { _mm256_loadu_si256(primary.as_ptr().cast()) },
        unsafe { _mm256_loadu_si256(residual.as_ptr().cast()) },
    )
}

#[target_feature(enable = "avx2")]
fn x8_weight_scales(packed: &[u8; 136]) -> __m256 {
    let scales = std::array::from_fn::<_, 8, _>(|lane| e8m0_scale(packed[lane]));
    // SAFETY: `scales` contains eight contiguous FP32 lanes.
    unsafe { _mm256_loadu_ps(scales.as_ptr()) }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mxfp4_q8_gemv_x8_avx2(
    weights: Mxfp4MatrixView<'_>,
    row_start: usize,
    activations: &[Q8Block],
    bias: &[f32],
    output: &mut [f32],
) {
    // SAFETY: the public projection entry point validates eight-element slices.
    let mut accumulator = unsafe { _mm256_loadu_ps(bias.as_ptr()) };
    let group = row_start / 8;
    for (block_index, activation) in activations.iter().enumerate() {
        let packed = weights.x8_block(group, block_index);
        // SAFETY: activation arrays and packed group sizes are fixed.
        let (dots, _) = unsafe { mxfp4_x8_block_dots(packed, &activation.values, None) };
        let mut contribution = _mm256_cvtepi32_ps(dots);
        contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
        contribution = _mm256_mul_ps(contribution, x8_weight_scales(packed));
        contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(activation.scale));
        accumulator = _mm256_add_ps(accumulator, contribution);
    }
    // SAFETY: the public projection entry point validates eight output lanes.
    unsafe { _mm256_storeu_ps(output.as_mut_ptr(), accumulator) };
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mxfp4_residual_q8_gemv_x8_avx2(
    weights: Mxfp4MatrixView<'_>,
    row_start: usize,
    activations: &[ResidualQ8Block],
    bias: &[f32],
    output: &mut [f32],
) {
    // SAFETY: the public projection entry point validates eight-element slices.
    let mut accumulator = unsafe { _mm256_loadu_ps(bias.as_ptr()) };
    let group = row_start / 8;
    for (block_index, activation) in activations.iter().enumerate() {
        let packed = weights.x8_block(group, block_index);
        // SAFETY: activation arrays and packed group sizes are fixed. Both dots
        // share each decoded weight vector before the K-block advances.
        let (primary, residual) = unsafe {
            mxfp4_x8_block_dots(
                packed,
                &activation.primary.values,
                Some(&activation.residual.values),
            )
        };
        let weight_scales = x8_weight_scales(packed);

        let mut contribution = _mm256_cvtepi32_ps(primary);
        contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
        contribution = _mm256_mul_ps(contribution, weight_scales);
        contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(activation.primary.scale));
        accumulator = _mm256_add_ps(accumulator, contribution);

        let mut contribution = _mm256_cvtepi32_ps(residual);
        contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
        contribution = _mm256_mul_ps(contribution, weight_scales);
        contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(activation.residual.scale));
        accumulator = _mm256_add_ps(accumulator, contribution);
    }
    // SAFETY: the public projection entry point validates eight output lanes.
    unsafe { _mm256_storeu_ps(output.as_mut_ptr(), accumulator) };
}

#[target_feature(enable = "avx2")]
unsafe fn decode_mxfp4_four_rows_avx2(packed: *const u8) -> (__m256i, __m256i) {
    const LUT: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    // SAFETY: callers identify one in-bounds 32-byte four-output segment and
    // the fixed LUT contains sixteen bytes.
    let packed = unsafe { _mm256_loadu_si256(packed.cast()) };
    let lut = _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(LUT.as_ptr().cast()) });
    let mask = _mm256_set1_epi8(0x0f);
    let low = _mm256_shuffle_epi8(lut, _mm256_and_si256(packed, mask));
    let high = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask));
    (low, high)
}

#[target_feature(enable = "avx2")]
unsafe fn mxfp4_x8_panel_block_dots_avx2(
    packed: &[u8; 136],
    primary: *const i8,
    residual: Option<*const i8>,
    input_rows: usize,
) -> ([[i32; 8]; 4], [[i32; 8]; 4]) {
    let mut primary_dots = [[0_i32; 8]; 4];
    let mut residual_dots = [[0_i32; 8]; 4];
    for chunk in 0..2 {
        for output_group in 0..2 {
            let offset = 8 + chunk * 64 + output_group * 32;
            // SAFETY: the x8 block contains the selected 32-byte segment.
            let (low_weights, high_weights) =
                unsafe { decode_mxfp4_four_rows_avx2(packed.as_ptr().add(offset)) };
            for input_row in 0..input_rows {
                // SAFETY: the packed panel contains four complete 32-byte rows.
                let row = unsafe { primary.add(input_row * QUANT_BLOCK_SIZE) };
                let low = unsafe { repeated_i8x8(row.add(chunk * 8)) };
                let high = unsafe { repeated_i8x8(row.add(16 + chunk * 8)) };
                let low_dots = dot_four_rows_eight(low_weights, low);
                let high_dots = dot_four_rows_eight(high_weights, high);
                for lane in 0..4 {
                    primary_dots[input_row][output_group * 4 + lane] +=
                        low_dots[lane] + high_dots[lane];
                }

                if let Some(residual) = residual {
                    // SAFETY: residual uses the same fixed panel extent.
                    let row = unsafe { residual.add(input_row * QUANT_BLOCK_SIZE) };
                    let low = unsafe { repeated_i8x8(row.add(chunk * 8)) };
                    let high = unsafe { repeated_i8x8(row.add(16 + chunk * 8)) };
                    let low_dots = dot_four_rows_eight(low_weights, low);
                    let high_dots = dot_four_rows_eight(high_weights, high);
                    for lane in 0..4 {
                        residual_dots[input_row][output_group * 4 + lane] +=
                            low_dots[lane] + high_dots[lane];
                    }
                }
            }
        }
    }
    (primary_dots, residual_dots)
}

fn packed_panel_scale(panel: &[u8], base: usize, row: usize) -> f32 {
    let start = base + row * std::mem::size_of::<f32>();
    f32::from_ne_bytes(
        panel[start..start + 4]
            .try_into()
            .expect("validated panel scale"),
    )
}

/// Four-input-row by eight-output-row MXFP4 matrix kernel.
///
/// Each x8 weight chunk is decoded once and reused for every input row and for
/// both residual passes before the next K block is loaded.
pub(super) struct MatmulDestination<'a> {
    pub(super) bias: Option<&'a [f32]>,
    pub(super) output: &'a mut [f32],
    pub(super) output_stride: usize,
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn mxfp4_matmul_x8_avx2(
    weights: Mxfp4MatrixView<'_>,
    input_start: usize,
    input_rows: usize,
    activations: Mxfp4ActivationMatrix<'_>,
    destination: MatmulDestination<'_>,
    panel: &[u8],
) {
    let (blocks, passes) = match activations {
        Mxfp4ActivationMatrix::Q8(view) => (view.blocks_per_row(), 1),
        Mxfp4ActivationMatrix::ResidualQ8(view) => (view.blocks_per_row(), 2),
    };
    let pass_bytes = crate::matmul::avx2_panel_pass_bytes();
    for group in 0..weights.complete_x8_groups() {
        let output_start = group * 8;
        let initial = if let Some(bias) = destination.bias {
            // SAFETY: every group is a complete eight-output span.
            unsafe { _mm256_loadu_ps(bias.as_ptr().add(output_start)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut accumulators = [initial; 4];
        for block in 0..blocks {
            let packed = weights.x8_block(group, block);
            let primary_base = (block * passes) * pass_bytes;
            let residual_base = (block * passes + 1) * pass_bytes;
            // SAFETY: the caller packed four 32-byte rows after sixteen scale
            // bytes in every required panel pass.
            let primary = unsafe { panel.as_ptr().add(primary_base + 16).cast::<i8>() };
            let residual = (passes == 2)
                .then(|| unsafe { panel.as_ptr().add(residual_base + 16).cast::<i8>() });
            let (primary_dots, residual_dots) =
                unsafe { mxfp4_x8_panel_block_dots_avx2(packed, primary, residual, input_rows) };
            let weight_scales = x8_weight_scales(packed);
            for input_row in 0..input_rows {
                // SAFETY: each fixed array contains eight INT32 values.
                let dots = unsafe {
                    _mm256_loadu_si256(primary_dots[input_row].as_ptr().cast::<__m256i>())
                };
                let mut contribution = _mm256_cvtepi32_ps(dots);
                contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
                contribution = _mm256_mul_ps(contribution, weight_scales);
                contribution = _mm256_mul_ps(
                    contribution,
                    _mm256_set1_ps(packed_panel_scale(panel, primary_base, input_row)),
                );
                accumulators[input_row] = _mm256_add_ps(accumulators[input_row], contribution);

                if passes == 2 {
                    // SAFETY: each fixed array contains eight INT32 values.
                    let dots = unsafe {
                        _mm256_loadu_si256(residual_dots[input_row].as_ptr().cast::<__m256i>())
                    };
                    let mut contribution = _mm256_cvtepi32_ps(dots);
                    contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
                    contribution = _mm256_mul_ps(contribution, weight_scales);
                    contribution = _mm256_mul_ps(
                        contribution,
                        _mm256_set1_ps(packed_panel_scale(panel, residual_base, input_row)),
                    );
                    accumulators[input_row] = _mm256_add_ps(accumulators[input_row], contribution);
                }
            }
        }
        for (input_row, accumulator) in accumulators.iter().enumerate().take(input_rows) {
            let offset = (input_start + input_row) * destination.output_stride + output_start;
            // SAFETY: the validated problem reserves eight output elements.
            unsafe { _mm256_storeu_ps(destination.output.as_mut_ptr().add(offset), *accumulator) };
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn decode_mxfp4_x8_chunk_avx512(packed: *const u8) -> (__m512i, __m512i) {
    const LUT: [i8; 64] = [
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0, -1,
        -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2,
        3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    ];
    // SAFETY: the caller passes one complete 64-byte x8 layout chunk. The
    // fixed LUT has one copy in every 128-bit VPSHUFB lane.
    let packed = unsafe { _mm512_loadu_si512(packed.cast()) };
    let lut = unsafe { _mm512_loadu_si512(LUT.as_ptr().cast()) };
    let mask = _mm512_set1_epi8(0x0f);
    let low = _mm512_shuffle_epi8(lut, _mm512_and_si512(packed, mask));
    let high = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask));
    (low, high)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot_eight_rows_eight_avx512_vnni(weights: __m512i, activations: *const i8) -> [i32; 8] {
    // Each output row occupies one 64-bit lane in `weights`. Replicating the
    // matching eight activations produces two four-byte VNNI partials per row.
    // SAFETY: the caller identifies an in-bounds eight-byte activation span.
    let activation_sum = (0..8)
        .map(|lane| {
            // SAFETY: the same eight-byte bound used for the lane load below.
            unsafe { *activations.add(lane) as i32 }
        })
        .sum::<i32>();
    let activation_lane = unsafe { std::ptr::read_unaligned(activations.cast::<i64>()) };
    let activations = _mm512_set1_epi64(activation_lane);
    let shifted_weights = _mm512_add_epi8(weights, _mm512_set1_epi8(12));
    let shifted_dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), shifted_weights, activations);

    let mut partial = [0_i32; 16];
    // SAFETY: `partial` has room for one 512-bit vector.
    unsafe { _mm512_storeu_si512(partial.as_mut_ptr().cast(), shifted_dot) };

    // The correction is identical in every replicated row lane. Accumulate it
    // in scalar i32 to avoid adding any capability beyond the ZMM body.
    std::array::from_fn(|row| partial[row * 2] + partial[row * 2 + 1] - 12 * activation_sum)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn mxfp4_x8_block_dots_avx512_vnni(
    packed: &[u8; 136],
    primary_values: &[i8; 32],
    residual_values: Option<&[i8; 32]>,
) -> ([i32; 8], [i32; 8]) {
    let mut primary = [0_i32; 8];
    let mut residual = [0_i32; 8];
    for chunk in 0..2 {
        // SAFETY: each iteration addresses one complete 64-byte value chunk
        // after the eight scale bytes in the fixed 136-byte record.
        let (low_weights, high_weights) =
            unsafe { decode_mxfp4_x8_chunk_avx512(packed.as_ptr().add(8 + chunk * 64)) };
        // SAFETY: each pointer begins an in-bounds eight-value segment.
        let low = unsafe {
            dot_eight_rows_eight_avx512_vnni(low_weights, primary_values.as_ptr().add(chunk * 8))
        };
        let high = unsafe {
            dot_eight_rows_eight_avx512_vnni(
                high_weights,
                primary_values.as_ptr().add(16 + chunk * 8),
            )
        };
        for row in 0..8 {
            primary[row] += low[row] + high[row];
        }

        if let Some(residual_values) = residual_values {
            // The decoded ZMM weights stay live and are reused for the second
            // activation dot before this K chunk advances.
            let low = unsafe {
                dot_eight_rows_eight_avx512_vnni(
                    low_weights,
                    residual_values.as_ptr().add(chunk * 8),
                )
            };
            let high = unsafe {
                dot_eight_rows_eight_avx512_vnni(
                    high_weights,
                    residual_values.as_ptr().add(16 + chunk * 8),
                )
            };
            for row in 0..8 {
                residual[row] += low[row] + high[row];
            }
        }
    }
    (primary, residual)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn mxfp4_x8_panel_block_dots_avx512_vnni(
    packed: &[u8; 136],
    primary: *const i8,
    residual: Option<*const i8>,
    input_rows: usize,
) -> ([[i32; 8]; 8], [[i32; 8]; 8]) {
    let mut primary_dots = [[0_i32; 8]; 8];
    let mut residual_dots = [[0_i32; 8]; 8];
    for chunk in 0..2 {
        // Decode each eight-output x8-v2 weight chunk once, then reuse both
        // ZMM vectors across all eight input rows and the residual pass.
        // SAFETY: the fixed x8 record contains both complete 64-byte chunks.
        let (low_weights, high_weights) =
            unsafe { decode_mxfp4_x8_chunk_avx512(packed.as_ptr().add(8 + chunk * 64)) };
        for input_row in 0..input_rows {
            // SAFETY: the caller supplies eight complete 32-byte panel rows.
            let row = unsafe { primary.add(input_row * QUANT_BLOCK_SIZE) };
            let low = unsafe { dot_eight_rows_eight_avx512_vnni(low_weights, row.add(chunk * 8)) };
            let high =
                unsafe { dot_eight_rows_eight_avx512_vnni(high_weights, row.add(16 + chunk * 8)) };
            for output in 0..8 {
                primary_dots[input_row][output] += low[output] + high[output];
            }

            if let Some(residual) = residual {
                // SAFETY: residual panel rows have the same fixed extent.
                let row = unsafe { residual.add(input_row * QUANT_BLOCK_SIZE) };
                let low =
                    unsafe { dot_eight_rows_eight_avx512_vnni(low_weights, row.add(chunk * 8)) };
                let high = unsafe {
                    dot_eight_rows_eight_avx512_vnni(high_weights, row.add(16 + chunk * 8))
                };
                for output in 0..8 {
                    residual_dots[input_row][output] += low[output] + high[output];
                }
            }
        }
    }
    (primary_dots, residual_dots)
}

/// Eight-input-row by eight-output-row AVX-512/VNNI MXFP4 matrix kernel.
///
/// The persistent x8-v2 layout is unchanged. Each decoded ZMM weight pair is
/// reused across the full input microtile and both residual-Q8 passes.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(super) unsafe fn mxfp4_matmul_x8_avx512_vnni(
    weights: Mxfp4MatrixView<'_>,
    input_start: usize,
    input_rows: usize,
    activations: Mxfp4ActivationMatrix<'_>,
    destination: MatmulDestination<'_>,
    panel: &[u8],
) {
    let (blocks, passes) = match activations {
        Mxfp4ActivationMatrix::Q8(view) => (view.blocks_per_row(), 1),
        Mxfp4ActivationMatrix::ResidualQ8(view) => (view.blocks_per_row(), 2),
    };
    let pass_bytes = crate::matmul::avx512_panel_pass_bytes();
    for group in 0..weights.complete_x8_groups() {
        let output_start = group * 8;
        let initial = if let Some(bias) = destination.bias {
            // SAFETY: every group spans eight validated outputs.
            unsafe { _mm256_loadu_ps(bias.as_ptr().add(output_start)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut accumulators = [initial; 8];
        for block in 0..blocks {
            let packed = weights.x8_block(group, block);
            let primary_base = (block * passes) * pass_bytes;
            let residual_base = (block * passes + 1) * pass_bytes;
            // SAFETY: the AVX-512 panel stores eight 32-byte rows after its
            // eight FP32 scales in every pass.
            let primary = unsafe { panel.as_ptr().add(primary_base + 32).cast::<i8>() };
            let residual = (passes == 2)
                .then(|| unsafe { panel.as_ptr().add(residual_base + 32).cast::<i8>() });
            let (primary_dots, residual_dots) = unsafe {
                mxfp4_x8_panel_block_dots_avx512_vnni(packed, primary, residual, input_rows)
            };
            let weight_scales = x8_weight_scales(packed);
            for input_row in 0..input_rows {
                // SAFETY: each fixed row contains eight INT32 dots.
                let dots = unsafe {
                    _mm256_loadu_si256(primary_dots[input_row].as_ptr().cast::<__m256i>())
                };
                let mut contribution = _mm256_cvtepi32_ps(dots);
                contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
                contribution = _mm256_mul_ps(contribution, weight_scales);
                contribution = _mm256_mul_ps(
                    contribution,
                    _mm256_set1_ps(packed_panel_scale(panel, primary_base, input_row)),
                );
                accumulators[input_row] = _mm256_add_ps(accumulators[input_row], contribution);

                if passes == 2 {
                    // SAFETY: each fixed row contains eight INT32 dots.
                    let dots = unsafe {
                        _mm256_loadu_si256(residual_dots[input_row].as_ptr().cast::<__m256i>())
                    };
                    let mut contribution = _mm256_cvtepi32_ps(dots);
                    contribution = _mm256_mul_ps(contribution, _mm256_set1_ps(0.5));
                    contribution = _mm256_mul_ps(contribution, weight_scales);
                    contribution = _mm256_mul_ps(
                        contribution,
                        _mm256_set1_ps(packed_panel_scale(panel, residual_base, input_row)),
                    );
                    accumulators[input_row] = _mm256_add_ps(accumulators[input_row], contribution);
                }
            }
        }
        for (input_row, accumulator) in accumulators.iter().enumerate().take(input_rows) {
            let offset = (input_start + input_row) * destination.output_stride + output_start;
            // SAFETY: the validated problem reserves eight output elements.
            unsafe { _mm256_storeu_ps(destination.output.as_mut_ptr().add(offset), *accumulator) };
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(super) unsafe fn mxfp4_q8_gemv_x8_avx512_vnni(
    weights: Mxfp4MatrixView<'_>,
    row_start: usize,
    activations: &[Q8Block],
    bias: &[f32],
    output: &mut [f32],
) {
    // SAFETY: the public projection entry point validates eight-element slices.
    output.copy_from_slice(bias);
    let group = row_start / 8;
    for (block_index, activation) in activations.iter().enumerate() {
        let packed = weights.x8_block(group, block_index);
        // SAFETY: activation arrays and packed x8 records have fixed sizes.
        let (dots, _) =
            unsafe { mxfp4_x8_block_dots_avx512_vnni(packed, &activation.values, None) };
        for row in 0..8 {
            output[row] += dots[row] as f32 * 0.5 * e8m0_scale(packed[row]) * activation.scale;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(super) unsafe fn mxfp4_residual_q8_gemv_x8_avx512_vnni(
    weights: Mxfp4MatrixView<'_>,
    row_start: usize,
    activations: &[ResidualQ8Block],
    bias: &[f32],
    output: &mut [f32],
) {
    // SAFETY: the public projection entry point validates eight-element slices.
    output.copy_from_slice(bias);
    let group = row_start / 8;
    for (block_index, activation) in activations.iter().enumerate() {
        let packed = weights.x8_block(group, block_index);
        // SAFETY: both activation arrays and the packed x8 record have fixed
        // sizes. The helper reuses decoded weights for both dots.
        let (primary, residual) = unsafe {
            mxfp4_x8_block_dots_avx512_vnni(
                packed,
                &activation.primary.values,
                Some(&activation.residual.values),
            )
        };
        for row in 0..8 {
            let weight_scale = 0.5 * e8m0_scale(packed[row]);
            output[row] += primary[row] as f32 * weight_scale * activation.primary.scale;
            output[row] += residual[row] as f32 * weight_scale * activation.residual.scale;
        }
    }
}

#[target_feature(enable = "avx2,avx512vl,avx512vnni")]
pub(super) unsafe fn mxfp4_q8_dot_avx512_vnni(weight: &Mxfp4Block, activation: &Q8Block) -> i32 {
    // VNNI's byte dot instruction accepts unsigned weights and signed
    // activations. Shift the small signed E2M1 integers by 12, then subtract
    // 12 times each activation sum. This keeps the entire dot exact in i32
    // while avoiding two 8-to-16-bit expansions.
    // SAFETY: caller verified AVX2, AVX-512 VL, and VNNI availability.
    let signed_weights = unsafe { unpack_mxfp4_avx2(weight) };
    // SAFETY: caller verified the VNNI feature set.
    unsafe { mxfp4_q8_dot_unpacked_avx512_vnni(signed_weights, activation) }
}

#[target_feature(enable = "avx2,avx512vl,avx512vnni")]
unsafe fn mxfp4_q8_dot_unpacked_avx512_vnni(signed_weights: __m256i, activation: &Q8Block) -> i32 {
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

#[target_feature(enable = "avx2,avx512vl,avx512vnni")]
pub(super) unsafe fn mxfp4_residual_q8_dot_avx512_vnni(
    weight: &Mxfp4Block,
    activation: &ResidualQ8Block,
) -> [i32; 2] {
    // SAFETY: caller verified the VNNI feature set. The unpacked vector is
    // reused for both integer dots.
    let weights = unsafe { unpack_mxfp4_avx2(weight) };
    [
        unsafe { mxfp4_q8_dot_unpacked_avx512_vnni(weights, &activation.primary) },
        unsafe { mxfp4_q8_dot_unpacked_avx512_vnni(weights, &activation.residual) },
    ]
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
