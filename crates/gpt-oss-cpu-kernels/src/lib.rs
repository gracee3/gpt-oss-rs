#![deny(unsafe_op_in_unsafe_fn)]
//! Safe dispatch and scalar reference kernels for GPT-OSS CPU inference.
//!
//! Unsafe code is confined to the x86 implementation module. Every public
//! entry point validates dimensions and ISA availability before dispatch.

use std::fmt;
use std::str::FromStr;

use half::bf16;
use thiserror::Error;

#[cfg(target_arch = "x86_64")]
mod x86;

pub const QUANT_BLOCK_SIZE: usize = 32;
pub const MXFP4_PACKED_BYTES: usize = QUANT_BLOCK_SIZE / 2;

/// E2M1 values used by the official GPT-OSS SafeTensors representation.
pub const MXFP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelPath {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Avx512Vnni,
}

impl KernelPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512Vnni => "avx512-vnni",
        }
    }
}

impl fmt::Display for KernelPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for KernelPath {
    type Err = KernelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "scalar" => Ok(Self::Scalar),
            "avx2" => Ok(Self::Avx2),
            "avx512-vnni" | "avx512_vnni" => Ok(Self::Avx512Vnni),
            _ => Err(KernelError::InvalidPath(value.to_string())),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    #[error("unknown CPU kernel path '{0}'")]
    InvalidPath(String),
    #[error("CPU kernel path '{0}' is unavailable on this host")]
    Unavailable(KernelPath),
    #[error("invalid CPU kernel dimensions: {0}")]
    InvalidDimensions(String),
    #[error("CPU kernel input contains a non-finite value")]
    NonFiniteInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Features {
    avx2_fma: bool,
    avx512_vnni: bool,
}

impl Features {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2_fma: std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma"),
                avx512_vnni: std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512dq")
                    && std::is_x86_feature_detected!("avx512vl")
                    && std::is_x86_feature_detected!("avx512vbmi")
                    && std::is_x86_feature_detected!("avx512vnni"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2_fma: false,
                avx512_vnni: false,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernels {
    path: KernelPath,
}

impl Kernels {
    pub fn new(requested: KernelPath) -> Result<Self, KernelError> {
        Self::with_features(requested, Features::detect())
    }

    /// Resolve `GPT_OSS_CPU_KERNEL`, falling back to automatic dispatch.
    pub fn from_env() -> Result<Self, KernelError> {
        match std::env::var("GPT_OSS_CPU_KERNEL") {
            Ok(value) => Self::new(value.parse()?),
            Err(std::env::VarError::NotPresent) => Self::new(KernelPath::Auto),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(KernelError::InvalidPath("non-Unicode value".into()))
            }
        }
    }

    fn with_features(requested: KernelPath, features: Features) -> Result<Self, KernelError> {
        let path = match requested {
            KernelPath::Auto if features.avx512_vnni => KernelPath::Avx512Vnni,
            KernelPath::Auto if features.avx2_fma => KernelPath::Avx2,
            KernelPath::Auto => KernelPath::Scalar,
            KernelPath::Scalar => KernelPath::Scalar,
            KernelPath::Avx2 if features.avx2_fma => KernelPath::Avx2,
            KernelPath::Avx512Vnni if features.avx512_vnni => KernelPath::Avx512Vnni,
            unavailable => return Err(KernelError::Unavailable(unavailable)),
        };
        Ok(Self { path })
    }

    pub const fn path(self) -> KernelPath {
        self.path
    }

    /// BF16 row-major matrix-vector multiplication with FP32 accumulation.
    pub fn bf16_matvec(
        self,
        weights: &[bf16],
        rows: usize,
        cols: usize,
        input: &[bf16],
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        if input.len() != cols || output.len() != rows || weights.len() != rows * cols {
            return Err(KernelError::InvalidDimensions(format!(
                "bf16 matvec expects weights={rows}x{cols}, input={cols}, output={rows}"
            )));
        }
        for (row, destination) in weights.chunks_exact(cols).zip(output.iter_mut()) {
            *destination = match self.path {
                KernelPath::Scalar => scalar_bf16_dot(row, input),
                #[cfg(target_arch = "x86_64")]
                KernelPath::Avx2 => {
                    // SAFETY: construction verifies AVX2 and FMA support.
                    unsafe { x86::bf16_dot_avx2(row, input) }
                }
                #[cfg(target_arch = "x86_64")]
                KernelPath::Avx512Vnni => {
                    // SAFETY: construction verifies the full AVX-512 feature set.
                    unsafe { x86::bf16_dot_avx512(row, input) }
                }
                _ => scalar_bf16_dot(row, input),
            };
        }
        Ok(())
    }

    /// Quantize an activation row into independent symmetric Q8 blocks.
    pub fn quantize_q8(self, input: &[f32]) -> Result<Vec<Q8Block>, KernelError> {
        if !input.len().is_multiple_of(QUANT_BLOCK_SIZE) {
            return Err(KernelError::InvalidDimensions(format!(
                "Q8 input length {} is not divisible by {QUANT_BLOCK_SIZE}",
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(KernelError::NonFiniteInput);
        }

        input
            .chunks_exact(QUANT_BLOCK_SIZE)
            .map(|block| {
                let max_abs = match self.path {
                    KernelPath::Scalar => scalar_max_abs(block),
                    #[cfg(target_arch = "x86_64")]
                    KernelPath::Avx2 => {
                        // SAFETY: construction verifies AVX2 and FMA support.
                        unsafe { x86::max_abs_avx2(block) }
                    }
                    #[cfg(target_arch = "x86_64")]
                    KernelPath::Avx512Vnni => {
                        // SAFETY: construction verifies the full AVX-512 feature set.
                        unsafe { x86::max_abs_avx512(block) }
                    }
                    _ => scalar_max_abs(block),
                };
                Ok(quantize_q8_block(block, max_abs))
            })
            .collect()
    }

    /// Exact integer dot product for one packed MXFP4/Q8 block.
    pub fn mxfp4_q8_block_dot_i32(self, weight: &Mxfp4Block, activation: &Q8Block) -> i32 {
        match self.path {
            KernelPath::Scalar => scalar_mxfp4_q8_dot_i32(weight, activation),
            #[cfg(target_arch = "x86_64")]
            KernelPath::Avx2 => {
                // SAFETY: construction verifies AVX2 and FMA support.
                unsafe { x86::mxfp4_q8_dot_avx2(weight, activation) }
            }
            #[cfg(target_arch = "x86_64")]
            KernelPath::Avx512Vnni => {
                // SAFETY: construction verifies the full AVX-512 feature set.
                unsafe { x86::mxfp4_q8_dot_avx512_vnni(weight, activation) }
            }
            _ => scalar_mxfp4_q8_dot_i32(weight, activation),
        }
    }

    /// Dot product across matching MXFP4 and Q8 block rows.
    pub fn mxfp4_q8_dot(
        self,
        weights: &[Mxfp4Block],
        activations: &[Q8Block],
    ) -> Result<f32, KernelError> {
        if weights.len() != activations.len() {
            return Err(KernelError::InvalidDimensions(format!(
                "MXFP4 blocks {} do not match Q8 blocks {}",
                weights.len(),
                activations.len()
            )));
        }
        let mut total = 0.0_f32;
        for (weight, activation) in weights.iter().zip(activations) {
            let integer = self.mxfp4_q8_block_dot_i32(weight, activation);
            // The integer LUT represents twice the E2M1 value so that 0.5 can
            // be accumulated exactly. Convert that doubled integer result back
            // to the official MXFP4 value before applying the two block scales.
            total += integer as f32 * 0.5 * e8m0_scale(weight.scale) * activation.scale;
        }
        Ok(total)
    }

    /// Matrix-vector projection for rows stored as contiguous MXFP4 blocks.
    pub fn mxfp4_matvec(
        self,
        weights: &[Mxfp4Block],
        rows: usize,
        blocks_per_row: usize,
        activations: &[Q8Block],
        bias: Option<&[f32]>,
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        if weights.len() != rows * blocks_per_row
            || activations.len() != blocks_per_row
            || output.len() != rows
            || bias.is_some_and(|bias| bias.len() != rows)
        {
            return Err(KernelError::InvalidDimensions(
                "invalid MXFP4 matrix-vector projection shape".into(),
            ));
        }
        for row in 0..rows {
            let start = row * blocks_per_row;
            output[row] = self
                .mxfp4_q8_dot(&weights[start..start + blocks_per_row], activations)?
                + bias.map_or(0.0, |bias| bias[row]);
        }
        Ok(())
    }

    /// RMS normalization with FP32 accumulation.
    pub fn rms_norm(
        self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        if input.len() != weight.len() || output.len() != input.len() || input.is_empty() {
            return Err(KernelError::InvalidDimensions(
                "RMS norm slices must have the same non-zero length".into(),
            ));
        }
        let sum_squares = match self.path {
            KernelPath::Scalar => input.iter().map(|value| value * value).sum(),
            #[cfg(target_arch = "x86_64")]
            KernelPath::Avx2 => {
                // SAFETY: construction verifies AVX2 and FMA support.
                unsafe { x86::sum_squares_avx2(input) }
            }
            #[cfg(target_arch = "x86_64")]
            KernelPath::Avx512Vnni => {
                // SAFETY: construction verifies the full AVX-512 feature set.
                unsafe { x86::sum_squares_avx512(input) }
            }
            _ => input.iter().map(|value| value * value).sum(),
        };
        let inverse_rms = (sum_squares / input.len() as f32 + epsilon).sqrt().recip();
        for ((destination, value), scale) in output.iter_mut().zip(input).zip(weight) {
            *destination = *value * inverse_rms * *scale;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8Block {
    pub scale: f32,
    pub values: [i8; QUANT_BLOCK_SIZE],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp4Block {
    pub scale: u8,
    /// Adjacent values are packed low nibble first, matching official
    /// SafeTensors (`[block, 16]`) storage.
    pub packed: [u8; MXFP4_PACKED_BYTES],
}

impl Mxfp4Block {
    pub fn unpack(&self) -> [i8; QUANT_BLOCK_SIZE] {
        let mut output = [0_i8; QUANT_BLOCK_SIZE];
        for (index, byte) in self.packed.iter().copied().enumerate() {
            output[index * 2] = mxfp4_integer(byte & 0x0f);
            output[index * 2 + 1] = mxfp4_integer(byte >> 4);
        }
        output
    }
}

/// Decode one FP4 E2M1 code.
pub const fn decode_mxfp4(code: u8) -> f32 {
    MXFP4_VALUES[(code & 0x0f) as usize]
}

/// Decode an E8M0 scale in the same bit-exact form as the official tensors.
pub const fn e8m0_scale(scale: u8) -> f32 {
    f32::from_bits((scale as u32) << 23)
}

pub fn bf16_to_f32(input: &[bf16], output: &mut [f32]) -> Result<(), KernelError> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidDimensions(
            "BF16 conversion slices have different lengths".into(),
        ));
    }
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = source.to_f32();
    }
    Ok(())
}

pub fn f32_to_bf16(input: &[f32], output: &mut [bf16]) -> Result<(), KernelError> {
    if input.len() != output.len() {
        return Err(KernelError::InvalidDimensions(
            "BF16 conversion slices have different lengths".into(),
        ));
    }
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = bf16::from_f32(*source);
    }
    Ok(())
}

pub fn softmax_in_place(values: &mut [f32]) -> Result<(), KernelError> {
    if values.is_empty() {
        return Err(KernelError::InvalidDimensions(
            "softmax input must not be empty".into(),
        ));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        denominator += *value;
    }
    for value in values {
        *value /= denominator;
    }
    Ok(())
}

fn scalar_bf16_dot(left: &[bf16], right: &[bf16]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.to_f32() * right.to_f32())
        .sum()
}

fn scalar_max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0, |maximum, value| maximum.max(value.abs()))
}

fn quantize_q8_block(values: &[f32], max_abs: f32) -> Q8Block {
    // Adapted from llama.cpp `quantize_row_q8_0_ref` in
    // `ggml/src/ggml-quants.c` at 030ebb558a5820b444a8f836ed5cdd46c9b4bd7a
    // (MIT): symmetric scale with nearest-integer signed activation values.
    let scale = max_abs / 127.0;
    let inverse_scale = if scale == 0.0 { 0.0 } else { scale.recip() };
    let mut quantized = [0_i8; QUANT_BLOCK_SIZE];
    for (destination, source) in quantized.iter_mut().zip(values) {
        *destination = (*source * inverse_scale).round().clamp(-127.0, 127.0) as i8;
    }
    Q8Block {
        scale,
        values: quantized,
    }
}

const fn mxfp4_integer(code: u8) -> i8 {
    // Semantically cross-checked against mistral.rs `mxfp4_dequantize` in
    // `mistralrs-quant/src/mxfp4/mod.rs` at
    // 8010b6a0578e416120b590ed72fd46ed5f24ee85 (MIT). Values are doubled so
    // the half step remains exact during integer accumulation.
    const VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    VALUES[(code & 0x0f) as usize]
}

fn scalar_mxfp4_q8_dot_i32(weight: &Mxfp4Block, activation: &Q8Block) -> i32 {
    weight
        .unpack()
        .iter()
        .zip(activation.values)
        .map(|(weight, activation)| *weight as i32 * activation as i32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn dispatch_rejects_synthetic_unavailable_paths() {
        let none = Features {
            avx2_fma: false,
            avx512_vnni: false,
        };
        assert_eq!(
            Kernels::with_features(KernelPath::Auto, none)
                .unwrap()
                .path(),
            KernelPath::Scalar
        );
        assert_eq!(
            Kernels::with_features(KernelPath::Avx2, none).unwrap_err(),
            KernelError::Unavailable(KernelPath::Avx2)
        );
        assert_eq!(
            Kernels::with_features(KernelPath::Avx512Vnni, none).unwrap_err(),
            KernelError::Unavailable(KernelPath::Avx512Vnni)
        );
    }

    #[test]
    fn environment_forcing_is_honored() {
        let kernels = Kernels::from_env().unwrap();
        if let Ok(value) = std::env::var("GPT_OSS_CPU_KERNEL") {
            let requested: KernelPath = value.parse().unwrap();
            if requested != KernelPath::Auto {
                assert_eq!(kernels.path(), requested);
            }
        }
    }

    #[test]
    fn all_mxfp4_codes_decode() {
        for code in 0_u8..16 {
            assert_eq!(
                decode_mxfp4(code).to_bits(),
                MXFP4_VALUES[code as usize].to_bits()
            );
        }
    }

    #[test]
    fn e8m0_scale_edges_are_bit_exact() {
        for scale in [0_u8, 1, 2, 126, 127, 128, 254, 255] {
            assert_eq!(e8m0_scale(scale).to_bits(), (scale as u32) << 23);
        }
    }

    #[test]
    fn q8_zero_block_is_stable() {
        let blocks = Kernels::new(KernelPath::Scalar)
            .unwrap()
            .quantize_q8(&[0.0; QUANT_BLOCK_SIZE])
            .unwrap();
        assert_eq!(blocks[0].scale, 0.0);
        assert_eq!(blocks[0].values, [0; QUANT_BLOCK_SIZE]);
    }

    #[test]
    fn mxfp4_integer_dot_has_official_e2m1_scale() {
        let weight = Mxfp4Block {
            scale: 127,
            packed: [0x11; MXFP4_PACKED_BYTES],
        };
        let activation = Q8Block {
            scale: 1.0,
            values: [1; QUANT_BLOCK_SIZE],
        };
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        assert_eq!(kernels.mxfp4_q8_block_dot_i32(&weight, &activation), 32);
        assert_eq!(
            kernels.mxfp4_q8_dot(&[weight], &[activation]).unwrap(),
            16.0
        );
    }

    #[test]
    fn detected_simd_matches_scalar_randomized() {
        let scalar = Kernels::new(KernelPath::Scalar).unwrap();
        let paths = [KernelPath::Avx2, KernelPath::Avx512Vnni]
            .into_iter()
            .filter_map(|path| Kernels::new(path).ok())
            .collect::<Vec<_>>();
        let mut rng = ChaCha8Rng::seed_from_u64(0x4750_544f_5353);

        for _ in 0..128 {
            let mut packed = [0_u8; MXFP4_PACKED_BYTES];
            rng.fill(&mut packed);
            let weight = Mxfp4Block {
                scale: rng.gen(),
                packed,
            };
            let mut q8 = [0_i8; QUANT_BLOCK_SIZE];
            for value in &mut q8 {
                *value = rng.gen_range(-127..=127);
            }
            let activation = Q8Block {
                scale: rng.gen_range(0.0001..2.0),
                values: q8,
            };
            for simd in &paths {
                assert_eq!(
                    simd.mxfp4_q8_block_dot_i32(&weight, &activation),
                    scalar.mxfp4_q8_block_dot_i32(&weight, &activation),
                    "{} integer dot mismatch",
                    simd.path()
                );
            }

            let activation_input = (0..64)
                .map(|_| rng.gen_range(-20.0..20.0))
                .collect::<Vec<_>>();
            let expected_q8 = scalar.quantize_q8(&activation_input).unwrap();
            for simd in &paths {
                assert_eq!(simd.quantize_q8(&activation_input).unwrap(), expected_q8);
            }

            let input = (0..64)
                .map(|_| bf16::from_f32(rng.gen_range(-2.0..2.0)))
                .collect::<Vec<_>>();
            let weights = (0..128)
                .map(|_| bf16::from_f32(rng.gen_range(-2.0..2.0)))
                .collect::<Vec<_>>();
            let mut expected = [0.0_f32; 2];
            let mut actual = [0.0_f32; 2];
            scalar
                .bf16_matvec(&weights, 2, 64, &input, &mut expected)
                .unwrap();
            for simd in &paths {
                simd.bf16_matvec(&weights, 2, 64, &input, &mut actual)
                    .unwrap();
                for (expected, actual) in expected.into_iter().zip(actual) {
                    let tolerance = 2e-4_f32.max(expected.abs() * 2e-5);
                    assert!(
                        (expected - actual).abs() <= tolerance,
                        "{}: {expected} vs {actual}",
                        simd.path()
                    );
                }
            }
        }
    }

    #[test]
    fn matvec_and_normalization_validate_shapes() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        assert!(kernels.bf16_matvec(&[], 1, 2, &[], &mut [0.0]).is_err());
        assert!(kernels.rms_norm(&[], &[], 1e-5, &mut []).is_err());
    }

    #[test]
    fn softmax_is_normalized() {
        let mut values = [1.0, 2.0, 3.0];
        softmax_in_place(&mut values).unwrap();
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
