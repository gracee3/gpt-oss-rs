#![deny(unsafe_op_in_unsafe_fn)]
//! Safe dispatch and scalar reference kernels for GPT-OSS CPU inference.
//!
//! Unsafe code is confined to the x86 implementation module. Every public
//! entry point validates dimensions and ISA availability before dispatch.

use std::fmt;
use std::str::FromStr;

use half::bf16;
use thiserror::Error;

mod amx;
mod features;
mod matmul;
mod tiger_lake;
#[cfg(target_arch = "x86_64")]
mod x86;

pub use amx::{initialize_amx_int8, AmxRuntimeError, AmxRuntimeStatus};
pub use features::{CpuFeatures, CpuHardwareIdentity, KernelRequirements};
pub use matmul::{
    Mxfp4ActivationMatrix, Mxfp4MatmulBackend, Mxfp4MatmulProblem, Mxfp4ScratchRequirement,
    Q8MatrixView, ResidualQ8MatrixView,
};
pub use tiger_lake::{
    tiger_lake_auto_matmul_backend, tiger_lake_profile_matches, Mxfp4PromotionRegion,
    TIGER_LAKE_MXFP4_PROMOTION_BENCHMARK_COMMIT, TIGER_LAKE_MXFP4_PROMOTION_EVIDENCE_SHA256,
    TIGER_LAKE_MXFP4_PROMOTION_REGIONS, TIGER_LAKE_PROFILE_KEY, TIGER_LAKE_THREAD_POLICY,
};

pub const QUANT_BLOCK_SIZE: usize = 32;
pub const MXFP4_PACKED_BYTES: usize = QUANT_BLOCK_SIZE / 2;

/// E2M1 values used by the official GPT-OSS SafeTensors representation.
pub const MXFP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp4GemvKernel {
    Scalar,
    Avx2Row,
    Avx512VnniRow,
    Avx2X8,
    Avx512VnniX8,
    ExactBf16,
}

impl Mxfp4GemvKernel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar-row",
            Self::Avx2Row => "avx2-row",
            Self::Avx512VnniRow => "avx512-vnni-row",
            Self::Avx2X8 => "avx2-x8",
            Self::Avx512VnniX8 => "avx512-vnni-x8",
            Self::ExactBf16 => "exact-bf16-row",
        }
    }
}

impl fmt::Display for Mxfp4GemvKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mxfp4WeightLayout {
    CanonicalAdjacentV1,
    InterleavedSplitX8V2,
}

impl Mxfp4WeightLayout {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalAdjacentV1 => "CanonicalAdjacentV1",
            Self::InterleavedSplitX8V2 => "InterleavedSplitX8V2",
        }
    }

    pub const fn identifier(self) -> u32 {
        match self {
            Self::CanonicalAdjacentV1 => 1,
            Self::InterleavedSplitX8V2 => 2,
        }
    }
}

impl fmt::Display for Mxfp4WeightLayout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

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
    #[error("unknown MXFP4 matrix backend '{0}'")]
    InvalidMatmulBackend(String),
    #[error("MXFP4 matrix backend '{backend}' is unavailable: {reason}")]
    UnavailableMatmulBackend {
        backend: Mxfp4MatmulBackend,
        reason: &'static str,
    },
    #[error("AMX-INT8 tile shim failed with status {0}")]
    AmxShim(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernels {
    path: KernelPath,
    dispatch_plan: DispatchPlan,
}

/// Per-operation CPU kernel selection resolved at startup.
///
/// The fields are private so callers can inspect, but cannot mutate, the plan
/// selected after host feature detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPlan {
    bf16_matvec: KernelPath,
    quantize_q8: KernelPath,
    mxfp4_q8_dot: KernelPath,
    mxfp4_gemv: Mxfp4GemvKernel,
    mxfp4_weight_layout: Mxfp4WeightLayout,
    rms_norm: KernelPath,
}

impl DispatchPlan {
    pub const fn bf16_matvec(self) -> KernelPath {
        self.bf16_matvec
    }

    pub const fn quantize_q8(self) -> KernelPath {
        self.quantize_q8
    }

    pub const fn mxfp4_q8_dot(self) -> KernelPath {
        self.mxfp4_q8_dot
    }

    pub const fn mxfp4_gemv(self) -> Mxfp4GemvKernel {
        self.mxfp4_gemv
    }

    pub const fn mxfp4_weight_layout(self) -> Mxfp4WeightLayout {
        self.mxfp4_weight_layout
    }

    pub const fn rms_norm(self) -> KernelPath {
        self.rms_norm
    }
}

impl fmt::Display for DispatchPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "bf16_matvec={}, quantize_q8={}, mxfp4_q8_dot={}, mxfp4_gemv={}, mxfp4_layout={}, rms_norm={}",
            self.bf16_matvec,
            self.quantize_q8,
            self.mxfp4_q8_dot,
            self.mxfp4_gemv,
            self.mxfp4_weight_layout,
            self.rms_norm
        )
    }
}

impl Kernels {
    pub fn new(requested: KernelPath) -> Result<Self, KernelError> {
        Self::with_features(requested, CpuFeatures::detect())
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

    fn with_features(requested: KernelPath, features: CpuFeatures) -> Result<Self, KernelError> {
        let path = match requested {
            KernelPath::Auto if features.supports(KernelRequirements::AVX512_VNNI_PATH) => {
                KernelPath::Avx512Vnni
            }
            KernelPath::Auto if features.supports(KernelRequirements::AVX2_FMA) => KernelPath::Avx2,
            KernelPath::Auto => KernelPath::Scalar,
            KernelPath::Scalar => KernelPath::Scalar,
            KernelPath::Avx2 if features.supports(KernelRequirements::AVX2_FMA) => KernelPath::Avx2,
            KernelPath::Avx512Vnni if features.supports(KernelRequirements::AVX512_VNNI_PATH) => {
                KernelPath::Avx512Vnni
            }
            unavailable => return Err(KernelError::Unavailable(unavailable)),
        };
        let dispatch_plan = if requested == KernelPath::Auto {
            DispatchPlan {
                bf16_matvec: path,
                quantize_q8: path,
                mxfp4_q8_dot: if features.supports(KernelRequirements::AVX2_MXFP4) {
                    KernelPath::Avx2
                } else {
                    path
                },
                mxfp4_gemv: if features.supports(KernelRequirements::AVX2_MXFP4) {
                    Mxfp4GemvKernel::Avx2X8
                } else {
                    Mxfp4GemvKernel::Scalar
                },
                mxfp4_weight_layout: if features.supports(KernelRequirements::AVX2_MXFP4) {
                    Mxfp4WeightLayout::InterleavedSplitX8V2
                } else {
                    Mxfp4WeightLayout::CanonicalAdjacentV1
                },
                rms_norm: path,
            }
        } else {
            DispatchPlan {
                bf16_matvec: path,
                quantize_q8: path,
                mxfp4_q8_dot: path,
                mxfp4_gemv: match path {
                    KernelPath::Scalar => Mxfp4GemvKernel::Scalar,
                    KernelPath::Avx2 => Mxfp4GemvKernel::Avx2X8,
                    KernelPath::Avx512Vnni => Mxfp4GemvKernel::Avx512VnniX8,
                    KernelPath::Auto => Mxfp4GemvKernel::Scalar,
                },
                mxfp4_weight_layout: match path {
                    KernelPath::Avx2 | KernelPath::Avx512Vnni => {
                        Mxfp4WeightLayout::InterleavedSplitX8V2
                    }
                    KernelPath::Auto | KernelPath::Scalar => Mxfp4WeightLayout::CanonicalAdjacentV1,
                },
                rms_norm: path,
            }
        };
        Ok(Self {
            path,
            dispatch_plan,
        })
    }

    pub const fn path(self) -> KernelPath {
        self.path
    }

    pub const fn dispatch_plan(self) -> DispatchPlan {
        self.dispatch_plan
    }

    /// Select the diagnostic exact-BF16 projection without changing the
    /// compatibility path or any non-MXFP4 operation.
    pub const fn with_exact_bf16_mxfp4(mut self) -> Self {
        self.dispatch_plan.mxfp4_gemv = Mxfp4GemvKernel::ExactBf16;
        self.dispatch_plan.mxfp4_weight_layout = Mxfp4WeightLayout::CanonicalAdjacentV1;
        self
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
            *destination = match self.dispatch_plan.bf16_matvec {
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
                let max_abs = match self.dispatch_plan.quantize_q8 {
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

    /// Quantize an activation row as a Q8 approximation plus a separately
    /// quantized reconstruction residual.
    pub fn quantize_residual_q8(self, input: &[f32]) -> Result<Vec<ResidualQ8Block>, KernelError> {
        if !input.len().is_multiple_of(QUANT_BLOCK_SIZE) {
            return Err(KernelError::InvalidDimensions(format!(
                "residual-Q8 input length {} is not divisible by {QUANT_BLOCK_SIZE}",
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(KernelError::NonFiniteInput);
        }

        input
            .chunks_exact(QUANT_BLOCK_SIZE)
            .map(|block| {
                let primary_max = self.max_abs(block);
                let primary = quantize_q8_block(block, primary_max);
                let mut residual_values = [0.0_f32; QUANT_BLOCK_SIZE];
                for ((destination, source), quantized) in
                    residual_values.iter_mut().zip(block).zip(primary.values)
                {
                    *destination = *source - quantized as f32 * primary.scale;
                }
                let residual_max = self.max_abs(&residual_values);
                let residual = quantize_q8_block(&residual_values, residual_max);
                Ok(ResidualQ8Block { primary, residual })
            })
            .collect()
    }

    fn max_abs(self, block: &[f32]) -> f32 {
        match self.dispatch_plan.quantize_q8 {
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
        }
    }

    /// Exact integer dot product for one packed MXFP4/Q8 block.
    pub fn mxfp4_q8_block_dot_i32(self, weight: &Mxfp4Block, activation: &Q8Block) -> i32 {
        match self.dispatch_plan.mxfp4_q8_dot {
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

    /// Two exact integer dots for one packed MXFP4/residual-Q8 block. The
    /// selected implementation unpacks the MXFP4 nibbles once for both dots.
    pub fn mxfp4_residual_q8_block_dot_i32(
        self,
        weight: &Mxfp4Block,
        activation: &ResidualQ8Block,
    ) -> [i32; 2] {
        match self.dispatch_plan.mxfp4_q8_dot {
            KernelPath::Scalar => scalar_mxfp4_residual_q8_dot_i32(weight, activation),
            #[cfg(target_arch = "x86_64")]
            KernelPath::Avx2 => {
                // SAFETY: construction verifies AVX2 and FMA support.
                unsafe { x86::mxfp4_residual_q8_dot_avx2(weight, activation) }
            }
            #[cfg(target_arch = "x86_64")]
            KernelPath::Avx512Vnni => {
                // SAFETY: construction verifies the full AVX-512 feature set.
                unsafe { x86::mxfp4_residual_q8_dot_avx512_vnni(weight, activation) }
            }
            _ => scalar_mxfp4_residual_q8_dot_i32(weight, activation),
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

    /// Dot product across matching MXFP4 and two-pass residual-Q8 rows.
    pub fn mxfp4_residual_q8_dot(
        self,
        weights: &[Mxfp4Block],
        activations: &[ResidualQ8Block],
    ) -> Result<f32, KernelError> {
        if weights.len() != activations.len() {
            return Err(KernelError::InvalidDimensions(format!(
                "MXFP4 blocks {} do not match residual-Q8 blocks {}",
                weights.len(),
                activations.len()
            )));
        }
        let mut total = 0.0_f32;
        for (weight, activation) in weights.iter().zip(activations) {
            let [primary, residual] = self.mxfp4_residual_q8_block_dot_i32(weight, activation);
            let weight_scale = 0.5 * e8m0_scale(weight.scale);
            total += primary as f32 * weight_scale * activation.primary.scale;
            total += residual as f32 * weight_scale * activation.residual.scale;
        }
        Ok(total)
    }

    /// Project one aligned output tile from an MXFP4 matrix and Q8 activation.
    ///
    /// The model runner owns parallelism and calls this once per eight-row
    /// tile. Feature and kernel selection therefore stay outside the K-block
    /// loop.
    pub fn mxfp4_q8_gemv_tile(
        self,
        weights: Mxfp4MatrixView<'_>,
        row_start: usize,
        activations: Q8ActivationView<'_>,
        bias: &[f32],
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        self.validate_mxfp4_gemv_tile(weights, row_start, activations.blocks.len(), bias, output)?;
        match self.dispatch_plan.mxfp4_gemv {
            #[cfg(target_arch = "x86_64")]
            Mxfp4GemvKernel::Avx2X8 if output.len() == 8 => {
                // SAFETY: construction verifies AVX2 support, validation
                // verifies an aligned complete x8 group and its packed layout.
                unsafe {
                    x86::mxfp4_q8_gemv_x8_avx2(
                        weights,
                        row_start,
                        activations.blocks,
                        &bias[row_start..row_start + 8],
                        output,
                    )
                };
                Ok(())
            }
            #[cfg(target_arch = "x86_64")]
            Mxfp4GemvKernel::Avx512VnniX8 if output.len() == 8 => {
                // SAFETY: construction verifies AVX-512F/BW/VNNI support;
                // validation verifies an aligned complete x8 packed group.
                unsafe {
                    x86::mxfp4_q8_gemv_x8_avx512_vnni(
                        weights,
                        row_start,
                        activations.blocks,
                        &bias[row_start..row_start + 8],
                        output,
                    )
                };
                Ok(())
            }
            kernel => self.mxfp4_q8_gemv_rows(
                weights,
                row_start,
                activations.blocks,
                bias,
                output,
                canonical_row_kernel(kernel),
            ),
        }
    }

    /// Project one aligned output tile from an MXFP4 matrix and residual-Q8
    /// activation, decoding each weight group once for both integer dots.
    pub fn mxfp4_residual_q8_gemv_tile(
        self,
        weights: Mxfp4MatrixView<'_>,
        row_start: usize,
        activations: ResidualQ8ActivationView<'_>,
        bias: &[f32],
        output: &mut [f32],
    ) -> Result<(), KernelError> {
        self.validate_mxfp4_gemv_tile(weights, row_start, activations.blocks.len(), bias, output)?;
        match self.dispatch_plan.mxfp4_gemv {
            #[cfg(target_arch = "x86_64")]
            Mxfp4GemvKernel::Avx2X8 if output.len() == 8 => {
                // SAFETY: construction verifies AVX2 support, validation
                // verifies an aligned complete x8 group and its packed layout.
                unsafe {
                    x86::mxfp4_residual_q8_gemv_x8_avx2(
                        weights,
                        row_start,
                        activations.blocks,
                        &bias[row_start..row_start + 8],
                        output,
                    )
                };
                Ok(())
            }
            #[cfg(target_arch = "x86_64")]
            Mxfp4GemvKernel::Avx512VnniX8 if output.len() == 8 => {
                // SAFETY: construction verifies AVX-512F/BW/VNNI support;
                // validation verifies an aligned complete x8 packed group.
                unsafe {
                    x86::mxfp4_residual_q8_gemv_x8_avx512_vnni(
                        weights,
                        row_start,
                        activations.blocks,
                        &bias[row_start..row_start + 8],
                        output,
                    )
                };
                Ok(())
            }
            kernel => self.mxfp4_residual_q8_gemv_rows(
                weights,
                row_start,
                activations.blocks,
                bias,
                output,
                canonical_row_kernel(kernel),
            ),
        }
    }

    fn validate_mxfp4_gemv_tile(
        self,
        weights: Mxfp4MatrixView<'_>,
        row_start: usize,
        activation_blocks: usize,
        bias: &[f32],
        output: &[f32],
    ) -> Result<(), KernelError> {
        if weights.layout != self.dispatch_plan.mxfp4_weight_layout
            || activation_blocks != weights.blocks
            || bias.len() != weights.rows
            || output.is_empty()
            || output.len() > 8
            || !row_start.is_multiple_of(8)
            || row_start
                .checked_add(output.len())
                .is_none_or(|end| end > weights.rows)
            || (output.len() < 8 && row_start + output.len() != weights.rows)
        {
            return Err(KernelError::InvalidDimensions(
                "invalid MXFP4 GEMV tile or packed layout".into(),
            ));
        }
        Ok(())
    }

    fn mxfp4_q8_gemv_rows(
        self,
        weights: Mxfp4MatrixView<'_>,
        row_start: usize,
        activations: &[Q8Block],
        bias: &[f32],
        output: &mut [f32],
        kernel: Mxfp4GemvKernel,
    ) -> Result<(), KernelError> {
        for (local_row, destination) in output.iter_mut().enumerate() {
            let row = row_start + local_row;
            let mut total = bias[row];
            for (block, activation) in activations.iter().enumerate() {
                let weight = weights.block(row, block)?;
                let integer = mxfp4_q8_block_dot_for(kernel, &weight, activation);
                total += integer as f32 * 0.5 * e8m0_scale(weight.scale) * activation.scale;
            }
            *destination = total;
        }
        Ok(())
    }

    fn mxfp4_residual_q8_gemv_rows(
        self,
        weights: Mxfp4MatrixView<'_>,
        row_start: usize,
        activations: &[ResidualQ8Block],
        bias: &[f32],
        output: &mut [f32],
        kernel: Mxfp4GemvKernel,
    ) -> Result<(), KernelError> {
        for (local_row, destination) in output.iter_mut().enumerate() {
            let row = row_start + local_row;
            let mut total = bias[row];
            for (block, activation) in activations.iter().enumerate() {
                let weight = weights.block(row, block)?;
                let [primary, residual] =
                    mxfp4_residual_q8_block_dot_for(kernel, &weight, activation);
                let weight_scale = 0.5 * e8m0_scale(weight.scale);
                total += primary as f32 * weight_scale * activation.primary.scale;
                total += residual as f32 * weight_scale * activation.residual.scale;
            }
            *destination = total;
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
        let sum_squares = match self.dispatch_plan.rms_norm {
            KernelPath::Scalar => scalar_sum_squares(input),
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
            _ => scalar_sum_squares(input),
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

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualQ8Block {
    pub primary: Q8Block,
    pub residual: Q8Block,
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

/// Borrowed view of one expert's packed MXFP4 projection matrix.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp4MatrixView<'a> {
    data: &'a [u8],
    rows: usize,
    blocks: usize,
    layout: Mxfp4WeightLayout,
}

impl<'a> Mxfp4MatrixView<'a> {
    pub fn new(
        data: &'a [u8],
        rows: usize,
        blocks: usize,
        layout: Mxfp4WeightLayout,
    ) -> Result<Self, KernelError> {
        let expected = rows
            .checked_mul(blocks)
            .and_then(|records| records.checked_mul(17))
            .ok_or_else(|| KernelError::InvalidDimensions("MXFP4 view size overflows".into()))?;
        if rows == 0 || blocks == 0 || data.len() != expected {
            return Err(KernelError::InvalidDimensions(format!(
                "MXFP4 view expects {rows}x{blocks} records ({} bytes), got {} bytes",
                expected,
                data.len()
            )));
        }
        Ok(Self {
            data,
            rows,
            blocks,
            layout,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn blocks(self) -> usize {
        self.blocks
    }

    pub const fn layout(self) -> Mxfp4WeightLayout {
        self.layout
    }

    /// Return one canonical owned block regardless of the mapped layout.
    pub fn block(self, row: usize, block: usize) -> Result<Mxfp4Block, KernelError> {
        if row >= self.rows || block >= self.blocks {
            return Err(KernelError::InvalidDimensions(format!(
                "MXFP4 block [{row}, {block}] exceeds [{}, {}]",
                self.rows, self.blocks
            )));
        }
        match self.layout {
            Mxfp4WeightLayout::CanonicalAdjacentV1 => {
                let start = (row * self.blocks + block) * 17;
                Ok(Mxfp4Block {
                    scale: self.data[start],
                    packed: self.data[start + 1..start + 17]
                        .try_into()
                        .expect("validated canonical MXFP4 record"),
                })
            }
            Mxfp4WeightLayout::InterleavedSplitX8V2 if row < self.complete_x8_rows() => {
                let group = row / 8;
                let lane = row % 8;
                let x8 = self.x8_block(group, block);
                let split = std::array::from_fn(|index| {
                    let chunk = index / 8;
                    x8[8 + chunk * 64 + lane * 8 + index % 8]
                });
                Ok(Mxfp4Block {
                    scale: x8[lane],
                    packed: mxfp4_split_to_adjacent(split),
                })
            }
            Mxfp4WeightLayout::InterleavedSplitX8V2 => {
                let tail_row = row - self.complete_x8_rows();
                let tail_start = self.complete_x8_groups() * self.blocks * 136;
                let start = tail_start + (tail_row * self.blocks + block) * 17;
                Ok(Mxfp4Block {
                    scale: self.data[start],
                    packed: self.data[start + 1..start + 17]
                        .try_into()
                        .expect("validated canonical MXFP4 tail record"),
                })
            }
        }
    }

    pub(crate) const fn complete_x8_groups(self) -> usize {
        self.rows / 8
    }

    pub(crate) const fn complete_x8_rows(self) -> usize {
        self.complete_x8_groups() * 8
    }

    pub(crate) fn x8_block(self, group: usize, block: usize) -> &'a [u8; 136] {
        debug_assert_eq!(self.layout, Mxfp4WeightLayout::InterleavedSplitX8V2);
        debug_assert!(group < self.complete_x8_groups());
        debug_assert!(block < self.blocks);
        let start = (group * self.blocks + block) * 136;
        self.data[start..start + 136]
            .try_into()
            .expect("validated x8 MXFP4 block group")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Q8ActivationView<'a> {
    blocks: &'a [Q8Block],
}

impl<'a> Q8ActivationView<'a> {
    pub const fn new(blocks: &'a [Q8Block]) -> Self {
        Self { blocks }
    }

    pub const fn blocks(self) -> &'a [Q8Block] {
        self.blocks
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResidualQ8ActivationView<'a> {
    blocks: &'a [ResidualQ8Block],
}

impl<'a> ResidualQ8ActivationView<'a> {
    pub const fn new(blocks: &'a [ResidualQ8Block]) -> Self {
        Self { blocks }
    }

    pub const fn blocks(self) -> &'a [ResidualQ8Block] {
        self.blocks
    }
}

/// Convert official adjacent-nibble packing to split-half packing.
pub fn mxfp4_adjacent_to_split(adjacent: [u8; MXFP4_PACKED_BYTES]) -> [u8; MXFP4_PACKED_BYTES] {
    let code = |index: usize| {
        let byte = adjacent[index / 2];
        if index.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        }
    };
    std::array::from_fn(|index| code(index) | (code(index + 16) << 4))
}

/// Convert split-half packing back to official adjacent-nibble packing.
pub fn mxfp4_split_to_adjacent(split: [u8; MXFP4_PACKED_BYTES]) -> [u8; MXFP4_PACKED_BYTES] {
    let code = |index: usize| {
        if index < 16 {
            split[index] & 0x0f
        } else {
            split[index - 16] >> 4
        }
    };
    std::array::from_fn(|index| code(index * 2) | (code(index * 2 + 1) << 4))
}

/// Accumulate one exact MXFP4×BF16 block into the deterministic FP32
/// reduction lanes used by the dense BF16 projection kernels.
pub fn accumulate_mxfp4_bf16_block(
    weight: &Mxfp4Block,
    activation: &[bf16; QUANT_BLOCK_SIZE],
    lanes: &mut [f32; 16],
) {
    let scale = e8m0_scale(weight.scale);
    for (packed_index, packed) in weight.packed.iter().copied().enumerate() {
        let value_index = packed_index * 2;
        let low = bf16::from_f32(decode_mxfp4(packed & 0x0f) * scale).to_f32();
        let high = bf16::from_f32(decode_mxfp4(packed >> 4) * scale).to_f32();
        lanes[value_index % lanes.len()] += low * activation[value_index].to_f32();
        lanes[(value_index + 1) % lanes.len()] += high * activation[value_index + 1].to_f32();
    }
}

/// Decode one FP4 E2M1 code.
pub const fn decode_mxfp4(code: u8) -> f32 {
    MXFP4_VALUES[(code & 0x0f) as usize]
}

/// Decode one OCP MX E8M0 scale.
///
/// Finite normal encodings are exact powers of two. The two special encodings
/// are not obtained by shifting the byte into an IEEE-754 exponent field:
/// zero denotes `2^-127`, while `0xff` is invalid and produces NaN.
pub const fn e8m0_scale(scale: u8) -> f32 {
    match scale {
        0 => f32::from_bits(0x0040_0000),
        0xff => f32::NAN,
        scale => f32::from_bits((scale as u32) << 23),
    }
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
    // Keep the logical reduction lanes identical to AVX-512 and to the two
    // AVX2 accumulators. Besides reproducibility, this prevents long model
    // projections from crossing a BF16 boundary solely because dispatch chose
    // a different horizontal-reduction tree.
    let mut lanes = [0.0_f32; 16];
    let mut left_chunks = left.chunks_exact(lanes.len());
    let mut right_chunks = right.chunks_exact(lanes.len());
    for (left, right) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        for lane in 0..lanes.len() {
            lanes[lane] += left[lane].to_f32() * right[lane].to_f32();
        }
    }
    for (lane, (left, right)) in left_chunks
        .remainder()
        .iter()
        .zip(right_chunks.remainder())
        .enumerate()
    {
        lanes[lane] += left.to_f32() * right.to_f32();
    }
    lanes.into_iter().sum()
}

fn scalar_sum_squares(values: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 16];
    let mut chunks = values.chunks_exact(lanes.len());
    for values in chunks.by_ref() {
        for lane in 0..lanes.len() {
            lanes[lane] += values[lane] * values[lane];
        }
    }
    for (lane, value) in chunks.remainder().iter().copied().enumerate() {
        lanes[lane] += value * value;
    }
    lanes.into_iter().sum()
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

fn scalar_mxfp4_residual_q8_dot_i32(weight: &Mxfp4Block, activation: &ResidualQ8Block) -> [i32; 2] {
    let unpacked = weight.unpack();
    let mut primary = 0_i32;
    let mut residual = 0_i32;
    for ((weight, primary_value), residual_value) in unpacked
        .iter()
        .zip(activation.primary.values)
        .zip(activation.residual.values)
    {
        primary += *weight as i32 * primary_value as i32;
        residual += *weight as i32 * residual_value as i32;
    }
    [primary, residual]
}

const fn canonical_row_kernel(kernel: Mxfp4GemvKernel) -> Mxfp4GemvKernel {
    match kernel {
        Mxfp4GemvKernel::Avx2X8 => Mxfp4GemvKernel::Avx2Row,
        Mxfp4GemvKernel::Avx512VnniX8 => Mxfp4GemvKernel::Avx512VnniRow,
        other => other,
    }
}

fn mxfp4_q8_block_dot_for(
    kernel: Mxfp4GemvKernel,
    weight: &Mxfp4Block,
    activation: &Q8Block,
) -> i32 {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Mxfp4GemvKernel::Avx2Row => {
            // SAFETY: this kernel is present in a dispatch plan only after
            // AVX2 capability validation.
            unsafe { x86::mxfp4_q8_dot_avx2(weight, activation) }
        }
        #[cfg(target_arch = "x86_64")]
        Mxfp4GemvKernel::Avx512VnniRow => {
            // SAFETY: this kernel is present in a dispatch plan only after its
            // exact AVX2+AVX-512 VL/VNNI requirements are validated.
            unsafe { x86::mxfp4_q8_dot_avx512_vnni(weight, activation) }
        }
        _ => scalar_mxfp4_q8_dot_i32(weight, activation),
    }
}

fn mxfp4_residual_q8_block_dot_for(
    kernel: Mxfp4GemvKernel,
    weight: &Mxfp4Block,
    activation: &ResidualQ8Block,
) -> [i32; 2] {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Mxfp4GemvKernel::Avx2Row => {
            // SAFETY: dispatch validated AVX2 and the implementation reuses
            // one decoded weight vector for both dots.
            unsafe { x86::mxfp4_residual_q8_dot_avx2(weight, activation) }
        }
        #[cfg(target_arch = "x86_64")]
        Mxfp4GemvKernel::Avx512VnniRow => {
            // SAFETY: dispatch validated AVX2+AVX-512 VL/VNNI.
            unsafe { x86::mxfp4_residual_q8_dot_avx512_vnni(weight, activation) }
        }
        _ => scalar_mxfp4_residual_q8_dot_i32(weight, activation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn pack_canonical(weights: &[Mxfp4Block]) -> Vec<u8> {
        let mut data = Vec::with_capacity(weights.len() * 17);
        for weight in weights {
            data.push(weight.scale);
            data.extend_from_slice(&weight.packed);
        }
        data
    }

    fn pack_x8(weights: &[Mxfp4Block], rows: usize, blocks: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(weights.len() * 17);
        for group in 0..rows / 8 {
            for block in 0..blocks {
                for lane in 0..8 {
                    data.push(weights[((group * 8 + lane) * blocks) + block].scale);
                }
                let split = std::array::from_fn::<_, 8, _>(|lane| {
                    mxfp4_adjacent_to_split(weights[((group * 8 + lane) * blocks) + block].packed)
                });
                for chunk in 0..2 {
                    for row in &split {
                        data.extend_from_slice(&row[chunk * 8..chunk * 8 + 8]);
                    }
                }
            }
        }
        for row in rows / 8 * 8..rows {
            for block in 0..blocks {
                let weight = &weights[row * blocks + block];
                data.push(weight.scale);
                data.extend_from_slice(&weight.packed);
            }
        }
        assert_eq!(data.len(), weights.len() * 17);
        data
    }

    fn project_legacy_q8(
        kernels: Kernels,
        weights: &[Mxfp4Block],
        rows: usize,
        blocks: usize,
        activations: &[Q8Block],
        bias: &[f32],
    ) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let mut total = bias[row];
                for block in 0..blocks {
                    let weight = &weights[row * blocks + block];
                    let integer = kernels.mxfp4_q8_block_dot_i32(weight, &activations[block]);
                    total +=
                        integer as f32 * 0.5 * e8m0_scale(weight.scale) * activations[block].scale;
                }
                total
            })
            .collect()
    }

    fn project_legacy_residual(
        kernels: Kernels,
        weights: &[Mxfp4Block],
        rows: usize,
        blocks: usize,
        activations: &[ResidualQ8Block],
        bias: &[f32],
    ) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let mut total = bias[row];
                for block in 0..blocks {
                    let weight = &weights[row * blocks + block];
                    let [primary, residual] =
                        kernels.mxfp4_residual_q8_block_dot_i32(weight, &activations[block]);
                    let weight_scale = 0.5 * e8m0_scale(weight.scale);
                    total += primary as f32 * weight_scale * activations[block].primary.scale;
                    total += residual as f32 * weight_scale * activations[block].residual.scale;
                }
                total
            })
            .collect()
    }

    fn run_x8_projection_case(rows: usize, blocks: usize, seed: u64) {
        run_x8_projection_case_for(KernelPath::Avx2, rows, blocks, seed);
    }

    fn run_x8_projection_case_for(path: KernelPath, rows: usize, blocks: usize, seed: u64) {
        let Ok(x8) = Kernels::new(path) else {
            return;
        };
        let scalar = Kernels::new(KernelPath::Scalar).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let weights = (0..rows * blocks)
            .map(|index| Mxfp4Block {
                scale: if index % 31 == 0 {
                    0
                } else {
                    rng.gen_range(110..=135)
                },
                packed: std::array::from_fn(|_| rng.gen()),
            })
            .collect::<Vec<_>>();
        let q8 = (0..blocks)
            .map(|index| Q8Block {
                scale: if index % 17 == 0 {
                    0.0
                } else {
                    rng.gen_range(0.00001..0.02)
                },
                values: std::array::from_fn(|lane| match (index + lane) % 19 {
                    0 => -127,
                    1 => 127,
                    _ => rng.gen_range(-127..=127),
                }),
            })
            .collect::<Vec<_>>();
        let residual = q8
            .iter()
            .cloned()
            .map(|primary| ResidualQ8Block {
                primary,
                residual: Q8Block {
                    scale: rng.gen_range(0.0000001..0.0002),
                    values: std::array::from_fn(|_| rng.gen_range(-127..=127)),
                },
            })
            .collect::<Vec<_>>();
        let bias = (0..rows)
            .map(|_| rng.gen_range(-0.25..0.25))
            .collect::<Vec<_>>();

        let x8_data = pack_x8(&weights, rows, blocks);
        let x8_view = Mxfp4MatrixView::new(
            &x8_data,
            rows,
            blocks,
            Mxfp4WeightLayout::InterleavedSplitX8V2,
        )
        .unwrap();
        for row in 0..rows {
            for block in 0..blocks {
                assert_eq!(
                    x8_view.block(row, block).unwrap(),
                    weights[row * blocks + block]
                );
            }
        }

        let expected_q8 = project_legacy_q8(x8, &weights, rows, blocks, &q8, &bias);
        let expected_residual =
            project_legacy_residual(x8, &weights, rows, blocks, &residual, &bias);
        let scalar_q8 = project_legacy_q8(scalar, &weights, rows, blocks, &q8, &bias);
        let scalar_residual =
            project_legacy_residual(scalar, &weights, rows, blocks, &residual, &bias);
        assert_eq!(expected_q8, scalar_q8);
        assert_eq!(expected_residual, scalar_residual);

        let mut actual_q8 = vec![0.0; rows];
        let mut actual_residual = vec![0.0; rows];
        for (tile, output) in actual_q8.chunks_mut(8).enumerate() {
            x8.mxfp4_q8_gemv_tile(x8_view, tile * 8, Q8ActivationView::new(&q8), &bias, output)
                .unwrap();
        }
        for (tile, output) in actual_residual.chunks_mut(8).enumerate() {
            x8.mxfp4_residual_q8_gemv_tile(
                x8_view,
                tile * 8,
                ResidualQ8ActivationView::new(&residual),
                &bias,
                output,
            )
            .unwrap();
        }
        assert_eq!(actual_q8, expected_q8);
        assert_eq!(actual_residual, expected_residual);

        let canonical_data = pack_canonical(&weights);
        let canonical_view = Mxfp4MatrixView::new(
            &canonical_data,
            rows,
            blocks,
            Mxfp4WeightLayout::CanonicalAdjacentV1,
        )
        .unwrap();
        let mut scalar_projection = vec![0.0; rows];
        for (tile, output) in scalar_projection.chunks_mut(8).enumerate() {
            scalar
                .mxfp4_q8_gemv_tile(
                    canonical_view,
                    tile * 8,
                    Q8ActivationView::new(&q8),
                    &bias,
                    output,
                )
                .unwrap();
        }
        assert_eq!(scalar_projection, expected_q8);
    }

    #[test]
    fn dispatch_rejects_synthetic_unavailable_paths() {
        let none = CpuFeatures::NONE;
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

        let avx2_without_fma = CpuFeatures {
            avx2: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(
            Kernels::with_features(KernelPath::Avx2, avx2_without_fma).unwrap_err(),
            KernelError::Unavailable(KernelPath::Avx2)
        );

        let avx512_without_bw = CpuFeatures {
            avx2: true,
            avx512_f: true,
            avx512_vl: true,
            avx512_vnni: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(
            Kernels::with_features(KernelPath::Avx512Vnni, avx512_without_bw).unwrap_err(),
            KernelError::Unavailable(KernelPath::Avx512Vnni)
        );
    }

    #[test]
    fn requirements_keep_extended_capabilities_independent() {
        let features = CpuFeatures {
            avx_vnni: true,
            avx512_vbmi2: true,
            amx_tile: true,
            amx_int8: true,
            avx10_2: true,
            avx10_256: true,
            ..CpuFeatures::NONE
        };
        let present = KernelRequirements::AVX_VNNI
            .union(KernelRequirements::AVX512_VBMI2)
            .union(KernelRequirements::AMX_TILE)
            .union(KernelRequirements::AMX_INT8)
            .union(KernelRequirements::AVX10_2)
            .union(KernelRequirements::AVX10_256);
        assert!(features.supports(present));
        assert!(!features.supports(present.union(KernelRequirements::AVX2)));
        assert!(!features.supports(present.union(KernelRequirements::AVX512_VBMI)));
        assert_eq!(CpuFeatures::detect(), CpuFeatures::detect());
    }

    #[test]
    fn automatic_dispatch_uses_exact_operation_requirements() {
        let avx2_without_fma = CpuFeatures {
            avx2: true,
            ..CpuFeatures::NONE
        };
        let kernels = Kernels::with_features(KernelPath::Auto, avx2_without_fma).unwrap();
        let plan = kernels.dispatch_plan();
        assert_eq!(kernels.path(), KernelPath::Scalar);
        assert_eq!(plan.bf16_matvec(), KernelPath::Scalar);
        assert_eq!(plan.quantize_q8(), KernelPath::Scalar);
        assert_eq!(plan.rms_norm(), KernelPath::Scalar);
        assert_eq!(plan.mxfp4_q8_dot(), KernelPath::Avx2);
        assert_eq!(plan.mxfp4_gemv(), Mxfp4GemvKernel::Avx2X8);
    }

    #[test]
    fn automatic_dispatch_uses_capabilities_without_generation_names() {
        let both = CpuFeatures {
            avx2: true,
            fma: true,
            avx512_f: true,
            avx512_bw: true,
            avx512_vl: true,
            avx512_vnni: true,
            ..CpuFeatures::NONE
        };
        let kernels = Kernels::with_features(KernelPath::Auto, both).unwrap();
        let plan = kernels.dispatch_plan();
        assert_eq!(kernels.path(), KernelPath::Avx512Vnni);
        assert_eq!(plan.bf16_matvec(), KernelPath::Avx512Vnni);
        assert_eq!(plan.quantize_q8(), KernelPath::Avx512Vnni);
        assert_eq!(plan.mxfp4_q8_dot(), KernelPath::Avx2);
        assert_eq!(plan.mxfp4_gemv(), Mxfp4GemvKernel::Avx2X8);
        assert_eq!(
            plan.mxfp4_weight_layout(),
            Mxfp4WeightLayout::InterleavedSplitX8V2
        );
        assert_eq!(plan.rms_norm(), KernelPath::Avx512Vnni);
        assert_eq!(
            plan.to_string(),
            "bf16_matvec=avx512-vnni, quantize_q8=avx512-vnni, mxfp4_q8_dot=avx2, mxfp4_gemv=avx2-x8, mxfp4_layout=InterleavedSplitX8V2, rms_norm=avx512-vnni"
        );
    }

    #[test]
    fn avx512_vnni_does_not_require_vbmi() {
        let without_vbmi = CpuFeatures {
            avx2: true,
            fma: true,
            avx512_f: true,
            avx512_bw: true,
            avx512_vl: true,
            avx512_vnni: true,
            avx512_vbmi: false,
            ..CpuFeatures::NONE
        };
        let forced = Kernels::with_features(KernelPath::Avx512Vnni, without_vbmi).unwrap();
        assert_eq!(forced.path(), KernelPath::Avx512Vnni);
        assert_eq!(
            forced.dispatch_plan().mxfp4_gemv(),
            Mxfp4GemvKernel::Avx512VnniX8
        );
        assert_eq!(
            forced.dispatch_plan().mxfp4_weight_layout(),
            Mxfp4WeightLayout::InterleavedSplitX8V2
        );
    }

    #[test]
    fn forced_dispatch_uses_one_path_for_every_operation() {
        let both = CpuFeatures {
            avx2: true,
            fma: true,
            avx512_f: true,
            avx512_bw: true,
            avx512_vl: true,
            avx512_vnni: true,
            ..CpuFeatures::NONE
        };
        for path in [KernelPath::Scalar, KernelPath::Avx2, KernelPath::Avx512Vnni] {
            let plan = Kernels::with_features(path, both).unwrap().dispatch_plan();
            assert_eq!(plan.bf16_matvec(), path);
            assert_eq!(plan.quantize_q8(), path);
            assert_eq!(plan.mxfp4_q8_dot(), path);
            assert_eq!(plan.rms_norm(), path);
        }
        let avx2 = Kernels::with_features(KernelPath::Avx2, both)
            .unwrap()
            .dispatch_plan();
        assert_eq!(avx2.mxfp4_gemv(), Mxfp4GemvKernel::Avx2X8);
        assert_eq!(
            avx2.mxfp4_weight_layout(),
            Mxfp4WeightLayout::InterleavedSplitX8V2
        );
        let avx512 = Kernels::with_features(KernelPath::Avx512Vnni, both)
            .unwrap()
            .dispatch_plan();
        assert_eq!(avx512.mxfp4_gemv(), Mxfp4GemvKernel::Avx512VnniX8);
        assert_eq!(
            avx512.mxfp4_weight_layout(),
            Mxfp4WeightLayout::InterleavedSplitX8V2
        );
    }

    #[test]
    fn exact_bf16_overrides_only_mxfp4_dispatch() {
        let features = CpuFeatures {
            avx2: true,
            fma: true,
            ..CpuFeatures::NONE
        };
        let normal = Kernels::with_features(KernelPath::Avx2, features).unwrap();
        let exact = normal.with_exact_bf16_mxfp4();
        assert_eq!(exact.path(), normal.path());
        assert_eq!(
            exact.dispatch_plan().bf16_matvec(),
            normal.dispatch_plan().bf16_matvec()
        );
        assert_eq!(
            exact.dispatch_plan().mxfp4_gemv(),
            Mxfp4GemvKernel::ExactBf16
        );
        assert_eq!(
            exact.dispatch_plan().mxfp4_weight_layout(),
            Mxfp4WeightLayout::CanonicalAdjacentV1
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
    fn adjacent_split_packing_round_trips_all_codes() {
        let adjacent = std::array::from_fn(|index| {
            let low = (index * 2) as u8 & 0x0f;
            let high = (index * 2 + 1) as u8 & 0x0f;
            low | (high << 4)
        });
        let split = mxfp4_adjacent_to_split(adjacent);
        for (index, byte) in split.into_iter().enumerate() {
            assert_eq!(byte & 0x0f, index as u8);
            assert_eq!(byte >> 4, index as u8);
        }
        assert_eq!(mxfp4_split_to_adjacent(split), adjacent);
    }

    #[test]
    fn x8_q8_and_residual_projections_match_canonical_for_every_tail() {
        for tail in 1..=7 {
            run_x8_projection_case(8 + tail, 5, 0x5820_0000 + tail as u64);
            run_x8_projection_case_for(
                KernelPath::Avx512Vnni,
                8 + tail,
                if tail <= 2 { tail } else { 5 },
                0x5120_0000 + tail as u64,
            );
        }
    }

    #[test]
    fn x8_e8m0_invalid_scale_propagates_nan_on_every_projection_path() {
        let weights = vec![
            Mxfp4Block {
                scale: 0xff,
                packed: [0x11; MXFP4_PACKED_BYTES],
            };
            8
        ];
        let activation = [Q8Block {
            scale: 1.0,
            values: [1; QUANT_BLOCK_SIZE],
        }];
        let bias = [0.0; 8];
        for path in [KernelPath::Scalar, KernelPath::Avx2, KernelPath::Avx512Vnni] {
            let Ok(kernels) = Kernels::new(path) else {
                continue;
            };
            let data = match kernels.dispatch_plan().mxfp4_weight_layout() {
                Mxfp4WeightLayout::CanonicalAdjacentV1 => pack_canonical(&weights),
                Mxfp4WeightLayout::InterleavedSplitX8V2 => pack_x8(&weights, 8, 1),
            };
            let view =
                Mxfp4MatrixView::new(&data, 8, 1, kernels.dispatch_plan().mxfp4_weight_layout())
                    .unwrap();
            let mut output = [0.0; 8];
            kernels
                .mxfp4_q8_gemv_tile(
                    view,
                    0,
                    Q8ActivationView::new(&activation),
                    &bias,
                    &mut output,
                )
                .unwrap();
            assert!(output.into_iter().all(f32::is_nan), "path={path}");
        }
    }

    #[test]
    fn x8_handles_zero_blocks_extrema_and_e8m0_exponents() {
        let Ok(x8) = Kernels::new(KernelPath::Avx2) else {
            return;
        };
        let rows = 8;
        let blocks = 8;
        let exponents = [0_u8, 1, 2, 100, 126, 127, 128, 200];
        let patterns = [0x00_u8, 0x88, 0x77, 0xff, 0x70, 0xf0, 0x07, 0x0f];
        let weights = (0..rows * blocks)
            .map(|index| Mxfp4Block {
                scale: exponents[index % blocks],
                packed: [patterns[(index / blocks + index) % patterns.len()]; 16],
            })
            .collect::<Vec<_>>();
        let q8 = (0..blocks)
            .map(|block| Q8Block {
                scale: if block == 0 { 0.0 } else { 0.125 },
                values: std::array::from_fn(|index| match index % 3 {
                    0 => -127,
                    1 => 127,
                    _ => 0,
                }),
            })
            .collect::<Vec<_>>();
        let residual = q8
            .iter()
            .cloned()
            .map(|primary| ResidualQ8Block {
                primary,
                residual: Q8Block {
                    scale: 0.000_976_562_5,
                    values: std::array::from_fn(|index| if index % 2 == 0 { 127 } else { -127 }),
                },
            })
            .collect::<Vec<_>>();
        let bias = [0.0_f32; 8];
        let data = pack_x8(&weights, rows, blocks);
        let view =
            Mxfp4MatrixView::new(&data, rows, blocks, Mxfp4WeightLayout::InterleavedSplitX8V2)
                .unwrap();
        let expected_q8 = project_legacy_q8(x8, &weights, rows, blocks, &q8, &bias);
        let expected_residual =
            project_legacy_residual(x8, &weights, rows, blocks, &residual, &bias);
        let mut actual_q8 = [0.0; 8];
        let mut actual_residual = [0.0; 8];
        x8.mxfp4_q8_gemv_tile(view, 0, Q8ActivationView::new(&q8), &bias, &mut actual_q8)
            .unwrap();
        x8.mxfp4_residual_q8_gemv_tile(
            view,
            0,
            ResidualQ8ActivationView::new(&residual),
            &bias,
            &mut actual_residual,
        )
        .unwrap();
        assert_eq!(actual_q8.as_slice(), expected_q8);
        assert_eq!(actual_residual.as_slice(), expected_residual);
    }

    #[test]
    fn x8_matches_gate_up_and_down_projection_shapes() {
        for (path, seed_offset) in [(KernelPath::Avx2, 0), (KernelPath::Avx512Vnni, 0x1000_0000)] {
            run_x8_projection_case_for(
                path,
                5760,
                2880 / QUANT_BLOCK_SIZE,
                0x5820_5760 + seed_offset,
            );
            run_x8_projection_case_for(
                path,
                2880,
                2880 / QUANT_BLOCK_SIZE,
                0x5820_2880 + seed_offset,
            );
        }
    }

    #[test]
    fn e8m0_scale_edges_are_bit_exact() {
        assert_eq!(e8m0_scale(0).to_bits(), 0x0040_0000);
        for scale in [1_u8, 2, 126, 127, 128, 254] {
            assert_eq!(e8m0_scale(scale).to_bits(), (scale as u32) << 23);
        }
        assert!(e8m0_scale(0xff).is_nan());
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
    fn residual_q8_reconstructs_better_than_one_pass_q8() {
        let input = std::array::from_fn::<_, QUANT_BLOCK_SIZE, _>(|index| {
            ((index as f32 * 0.731).sin() * 9.0) + index as f32 * 0.013
        });
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let q8 = kernels.quantize_q8(&input).unwrap().remove(0);
        let residual = kernels.quantize_residual_q8(&input).unwrap().remove(0);
        let q8_error = input
            .iter()
            .zip(q8.values)
            .map(|(source, value)| (source - value as f32 * q8.scale).abs())
            .sum::<f32>();
        let residual_error = input
            .iter()
            .zip(residual.primary.values)
            .zip(residual.residual.values)
            .map(|((source, primary), correction)| {
                (source
                    - primary as f32 * residual.primary.scale
                    - correction as f32 * residual.residual.scale)
                    .abs()
            })
            .sum::<f32>();
        assert!(
            residual_error < q8_error * 0.02,
            "{residual_error} vs {q8_error}"
        );
    }

    #[test]
    fn residual_q8_zero_block_is_stable() {
        let block = Kernels::new(KernelPath::Scalar)
            .unwrap()
            .quantize_residual_q8(&[0.0; QUANT_BLOCK_SIZE])
            .unwrap()
            .remove(0);
        assert_eq!(block.primary.scale, 0.0);
        assert_eq!(block.primary.values, [0; QUANT_BLOCK_SIZE]);
        assert_eq!(block.residual.scale, 0.0);
        assert_eq!(block.residual.values, [0; QUANT_BLOCK_SIZE]);
    }

    #[test]
    fn residual_q8_rejects_non_finite_input() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut input = [0.0; QUANT_BLOCK_SIZE];
            input[7] = value;
            assert_eq!(
                kernels.quantize_residual_q8(&input).unwrap_err(),
                KernelError::NonFiniteInput
            );
        }
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
    fn fused_residual_dot_matches_two_independent_dots() {
        let weight = Mxfp4Block {
            scale: 129,
            packed: std::array::from_fn(|index| (index as u8) | ((15 - index) as u8) << 4),
        };
        let activation = ResidualQ8Block {
            primary: Q8Block {
                scale: 0.125,
                values: std::array::from_fn(|index| index as i8 - 16),
            },
            residual: Q8Block {
                scale: 0.001,
                values: std::array::from_fn(|index| 31 - index as i8),
            },
        };
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        assert_eq!(
            kernels.mxfp4_residual_q8_block_dot_i32(&weight, &activation),
            [
                kernels.mxfp4_q8_block_dot_i32(&weight, &activation.primary),
                kernels.mxfp4_q8_block_dot_i32(&weight, &activation.residual),
            ]
        );
    }

    #[test]
    fn exact_bf16_block_dot_uses_bf16_weight_boundaries_and_reduction_lanes() {
        let weight = Mxfp4Block {
            scale: 124,
            packed: [
                0x75, 0x31, 0xed, 0xa6, 0x42, 0xf9, 0x8b, 0xc4, 0x17, 0x5e, 0x20, 0xda, 0x63, 0xbf,
                0x94, 0x28,
            ],
        };
        let activation = std::array::from_fn(|index| {
            bf16::from_f32(((index as f32 - 15.5) * 0.137).sin() * 2.25)
        });
        let mut lanes = [0.0_f32; 16];
        accumulate_mxfp4_bf16_block(&weight, &activation, &mut lanes);
        assert_eq!(
            lanes.map(f32::to_bits),
            [
                3206160384, 3217183744, 3204644864, 3169845248, 1062731776, 1067950080, 3215392768,
                3203301376, 1041563648, 1059454976, 3217698816, 1057914880, 1061216256, 3188326400,
                3186294784, 1049509888,
            ]
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

                let residual = ResidualQ8Block {
                    primary: activation.clone(),
                    residual: Q8Block {
                        scale: rng.gen_range(0.000001..0.01),
                        values: std::array::from_fn(|_| rng.gen_range(-127..=127)),
                    },
                };
                assert_eq!(
                    simd.mxfp4_residual_q8_block_dot_i32(&weight, &residual),
                    scalar.mxfp4_residual_q8_block_dot_i32(&weight, &residual),
                    "{} fused residual dot mismatch",
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
                    assert_eq!(
                        expected.to_bits(),
                        actual.to_bits(),
                        "{}: {expected} vs {actual}",
                        simd.path()
                    );
                }

                let rms_input = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
                let rms_weight = weights[..64]
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>();
                let mut expected_rms = [0.0_f32; 64];
                let mut actual_rms = [0.0_f32; 64];
                scalar
                    .rms_norm(&rms_input, &rms_weight, 1e-5, &mut expected_rms)
                    .unwrap();
                simd.rms_norm(&rms_input, &rms_weight, 1e-5, &mut actual_rms)
                    .unwrap();
                assert_eq!(actual_rms, expected_rms, "{} RMS mismatch", simd.path());
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
