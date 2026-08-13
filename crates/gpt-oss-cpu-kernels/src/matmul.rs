use std::fmt;
use std::str::FromStr;

use crate::{
    e8m0_scale, CpuFeatures, KernelError, KernelRequirements, Kernels, Mxfp4MatrixView,
    Mxfp4WeightLayout, Q8ActivationView, Q8Block, ResidualQ8ActivationView, ResidualQ8Block,
    QUANT_BLOCK_SIZE,
};

const AVX2_PANEL_ROWS: usize = 4;
const AVX2_PANEL_PASS_BYTES: usize =
    4 * std::mem::size_of::<f32>() + AVX2_PANEL_ROWS * QUANT_BLOCK_SIZE;
#[cfg(feature = "amx-int8")]
const AMX_TILE_ROWS: usize = 16;
#[cfg(feature = "amx-int8")]
const AMX_TILE_OUTPUTS: usize = 16;
#[cfg(feature = "amx-int8")]
const AMX_A_PANEL_BYTES: usize = AMX_TILE_ROWS * QUANT_BLOCK_SIZE;
#[cfg(feature = "amx-int8")]
const AMX_B_PANEL_BYTES: usize = 8 * 64;
#[cfg(feature = "amx-int8")]
const AMX_C_TILE_BYTES: usize = AMX_TILE_ROWS * AMX_TILE_OUTPUTS * 4;
#[cfg(feature = "amx-int8")]
const AMX_SCRATCH_BYTES: usize = AMX_A_PANEL_BYTES + AMX_B_PANEL_BYTES + AMX_C_TILE_BYTES;

/// Explicit MXFP4 matrix implementation preference.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp4MatmulBackend {
    #[default]
    Auto,
    Scalar,
    Avx2,
    Avx512Vnni,
    AmxInt8,
}

impl Mxfp4MatmulBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512Vnni => "avx512-vnni",
            Self::AmxInt8 => "amx-int8",
        }
    }

    const fn resolve(self, rows: usize) -> Self {
        match self {
            Self::Auto if rows == 1 => Self::Auto,
            Self::Auto => Self::Scalar,
            backend => backend,
        }
    }

    /// Query transient caller-owned storage before executing a problem.
    pub fn scratch_requirement(
        self,
        problem: &Mxfp4MatmulProblem<'_>,
    ) -> Result<Mxfp4ScratchRequirement, KernelError> {
        match self.resolve(problem.m()) {
            Self::Auto | Self::Scalar => Ok(Mxfp4ScratchRequirement::NONE),
            Self::Avx2 | Self::Avx512Vnni => {
                let passes = match problem.activations {
                    Mxfp4ActivationMatrix::Q8(_) => 1,
                    Mxfp4ActivationMatrix::ResidualQ8(_) => 2,
                };
                let bytes = problem
                    .weights
                    .blocks()
                    .checked_mul(passes)
                    .and_then(|value| value.checked_mul(AVX2_PANEL_PASS_BYTES))
                    .ok_or_else(|| {
                        KernelError::InvalidDimensions(
                            "AVX2 MXFP4 activation-panel size overflows".into(),
                        )
                    })?;
                Ok(Mxfp4ScratchRequirement {
                    size: bytes,
                    alignment: 32,
                })
            }
            Self::AmxInt8 => amx_scratch_requirement(problem),
        }
    }
}

impl fmt::Display for Mxfp4MatmulBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Mxfp4MatmulBackend {
    type Err = KernelError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "scalar" => Ok(Self::Scalar),
            "avx2" => Ok(Self::Avx2),
            "avx512-vnni" | "avx512_vnni" => Ok(Self::Avx512Vnni),
            "amx-int8" | "amx_int8" => Ok(Self::AmxInt8),
            _ => Err(KernelError::InvalidMatmulBackend(value.to_string())),
        }
    }
}

/// Required byte count and starting-address alignment for transient scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp4ScratchRequirement {
    pub size: usize,
    pub alignment: usize,
}

impl Mxfp4ScratchRequirement {
    pub const NONE: Self = Self {
        size: 0,
        alignment: 1,
    };
}

/// Row-major typed view over Q8 activation blocks.
#[derive(Debug, Clone, Copy)]
pub struct Q8MatrixView<'a> {
    blocks: &'a [Q8Block],
    rows: usize,
    blocks_per_row: usize,
    row_stride: usize,
}

impl<'a> Q8MatrixView<'a> {
    pub fn new(
        blocks: &'a [Q8Block],
        rows: usize,
        blocks_per_row: usize,
        row_stride: usize,
    ) -> Result<Self, KernelError> {
        validate_activation_extent(blocks.len(), rows, blocks_per_row, row_stride, "Q8")?;
        Ok(Self {
            blocks,
            rows,
            blocks_per_row,
            row_stride,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn blocks_per_row(self) -> usize {
        self.blocks_per_row
    }

    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    pub fn row(self, row: usize) -> Result<&'a [Q8Block], KernelError> {
        activation_row(
            self.blocks,
            self.rows,
            self.blocks_per_row,
            self.row_stride,
            row,
            "Q8",
        )
    }
}

/// Row-major typed view over residual-Q8 activation blocks.
#[derive(Debug, Clone, Copy)]
pub struct ResidualQ8MatrixView<'a> {
    blocks: &'a [ResidualQ8Block],
    rows: usize,
    blocks_per_row: usize,
    row_stride: usize,
}

impl<'a> ResidualQ8MatrixView<'a> {
    pub fn new(
        blocks: &'a [ResidualQ8Block],
        rows: usize,
        blocks_per_row: usize,
        row_stride: usize,
    ) -> Result<Self, KernelError> {
        validate_activation_extent(
            blocks.len(),
            rows,
            blocks_per_row,
            row_stride,
            "residual-Q8",
        )?;
        Ok(Self {
            blocks,
            rows,
            blocks_per_row,
            row_stride,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn blocks_per_row(self) -> usize {
        self.blocks_per_row
    }

    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    pub fn row(self, row: usize) -> Result<&'a [ResidualQ8Block], KernelError> {
        activation_row(
            self.blocks,
            self.rows,
            self.blocks_per_row,
            self.row_stride,
            row,
            "residual-Q8",
        )
    }
}

fn validate_activation_extent(
    len: usize,
    rows: usize,
    blocks_per_row: usize,
    row_stride: usize,
    kind: &str,
) -> Result<(), KernelError> {
    let required = rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(row_stride))
        .and_then(|offset| offset.checked_add(blocks_per_row));
    if rows == 0
        || blocks_per_row == 0
        || row_stride < blocks_per_row
        || required.is_none_or(|required| required > len)
    {
        return Err(KernelError::InvalidDimensions(format!(
            "invalid {kind} matrix view: rows={rows}, blocks={blocks_per_row}, stride={row_stride}, len={len}"
        )));
    }
    Ok(())
}

fn activation_row<'a, T>(
    values: &'a [T],
    rows: usize,
    blocks_per_row: usize,
    row_stride: usize,
    row: usize,
    kind: &str,
) -> Result<&'a [T], KernelError> {
    if row >= rows {
        return Err(KernelError::InvalidDimensions(format!(
            "{kind} activation row {row} exceeds {rows}"
        )));
    }
    let start = row * row_stride;
    Ok(&values[start..start + blocks_per_row])
}

#[derive(Debug, Clone, Copy)]
pub enum Mxfp4ActivationMatrix<'a> {
    Q8(Q8MatrixView<'a>),
    ResidualQ8(ResidualQ8MatrixView<'a>),
}

impl Mxfp4ActivationMatrix<'_> {
    const fn rows(self) -> usize {
        match self {
            Self::Q8(view) => view.rows(),
            Self::ResidualQ8(view) => view.rows(),
        }
    }

    const fn blocks_per_row(self) -> usize {
        match self {
            Self::Q8(view) => view.blocks_per_row(),
            Self::ResidualQ8(view) => view.blocks_per_row(),
        }
    }
}

/// Fully validated row-major MXFP4 matrix multiplication problem.
pub struct Mxfp4MatmulProblem<'a> {
    weights: Mxfp4MatrixView<'a>,
    activations: Mxfp4ActivationMatrix<'a>,
    bias: Option<&'a [f32]>,
    output: &'a mut [f32],
    output_stride: usize,
}

impl<'a> Mxfp4MatmulProblem<'a> {
    pub fn new_q8(
        weights: Mxfp4MatrixView<'a>,
        activations: Q8MatrixView<'a>,
        bias: Option<&'a [f32]>,
        output: &'a mut [f32],
        output_stride: usize,
    ) -> Result<Self, KernelError> {
        Self::new(
            weights,
            Mxfp4ActivationMatrix::Q8(activations),
            bias,
            output,
            output_stride,
        )
    }

    pub fn new_residual_q8(
        weights: Mxfp4MatrixView<'a>,
        activations: ResidualQ8MatrixView<'a>,
        bias: Option<&'a [f32]>,
        output: &'a mut [f32],
        output_stride: usize,
    ) -> Result<Self, KernelError> {
        Self::new(
            weights,
            Mxfp4ActivationMatrix::ResidualQ8(activations),
            bias,
            output,
            output_stride,
        )
    }

    fn new(
        weights: Mxfp4MatrixView<'a>,
        activations: Mxfp4ActivationMatrix<'a>,
        bias: Option<&'a [f32]>,
        output: &'a mut [f32],
        output_stride: usize,
    ) -> Result<Self, KernelError> {
        let m = activations.rows();
        let n = weights.rows();
        let required = m
            .checked_sub(1)
            .and_then(|last| last.checked_mul(output_stride))
            .and_then(|offset| offset.checked_add(n));
        if activations.blocks_per_row() != weights.blocks()
            || output_stride < n
            || required.is_none_or(|required| required > output.len())
            || bias.is_some_and(|bias| bias.len() != n)
        {
            return Err(KernelError::InvalidDimensions(format!(
                "invalid MXFP4 matmul: M={m}, N={n}, K={}, output_stride={output_stride}",
                weights.blocks() * QUANT_BLOCK_SIZE
            )));
        }
        Ok(Self {
            weights,
            activations,
            bias,
            output,
            output_stride,
        })
    }

    pub const fn m(&self) -> usize {
        self.activations.rows()
    }

    pub const fn n(&self) -> usize {
        self.weights.rows()
    }

    pub const fn k(&self) -> usize {
        self.weights.blocks() * QUANT_BLOCK_SIZE
    }

    pub const fn output_stride(&self) -> usize {
        self.output_stride
    }

    pub const fn weights(&self) -> Mxfp4MatrixView<'a> {
        self.weights
    }

    pub const fn activations(&self) -> Mxfp4ActivationMatrix<'a> {
        self.activations
    }
}

impl Kernels {
    /// Execute one validated MXFP4 matrix problem without allocating.
    pub fn mxfp4_matmul(
        self,
        backend: Mxfp4MatmulBackend,
        mut problem: Mxfp4MatmulProblem<'_>,
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        let requirement = backend.scratch_requirement(&problem)?;
        validate_scratch(requirement, scratch)?;
        match backend.resolve(problem.m()) {
            Mxfp4MatmulBackend::Auto if problem.m() == 1 => self.gemv_matmul(&mut problem),
            Mxfp4MatmulBackend::Scalar => scalar_matmul(&mut problem),
            Mxfp4MatmulBackend::Avx2 => avx2_matmul(&mut problem, scratch),
            Mxfp4MatmulBackend::Avx512Vnni => {
                if !CpuFeatures::detect().supports(KernelRequirements::AVX512_VNNI_PATH) {
                    return Err(KernelError::UnavailableMatmulBackend {
                        backend,
                        reason: "AVX-512F/BW/VL/VNNI and OS vector state are required",
                    });
                }
                // Forced research candidate: retain the exact x8-v2 arithmetic
                // contract while the wider microkernel is evaluated.
                avx2_matmul(&mut problem, scratch)
            }
            Mxfp4MatmulBackend::AmxInt8 => amx_matmul(&mut problem, scratch),
            Mxfp4MatmulBackend::Auto => unreachable!("multi-row auto resolves to scalar"),
        }
    }

    /// Execute the portable AMX-INT8 tile emulator without capability or
    /// permission checks. This is a correctness oracle, not a serving path.
    #[cfg(feature = "amx-int8")]
    pub fn mxfp4_matmul_amx_int8_emulated(
        self,
        mut problem: Mxfp4MatmulProblem<'_>,
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        let requirement = Mxfp4MatmulBackend::AmxInt8.scratch_requirement(&problem)?;
        validate_scratch(requirement, scratch)?;
        emulated_amx_matmul(&mut problem, scratch)
    }

    fn gemv_matmul(self, problem: &mut Mxfp4MatmulProblem<'_>) -> Result<(), KernelError> {
        let Some(bias) = problem.bias else {
            return scalar_matmul(problem);
        };
        let output = &mut problem.output[..problem.weights.rows()];
        match problem.activations {
            Mxfp4ActivationMatrix::Q8(activations) => {
                let activations = Q8ActivationView::new(activations.row(0)?);
                for (tile, output) in output.chunks_mut(8).enumerate() {
                    self.mxfp4_q8_gemv_tile(problem.weights, tile * 8, activations, bias, output)?;
                }
            }
            Mxfp4ActivationMatrix::ResidualQ8(activations) => {
                let activations = ResidualQ8ActivationView::new(activations.row(0)?);
                for (tile, output) in output.chunks_mut(8).enumerate() {
                    self.mxfp4_residual_q8_gemv_tile(
                        problem.weights,
                        tile * 8,
                        activations,
                        bias,
                        output,
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(feature = "amx-int8"))]
fn amx_scratch_requirement(
    _problem: &Mxfp4MatmulProblem<'_>,
) -> Result<Mxfp4ScratchRequirement, KernelError> {
    Err(KernelError::UnavailableMatmulBackend {
        backend: Mxfp4MatmulBackend::AmxInt8,
        reason: "the amx-int8 Cargo feature is not enabled",
    })
}

#[cfg(feature = "amx-int8")]
fn amx_scratch_requirement(
    problem: &Mxfp4MatmulProblem<'_>,
) -> Result<Mxfp4ScratchRequirement, KernelError> {
    if problem.weights.layout() != Mxfp4WeightLayout::InterleavedSplitX8V2 {
        return Err(KernelError::InvalidDimensions(
            "AMX-INT8 MXFP4 matmul requires InterleavedSplitX8V2 weights".into(),
        ));
    }
    if problem.m() == 1 || problem.n() < AMX_TILE_OUTPUTS {
        return Ok(Mxfp4ScratchRequirement::NONE);
    }
    Ok(Mxfp4ScratchRequirement {
        size: AMX_SCRATCH_BYTES,
        alignment: 64,
    })
}

#[cfg(not(feature = "amx-int8"))]
fn amx_matmul(
    _problem: &mut Mxfp4MatmulProblem<'_>,
    _scratch: &mut [u8],
) -> Result<(), KernelError> {
    Err(KernelError::UnavailableMatmulBackend {
        backend: Mxfp4MatmulBackend::AmxInt8,
        reason: "the amx-int8 Cargo feature is not enabled",
    })
}

#[cfg(feature = "amx-int8")]
fn amx_matmul(problem: &mut Mxfp4MatmulProblem<'_>, scratch: &mut [u8]) -> Result<(), KernelError> {
    crate::initialize_amx_int8().map_err(|error| KernelError::UnavailableMatmulBackend {
        backend: Mxfp4MatmulBackend::AmxInt8,
        reason: error.reason(),
    })?;
    amx_matmul_with_tile(problem, scratch, crate::amx::execute_amx_int8_tile)
}

fn validate_scratch(
    requirement: Mxfp4ScratchRequirement,
    scratch: &[u8],
) -> Result<(), KernelError> {
    if scratch.len() < requirement.size
        || (requirement.size != 0
            && !(scratch.as_ptr() as usize).is_multiple_of(requirement.alignment))
    {
        return Err(KernelError::InvalidDimensions(format!(
            "MXFP4 scratch requires {} bytes aligned to {}, got {} bytes at alignment offset {}",
            requirement.size,
            requirement.alignment,
            scratch.len(),
            scratch.as_ptr() as usize % requirement.alignment
        )));
    }
    Ok(())
}

fn scalar_matmul(problem: &mut Mxfp4MatmulProblem<'_>) -> Result<(), KernelError> {
    scalar_matmul_range(problem, 0, problem.m(), 0, problem.n())
}

fn scalar_matmul_range(
    problem: &mut Mxfp4MatmulProblem<'_>,
    input_start: usize,
    input_rows: usize,
    output_start: usize,
    output_rows: usize,
) -> Result<(), KernelError> {
    for input_row in input_start..input_start + input_rows {
        for output_row in output_start..output_start + output_rows {
            let mut total = problem.bias.map_or(0.0, |bias| bias[output_row]);
            match problem.activations {
                Mxfp4ActivationMatrix::Q8(activations) => {
                    for (block_index, activation) in activations.row(input_row)?.iter().enumerate()
                    {
                        let weight = problem.weights.block(output_row, block_index)?;
                        let integer = weight
                            .unpack()
                            .into_iter()
                            .zip(activation.values)
                            .map(|(weight, activation)| weight as i32 * activation as i32)
                            .sum::<i32>();
                        total += integer as f32 * 0.5 * e8m0_scale(weight.scale) * activation.scale;
                    }
                }
                Mxfp4ActivationMatrix::ResidualQ8(activations) => {
                    for (block_index, activation) in activations.row(input_row)?.iter().enumerate()
                    {
                        let weight = problem.weights.block(output_row, block_index)?;
                        let unpacked = weight.unpack();
                        let primary = unpacked
                            .iter()
                            .zip(activation.primary.values)
                            .map(|(weight, activation)| *weight as i32 * activation as i32)
                            .sum::<i32>();
                        let residual = unpacked
                            .iter()
                            .zip(activation.residual.values)
                            .map(|(weight, activation)| *weight as i32 * activation as i32)
                            .sum::<i32>();
                        let weight_scale = 0.5 * e8m0_scale(weight.scale);
                        total += primary as f32 * weight_scale * activation.primary.scale;
                        total += residual as f32 * weight_scale * activation.residual.scale;
                    }
                }
            }
            problem.output[input_row * problem.output_stride + output_row] = total;
        }
    }
    Ok(())
}

#[cfg(feature = "amx-int8")]
fn emulated_amx_matmul(
    problem: &mut Mxfp4MatmulProblem<'_>,
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    amx_matmul_with_tile(problem, scratch, emulate_amx_tile)
}

#[cfg(feature = "amx-int8")]
fn amx_matmul_with_tile(
    problem: &mut Mxfp4MatmulProblem<'_>,
    scratch: &mut [u8],
    tile: impl Fn(usize, &[u8], &[u8], &mut [u8]) -> Result<(), KernelError>,
) -> Result<(), KernelError> {
    if problem.m() == 1 || problem.n() < AMX_TILE_OUTPUTS {
        return scalar_matmul(problem);
    }
    let complete_outputs = problem.n() / AMX_TILE_OUTPUTS * AMX_TILE_OUTPUTS;
    for input_row in 0..problem.m() {
        for output_row in 0..complete_outputs {
            problem.output[input_row * problem.output_stride + output_row] =
                problem.bias.map_or(0.0, |bias| bias[output_row]);
        }
    }

    let (a_panel, rest) = scratch.split_at_mut(AMX_A_PANEL_BYTES);
    let (b_panel, c_tile) = rest.split_at_mut(AMX_B_PANEL_BYTES);
    for output_start in (0..complete_outputs).step_by(AMX_TILE_OUTPUTS) {
        for block_index in 0..problem.weights.blocks() {
            let weight_scales =
                pack_amx_b_panel(problem.weights, output_start, block_index, b_panel)?;
            for input_start in (0..problem.m()).step_by(AMX_TILE_ROWS) {
                let input_rows = (problem.m() - input_start).min(AMX_TILE_ROWS);
                match problem.activations {
                    Mxfp4ActivationMatrix::Q8(activations) => {
                        let activation_scales = pack_amx_q8_a_panel(
                            activations,
                            input_start,
                            input_rows,
                            block_index,
                            a_panel,
                        )?;
                        tile(input_rows, a_panel, b_panel, c_tile)?;
                        accumulate_amx_tile(
                            problem,
                            input_start,
                            input_rows,
                            output_start,
                            &activation_scales,
                            &weight_scales,
                            c_tile,
                            false,
                        );
                    }
                    Mxfp4ActivationMatrix::ResidualQ8(activations) => {
                        for residual_pass in [false, true] {
                            let activation_scales = pack_amx_residual_a_panel(
                                activations,
                                input_start,
                                input_rows,
                                block_index,
                                residual_pass,
                                a_panel,
                            )?;
                            tile(input_rows, a_panel, b_panel, c_tile)?;
                            accumulate_amx_tile(
                                problem,
                                input_start,
                                input_rows,
                                output_start,
                                &activation_scales,
                                &weight_scales,
                                c_tile,
                                true,
                            );
                        }
                    }
                }
            }
        }
    }
    if complete_outputs != problem.n() {
        scalar_matmul_range(
            problem,
            0,
            problem.m(),
            complete_outputs,
            problem.n() - complete_outputs,
        )?;
    }
    Ok(())
}

#[cfg(feature = "amx-int8")]
fn pack_amx_b_panel(
    weights: Mxfp4MatrixView<'_>,
    output_start: usize,
    block_index: usize,
    panel: &mut [u8],
) -> Result<[f32; AMX_TILE_OUTPUTS], KernelError> {
    if panel.len() < AMX_B_PANEL_BYTES
        || output_start
            .checked_add(AMX_TILE_OUTPUTS)
            .is_none_or(|end| end > weights.rows())
        || block_index >= weights.blocks()
    {
        return Err(KernelError::InvalidDimensions(
            "invalid AMX-INT8 B-panel bounds".into(),
        ));
    }
    panel[..AMX_B_PANEL_BYTES].fill(0);
    let mut scales = [0.0; AMX_TILE_OUTPUTS];
    for (output, scale) in scales.iter_mut().enumerate() {
        let weight = weights.block(output_start + output, block_index)?;
        *scale = e8m0_scale(weight.scale);
        for (k, value) in weight.unpack().into_iter().enumerate() {
            let index = (k / 4) * 64 + output * 4 + k % 4;
            panel[index] = value.to_ne_bytes()[0];
        }
    }
    Ok(scales)
}

#[cfg(feature = "amx-int8")]
fn pack_amx_q8_a_panel(
    activations: Q8MatrixView<'_>,
    input_start: usize,
    input_rows: usize,
    block_index: usize,
    panel: &mut [u8],
) -> Result<[f32; AMX_TILE_ROWS], KernelError> {
    pack_amx_a_panel(input_rows, panel, |row| {
        activations
            .row(input_start + row)?
            .get(block_index)
            .ok_or_else(|| KernelError::InvalidDimensions("invalid AMX Q8 block".into()))
    })
}

#[cfg(feature = "amx-int8")]
fn pack_amx_residual_a_panel(
    activations: ResidualQ8MatrixView<'_>,
    input_start: usize,
    input_rows: usize,
    block_index: usize,
    residual_pass: bool,
    panel: &mut [u8],
) -> Result<[f32; AMX_TILE_ROWS], KernelError> {
    pack_amx_a_panel(input_rows, panel, |row| {
        let block = activations
            .row(input_start + row)?
            .get(block_index)
            .ok_or_else(|| {
                KernelError::InvalidDimensions("invalid AMX residual-Q8 block".into())
            })?;
        Ok(if residual_pass {
            &block.residual
        } else {
            &block.primary
        })
    })
}

#[cfg(feature = "amx-int8")]
fn pack_amx_a_panel<'a>(
    input_rows: usize,
    panel: &mut [u8],
    mut block: impl FnMut(usize) -> Result<&'a Q8Block, KernelError>,
) -> Result<[f32; AMX_TILE_ROWS], KernelError> {
    if input_rows == 0 || input_rows > AMX_TILE_ROWS || panel.len() < AMX_A_PANEL_BYTES {
        return Err(KernelError::InvalidDimensions(
            "invalid AMX-INT8 A-panel bounds".into(),
        ));
    }
    panel[..AMX_A_PANEL_BYTES].fill(0);
    let mut scales = [0.0; AMX_TILE_ROWS];
    for row in 0..input_rows {
        let block = block(row)?;
        scales[row] = block.scale;
        for (destination, value) in panel[row * QUANT_BLOCK_SIZE..(row + 1) * QUANT_BLOCK_SIZE]
            .iter_mut()
            .zip(block.values)
        {
            *destination = value.to_ne_bytes()[0];
        }
    }
    Ok(scales)
}

#[cfg(feature = "amx-int8")]
fn emulate_amx_tile(
    input_rows: usize,
    a_panel: &[u8],
    b_panel: &[u8],
    c_tile: &mut [u8],
) -> Result<(), KernelError> {
    if input_rows == 0
        || input_rows > AMX_TILE_ROWS
        || a_panel.len() < AMX_A_PANEL_BYTES
        || b_panel.len() < AMX_B_PANEL_BYTES
        || c_tile.len() < AMX_C_TILE_BYTES
    {
        return Err(KernelError::InvalidDimensions(
            "invalid AMX-INT8 tile-emulation bounds".into(),
        ));
    }
    c_tile[..AMX_C_TILE_BYTES].fill(0);
    for row in 0..input_rows {
        for output in 0..AMX_TILE_OUTPUTS {
            let mut integer = 0_i32;
            for k in 0..QUANT_BLOCK_SIZE {
                let activation = i8::from_ne_bytes([a_panel[row * QUANT_BLOCK_SIZE + k]]);
                let weight = i8::from_ne_bytes([b_panel[(k / 4) * 64 + output * 4 + k % 4]]);
                integer += activation as i32 * weight as i32;
            }
            let index = (row * AMX_TILE_OUTPUTS + output) * 4;
            c_tile[index..index + 4].copy_from_slice(&integer.to_ne_bytes());
        }
    }
    Ok(())
}

#[cfg(feature = "amx-int8")]
#[allow(clippy::too_many_arguments)]
fn accumulate_amx_tile(
    problem: &mut Mxfp4MatmulProblem<'_>,
    input_start: usize,
    input_rows: usize,
    output_start: usize,
    activation_scales: &[f32; AMX_TILE_ROWS],
    weight_scales: &[f32; AMX_TILE_OUTPUTS],
    c_tile: &[u8],
    residual_contract: bool,
) {
    for (row, &activation_scale) in activation_scales.iter().take(input_rows).enumerate() {
        for (output, &weight_scale) in weight_scales.iter().enumerate() {
            let index = (row * AMX_TILE_OUTPUTS + output) * 4;
            let integer = i32::from_ne_bytes(c_tile[index..index + 4].try_into().unwrap());
            let contribution = if residual_contract {
                integer as f32 * (0.5 * weight_scale) * activation_scale
            } else {
                integer as f32 * 0.5 * weight_scale * activation_scale
            };
            problem.output[(input_start + row) * problem.output_stride + output_start + output] +=
                contribution;
        }
    }
}

fn avx2_matmul(
    problem: &mut Mxfp4MatmulProblem<'_>,
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    if !CpuFeatures::detect().supports(KernelRequirements::AVX2_MXFP4) {
        return Err(KernelError::UnavailableMatmulBackend {
            backend: Mxfp4MatmulBackend::Avx2,
            reason: "AVX2 is unavailable on this host",
        });
    }
    if problem.weights.layout() != Mxfp4WeightLayout::InterleavedSplitX8V2 {
        return Err(KernelError::InvalidDimensions(
            "AVX2 MXFP4 matmul requires InterleavedSplitX8V2 weights".into(),
        ));
    }

    let complete_outputs = problem.n() / 8 * 8;
    for input_start in (0..problem.m()).step_by(AVX2_PANEL_ROWS) {
        let input_rows = (problem.m() - input_start).min(AVX2_PANEL_ROWS);
        pack_avx2_panel(problem.activations, input_start, input_rows, scratch)?;
        if complete_outputs != 0 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: feature and layout checks above plus the validated
            // problem and scratch contract establish every pointer bound.
            unsafe {
                crate::x86::mxfp4_matmul_x8_avx2(
                    problem.weights,
                    input_start,
                    input_rows,
                    problem.activations,
                    crate::x86::Avx2MatmulDestination {
                        bias: problem.bias,
                        output: problem.output,
                        output_stride: problem.output_stride,
                    },
                    scratch,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            return Err(KernelError::UnavailableMatmulBackend {
                backend: Mxfp4MatmulBackend::Avx2,
                reason: "the AVX2 matrix kernel requires x86-64",
            });
        }
        if complete_outputs != problem.n() {
            scalar_matmul_range(
                problem,
                input_start,
                input_rows,
                complete_outputs,
                problem.n() - complete_outputs,
            )?;
        }
    }
    Ok(())
}

fn pack_avx2_panel(
    activations: Mxfp4ActivationMatrix<'_>,
    input_start: usize,
    input_rows: usize,
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let passes = match activations {
        Mxfp4ActivationMatrix::Q8(_) => 1,
        Mxfp4ActivationMatrix::ResidualQ8(_) => 2,
    };
    let blocks = activations.blocks_per_row();
    let required = blocks * passes * AVX2_PANEL_PASS_BYTES;
    let panel = &mut scratch[..required];
    panel.fill(0);
    for block in 0..blocks {
        for row in 0..input_rows {
            match activations {
                Mxfp4ActivationMatrix::Q8(view) => {
                    pack_q8_panel_block(
                        panel,
                        passes,
                        block,
                        0,
                        row,
                        &view.row(input_start + row)?[block],
                    );
                }
                Mxfp4ActivationMatrix::ResidualQ8(view) => {
                    let activation = &view.row(input_start + row)?[block];
                    pack_q8_panel_block(panel, passes, block, 0, row, &activation.primary);
                    pack_q8_panel_block(panel, passes, block, 1, row, &activation.residual);
                }
            }
        }
    }
    Ok(())
}

fn pack_q8_panel_block(
    panel: &mut [u8],
    passes: usize,
    block: usize,
    pass: usize,
    row: usize,
    activation: &Q8Block,
) {
    let base = (block * passes + pass) * AVX2_PANEL_PASS_BYTES;
    let scale_start = base + row * std::mem::size_of::<f32>();
    panel[scale_start..scale_start + 4].copy_from_slice(&activation.scale.to_ne_bytes());
    let values_start = base + 16 + row * QUANT_BLOCK_SIZE;
    // SAFETY: i8 and u8 have identical layout and both fixed extents contain
    // exactly one quantization block.
    let values = unsafe {
        std::slice::from_raw_parts(activation.values.as_ptr().cast::<u8>(), QUANT_BLOCK_SIZE)
    };
    panel[values_start..values_start + QUANT_BLOCK_SIZE].copy_from_slice(values);
}

pub(super) const fn avx2_panel_pass_bytes() -> usize {
    AVX2_PANEL_PASS_BYTES
}

#[cfg(test)]
mod tests {
    use crate::{
        mxfp4_adjacent_to_split, KernelPath, Mxfp4Block, Mxfp4WeightLayout, MXFP4_PACKED_BYTES,
    };

    use super::*;

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
        data
    }

    fn fixtures(rows: usize, blocks: usize) -> (Vec<Mxfp4Block>, Vec<Q8Block>) {
        let weights = (0..rows * blocks)
            .map(|index| Mxfp4Block {
                scale: if index % 17 == 0 {
                    0
                } else {
                    126 + index as u8 % 4
                },
                packed: std::array::from_fn::<_, MXFP4_PACKED_BYTES, _>(|byte| {
                    ((index + byte) as u8 & 0x0f) | (((index + byte + 3) as u8 & 0x0f) << 4)
                }),
            })
            .collect();
        let activations = (0..16 * blocks)
            .map(|index| Q8Block {
                scale: 0.01 + index as f32 * 0.001,
                values: std::array::from_fn(|lane| ((index * 31 + lane) % 255) as i16 as i8),
            })
            .collect();
        (weights, activations)
    }

    #[test]
    fn backend_parse_display_and_auto_policy_are_explicit() {
        for (text, backend) in [
            ("auto", Mxfp4MatmulBackend::Auto),
            ("scalar", Mxfp4MatmulBackend::Scalar),
            ("avx2", Mxfp4MatmulBackend::Avx2),
            ("amx-int8", Mxfp4MatmulBackend::AmxInt8),
        ] {
            assert_eq!(text.parse::<Mxfp4MatmulBackend>().unwrap(), backend);
            assert_eq!(backend.to_string(), text);
        }
        assert!("neon".parse::<Mxfp4MatmulBackend>().is_err());
    }

    #[test]
    fn scalar_matrix_matches_individual_dots_for_shapes_and_tails() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        for m in [1, 2, 4, 5] {
            for n in [1, 7, 8, 13, 16] {
                let blocks = 3;
                let (weights, activations) = fixtures(n, blocks);
                let data = pack_x8(&weights, n, blocks);
                let weights =
                    Mxfp4MatrixView::new(&data, n, blocks, Mxfp4WeightLayout::InterleavedSplitX8V2)
                        .unwrap();
                let activations = Q8MatrixView::new(&activations, m, blocks, blocks).unwrap();
                let bias = (0..n).map(|index| index as f32 * 0.25).collect::<Vec<_>>();
                let stride = n + 3;
                let mut output = vec![1234.0; m * stride + 2];
                let problem = Mxfp4MatmulProblem::new_q8(
                    weights,
                    activations,
                    Some(&bias),
                    &mut output,
                    stride,
                )
                .unwrap();
                kernels
                    .mxfp4_matmul(Mxfp4MatmulBackend::Scalar, problem, &mut [])
                    .unwrap();
                for row in 0..m {
                    for output_row in 0..n {
                        let mut expected = bias[output_row];
                        for (block, activation) in activations.row(row).unwrap().iter().enumerate()
                        {
                            let weight = weights.block(output_row, block).unwrap();
                            let integer = kernels.mxfp4_q8_block_dot_i32(&weight, activation);
                            expected +=
                                integer as f32 * 0.5 * e8m0_scale(weight.scale) * activation.scale;
                        }
                        assert_eq!(output[row * stride + output_row], expected);
                    }
                    assert!(output[row * stride + n..(row + 1) * stride]
                        .iter()
                        .all(|value| *value == 1234.0));
                }
            }
        }
    }

    #[test]
    fn residual_scalar_preserves_primary_then_residual_order_without_bias() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let (weights, q8) = fixtures(9, 2);
        let residual = q8
            .into_iter()
            .map(|primary| ResidualQ8Block {
                residual: Q8Block {
                    scale: primary.scale / 17.0,
                    values: primary.values.map(|value| value / 3),
                },
                primary,
            })
            .collect::<Vec<_>>();
        let data = pack_x8(&weights, 9, 2);
        let weights =
            Mxfp4MatrixView::new(&data, 9, 2, Mxfp4WeightLayout::InterleavedSplitX8V2).unwrap();
        let activations = ResidualQ8MatrixView::new(&residual, 5, 2, 2).unwrap();
        let mut output = vec![0.0; 45];
        let problem =
            Mxfp4MatmulProblem::new_residual_q8(weights, activations, None, &mut output, 9)
                .unwrap();
        kernels
            .mxfp4_matmul(Mxfp4MatmulBackend::Auto, problem, &mut [])
            .unwrap();
        for row in 0..5 {
            for output_row in 0..9 {
                let blocks = (0..2)
                    .map(|block| weights.block(output_row, block).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    output[row * 9 + output_row],
                    kernels
                        .mxfp4_residual_q8_dot(&blocks, activations.row(row).unwrap())
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn views_and_output_strides_reject_invalid_bounds() {
        let (_, activations) = fixtures(8, 2);
        assert!(Q8MatrixView::new(&activations, 2, 2, 1).is_err());
        assert!(Q8MatrixView::new(&activations[..3], 2, 2, 2).is_err());
        assert!(Q8MatrixView::new(&activations, usize::MAX, 2, 2).is_err());

        let (weights, _) = fixtures(8, 2);
        let data = pack_x8(&weights, 8, 2);
        let weights =
            Mxfp4MatrixView::new(&data, 8, 2, Mxfp4WeightLayout::InterleavedSplitX8V2).unwrap();
        let view = Q8MatrixView::new(&activations, 2, 2, 2).unwrap();
        let mut short = vec![0.0; 15];
        assert!(Mxfp4MatmulProblem::new_q8(weights, view, None, &mut short, 8).is_err());
        let mut output = vec![0.0; 16];
        assert!(Mxfp4MatmulProblem::new_q8(weights, view, None, &mut output, 7).is_err());
        let mut output = vec![0.0; 16];
        assert!(
            Mxfp4MatmulProblem::new_q8(weights, view, Some(&[0.0; 7]), &mut output, 8,).is_err()
        );
    }

    fn aligned_offset(bytes: &[u8], alignment: usize) -> usize {
        (alignment - bytes.as_ptr() as usize % alignment) % alignment
    }

    #[test]
    fn avx2_matrix_matches_scalar_for_panels_and_all_tails() {
        if !CpuFeatures::detect().supports(KernelRequirements::AVX2_MXFP4) {
            return;
        }
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        for blocks in [1, 3] {
            for m in [1, 2, 3, 4, 5] {
                for n in [1, 7, 8, 13, 16] {
                    let (canonical, q8) = fixtures(n, blocks);
                    let data = pack_x8(&canonical, n, blocks);
                    let weights = Mxfp4MatrixView::new(
                        &data,
                        n,
                        blocks,
                        Mxfp4WeightLayout::InterleavedSplitX8V2,
                    )
                    .unwrap();
                    let q8 = Q8MatrixView::new(&q8, m, blocks, blocks).unwrap();
                    let bias_values = (0..n)
                        .map(|index| index as f32 * 0.03125 - 0.25)
                        .collect::<Vec<_>>();
                    let bias = (m + n + blocks)
                        .is_multiple_of(2)
                        .then_some(bias_values.as_slice());
                    let stride = n + 3;
                    let mut expected = vec![9876.0; m * stride];
                    let scalar =
                        Mxfp4MatmulProblem::new_q8(weights, q8, bias, &mut expected, stride)
                            .unwrap();
                    kernels
                        .mxfp4_matmul(Mxfp4MatmulBackend::Scalar, scalar, &mut [])
                        .unwrap();

                    let mut actual = vec![9876.0; m * stride];
                    let problem =
                        Mxfp4MatmulProblem::new_q8(weights, q8, bias, &mut actual, stride).unwrap();
                    let requirement = Mxfp4MatmulBackend::Avx2
                        .scratch_requirement(&problem)
                        .unwrap();
                    assert_eq!(requirement.size, blocks * AVX2_PANEL_PASS_BYTES);
                    assert_eq!(requirement.alignment, 32);
                    let mut storage = vec![0xa5; requirement.size + requirement.alignment * 2];
                    let offset = aligned_offset(&storage, requirement.alignment);
                    kernels
                        .mxfp4_matmul(
                            Mxfp4MatmulBackend::Avx2,
                            problem,
                            &mut storage[offset..offset + requirement.size],
                        )
                        .unwrap();
                    assert_eq!(actual, expected, "M={m}, N={n}, blocks={blocks}");
                    assert!(storage[..offset].iter().all(|byte| *byte == 0xa5));
                    assert!(storage[offset + requirement.size..]
                        .iter()
                        .all(|byte| *byte == 0xa5));
                }
            }
        }
    }

    #[test]
    fn avx2_residual_matrix_matches_scalar_and_reuses_exact_scratch() {
        if !CpuFeatures::detect().supports(KernelRequirements::AVX2_MXFP4) {
            return;
        }
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let blocks = 3;
        let (canonical, q8) = fixtures(13, blocks);
        let data = pack_x8(&canonical, 13, blocks);
        let weights =
            Mxfp4MatrixView::new(&data, 13, blocks, Mxfp4WeightLayout::InterleavedSplitX8V2)
                .unwrap();
        let residual = q8
            .into_iter()
            .map(|primary| ResidualQ8Block {
                residual: Q8Block {
                    scale: primary.scale / 97.0,
                    values: primary.values.map(|value| value.wrapping_mul(3)),
                },
                primary,
            })
            .collect::<Vec<_>>();
        let activations = ResidualQ8MatrixView::new(&residual, 5, blocks, blocks).unwrap();
        let bias = (0..13).map(|index| index as f32 / 32.0).collect::<Vec<_>>();
        let mut expected = vec![0.0; 5 * 15];
        let scalar = Mxfp4MatmulProblem::new_residual_q8(
            weights,
            activations,
            Some(&bias),
            &mut expected,
            15,
        )
        .unwrap();
        kernels
            .mxfp4_matmul(Mxfp4MatmulBackend::Scalar, scalar, &mut [])
            .unwrap();

        let mut actual = vec![0.0; 5 * 15];
        let problem =
            Mxfp4MatmulProblem::new_residual_q8(weights, activations, Some(&bias), &mut actual, 15)
                .unwrap();
        let requirement = Mxfp4MatmulBackend::Avx2
            .scratch_requirement(&problem)
            .unwrap();
        assert_eq!(requirement.size, blocks * 2 * AVX2_PANEL_PASS_BYTES);
        let mut storage = vec![0; requirement.size + requirement.alignment];
        let offset = aligned_offset(&storage, requirement.alignment);
        kernels
            .mxfp4_matmul(
                Mxfp4MatmulBackend::Avx2,
                problem,
                &mut storage[offset..offset + requirement.size],
            )
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_scratch_rejects_short_and_misaligned_views() {
        if !CpuFeatures::detect().supports(KernelRequirements::AVX2_MXFP4) {
            return;
        }
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let (canonical, q8) = fixtures(8, 2);
        let data = pack_x8(&canonical, 8, 2);
        let weights =
            Mxfp4MatrixView::new(&data, 8, 2, Mxfp4WeightLayout::InterleavedSplitX8V2).unwrap();
        let activations = Q8MatrixView::new(&q8, 2, 2, 2).unwrap();
        let requirement = Mxfp4ScratchRequirement {
            size: 2 * AVX2_PANEL_PASS_BYTES,
            alignment: 32,
        };
        let mut storage = vec![0; requirement.size + requirement.alignment + 1];
        let offset = aligned_offset(&storage, requirement.alignment);

        let mut output = vec![0.0; 16];
        let problem =
            Mxfp4MatmulProblem::new_q8(weights, activations, None, &mut output, 8).unwrap();
        assert!(kernels
            .mxfp4_matmul(
                Mxfp4MatmulBackend::Avx2,
                problem,
                &mut storage[offset..offset + requirement.size - 1],
            )
            .is_err());

        let mut output = vec![0.0; 16];
        let problem =
            Mxfp4MatmulProblem::new_q8(weights, activations, None, &mut output, 8).unwrap();
        assert!(kernels
            .mxfp4_matmul(
                Mxfp4MatmulBackend::Avx2,
                problem,
                &mut storage[offset + 1..offset + 1 + requirement.size],
            )
            .is_err());
    }

    #[cfg(feature = "amx-int8")]
    #[test]
    fn amx_panels_preserve_semantic_a_and_vnni_b_order() {
        let blocks = 2;
        let (canonical, q8) = fixtures(16, blocks);
        let data = pack_x8(&canonical, 16, blocks);
        let weights =
            Mxfp4MatrixView::new(&data, 16, blocks, Mxfp4WeightLayout::InterleavedSplitX8V2)
                .unwrap();
        let activations = Q8MatrixView::new(&q8, 16, blocks, blocks).unwrap();
        let mut a = [0xa5; AMX_A_PANEL_BYTES];
        let scales = pack_amx_q8_a_panel(activations, 0, 15, 1, &mut a).unwrap();
        for row in 0..15 {
            assert_eq!(scales[row], activations.row(row).unwrap()[1].scale);
            for k in 0..QUANT_BLOCK_SIZE {
                assert_eq!(
                    i8::from_ne_bytes([a[row * QUANT_BLOCK_SIZE + k]]),
                    activations.row(row).unwrap()[1].values[k]
                );
            }
        }
        assert!(a[15 * QUANT_BLOCK_SIZE..].iter().all(|byte| *byte == 0));

        let mut b = [0xa5; AMX_B_PANEL_BYTES];
        let scales = pack_amx_b_panel(weights, 0, 1, &mut b).unwrap();
        for output in 0..16 {
            let block = weights.block(output, 1).unwrap();
            assert_eq!(scales[output], e8m0_scale(block.scale));
            let unpacked = block.unpack();
            for k in 0..QUANT_BLOCK_SIZE {
                assert_eq!(
                    i8::from_ne_bytes([b[(k / 4) * 64 + output * 4 + k % 4]]),
                    unpacked[k]
                );
            }
        }
    }

    #[cfg(feature = "amx-int8")]
    #[test]
    fn amx_tile_emulator_matches_scalar_q8_for_tiles_blocks_and_tails() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        for m in [2, 4, 15, 16] {
            for n in [16, 19] {
                for blocks in [1, 3] {
                    let (canonical, q8) = fixtures(n, blocks);
                    let data = pack_x8(&canonical, n, blocks);
                    let weights = Mxfp4MatrixView::new(
                        &data,
                        n,
                        blocks,
                        Mxfp4WeightLayout::InterleavedSplitX8V2,
                    )
                    .unwrap();
                    let activations = Q8MatrixView::new(&q8, m, blocks, blocks).unwrap();
                    let bias = (0..n)
                        .map(|index| index as f32 * 0.03125 - 0.125)
                        .collect::<Vec<_>>();
                    let stride = n + 3;
                    let mut expected = vec![4567.0; m * stride];
                    let scalar = Mxfp4MatmulProblem::new_q8(
                        weights,
                        activations,
                        Some(&bias),
                        &mut expected,
                        stride,
                    )
                    .unwrap();
                    kernels
                        .mxfp4_matmul(Mxfp4MatmulBackend::Scalar, scalar, &mut [])
                        .unwrap();

                    let mut actual = vec![4567.0; m * stride];
                    let problem = Mxfp4MatmulProblem::new_q8(
                        weights,
                        activations,
                        Some(&bias),
                        &mut actual,
                        stride,
                    )
                    .unwrap();
                    let requirement = Mxfp4MatmulBackend::AmxInt8
                        .scratch_requirement(&problem)
                        .unwrap();
                    assert_eq!(requirement.size, AMX_SCRATCH_BYTES);
                    assert_eq!(requirement.alignment, 64);
                    let mut storage = vec![0xa5; requirement.size + 2 * requirement.alignment];
                    let offset = aligned_offset(&storage, requirement.alignment);
                    kernels
                        .mxfp4_matmul_amx_int8_emulated(
                            problem,
                            &mut storage[offset..offset + requirement.size],
                        )
                        .unwrap();
                    assert_eq!(actual, expected, "M={m}, N={n}, blocks={blocks}");
                    assert!(storage[..offset].iter().all(|byte| *byte == 0xa5));
                    assert!(storage[offset + requirement.size..]
                        .iter()
                        .all(|byte| *byte == 0xa5));
                }
            }
        }
    }

    #[cfg(feature = "amx-int8")]
    #[test]
    fn amx_tile_emulator_matches_scalar_residual_primary_then_residual() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let m = 15;
        let n = 21;
        let blocks = 3;
        let (canonical, q8) = fixtures(n, blocks);
        let data = pack_x8(&canonical, n, blocks);
        let weights =
            Mxfp4MatrixView::new(&data, n, blocks, Mxfp4WeightLayout::InterleavedSplitX8V2)
                .unwrap();
        let residual = q8
            .into_iter()
            .map(|primary| ResidualQ8Block {
                residual: Q8Block {
                    scale: primary.scale / 113.0,
                    values: primary.values.map(|value| value.wrapping_mul(-3)),
                },
                primary,
            })
            .collect::<Vec<_>>();
        let activations = ResidualQ8MatrixView::new(&residual, m, blocks, blocks).unwrap();
        let bias = (0..n).map(|index| index as f32 / 64.0).collect::<Vec<_>>();
        let mut expected = vec![0.0; m * n];
        let scalar = Mxfp4MatmulProblem::new_residual_q8(
            weights,
            activations,
            Some(&bias),
            &mut expected,
            n,
        )
        .unwrap();
        kernels
            .mxfp4_matmul(Mxfp4MatmulBackend::Scalar, scalar, &mut [])
            .unwrap();
        let mut actual = vec![0.0; m * n];
        let problem =
            Mxfp4MatmulProblem::new_residual_q8(weights, activations, Some(&bias), &mut actual, n)
                .unwrap();
        let requirement = Mxfp4MatmulBackend::AmxInt8
            .scratch_requirement(&problem)
            .unwrap();
        let mut storage = vec![0; requirement.size + requirement.alignment];
        let offset = aligned_offset(&storage, requirement.alignment);
        kernels
            .mxfp4_matmul_amx_int8_emulated(
                problem,
                &mut storage[offset..offset + requirement.size],
            )
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "amx-int8")]
    #[test]
    fn amx_scratch_fallback_bounds_alignment_and_integer_extrema() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let (canonical, q8) = fixtures(16, 1);
        let data = pack_x8(&canonical, 16, 1);
        let weights =
            Mxfp4MatrixView::new(&data, 16, 1, Mxfp4WeightLayout::InterleavedSplitX8V2).unwrap();

        let one = Q8MatrixView::new(&q8, 1, 1, 1).unwrap();
        let mut one_output = vec![0.0; 16];
        let one_problem =
            Mxfp4MatmulProblem::new_q8(weights, one, None, &mut one_output, 16).unwrap();
        assert_eq!(
            Mxfp4MatmulBackend::AmxInt8
                .scratch_requirement(&one_problem)
                .unwrap(),
            Mxfp4ScratchRequirement::NONE
        );
        kernels
            .mxfp4_matmul_amx_int8_emulated(one_problem, &mut [])
            .unwrap();

        let activations = Q8MatrixView::new(&q8, 2, 1, 1).unwrap();
        let mut storage = vec![0; AMX_SCRATCH_BYTES + 65];
        let offset = aligned_offset(&storage, 64);
        for (start, len) in [
            (offset, AMX_SCRATCH_BYTES - 1),
            (offset + 1, AMX_SCRATCH_BYTES),
        ] {
            let mut output = vec![0.0; 32];
            let problem =
                Mxfp4MatmulProblem::new_q8(weights, activations, None, &mut output, 16).unwrap();
            assert!(kernels
                .mxfp4_matmul_amx_int8_emulated(problem, &mut storage[start..start + len],)
                .is_err());
        }

        let a = [127_u8; AMX_A_PANEL_BYTES];
        let b = [12_u8; AMX_B_PANEL_BYTES];
        let mut c = [0_u8; AMX_C_TILE_BYTES];
        emulate_amx_tile(16, &a, &b, &mut c).unwrap();
        for value in c.chunks_exact(4) {
            assert_eq!(i32::from_ne_bytes(value.try_into().unwrap()), 48_768);
        }
    }

    #[cfg(feature = "amx-int8")]
    #[test]
    fn forced_amx_checks_runtime_before_scalar_fallback() {
        let kernels = Kernels::new(KernelPath::Scalar).unwrap();
        let (canonical, q8) = fixtures(16, 1);
        let data = pack_x8(&canonical, 16, 1);
        let weights =
            Mxfp4MatrixView::new(&data, 16, 1, Mxfp4WeightLayout::InterleavedSplitX8V2).unwrap();
        let activations = Q8MatrixView::new(&q8, 1, 1, 1).unwrap();
        let mut output = vec![0.0; 16];
        let problem =
            Mxfp4MatmulProblem::new_q8(weights, activations, None, &mut output, 16).unwrap();
        match crate::initialize_amx_int8() {
            Ok(_) => kernels
                .mxfp4_matmul(Mxfp4MatmulBackend::AmxInt8, problem, &mut [])
                .unwrap(),
            Err(expected) => {
                let error = kernels
                    .mxfp4_matmul(Mxfp4MatmulBackend::AmxInt8, problem, &mut [])
                    .unwrap_err();
                assert!(error.to_string().contains(expected.reason()));
            }
        }
    }
}
