use std::fmt;
use std::str::FromStr;

use crate::{
    e8m0_scale, KernelError, Kernels, Mxfp4MatrixView, Q8Block, ResidualQ8Block, QUANT_BLOCK_SIZE,
};

/// Explicit MXFP4 matrix implementation preference.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Mxfp4MatmulBackend {
    #[default]
    Auto,
    Scalar,
    Avx2,
    AmxInt8,
}

impl Mxfp4MatmulBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
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
            Self::Avx2 => {
                let passes = match problem.activations {
                    Mxfp4ActivationMatrix::Q8(_) => 1,
                    Mxfp4ActivationMatrix::ResidualQ8(_) => 2,
                };
                let bytes = problem
                    .weights
                    .blocks()
                    .checked_mul(passes)
                    .and_then(|value| value.checked_mul(144))
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
            Self::AmxInt8 => Err(KernelError::UnavailableMatmulBackend {
                backend: Self::AmxInt8,
                reason: "the amx-int8 Cargo feature is not enabled",
            }),
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
            Mxfp4MatmulBackend::Auto if problem.m() == 1 => scalar_matmul(&mut problem),
            Mxfp4MatmulBackend::Scalar => scalar_matmul(&mut problem),
            Mxfp4MatmulBackend::Avx2 => Err(KernelError::UnavailableMatmulBackend {
                backend: Mxfp4MatmulBackend::Avx2,
                reason: "the AVX2 matrix kernel is not implemented",
            }),
            Mxfp4MatmulBackend::AmxInt8 => Err(KernelError::UnavailableMatmulBackend {
                backend: Mxfp4MatmulBackend::AmxInt8,
                reason: "the amx-int8 Cargo feature is not enabled",
            }),
            Mxfp4MatmulBackend::Auto => unreachable!("multi-row auto resolves to scalar"),
        }
    }
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
    for input_row in 0..problem.m() {
        for output_row in 0..problem.n() {
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
        let activations = (0..5 * blocks)
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
}
