use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use gpt_oss_cpu_kernels::{
    mxfp4_adjacent_to_split, KernelPath, Kernels, Mxfp4MatmulBackend, Mxfp4MatmulProblem,
    Mxfp4MatrixView, Mxfp4WeightLayout, ResidualQ8Block, ResidualQ8MatrixView,
};
use half::bf16;
use memmap2::MmapOptions;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

pub const N: usize = 5_760;
pub const K: usize = 2_880;
pub const BLOCKS: usize = K / 32;
pub const EXPERT: usize = 0;
pub const EXPERTS: usize = 32;
pub const XE_TILE: usize = 32;
pub const XE_WEIGHT_PLANES: usize = 17;
pub const XE_ACTIVATION_RECORD_BYTES: usize = 72;

const BLOCKS_NAME: &str = "model.layers.0.mlp.experts.gate_up_proj_blocks";
const SCALES_NAME: &str = "model.layers.0.mlp.experts.gate_up_proj_scales";
const BIAS_NAME: &str = "model.layers.0.mlp.experts.gate_up_proj_bias";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub selected_expert: usize,
    pub selected_bytes: usize,
    pub shard: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionDescriptor {
    pub layer: usize,
    pub expert: usize,
    pub m_values: Vec<usize>,
    pub n: usize,
    pub k: usize,
    pub blocks_per_output: usize,
    pub blocks: TensorDescriptor,
    pub scales: TensorDescriptor,
    pub bias: TensorDescriptor,
    pub canonical_compact_bytes: usize,
    pub cpu_x8_bytes: usize,
    pub xe_v2_bytes: usize,
}

pub struct ProjectionBundle {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub bias: Vec<f32>,
    pub canonical_records: Vec<u8>,
    pub cpu_x8_records: Vec<u8>,
    pub xe_v2_records: Vec<u8>,
    pub descriptor: ProjectionDescriptor,
}

impl ProjectionBundle {
    pub fn open(snapshot: &Path) -> Result<Self> {
        let shard = locate_layer_zero_shard(snapshot)?;
        let file = File::open(&shard)
            .with_context(|| format!("open checkpoint shard {}", shard.display()))?;
        // SAFETY: this is a read-only mapping and the checkpoint is treated as immutable.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap checkpoint shard {}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mapping).context("parse SafeTensors shard")?;

        let blocks_tensor = tensors
            .tensor(BLOCKS_NAME)
            .context("load layer-0 gate_up blocks")?;
        let scales_tensor = tensors
            .tensor(SCALES_NAME)
            .context("load layer-0 gate_up scales")?;
        let bias_tensor = tensors
            .tensor(BIAS_NAME)
            .context("load layer-0 gate_up bias")?;
        validate_tensor(
            &blocks_tensor,
            Dtype::U8,
            &[EXPERTS, N, BLOCKS, 16],
            BLOCKS_NAME,
        )?;
        validate_tensor(
            &scales_tensor,
            Dtype::U8,
            &[EXPERTS, N, BLOCKS],
            SCALES_NAME,
        )?;
        if bias_tensor.shape() != [EXPERTS, N] {
            bail!("{BIAS_NAME} has unexpected shape {:?}", bias_tensor.shape());
        }
        if !matches!(bias_tensor.dtype(), Dtype::BF16 | Dtype::F16 | Dtype::F32) {
            bail!(
                "{BIAS_NAME} has unsupported dtype {:?}",
                bias_tensor.dtype()
            );
        }

        let block_expert_bytes = N * BLOCKS * 16;
        let scale_expert_bytes = N * BLOCKS;
        let bias_element_bytes = bias_tensor.dtype().size();
        let bias_expert_bytes = N * bias_element_bytes;
        let block_start = EXPERT * block_expert_bytes;
        let scale_start = EXPERT * scale_expert_bytes;
        let bias_start = EXPERT * bias_expert_bytes;
        let packed = blocks_tensor.data()[block_start..block_start + block_expert_bytes].to_vec();
        let scales = scales_tensor.data()[scale_start..scale_start + scale_expert_bytes].to_vec();
        let bias_bytes = &bias_tensor.data()[bias_start..bias_start + bias_expert_bytes];
        let bias_values = decode_floats(bias_tensor.dtype(), bias_bytes)?;
        if bias_values.len() != N {
            bail!(
                "decoded layer-0 expert-0 bias has {} values",
                bias_values.len()
            );
        }

        let canonical_records = canonical_records(&packed, &scales);
        let cpu_x8_records = x8_records(&packed, &scales);
        let xe_v2_records = xe_v2_records(&packed, &scales);
        let descriptor = ProjectionDescriptor {
            layer: 0,
            expert: EXPERT,
            m_values: vec![1, 2, 4, 8, 16, 32, 64, 128],
            n: N,
            k: K,
            blocks_per_output: BLOCKS,
            blocks: tensor_descriptor(BLOCKS_NAME, &blocks_tensor, block_expert_bytes, &shard),
            scales: tensor_descriptor(SCALES_NAME, &scales_tensor, scale_expert_bytes, &shard),
            bias: tensor_descriptor(BIAS_NAME, &bias_tensor, bias_expert_bytes, &shard),
            canonical_compact_bytes: packed.len() + scales.len() + bias_values.len() * 4,
            cpu_x8_bytes: cpu_x8_records.len() + bias_values.len() * 4,
            xe_v2_bytes: xe_v2_records.len() + bias_values.len() * 4,
        };
        Ok(Self {
            packed,
            scales,
            bias: bias_values,
            canonical_records,
            cpu_x8_records,
            xe_v2_records,
            descriptor,
        })
    }

    pub fn cpu_projection_into(
        &self,
        backend: Mxfp4MatmulBackend,
        activations: &[ResidualQ8Block],
        rows: usize,
        output: &mut [f32],
        scratch: &mut [u8],
    ) -> Result<()> {
        let (records, layout) = self.records_for_backend(backend)?;
        if output.len() != rows * N {
            bail!(
                "CPU projection output has {} values, expected {}",
                output.len(),
                rows * N
            );
        }
        let weights = Mxfp4MatrixView::new(records, N, BLOCKS, layout)?;
        let activation_view = ResidualQ8MatrixView::new(activations, rows, BLOCKS, BLOCKS)?;
        let problem = Mxfp4MatmulProblem::new_residual_q8(
            weights,
            activation_view,
            Some(&self.bias),
            output,
            N,
        )?;
        Kernels::new(match backend {
            Mxfp4MatmulBackend::Scalar => KernelPath::Scalar,
            Mxfp4MatmulBackend::Avx2 => KernelPath::Avx2,
            other => bail!("research projection does not support CPU backend {other}"),
        })?
        .mxfp4_matmul(backend, problem, scratch)?;
        Ok(())
    }

    fn records_for_backend(
        &self,
        backend: Mxfp4MatmulBackend,
    ) -> Result<(&[u8], Mxfp4WeightLayout)> {
        match backend {
            Mxfp4MatmulBackend::Scalar => Ok((
                self.canonical_records.as_slice(),
                Mxfp4WeightLayout::CanonicalAdjacentV1,
            )),
            Mxfp4MatmulBackend::Avx2 => Ok((
                self.cpu_x8_records.as_slice(),
                Mxfp4WeightLayout::InterleavedSplitX8V2,
            )),
            other => bail!("research projection does not support CPU backend {other}"),
        }
    }
}

pub fn deterministic_activations(rows: usize, seed: u64) -> Vec<f32> {
    let mut random = ChaCha8Rng::seed_from_u64(seed ^ rows as u64);
    (0..rows * K)
        .map(|index| {
            let structured = ((index % 97) as f32 - 48.0) / 31.0;
            let noise = random.gen_range(-0.125_f32..=0.125_f32);
            bf16::from_f32(structured + noise).to_f32()
        })
        .collect()
}

pub fn quantize_residual_rows(values: &[f32]) -> Result<Vec<ResidualQ8Block>> {
    Kernels::new(KernelPath::Avx2)
        .context("AVX2 is required for the declared CPU baseline")?
        .quantize_residual_q8(values)
        .context("quantize deterministic residual-Q8 activations")
}

pub fn split_residual_activations(
    blocks: &[ResidualQ8Block],
) -> (Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>) {
    let mut primary = Vec::with_capacity(blocks.len() * 32);
    let mut residual = Vec::with_capacity(blocks.len() * 32);
    let mut primary_scales = Vec::with_capacity(blocks.len());
    let mut residual_scales = Vec::with_capacity(blocks.len());
    for block in blocks {
        primary.extend_from_slice(&block.primary.values);
        residual.extend_from_slice(&block.residual.values);
        primary_scales.push(block.primary.scale);
        residual_scales.push(block.residual.scale);
    }
    (primary, residual, primary_scales, residual_scales)
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct XeActivationRecordV2 {
    pub primary: [i8; 32],
    pub residual: [i8; 32],
    pub primary_scale: f32,
    pub residual_scale: f32,
}

pub fn pack_activation_records(blocks: &[ResidualQ8Block]) -> Vec<XeActivationRecordV2> {
    assert_eq!(
        std::mem::size_of::<XeActivationRecordV2>(),
        XE_ACTIVATION_RECORD_BYTES
    );
    blocks
        .iter()
        .map(|block| XeActivationRecordV2 {
            primary: block.primary.values,
            residual: block.residual.values,
            primary_scale: block.primary.scale,
            residual_scale: block.residual.scale,
        })
        .collect()
}

fn locate_layer_zero_shard(snapshot: &Path) -> Result<PathBuf> {
    let expected = snapshot.join("model-00000-of-00002.safetensors");
    if expected.is_file() {
        return Ok(expected);
    }
    let mut shards = std::fs::read_dir(snapshot)
        .with_context(|| format!("read snapshot {}", snapshot.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    shards.sort();
    for shard in shards {
        let file = File::open(&shard)?;
        // SAFETY: temporary read-only mapping is dropped after inspection.
        let mapping = unsafe { MmapOptions::new().map(&file) }?;
        if SafeTensors::deserialize(&mapping)
            .is_ok_and(|tensors| tensors.tensor(BLOCKS_NAME).is_ok())
        {
            return Ok(shard);
        }
    }
    bail!(
        "could not locate {BLOCKS_NAME} under {}",
        snapshot.display()
    )
}

fn validate_tensor(
    tensor: &safetensors::tensor::TensorView<'_>,
    dtype: Dtype,
    shape: &[usize],
    name: &str,
) -> Result<()> {
    if tensor.dtype() != dtype || tensor.shape() != shape {
        bail!(
            "{name} expected {dtype:?} {shape:?}, got {:?} {:?}",
            tensor.dtype(),
            tensor.shape()
        );
    }
    Ok(())
}

fn tensor_descriptor(
    name: &str,
    tensor: &safetensors::tensor::TensorView<'_>,
    selected_bytes: usize,
    shard: &Path,
) -> TensorDescriptor {
    TensorDescriptor {
        name: name.to_string(),
        dtype: format!("{:?}", tensor.dtype()),
        shape: tensor.shape().to_vec(),
        selected_expert: EXPERT,
        selected_bytes,
        shard: shard.to_path_buf(),
    }
}

fn decode_floats(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|value| bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
            .collect()),
        Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|value| half::f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
            .collect()),
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte chunk")))
            .collect()),
        other => bail!("unsupported floating tensor dtype {other:?}"),
    }
}

fn canonical_records(packed: &[u8], scales: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(scales.len() * 17);
    for (record, scale) in packed.chunks_exact(16).zip(scales) {
        output.push(*scale);
        output.extend_from_slice(record);
    }
    output
}

fn x8_records(packed: &[u8], scales: &[u8]) -> Vec<u8> {
    let source_record = |row: usize, block: usize| row * BLOCKS + block;
    let mut output = Vec::with_capacity(scales.len() * 17);
    for group in 0..N / 8 {
        for block in 0..BLOCKS {
            for lane in 0..8 {
                output.push(scales[source_record(group * 8 + lane, block)]);
            }
            let split = std::array::from_fn::<_, 8, _>(|lane| {
                let record = source_record(group * 8 + lane, block);
                mxfp4_adjacent_to_split(
                    packed[record * 16..record * 16 + 16]
                        .try_into()
                        .expect("validated record"),
                )
            });
            for chunk in 0..2 {
                for row in &split {
                    output.extend_from_slice(&row[chunk * 8..chunk * 8 + 8]);
                }
            }
        }
    }
    output
}

/// Repack canonical `[output][K-block][scale + 16 packed bytes]` records into
/// the immutable Xe ABI v2 layout `[output-tile][K-block][17 planes][32 lanes]`.
pub fn xe_v2_records(packed: &[u8], scales: &[u8]) -> Vec<u8> {
    xe_v2_records_for_dimensions(packed, scales, N, BLOCKS)
}

pub fn xe_v2_records_for_dimensions(
    packed: &[u8],
    scales: &[u8],
    columns: usize,
    blocks: usize,
) -> Vec<u8> {
    assert_eq!(packed.len(), columns * blocks * 16);
    assert_eq!(scales.len(), columns * blocks);
    assert_eq!(columns % XE_TILE, 0);
    let mut output = vec![0_u8; scales.len() * XE_WEIGHT_PLANES];
    for tile in 0..columns / XE_TILE {
        for block in 0..blocks {
            let destination = (tile * blocks + block) * XE_WEIGHT_PLANES * XE_TILE;
            for lane in 0..XE_TILE {
                let source = (tile * XE_TILE + lane) * blocks + block;
                output[destination + lane] = scales[source];
                for byte in 0..16 {
                    output[destination + (byte + 1) * XE_TILE + lane] = packed[source * 16 + byte];
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_cpu_kernels::{Mxfp4Block, Q8Block};

    #[test]
    fn layout_conversion_preserves_every_block() {
        let mut packed = vec![0_u8; N * BLOCKS * 16];
        let mut scales = vec![0_u8; N * BLOCKS];
        for (index, byte) in packed.iter_mut().enumerate() {
            *byte = (index.wrapping_mul(37)) as u8;
        }
        for (index, scale) in scales.iter_mut().enumerate() {
            *scale = (index.wrapping_mul(13)) as u8;
        }
        let records = x8_records(&packed, &scales);
        let view =
            Mxfp4MatrixView::new(&records, N, BLOCKS, Mxfp4WeightLayout::InterleavedSplitX8V2)
                .unwrap();
        for &(row, block) in &[(0, 0), (7, 89), (8, 2), (5759, 42)] {
            assert_eq!(
                view.block(row, block).unwrap(),
                Mxfp4Block {
                    scale: scales[row * BLOCKS + block],
                    packed: packed[(row * BLOCKS + block) * 16..(row * BLOCKS + block + 1) * 16]
                        .try_into()
                        .unwrap(),
                }
            );
        }
    }

    #[test]
    fn deterministic_activation_contract_is_block_aligned() {
        let values = deterministic_activations(2, 0x5845);
        assert_eq!(values.len(), 2 * K);
        let blocks = quantize_residual_rows(&values).unwrap();
        assert_eq!(blocks.len(), 2 * BLOCKS);
        assert!(blocks.iter().all(|block| block.primary.scale.is_finite()));
        let _type_contract: &Q8Block = &blocks[0].primary;
        let records = pack_activation_records(&blocks);
        assert_eq!(records.len(), 2 * BLOCKS);
        assert_eq!(std::mem::size_of_val(records.as_slice()), 2 * BLOCKS * 72);
        assert_eq!(records[0].primary, blocks[0].primary.values);
        assert_eq!(records[0].residual, blocks[0].residual.values);
        assert_eq!(
            records[0].primary_scale.to_bits(),
            blocks[0].primary.scale.to_bits()
        );
        assert_eq!(
            records[0].residual_scale.to_bits(),
            blocks[0].residual.scale.to_bits()
        );
    }

    #[test]
    fn xe_v2_layout_round_trips_every_plane() {
        let mut packed = vec![0_u8; N * BLOCKS * 16];
        let mut scales = vec![0_u8; N * BLOCKS];
        for (index, byte) in packed.iter_mut().enumerate() {
            *byte = index.wrapping_mul(29) as u8;
        }
        for (index, scale) in scales.iter_mut().enumerate() {
            *scale = index.wrapping_mul(11) as u8;
        }
        let records = xe_v2_records(&packed, &scales);
        assert_eq!(records.len(), packed.len() + scales.len());
        for row in 0..N {
            for block in 0..BLOCKS {
                let tile = row / XE_TILE;
                let lane = row % XE_TILE;
                let base = (tile * BLOCKS + block) * XE_WEIGHT_PLANES * XE_TILE;
                let source = row * BLOCKS + block;
                assert_eq!(records[base + lane], scales[source]);
                for byte in 0..16 {
                    assert_eq!(
                        records[base + (byte + 1) * XE_TILE + lane],
                        packed[source * 16 + byte]
                    );
                }
            }
        }
    }
}
