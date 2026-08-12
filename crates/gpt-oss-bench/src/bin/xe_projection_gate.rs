//! Transfer-inclusive real-checkpoint CPU/Xe projection promotion gate.

use std::fs::File;
use std::hint::black_box;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_cpu_kernels::{
    mxfp4_adjacent_to_split, KernelPath, Kernels, Mxfp4MatmulBackend, Mxfp4MatmulProblem,
    Mxfp4MatrixView, Mxfp4WeightLayout, ResidualQ8MatrixView,
};
use gpt_oss_xe::{
    ActivationRecordV2, AttachConfig, AttachmentMode, ProjectionRequest, ProjectionRole,
    XeProjectionEngine,
};
use half::{bf16, f16};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

const EXPERTS: usize = 32;
const EXPERT: usize = 0;
const TRIALS: usize = 3;
const WARMUPS: usize = 10;
const SAMPLES: usize = 30;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    repack_cache: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_delimiter = ',', default_values_t = [4, 8, 16, 32, 64, 128])]
    rows: Vec<usize>,
    #[arg(long, default_value_t = 128)]
    xe_max_resident_mib: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Method {
    Scalar,
    CpuAuto,
    Avx2,
    Xe,
}

impl Method {
    const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::CpuAuto => "cpu_auto",
            Self::Avx2 => "avx2",
            Self::Xe => "xe",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Role {
    GateUp,
    Down,
}

impl Role {
    const fn name(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::Down => "down",
        }
    }

    const fn xe(self) -> ProjectionRole {
        match self {
            Self::GateUp => ProjectionRole::GateUp,
            Self::Down => ProjectionRole::Down,
        }
    }

    fn tensor_name(self, suffix: &str) -> String {
        format!(
            "model.layers.0.mlp.experts.{}_{}",
            match self {
                Self::GateUp => "gate_up_proj",
                Self::Down => "down_proj",
            },
            suffix
        )
    }
}

struct Projection {
    role: Role,
    columns: usize,
    blocks: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    bias: Vec<f32>,
    canonical: Vec<u8>,
    x8: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct TensorHeader {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Debug, Serialize)]
struct Sample {
    trial: usize,
    method: &'static str,
    sample: usize,
    total_ns: u64,
}

struct ProjectionOutput {
    raw: Vec<f32>,
    bf16_boundary: Vec<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.rows.is_empty() || cli.rows.iter().any(|rows| *rows < 4 || rows % 4 != 0) {
        bail!("--rows must contain positive multiples of four at or above four");
    }
    let projections = [
        Projection::open(&cli.model, Role::GateUp)?,
        Projection::open(&cli.model, Role::Down)?,
    ];
    let max_columns = projections.iter().map(|value| value.columns).max().unwrap();
    let max_blocks = projections.iter().map(|value| value.blocks).max().unwrap();
    let xe = XeProjectionEngine::attach(AttachConfig::new(
        AttachmentMode::Explicit,
        &cli.repack_cache,
        cli.xe_max_resident_mib
            .checked_mul(1024 * 1024)
            .context("Xe resident cap overflows bytes")?,
        max_columns,
        max_blocks,
    ))?;
    let xe_descriptor = xe.descriptor()?;
    let orders = [
        [Method::Scalar, Method::CpuAuto, Method::Avx2, Method::Xe],
        [Method::CpuAuto, Method::Avx2, Method::Xe, Method::Scalar],
        [Method::Xe, Method::Scalar, Method::CpuAuto, Method::Avx2],
    ];
    let mut cpu_scratch = vec![0_u8; (1 << 20) + 4096];
    let mut reports = Vec::new();
    for projection in &projections {
        for &rows in &cli.rows {
            let inputs = deterministic_activations(rows, projection.blocks * 32, projection.role);
            let expected = run_projection(
                projection,
                rows,
                &inputs,
                Method::Scalar,
                &xe,
                &mut cpu_scratch,
            )?
            .0;
            for method in [Method::CpuAuto, Method::Avx2, Method::Xe] {
                let actual =
                    run_projection(projection, rows, &inputs, method, &xe, &mut cpu_scratch)?.0;
                compare_outputs(&expected, &actual).with_context(|| {
                    format!(
                        "{} M={rows} {} correctness",
                        projection.role.name(),
                        method.name()
                    )
                })?;
            }
            let mut samples = Vec::with_capacity(TRIALS * 4 * SAMPLES);
            for (trial, order) in orders.into_iter().enumerate() {
                for method in order {
                    for _ in 0..WARMUPS {
                        black_box(
                            run_projection(
                                projection,
                                rows,
                                &inputs,
                                method,
                                &xe,
                                &mut cpu_scratch,
                            )?
                            .0,
                        );
                    }
                    for sample in 0..SAMPLES {
                        let (_, elapsed) = run_projection(
                            projection,
                            rows,
                            &inputs,
                            method,
                            &xe,
                            &mut cpu_scratch,
                        )?;
                        samples.push(Sample {
                            trial,
                            method: method.name(),
                            sample,
                            total_ns: elapsed,
                        });
                    }
                }
            }
            reports.push(json!({
                "projection": projection.role.name(),
                "rows": rows,
                "columns": projection.columns,
                "blocks": projection.blocks,
                "correctness": {
                    "scalar_cpu_auto_avx2_xe_bf16_identical": true,
                    "xe_float_bound": "four ULP or 1e-6",
                },
                "samples": samples,
            }));
        }
    }
    xe.shutdown()?;
    let result = json!({
        "schema": "gpt-oss-rs.xe-transfer-inclusive-projection/v1",
        "status": "pass",
        "source": source_provenance(),
        "model": cli.model,
        "selected_layer": 0,
        "selected_expert": EXPERT,
        "xe": xe_descriptor,
        "trial_count": TRIALS,
        "warmups_per_method_per_trial": WARMUPS,
        "samples_per_method_per_trial": SAMPLES,
        "methods": ["scalar", "cpu_auto", "avx2", "xe"],
        "timed_scope": {
            "all": ["residual-Q8 activation preparation", "projection", "BF16 conversion"],
            "xe_additional": ["weight repack", "weight and bias staging", "activation record packing and staging", "argument setup", "submission", "terminal wait", "readback"],
            "excluded": ["model mapping", "OpenCL context/program creation", "persistent slab allocation", "reusable CPU scratch allocation"],
        },
        "reports": reports,
    });
    let encoded = serde_json::to_vec_pretty(&result)?;
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    gpt_oss_evidence::atomic_write(&cli.output, &encoded)?;
    println!("{}", cli.output.display());
    Ok(())
}

fn source_provenance() -> serde_json::Value {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_string())
            .unwrap_or_default()
    };
    let dirty = !git(&["status", "--porcelain"]).is_empty();
    let binary_sha256 = std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::read(path).ok())
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .unwrap_or_default();
    json!({
        "repository_commit": git(&["rev-parse", "HEAD"]),
        "dirty": dirty,
        "binary_sha256": binary_sha256,
        "cargo_lock_sha256": std::fs::read("Cargo.lock")
            .ok()
            .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
            .unwrap_or_default(),
        "toolchain": std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_string())
            .unwrap_or_default(),
        "profile": "release",
        "features": ["xe"],
    })
}

fn run_projection(
    projection: &Projection,
    rows: usize,
    inputs: &[f32],
    method: Method,
    xe: &XeProjectionEngine,
    cpu_scratch: &mut [u8],
) -> Result<(ProjectionOutput, u64)> {
    let started = Instant::now();
    let kernels = Kernels::new(match method {
        Method::Scalar => KernelPath::Scalar,
        _ => KernelPath::Auto,
    })?;
    let activations = kernels.quantize_residual_q8(inputs)?;
    let output = match method {
        Method::Scalar | Method::CpuAuto | Method::Avx2 => {
            let (records, layout, backend) = match method {
                Method::Scalar => (
                    projection.canonical.as_slice(),
                    Mxfp4WeightLayout::CanonicalAdjacentV1,
                    Mxfp4MatmulBackend::Scalar,
                ),
                Method::CpuAuto => (
                    projection.x8.as_slice(),
                    Mxfp4WeightLayout::InterleavedSplitX8V2,
                    Mxfp4MatmulBackend::Auto,
                ),
                Method::Avx2 => (
                    projection.x8.as_slice(),
                    Mxfp4WeightLayout::InterleavedSplitX8V2,
                    Mxfp4MatmulBackend::Avx2,
                ),
                Method::Xe => unreachable!(),
            };
            let weights =
                Mxfp4MatrixView::new(records, projection.columns, projection.blocks, layout)?;
            let activation_view = ResidualQ8MatrixView::new(
                &activations,
                rows,
                projection.blocks,
                projection.blocks,
            )?;
            let mut output = vec![0.0; rows * projection.columns];
            let problem = Mxfp4MatmulProblem::new_residual_q8(
                weights,
                activation_view,
                Some(&projection.bias),
                &mut output,
                projection.columns,
            )?;
            let requirement = backend.scratch_requirement(&problem)?;
            let offset = cpu_scratch.as_ptr().align_offset(requirement.alignment);
            if offset == usize::MAX || offset + requirement.size > cpu_scratch.len() {
                bail!("reusable CPU projection scratch is undersized");
            }
            kernels.mxfp4_matmul(
                backend,
                problem,
                &mut cpu_scratch[offset..offset + requirement.size],
            )?;
            output
        }
        Method::Xe => {
            let weights_v2 =
                gpt_oss_xe::repack_v2(projection.columns, projection.blocks, |column, block| {
                    let source = column * projection.blocks + block;
                    Ok((
                        projection.scales[source],
                        projection.packed[source * 16..source * 16 + 16]
                            .try_into()
                            .expect("validated MXFP4 record"),
                    ))
                })?;
            let activation_records = activations
                .iter()
                .map(|block| ActivationRecordV2 {
                    primary: block.primary.values,
                    residual: block.residual.values,
                    primary_scale: block.primary.scale,
                    residual_scale: block.residual.scale,
                })
                .collect::<Vec<_>>();
            xe.project(ProjectionRequest {
                role: projection.role.xe(),
                rows,
                columns: projection.columns,
                blocks: projection.blocks,
                weights_v2: &weights_v2,
                activations_v2: &activation_records,
                bias: &projection.bias,
            })?
        }
    };
    let bf16_boundary = output
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    Ok((
        ProjectionOutput {
            raw: output,
            bf16_boundary,
        },
        started.elapsed().as_nanos() as u64,
    ))
}

fn compare_outputs(expected: &ProjectionOutput, actual: &ProjectionOutput) -> Result<()> {
    if expected.raw.len() != actual.raw.len() {
        bail!("output length mismatch");
    }
    for (index, (&expected_value, &actual_value)) in
        expected.raw.iter().zip(&actual.raw).enumerate()
    {
        if !actual_value.is_finite() {
            bail!("non-finite output at {index}");
        }
        let ulp = expected_value.to_bits().abs_diff(actual_value.to_bits());
        if (expected_value - actual_value).abs() > 1e-6 && ulp > 4 {
            bail!("output mismatch at {index}: {expected_value} != {actual_value} ({ulp} ULP)");
        }
        if expected.bf16_boundary[index].to_bits() != actual.bf16_boundary[index].to_bits() {
            bail!("BF16 boundary mismatch at {index}");
        }
    }
    Ok(())
}

impl Projection {
    fn open(snapshot: &Path, role: Role) -> Result<Self> {
        let blocks_name = role.tensor_name("blocks");
        let scales_name = role.tensor_name("scales");
        let bias_name = role.tensor_name("bias");
        for entry in std::fs::read_dir(snapshot)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("safetensors") {
                continue;
            }
            let mut file = File::open(&path)?;
            let file_len = usize::try_from(file.metadata()?.len())
                .context("checkpoint shard is too large for this platform")?;
            let mut length_bytes = [0_u8; 8];
            file.read_exact(&mut length_bytes)?;
            let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
                .context("safetensors header is too large for this platform")?;
            if header_len == 0 || header_len > 128 * 1024 * 1024 {
                bail!(
                    "{} has an invalid safetensors header length",
                    path.display()
                );
            }
            let data_start = 8_usize
                .checked_add(header_len)
                .context("safetensors data offset overflow")?;
            if data_start > file_len {
                bail!("{} has a truncated safetensors header", path.display());
            }
            let mut header_bytes = vec![0_u8; header_len];
            file.read_exact(&mut header_bytes)?;
            let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
            let Some(blocks_value) = header.get(&blocks_name) else {
                continue;
            };
            let blocks_tensor: TensorHeader = serde_json::from_value(blocks_value.clone())?;
            let scales_tensor: TensorHeader = serde_json::from_value(
                header
                    .get(&scales_name)
                    .with_context(|| format!("{path:?} is missing {scales_name}"))?
                    .clone(),
            )?;
            let bias_tensor: TensorHeader = serde_json::from_value(
                header
                    .get(&bias_name)
                    .with_context(|| format!("{path:?} is missing {bias_name}"))?
                    .clone(),
            )?;
            let shape = &blocks_tensor.shape;
            if shape.len() != 4 || shape[0] != EXPERTS || shape[3] != 16 {
                bail!("{blocks_name} has unexpected shape {shape:?}");
            }
            let columns = shape[1];
            let blocks = shape[2];
            if !columns.is_multiple_of(32)
                || scales_tensor.shape != [EXPERTS, columns, blocks]
                || bias_tensor.shape != [EXPERTS, columns]
                || blocks_tensor.dtype.size() != 1
                || scales_tensor.dtype.size() != 1
            {
                bail!("{role:?} tensor extents disagree");
            }
            let block_bytes = columns
                .checked_mul(blocks)
                .and_then(|value| value.checked_mul(16))
                .context("block tensor extent overflow")?;
            let scale_bytes = columns
                .checked_mul(blocks)
                .context("scale tensor extent overflow")?;
            let bias_bytes = columns
                .checked_mul(bias_tensor.dtype.size())
                .context("bias tensor extent overflow")?;
            let packed =
                read_expert_tensor(&mut file, file_len, data_start, &blocks_tensor, block_bytes)?;
            let scales =
                read_expert_tensor(&mut file, file_len, data_start, &scales_tensor, scale_bytes)?;
            let bias_bytes =
                read_expert_tensor(&mut file, file_len, data_start, &bias_tensor, bias_bytes)?;
            let bias = decode_floats(bias_tensor.dtype, &bias_bytes)?;
            let canonical = canonical_records(&packed, &scales);
            let x8 = x8_records(&packed, &scales, columns, blocks);
            return Ok(Self {
                role,
                columns,
                blocks,
                packed,
                scales,
                bias,
                canonical,
                x8,
            });
        }
        bail!(
            "could not locate {blocks_name} under {}",
            snapshot.display()
        )
    }
}

fn read_expert_tensor(
    file: &mut File,
    file_len: usize,
    data_start: usize,
    tensor: &TensorHeader,
    expert_bytes: usize,
) -> Result<Vec<u8>> {
    let expected_bytes = expert_bytes
        .checked_mul(EXPERTS)
        .context("expert tensor extent overflow")?;
    let (tensor_start, tensor_end) = tensor.data_offsets;
    if tensor_end < tensor_start || tensor_end - tensor_start != expected_bytes {
        bail!("expert tensor data extent disagrees with its shape");
    }
    let expert_offset = EXPERT
        .checked_mul(expert_bytes)
        .context("expert tensor offset overflow")?;
    let start = data_start
        .checked_add(tensor_start)
        .and_then(|value| value.checked_add(expert_offset))
        .context("expert tensor file offset overflow")?;
    let end = start
        .checked_add(expert_bytes)
        .context("expert tensor end offset overflow")?;
    let tensor_file_end = data_start
        .checked_add(tensor_end)
        .context("expert tensor file extent overflow")?;
    if end > file_len || end > tensor_file_end {
        bail!("expert tensor data is truncated");
    }
    file.seek(SeekFrom::Start(
        u64::try_from(start).context("expert tensor offset exceeds u64")?,
    ))?;
    let mut bytes = vec![0_u8; expert_bytes];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn decode_floats(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    let values = match dtype {
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        other => bail!("unsupported bias dtype {other:?}"),
    };
    Ok(values)
}

fn canonical_records(packed: &[u8], scales: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(scales.len() * 17);
    for (record, scale) in packed.chunks_exact(16).zip(scales) {
        output.push(*scale);
        output.extend_from_slice(record);
    }
    output
}

fn x8_records(packed: &[u8], scales: &[u8], columns: usize, blocks: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(scales.len() * 17);
    for group in 0..columns / 8 {
        for block in 0..blocks {
            for lane in 0..8 {
                output.push(scales[(group * 8 + lane) * blocks + block]);
            }
            let split = std::array::from_fn::<_, 8, _>(|lane| {
                let source = ((group * 8 + lane) * blocks + block) * 16;
                mxfp4_adjacent_to_split(packed[source..source + 16].try_into().unwrap())
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

fn deterministic_activations(rows: usize, width: usize, role: Role) -> Vec<f32> {
    let role_seed = match role {
        Role::GateUp => 0x4741_5445,
        Role::Down => 0x444f_574e,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(0x5845_5052_4f44 ^ rows as u64 ^ role_seed);
    (0..rows * width)
        .map(|index| {
            let structured = ((index % 97) as f32 - 48.0) / 31.0;
            bf16::from_f32(structured + rng.gen_range(-0.125..=0.125)).to_f32()
        })
        .collect()
}
