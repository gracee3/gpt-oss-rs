use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use serde_json::{json, Value};

const M: usize = 74;
const HIDDEN: usize = 2880;
const Q_OUT: usize = 4096;
const KV_OUT: usize = 512;
const HEAD_DIM: usize = 64;

#[derive(Debug, Parser)]
#[command(
    name = "qkv_projection_policy_compare",
    about = "Validate Q/K/V projection policy comparison inputs and emit a skeleton status artifact"
)]
struct Cli {
    #[arg(long)]
    norm_input: PathBuf,

    #[arg(long)]
    q_weight: PathBuf,

    #[arg(long)]
    k_weight: PathBuf,

    #[arg(long)]
    v_weight: PathBuf,

    #[arg(long)]
    q_oracle: PathBuf,

    #[arg(long)]
    k_oracle: PathBuf,

    #[arg(long)]
    v_oracle: PathBuf,

    #[arg(long, default_value = "current")]
    policy: String,

    #[arg(long, default_value = "all")]
    projection: String,

    #[arg(long, default_value = "f16", value_parser = ["f16", "bf16"])]
    storage_dtype: String,

    #[arg(long)]
    execute: bool,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct Status {
    mode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    projection_execution: bool,
    projection: String,
    storage_dtype: String,
    artifact_metadata_loaded: bool,
    policies_requested: Vec<String>,
    inputs: Inputs,
    shape_validation: ShapeValidation,
    artifacts: BTreeMap<&'static str, ArtifactStatus>,
    expected_shapes: ExpectedShapes,
    k_contract: Value,
    comparisons: Vec<serde_json::Value>,
    latency: BTreeMap<String, serde_json::Value>,
    runtime_forward_reference: Value,
    warnings: Vec<String>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct Inputs {
    norm_input: String,
    q_weight: String,
    k_weight: String,
    v_weight: String,
    q_oracle: String,
    k_oracle: String,
    v_oracle: String,
}

#[derive(Debug, Serialize)]
struct ExpectedShapes {
    norm_input: [usize; 2],
    q: ProjectionShape,
    k: ProjectionShape,
    v: ProjectionShape,
}

#[derive(Debug, Serialize)]
struct ProjectionShape {
    activation: [usize; 2],
    weight: [usize; 2],
    output: [usize; 2],
}

#[derive(Debug, Serialize)]
struct ShapeValidation {
    all_available_shapes_match: bool,
    all_shapes_available: bool,
}

#[derive(Debug, Serialize)]
struct ArtifactStatus {
    path: String,
    json_loaded: bool,
    shape_available: bool,
    discovered_shape: Option<Vec<usize>>,
    expected_shape: [usize; 2],
    shape_matched: Option<bool>,
    dtype: Option<String>,
    digest: Option<String>,
    value_count: Option<usize>,
    warnings: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_inputs(&cli)?;
    let artifacts = load_artifacts(&cli);
    let warnings = collect_warnings(&artifacts);
    let shape_validation = ShapeValidation {
        all_available_shapes_match: artifacts
            .values()
            .all(|artifact| artifact.shape_matched.unwrap_or(true)),
        all_shapes_available: artifacts.values().all(|artifact| artifact.shape_available),
    };
    let artifact_metadata_loaded = artifacts.values().all(|artifact| artifact.json_loaded);
    let policies_requested = parse_policies(&cli.policy);
    let mut projection_execution = false;
    let mut classification = classify(&artifacts);
    let mut comparisons = Vec::new();
    let mut latency = BTreeMap::new();
    let mut k_contract = Value::Null;
    let mut runtime_forward_reference = Value::Null;
    let mut next_bounded_step =
        "implement baseline current CUDA helper comparison against loaded artifacts";

    if cli.execute {
        let execution = run_execution(&cli, &policies_requested)?;
        projection_execution = execution.projection_execution;
        classification = execution.classification;
        comparisons = execution.comparisons;
        latency = execution.latency;
        k_contract = execution.k_contract;
        runtime_forward_reference = execution.runtime_forward_reference;
        next_bounded_step = execution.next_bounded_step;
    }

    let status = Status {
        mode: "qkv_projection_policy_compare",
        classification,
        implemented: projection_execution,
        runtime_behavior_changed: false,
        projection_execution,
        projection: cli.projection.clone(),
        storage_dtype: cli.storage_dtype.clone(),
        artifact_metadata_loaded,
        policies_requested,
        inputs: Inputs {
            norm_input: path_string(&cli.norm_input),
            q_weight: path_string(&cli.q_weight),
            k_weight: path_string(&cli.k_weight),
            v_weight: path_string(&cli.v_weight),
            q_oracle: path_string(&cli.q_oracle),
            k_oracle: path_string(&cli.k_oracle),
            v_oracle: path_string(&cli.v_oracle),
        },
        shape_validation,
        artifacts,
        expected_shapes: ExpectedShapes {
            norm_input: [M, HIDDEN],
            q: ProjectionShape {
                activation: [M, HIDDEN],
                weight: [Q_OUT, HIDDEN],
                output: [M, Q_OUT],
            },
            k: ProjectionShape {
                activation: [M, HIDDEN],
                weight: [KV_OUT, HIDDEN],
                output: [M, KV_OUT],
            },
            v: ProjectionShape {
                activation: [M, HIDDEN],
                weight: [KV_OUT, HIDDEN],
                output: [M, KV_OUT],
            },
        },
        k_contract,
        comparisons,
        latency,
        runtime_forward_reference,
        warnings,
        next_bounded_step,
    };

    write_status(&cli.output, &status)?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

fn validate_inputs(cli: &Cli) -> Result<()> {
    let required = [
        ("norm_input", cli.norm_input.as_path()),
        ("q_weight", cli.q_weight.as_path()),
        ("k_weight", cli.k_weight.as_path()),
        ("v_weight", cli.v_weight.as_path()),
        ("q_oracle", cli.q_oracle.as_path()),
        ("k_oracle", cli.k_oracle.as_path()),
        ("v_oracle", cli.v_oracle.as_path()),
    ];

    let missing: Vec<_> = required
        .iter()
        .filter(|(_, path)| !path.exists())
        .map(|(name, path)| format!("{name}={}", path.display()))
        .collect();

    if !missing.is_empty() {
        bail!(
            "classification=qkv_projection_policy_compare_missing_inputs; missing required input path(s): {}",
            missing.join(", ")
        );
    }

    Ok(())
}

fn write_status(output: &Path, status: &Status) -> Result<()> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "classification=qkv_projection_policy_compare_output_write_failed; failed to create output directory {}",
                parent.display()
            )
        })?;
    }

    let bytes = serde_json::to_vec_pretty(status).with_context(|| {
        "classification=qkv_projection_policy_compare_output_write_failed; failed to serialize status"
    })?;
    std::fs::write(output, bytes).with_context(|| {
        format!(
            "classification=qkv_projection_policy_compare_output_write_failed; failed to write {}",
            output.display()
        )
    })?;
    Ok(())
}

fn load_artifacts(cli: &Cli) -> BTreeMap<&'static str, ArtifactStatus> {
    [
        ("norm_input", cli.norm_input.as_path(), [M, HIDDEN]),
        ("q_weight", cli.q_weight.as_path(), [Q_OUT, HIDDEN]),
        ("k_weight", cli.k_weight.as_path(), [KV_OUT, HIDDEN]),
        ("v_weight", cli.v_weight.as_path(), [KV_OUT, HIDDEN]),
        ("q_oracle", cli.q_oracle.as_path(), [M, Q_OUT]),
        ("k_oracle", cli.k_oracle.as_path(), [M, KV_OUT]),
        ("v_oracle", cli.v_oracle.as_path(), [M, KV_OUT]),
    ]
    .into_iter()
    .map(|(name, path, expected_shape)| (name, load_artifact(path, expected_shape)))
    .collect()
}

fn load_artifact(path: &Path, expected_shape: [usize; 2]) -> ArtifactStatus {
    let mut warnings = Vec::new();
    let mut json_loaded = false;
    let mut shape = None;
    let mut dtype = None;
    let mut digest = None;
    let mut value_count = None;

    match std::fs::read_to_string(path) {
        Ok(text) if text.trim().is_empty() => {
            warnings.push("artifact is empty; metadata unavailable".to_string());
        }
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(value) => {
                json_loaded = true;
                let shape_candidates = collect_shapes(&value);
                shape = shape_candidates
                    .iter()
                    .find(|candidate| candidate.as_slice() == expected_shape)
                    .cloned()
                    .or_else(|| shape_candidates.into_iter().next());
                dtype = find_string_field(&value, &is_dtype_key);
                digest = find_string_field(&value, &is_digest_key);
                value_count = find_value_count(&value);
            }
            Err(err) => {
                warnings.push(format!("artifact is not valid JSON: {err}"));
            }
        },
        Err(err) => {
            warnings.push(format!("failed to read artifact: {err}"));
        }
    }

    if shape.is_none() {
        warnings.push("shape metadata not found".to_string());
    }
    if dtype.is_none() {
        warnings.push("dtype metadata not found".to_string());
    }
    if digest.is_none() {
        warnings.push("digest/checksum metadata not found".to_string());
    }

    let shape_matched = shape
        .as_ref()
        .map(|discovered| discovered.as_slice() == expected_shape);
    if shape_matched == Some(false) {
        warnings.push(format!(
            "shape mismatch: expected {:?}, discovered {:?}",
            expected_shape,
            shape.as_ref().unwrap()
        ));
    }

    ArtifactStatus {
        path: path_string(path),
        json_loaded,
        shape_available: shape.is_some(),
        discovered_shape: shape,
        expected_shape,
        shape_matched,
        dtype,
        digest,
        value_count,
        warnings,
    }
}

fn classify(artifacts: &BTreeMap<&'static str, ArtifactStatus>) -> &'static str {
    if artifacts
        .values()
        .any(|artifact| artifact.shape_matched == Some(false))
    {
        "qkv_projection_policy_compare_artifact_shape_mismatch"
    } else if !artifacts
        .values()
        .all(|artifact| artifact.json_loaded && artifact.shape_available)
    {
        "qkv_projection_policy_compare_artifact_metadata_incomplete"
    } else {
        "qkv_projection_policy_compare_artifact_metadata_loaded"
    }
}

struct ExecutionResult {
    classification: &'static str,
    projection_execution: bool,
    comparisons: Vec<Value>,
    latency: BTreeMap<String, Value>,
    k_contract: Value,
    runtime_forward_reference: Value,
    next_bounded_step: &'static str,
}

fn run_execution(cli: &Cli, policies: &[String]) -> Result<ExecutionResult> {
    if cli.projection != "k" {
        let classification = if cli.storage_dtype == "bf16" {
            "qkv_projection_policy_compare_k_bf16_cuda_projection_not_implemented"
        } else {
            "qkv_projection_policy_compare_projection_not_implemented"
        };
        return Ok(ExecutionResult {
            classification,
            projection_execution: false,
            comparisons: Vec::new(),
            latency: BTreeMap::new(),
            k_contract: Value::Null,
            runtime_forward_reference: Value::Null,
            next_bounded_step: "fix artifact/API issue before CUDA policy work",
        });
    }

    if policies != ["current"] {
        let classification = if cli.storage_dtype == "bf16" {
            "qkv_projection_policy_compare_k_bf16_cuda_policy_not_implemented"
        } else {
            "qkv_projection_policy_compare_policy_not_implemented"
        };
        return Ok(ExecutionResult {
            classification,
            projection_execution: false,
            comparisons: Vec::new(),
            latency: BTreeMap::new(),
            k_contract: Value::Null,
            runtime_forward_reference: Value::Null,
            next_bounded_step: "fix artifact/API issue before CUDA policy work",
        });
    }

    run_current_k(cli)
}

#[cfg(feature = "cuda")]
fn run_current_k(cli: &Cli) -> Result<ExecutionResult> {
    if cli.storage_dtype == "bf16" {
        return run_current_k_bf16(cli);
    }
    run_current_k_f16(cli)
}

#[cfg(feature = "cuda")]
fn run_current_k_f16(cli: &Cli) -> Result<ExecutionResult> {
    let norm = load_values(
        &cli.norm_input,
        M * HIDDEN,
        "norm_input",
        "qkv_projection_policy_compare_k_current_blocked_by_artifact_values",
    )?;
    let k_weight = load_values(
        &cli.k_weight,
        KV_OUT * HIDDEN,
        "k_weight",
        "qkv_projection_policy_compare_k_current_blocked_by_artifact_values",
    )?;
    let k_oracle = load_values(
        &cli.k_oracle,
        M * KV_OUT,
        "k_oracle",
        "qkv_projection_policy_compare_k_current_blocked_by_artifact_values",
    )?;

    let norm_f16 = to_f16_values(&norm);
    let k_weight_f16 = to_f16_values(&k_weight);

    let context = gpt_oss_gpu::CudaContext::new(0)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; CUDA context init failed: {err}"))?;
    let stream = context
        .new_stream()
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; CUDA stream init failed: {err}"))?;
    let blas = gpt_oss_gpu::cublas::CublasHandle::new(stream.clone())
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; cuBLAS handle init failed: {err}"))?;

    let norm_gpu = stream
        .clone_htod(&norm_f16)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; norm input upload failed: {err}"))?;
    let k_weight_gpu = stream
        .clone_htod(&k_weight_f16)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; K weight upload failed: {err}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<half::f16>(M * KV_OUT)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; output allocation failed: {err}"))?;

    let warmup_iters = 3usize;
    let timed_iters = 10usize;
    for _ in 0..warmup_iters {
        blas.hgemm(
            M,
            KV_OUT,
            HIDDEN,
            half::f16::from_f32(1.0),
            &norm_gpu,
            &k_weight_gpu,
            half::f16::from_f32(0.0),
            &mut output_gpu,
        )
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; current K hgemm warmup failed: {err}"))?;
    }
    stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; warmup synchronize failed: {err}"))?;

    let start = Instant::now();
    for _ in 0..timed_iters {
        blas.hgemm(
            M,
            KV_OUT,
            HIDDEN,
            half::f16::from_f32(1.0),
            &norm_gpu,
            &k_weight_gpu,
            half::f16::from_f32(0.0),
            &mut output_gpu,
        )
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; current K hgemm timed run failed: {err}"))?;
    }
    stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; timed synchronize failed: {err}"))?;
    let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / timed_iters as f64;

    let output_f16 = stream
        .clone_dtoh(&output_gpu)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_current_cuda_execution_failed; output download failed: {err}"))?;
    let output: Vec<f32> = output_f16.iter().map(|value| value.to_f32()).collect();

    let cuda_vs_oracle = compare_outputs(&output, &k_oracle);
    let cpu_f32 = cpu_k_projection(
        &norm,
        &k_weight,
        RoundingPolicy::F32,
        WeightLayout::RowMajor,
    );
    let cpu_f16 = cpu_k_projection(
        &norm,
        &k_weight,
        RoundingPolicy::F16,
        WeightLayout::RowMajor,
    );
    let cpu_bf16 = cpu_k_projection(
        &norm,
        &k_weight,
        RoundingPolicy::BFloat16,
        WeightLayout::RowMajor,
    );
    let cpu_f32_transposed_flat = cpu_k_projection(
        &norm,
        &k_weight,
        RoundingPolicy::F32,
        WeightLayout::TransposedFlat,
    );

    let cpu_f32_vs_oracle = compare_outputs(&cpu_f32, &k_oracle);
    let cpu_f16_vs_oracle = compare_outputs(&cpu_f16, &k_oracle);
    let cpu_bf16_vs_oracle = compare_outputs(&cpu_bf16, &k_oracle);
    let cpu_transposed_vs_oracle = compare_outputs(&cpu_f32_transposed_flat, &k_oracle);
    let cuda_vs_cpu_f32 = compare_outputs(&output, &cpu_f32);
    let cuda_vs_cpu_f16 = compare_outputs(&output, &cpu_f16);
    let cuda_vs_cpu_bf16 = compare_outputs(&output, &cpu_bf16);

    let classification = classify_k_contract(
        &cuda_vs_oracle,
        &cpu_f16_vs_oracle,
        &cpu_bf16_vs_oracle,
        &cpu_transposed_vs_oracle,
        &cuda_vs_cpu_f16,
    );
    let next_bounded_step = if classification
        == "qkv_projection_policy_compare_k_current_matches_oracle"
    {
        "implement Q/V current CUDA baseline comparisons"
    } else if classification
        == "qkv_projection_policy_compare_k_current_layout_or_orientation_mismatch"
    {
        "fix harness layout/orientation contract before CUDA policy work"
    } else if classification == "qkv_projection_policy_compare_k_current_dtype_policy_mismatch" {
        "design a K-only dtype-policy candidate comparison before runtime projection policy work"
    } else if classification
        == "qkv_projection_policy_compare_k_current_f16_contract_vs_bf16_oracle"
    {
        "implement K-only candidate policy comparison, starting with scoped BF16-compatible projection policy if approved"
    } else {
        "reconcile CUDA helper contract against runtime-forward helper artifacts before candidate policy work"
    };

    let k_contract = json!({
        "activation_shape": [M, HIDDEN],
        "weight_shape": [KV_OUT, HIDDEN],
        "oracle_shape": [M, KV_OUT],
        "activation_conversion": "JSON f32 values rounded to IEEE f16 before CUDA upload",
        "weight_conversion": "JSON f32 values rounded to IEEE f16 before CUDA upload",
        "cublas_input_dtype": "CUDA_R_16F via CublasHandle::hgemm",
        "cublas_output_dtype": "CUDA_R_16F downloaded and widened to f32 for comparison",
        "gemm_operation": "norm_input @ k_weight.T",
        "transa": "CUBLAS_OP_T on K weight in row-major [512, 2880]",
        "transb": "CUBLAS_OP_N on activation in row-major [74, 2880]",
        "m": KV_OUT,
        "n": M,
        "k": HIDDEN,
        "logical_m": M,
        "logical_n": KV_OUT,
        "lda": HIDDEN,
        "ldb": HIDDEN,
        "ldc": KV_OUT,
        "bias_applied": false,
        "output_layout": "[token, feature]",
        "oracle_layout": "[token, feature] flat row-major values loaded from K oracle artifact",
        "notes": [
            "This harness uses current public hgemm, which is f16 input/output, not BF16 storage.",
            "No K bias is applied in this baseline.",
            "The runtime-forward proof artifacts identify oneDNN/PyTorch BF16 linear behavior as the oracle target."
        ]
    });

    let runtime_forward_reference = load_runtime_forward_reference();

    let matched = cuda_vs_oracle.mismatching_element_count == 0;
    let legacy_classification = if matched {
        "qkv_projection_policy_compare_k_current_matches_oracle"
    } else {
        "qkv_projection_policy_compare_k_current_mismatches_oracle"
    };

    let comparison_value = json!({
        "projection": "k",
        "policy": "current",
        "shape": [M, KV_OUT],
        "output_checksum": digest_f32_le(&output),
        "legacy_classification": legacy_classification,
        "gemm": {
            "m": M,
            "n": KV_OUT,
            "k": HIDDEN,
            "operation": "norm_input @ k_weight.T",
            "backend": "current CublasHandle hgemm",
        },
        "vs_oracle": cuda_vs_oracle,
    });
    let cpu_f32_value = replay_json("cpu_f32_accum_f32_output", &cpu_f32, &cpu_f32_vs_oracle);
    let cpu_f16_value = replay_json(
        "cpu_f16_rounded_inputs_f32_accum_f16_output",
        &cpu_f16,
        &cpu_f16_vs_oracle,
    );
    let cpu_bf16_value = replay_json(
        "cpu_bf16_rounded_inputs_f32_accum_bf16_output",
        &cpu_bf16,
        &cpu_bf16_vs_oracle,
    );
    let cpu_transposed_value = replay_json(
        "cpu_f32_transposed_flat_weight_interpretation",
        &cpu_f32_transposed_flat,
        &cpu_transposed_vs_oracle,
    );
    let cuda_vs_cpu_f32_value = comparison_json("cuda_current_vs_cpu_f32", &cuda_vs_cpu_f32);
    let cuda_vs_cpu_f16_value = comparison_json("cuda_current_vs_cpu_f16", &cuda_vs_cpu_f16);
    let cuda_vs_cpu_bf16_value = comparison_json("cuda_current_vs_cpu_bf16", &cuda_vs_cpu_bf16);
    let mut latency = BTreeMap::new();
    latency.insert(
        "current_k".to_string(),
        json!({
            "available": true,
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
            "mean_ms": mean_ms,
        }),
    );
    Ok(ExecutionResult {
        classification,
        projection_execution: true,
        comparisons: vec![
            comparison_value,
            cpu_f32_value,
            cpu_f16_value,
            cpu_bf16_value,
            cpu_transposed_value,
            cuda_vs_cpu_f32_value,
            cuda_vs_cpu_f16_value,
            cuda_vs_cpu_bf16_value,
        ],
        latency,
        k_contract,
        runtime_forward_reference,
        next_bounded_step,
    })
}

#[cfg(feature = "cuda")]
fn run_current_k_bf16(cli: &Cli) -> Result<ExecutionResult> {
    let norm = load_values(
        &cli.norm_input,
        M * HIDDEN,
        "norm_input",
        "qkv_projection_policy_compare_k_bf16_cuda_blocked_by_artifact_values",
    )?;
    let k_weight = load_values(
        &cli.k_weight,
        KV_OUT * HIDDEN,
        "k_weight",
        "qkv_projection_policy_compare_k_bf16_cuda_blocked_by_artifact_values",
    )?;
    let k_oracle = load_values(
        &cli.k_oracle,
        M * KV_OUT,
        "k_oracle",
        "qkv_projection_policy_compare_k_bf16_cuda_blocked_by_artifact_values",
    )?;

    let norm_bf16 = to_bf16_values(&norm);
    let k_weight_bf16 = to_bf16_values(&k_weight);

    let context = gpt_oss_gpu::CudaContext::new(0)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; CUDA context init failed: {err}"))?;
    let stream = context
        .new_stream()
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; CUDA stream init failed: {err}"))?;
    let blas = gpt_oss_gpu::cublas::CublasHandle::new(stream.clone())
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; cuBLAS handle init failed: {err}"))?;

    let norm_gpu = stream
        .clone_htod(&norm_bf16)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; norm input upload failed: {err}"))?;
    let k_weight_gpu = stream
        .clone_htod(&k_weight_bf16)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; K weight upload failed: {err}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<half::bf16>(M * KV_OUT)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; output allocation failed: {err}"))?;

    let warmup_iters = 3usize;
    let timed_iters = 10usize;
    for _ in 0..warmup_iters {
        blas.bf16_gemm_into(
            M,
            KV_OUT,
            HIDDEN,
            1.0,
            &norm_gpu,
            &k_weight_gpu,
            0.0,
            &mut output_gpu,
        )
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; current K BF16 GEMM warmup failed: {err}"))?;
    }
    stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; warmup synchronize failed: {err}"))?;

    let start = Instant::now();
    for _ in 0..timed_iters {
        blas.bf16_gemm_into(
            M,
            KV_OUT,
            HIDDEN,
            1.0,
            &norm_gpu,
            &k_weight_gpu,
            0.0,
            &mut output_gpu,
        )
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; current K BF16 GEMM timed run failed: {err}"))?;
    }
    stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; timed synchronize failed: {err}"))?;
    let mean_ms = start.elapsed().as_secs_f64() * 1000.0 / timed_iters as f64;

    let output_bf16 = stream
        .clone_dtoh(&output_gpu)
        .map_err(|err| anyhow::anyhow!("classification=qkv_projection_policy_compare_k_bf16_cuda_execution_failed; output download failed: {err}"))?;
    let output: Vec<f32> = output_bf16.iter().map(|value| value.to_f32()).collect();

    let cpu_bf16 = cpu_k_projection(
        &norm,
        &k_weight,
        RoundingPolicy::BFloat16,
        WeightLayout::RowMajor,
    );
    let cuda_vs_oracle = compare_outputs(&output, &k_oracle);
    let cuda_vs_cpu_bf16 = compare_outputs(&output, &cpu_bf16);
    let cpu_bf16_vs_oracle = compare_outputs(&cpu_bf16, &k_oracle);

    let classification =
        classify_k_bf16_contract(&cuda_vs_oracle, &cuda_vs_cpu_bf16, &cpu_bf16_vs_oracle);
    let next_bounded_step = if classification
        == "qkv_projection_policy_compare_k_bf16_cuda_matches_oracle"
    {
        "implement Q/V current CUDA baseline comparisons"
    } else if classification
        == "qkv_projection_policy_compare_k_bf16_cuda_matches_legacy_bf16_contract"
    {
        "implement K-only candidate policy comparison, starting with scoped cuBLAS pedantic/no-tensor-op if approved"
    } else {
        "reconcile BF16 CUDA helper contract before candidate policy work"
    };

    let k_contract = json!({
        "storage_dtype": "bf16",
        "activation_shape": [M, HIDDEN],
        "weight_shape": [KV_OUT, HIDDEN],
        "oracle_shape": [M, KV_OUT],
        "activation_conversion": "JSON f32 values rounded to BF16 before CUDA upload",
        "weight_conversion": "JSON f32 values rounded to BF16 before CUDA upload",
        "bf16_conversion_policy": "half::bf16::from_f32 on JSON f32 values",
        "cublas_input_dtype": "CUDA_R_16BF via CublasHandle::bf16_gemm_into",
        "cublas_output_dtype": "CUDA_R_16BF downloaded and widened to f32 for comparison",
        "compute_dtype": "CUBLAS_COMPUTE_32F",
        "algorithm": "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        "gemm_operation": "norm_input @ k_weight.T",
        "transa": "CUBLAS_OP_T on K weight in row-major [512, 2880]",
        "transb": "CUBLAS_OP_N on activation in row-major [74, 2880]",
        "m": KV_OUT,
        "n": M,
        "k": HIDDEN,
        "logical_m": M,
        "logical_n": KV_OUT,
        "lda": HIDDEN,
        "ldb": HIDDEN,
        "ldc": KV_OUT,
        "bias_applied": false,
        "output_layout": "[token, feature]",
        "oracle_layout": "[token, feature] flat row-major values loaded from K oracle artifact",
        "notes": [
            "This harness uses the public BF16 validation API, not runtime projection routing.",
            "No K bias is applied in this baseline.",
            "No pedantic/no-tensor-op cuBLAS policy is used."
        ]
    });

    let comparison_value = json!({
        "projection": "k",
        "policy": "current",
        "storage_dtype": "bf16",
        "shape": [M, KV_OUT],
        "output_checksum": digest_f32_le(&output),
        "gemm": {
            "m": M,
            "n": KV_OUT,
            "k": HIDDEN,
            "operation": "norm_input @ k_weight.T",
            "backend": "CublasHandle::bf16_gemm_into",
            "input_dtype": "CUDA_R_16BF",
            "output_dtype": "CUDA_R_16BF",
            "compute_dtype": "CUBLAS_COMPUTE_32F",
            "algorithm": "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        },
        "vs_oracle": cuda_vs_oracle,
    });
    let cuda_vs_cpu_bf16_value =
        comparison_json("cuda_bf16_current_vs_cpu_bf16", &cuda_vs_cpu_bf16);
    let cpu_bf16_value = replay_json(
        "cpu_bf16_rounded_inputs_f32_accum_bf16_output",
        &cpu_bf16,
        &cpu_bf16_vs_oracle,
    );
    let mut latency = BTreeMap::new();
    latency.insert(
        "current_k_bf16".to_string(),
        json!({
            "available": true,
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
            "mean_ms": mean_ms,
        }),
    );

    Ok(ExecutionResult {
        classification,
        projection_execution: true,
        comparisons: vec![comparison_value, cuda_vs_cpu_bf16_value, cpu_bf16_value],
        latency,
        k_contract,
        runtime_forward_reference: load_runtime_forward_reference(),
        next_bounded_step,
    })
}

#[cfg(not(feature = "cuda"))]
fn run_current_k(_cli: &Cli) -> Result<ExecutionResult> {
    Ok(ExecutionResult {
        classification: "qkv_projection_policy_compare_k_current_cuda_execution_failed",
        projection_execution: false,
        comparisons: Vec::new(),
        latency: BTreeMap::new(),
        k_contract: Value::Null,
        runtime_forward_reference: Value::Null,
        next_bounded_step: "fix artifact/API issue before CUDA policy work",
    })
}

#[derive(Clone, Debug, Serialize)]
struct ComparisonMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    matched: bool,
    mismatching_element_count: usize,
    first_mismatch: Option<Mismatch>,
    worst_mismatch: Option<Mismatch>,
    mismatch_table: Vec<Mismatch>,
}

#[derive(Clone, Debug, Serialize)]
struct Mismatch {
    token: usize,
    feature: usize,
    kv_head: usize,
    lane: usize,
    local: f32,
    oracle: f32,
    abs_diff: f32,
}

fn compare_outputs(local: &[f32], oracle: &[f32]) -> ComparisonMetrics {
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut mismatching_element_count = 0usize;
    let mut first_mismatch = None;
    let mut worst_mismatch = None;
    let mut mismatch_table = Vec::new();

    for (index, (&local_value, &oracle_value)) in local.iter().zip(oracle.iter()).enumerate() {
        let abs_diff = (local_value - oracle_value).abs();
        sum_abs_diff += f64::from(abs_diff);
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            worst_mismatch = Some(make_mismatch(index, local_value, oracle_value, abs_diff));
        }
        if local_value.to_bits() != oracle_value.to_bits() {
            mismatching_element_count += 1;
            let mismatch = make_mismatch(index, local_value, oracle_value, abs_diff);
            first_mismatch.get_or_insert_with(|| mismatch.clone());
            if mismatch_table.len() < 20 {
                mismatch_table.push(mismatch);
            }
        }
    }

    ComparisonMetrics {
        max_abs_diff,
        mean_abs_diff: (sum_abs_diff / local.len() as f64) as f32,
        matched: mismatching_element_count == 0,
        mismatching_element_count,
        first_mismatch,
        worst_mismatch,
        mismatch_table,
    }
}

fn make_mismatch(index: usize, local: f32, oracle: f32, abs_diff: f32) -> Mismatch {
    let token = index / KV_OUT;
    let feature = index % KV_OUT;
    Mismatch {
        token,
        feature,
        kv_head: feature / HEAD_DIM,
        lane: feature % HEAD_DIM,
        local,
        oracle,
        abs_diff,
    }
}

#[derive(Clone, Copy)]
enum RoundingPolicy {
    F32,
    F16,
    BFloat16,
}

#[derive(Clone, Copy)]
enum WeightLayout {
    RowMajor,
    TransposedFlat,
}

fn cpu_k_projection(
    input: &[f32],
    weight: &[f32],
    policy: RoundingPolicy,
    layout: WeightLayout,
) -> Vec<f32> {
    let input = round_values(input, policy);
    let weight = round_values(weight, policy);
    let mut output = vec![0.0f32; M * KV_OUT];

    for token in 0..M {
        for feature in 0..KV_OUT {
            let mut sum = 0.0f32;
            for hidden in 0..HIDDEN {
                let input_value = input[token * HIDDEN + hidden];
                let weight_value = match layout {
                    WeightLayout::RowMajor => weight[feature * HIDDEN + hidden],
                    WeightLayout::TransposedFlat => weight[hidden * KV_OUT + feature],
                };
                sum += input_value * weight_value;
            }
            output[token * KV_OUT + feature] = round_output(sum, policy);
        }
    }

    output
}

fn round_values(values: &[f32], policy: RoundingPolicy) -> Vec<f32> {
    values
        .iter()
        .map(|value| round_output(*value, policy))
        .collect()
}

fn round_output(value: f32, policy: RoundingPolicy) -> f32 {
    match policy {
        RoundingPolicy::F32 => value,
        RoundingPolicy::F16 => half::f16::from_f32(value).to_f32(),
        RoundingPolicy::BFloat16 => half::bf16::from_f32(value).to_f32(),
    }
}

fn replay_json(name: &'static str, output: &[f32], vs_oracle: &ComparisonMetrics) -> Value {
    json!({
        "projection": "k",
        "replay": name,
        "shape": [M, KV_OUT],
        "output_checksum": digest_f32_le(output),
        "vs_oracle": vs_oracle,
    })
}

fn comparison_json(name: &'static str, metrics: &ComparisonMetrics) -> Value {
    json!({
        "projection": "k",
        "comparison": name,
        "shape": [M, KV_OUT],
        "metrics": metrics,
    })
}

fn classify_k_contract(
    cuda_vs_oracle: &ComparisonMetrics,
    cpu_f16_vs_oracle: &ComparisonMetrics,
    cpu_bf16_vs_oracle: &ComparisonMetrics,
    cpu_transposed_vs_oracle: &ComparisonMetrics,
    cuda_vs_cpu_f16: &ComparisonMetrics,
) -> &'static str {
    if cuda_vs_oracle.matched {
        "qkv_projection_policy_compare_k_current_matches_oracle"
    } else if cpu_bf16_vs_oracle.matched && !cpu_f16_vs_oracle.matched {
        "qkv_projection_policy_compare_k_current_dtype_policy_mismatch"
    } else if cpu_transposed_vs_oracle.mismatching_element_count
        < cuda_vs_oracle.mismatching_element_count / 2
    {
        "qkv_projection_policy_compare_k_current_layout_or_orientation_mismatch"
    } else if cuda_vs_cpu_f16.matched && !cpu_f16_vs_oracle.matched {
        "qkv_projection_policy_compare_k_current_f16_contract_vs_bf16_oracle"
    } else {
        "qkv_projection_policy_compare_k_current_cuda_contract_unmodeled"
    }
}

fn classify_k_bf16_contract(
    cuda_vs_oracle: &ComparisonMetrics,
    cuda_vs_cpu_bf16: &ComparisonMetrics,
    cpu_bf16_vs_oracle: &ComparisonMetrics,
) -> &'static str {
    if cuda_vs_oracle.matched {
        "qkv_projection_policy_compare_k_bf16_cuda_matches_oracle"
    } else if cuda_vs_cpu_bf16.matched && !cpu_bf16_vs_oracle.matched {
        "qkv_projection_policy_compare_k_bf16_cuda_matches_legacy_bf16_contract"
    } else {
        "qkv_projection_policy_compare_k_bf16_cuda_contract_unmodeled"
    }
}

fn load_runtime_forward_reference() -> Value {
    let helper_path = Path::new("/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-onednn-oracle-helper-proof-status.json");
    let scoped_path = Path::new("/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-onednn-oracle-scoped-helper-fix-status.json");

    let helper = read_json_file(helper_path);
    let scoped = read_json_file(scoped_path);

    json!({
        "available": helper.is_some() || scoped.is_some(),
        "helper_artifact": helper_path.display().to_string(),
        "scoped_candidate_artifact": scoped_path.display().to_string(),
        "known_legacy_vs_oracle_metrics": helper
            .as_ref()
            .and_then(|value| value.pointer("/helper_replay_vs_oracle_metrics/rust_cpu_replay_vs_onednn_oracle/metrics"))
            .cloned()
            .or_else(|| scoped
                .as_ref()
                .and_then(|value| value.pointer("/legacy_helper_replay_metrics/legacy_rust_cpu_replay_vs_onednn_oracle"))
                .cloned()),
        "known_candidate_vs_oracle_metrics": scoped
            .as_ref()
            .and_then(|value| value.pointer("/candidate_helper_projection_metrics/scoped_candidate_vs_onednn_oracle"))
            .cloned(),
        "known_six_lane_table": helper
            .as_ref()
            .and_then(|value| value.pointer("/helper_replay_vs_oracle_metrics/rust_cpu_replay_vs_onednn_oracle/known_six_lane_values"))
            .cloned(),
        "notes": [
            "Runtime-forward artifacts report a six-lane legacy/helper-vs-oneDNN K mismatch, not the broad mismatch produced by an uncalibrated f16 hgemm artifact harness.",
            "The scoped oneDNN candidate path is proof-only and is not imported by this harness."
        ],
    })
}

fn read_json_file(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn load_values(
    path: &Path,
    expected_count: usize,
    name: &str,
    classification: &str,
) -> Result<Vec<f32>> {
    let text = std::fs::read_to_string(path).with_context(|| {
        format!(
            "classification={classification}; failed to read {name} values from {}",
            path.display()
        )
    })?;
    let json: Value = serde_json::from_str(&text)
        .with_context(|| format!("classification={classification}; {name} is not valid JSON"))?;
    let values = find_numeric_values_for_artifact(&json, name).with_context(|| {
        format!(
            "classification={classification}; {name} has no supported values/tensor_values/data/output_values/value field"
        )
    })?;
    if values.len() != expected_count {
        bail!(
            "classification={classification}; {name} value count mismatch: expected {expected_count}, found {}",
            values.len()
        );
    }
    Ok(values)
}

fn find_numeric_values_for_artifact(value: &Value, name: &str) -> Option<Vec<f32>> {
    let preferred_keys: &[&str] = match name {
        // Runtime-forward local artifacts use this key for the normalized layer0 attention input
        // consumed by Q/K/V projections.
        "norm_input" => &["layer0_attn_norm_output_f32"],
        "k_weight" => &["official_k_weight_f32"],
        "k_oracle" => &[
            "official_module_k_output_f32",
            "official_manual_k_output_f32",
            "onednn_k_output_f32",
        ],
        _ => &[],
    };
    for key in preferred_keys {
        if let Some(values) = find_numeric_values_by_key(value, key) {
            return Some(values);
        }
    }
    find_numeric_values_inner(value, None, &is_value_key)
}

fn find_numeric_values_by_key(value: &Value, target_key: &str) -> Option<Vec<f32>> {
    find_numeric_values_inner(value, None, &|key| key == target_key)
}

fn find_numeric_values_inner(
    value: &Value,
    key: Option<&str>,
    key_predicate: &dyn Fn(&str) -> bool,
) -> Option<Vec<f32>> {
    if key.is_some_and(key_predicate) {
        if let Some(values) = flatten_f32_values(value) {
            return Some(values);
        }
    }

    match value {
        Value::Object(map) => map.iter().find_map(|(child_key, child_value)| {
            find_numeric_values_inner(child_value, Some(child_key.as_str()), key_predicate)
        }),
        Value::Array(items) => items
            .iter()
            .find_map(|item| find_numeric_values_inner(item, None, key_predicate)),
        _ => None,
    }
}

fn flatten_f32_values(value: &Value) -> Option<Vec<f32>> {
    match value {
        Value::Number(number) => number.as_f64().map(|number| vec![number as f32]),
        Value::Array(items) => {
            let mut values = Vec::new();
            for item in items {
                values.extend(flatten_f32_values(item)?);
            }
            Some(values)
        }
        _ => None,
    }
}

fn to_f16_values(values: &[f32]) -> Vec<half::f16> {
    values
        .iter()
        .map(|value| half::f16::from_f32(*value))
        .collect()
}

fn to_bf16_values(values: &[f32]) -> Vec<half::bf16> {
    values
        .iter()
        .map(|value| half::bf16::from_f32(*value))
        .collect()
}

fn digest_f32_le(values: &[f32]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn collect_warnings(artifacts: &BTreeMap<&'static str, ArtifactStatus>) -> Vec<String> {
    artifacts
        .iter()
        .flat_map(|(name, artifact)| {
            artifact
                .warnings
                .iter()
                .map(move |warning| format!("{name}: {warning}"))
        })
        .collect()
}

fn collect_shapes(value: &Value) -> Vec<Vec<usize>> {
    let mut shapes = Vec::new();
    collect_shapes_inner(value, None, &mut shapes);
    shapes
}

fn collect_shapes_inner(value: &Value, key: Option<&str>, shapes: &mut Vec<Vec<usize>>) {
    if key.is_some_and(is_shape_key) {
        if let Some(shape) = as_usize_array(value) {
            shapes.push(shape);
        }
    }

    match value {
        Value::Object(map) => {
            for (child_key, child_value) in map {
                collect_shapes_inner(child_value, Some(child_key.as_str()), shapes);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_shapes_inner(item, None, shapes);
            }
        }
        _ => {}
    }
}

fn as_usize_array(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Array(items) => {
            let shape: Option<Vec<_>> = items
                .iter()
                .map(|item| item.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect();
            shape.filter(|shape| !shape.is_empty())
        }
        _ => None,
    }
}

fn find_string_field(value: &Value, key_predicate: &dyn Fn(&str) -> bool) -> Option<String> {
    find_string_field_inner(value, None, key_predicate)
}

fn find_string_field_inner(
    value: &Value,
    key: Option<&str>,
    key_predicate: &dyn Fn(&str) -> bool,
) -> Option<String> {
    if key.is_some_and(key_predicate) {
        if let Some(text) = value.as_str() {
            return Some(text.to_string());
        }
    }

    match value {
        Value::Object(map) => map.iter().find_map(|(child_key, child_value)| {
            find_string_field_inner(child_value, Some(child_key.as_str()), key_predicate)
        }),
        Value::Array(items) => items
            .iter()
            .find_map(|item| find_string_field_inner(item, None, key_predicate)),
        _ => None,
    }
}

fn find_value_count(value: &Value) -> Option<usize> {
    find_value_count_inner(value, None)
}

fn find_value_count_inner(value: &Value, key: Option<&str>) -> Option<usize> {
    if key.is_some_and(is_count_key) {
        if let Some(count) = value.as_u64().and_then(|n| usize::try_from(n).ok()) {
            return Some(count);
        }
    }
    if key.is_some_and(is_value_key) {
        if let Some(values) = value.as_array() {
            return Some(flat_value_count(values));
        }
    }

    match value {
        Value::Object(map) => map.iter().find_map(|(child_key, child_value)| {
            find_value_count_inner(child_value, Some(child_key.as_str()))
        }),
        Value::Array(items) => items
            .iter()
            .find_map(|item| find_value_count_inner(item, None)),
        _ => None,
    }
}

fn flat_value_count(values: &[Value]) -> usize {
    values
        .iter()
        .map(|value| match value {
            Value::Array(items) => flat_value_count(items),
            _ => 1,
        })
        .sum()
}

fn is_shape_key(key: &str) -> bool {
    matches!(key, "shape" | "tensor_shape" | "output_shape")
}

fn is_dtype_key(key: &str) -> bool {
    key == "dtype" || key == "serialization_dtype" || key.ends_with("_dtype")
}

fn is_digest_key(key: &str) -> bool {
    key == "digest" || key == "checksum" || key.ends_with("_digest")
}

fn is_count_key(key: &str) -> bool {
    matches!(key, "count" | "value_count" | "num_values")
}

fn is_value_key(key: &str) -> bool {
    matches!(
        key,
        "values" | "tensor_values" | "data" | "output_values" | "value"
    )
}

fn parse_policies(policy: &str) -> Vec<String> {
    policy
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn path_string(path: &Path) -> String {
    path.display().to_string()
}
