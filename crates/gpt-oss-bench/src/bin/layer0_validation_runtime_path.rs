use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(
    name = "layer0_validation_runtime_path",
    about = "Emit the validation-only layer0 runtime-path skeleton status"
)]
struct Cli {
    /// Validation submode to run.
    #[arg(long, default_value = "skeleton")]
    mode: Mode,

    /// Exact layer0 input/residual artifact for the validation case.
    #[arg(long)]
    layer0_input: Option<PathBuf>,

    /// Official layer0 final-token output artifact to compare against later.
    #[arg(long)]
    official_layer0_output: Option<PathBuf>,

    /// Official K pre-RoPE artifact for the runtime/kernel RoPE parity guard.
    #[arg(long)]
    k_pre_rope: Option<PathBuf>,

    /// Official K post-RoPE oracle artifact for the runtime/kernel RoPE parity guard.
    #[arg(long)]
    k_post_rope_oracle: Option<PathBuf>,

    /// Token count for K RoPE validation.
    #[arg(long, default_value_t = 74)]
    token_count: usize,

    /// Number of grouped-query K/V heads for K RoPE validation.
    #[arg(long, default_value_t = 8)]
    kv_heads: usize,

    /// Per-head dimension for K RoPE validation.
    #[arg(long, default_value_t = 64)]
    head_dim: usize,

    /// Logical position range for K RoPE validation.
    #[arg(long, default_value = "0:74")]
    positions: String,

    /// Storage dtype for K RoPE validation.
    #[arg(long, default_value = "f16", value_parser = ["f16"])]
    dtype: String,

    /// RoPE theta used by the current runtime table helper.
    #[arg(long, default_value_t = 150000.0)]
    rope_theta: f32,

    /// JSON status output path.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Skeleton,
    KRope,
}

#[derive(Debug, Serialize)]
struct Status {
    mode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    cuda_execution: bool,
    model_runner_routing_changed: bool,
    inputs: Inputs,
    target: Target,
    planned_stages: Vec<&'static str>,
    do_not_promote_boundaries: Vec<&'static str>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct KRopeStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    cuda_execution: bool,
    kernel_used: Option<&'static str>,
    api_used: &'static str,
    artifacts: KRopeArtifacts,
    rope_config: KRopeConfig,
    metrics: Option<ComparisonMetrics>,
    first_mismatch: Option<LogicalDiff>,
    worst_mismatch: Option<LogicalDiff>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct KRopeArtifacts {
    k_pre_rope: KArtifactStatus,
    k_post_rope_oracle: KArtifactStatus,
}

#[derive(Debug, Serialize)]
struct KArtifactStatus {
    path: String,
    json_loaded: bool,
    shape: Option<Vec<usize>>,
    value_count: Option<usize>,
    expected_value_count: usize,
    shape_or_count_matched: bool,
    value_key: Option<String>,
}

#[derive(Debug, Serialize)]
struct KRopeConfig {
    token_count: usize,
    kv_heads: usize,
    head_dim: usize,
    positions: String,
    rope_theta: f32,
    table_source: &'static str,
    lane_pairing: &'static str,
    dtype: String,
}

#[derive(Debug, Serialize)]
struct ComparisonMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatches: usize,
}

#[derive(Debug, Serialize)]
struct LogicalDiff {
    token: usize,
    kv_head: usize,
    lane: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct Blocker {
    kind: &'static str,
    detail: &'static str,
}

#[derive(Debug, Serialize)]
struct Inputs {
    layer0_input: ArtifactPath,
    official_layer0_output: ArtifactPath,
}

#[derive(Debug, Serialize)]
struct ArtifactPath {
    path: String,
    exists: bool,
}

#[derive(Debug, Serialize)]
struct Target {
    exact_case: &'static str,
    output_boundary: &'static str,
    output_shape: [usize; 1],
    comparison: &'static str,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.mode {
        Mode::Skeleton => run_skeleton(&cli),
        Mode::KRope => run_k_rope(&cli),
    }
}

fn run_skeleton(cli: &Cli) -> Result<()> {
    let layer0_input = required_path(&cli.layer0_input, "layer0 input")?;
    let official_layer0_output =
        required_path(&cli.official_layer0_output, "official layer0 output")?;
    validate_path(layer0_input, "layer0 input")?;
    validate_path(official_layer0_output, "official layer0 output")?;

    let status = Status {
        mode: "layer0_validation_runtime_path",
        classification: "layer0_validation_runtime_path_skeleton_ready",
        implemented: false,
        runtime_behavior_changed: false,
        validation_only: true,
        cuda_execution: false,
        model_runner_routing_changed: false,
        inputs: Inputs {
            layer0_input: artifact_path(layer0_input),
            official_layer0_output: artifact_path(official_layer0_output),
        },
        target: Target {
            exact_case: "developer-message-user-smoke",
            output_boundary: "layer0_final_token_hidden_state_after_mlp_residual_add",
            output_shape: [2880],
            comparison: "planned exact BF16-boundary comparison against official oracle",
        },
        planned_stages: vec![
            "skeleton_status_binary",
            "layer0_attention_only_validation",
            "layer0_mlp_validation",
            "full_layer0_final_token_validation",
            "layer_ladder_and_final_logits_later",
        ],
        do_not_promote_boundaries: vec![
            "no_production_runtime_routing",
            "no_default_model_runner_behavior_change",
            "no_cuda_kernel_replacement",
            "no_onednn_or_torch_runtime_dependency",
            "no_runtime_forward_proof_binary_import",
            "no_raw_artifacts_committed",
            "no_4097_work",
        ],
        next_bounded_step:
            "implement layer0 attention-only validation using runtime RoPE and validation-only projection policy",
    };

    write_json(&cli.output, &status)
}

fn run_k_rope(cli: &Cli) -> Result<()> {
    let k_pre_rope = required_path(&cli.k_pre_rope, "K pre-RoPE")?;
    let k_post_rope_oracle = required_path(&cli.k_post_rope_oracle, "K post-RoPE oracle")?;
    validate_path(k_pre_rope, "K pre-RoPE")?;
    validate_path(k_post_rope_oracle, "K post-RoPE oracle")?;

    let expected_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let (pre, pre_values) = load_k_artifact(k_pre_rope, expected_count, true)?;
    let (post, oracle_values) = load_k_artifact(k_post_rope_oracle, expected_count, false)?;
    let execution = if pre.shape_or_count_matched && post.shape_or_count_matched {
        execute_k_rope(&pre_values, &oracle_values, cli)
    } else {
        KRopeExecution {
            classification: "layer0_validation_k_rope_blocked_by_artifacts",
            metrics: None,
            first_mismatch: None,
            worst_mismatch: None,
            blocker: Some(Blocker {
                kind: "artifact_shape_or_values",
                detail: "K pre/post RoPE artifacts did not expose a supported value key or expected value count",
            }),
        }
    };

    let next_bounded_step = match execution.classification {
        "layer0_validation_k_rope_matches_oracle" => {
            "implement layer0 attention-only validation through raw QK guard using the same runtime RoPE helper"
        }
        "layer0_validation_k_rope_mismatch" => {
            "pin K pre-RoPE artifact identity; then align runtime RoPE table source if artifact identity holds"
        }
        _ => "resolve the reported K RoPE validation blocker",
    };

    let status = KRopeStatus {
        mode: "layer0_validation_runtime_path",
        submode: "k-rope",
        classification: execution.classification,
        implemented: execution.blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        cuda_execution: execution.blocker.is_none(),
        kernel_used: execution
            .blocker
            .is_none()
            .then_some("rotary_embedding_f16_kernel"),
        api_used: if execution.blocker.is_none() {
            "gpt_oss_model_runner::rope_validation::apply_k_rope_f16_validation"
        } else {
            "blocked before launch"
        },
        artifacts: KRopeArtifacts {
            k_pre_rope: pre,
            k_post_rope_oracle: post,
        },
        rope_config: KRopeConfig {
            token_count: cli.token_count,
            kv_heads: cli.kv_heads,
            head_dim: cli.head_dim,
            positions: cli.positions.clone(),
            rope_theta: cli.rope_theta,
            table_source: "gpt_oss_model_runner::rope_validation::build_runtime_rope_tables",
            lane_pairing: "half_split",
            dtype: cli.dtype.clone(),
        },
        metrics: execution.metrics,
        first_mismatch: execution.first_mismatch,
        worst_mismatch: execution.worst_mismatch,
        blocker: execution.blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

struct KRopeExecution {
    classification: &'static str,
    metrics: Option<ComparisonMetrics>,
    first_mismatch: Option<LogicalDiff>,
    worst_mismatch: Option<LogicalDiff>,
    blocker: Option<Blocker>,
}

#[cfg(feature = "cuda")]
fn execute_k_rope(k_pre: &[f32], oracle: &[f32], cli: &Cli) -> KRopeExecution {
    let result = gpt_oss_model_runner::rope_validation::apply_k_rope_f16_validation(
        k_pre,
        cli.token_count,
        cli.kv_heads,
        cli.head_dim,
        cli.rope_theta,
    );
    match result {
        Ok(actual) => {
            let comparison = compare_k_rope(&actual, oracle, cli.kv_heads, cli.head_dim);
            let matched = comparison.metrics.mismatches == 0;
            KRopeExecution {
                classification: if matched {
                    "layer0_validation_k_rope_matches_oracle"
                } else {
                    "layer0_validation_k_rope_mismatch"
                },
                metrics: Some(comparison.metrics),
                first_mismatch: comparison.first_mismatch,
                worst_mismatch: comparison.worst_mismatch,
                blocker: None,
            }
        }
        Err(_) => KRopeExecution {
            classification: "layer0_validation_k_rope_cuda_execution_failed",
            metrics: None,
            first_mismatch: None,
            worst_mismatch: None,
            blocker: Some(Blocker {
                kind: "cuda_execution_failed",
                detail: "runtime RoPE validation helper failed; rerun with stderr for detailed CUDA error",
            }),
        },
    }
}

#[cfg(not(feature = "cuda"))]
fn execute_k_rope(_k_pre: &[f32], _oracle: &[f32], _cli: &Cli) -> KRopeExecution {
    KRopeExecution {
        classification: "layer0_validation_k_rope_cuda_execution_failed",
        metrics: None,
        first_mismatch: None,
        worst_mismatch: None,
        blocker: Some(Blocker {
            kind: "cuda_feature_disabled",
            detail: "k-rope execution requires the cuda feature",
        }),
    }
}

fn validate_path(path: &Path, label: &str) -> Result<()> {
    anyhow::ensure!(
        path.exists(),
        "{} artifact does not exist: {}",
        label,
        path.display()
    );
    anyhow::ensure!(
        path.is_file(),
        "{} artifact is not a file: {}",
        label,
        path.display()
    );
    Ok(())
}

fn required_path<'a>(path: &'a Option<PathBuf>, label: &str) -> Result<&'a Path> {
    path.as_deref()
        .with_context(|| format!("--{} is required", label.replace(' ', "-").to_lowercase()))
}

fn artifact_path(path: &Path) -> ArtifactPath {
    ArtifactPath {
        path: path.display().to_string(),
        exists: path.exists(),
    }
}

fn write_json<T: Serialize>(output: &Path, status: &T) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!("failed to create output directory for {}", output.display())
        })?;
    }
    let payload = serde_json::to_vec_pretty(status)?;
    fs::write(output, &payload).with_context(|| format!("failed to write {}", output.display()))?;
    println!("{}", String::from_utf8_lossy(&payload));
    Ok(())
}

fn load_k_artifact(
    path: &Path,
    expected_count: usize,
    allow_projection_key: bool,
) -> Result<(KArtifactStatus, Vec<f32>)> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let shape = extract_shape(&value);
    let (values, value_key) = extract_values(&value, allow_projection_key);
    let value_count = values.as_ref().map(Vec::len);
    let shape_matches = matches!(shape.as_deref(), Some([74, 512]) | Some([74, 8, 64]));
    let count_matches = value_count == Some(expected_count);
    Ok((
        KArtifactStatus {
            path: path.display().to_string(),
            json_loaded: true,
            shape,
            value_count,
            expected_value_count: expected_count,
            shape_or_count_matched: shape_matches || count_matches,
            value_key,
        },
        values.unwrap_or_default(),
    ))
}

fn extract_shape(value: &Value) -> Option<Vec<usize>> {
    ["shape", "tensor_shape", "output_shape"]
        .iter()
        .find_map(|key| value.get(*key).and_then(json_array_to_usize_vec))
        .or_else(|| {
            value
                .get("metadata")
                .and_then(|metadata| metadata.get("shape"))
                .and_then(json_array_to_usize_vec)
        })
        .or_else(|| {
            let token_count = value.get("token_count")?.as_u64()? as usize;
            let kv_dim = value.get("kv_dim")?.as_u64()? as usize;
            Some(vec![token_count, kv_dim])
        })
}

fn json_array_to_usize_vec(value: &Value) -> Option<Vec<usize>> {
    value
        .as_array()?
        .iter()
        .map(|entry| entry.as_u64().map(|dim| dim as usize))
        .collect()
}

fn extract_values(value: &Value, allow_projection_key: bool) -> (Option<Vec<f32>>, Option<String>) {
    if let Some(values) = value.get("values").and_then(Value::as_array) {
        return (Some(json_values_to_f32(values)), Some("values".to_string()));
    }
    if allow_projection_key {
        let key = "official_projection_outputs.official_module_k_output_f32";
        if let Some(values) = value
            .get("official_projection_outputs")
            .and_then(|outputs| outputs.get("official_module_k_output_f32"))
            .and_then(Value::as_array)
        {
            return (Some(json_values_to_f32(values)), Some(key.to_string()));
        }
    }
    (None, None)
}

fn json_values_to_f32(values: &[Value]) -> Vec<f32> {
    values
        .iter()
        .map(|value| value.as_f64().unwrap_or(f64::NAN) as f32)
        .collect()
}

struct KRopeComparison {
    metrics: ComparisonMetrics,
    first_mismatch: Option<LogicalDiff>,
    worst_mismatch: Option<LogicalDiff>,
}

fn compare_k_rope(
    actual: &[f32],
    expected: &[f32],
    kv_heads: usize,
    head_dim: usize,
) -> KRopeComparison {
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    let mut worst_mismatch = None;

    for (idx, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs_diff = (actual_value - expected_value).abs();
        sum_abs_diff += abs_diff as f64;
        if abs_diff != 0.0 {
            mismatches += 1;
            let diff = logical_diff(
                idx,
                kv_heads,
                head_dim,
                actual_value,
                expected_value,
                abs_diff,
            );
            if first_mismatch.is_none() {
                first_mismatch = Some(LogicalDiff { ..diff });
            }
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
                worst_mismatch = Some(diff);
            }
        }
    }

    let len = actual.len().min(expected.len()).max(1);
    KRopeComparison {
        metrics: ComparisonMetrics {
            max_abs_diff,
            mean_abs_diff: (sum_abs_diff / len as f64) as f32,
            mismatches,
        },
        first_mismatch,
        worst_mismatch,
    }
}

fn logical_diff(
    idx: usize,
    kv_heads: usize,
    head_dim: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
) -> LogicalDiff {
    let per_token = kv_heads * head_dim;
    let token = idx / per_token;
    let feature = idx % per_token;
    LogicalDiff {
        token,
        kv_head: feature / head_dim,
        lane: feature % head_dim,
        actual,
        expected,
        abs_diff,
    }
}
