use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
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

    /// Q pre-RoPE artifact for raw-QK validation.
    #[arg(long)]
    q_pre_rope: Option<PathBuf>,

    /// Official raw scaled QK logits pre-mask oracle artifact.
    #[arg(long)]
    raw_qk_oracle: Option<PathBuf>,

    /// Official masked scaled QK logits pre-softmax oracle artifact.
    #[arg(long)]
    masked_logits_oracle: Option<PathBuf>,

    /// Official attention probabilities post-softmax oracle artifact.
    #[arg(long)]
    attention_probs_oracle: Option<PathBuf>,

    /// Token count for K RoPE validation.
    #[arg(long, default_value_t = 74)]
    token_count: usize,

    /// Query head count for raw-QK validation.
    #[arg(long, default_value_t = 64)]
    query_heads: usize,

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

    /// RoPE scaling type for validation table construction.
    #[arg(long, default_value = "yarn")]
    rope_scaling_type: String,

    /// RoPE scaling factor for YaRN validation tables.
    #[arg(long, default_value_t = 32.0)]
    rope_scaling_factor: f32,

    /// YaRN beta_slow / alpha parameter for validation tables.
    #[arg(long, default_value_t = 1.0)]
    rope_ntk_alpha: f32,

    /// YaRN beta_fast / beta parameter for validation tables.
    #[arg(long, default_value_t = 32.0)]
    rope_ntk_beta: f32,

    /// Original pre-YaRN context length for validation tables.
    #[arg(long, default_value_t = 4096)]
    initial_context_length: usize,

    /// Whether to truncate YaRN low/high ramp bounds.
    #[arg(long, default_value_t = false)]
    rope_scaling_truncate: bool,

    /// Final token index for raw-QK validation.
    #[arg(long, default_value_t = 73)]
    final_token_index: usize,

    /// Raw QK scale.
    #[arg(long, default_value_t = 0.125)]
    scale: f32,

    /// JSON status output path.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Skeleton,
    KRope,
    RawQk,
    AttentionProbs,
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
    table_source: &'static str,
    rope_application_policy: &'static str,
    artifacts: KRopeArtifacts,
    rope_config: KRopeConfig,
    f16_kernel_metrics: Option<ComparisonMetrics>,
    f16_kernel_first_mismatch: Option<LogicalDiff>,
    f16_kernel_worst_mismatch: Option<LogicalDiff>,
    cpu_discriminator: Vec<RopeDiscriminatorResult>,
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
    rope_scaling_type: Option<String>,
    rope_scaling_factor: f32,
    rope_ntk_alpha: f32,
    rope_ntk_beta: f32,
    initial_context_length: usize,
    rope_scaling_truncate: bool,
    table_source: &'static str,
    lane_pairing: &'static str,
    dtype: String,
    output_boundary: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatches: usize,
}

#[derive(Debug, Clone, Serialize)]
struct LogicalDiff {
    token: usize,
    kv_head: usize,
    lane: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct RopeDiscriminatorResult {
    variant: &'static str,
    policy: &'static str,
    metrics: ComparisonMetrics,
    first_mismatch: Option<LogicalDiff>,
    worst_mismatch: Option<LogicalDiff>,
    matched: bool,
}

#[derive(Debug, Serialize)]
struct Blocker {
    kind: &'static str,
    detail: &'static str,
}

#[derive(Debug, Serialize)]
struct RawQkStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    table_source: &'static str,
    rope_policy: &'static str,
    q_source: TensorArtifactStatus,
    k_source: TensorArtifactStatus,
    raw_qk_oracle: TensorArtifactStatus,
    raw_qk_config: RawQkConfig,
    metrics: Option<RawQkMetrics>,
    first_mismatch: Option<RawQkDiff>,
    worst_mismatch: Option<RawQkDiff>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct AttentionProbsStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    table_source: &'static str,
    rope_policy: &'static str,
    q_source: TensorArtifactStatus,
    k_source: TensorArtifactStatus,
    raw_qk_oracle: Option<TensorArtifactStatus>,
    masked_logits_oracle: TensorArtifactStatus,
    attention_probs_oracle: TensorArtifactStatus,
    config: AttentionProbsConfig,
    raw_qk_guard: Option<MatrixComparisonStatus>,
    masked_logits: Option<MaskedLogitsStatus>,
    attention_probs: Option<AttentionProbabilityStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct AttentionProbsConfig {
    token_count: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    final_token_index: usize,
    scale: f32,
    masked_width: usize,
    output_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct MaskedLogitsStatus {
    real_keys: MatrixComparisonStatus,
    sink: MatrixComparisonStatus,
    all: MatrixComparisonStatus,
    sink_source: &'static str,
    masked_real_key_count: usize,
}

#[derive(Debug, Serialize)]
struct AttentionProbabilityStatus {
    real_keys: MatrixComparisonStatus,
    sink: MatrixComparisonStatus,
    all: MatrixComparisonStatus,
    row_sums: RowSumSummary,
    softmax_policy: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixComparisonStatus {
    metrics: MatrixMetrics,
    first_mismatch: Option<MatrixDiff>,
    worst_mismatch: Option<MatrixDiff>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatches: usize,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixDiff {
    q_head: usize,
    column: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct RowSumSummary {
    min: f32,
    max: f32,
    mean: f32,
}

#[derive(Debug, Serialize)]
struct RawQkConfig {
    token_count: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    final_token_index: usize,
    scale: f32,
    output_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct TensorArtifactStatus {
    path: String,
    json_loaded: bool,
    shape: Option<Vec<usize>>,
    value_count: Option<usize>,
    expected_value_counts: Vec<usize>,
    shape_or_count_matched: bool,
    value_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RawQkMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatches: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RawQkDiff {
    q_head: usize,
    token: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
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
        Mode::RawQk => run_raw_qk(&cli),
        Mode::AttentionProbs => run_attention_probs(&cli),
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
            table_source: "unavailable",
            rope_application_policy: "blocked_before_launch",
            f16_kernel_metrics: None,
            f16_kernel_first_mismatch: None,
            f16_kernel_worst_mismatch: None,
            cpu_discriminator: Vec::new(),
        }
    };

    let next_bounded_step = match execution.classification {
        "layer0_validation_k_rope_matches_oracle" => {
            "implement layer0 attention-only validation through raw QK guard using the same runtime RoPE helper"
        }
        "layer0_validation_k_rope_bf16_boundary_matches_oracle" => {
            "use the BF16-boundary RoPE helper in the layer0 attention-only validation path through raw QK"
        }
        "layer0_validation_k_rope_bf16_boundary_mismatch" => {
            "reconcile the BF16-boundary RoPE helper against the official/model application path"
        }
        "layer0_validation_k_rope_dtype_policy_unresolved"
        | "layer0_validation_k_rope_mismatch" => {
            "compare the best discriminator policy against the official/model RoPE call to identify the remaining tensor-operation boundary"
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
            "gpt_oss_model_runner::rope_validation::apply_k_rope_bf16_boundary_validation"
        } else {
            "blocked before launch"
        },
        table_source: execution.table_source,
        rope_application_policy: execution.rope_application_policy,
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
            rope_scaling_type: rope_scaling_type(cli),
            rope_scaling_factor: cli.rope_scaling_factor,
            rope_ntk_alpha: cli.rope_ntk_alpha,
            rope_ntk_beta: cli.rope_ntk_beta,
            initial_context_length: cli.initial_context_length,
            rope_scaling_truncate: cli.rope_scaling_truncate,
            table_source: execution.table_source,
            lane_pairing: "half_split",
            dtype: cli.dtype.clone(),
            output_boundary: "bf16_input_bf16_factors_bf16_rounded_math_bf16_output",
        },
        f16_kernel_metrics: execution.f16_kernel_metrics,
        f16_kernel_first_mismatch: execution.f16_kernel_first_mismatch,
        f16_kernel_worst_mismatch: execution.f16_kernel_worst_mismatch,
        cpu_discriminator: execution.cpu_discriminator,
        metrics: execution.metrics,
        first_mismatch: execution.first_mismatch,
        worst_mismatch: execution.worst_mismatch,
        blocker: execution.blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_raw_qk(cli: &Cli) -> Result<()> {
    let q_pre_rope = required_path(&cli.q_pre_rope, "Q pre-RoPE")?;
    let k_pre_rope = required_path(&cli.k_pre_rope, "K pre-RoPE")?;
    let raw_qk_oracle = required_path(&cli.raw_qk_oracle, "raw QK oracle")?;
    validate_path(q_pre_rope, "Q pre-RoPE")?;
    validate_path(k_pre_rope, "K pre-RoPE")?;
    validate_path(raw_qk_oracle, "raw QK oracle")?;

    let q_full_count = cli.token_count * cli.query_heads * cli.head_dim;
    let q_final_count = cli.query_heads * cli.head_dim;
    let k_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let raw_qk_count = cli.query_heads * cli.token_count;

    let (q_status, q_values) = load_tensor_artifact(
        q_pre_rope,
        &[q_full_count, q_final_count],
        &["values", "local_q_pre_rope_f32"],
    )?;
    let (k_status, k_values) = load_tensor_artifact(
        k_pre_rope,
        &[k_count],
        &[
            "values",
            "official_projection_outputs.official_module_k_output_f32",
        ],
    )?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(raw_qk_oracle, &[raw_qk_count], &["values"])?;

    let execution = if !q_status.shape_or_count_matched {
        RawQkExecution::blocked(
            "layer0_validation_raw_qk_blocked_by_q_pre_rope",
            "q_pre_rope_artifact",
            "Q pre-RoPE artifact did not expose a supported value key or expected value count",
        )
    } else if !k_status.shape_or_count_matched || !oracle_status.shape_or_count_matched {
        RawQkExecution::blocked(
            "layer0_validation_raw_qk_blocked_by_artifacts",
            "k_or_raw_qk_artifact",
            "K pre-RoPE or raw-QK oracle artifact did not expose a supported value key or expected value count",
        )
    } else {
        execute_raw_qk(&q_values, &k_values, &oracle_values, cli)
    };

    let next_bounded_step = match execution.classification {
        "layer0_validation_raw_qk_matches_oracle" => {
            "extend attention-only validation to mask and softmax probability boundary"
        }
        "layer0_validation_raw_qk_mismatch" => {
            "localize raw-QK mismatch between Q source, K source, RoPE, and score rounding"
        }
        _ => "resolve the reported raw-QK validation blocker",
    };

    let status = RawQkStatus {
        mode: "layer0_validation_runtime_path",
        submode: "raw-qk",
        classification: execution.classification,
        implemented: execution.blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        table_source: execution.table_source,
        rope_policy: "bf16_input_bf16_factors_bf16_rounded_math_bf16_output",
        q_source: q_status,
        k_source: k_status,
        raw_qk_oracle: oracle_status,
        raw_qk_config: RawQkConfig {
            token_count: cli.token_count,
            query_heads: cli.query_heads,
            kv_heads: cli.kv_heads,
            head_dim: cli.head_dim,
            final_token_index: cli.final_token_index,
            scale: cli.scale,
            output_boundary: "bf16_rounded_score_before_mask",
        },
        metrics: execution.metrics,
        first_mismatch: execution.first_mismatch,
        worst_mismatch: execution.worst_mismatch,
        blocker: execution.blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_attention_probs(cli: &Cli) -> Result<()> {
    let q_pre_rope = required_path(&cli.q_pre_rope, "Q pre-RoPE")?;
    let k_pre_rope = required_path(&cli.k_pre_rope, "K pre-RoPE")?;
    let masked_logits_oracle = required_path(&cli.masked_logits_oracle, "masked logits oracle")?;
    let attention_probs_oracle =
        required_path(&cli.attention_probs_oracle, "attention probs oracle")?;
    validate_path(q_pre_rope, "Q pre-RoPE")?;
    validate_path(k_pre_rope, "K pre-RoPE")?;
    validate_path(masked_logits_oracle, "masked logits oracle")?;
    validate_path(attention_probs_oracle, "attention probs oracle")?;
    if let Some(raw_qk_oracle) = &cli.raw_qk_oracle {
        validate_path(raw_qk_oracle, "raw QK oracle")?;
    }

    let q_full_count = cli.token_count * cli.query_heads * cli.head_dim;
    let q_final_count = cli.query_heads * cli.head_dim;
    let k_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let raw_qk_count = cli.query_heads * cli.token_count;
    let masked_width = cli.token_count + 1;
    let masked_count = cli.query_heads * masked_width;

    let (q_status, q_values) = load_tensor_artifact(
        q_pre_rope,
        &[q_full_count, q_final_count],
        &["values", "local_q_pre_rope_f32"],
    )?;
    let (k_status, k_values) = load_tensor_artifact(
        k_pre_rope,
        &[k_count],
        &[
            "values",
            "official_projection_outputs.official_module_k_output_f32",
        ],
    )?;
    let (raw_oracle_status, raw_oracle_values) = match &cli.raw_qk_oracle {
        Some(path) => {
            let (status, values) = load_tensor_artifact(path, &[raw_qk_count], &["values"])?;
            (Some(status), Some(values))
        }
        None => (None, None),
    };
    let (masked_status, masked_values) =
        load_tensor_artifact(masked_logits_oracle, &[masked_count], &["values"])?;
    let (probs_status, probs_values) =
        load_tensor_artifact(attention_probs_oracle, &[masked_count], &["values"])?;

    let execution = if !q_status.shape_or_count_matched {
        AttentionProbsExecution::blocked(
            "layer0_validation_attention_probs_blocked_by_artifacts",
            "q_pre_rope_artifact",
            "Q pre-RoPE artifact did not expose a supported value key or expected value count",
        )
    } else if !k_status.shape_or_count_matched
        || !masked_status.shape_or_count_matched
        || !probs_status.shape_or_count_matched
        || raw_oracle_status
            .as_ref()
            .map(|status| !status.shape_or_count_matched)
            .unwrap_or(false)
    {
        AttentionProbsExecution::blocked(
            "layer0_validation_attention_probs_blocked_by_artifacts",
            "attention_artifacts",
            "K pre-RoPE, raw-QK, masked-logits, or attention-probs artifact did not expose a supported value key or expected value count",
        )
    } else {
        execute_attention_probs(
            &q_values,
            &k_values,
            raw_oracle_values.as_deref(),
            &masked_values,
            &probs_values,
            cli,
        )
    };

    let next_bounded_step = match execution.classification {
        "layer0_validation_attention_probs_match_oracle" => {
            "extend attention-only validation to weighted V using the exact attention probabilities"
        }
        "layer0_validation_masked_logits_mismatch" => {
            "localize masked-logit mismatch between raw-QK real keys and sink-source handling"
        }
        "layer0_validation_attention_probs_mismatch"
        | "layer0_validation_attention_probs_softmax_policy_unresolved" => {
            "reconcile softmax dtype/output policy against the official attention probability boundary"
        }
        _ => "resolve the reported attention probability validation blocker",
    };

    let status = AttentionProbsStatus {
        mode: "layer0_validation_runtime_path",
        submode: "attention-probs",
        classification: execution.classification,
        implemented: execution.blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        table_source: execution.table_source,
        rope_policy: "bf16_input_bf16_factors_bf16_rounded_math_bf16_output",
        q_source: q_status,
        k_source: k_status,
        raw_qk_oracle: raw_oracle_status,
        masked_logits_oracle: masked_status,
        attention_probs_oracle: probs_status,
        config: AttentionProbsConfig {
            token_count: cli.token_count,
            query_heads: cli.query_heads,
            kv_heads: cli.kv_heads,
            head_dim: cli.head_dim,
            final_token_index: cli.final_token_index,
            scale: cli.scale,
            masked_width,
            output_boundary: "bf16_masked_logits_and_bf16_attention_probabilities",
        },
        raw_qk_guard: execution.raw_qk_guard,
        masked_logits: execution.masked_logits,
        attention_probs: execution.attention_probs,
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
    table_source: &'static str,
    rope_application_policy: &'static str,
    f16_kernel_metrics: Option<ComparisonMetrics>,
    f16_kernel_first_mismatch: Option<LogicalDiff>,
    f16_kernel_worst_mismatch: Option<LogicalDiff>,
    cpu_discriminator: Vec<RopeDiscriminatorResult>,
}

struct RawQkExecution {
    classification: &'static str,
    metrics: Option<RawQkMetrics>,
    first_mismatch: Option<RawQkDiff>,
    worst_mismatch: Option<RawQkDiff>,
    blocker: Option<Blocker>,
    table_source: &'static str,
}

struct RawQkValues {
    values: Vec<f32>,
    table_source: &'static str,
}

struct AttentionProbsExecution {
    classification: &'static str,
    table_source: &'static str,
    raw_qk_guard: Option<MatrixComparisonStatus>,
    masked_logits: Option<MaskedLogitsStatus>,
    attention_probs: Option<AttentionProbabilityStatus>,
    blocker: Option<Blocker>,
}

impl AttentionProbsExecution {
    fn blocked(
        classification: &'static str,
        kind: &'static str,
        detail: &'static str,
    ) -> AttentionProbsExecution {
        AttentionProbsExecution {
            classification,
            table_source: "unavailable",
            raw_qk_guard: None,
            masked_logits: None,
            attention_probs: None,
            blocker: Some(Blocker { kind, detail }),
        }
    }
}

impl RawQkExecution {
    fn blocked(
        classification: &'static str,
        kind: &'static str,
        detail: &'static str,
    ) -> RawQkExecution {
        RawQkExecution {
            classification,
            metrics: None,
            first_mismatch: None,
            worst_mismatch: None,
            blocker: Some(Blocker { kind, detail }),
            table_source: "unavailable",
        }
    }
}

#[cfg(feature = "cuda")]
fn execute_k_rope(k_pre: &[f32], oracle: &[f32], cli: &Cli) -> KRopeExecution {
    let config = validation_rope_config(cli);
    let (cos_table, sin_table, table_source) =
        gpt_oss_model_runner::rope_validation::build_validation_rope_tables_from_config(
            &config,
            cli.token_count,
        );
    let cpu_discriminator = run_rope_cpu_discriminator(k_pre, oracle, &cos_table, &sin_table, cli);
    let best_cpu = cpu_discriminator.iter().min_by(|left, right| {
        left.metrics
            .mismatches
            .cmp(&right.metrics.mismatches)
            .then_with(|| {
                left.metrics
                    .max_abs_diff
                    .total_cmp(&right.metrics.max_abs_diff)
            })
            .then_with(|| {
                left.metrics
                    .mean_abs_diff
                    .total_cmp(&right.metrics.mean_abs_diff)
            })
    });

    let f16_result = gpt_oss_model_runner::rope_validation::apply_k_rope_f16_validation_with_config(
        k_pre,
        cli.token_count,
        cli.kv_heads,
        &config,
    );
    let (f16_kernel_metrics, f16_kernel_first_mismatch, f16_kernel_worst_mismatch) =
        match f16_result {
            Ok((actual, _)) => {
                let comparison = compare_k_rope(&actual, oracle, cli.kv_heads, cli.head_dim);
                (
                    Some(comparison.metrics),
                    comparison.first_mismatch,
                    comparison.worst_mismatch,
                )
            }
            Err(_) => (None, None, None),
        };

    let result = gpt_oss_model_runner::rope_validation::apply_k_rope_bf16_boundary_validation(
        k_pre,
        cli.token_count,
        cli.kv_heads,
        &config,
    );
    match result {
        Ok((actual, boundary_table_source)) => {
            let comparison = compare_k_rope(&actual, oracle, cli.kv_heads, cli.head_dim);
            let matched = comparison.metrics.mismatches == 0;
            let classification = if matched {
                "layer0_validation_k_rope_bf16_boundary_matches_oracle"
            } else {
                "layer0_validation_k_rope_bf16_boundary_mismatch"
            };
            KRopeExecution {
                classification,
                metrics: Some(comparison.metrics),
                first_mismatch: comparison.first_mismatch,
                worst_mismatch: comparison.worst_mismatch,
                blocker: None,
                table_source: boundary_table_source,
                rope_application_policy: "bf16_input_bf16_factors_bf16_rounded_math_bf16_output",
                f16_kernel_metrics,
                f16_kernel_first_mismatch,
                f16_kernel_worst_mismatch,
                cpu_discriminator,
            }
        }
        Err(_) => KRopeExecution {
            classification: "layer0_validation_k_rope_bf16_boundary_blocked",
            metrics: best_cpu.map(|result| result.metrics.clone()),
            first_mismatch: best_cpu.and_then(|result| result.first_mismatch.clone()),
            worst_mismatch: best_cpu.and_then(|result| result.worst_mismatch.clone()),
            blocker: Some(Blocker {
                kind: "bf16_boundary_validation_failed",
                detail: "BF16-boundary RoPE validation helper failed; rerun with stderr for detailed error",
            }),
            table_source,
            rope_application_policy: "bf16_boundary_validation_failed",
            f16_kernel_metrics,
            f16_kernel_first_mismatch,
            f16_kernel_worst_mismatch,
            cpu_discriminator,
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
        table_source: "unavailable",
        rope_application_policy: "cuda_feature_disabled",
        f16_kernel_metrics: None,
        f16_kernel_first_mismatch: None,
        f16_kernel_worst_mismatch: None,
        cpu_discriminator: Vec::new(),
    }
}

fn validation_rope_config(cli: &Cli) -> gpt_oss_model_runner::runner::ModelRunnerConfig {
    let mut config = gpt_oss_model_runner::runner::ModelRunnerConfig::default();
    config.head_dim = cli.head_dim;
    config.rope_theta = cli.rope_theta;
    config.rope_scaling_type = rope_scaling_type(cli);
    config.rope_scaling_factor = cli.rope_scaling_factor;
    config.rope_ntk_alpha = cli.rope_ntk_alpha;
    config.rope_ntk_beta = cli.rope_ntk_beta;
    config.initial_context_length = cli.initial_context_length;
    config.rope_scaling_truncate = cli.rope_scaling_truncate;
    config
}

fn execute_raw_qk(q_pre: &[f32], k_pre: &[f32], oracle: &[f32], cli: &Cli) -> RawQkExecution {
    let raw_values = match compute_raw_qk_values(q_pre, k_pre, cli) {
        Ok(values) => values,
        Err(execution) => return execution,
    };
    let comparison = compare_raw_qk(&raw_values.values, oracle, cli);
    RawQkExecution {
        classification: if comparison.metrics.mismatches == 0 {
            "layer0_validation_raw_qk_matches_oracle"
        } else {
            "layer0_validation_raw_qk_mismatch"
        },
        metrics: Some(comparison.metrics),
        first_mismatch: comparison.first_mismatch,
        worst_mismatch: comparison.worst_mismatch,
        blocker: None,
        table_source: raw_values.table_source,
    }
}

fn compute_raw_qk_values(
    q_pre: &[f32],
    k_pre: &[f32],
    cli: &Cli,
) -> Result<RawQkValues, RawQkExecution> {
    let config = validation_rope_config(cli);
    let q_rope_input = q_rope_input_for_helper(q_pre, cli);
    let q_result = gpt_oss_model_runner::rope_validation::apply_k_rope_bf16_boundary_validation(
        &q_rope_input,
        q_rope_token_count(q_pre, cli),
        cli.query_heads,
        &config,
    );
    let k_result = gpt_oss_model_runner::rope_validation::apply_k_rope_bf16_boundary_validation(
        k_pre,
        cli.token_count,
        cli.kv_heads,
        &config,
    );
    let (q_post, q_table_source) = match q_result {
        Ok(result) => result,
        Err(_) => {
            return Err(RawQkExecution::blocked(
                "layer0_validation_raw_qk_execution_failed",
                "q_rope_failed",
                "BF16-boundary Q RoPE helper failed",
            ));
        }
    };
    let (k_post, k_table_source) = match k_result {
        Ok(result) => result,
        Err(_) => {
            return Err(RawQkExecution::blocked(
                "layer0_validation_raw_qk_execution_failed",
                "k_rope_failed",
                "BF16-boundary K RoPE helper failed",
            ));
        }
    };
    let q_final = q_final_post_rope_slice(&q_post, q_pre.len(), cli);
    let actual = compute_raw_qk(q_final, &k_post, cli);
    Ok(RawQkValues {
        values: actual,
        table_source: if q_table_source == k_table_source {
            q_table_source
        } else {
            "mixed_table_sources"
        },
    })
}

fn execute_attention_probs(
    q_pre: &[f32],
    k_pre: &[f32],
    raw_oracle: Option<&[f32]>,
    masked_oracle: &[f32],
    probs_oracle: &[f32],
    cli: &Cli,
) -> AttentionProbsExecution {
    let raw_values = match compute_raw_qk_values(q_pre, k_pre, cli) {
        Ok(values) => values,
        Err(execution) => {
            return AttentionProbsExecution::blocked(
                "layer0_validation_attention_probs_execution_failed",
                execution
                    .blocker
                    .map(|blocker| blocker.kind)
                    .unwrap_or("raw_qk_execution_failed"),
                "raw-QK generation failed before mask/softmax validation",
            );
        }
    };
    let raw_qk_guard = raw_oracle.map(|oracle| {
        compare_matrix(
            &raw_values.values,
            oracle,
            cli.query_heads,
            cli.token_count,
            MatrixSelection::All,
        )
    });

    let masked_width = cli.token_count + 1;
    let masked_logits = build_masked_logits_from_raw_qk(
        &raw_values.values,
        masked_oracle,
        cli.query_heads,
        cli.token_count,
    );
    let masked_real_keys = compare_matrix(
        &masked_logits,
        masked_oracle,
        cli.query_heads,
        masked_width,
        MatrixSelection::Columns {
            start: 0,
            end: cli.token_count,
        },
    );
    let masked_sink = compare_matrix(
        &masked_logits,
        masked_oracle,
        cli.query_heads,
        masked_width,
        MatrixSelection::Column(cli.token_count),
    );
    let masked_all = compare_matrix(
        &masked_logits,
        masked_oracle,
        cli.query_heads,
        masked_width,
        MatrixSelection::All,
    );
    let masked_status = MaskedLogitsStatus {
        real_keys: masked_real_keys,
        sink: masked_sink,
        all: masked_all,
        sink_source: "official_masked_logits_oracle_sink_column",
        masked_real_key_count: 0,
    };

    if masked_status.all.metrics.mismatches != 0 {
        return AttentionProbsExecution {
            classification: "layer0_validation_masked_logits_mismatch",
            table_source: raw_values.table_source,
            raw_qk_guard,
            masked_logits: Some(masked_status),
            attention_probs: None,
            blocker: None,
        };
    }

    let probs = softmax_rows_bf16_output(&masked_logits, cli.query_heads, masked_width);
    let probs_real_keys = compare_matrix(
        &probs,
        probs_oracle,
        cli.query_heads,
        masked_width,
        MatrixSelection::Columns {
            start: 0,
            end: cli.token_count,
        },
    );
    let probs_sink = compare_matrix(
        &probs,
        probs_oracle,
        cli.query_heads,
        masked_width,
        MatrixSelection::Column(cli.token_count),
    );
    let probs_all = compare_matrix(
        &probs,
        probs_oracle,
        cli.query_heads,
        masked_width,
        MatrixSelection::All,
    );
    let probs_status = AttentionProbabilityStatus {
        real_keys: probs_real_keys,
        sink: probs_sink,
        all: probs_all,
        row_sums: row_sum_summary(&probs, cli.query_heads, masked_width),
        softmax_policy: "f32_subtract_max_exp_sum_bf16_output",
    };
    let classification = if probs_status.all.metrics.mismatches == 0 {
        "layer0_validation_attention_probs_match_oracle"
    } else {
        "layer0_validation_attention_probs_mismatch"
    };

    AttentionProbsExecution {
        classification,
        table_source: raw_values.table_source,
        raw_qk_guard,
        masked_logits: Some(masked_status),
        attention_probs: Some(probs_status),
        blocker: None,
    }
}

fn q_rope_token_count(q_pre: &[f32], cli: &Cli) -> usize {
    if q_pre.len() == cli.query_heads * cli.head_dim {
        cli.final_token_index + 1
    } else {
        cli.token_count
    }
}

fn q_rope_input_for_helper(q_pre: &[f32], cli: &Cli) -> Vec<f32> {
    let q_final_count = cli.query_heads * cli.head_dim;
    if q_pre.len() != q_final_count {
        return q_pre.to_vec();
    }
    let mut padded = vec![0.0f32; (cli.final_token_index + 1) * q_final_count];
    let start = cli.final_token_index * q_final_count;
    padded[start..start + q_final_count].copy_from_slice(q_pre);
    padded
}

fn q_final_post_rope_slice<'a>(q_post: &'a [f32], original_q_len: usize, cli: &Cli) -> &'a [f32] {
    let q_final_count = cli.query_heads * cli.head_dim;
    if original_q_len == q_final_count {
        let start = cli.final_token_index * q_final_count;
        &q_post[start..start + q_final_count]
    } else {
        let start = cli.final_token_index * q_final_count;
        &q_post[start..start + q_final_count]
    }
}

fn compute_raw_qk(q_final: &[f32], k_post: &[f32], cli: &Cli) -> Vec<f32> {
    let mut output = vec![0.0f32; cli.query_heads * cli.token_count];
    for q_head in 0..cli.query_heads {
        let kv_head = q_head / (cli.query_heads / cli.kv_heads);
        for token in 0..cli.token_count {
            let mut sum = 0.0f32;
            let q_base = q_head * cli.head_dim;
            let k_base = (token * cli.kv_heads + kv_head) * cli.head_dim;
            for lane in 0..cli.head_dim {
                sum += q_final[q_base + lane] * k_post[k_base + lane];
            }
            output[q_head * cli.token_count + token] = round_bf16(sum * cli.scale);
        }
    }
    output
}

fn build_masked_logits_from_raw_qk(
    raw_qk: &[f32],
    masked_oracle: &[f32],
    query_heads: usize,
    token_count: usize,
) -> Vec<f32> {
    let masked_width = token_count + 1;
    let mut output = vec![0.0f32; query_heads * masked_width];
    for q_head in 0..query_heads {
        let raw_base = q_head * token_count;
        let masked_base = q_head * masked_width;
        output[masked_base..masked_base + token_count]
            .copy_from_slice(&raw_qk[raw_base..raw_base + token_count]);
        output[masked_base + token_count] = masked_oracle[masked_base + token_count];
    }
    output
}

fn softmax_rows_bf16_output(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let row_values = &values[base..base + cols];
        let max_value = row_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut exp_values = vec![0.0f32; cols];
        for (idx, value) in row_values.iter().copied().enumerate() {
            let exp_value = (value - max_value).exp();
            exp_values[idx] = exp_value;
            sum += exp_value;
        }
        for (idx, exp_value) in exp_values.into_iter().enumerate() {
            output[base + idx] = round_bf16(exp_value / sum);
        }
    }
    output
}

fn row_sum_summary(values: &[f32], rows: usize, cols: usize) -> RowSumSummary {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for row in 0..rows {
        let base = row * cols;
        let row_sum = values[base..base + cols].iter().sum::<f32>();
        min = min.min(row_sum);
        max = max.max(row_sum);
        sum += row_sum as f64;
    }
    RowSumSummary {
        min,
        max,
        mean: (sum / rows.max(1) as f64) as f32,
    }
}

fn rope_scaling_type(cli: &Cli) -> Option<String> {
    let value = cli.rope_scaling_type.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(value.to_string())
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

fn load_tensor_artifact(
    path: &Path,
    expected_counts: &[usize],
    value_keys: &[&str],
) -> Result<(TensorArtifactStatus, Vec<f32>)> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let shape = extract_shape(&value);
    let (values, value_key) = extract_values_by_keys(&value, value_keys);
    let value_count = values.as_ref().map(Vec::len);
    let count_matches = value_count
        .map(|count| expected_counts.contains(&count))
        .unwrap_or(false);
    Ok((
        TensorArtifactStatus {
            path: path.display().to_string(),
            json_loaded: true,
            shape,
            value_count,
            expected_value_counts: expected_counts.to_vec(),
            shape_or_count_matched: count_matches,
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

fn extract_values_by_keys(value: &Value, keys: &[&str]) -> (Option<Vec<f32>>, Option<String>) {
    for key in keys {
        if let Some(values) = value.pointer(&format!("/{}", key.replace('.', "/"))) {
            if let Some(array) = values.as_array() {
                return (Some(json_values_to_f32(array)), Some((*key).to_string()));
            }
        }
        let mut current = value;
        let mut found = true;
        for part in key.split('.') {
            if let Some(next) = current.get(part) {
                current = next;
            } else {
                found = false;
                break;
            }
        }
        if found {
            if let Some(array) = current.as_array() {
                return (Some(json_values_to_f32(array)), Some((*key).to_string()));
            }
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

fn run_rope_cpu_discriminator(
    k_pre: &[f32],
    oracle: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    cli: &Cli,
) -> Vec<RopeDiscriminatorResult> {
    let variants = [
        (
            "f32_input_f32_factors_f32_math_bf16_output",
            "f32 input, f32 cos/sin, f32 math, BF16 output",
            RopeCpuPolicy::F32MathBf16Output,
        ),
        (
            "bf16_input_bf16_factors_bf16ish_math_bf16_output",
            "BF16 input, BF16 cos/sin, BF16-rounded multiply/add, BF16 output",
            RopeCpuPolicy::Bf16ishMath,
        ),
        (
            "bf16_input_bf16_factors_f32_math_bf16_output",
            "BF16 input, BF16 cos/sin widened to f32 math, BF16 output",
            RopeCpuPolicy::Bf16FactorsF32Math,
        ),
        (
            "f16_input_f16_factors_f16ish_output",
            "F16 input, F16 cos/sin widened to f32 math, F16 output",
            RopeCpuPolicy::F16FactorsF16Output,
        ),
    ];

    variants
        .into_iter()
        .map(|(variant, policy, cpu_policy)| {
            let actual = apply_rope_cpu_policy(k_pre, cos_table, sin_table, cli, cpu_policy);
            let comparison = compare_k_rope(&actual, oracle, cli.kv_heads, cli.head_dim);
            RopeDiscriminatorResult {
                variant,
                policy,
                matched: comparison.metrics.mismatches == 0,
                metrics: comparison.metrics,
                first_mismatch: comparison.first_mismatch,
                worst_mismatch: comparison.worst_mismatch,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum RopeCpuPolicy {
    F32MathBf16Output,
    Bf16ishMath,
    Bf16FactorsF32Math,
    F16FactorsF16Output,
}

fn apply_rope_cpu_policy(
    k_pre: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    cli: &Cli,
    policy: RopeCpuPolicy,
) -> Vec<f32> {
    let half_dim = cli.head_dim / 2;
    let mut output = vec![0.0f32; k_pre.len()];
    for token in 0..cli.token_count {
        for kv_head in 0..cli.kv_heads {
            let head_base = (token * cli.kv_heads + kv_head) * cli.head_dim;
            let table_base = token * half_dim;
            for lane in 0..half_dim {
                let x1 = k_pre[head_base + lane];
                let x2 = k_pre[head_base + half_dim + lane];
                let cos = cos_table[table_base + lane];
                let sin = sin_table[table_base + lane];
                let (out1, out2) = apply_rope_pair(policy, x1, x2, cos, sin);
                output[head_base + lane] = out1;
                output[head_base + half_dim + lane] = out2;
            }
        }
    }
    output
}

fn apply_rope_pair(policy: RopeCpuPolicy, x1: f32, x2: f32, cos: f32, sin: f32) -> (f32, f32) {
    match policy {
        RopeCpuPolicy::F32MathBf16Output => {
            let out1 = x1 * cos - x2 * sin;
            let out2 = x2 * cos + x1 * sin;
            (round_bf16(out1), round_bf16(out2))
        }
        RopeCpuPolicy::Bf16ishMath => {
            let x1 = round_bf16(x1);
            let x2 = round_bf16(x2);
            let cos = round_bf16(cos);
            let sin = round_bf16(sin);
            let out1 = round_bf16(round_bf16(x1 * cos) - round_bf16(x2 * sin));
            let out2 = round_bf16(round_bf16(x2 * cos) + round_bf16(x1 * sin));
            (out1, out2)
        }
        RopeCpuPolicy::Bf16FactorsF32Math => {
            let x1 = round_bf16(x1);
            let x2 = round_bf16(x2);
            let cos = round_bf16(cos);
            let sin = round_bf16(sin);
            let out1 = x1 * cos - x2 * sin;
            let out2 = x2 * cos + x1 * sin;
            (round_bf16(out1), round_bf16(out2))
        }
        RopeCpuPolicy::F16FactorsF16Output => {
            let x1 = round_f16(x1);
            let x2 = round_f16(x2);
            let cos = round_f16(cos);
            let sin = round_f16(sin);
            let out1 = x1 * cos - x2 * sin;
            let out2 = x2 * cos + x1 * sin;
            (round_f16(out1), round_f16(out2))
        }
    }
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn round_f16(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

struct KRopeComparison {
    metrics: ComparisonMetrics,
    first_mismatch: Option<LogicalDiff>,
    worst_mismatch: Option<LogicalDiff>,
}

struct RawQkComparison {
    metrics: RawQkMetrics,
    first_mismatch: Option<RawQkDiff>,
    worst_mismatch: Option<RawQkDiff>,
}

enum MatrixSelection {
    All,
    Column(usize),
    Columns { start: usize, end: usize },
}

impl MatrixSelection {
    fn includes(&self, column: usize) -> bool {
        match *self {
            MatrixSelection::All => true,
            MatrixSelection::Column(selected) => column == selected,
            MatrixSelection::Columns { start, end } => (start..end).contains(&column),
        }
    }
}

fn compare_matrix(
    actual: &[f32],
    expected: &[f32],
    rows: usize,
    cols: usize,
    selection: MatrixSelection,
) -> MatrixComparisonStatus {
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut compared = 0usize;
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    let mut worst_mismatch = None;

    for row in 0..rows {
        for col in 0..cols {
            if !selection.includes(col) {
                continue;
            }
            let idx = row * cols + col;
            let actual_value = actual[idx];
            let expected_value = expected[idx];
            let abs_diff = (actual_value - expected_value).abs();
            compared += 1;
            sum_abs_diff += abs_diff as f64;
            if abs_diff != 0.0 {
                mismatches += 1;
                let diff = MatrixDiff {
                    q_head: row,
                    column: col,
                    actual: actual_value,
                    expected: expected_value,
                    abs_diff,
                };
                if first_mismatch.is_none() {
                    first_mismatch = Some(diff.clone());
                }
                if abs_diff > max_abs_diff {
                    max_abs_diff = abs_diff;
                    worst_mismatch = Some(diff);
                }
            }
        }
    }

    MatrixComparisonStatus {
        metrics: MatrixMetrics {
            max_abs_diff,
            mean_abs_diff: (sum_abs_diff / compared.max(1) as f64) as f32,
            mismatches,
        },
        first_mismatch,
        worst_mismatch,
    }
}

fn compare_raw_qk(actual: &[f32], expected: &[f32], cli: &Cli) -> RawQkComparison {
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
            let diff = RawQkDiff {
                q_head: idx / cli.token_count,
                token: idx % cli.token_count,
                actual: actual_value,
                expected: expected_value,
                abs_diff,
            };
            if first_mismatch.is_none() {
                first_mismatch = Some(diff.clone());
            }
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
                worst_mismatch = Some(diff);
            }
        }
    }

    let len = actual.len().min(expected.len()).max(1);
    RawQkComparison {
        metrics: RawQkMetrics {
            max_abs_diff,
            mean_abs_diff: (sum_abs_diff / len as f64) as f32,
            mismatches,
        },
        first_mismatch,
        worst_mismatch,
    }
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
