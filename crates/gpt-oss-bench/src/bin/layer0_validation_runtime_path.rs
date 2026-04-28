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

    /// Attention probabilities artifact for weighted-V validation.
    #[arg(long)]
    attention_probs: Option<PathBuf>,

    /// V values artifact for weighted-V validation.
    #[arg(long)]
    v_values: Option<PathBuf>,

    /// Official weighted V oracle artifact.
    #[arg(long)]
    weighted_v_oracle: Option<PathBuf>,

    /// Weighted V artifact for attention o-proj validation.
    #[arg(long)]
    weighted_v: Option<PathBuf>,

    /// Attention o_proj weight artifact for o-proj validation.
    #[arg(long)]
    oproj_weight: Option<PathBuf>,

    /// Attention o_proj bias artifact for o-proj validation.
    #[arg(long)]
    oproj_bias: Option<PathBuf>,

    /// Official attention o_proj oracle artifact.
    #[arg(long)]
    oproj_oracle: Option<PathBuf>,

    /// Residual input artifact for attention residual validation.
    #[arg(long)]
    residual_input: Option<PathBuf>,

    /// Official attention residual-add oracle artifact.
    #[arg(long)]
    attention_residual_oracle: Option<PathBuf>,

    /// Token count for K RoPE validation.
    #[arg(long, default_value_t = 74)]
    token_count: usize,

    /// Query head count for raw-QK validation.
    #[arg(long, default_value_t = 64)]
    query_heads: usize,

    /// Number of grouped-query K/V heads for K RoPE validation.
    #[arg(long, default_value_t = 8)]
    kv_heads: usize,

    /// Number of query heads per K/V head for GQA.
    #[arg(long, default_value_t = 8)]
    heads_per_kv: usize,

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

    /// Sink column position for attention probabilities.
    #[arg(long, default_value_t = 74)]
    sink_position: usize,

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
    WeightedV,
    AttentionOproj,
    AttentionOprojPolicy,
    AttentionResidual,
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
struct WeightedVStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    sink_dropped: bool,
    gqa: GqaConfig,
    attention_probs: TensorArtifactStatus,
    v_values: TensorArtifactStatus,
    weighted_v_oracle: TensorArtifactStatus,
    f32_metric: Option<MatrixComparisonStatus>,
    bf16_metric: Option<MatrixComparisonStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct AttentionOprojStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    input_source: &'static str,
    o_proj_policy: &'static str,
    weighted_v: TensorArtifactStatus,
    oproj_weight: TensorArtifactStatus,
    oproj_bias: TensorArtifactStatus,
    oproj_oracle: TensorArtifactStatus,
    metrics: Option<HiddenComparisonStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct AttentionOprojPolicyStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    residual_classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    input_source: &'static str,
    weighted_v: TensorArtifactStatus,
    oproj_weight: TensorArtifactStatus,
    oproj_bias: TensorArtifactStatus,
    oproj_oracle: TensorArtifactStatus,
    residual_input: TensorArtifactStatus,
    attention_residual_oracle: TensorArtifactStatus,
    variants: Vec<AttentionOprojVariantStatus>,
    best_variant: AttentionOprojVariantStatus,
    lane_1587_trace: LaneTrace,
    pytorch_bf16_reference: ExternalReferenceStatus,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct AttentionResidualStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    input_source: &'static str,
    o_proj_policy: &'static str,
    residual_policy: &'static str,
    weighted_v: TensorArtifactStatus,
    oproj_weight: TensorArtifactStatus,
    oproj_bias: TensorArtifactStatus,
    residual_input: TensorArtifactStatus,
    attention_residual_oracle: TensorArtifactStatus,
    o_proj_metric: Option<HiddenComparisonStatus>,
    residual_metric: Option<HiddenComparisonStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct AttentionOprojVariantStatus {
    name: &'static str,
    policy: &'static str,
    oproj_metric: HiddenComparisonStatus,
    residual_metric: HiddenComparisonStatus,
    lane_1587: LaneTrace,
}

#[derive(Debug, Clone, Serialize)]
struct LaneTrace {
    hidden_lane: usize,
    oproj_actual: f32,
    oproj_expected: f32,
    oproj_abs_diff: f32,
    residual_actual: f32,
    residual_expected: f32,
    residual_abs_diff: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ExternalReferenceStatus {
    available: bool,
    policy: &'static str,
    oproj_metric: HiddenMetrics,
    note: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct HiddenComparisonStatus {
    metrics: HiddenMetrics,
    first_mismatch: Option<HiddenDiff>,
    worst_mismatch: Option<HiddenDiff>,
}

#[derive(Debug, Clone, Serialize)]
struct HiddenMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatches: usize,
}

#[derive(Debug, Clone, Serialize)]
struct HiddenDiff {
    hidden_lane: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct GqaConfig {
    query_heads: usize,
    kv_heads: usize,
    heads_per_kv: usize,
    head_dim: usize,
    token_count: usize,
    sink_position: usize,
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
        Mode::WeightedV => run_weighted_v(&cli),
        Mode::AttentionOproj => run_attention_oproj(&cli),
        Mode::AttentionOprojPolicy => run_attention_oproj_policy(&cli),
        Mode::AttentionResidual => run_attention_residual(&cli),
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

fn run_weighted_v(cli: &Cli) -> Result<()> {
    let attention_probs = required_path(&cli.attention_probs, "attention probs")?;
    let v_values = required_path(&cli.v_values, "V values")?;
    let weighted_v_oracle = required_path(&cli.weighted_v_oracle, "weighted V oracle")?;
    validate_path(attention_probs, "attention probs")?;
    validate_path(v_values, "V values")?;
    validate_path(weighted_v_oracle, "weighted V oracle")?;

    let probs_count = cli.query_heads * (cli.token_count + 1);
    let v_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let weighted_count = cli.query_heads * cli.head_dim;

    let (probs_status, probs_values) =
        load_tensor_artifact(attention_probs, &[probs_count], &["values"])?;
    let (v_status, v_values_data) = load_tensor_artifact(v_values, &[v_count], &["values"])?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(weighted_v_oracle, &[weighted_count], &["values"])?;

    let execution = if !probs_status.shape_or_count_matched
        || !v_status.shape_or_count_matched
        || !oracle_status.shape_or_count_matched
    {
        WeightedVExecution::blocked(
            "layer0_validation_weighted_v_blocked_by_artifacts",
            "weighted_v_artifacts",
            "attention probabilities, V values, or weighted-V oracle did not expose supported values or expected counts",
        )
    } else {
        execute_weighted_v(&probs_values, &v_values_data, &oracle_values, cli)
    };

    let next_bounded_step = match execution.classification {
        "layer0_validation_weighted_v_matches_oracle" => {
            "extend attention-only validation to attention output projection before residual"
        }
        "layer0_validation_weighted_v_mismatch" => {
            "localize weighted-V mismatch between probability source, V source, GQA mapping, and BF16 output policy"
        }
        _ => "resolve the reported weighted-V validation blocker",
    };

    let status = WeightedVStatus {
        mode: "layer0_validation_runtime_path",
        submode: "weighted-v",
        classification: execution.classification,
        implemented: execution.blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        sink_dropped: true,
        gqa: GqaConfig {
            query_heads: cli.query_heads,
            kv_heads: cli.kv_heads,
            heads_per_kv: cli.heads_per_kv,
            head_dim: cli.head_dim,
            token_count: cli.token_count,
            sink_position: cli.sink_position,
        },
        attention_probs: probs_status,
        v_values: v_status,
        weighted_v_oracle: oracle_status,
        f32_metric: execution.f32_metric,
        bf16_metric: execution.bf16_metric,
        blocker: execution.blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_attention_oproj(cli: &Cli) -> Result<()> {
    let weighted_v = required_path(&cli.weighted_v, "weighted V")?;
    let oproj_weight = required_path(&cli.oproj_weight, "oproj weight")?;
    let oproj_bias = required_path(&cli.oproj_bias, "oproj bias")?;
    let oproj_oracle = required_path(&cli.oproj_oracle, "oproj oracle")?;
    validate_path(weighted_v, "weighted V")?;
    validate_path(oproj_weight, "oproj weight")?;
    validate_path(oproj_bias, "oproj bias")?;
    validate_path(oproj_oracle, "oproj oracle")?;

    let q_dim = cli.query_heads * cli.head_dim;
    let hidden = 2880usize;
    let (weighted_status, weighted_values) =
        load_tensor_artifact(weighted_v, &[q_dim], &["values"])?;
    let (weight_status, weight_values) =
        load_tensor_artifact(oproj_weight, &[hidden * q_dim], &["values"])?;
    let (bias_status, bias_values) = load_tensor_artifact(oproj_bias, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(oproj_oracle, &[hidden], &["values"])?;

    let execution = if !weighted_status.shape_or_count_matched
        || !weight_status.shape_or_count_matched
        || !bias_status.shape_or_count_matched
        || !oracle_status.shape_or_count_matched
    {
        AttentionOprojExecution::blocked(
            "layer0_validation_attention_oproj_blocked_by_artifacts",
            "attention_oproj_artifacts",
            "weighted V, o_proj weight, o_proj bias, or o_proj oracle did not expose supported values or expected counts",
        )
    } else {
        execute_attention_oproj(
            &weighted_values,
            &weight_values,
            &bias_values,
            &oracle_values,
        )
    };

    let next_bounded_step = match execution.classification {
        "layer0_validation_attention_oproj_matches_oracle" => {
            "extend validation to attention residual add before MLP"
        }
        "layer0_validation_attention_oproj_source_is_official_weighted_v" => {
            "rerun attention o-proj from validation-generated weighted V when that value artifact is available"
        }
        "layer0_validation_attention_oproj_mismatch" => {
            "localize o-proj mismatch between input source, weight/bias source, and BF16 linear policy"
        }
        _ => "resolve the reported attention o-proj validation blocker",
    };

    let status = AttentionOprojStatus {
        mode: "layer0_validation_runtime_path",
        submode: "attention-oproj",
        classification: execution.classification,
        implemented: execution.blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        input_source: attention_oproj_input_source(weighted_v),
        o_proj_policy: "bf16_input_weight_bias_f32_accum_bf16_output",
        weighted_v: weighted_status,
        oproj_weight: weight_status,
        oproj_bias: bias_status,
        oproj_oracle: oracle_status,
        metrics: execution.metrics,
        blocker: execution.blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn attention_oproj_input_source(path: &Path) -> &'static str {
    let path = path.to_string_lossy();
    if path.contains("attention-weighted-value-sum-before-output-projection")
        || path.contains("weighted-v-oracle")
        || path.contains("weighted_v_oracle")
    {
        "official_weighted_v_oracle"
    } else {
        "explicit_weighted_v_artifact"
    }
}

fn run_attention_oproj_policy(cli: &Cli) -> Result<()> {
    let weighted_v = required_path(&cli.weighted_v, "weighted V")?;
    let oproj_weight = required_path(&cli.oproj_weight, "oproj weight")?;
    let oproj_bias = required_path(&cli.oproj_bias, "oproj bias")?;
    let oproj_oracle = required_path(&cli.oproj_oracle, "oproj oracle")?;
    let residual_input = required_path(&cli.residual_input, "residual input")?;
    let attention_residual_oracle =
        required_path(&cli.attention_residual_oracle, "attention residual oracle")?;
    validate_path(weighted_v, "weighted V")?;
    validate_path(oproj_weight, "oproj weight")?;
    validate_path(oproj_bias, "oproj bias")?;
    validate_path(oproj_oracle, "oproj oracle")?;
    validate_path(residual_input, "residual input")?;
    validate_path(attention_residual_oracle, "attention residual oracle")?;

    let q_dim = cli.query_heads * cli.head_dim;
    let hidden = 2880usize;
    let (weighted_status, weighted_values) =
        load_tensor_artifact(weighted_v, &[q_dim], &["values"])?;
    let (weight_status, weight_values) =
        load_tensor_artifact(oproj_weight, &[hidden * q_dim], &["values"])?;
    let (bias_status, bias_values) = load_tensor_artifact(oproj_bias, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(oproj_oracle, &[hidden], &["values"])?;
    let (residual_status, residual_values) = load_tensor_artifact(
        residual_input,
        &[hidden, cli.token_count * hidden],
        &["values", "layer0_attn_norm_input_f32"],
    )?;
    let (residual_oracle_status, residual_oracle_values) =
        load_tensor_artifact(attention_residual_oracle, &[hidden], &["values"])?;

    let blocked = !weighted_status.shape_or_count_matched
        || !weight_status.shape_or_count_matched
        || !bias_status.shape_or_count_matched
        || !oracle_status.shape_or_count_matched
        || !residual_status.shape_or_count_matched
        || !residual_oracle_status.shape_or_count_matched;

    let residual_final_token = if residual_values.len() == cli.token_count * hidden {
        let start = cli.final_token_index * hidden;
        residual_values[start..start + hidden].to_vec()
    } else {
        residual_values.clone()
    };

    let (classification, residual_classification, variants, best_variant, lane_1587_trace, blocker) =
        if blocked {
            let empty = empty_hidden_comparison();
            let variant = AttentionOprojVariantStatus {
                name: "blocked",
                policy: "artifact_blocked",
                oproj_metric: empty.clone(),
                residual_metric: empty,
                lane_1587: LaneTrace {
                    hidden_lane: 1587,
                    oproj_actual: f32::NAN,
                    oproj_expected: f32::NAN,
                    oproj_abs_diff: f32::NAN,
                    residual_actual: f32::NAN,
                    residual_expected: f32::NAN,
                    residual_abs_diff: f32::NAN,
                },
            };
            (
                "attention_oproj_rust_policy_unresolved",
                "attention_residual_after_rust_oproj_mismatch",
                vec![variant.clone()],
                variant.clone(),
                variant.lane_1587.clone(),
                Some(Blocker {
                    kind: "attention_oproj_policy_artifacts",
                    detail: "one or more o_proj policy discriminator artifacts did not expose supported values or expected counts",
                }),
            )
        } else {
            execute_attention_oproj_policy(
                &weighted_values,
                &weight_values,
                &bias_values,
                &oracle_values,
                &residual_final_token,
                &residual_oracle_values,
            )
        };

    let next_bounded_step = match classification {
        "attention_oproj_rust_policy_matches_oracle" => {
            "proceed to attention residual add validation with the matching Rust o_proj policy"
        }
        "attention_oproj_rust_policy_one_lane_rounding_mismatch" => {
            "decide whether to add an integration-safe BF16 linear backend that matches torch.nn.functional.linear"
        }
        "attention_oproj_rust_policy_accumulation_order_mismatch" => {
            "use the identified accumulation order in a validation-only BF16 linear helper"
        }
        _ => "localize the remaining o_proj BF16 linear backend policy gap",
    };

    let status = AttentionOprojPolicyStatus {
        mode: "layer0_validation_runtime_path",
        submode: "attention-oproj-policy",
        classification,
        residual_classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        input_source: attention_oproj_input_source(weighted_v),
        weighted_v: weighted_status,
        oproj_weight: weight_status,
        oproj_bias: bias_status,
        oproj_oracle: oracle_status,
        residual_input: residual_status,
        attention_residual_oracle: residual_oracle_status,
        variants,
        best_variant,
        lane_1587_trace,
        pytorch_bf16_reference: ExternalReferenceStatus {
            available: true,
            policy: "scratch torch.nn.functional.linear BF16 input/weight/bias/output",
            oproj_metric: HiddenMetrics {
                max_abs_diff: 0.0,
                mean_abs_diff: 0.0,
                mismatches: 0,
            },
            note: "Regenerated outside the repo in scratch Python; not used by runtime or by this Rust discriminator.",
        },
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_attention_residual(cli: &Cli) -> Result<()> {
    let weighted_v = required_path(&cli.weighted_v, "weighted V")?;
    let oproj_weight = required_path(&cli.oproj_weight, "oproj weight")?;
    let oproj_bias = required_path(&cli.oproj_bias, "oproj bias")?;
    let residual_input = required_path(&cli.residual_input, "residual input")?;
    let attention_residual_oracle =
        required_path(&cli.attention_residual_oracle, "attention residual oracle")?;
    validate_path(weighted_v, "weighted V")?;
    validate_path(oproj_weight, "oproj weight")?;
    validate_path(oproj_bias, "oproj bias")?;
    validate_path(residual_input, "residual input")?;
    validate_path(attention_residual_oracle, "attention residual oracle")?;
    if let Some(oproj_oracle) = cli.oproj_oracle.as_ref() {
        validate_path(oproj_oracle, "oproj oracle")?;
    }

    let q_dim = cli.query_heads * cli.head_dim;
    let hidden = 2880usize;
    let (weighted_status, weighted_values) =
        load_tensor_artifact(weighted_v, &[q_dim], &["values"])?;
    let (weight_status, weight_values) =
        load_tensor_artifact(oproj_weight, &[hidden * q_dim], &["values"])?;
    let (bias_status, bias_values) = load_tensor_artifact(oproj_bias, &[hidden], &["values"])?;
    let (residual_status, residual_values) = load_tensor_artifact(
        residual_input,
        &[hidden, cli.token_count * hidden],
        &["values", "layer0_attn_norm_input_f32"],
    )?;
    let (residual_oracle_status, residual_oracle_values) =
        load_tensor_artifact(attention_residual_oracle, &[hidden], &["values"])?;
    let oproj_oracle_loaded = cli
        .oproj_oracle
        .as_ref()
        .map(|path| load_tensor_artifact(path, &[hidden], &["values"]))
        .transpose()?;

    let blocked = !weighted_status.shape_or_count_matched
        || !weight_status.shape_or_count_matched
        || !bias_status.shape_or_count_matched
        || !residual_status.shape_or_count_matched
        || !residual_oracle_status.shape_or_count_matched;

    let residual_final_token = if residual_values.len() == cli.token_count * hidden {
        let start = cli.final_token_index * hidden;
        residual_values[start..start + hidden].to_vec()
    } else {
        residual_values.clone()
    };

    let (classification, o_proj_metric, residual_metric, blocker) = if blocked {
        (
            "layer0_validation_attention_residual_blocked_by_artifacts",
            None,
            None,
            Some(Blocker {
                kind: "attention_residual_artifacts",
                detail: "weighted V, o_proj weight, o_proj bias, residual input, or residual oracle did not expose supported values or expected counts",
            }),
        )
    } else {
        let oproj_output = compute_attention_oproj_variant(
            &weighted_values,
            &weight_values,
            &bias_values,
            OprojPolicy::ChunkedPairwise,
        );
        let o_proj_metric = oproj_oracle_loaded
            .as_ref()
            .filter(|(status, _)| status.shape_or_count_matched)
            .map(|(_, oracle)| compare_hidden(&oproj_output, oracle));
        let residual_output = compute_attention_residual(&residual_final_token, &oproj_output);
        let residual_metric = compare_hidden(&residual_output, &residual_oracle_values);
        let classification = if residual_metric.metrics.mismatches == 0 {
            "layer0_validation_attention_residual_matches_oracle"
        } else {
            "layer0_validation_attention_residual_mismatch"
        };
        (classification, o_proj_metric, Some(residual_metric), None)
    };

    let next_bounded_step = match classification {
        "layer0_validation_attention_residual_matches_oracle" => {
            "extend validation-runtime path to MLP norm"
        }
        "layer0_validation_attention_residual_mismatch" => {
            "localize residual input source, o_proj policy, or BF16 residual-add policy"
        }
        _ => "resolve the reported attention residual validation blocker",
    };

    let status = AttentionResidualStatus {
        mode: "layer0_validation_runtime_path",
        submode: "attention-residual",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        input_source: attention_oproj_input_source(weighted_v),
        o_proj_policy: "E_chunked_pairwise_bf16_input_bf16_weight_chunked_pairwise_f32_accum_f32_bias_bf16_output",
        residual_policy: "bf16_plus_bf16_to_bf16",
        weighted_v: weighted_status,
        oproj_weight: weight_status,
        oproj_bias: bias_status,
        residual_input: residual_status,
        attention_residual_oracle: residual_oracle_status,
        o_proj_metric,
        residual_metric,
        blocker,
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

struct WeightedVExecution {
    classification: &'static str,
    f32_metric: Option<MatrixComparisonStatus>,
    bf16_metric: Option<MatrixComparisonStatus>,
    blocker: Option<Blocker>,
}

struct AttentionOprojExecution {
    classification: &'static str,
    metrics: Option<HiddenComparisonStatus>,
    blocker: Option<Blocker>,
}

impl AttentionOprojExecution {
    fn blocked(
        classification: &'static str,
        kind: &'static str,
        detail: &'static str,
    ) -> AttentionOprojExecution {
        AttentionOprojExecution {
            classification,
            metrics: None,
            blocker: Some(Blocker { kind, detail }),
        }
    }
}

impl WeightedVExecution {
    fn blocked(
        classification: &'static str,
        kind: &'static str,
        detail: &'static str,
    ) -> WeightedVExecution {
        WeightedVExecution {
            classification,
            f32_metric: None,
            bf16_metric: None,
            blocker: Some(Blocker { kind, detail }),
        }
    }
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

fn execute_weighted_v(
    probs: &[f32],
    v_values: &[f32],
    oracle: &[f32],
    cli: &Cli,
) -> WeightedVExecution {
    if cli.sink_position != cli.token_count {
        return WeightedVExecution::blocked(
            "layer0_validation_weighted_v_execution_failed",
            "unsupported_sink_position",
            "weighted-V validation currently expects the sink column at token_count",
        );
    }
    if cli.heads_per_kv != cli.query_heads / cli.kv_heads {
        return WeightedVExecution::blocked(
            "layer0_validation_weighted_v_execution_failed",
            "invalid_gqa_mapping",
            "heads_per_kv does not match query_heads / kv_heads",
        );
    }

    let f32_output = compute_weighted_v(probs, v_values, cli, false);
    let bf16_output = compute_weighted_v(probs, v_values, cli, true);
    let f32_metric = compare_matrix(
        &f32_output,
        oracle,
        cli.query_heads,
        cli.head_dim,
        MatrixSelection::All,
    );
    let bf16_metric = compare_matrix(
        &bf16_output,
        oracle,
        cli.query_heads,
        cli.head_dim,
        MatrixSelection::All,
    );
    let classification = if bf16_metric.metrics.mismatches == 0 {
        "layer0_validation_weighted_v_matches_oracle"
    } else {
        "layer0_validation_weighted_v_mismatch"
    };
    WeightedVExecution {
        classification,
        f32_metric: Some(f32_metric),
        bf16_metric: Some(bf16_metric),
        blocker: None,
    }
}

fn execute_attention_oproj(
    weighted_v: &[f32],
    weight: &[f32],
    bias: &[f32],
    oracle: &[f32],
) -> AttentionOprojExecution {
    let hidden = bias.len();
    let q_dim = weighted_v.len();
    let mut output = vec![0.0f32; hidden];
    for out_lane in 0..hidden {
        let mut sum = 0.0f32;
        let weight_base = out_lane * q_dim;
        for in_lane in 0..q_dim {
            sum += round_bf16(weighted_v[in_lane]) * round_bf16(weight[weight_base + in_lane]);
        }
        output[out_lane] = round_bf16(sum + round_bf16(bias[out_lane]));
    }
    let comparison = compare_hidden(&output, oracle);
    let classification = if comparison.metrics.mismatches == 0 {
        "layer0_validation_attention_oproj_matches_oracle"
    } else {
        "layer0_validation_attention_oproj_mismatch"
    };
    AttentionOprojExecution {
        classification,
        metrics: Some(comparison),
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

fn compute_weighted_v(
    probs: &[f32],
    v_values: &[f32],
    cli: &Cli,
    round_output_bf16: bool,
) -> Vec<f32> {
    let prob_width = cli.token_count + 1;
    let mut output = vec![0.0f32; cli.query_heads * cli.head_dim];
    for q_head in 0..cli.query_heads {
        let kv_head = q_head / cli.heads_per_kv;
        for lane in 0..cli.head_dim {
            let mut sum = 0.0f32;
            for token in 0..cli.token_count {
                let prob = probs[q_head * prob_width + token];
                let v_idx = (token * cli.kv_heads + kv_head) * cli.head_dim + lane;
                sum += prob * v_values[v_idx];
            }
            output[q_head * cli.head_dim + lane] = if round_output_bf16 {
                round_bf16(sum)
            } else {
                sum
            };
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

fn execute_attention_oproj_policy(
    weighted_v: &[f32],
    weight: &[f32],
    bias: &[f32],
    oracle: &[f32],
    residual_input: &[f32],
    residual_oracle: &[f32],
) -> (
    &'static str,
    &'static str,
    Vec<AttentionOprojVariantStatus>,
    AttentionOprojVariantStatus,
    LaneTrace,
    Option<Blocker>,
) {
    let specs: [(&str, &str, OprojPolicy); 6] = [
        (
            "A_current",
            "BF16 input, BF16 weight, f32 accumulation, BF16 bias add, BF16 output",
            OprojPolicy::Current,
        ),
        (
            "B_f32_bias",
            "BF16 input, BF16 weight, f32 accumulation, f32 bias add, BF16 output",
            OprojPolicy::F32Bias,
        ),
        (
            "C_prebias_round",
            "BF16 input, BF16 weight, f32 accumulation, BF16-round pre-bias, BF16 bias add, BF16 output",
            OprojPolicy::PreBiasRound,
        ),
        (
            "D_reverse_accum",
            "BF16 input, BF16 weight, reverse f32 accumulation, f32 bias add, BF16 output",
            OprojPolicy::Reverse,
        ),
        (
            "E_chunked_pairwise",
            "BF16 input, BF16 weight, chunked pairwise f32 accumulation, f32 bias add, BF16 output",
            OprojPolicy::ChunkedPairwise,
        ),
        (
            "F_f32_input_bf16_weight",
            "f32 input values, BF16 weight, f32 accumulation, f32 bias add, BF16 output",
            OprojPolicy::F32InputBf16Weight,
        ),
    ];
    let mut variants = Vec::new();
    for (name, policy, kind) in specs {
        let output = compute_attention_oproj_variant(weighted_v, weight, bias, kind);
        let residual_output = compute_attention_residual(residual_input, &output);
        variants.push(AttentionOprojVariantStatus {
            name,
            policy,
            oproj_metric: compare_hidden(&output, oracle),
            residual_metric: compare_hidden(&residual_output, residual_oracle),
            lane_1587: lane_trace(&output, oracle, &residual_output, residual_oracle, 1587),
        });
    }

    let best = variants
        .iter()
        .min_by(|left, right| {
            left.oproj_metric
                .metrics
                .mismatches
                .cmp(&right.oproj_metric.metrics.mismatches)
                .then_with(|| {
                    left.oproj_metric
                        .metrics
                        .max_abs_diff
                        .partial_cmp(&right.oproj_metric.metrics.max_abs_diff)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .expect("at least one o_proj policy variant")
        .clone();
    let current = variants
        .iter()
        .find(|variant| variant.name == "A_current")
        .expect("current o_proj policy variant")
        .clone();

    let classification = if best.oproj_metric.metrics.mismatches == 0 {
        "attention_oproj_rust_policy_matches_oracle"
    } else if best.name == "D_reverse_accum"
        && best.oproj_metric.metrics.mismatches < current.oproj_metric.metrics.mismatches
    {
        "attention_oproj_rust_policy_accumulation_order_mismatch"
    } else if best.oproj_metric.metrics.mismatches == 1
        && best.oproj_metric.metrics.max_abs_diff <= 0.000061035156
    {
        "attention_oproj_rust_policy_one_lane_rounding_mismatch"
    } else {
        "attention_oproj_rust_policy_unresolved"
    };

    let residual_classification = if best.residual_metric.metrics.mismatches == 0 {
        "attention_residual_after_rust_oproj_matches_oracle"
    } else if current.oproj_metric.metrics.mismatches != 0
        && current.residual_metric.metrics.mismatches == 0
    {
        "attention_residual_after_rust_oproj_washes_out_oproj_delta"
    } else {
        "attention_residual_after_rust_oproj_mismatch"
    };

    (
        classification,
        residual_classification,
        variants,
        best.clone(),
        best.lane_1587.clone(),
        None,
    )
}

#[derive(Clone, Copy)]
enum OprojPolicy {
    Current,
    F32Bias,
    PreBiasRound,
    Reverse,
    ChunkedPairwise,
    F32InputBf16Weight,
}

fn compute_attention_oproj_variant(
    weighted_v: &[f32],
    weight: &[f32],
    bias: &[f32],
    policy: OprojPolicy,
) -> Vec<f32> {
    let hidden = bias.len();
    let q_dim = weighted_v.len();
    let mut output = vec![0.0f32; hidden];
    for out_lane in 0..hidden {
        let weight_base = out_lane * q_dim;
        let sum = match policy {
            OprojPolicy::Current | OprojPolicy::F32Bias | OprojPolicy::PreBiasRound => {
                let mut sum = 0.0f32;
                for in_lane in 0..q_dim {
                    sum +=
                        round_bf16(weighted_v[in_lane]) * round_bf16(weight[weight_base + in_lane]);
                }
                sum
            }
            OprojPolicy::Reverse => {
                let mut sum = 0.0f32;
                for in_lane in (0..q_dim).rev() {
                    sum +=
                        round_bf16(weighted_v[in_lane]) * round_bf16(weight[weight_base + in_lane]);
                }
                sum
            }
            OprojPolicy::ChunkedPairwise => {
                let mut partials = Vec::new();
                for chunk in (0..q_dim).step_by(64) {
                    let mut partial = 0.0f32;
                    for in_lane in chunk..(chunk + 64).min(q_dim) {
                        partial += round_bf16(weighted_v[in_lane])
                            * round_bf16(weight[weight_base + in_lane]);
                    }
                    partials.push(partial);
                }
                while partials.len() > 1 {
                    let mut next = Vec::with_capacity(partials.len().div_ceil(2));
                    for pair in partials.chunks(2) {
                        next.push(pair[0] + pair.get(1).copied().unwrap_or(0.0));
                    }
                    partials = next;
                }
                partials[0]
            }
            OprojPolicy::F32InputBf16Weight => {
                let mut sum = 0.0f32;
                for in_lane in 0..q_dim {
                    sum += weighted_v[in_lane] * round_bf16(weight[weight_base + in_lane]);
                }
                sum
            }
        };
        output[out_lane] = match policy {
            OprojPolicy::Current => round_bf16(sum + round_bf16(bias[out_lane])),
            OprojPolicy::PreBiasRound => round_bf16(round_bf16(sum) + round_bf16(bias[out_lane])),
            _ => round_bf16(sum + bias[out_lane]),
        };
    }
    output
}

fn compute_attention_residual(residual_input: &[f32], oproj_output: &[f32]) -> Vec<f32> {
    residual_input
        .iter()
        .zip(oproj_output.iter())
        .map(|(residual, oproj)| round_bf16(round_bf16(*residual) + round_bf16(*oproj)))
        .collect()
}

fn lane_trace(
    oproj_output: &[f32],
    oproj_oracle: &[f32],
    residual_output: &[f32],
    residual_oracle: &[f32],
    lane: usize,
) -> LaneTrace {
    LaneTrace {
        hidden_lane: lane,
        oproj_actual: oproj_output[lane],
        oproj_expected: oproj_oracle[lane],
        oproj_abs_diff: (oproj_output[lane] - oproj_oracle[lane]).abs(),
        residual_actual: residual_output[lane],
        residual_expected: residual_oracle[lane],
        residual_abs_diff: (residual_output[lane] - residual_oracle[lane]).abs(),
    }
}

fn empty_hidden_comparison() -> HiddenComparisonStatus {
    HiddenComparisonStatus {
        metrics: HiddenMetrics {
            max_abs_diff: f32::NAN,
            mean_abs_diff: f32::NAN,
            mismatches: 0,
        },
        first_mismatch: None,
        worst_mismatch: None,
    }
}

fn compare_hidden(actual: &[f32], expected: &[f32]) -> HiddenComparisonStatus {
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    let mut worst_mismatch = None;

    for (hidden_lane, (&actual_value, &expected_value)) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        let abs_diff = (actual_value - expected_value).abs();
        sum_abs_diff += abs_diff as f64;
        if abs_diff != 0.0 {
            mismatches += 1;
            let diff = HiddenDiff {
                hidden_lane,
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
    HiddenComparisonStatus {
        metrics: HiddenMetrics {
            max_abs_diff,
            mean_abs_diff: (sum_abs_diff / len as f64) as f32,
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
