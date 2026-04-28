use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "cuda")]
use gpt_oss_model_runner::mxfp4_validation::{
    load_selected_experts_mxfp4_validation, Mxfp4SelectedExpertWeights,
};

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

    /// Attention residual artifact for MLP norm validation.
    #[arg(long)]
    attention_residual: Option<PathBuf>,

    /// Official MLP norm output oracle artifact.
    #[arg(long)]
    mlp_norm_oracle: Option<PathBuf>,

    /// MLP norm artifact for router validation.
    #[arg(long)]
    mlp_norm: Option<PathBuf>,

    /// Official router logits oracle artifact.
    #[arg(long)]
    router_logits_oracle: Option<PathBuf>,

    /// Official top-k indices and routing weights oracle artifact.
    #[arg(long)]
    topk_routing_oracle: Option<PathBuf>,

    /// Official selected expert outputs oracle artifact.
    #[arg(long)]
    selected_experts_oracle: Option<PathBuf>,

    /// Optional official expert30 MLP1 output before SwiGLU oracle artifact.
    #[arg(long)]
    expert30_mlp1_oracle: Option<PathBuf>,

    /// Optional official expert30 SwiGLU output before MLP2 oracle artifact.
    #[arg(long)]
    expert30_swiglu_oracle: Option<PathBuf>,

    /// Optional official expert30 MLP2 output before bias oracle artifact.
    #[arg(long)]
    expert30_mlp2_pre_bias_oracle: Option<PathBuf>,

    /// Optional official expert30 selected output after bias oracle artifact.
    #[arg(long)]
    expert30_selected_output_oracle: Option<PathBuf>,

    /// Optional official weighted expert sum oracle artifact.
    #[arg(long)]
    weighted_expert_sum_oracle: Option<PathBuf>,

    /// Selected expert ids in rank order.
    #[arg(long, default_value = "3,30,11,27")]
    selected_experts: String,

    /// Local model/checkpoint directory for validation-only tensor loading.
    #[arg(long)]
    model: Option<PathBuf>,

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
    MlpNorm,
    Router,
    SelectedExperts,
    SelectedExpertsDebug,
    SwigluDebug,
    Expert30Mlp2Debug,
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

#[derive(Debug, Serialize)]
struct MlpNormStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    input_source: &'static str,
    norm_policy: &'static str,
    epsilon: f32,
    attention_residual: TensorArtifactStatus,
    mlp_norm_oracle: TensorArtifactStatus,
    weight_source: Option<ModelTensorStatus>,
    metrics: Option<HiddenComparisonStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct RouterStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    input_source: &'static str,
    router_policy: &'static str,
    mlp_norm: TensorArtifactStatus,
    router_logits_oracle: TensorArtifactStatus,
    topk_routing_oracle: TensorArtifactStatus,
    router_weight_source: Option<ModelTensorStatus>,
    router_bias_source: Option<ModelTensorStatus>,
    logits_metric: Option<HiddenComparisonStatus>,
    topk: Option<TopkStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectedExpertsStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    selected_experts: Vec<usize>,
    input_source: &'static str,
    mlp_norm: TensorArtifactStatus,
    selected_experts_oracle: TensorArtifactStatus,
    tensor_sources: SelectedExpertTensorSources,
    mxfp4_loader: Option<Mxfp4LoaderStatus>,
    expert_formula: ExpertFormulaStatus,
    overall_metric: Option<SelectedExpertsComparisonStatus>,
    per_rank_metrics: Vec<SelectedExpertRankMetric>,
    expert30_internal_guard: Option<Value>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectedExpertsDebugStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    selected_experts: Vec<usize>,
    debug_expert: usize,
    input_source: &'static str,
    mlp_norm: TensorArtifactStatus,
    selected_experts_oracle: TensorArtifactStatus,
    mxfp4_loader: Option<Mxfp4LoaderStatus>,
    dequant_diagnostics: Option<SelectedExpertDequantDiagnostics>,
    expert30_boundary_metrics: Expert30BoundaryMetrics,
    variant_results: Vec<SelectedExpertDebugVariantStatus>,
    root_cause_summary: &'static str,
    weighted_sum_replacement: Option<Value>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectedExpertDequantDiagnostics {
    gate_up_weight: FiniteSummary,
    gate_up_bias: FiniteSummary,
    down_weight: FiniteSummary,
    down_bias: FiniteSummary,
    decoded_gate_up_dtype_shape: &'static str,
    decoded_down_dtype_shape: &'static str,
}

#[derive(Debug, Serialize)]
struct FiniteSummary {
    len: usize,
    finite_count: usize,
    non_finite_count: usize,
    min: f32,
    max: f32,
    mean: f32,
}

#[derive(Debug, Serialize)]
struct Expert30BoundaryMetrics {
    mlp1_before_swiglu: Option<HiddenComparisonStatus>,
    swiglu_before_mlp2: Option<HiddenComparisonStatus>,
    mlp2_before_bias: Option<HiddenComparisonStatus>,
    selected_output_after_bias: HiddenComparisonStatus,
}

#[derive(Debug, Serialize)]
struct SelectedExpertDebugVariantStatus {
    name: &'static str,
    policy: &'static str,
    mlp1_before_swiglu: Option<HiddenComparisonStatus>,
    swiglu_from_official_mlp1: Option<HiddenComparisonStatus>,
}

#[derive(Debug, Serialize)]
struct SwigluDebugStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    official_summary: OfficialSwigluSummary,
    mlp1_oracle: TensorArtifactStatus,
    swiglu_oracle: TensorArtifactStatus,
    variant_results: Vec<SwigluVariantStatus>,
    best_variant: SwigluVariantStatus,
    selected_experts_rerun: Option<Value>,
    weighted_sum_rerun: Option<Value>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct OfficialSwigluSummary {
    gate_up_split: &'static str,
    gate_clamp: &'static str,
    up_clamp: &'static str,
    sigmoid_scale: f32,
    multiply_order: &'static str,
    dtype_cast_behavior: &'static str,
    output_dtype: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct SwigluVariantStatus {
    name: &'static str,
    policy: &'static str,
    metric: HiddenComparisonStatus,
}

#[derive(Debug, Serialize)]
struct Expert30Mlp2DebugStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    official_swiglu_input_path: String,
    swiglu_oracle: TensorArtifactStatus,
    mlp2_pre_bias_oracle: TensorArtifactStatus,
    selected_experts_oracle: TensorArtifactStatus,
    down_projection_tensor_metadata: Option<Expert30DownProjectionMetadata>,
    variant_table: Vec<Expert30Mlp2VariantStatus>,
    best_variant: Expert30Mlp2VariantStatus,
    best_mlp2_pre_bias_metric: HiddenComparisonStatus,
    best_selected_output_metric: HiddenComparisonStatus,
    weighted_sum_replacement: Option<Value>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Expert30DownProjectionMetadata {
    helper_name: &'static str,
    decode_source: &'static str,
    expert: usize,
    down_weight_shape: [usize; 2],
    down_weight_dtype: &'static str,
    down_bias_shape: [usize; 1],
    down_bias_dtype: &'static str,
    layout_convention: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Expert30Mlp2VariantStatus {
    name: &'static str,
    policy: &'static str,
    mlp2_pre_bias_metric: HiddenComparisonStatus,
    selected_output_metric: HiddenComparisonStatus,
}

#[derive(Debug, Serialize)]
struct Mxfp4LoaderStatus {
    helper_name: &'static str,
    decode_source: &'static str,
    selected_experts: Vec<usize>,
    dtype_outputs: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectedExpertTensorSources {
    gate_up_proj_blocks: Option<ModelTensorStatus>,
    gate_up_proj_scales: Option<ModelTensorStatus>,
    gate_up_proj_bias: Option<ModelTensorStatus>,
    down_proj_blocks: Option<ModelTensorStatus>,
    down_proj_scales: Option<ModelTensorStatus>,
    down_proj_bias: Option<ModelTensorStatus>,
    loader_status: &'static str,
}

#[derive(Debug, Serialize)]
struct ExpertFormulaStatus {
    mlp1_fused_output_shape: [usize; 1],
    gate_slice: &'static str,
    up_slice: &'static str,
    gate_clamp_max: f32,
    up_clamp_min: f32,
    up_clamp_max: f32,
    sigmoid_scale: f32,
    swiglu: &'static str,
    output_dtype: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectedExpertsComparisonStatus {
    metrics: SelectedExpertsMetrics,
    first_mismatch: Option<SelectedExpertsDiff>,
    worst_mismatch: Option<SelectedExpertsDiff>,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedExpertsMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatches: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedExpertsDiff {
    rank: usize,
    expert: usize,
    hidden_lane: usize,
    actual: f32,
    expected: f32,
    abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct SelectedExpertRankMetric {
    rank: usize,
    expert: usize,
    metrics: Option<SelectedExpertsMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct TopkStatus {
    selected_experts_local: Vec<i64>,
    selected_experts_official: Vec<i64>,
    ordered_match: bool,
    selected_logits_metric: HiddenComparisonStatus,
    routing_weights_metric: HiddenComparisonStatus,
    local_weight_sum: f32,
    official_weight_sum: f32,
}

#[derive(Debug, Clone, Serialize)]
struct ModelTensorStatus {
    model_path: String,
    shard_path: String,
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
    value_count: usize,
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
        Mode::MlpNorm => run_mlp_norm(&cli),
        Mode::Router => run_router(&cli),
        Mode::SelectedExperts => run_selected_experts(&cli),
        Mode::SelectedExpertsDebug => run_selected_experts_debug(&cli),
        Mode::SwigluDebug => run_swiglu_debug(&cli),
        Mode::Expert30Mlp2Debug => run_expert30_mlp2_debug(&cli),
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

fn mlp_norm_input_source(path: &Path) -> &'static str {
    let display = path.display().to_string();
    if display.contains("hidden-state-after-attention-residual-add-before-mlp")
        || display.contains("attention-residual-add-before-mlp")
    {
        "official_attention_residual_oracle_because_prior_mode_exact"
    } else {
        "validation_attention_residual_output"
    }
}

fn router_input_source(path: &Path) -> &'static str {
    let display = path.display().to_string();
    if display.contains("mlp-norm-output-before-mlp-projections") {
        "official_mlp_norm_oracle_because_prior_mode_exact"
    } else {
        "validation_mlp_norm_output"
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

fn run_mlp_norm(cli: &Cli) -> Result<()> {
    let attention_residual = required_path(&cli.attention_residual, "attention residual")?;
    let mlp_norm_oracle = required_path(&cli.mlp_norm_oracle, "MLP norm oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(attention_residual, "attention residual")?;
    validate_path(mlp_norm_oracle, "MLP norm oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let (attention_status, attention_values) = load_tensor_artifact(
        attention_residual,
        &[hidden, cli.token_count * hidden],
        &["values", "layer0_attn_norm_input_f32"],
    )?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(mlp_norm_oracle, &[hidden], &["values"])?;
    let weight_result = load_model_tensor_f32(
        model,
        &[
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.norm.weight",
            "block.0.mlp.norm.scale",
            "post_attention_layernorm.weight",
        ],
    );

    let (classification, metrics, weight_source, blocker) =
        match (attention_status.shape_or_count_matched, oracle_status.shape_or_count_matched, weight_result) {
            (true, true, Ok((weight_source, weight_values))) if weight_values.len() == hidden => {
                let residual_final_token = if attention_values.len() == cli.token_count * hidden {
                    let start = cli.final_token_index * hidden;
                    attention_values[start..start + hidden].to_vec()
                } else {
                    attention_values.clone()
                };
                let output = compute_mlp_rms_norm(&residual_final_token, &weight_values, 1e-5);
                let metrics = compare_hidden(&output, &oracle_values);
                let classification = if metrics.metrics.mismatches == 0 {
                    "layer0_validation_mlp_norm_matches_oracle"
                } else {
                    "layer0_validation_mlp_norm_mismatch"
                };
                (classification, Some(metrics), Some(weight_source), None)
            }
            (_, _, Err(_)) => (
                "layer0_validation_mlp_norm_blocked_by_artifacts",
                None,
                None,
                Some(Blocker {
                    kind: "mlp_norm_weight",
                    detail: "could not load a supported layer0 MLP/post-attention norm weight tensor from --model",
                }),
            ),
            _ => (
                "layer0_validation_mlp_norm_blocked_by_artifacts",
                None,
                None,
                Some(Blocker {
                    kind: "mlp_norm_artifacts",
                    detail: "attention residual input or MLP norm oracle did not expose supported values or expected counts",
                }),
            ),
        };

    let next_bounded_step = match classification {
        "layer0_validation_mlp_norm_matches_oracle" => {
            "extend validation-runtime path to router logits and top-k"
        }
        "layer0_validation_mlp_norm_mismatch" => {
            "localize RMSNorm reduction, scale dtype, or BF16 output boundary policy"
        }
        _ => "resolve the reported MLP norm validation blocker",
    };

    let status = MlpNormStatus {
        mode: "layer0_validation_runtime_path",
        submode: "mlp-norm",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        input_source: mlp_norm_input_source(attention_residual),
        norm_policy: "bf16_input_fp32_rms_reduction_x_times_inverse_rms_then_scale_bf16_output",
        epsilon: 1e-5,
        attention_residual: attention_status,
        mlp_norm_oracle: oracle_status,
        weight_source,
        metrics,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_router(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let router_logits_oracle = required_path(&cli.router_logits_oracle, "router logits oracle")?;
    let topk_routing_oracle = required_path(&cli.topk_routing_oracle, "topk routing oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(router_logits_oracle, "router logits oracle")?;
    validate_path(topk_routing_oracle, "topk routing oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let experts = 32usize;
    let top_k = 4usize;
    let (mlp_norm_status, mlp_norm_values) =
        load_tensor_artifact(mlp_norm, &[hidden], &["values"])?;
    let (logits_oracle_status, logits_oracle_values) =
        load_tensor_artifact(router_logits_oracle, &[experts], &["values"])?;
    let (topk_oracle_status, topk_oracle) = load_topk_oracle(topk_routing_oracle, top_k)?;
    let weight_result = load_model_tensor_f32(model, &["model.layers.0.mlp.router.weight"]);
    let bias_result = load_model_tensor_f32(model, &["model.layers.0.mlp.router.bias"]);

    let (classification, logits_metric, topk, weight_source, bias_source, blocker) = match (
        mlp_norm_status.shape_or_count_matched,
        logits_oracle_status.shape_or_count_matched,
        topk_oracle_status.shape_or_count_matched,
        weight_result,
        bias_result,
    ) {
        (true, true, true, Ok((weight_source, weight_values)), Ok((bias_source, bias_values)))
            if weight_values.len() == experts * hidden && bias_values.len() == experts =>
        {
            let logits =
                compute_router_logits_bf16_linear(&mlp_norm_values, &weight_values, &bias_values);
            let logits_metric = compare_hidden(&logits, &logits_oracle_values);
            let local_topk = compute_router_topk(&logits, top_k);
            let ordered_match = local_topk.indices == topk_oracle.indices;
            let topk = TopkStatus {
                selected_experts_local: local_topk.indices.clone(),
                selected_experts_official: topk_oracle.indices,
                ordered_match,
                selected_logits_metric: compare_hidden(&local_topk.logits, &topk_oracle.logits),
                routing_weights_metric: compare_hidden(
                    &local_topk.routing_weights,
                    &topk_oracle.routing_weights,
                ),
                local_weight_sum: local_topk.routing_weights.iter().sum(),
                official_weight_sum: topk_oracle.routing_weights.iter().sum(),
            };
            let classification = if logits_metric.metrics.mismatches != 0 {
                "layer0_validation_router_logits_mismatch"
            } else if !topk.ordered_match
                || topk.selected_logits_metric.metrics.mismatches != 0
                || topk.routing_weights_metric.metrics.mismatches != 0
            {
                "layer0_validation_topk_routing_mismatch"
            } else {
                "layer0_validation_router_and_topk_match_oracle"
            };
            (
                classification,
                Some(logits_metric),
                Some(topk),
                Some(weight_source),
                Some(bias_source),
                None,
            )
        }
        (_, _, _, Err(_), _) => (
            "layer0_validation_router_blocked_by_artifacts",
            None,
            None,
            None,
            None,
            Some(Blocker {
                kind: "router_weight",
                detail: "could not load model.layers.0.mlp.router.weight from --model",
            }),
        ),
        (_, _, _, _, Err(_)) => (
            "layer0_validation_router_blocked_by_artifacts",
            None,
            None,
            None,
            None,
            Some(Blocker {
                kind: "router_bias",
                detail: "could not load model.layers.0.mlp.router.bias from --model",
            }),
        ),
        _ => (
            "layer0_validation_router_blocked_by_artifacts",
            None,
            None,
            None,
            None,
            Some(Blocker {
                kind: "router_artifacts",
                detail: "MLP norm, router logits oracle, or top-k/routing oracle did not expose supported values or expected counts",
            }),
        ),
    };

    let next_bounded_step = match classification {
        "layer0_validation_router_and_topk_match_oracle" => {
            "extend validation-runtime path to selected expert outputs"
        }
        "layer0_validation_router_logits_mismatch" => {
            "localize router BF16 linear accumulation or bias policy"
        }
        "layer0_validation_topk_routing_mismatch" => {
            "localize top-k ordering or selected-logit softmax policy"
        }
        _ => "resolve the reported router validation blocker",
    };

    let status = RouterStatus {
        mode: "layer0_validation_runtime_path",
        submode: "router",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        input_source: router_input_source(mlp_norm),
        router_policy: "bf16_input_bf16_weight_bf16_bias_f32_accum_bf16_output_topk_sorted_softmax_bf16_weights",
        mlp_norm: mlp_norm_status,
        router_logits_oracle: logits_oracle_status,
        topk_routing_oracle: topk_oracle_status,
        router_weight_source: weight_source,
        router_bias_source: bias_source,
        logits_metric,
        topk,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_selected_experts(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let selected_experts = parse_selected_experts(&cli.selected_experts)?;
    let hidden = 2880usize;
    let selected_count = selected_experts.len();
    let (mlp_norm_status, mlp_norm_values) =
        load_tensor_artifact(mlp_norm, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) = load_tensor_artifact(
        selected_experts_oracle,
        &[selected_count * hidden],
        &["values"],
    )?;

    let gate_up_proj_blocks =
        find_model_tensor_status(model, &["model.layers.0.mlp.experts.gate_up_proj_blocks"]).ok();
    let gate_up_proj_scales =
        find_model_tensor_status(model, &["model.layers.0.mlp.experts.gate_up_proj_scales"]).ok();
    let gate_up_proj_bias =
        find_model_tensor_status(model, &["model.layers.0.mlp.experts.gate_up_proj_bias"]).ok();
    let down_proj_blocks =
        find_model_tensor_status(model, &["model.layers.0.mlp.experts.down_proj_blocks"]).ok();
    let down_proj_scales =
        find_model_tensor_status(model, &["model.layers.0.mlp.experts.down_proj_scales"]).ok();
    let down_proj_bias =
        find_model_tensor_status(model, &["model.layers.0.mlp.experts.down_proj_bias"]).ok();

    let mx_fp4_sources = [
        gate_up_proj_blocks.as_ref(),
        gate_up_proj_scales.as_ref(),
        down_proj_blocks.as_ref(),
        down_proj_scales.as_ref(),
    ]
    .into_iter()
    .flatten()
    .any(|source| source.dtype == "U8");

    let missing_required_sources = gate_up_proj_blocks.is_none()
        || gate_up_proj_scales.is_none()
        || gate_up_proj_bias.is_none()
        || down_proj_blocks.is_none()
        || down_proj_scales.is_none()
        || down_proj_bias.is_none();
    let artifact_blocked =
        !mlp_norm_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;

    let (
        classification,
        overall_metric,
        per_rank_metrics,
        blocker,
        loader_status,
        mxfp4_loader,
        next_bounded_step,
    ) = if artifact_blocked {
        (
            "layer0_validation_selected_experts_blocked_by_artifacts",
            None,
            Vec::new(),
            Some(Blocker {
                kind: "selected_expert_artifacts",
                detail: "MLP norm input or selected expert oracle did not expose supported values or expected counts",
            }),
            "not_reached",
            None,
            "resolve the reported selected expert artifact blocker",
        )
    } else if missing_required_sources {
        (
            "layer0_validation_selected_experts_blocked_by_artifacts",
            None,
            Vec::new(),
            Some(Blocker {
                kind: "selected_expert_tensors",
                detail: "could not find the complete layer0 expert tensor set in --model",
            }),
            "missing_required_tensors",
            None,
            "resolve the reported selected expert tensor source blocker",
        )
    } else if mx_fp4_sources {
        match execute_selected_experts_mxfp4(model, &selected_experts, &mlp_norm_values, &oracle_values) {
            Ok((overall_metric, per_rank_metrics, mxfp4_loader)) => {
                let classification = if overall_metric.metrics.mismatches == 0 {
                    "layer0_validation_selected_experts_match_oracle"
                } else if per_rank_metrics
                    .iter()
                    .filter_map(|metric| metric.metrics.as_ref())
                    .filter(|metric| metric.mismatches != 0)
                    .count()
                    == 1
                {
                    "layer0_validation_selected_experts_mismatch_isolated"
                } else {
                    "layer0_validation_selected_experts_mismatch_large"
                };
                let next_bounded_step = match classification {
                    "layer0_validation_selected_experts_match_oracle" => {
                        "extend validation-runtime path to weighted expert sum"
                    }
                    "layer0_validation_selected_experts_mismatch_isolated" => {
                        "localize the isolated selected expert replay mismatch"
                    }
                    _ => "localize selected expert MXFP4 replay or BF16 boundary policy",
                };
                (
                    classification,
                    Some(overall_metric),
                    per_rank_metrics,
                    None,
                    "mxfp4_validation_loader_used",
                    Some(mxfp4_loader),
                    next_bounded_step,
                )
            }
            Err(_) => (
                "layer0_validation_selected_experts_blocked_by_mxfp4_loader_api",
                None,
                Vec::new(),
                Some(Blocker {
                    kind: "mxfp4_loader_api",
                    detail: "validation MXFP4 selected-expert loader/replay helper failed before comparison",
                }),
                "mxfp4_validation_loader_failed",
                None,
                "fix the narrow validation MXFP4 selected-expert loader/replay helper",
            ),
        }
    } else {
        (
            "layer0_validation_selected_experts_blocked_by_artifacts",
            None,
            Vec::new(),
            Some(Blocker {
                kind: "unsupported_dense_expert_replay",
                detail: "expert tensor metadata did not match the current MXFP4 blocker path, but dense selected-expert replay is not implemented in this slice",
            }),
            "unsupported_dense_replay",
            None,
            "add a bounded selected-expert replay implementation for the detected expert tensor layout",
        )
    };

    let status = SelectedExpertsStatus {
        mode: "layer0_validation_runtime_path",
        submode: "selected-experts",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        selected_experts,
        input_source: router_input_source(mlp_norm),
        mlp_norm: mlp_norm_status,
        selected_experts_oracle: oracle_status,
        tensor_sources: SelectedExpertTensorSources {
            gate_up_proj_blocks,
            gate_up_proj_scales,
            gate_up_proj_bias,
            down_proj_blocks,
            down_proj_scales,
            down_proj_bias,
            loader_status,
        },
        mxfp4_loader,
        expert_formula: ExpertFormulaStatus {
            mlp1_fused_output_shape: [5760],
            gate_slice: "values[0::2]",
            up_slice: "values[1::2]",
            gate_clamp_max: 7.0,
            up_clamp_min: -7.0,
            up_clamp_max: 7.0,
            sigmoid_scale: 1.702,
            swiglu: "gate * sigmoid(1.702 * gate) * (up + 1)",
            output_dtype: "BF16",
        },
        overall_metric,
        per_rank_metrics,
        expert30_internal_guard: None,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_selected_experts_debug(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let selected_experts = parse_selected_experts(&cli.selected_experts)?;
    let hidden = 2880usize;
    let selected_count = selected_experts.len();
    let (mlp_norm_status, mlp_norm_values) =
        load_tensor_artifact(mlp_norm, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) = load_tensor_artifact(
        selected_experts_oracle,
        &[selected_count * hidden],
        &["values"],
    )?;

    let expert30_rank = selected_experts.iter().position(|&expert| expert == 30);
    let artifact_blocked =
        !mlp_norm_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;

    let (
        classification,
        loader,
        dequant_diagnostics,
        boundary_metrics,
        variant_results,
        root_cause_summary,
        blocker,
        next_bounded_step,
    ) = if artifact_blocked {
        (
            "layer0_validation_selected_experts_blocked_by_artifacts",
            None,
            None,
            empty_expert30_boundary_metrics(),
            Vec::new(),
            "MLP norm input or selected expert oracle artifact did not expose supported values",
            Some(Blocker {
                kind: "selected_expert_debug_artifacts",
                detail: "MLP norm input or selected expert oracle did not expose supported values or expected counts",
            }),
            "resolve the reported selected expert debug artifact blocker",
        )
    } else if expert30_rank.is_none() {
        (
            "selected_expert_debug_blocked_by_missing_internal_oracles",
            None,
            None,
            empty_expert30_boundary_metrics(),
            Vec::new(),
            "selected expert list did not include expert30, so the focused expert30 localization could not run",
            Some(Blocker {
                kind: "selected_expert_debug_expert30",
                detail: "selected expert list must include expert30 for this focused debug mode",
            }),
            "rerun selected-experts-debug with expert30 included in --selected-experts",
        )
    } else {
        match execute_selected_experts_debug_mxfp4(
            model,
            &selected_experts,
            &mlp_norm_values,
            &oracle_values,
            expert30_rank.unwrap(),
            cli.expert30_mlp1_oracle.as_deref(),
            cli.expert30_swiglu_oracle.as_deref(),
            cli.expert30_mlp2_pre_bias_oracle.as_deref(),
        ) {
            Ok(debug) => debug,
            Err(_) => (
                "selected_expert_debug_blocked_by_mxfp4_api",
                None,
                None,
                empty_expert30_boundary_metrics(),
                Vec::new(),
                "MXFP4 validation helper failed before expert30 internal-boundary comparison",
                Some(Blocker {
                    kind: "mxfp4_debug_loader_api",
                    detail:
                        "validation MXFP4 selected-expert debug helper failed before comparison",
                }),
                "fix the narrow validation MXFP4 selected-expert loader/debug helper",
            ),
        }
    };

    let status = SelectedExpertsDebugStatus {
        mode: "layer0_validation_runtime_path",
        submode: "selected-experts-debug",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        selected_experts,
        debug_expert: 30,
        input_source: router_input_source(mlp_norm),
        mlp_norm: mlp_norm_status,
        selected_experts_oracle: oracle_status,
        mxfp4_loader: loader,
        dequant_diagnostics,
        expert30_boundary_metrics: boundary_metrics,
        variant_results,
        root_cause_summary,
        weighted_sum_replacement: None,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_swiglu_debug(cli: &Cli) -> Result<()> {
    let mlp1_oracle = required_path(&cli.expert30_mlp1_oracle, "expert30 MLP1 oracle")?;
    let swiglu_oracle = required_path(&cli.expert30_swiglu_oracle, "expert30 SwiGLU oracle")?;
    validate_path(mlp1_oracle, "expert30 MLP1 oracle")?;
    validate_path(swiglu_oracle, "expert30 SwiGLU oracle")?;

    let (mlp1_status, mlp1_values) = load_tensor_artifact(mlp1_oracle, &[5760], &["values"])?;
    let (swiglu_status, swiglu_values) = load_tensor_artifact(swiglu_oracle, &[2880], &["values"])?;
    let artifact_blocked =
        !mlp1_status.shape_or_count_matched || !swiglu_status.shape_or_count_matched;
    let (classification, variant_results, best_variant, blocker, next_bounded_step) =
        if artifact_blocked {
            let empty = SwigluVariantStatus {
                name: "not_run",
                policy: "artifact blocker",
                metric: empty_hidden_comparison(),
            };
            (
                "swiglu_policy_unresolved",
                Vec::new(),
                empty,
                Some(Blocker {
                    kind: "swiglu_debug_artifacts",
                    detail: "expert30 MLP1 or SwiGLU oracle did not expose supported values or expected counts",
                }),
                "resolve the reported SwiGLU debug artifact blocker",
            )
        } else {
            let variants = compute_swiglu_policy_variants(&mlp1_values, &swiglu_values);
            let best = variants
                .iter()
                .min_by(|left, right| {
                    left.metric
                        .metrics
                        .mismatches
                        .cmp(&right.metric.metrics.mismatches)
                        .then_with(|| {
                            left.metric
                                .metrics
                                .max_abs_diff
                                .partial_cmp(&right.metric.metrics.max_abs_diff)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                })
                .cloned()
                .expect("SwiGLU discriminator must include variants");
            let classification = classify_swiglu_policy(&variants, &best);
            let next_bounded_step = match classification {
                "swiglu_policy_matches_oracle" => {
                    "encode the exact SwiGLU policy in selected expert validation replay and rerun selected-experts"
                }
                "swiglu_split_policy_mismatch" => {
                    "update selected expert validation split policy only after confirming downstream selected-output impact"
                }
                "swiglu_clamp_policy_mismatch" => {
                    "update selected expert validation clamp policy only after confirming downstream selected-output impact"
                }
                "swiglu_dtype_rounding_policy_mismatch" => {
                    "inspect PyTorch BF16 elementwise SwiGLU behavior or use official SwiGLU as a temporary seam input for MLP2 localization"
                }
                _ => "inspect official SwiGLU operator/dtype behavior beyond the bounded Rust variants",
            };
            (classification, variants, best, None, next_bounded_step)
        };

    let status = SwigluDebugStatus {
        mode: "layer0_validation_runtime_path",
        submode: "swiglu-debug",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        official_summary: OfficialSwigluSummary {
            gate_up_split: "x_glu = x[..., ::2], x_linear = x[..., 1::2]",
            gate_clamp: "x_glu.clamp(min=None, max=limit)",
            up_clamp: "x_linear.clamp(min=-limit, max=limit)",
            sigmoid_scale: 1.702,
            multiply_order: "x_glu * sigmoid(alpha * x_glu), then multiply by x_linear + 1",
            dtype_cast_behavior: "no explicit cast in swiglu; torch operations inherit BF16 tensor semantics from MLP1 boundary",
            output_dtype: "BF16 official boundary",
        },
        mlp1_oracle: mlp1_status,
        swiglu_oracle: swiglu_status,
        variant_results,
        best_variant,
        selected_experts_rerun: None,
        weighted_sum_rerun: None,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_expert30_mlp2_debug(cli: &Cli) -> Result<()> {
    let swiglu_oracle = required_path(&cli.expert30_swiglu_oracle, "expert30 SwiGLU oracle")?;
    let mlp2_oracle = required_path(
        &cli.expert30_mlp2_pre_bias_oracle,
        "expert30 MLP2 pre-bias oracle",
    )?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(swiglu_oracle, "expert30 SwiGLU oracle")?;
    validate_path(mlp2_oracle, "expert30 MLP2 pre-bias oracle")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let (swiglu_status, swiglu_values) = load_tensor_artifact(swiglu_oracle, &[2880], &["values"])?;
    let (mlp2_status, mlp2_values) = load_tensor_artifact(mlp2_oracle, &[2880], &["values"])?;
    let (selected_status, selected_values) =
        load_tensor_artifact(selected_experts_oracle, &[4 * 2880, 2880], &["values"])?;
    let artifact_blocked = !swiglu_status.shape_or_count_matched
        || !mlp2_status.shape_or_count_matched
        || !selected_status.shape_or_count_matched;

    let selected_expert30 = if selected_values.len() == 4 * 2880 {
        selected_values[2880..(2 * 2880)].to_vec()
    } else {
        selected_values.clone()
    };

    let (
        classification,
        metadata,
        variant_table,
        best_variant,
        best_mlp2_pre_bias_metric,
        best_selected_output_metric,
        blocker,
        next_bounded_step,
    ) = if artifact_blocked {
        let empty_variant = Expert30Mlp2VariantStatus {
            name: "not_run",
            policy: "artifact blocker",
            mlp2_pre_bias_metric: empty_hidden_comparison(),
            selected_output_metric: empty_hidden_comparison(),
        };
        (
            "expert30_mlp2_blocked_by_artifacts",
            None,
            Vec::new(),
            empty_variant,
            empty_hidden_comparison(),
            empty_hidden_comparison(),
            Some(Blocker {
                kind: "expert30_mlp2_artifacts",
                detail: "expert30 SwiGLU, MLP2 pre-bias, or selected-output oracle did not expose supported values",
            }),
            "resolve the reported expert30 MLP2 artifact blocker",
        )
    } else {
        match execute_expert30_mlp2_debug_mxfp4(
            model,
            &swiglu_values,
            &mlp2_values,
            &selected_expert30,
        ) {
            Ok(result) => result,
            Err(_) => {
                let empty_variant = Expert30Mlp2VariantStatus {
                    name: "not_run",
                    policy: "MXFP4 decode blocker",
                    mlp2_pre_bias_metric: empty_hidden_comparison(),
                    selected_output_metric: empty_hidden_comparison(),
                };
                (
                    "expert30_mlp2_blocked_by_mxfp4_decode",
                    None,
                    Vec::new(),
                    empty_variant,
                    empty_hidden_comparison(),
                    empty_hidden_comparison(),
                    Some(Blocker {
                        kind: "expert30_mlp2_mxfp4_decode",
                        detail: "validation MXFP4 helper failed while loading expert30 down-projection weights",
                    }),
                    "fix the narrow validation MXFP4 down-projection loading path",
                )
            }
        }
    };

    let status = Expert30Mlp2DebugStatus {
        mode: "layer0_validation_runtime_path",
        submode: "expert30-mlp2-debug",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        official_swiglu_input_path: swiglu_oracle.display().to_string(),
        swiglu_oracle: swiglu_status,
        mlp2_pre_bias_oracle: mlp2_status,
        selected_experts_oracle: selected_status,
        down_projection_tensor_metadata: metadata,
        variant_table,
        best_variant,
        best_mlp2_pre_bias_metric,
        best_selected_output_metric,
        weighted_sum_replacement: None,
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

#[derive(Debug, Deserialize)]
struct SafetensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn load_model_tensor_f32(
    model_path: &Path,
    candidate_names: &[&str],
) -> Result<(ModelTensorStatus, Vec<f32>)> {
    let mut shards = if model_path.is_file() {
        vec![model_path.to_path_buf()]
    } else {
        let mut paths = fs::read_dir(model_path)
            .with_context(|| format!("failed to read model directory {}", model_path.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    };
    anyhow::ensure!(
        !shards.is_empty(),
        "no safetensors shards found in {}",
        model_path.display()
    );

    for shard in shards.drain(..) {
        if let Some(result) = try_load_tensor_from_safetensors(&shard, candidate_names)
            .with_context(|| format!("failed to inspect {}", shard.display()))?
        {
            let (tensor_name, entry, values) = result;
            let status = ModelTensorStatus {
                model_path: model_path.display().to_string(),
                shard_path: shard.display().to_string(),
                tensor_name,
                dtype: entry.dtype,
                shape: entry.shape,
                value_count: values.len(),
            };
            return Ok((status, values));
        }
    }

    anyhow::bail!(
        "none of {:?} found in {}",
        candidate_names,
        model_path.display()
    )
}

fn find_model_tensor_status(
    model_path: &Path,
    candidate_names: &[&str],
) -> Result<ModelTensorStatus> {
    let mut shards = if model_path.is_file() {
        vec![model_path.to_path_buf()]
    } else {
        let mut paths = fs::read_dir(model_path)
            .with_context(|| format!("failed to read model directory {}", model_path.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    };
    anyhow::ensure!(
        !shards.is_empty(),
        "no safetensors shards found in {}",
        model_path.display()
    );

    for shard in shards.drain(..) {
        if let Some((tensor_name, entry)) = try_find_tensor_metadata(&shard, candidate_names)
            .with_context(|| format!("failed to inspect {}", shard.display()))?
        {
            return Ok(ModelTensorStatus {
                model_path: model_path.display().to_string(),
                shard_path: shard.display().to_string(),
                tensor_name,
                dtype: entry.dtype,
                value_count: entry.shape.iter().product(),
                shape: entry.shape,
            });
        }
    }

    anyhow::bail!(
        "none of {:?} found in {}",
        candidate_names,
        model_path.display()
    )
}

fn try_find_tensor_metadata(
    shard: &Path,
    candidate_names: &[&str],
) -> Result<Option<(String, SafetensorEntry)>> {
    let mut file =
        fs::File::open(shard).with_context(|| format!("failed to open {}", shard.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).with_context(|| {
        format!(
            "failed to read safetensors header length from {}",
            shard.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .with_context(|| format!("failed to read safetensors header from {}", shard.display()))?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&header_bytes)
        .with_context(|| {
            format!(
                "failed to parse safetensors header from {}",
                shard.display()
            )
        })?;

    for name in candidate_names {
        if let Some(value) = header.get(*name) {
            let entry: SafetensorEntry = serde_json::from_value(value.clone())
                .with_context(|| format!("failed to parse tensor metadata for {}", name))?;
            return Ok(Some(((*name).to_string(), entry)));
        }
    }

    Ok(None)
}

fn try_load_tensor_from_safetensors(
    shard: &Path,
    candidate_names: &[&str],
) -> Result<Option<(String, SafetensorEntry, Vec<f32>)>> {
    let mut file =
        fs::File::open(shard).with_context(|| format!("failed to open {}", shard.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).with_context(|| {
        format!(
            "failed to read safetensors header length from {}",
            shard.display()
        )
    })?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .with_context(|| format!("failed to read safetensors header from {}", shard.display()))?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&header_bytes)
        .with_context(|| {
            format!(
                "failed to parse safetensors header from {}",
                shard.display()
            )
        })?;
    let data_start = 8u64 + header_len as u64;

    for name in candidate_names {
        if let Some(value) = header.get(*name) {
            let entry: SafetensorEntry = serde_json::from_value(value.clone())
                .with_context(|| format!("failed to parse tensor metadata for {}", name))?;
            let byte_len = entry.data_offsets[1] - entry.data_offsets[0];
            let mut bytes = vec![0u8; byte_len];
            file.seek(SeekFrom::Start(data_start + entry.data_offsets[0] as u64))
                .with_context(|| {
                    format!("failed to seek to tensor {} in {}", name, shard.display())
                })?;
            file.read_exact(&mut bytes).with_context(|| {
                format!("failed to read tensor {} from {}", name, shard.display())
            })?;
            let numel = entry.shape.iter().product::<usize>();
            let values = convert_safetensor_values_to_f32(&bytes, &entry.dtype, numel, name)?;
            return Ok(Some(((*name).to_string(), entry, values)));
        }
    }

    Ok(None)
}

fn convert_safetensor_values_to_f32(
    bytes: &[u8],
    dtype: &str,
    numel: usize,
    tensor_name: &str,
) -> Result<Vec<f32>> {
    match dtype {
        "F32" => {
            anyhow::ensure!(
                bytes.len() == numel * 4,
                "{} F32 byte count mismatch",
                tensor_name
            );
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        "F16" => {
            anyhow::ensure!(
                bytes.len() == numel * 2,
                "{} F16 byte count mismatch",
                tensor_name
            );
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect())
        }
        "BF16" => {
            anyhow::ensure!(
                bytes.len() == numel * 2,
                "{} BF16 byte count mismatch",
                tensor_name
            );
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
                .collect())
        }
        _ => anyhow::bail!("unsupported dtype {} for {}", dtype, tensor_name),
    }
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

#[derive(Debug)]
struct TopkOracle {
    indices: Vec<i64>,
    logits: Vec<f32>,
    routing_weights: Vec<f32>,
}

fn load_topk_oracle(path: &Path, top_k: usize) -> Result<(TensorArtifactStatus, TopkOracle)> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let indices: Vec<i64> = value
        .get("selected_expert_indices")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_i64(values))
        .unwrap_or_default();
    let logits: Vec<f32> = value
        .get("selected_expert_logits")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_f32(values))
        .unwrap_or_default();
    let routing_weights: Vec<f32> = value
        .get("routing_weights")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_f32(values))
        .unwrap_or_default();
    let matched = indices.len() == top_k && logits.len() == top_k && routing_weights.len() == top_k;
    Ok((
        TensorArtifactStatus {
            path: path.display().to_string(),
            json_loaded: true,
            shape: Some(vec![top_k]),
            value_count: Some(routing_weights.len()),
            expected_value_counts: vec![top_k],
            shape_or_count_matched: matched,
            value_key: Some(
                "selected_expert_indices.values,selected_expert_logits.values,routing_weights.values"
                    .to_string(),
            ),
        },
        TopkOracle {
            indices,
            logits,
            routing_weights,
        },
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

fn json_values_to_i64(values: &[Value]) -> Vec<i64> {
    values
        .iter()
        .map(|value| value.as_i64().unwrap_or_default())
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

fn compute_mlp_rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mut square_sum = 0.0f32;
    for value in input {
        let x = round_bf16(*value);
        square_sum += x * x;
    }
    let mean_square = square_sum / input.len().max(1) as f32;
    let inverse_rms = 1.0f32 / (mean_square + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, scale)| round_bf16(round_bf16(*x) * inverse_rms * *scale))
        .collect()
}

fn compute_router_logits_bf16_linear(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let experts = bias.len();
    let hidden = input.len();
    let mut output = vec![0.0f32; experts];
    for expert in 0..experts {
        let weight_base = expert * hidden;
        let mut sum = 0.0f32;
        for hidden_lane in 0..hidden {
            sum += round_bf16(input[hidden_lane]) * round_bf16(weight[weight_base + hidden_lane]);
        }
        output[expert] = round_bf16(sum + round_bf16(bias[expert]));
    }
    output
}

#[derive(Debug)]
struct RouterTopk {
    indices: Vec<i64>,
    logits: Vec<f32>,
    routing_weights: Vec<f32>,
}

fn compute_router_topk(logits: &[f32], top_k: usize) -> RouterTopk {
    let mut indexed = logits
        .iter()
        .enumerate()
        .map(|(index, value)| (index as i64, *value))
        .collect::<Vec<_>>();
    indexed.sort_by(|(left_index, left_value), (right_index, right_value)| {
        right_value
            .partial_cmp(left_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
    let selected = &indexed[..top_k.min(indexed.len())];
    let indices = selected.iter().map(|(index, _)| *index).collect::<Vec<_>>();
    let selected_logits = selected
        .iter()
        .map(|(_, value)| round_bf16(*value))
        .collect::<Vec<_>>();
    let max_logit = selected_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_values = selected_logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect::<Vec<_>>();
    let exp_sum = exp_values.iter().sum::<f32>();
    let routing_weights = exp_values
        .iter()
        .map(|value| round_bf16(*value / exp_sum))
        .collect::<Vec<_>>();
    RouterTopk {
        indices,
        logits: selected_logits,
        routing_weights,
    }
}

fn parse_selected_experts(value: &str) -> Result<Vec<usize>> {
    let experts = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid selected expert id: {part}"))
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(!experts.is_empty(), "--selected-experts cannot be empty");
    Ok(experts)
}

#[cfg(feature = "cuda")]
fn execute_selected_experts_mxfp4(
    model: &Path,
    selected_experts: &[usize],
    mlp_norm: &[f32],
    oracle: &[f32],
) -> Result<(
    SelectedExpertsComparisonStatus,
    Vec<SelectedExpertRankMetric>,
    Mxfp4LoaderStatus,
)> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut output = Vec::with_capacity(selected_experts.len() * 2880);
    for expert in &loaded.experts {
        output.extend(compute_selected_expert_output(mlp_norm, expert));
    }
    let overall = compare_selected_experts(&output, oracle, selected_experts);
    let per_rank_metrics = selected_experts
        .iter()
        .enumerate()
        .map(|(rank, &expert)| {
            let start = rank * 2880;
            let end = start + 2880;
            let comparison =
                compare_selected_experts(&output[start..end], &oracle[start..end], &[expert]);
            SelectedExpertRankMetric {
                rank,
                expert,
                metrics: Some(comparison.metrics),
            }
        })
        .collect::<Vec<_>>();
    let loader = Mxfp4LoaderStatus {
        helper_name: loaded.helper_name,
        decode_source: loaded.decode_source,
        selected_experts: loaded.selected_experts,
        dtype_outputs: loaded.dtype_outputs,
    };
    Ok((overall, per_rank_metrics, loader))
}

#[cfg(not(feature = "cuda"))]
fn execute_selected_experts_mxfp4(
    _model: &Path,
    _selected_experts: &[usize],
    _mlp_norm: &[f32],
    _oracle: &[f32],
) -> Result<(
    SelectedExpertsComparisonStatus,
    Vec<SelectedExpertRankMetric>,
    Mxfp4LoaderStatus,
)> {
    anyhow::bail!("MXFP4 selected expert validation requires the cuda feature")
}

type SelectedExpertsDebugResult = (
    &'static str,
    Option<Mxfp4LoaderStatus>,
    Option<SelectedExpertDequantDiagnostics>,
    Expert30BoundaryMetrics,
    Vec<SelectedExpertDebugVariantStatus>,
    &'static str,
    Option<Blocker>,
    &'static str,
);

type Expert30Mlp2DebugResult = (
    &'static str,
    Option<Expert30DownProjectionMetadata>,
    Vec<Expert30Mlp2VariantStatus>,
    Expert30Mlp2VariantStatus,
    HiddenComparisonStatus,
    HiddenComparisonStatus,
    Option<Blocker>,
    &'static str,
);

#[cfg(feature = "cuda")]
fn execute_selected_experts_debug_mxfp4(
    model: &Path,
    selected_experts: &[usize],
    mlp_norm: &[f32],
    selected_oracle: &[f32],
    expert30_rank: usize,
    mlp1_oracle_path: Option<&Path>,
    swiglu_oracle_path: Option<&Path>,
    mlp2_pre_bias_oracle_path: Option<&Path>,
) -> Result<SelectedExpertsDebugResult> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let expert = loaded
        .experts
        .iter()
        .find(|expert| expert.expert == 30)
        .context("MXFP4 validation loader did not return expert30")?;
    let selected_start = expert30_rank * 2880;
    let selected_end = selected_start + 2880;
    let selected_oracle = &selected_oracle[selected_start..selected_end];

    let trace = compute_selected_expert_trace(mlp_norm, expert, MlpReplayPolicy::Current);
    let selected_output_metric = compare_hidden(&trace.selected_output, selected_oracle);

    let mlp1_oracle = load_optional_debug_oracle(mlp1_oracle_path, 5760)?;
    let swiglu_oracle = load_optional_debug_oracle(swiglu_oracle_path, 2880)?;
    let mlp2_pre_bias_oracle = load_optional_debug_oracle(mlp2_pre_bias_oracle_path, 2880)?;

    let mlp1_metric = mlp1_oracle
        .as_ref()
        .map(|oracle| compare_hidden(&trace.mlp1_before_swiglu, oracle));
    let swiglu_metric = swiglu_oracle
        .as_ref()
        .map(|oracle| compare_hidden(&trace.swiglu_before_mlp2, oracle));
    let mlp2_metric = mlp2_pre_bias_oracle
        .as_ref()
        .map(|oracle| compare_hidden(&trace.mlp2_before_bias, oracle));

    let variant_specs = [
        (
            "current",
            "BF16 input, f16-dequant weight widened to f32, f32 accumulation, BF16 bias add, BF16 output",
            MlpReplayPolicy::Current,
        ),
        (
            "weight_bf16",
            "BF16 input, BF16-rounded dequant weight, f32 accumulation, BF16 bias add, BF16 output",
            MlpReplayPolicy::WeightBf16,
        ),
        (
            "f32_input",
            "f32 input, f16-dequant weight widened to f32, f32 accumulation, BF16 bias add, BF16 output",
            MlpReplayPolicy::F32Input,
        ),
        (
            "output_f16",
            "BF16 input, f16-dequant weight widened to f32, f32 accumulation, BF16 bias add, f16 output",
            MlpReplayPolicy::OutputF16,
        ),
    ];
    let mut variant_results = variant_specs
        .into_iter()
        .map(|(name, policy, replay_policy)| {
            let variant_trace = compute_selected_expert_trace(mlp_norm, expert, replay_policy);
            SelectedExpertDebugVariantStatus {
                name,
                policy,
                mlp1_before_swiglu: mlp1_oracle
                    .as_ref()
                    .map(|oracle| compare_hidden(&variant_trace.mlp1_before_swiglu, oracle)),
                swiglu_from_official_mlp1: None,
            }
        })
        .collect::<Vec<_>>();
    if let (Some(mlp1_oracle), Some(swiglu_oracle)) = (mlp1_oracle.as_ref(), swiglu_oracle.as_ref())
    {
        let swiglu_specs = [
            (
                "swiglu_current_from_official_mlp1",
                "official MLP1 input, gate/up interleaved, BF16 inputs, sigmoid scale 1.702, BF16 output",
                SwigluReplayPolicy::Current,
            ),
            (
                "swiglu_mul_round_from_official_mlp1",
                "official MLP1 input, BF16-round gate*sigmoid before multiplying by up+1, BF16 output",
                SwigluReplayPolicy::MulRound,
            ),
            (
                "swiglu_sigmoid_round_from_official_mlp1",
                "official MLP1 input, BF16-round sigmoid value, BF16 output",
                SwigluReplayPolicy::SigmoidRound,
            ),
            (
                "swiglu_swapped_from_official_mlp1",
                "official MLP1 input, swapped gate/up lane order, BF16 output",
                SwigluReplayPolicy::Swapped,
            ),
        ];
        variant_results.extend(
            swiglu_specs
                .into_iter()
                .map(|(name, policy, swiglu_policy)| {
                    let swiglu = compute_swiglu_with_policy(mlp1_oracle, swiglu_policy);
                    SelectedExpertDebugVariantStatus {
                        name,
                        policy,
                        mlp1_before_swiglu: None,
                        swiglu_from_official_mlp1: Some(compare_hidden(&swiglu, swiglu_oracle)),
                    }
                }),
        );
    }

    let boundary_metrics = Expert30BoundaryMetrics {
        mlp1_before_swiglu: mlp1_metric.clone(),
        swiglu_before_mlp2: swiglu_metric,
        mlp2_before_bias: mlp2_metric,
        selected_output_after_bias: selected_output_metric,
    };

    let classification = if let Some(metric) = &boundary_metrics.mlp1_before_swiglu {
        if metric.metrics.mismatches != 0 {
            "selected_expert_mismatch_starts_at_mxfp4_dequant_or_mlp1"
        } else if boundary_metrics
            .swiglu_before_mlp2
            .as_ref()
            .map(|metric| metric.metrics.mismatches != 0)
            .unwrap_or(false)
        {
            "selected_expert_mismatch_starts_at_swiglu"
        } else if boundary_metrics
            .mlp2_before_bias
            .as_ref()
            .map(|metric| metric.metrics.mismatches != 0)
            .unwrap_or(false)
        {
            "selected_expert_mismatch_starts_at_down_projection"
        } else if boundary_metrics
            .selected_output_after_bias
            .metrics
            .mismatches
            != 0
        {
            "selected_expert_mismatch_starts_at_bias_or_output"
        } else {
            "selected_expert_replay_policy_identified"
        }
    } else {
        "selected_expert_debug_blocked_by_missing_internal_oracles"
    };
    let root_cause_summary = match classification {
        "selected_expert_mismatch_starts_at_mxfp4_dequant_or_mlp1" => {
            "expert30 MLP1 output already diverges, so the next bounded target is MXFP4 dequant layout/scale interpretation or MLP1 replay policy"
        }
        "selected_expert_mismatch_starts_at_swiglu" => {
            "expert30 MLP1 matches but SwiGLU diverges, so the next bounded target is clamp/activation/dtype policy"
        }
        "selected_expert_mismatch_starts_at_down_projection" => {
            "expert30 MLP1 and SwiGLU match but MLP2 pre-bias diverges, so the next bounded target is down projection layout or replay policy"
        }
        "selected_expert_mismatch_starts_at_bias_or_output" => {
            "expert30 pre-bias boundaries match but selected output diverges, so the next bounded target is bias/output boundary policy"
        }
        "selected_expert_replay_policy_identified" => {
            "expert30 internal boundaries and selected output match with the current validation replay"
        }
        _ => "expert30 internal boundary oracles were not provided, so first divergence could not be localized",
    };
    let next_bounded_step = match classification {
        "selected_expert_mismatch_starts_at_mxfp4_dequant_or_mlp1" => {
            "inspect MXFP4 dequant layout and scale interpretation against proven/runtime semantics"
        }
        "selected_expert_debug_blocked_by_missing_internal_oracles" => {
            "rerun selected-experts-debug with expert30 internal boundary oracle paths"
        }
        _ => "continue the selected expert boundary indicated by the root classification",
    };

    let loader = Mxfp4LoaderStatus {
        helper_name: loaded.helper_name,
        decode_source: loaded.decode_source,
        selected_experts: loaded.selected_experts,
        dtype_outputs: loaded.dtype_outputs,
    };
    let dequant_diagnostics = SelectedExpertDequantDiagnostics {
        gate_up_weight: finite_summary(&expert.gate_up_weight),
        gate_up_bias: finite_summary(&expert.gate_up_bias),
        down_weight: finite_summary(&expert.down_weight),
        down_bias: finite_summary(&expert.down_bias),
        decoded_gate_up_dtype_shape: "f16_dequant_widened_to_f32 [5760,2880]",
        decoded_down_dtype_shape: "f16_dequant_widened_to_f32 [2880,2880]",
    };

    Ok((
        classification,
        Some(loader),
        Some(dequant_diagnostics),
        boundary_metrics,
        variant_results,
        root_cause_summary,
        None,
        next_bounded_step,
    ))
}

#[cfg(feature = "cuda")]
fn execute_expert30_mlp2_debug_mxfp4(
    model: &Path,
    swiglu: &[f32],
    mlp2_oracle: &[f32],
    selected_oracle: &[f32],
) -> Result<Expert30Mlp2DebugResult> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, &[30])
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let expert = loaded
        .experts
        .iter()
        .find(|expert| expert.expert == 30)
        .context("MXFP4 validation loader did not return expert30")?;
    let metadata = Expert30DownProjectionMetadata {
        helper_name: loaded.helper_name,
        decode_source: loaded.decode_source,
        expert: 30,
        down_weight_shape: [2880, 2880],
        down_weight_dtype: "dequantized_f16_widened_to_f32",
        down_bias_shape: [2880],
        down_bias_dtype: "BF16_widened_to_f32",
        layout_convention: "down_weight is [out_hidden, in_intermediate] row-major",
    };
    let specs = [
        (
            "A_current",
            "current decoded down weight, BF16 input, f32 accumulation, BF16 pre-bias/output, BF16 bias add",
            Expert30Mlp2Policy::Current,
        ),
        (
            "B_weight_bf16_round",
            "decoded down weight BF16-rounded before matmul, BF16 input, f32 accumulation, BF16 output",
            Expert30Mlp2Policy::WeightBf16Round,
        ),
        (
            "C_weight_f16",
            "decoded down weight treated as f16 before matmul, BF16 input, f32 accumulation, BF16 output",
            Expert30Mlp2Policy::WeightF16,
        ),
        (
            "D_f32_accum_bf16_output",
            "BF16 input, decoded f16 weight widened to f32, f32 accumulation, BF16 pre-bias/output",
            Expert30Mlp2Policy::F32AccumBf16Output,
        ),
        (
            "E_chunked_pairwise",
            "BF16 input, BF16 weight, chunked pairwise f32 accumulation, f32 bias add, BF16 output",
            Expert30Mlp2Policy::ChunkedPairwise,
        ),
        (
            "F1_bf16_prebias_bf16_bias",
            "BF16 pre-bias plus BF16 bias to BF16 selected output",
            Expert30Mlp2Policy::Bf16PreBiasBf16Bias,
        ),
        (
            "F2_f32_prebias_f32_bias",
            "f32 pre-bias plus f32 bias to BF16 selected output",
            Expert30Mlp2Policy::F32PreBiasF32Bias,
        ),
    ];
    let mut variants = Vec::with_capacity(specs.len());
    for (name, policy, kind) in specs {
        let pre_bias = compute_expert30_mlp2_prebias_variant(swiglu, &expert.down_weight, kind);
        let selected = compute_expert30_selected_output_variant(&pre_bias, &expert.down_bias, kind);
        variants.push(Expert30Mlp2VariantStatus {
            name,
            policy,
            mlp2_pre_bias_metric: compare_hidden(&pre_bias, mlp2_oracle),
            selected_output_metric: compare_hidden(&selected, selected_oracle),
        });
    }
    let best = variants
        .iter()
        .min_by(|left, right| {
            left.selected_output_metric
                .metrics
                .mismatches
                .cmp(&right.selected_output_metric.metrics.mismatches)
                .then_with(|| {
                    left.selected_output_metric
                        .metrics
                        .max_abs_diff
                        .partial_cmp(&right.selected_output_metric.metrics.max_abs_diff)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .cloned()
        .expect("expert30 MLP2 debug must include variants");
    let classification = if best.mlp2_pre_bias_metric.metrics.mismatches == 0
        && best.selected_output_metric.metrics.mismatches == 0
    {
        "expert30_mlp2_from_official_swiglu_matches_oracle"
    } else if best.mlp2_pre_bias_metric.metrics.max_abs_diff <= 0.00390625 {
        "expert30_mlp2_from_official_swiglu_mismatch_small"
    } else {
        "expert30_mlp2_from_official_swiglu_mismatch_large"
    };
    let next_bounded_step = match classification {
        "expert30_mlp2_from_official_swiglu_matches_oracle" => {
            "use official SwiGLU as a temporary seam input to validate selected-output and weighted-sum replacement"
        }
        "expert30_mlp2_from_official_swiglu_mismatch_small" => {
            "pin the remaining down-projection accumulation/bias boundary policy"
        }
        _ => "inspect expert30 down-projection MXFP4 layout/decode or replay policy",
    };
    Ok((
        classification,
        Some(metadata),
        variants,
        best.clone(),
        best.mlp2_pre_bias_metric.clone(),
        best.selected_output_metric.clone(),
        None,
        next_bounded_step,
    ))
}

#[cfg(not(feature = "cuda"))]
fn execute_expert30_mlp2_debug_mxfp4(
    _model: &Path,
    _swiglu: &[f32],
    _mlp2_oracle: &[f32],
    _selected_oracle: &[f32],
) -> Result<Expert30Mlp2DebugResult> {
    anyhow::bail!("expert30 MLP2 debug validation requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
fn execute_selected_experts_debug_mxfp4(
    _model: &Path,
    _selected_experts: &[usize],
    _mlp_norm: &[f32],
    _selected_oracle: &[f32],
    _expert30_rank: usize,
    _mlp1_oracle_path: Option<&Path>,
    _swiglu_oracle_path: Option<&Path>,
    _mlp2_pre_bias_oracle_path: Option<&Path>,
) -> Result<SelectedExpertsDebugResult> {
    anyhow::bail!("MXFP4 selected expert debug validation requires the cuda feature")
}

#[cfg(feature = "cuda")]
struct SelectedExpertTrace {
    mlp1_before_swiglu: Vec<f32>,
    swiglu_before_mlp2: Vec<f32>,
    mlp2_before_bias: Vec<f32>,
    selected_output: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum MlpReplayPolicy {
    Current,
    WeightBf16,
    F32Input,
    OutputF16,
}

#[derive(Clone, Copy)]
enum Expert30Mlp2Policy {
    Current,
    WeightBf16Round,
    WeightF16,
    F32AccumBf16Output,
    ChunkedPairwise,
    Bf16PreBiasBf16Bias,
    F32PreBiasF32Bias,
}

#[derive(Clone, Copy)]
enum SwigluReplayPolicy {
    Current,
    MulRound,
    SigmoidRound,
    Swapped,
}

#[derive(Clone, Copy)]
enum SwigluSplit {
    Interleaved,
    Half,
}

#[derive(Clone, Copy)]
enum SwigluClamp {
    Official,
    None,
    BothMinMax,
}

#[derive(Clone, Copy)]
enum SwigluRound {
    F32AllBf16Output,
    Bf16InputF32MathBf16Output,
    Bf16AfterClamp,
    Bf16SigmoidArg,
    Bf16SigmoidOutput,
    Bf16MulStage,
    Bf16UpPlusOne,
}

struct SwigluVariantSpec {
    name: &'static str,
    policy: &'static str,
    split: SwigluSplit,
    clamp: SwigluClamp,
    rounding: SwigluRound,
}

#[cfg(feature = "cuda")]
fn compute_selected_expert_trace(
    mlp_norm: &[f32],
    expert: &Mxfp4SelectedExpertWeights,
    policy: MlpReplayPolicy,
) -> SelectedExpertTrace {
    let mlp1_pre_bias =
        linear_out_in_prebias_with_policy(mlp_norm, &expert.gate_up_weight, 2880, 5760, policy);
    let mlp1_before_swiglu =
        add_bias_with_output_round(&mlp1_pre_bias, &expert.gate_up_bias, policy);
    let swiglu_before_mlp2 = compute_swiglu_bf16(&mlp1_before_swiglu);
    let mlp2_before_bias = linear_out_in_prebias_with_policy(
        &swiglu_before_mlp2,
        &expert.down_weight,
        2880,
        2880,
        policy,
    );
    let selected_output = add_bias_with_output_round(&mlp2_before_bias, &expert.down_bias, policy);
    SelectedExpertTrace {
        mlp1_before_swiglu,
        swiglu_before_mlp2,
        mlp2_before_bias,
        selected_output,
    }
}

#[cfg(feature = "cuda")]
fn linear_out_in_prebias_with_policy(
    input: &[f32],
    weight_out_in: &[f32],
    in_features: usize,
    out_features: usize,
    policy: MlpReplayPolicy,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_features];
    for out_idx in 0..out_features {
        let mut acc = 0.0f32;
        let weight_offset = out_idx * in_features;
        for in_idx in 0..in_features {
            let input_value = match policy {
                MlpReplayPolicy::F32Input => input[in_idx],
                _ => round_bf16(input[in_idx]),
            };
            let weight_value = match policy {
                MlpReplayPolicy::WeightBf16 => round_bf16(weight_out_in[weight_offset + in_idx]),
                _ => weight_out_in[weight_offset + in_idx],
            };
            acc += input_value * weight_value;
        }
        out[out_idx] = match policy {
            MlpReplayPolicy::OutputF16 => round_f16(acc),
            _ => round_bf16(acc),
        };
    }
    out
}

#[cfg(feature = "cuda")]
fn add_bias_with_output_round(pre_bias: &[f32], bias: &[f32], policy: MlpReplayPolicy) -> Vec<f32> {
    pre_bias
        .iter()
        .zip(bias.iter())
        .map(|(&value, &bias)| {
            let output = value + round_bf16(bias);
            match policy {
                MlpReplayPolicy::OutputF16 => round_f16(output),
                _ => round_bf16(output),
            }
        })
        .collect()
}

fn compute_expert30_mlp2_prebias_variant(
    swiglu: &[f32],
    down_weight: &[f32],
    policy: Expert30Mlp2Policy,
) -> Vec<f32> {
    let hidden = 2880usize;
    let mut out = vec![0.0f32; hidden];
    for out_idx in 0..hidden {
        let weight_base = out_idx * hidden;
        let sum = match policy {
            Expert30Mlp2Policy::ChunkedPairwise => {
                let mut partials = Vec::new();
                for chunk in (0..hidden).step_by(64) {
                    let mut partial = 0.0f32;
                    for in_idx in chunk..(chunk + 64).min(hidden) {
                        partial += round_bf16(swiglu[in_idx])
                            * round_bf16(down_weight[weight_base + in_idx]);
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
            _ => {
                let mut sum = 0.0f32;
                for in_idx in 0..hidden {
                    let weight = match policy {
                        Expert30Mlp2Policy::WeightBf16Round
                        | Expert30Mlp2Policy::ChunkedPairwise => {
                            round_bf16(down_weight[weight_base + in_idx])
                        }
                        Expert30Mlp2Policy::WeightF16 => {
                            round_f16(down_weight[weight_base + in_idx])
                        }
                        _ => down_weight[weight_base + in_idx],
                    };
                    sum += round_bf16(swiglu[in_idx]) * weight;
                }
                sum
            }
        };
        out[out_idx] = match policy {
            Expert30Mlp2Policy::F32PreBiasF32Bias => sum,
            _ => round_bf16(sum),
        };
    }
    out
}

fn compute_expert30_selected_output_variant(
    pre_bias: &[f32],
    down_bias: &[f32],
    policy: Expert30Mlp2Policy,
) -> Vec<f32> {
    pre_bias
        .iter()
        .zip(down_bias.iter())
        .map(|(&value, &bias)| match policy {
            Expert30Mlp2Policy::F32PreBiasF32Bias | Expert30Mlp2Policy::ChunkedPairwise => {
                round_bf16(value + bias)
            }
            _ => round_bf16(round_bf16(value) + round_bf16(bias)),
        })
        .collect()
}

fn load_optional_debug_oracle(
    path: Option<&Path>,
    expected_count: usize,
) -> Result<Option<Vec<f32>>> {
    path.map(|path| {
        load_tensor_artifact(path, &[expected_count], &["values"]).map(|(_, values)| values)
    })
    .transpose()
}

fn empty_expert30_boundary_metrics() -> Expert30BoundaryMetrics {
    Expert30BoundaryMetrics {
        mlp1_before_swiglu: None,
        swiglu_before_mlp2: None,
        mlp2_before_bias: None,
        selected_output_after_bias: empty_hidden_comparison(),
    }
}

fn finite_summary(values: &[f32]) -> FiniteSummary {
    let mut finite_count = 0usize;
    let mut non_finite_count = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &value in values {
        if value.is_finite() {
            finite_count += 1;
            min = min.min(value);
            max = max.max(value);
            sum += value as f64;
        } else {
            non_finite_count += 1;
        }
    }
    if finite_count == 0 {
        min = f32::NAN;
        max = f32::NAN;
    }
    FiniteSummary {
        len: values.len(),
        finite_count,
        non_finite_count,
        min,
        max,
        mean: if finite_count == 0 {
            f32::NAN
        } else {
            (sum / finite_count as f64) as f32
        },
    }
}

#[cfg(feature = "cuda")]
fn compute_selected_expert_output(
    mlp_norm: &[f32],
    expert: &Mxfp4SelectedExpertWeights,
) -> Vec<f32> {
    let gate_up = linear_out_in_bf16_output(
        mlp_norm,
        &expert.gate_up_weight,
        &expert.gate_up_bias,
        2880,
        5760,
    );
    let swiglu = compute_swiglu_bf16(&gate_up);
    linear_out_in_bf16_output(&swiglu, &expert.down_weight, &expert.down_bias, 2880, 2880)
}

fn linear_out_in_bf16_output(
    input: &[f32],
    weight_out_in: &[f32],
    bias: &[f32],
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_features];
    for out_idx in 0..out_features {
        let mut acc = 0.0f32;
        let weight_offset = out_idx * in_features;
        for in_idx in 0..in_features {
            acc += round_bf16(input[in_idx]) * weight_out_in[weight_offset + in_idx];
        }
        out[out_idx] = round_bf16(acc + round_bf16(bias[out_idx]));
    }
    out
}

fn compute_swiglu_bf16(gate_up: &[f32]) -> Vec<f32> {
    compute_swiglu_with_policy(gate_up, SwigluReplayPolicy::Current)
}

fn compute_swiglu_with_policy(gate_up: &[f32], policy: SwigluReplayPolicy) -> Vec<f32> {
    let intermediate = gate_up.len() / 2;
    let mut out = vec![0.0f32; intermediate];
    for idx in 0..intermediate {
        let (gate_lane, up_lane) = match policy {
            SwigluReplayPolicy::Swapped => (2 * idx + 1, 2 * idx),
            _ => (2 * idx, 2 * idx + 1),
        };
        let gate = round_bf16(gate_up[gate_lane]).min(7.0);
        let up = round_bf16(gate_up[up_lane]).clamp(-7.0, 7.0);
        let mut sigmoid = 1.0 / (1.0 + (-(1.702 * gate)).exp());
        if matches!(policy, SwigluReplayPolicy::SigmoidRound) {
            sigmoid = round_bf16(sigmoid);
        }
        let mut glu = gate * sigmoid;
        if matches!(policy, SwigluReplayPolicy::MulRound) {
            glu = round_bf16(glu);
        }
        out[idx] = round_bf16(glu * (up + 1.0));
    }
    out
}

fn compute_swiglu_policy_variants(mlp1: &[f32], oracle: &[f32]) -> Vec<SwigluVariantStatus> {
    let specs = [
        SwigluVariantSpec {
            name: "interleaved_official_f32_all_bf16_output",
            policy: "interleaved gate/up, official clamp, f32 sigmoid/multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::F32AllBf16Output,
        },
        SwigluVariantSpec {
            name: "interleaved_official_bf16_input_f32_math_bf16_output",
            policy: "interleaved gate/up, BF16 input slices, official clamp, f32 sigmoid/multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16InputF32MathBf16Output,
        },
        SwigluVariantSpec {
            name: "interleaved_official_bf16_after_clamp",
            policy: "interleaved gate/up, BF16 input and BF16-round after clamp, f32 sigmoid/multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16AfterClamp,
        },
        SwigluVariantSpec {
            name: "interleaved_official_bf16_sigmoid_arg",
            policy: "interleaved gate/up, official clamp, BF16-round 1.702*gate before sigmoid, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16SigmoidArg,
        },
        SwigluVariantSpec {
            name: "interleaved_official_bf16_sigmoid_output",
            policy: "interleaved gate/up, official clamp, BF16-round sigmoid output before multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16SigmoidOutput,
        },
        SwigluVariantSpec {
            name: "interleaved_official_bf16_mul_stage",
            policy: "interleaved gate/up, official clamp, BF16-round gate*sigmoid before final multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16MulStage,
        },
        SwigluVariantSpec {
            name: "interleaved_official_bf16_up_plus_one",
            policy: "interleaved gate/up, official clamp, BF16-round up+1 before final multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16UpPlusOne,
        },
        SwigluVariantSpec {
            name: "interleaved_no_clamp_bf16_input_f32_math_bf16_output",
            policy: "interleaved gate/up, no clamp, BF16 input slices, f32 sigmoid/multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::None,
            rounding: SwigluRound::Bf16InputF32MathBf16Output,
        },
        SwigluVariantSpec {
            name: "interleaved_clamp_both_minmax_bf16_input_f32_math_bf16_output",
            policy: "interleaved gate/up, clamp both gate and up to [-7,7], BF16 input slices, f32 sigmoid/multiply, BF16 output",
            split: SwigluSplit::Interleaved,
            clamp: SwigluClamp::BothMinMax,
            rounding: SwigluRound::Bf16InputF32MathBf16Output,
        },
        SwigluVariantSpec {
            name: "half_split_official_bf16_input_f32_math_bf16_output",
            policy: "half-split gate/up, official clamp, BF16 input slices, f32 sigmoid/multiply, BF16 output",
            split: SwigluSplit::Half,
            clamp: SwigluClamp::Official,
            rounding: SwigluRound::Bf16InputF32MathBf16Output,
        },
    ];

    specs
        .iter()
        .map(|spec| {
            let output = compute_swiglu_variant(mlp1, spec);
            SwigluVariantStatus {
                name: spec.name,
                policy: spec.policy,
                metric: compare_hidden(&output, oracle),
            }
        })
        .collect()
}

fn compute_swiglu_variant(mlp1: &[f32], spec: &SwigluVariantSpec) -> Vec<f32> {
    let intermediate = mlp1.len() / 2;
    let mut out = vec![0.0f32; intermediate];
    for idx in 0..intermediate {
        let (gate_lane, up_lane) = match spec.split {
            SwigluSplit::Interleaved => (2 * idx, 2 * idx + 1),
            SwigluSplit::Half => (idx, idx + intermediate),
        };
        let mut gate = mlp1[gate_lane];
        let mut up = mlp1[up_lane];
        if !matches!(spec.rounding, SwigluRound::F32AllBf16Output) {
            gate = round_bf16(gate);
            up = round_bf16(up);
        }
        match spec.clamp {
            SwigluClamp::Official => {
                gate = gate.min(7.0);
                up = up.clamp(-7.0, 7.0);
            }
            SwigluClamp::None => {}
            SwigluClamp::BothMinMax => {
                gate = gate.clamp(-7.0, 7.0);
                up = up.clamp(-7.0, 7.0);
            }
        }
        if matches!(spec.rounding, SwigluRound::Bf16AfterClamp) {
            gate = round_bf16(gate);
            up = round_bf16(up);
        }

        let mut sigmoid_arg = 1.702 * gate;
        if matches!(spec.rounding, SwigluRound::Bf16SigmoidArg) {
            sigmoid_arg = round_bf16(sigmoid_arg);
        }
        let mut sigmoid = 1.0 / (1.0 + (-sigmoid_arg).exp());
        if matches!(spec.rounding, SwigluRound::Bf16SigmoidOutput) {
            sigmoid = round_bf16(sigmoid);
        }
        let mut gate_sigmoid = gate * sigmoid;
        if matches!(spec.rounding, SwigluRound::Bf16MulStage) {
            gate_sigmoid = round_bf16(gate_sigmoid);
        }
        let mut up_plus_one = up + 1.0;
        if matches!(spec.rounding, SwigluRound::Bf16UpPlusOne) {
            up_plus_one = round_bf16(up_plus_one);
        }
        out[idx] = round_bf16(gate_sigmoid * up_plus_one);
    }
    out
}

fn classify_swiglu_policy(
    variants: &[SwigluVariantStatus],
    best: &SwigluVariantStatus,
) -> &'static str {
    if best.metric.metrics.mismatches == 0 {
        return "swiglu_policy_matches_oracle";
    }
    let half_split_best = variants
        .iter()
        .filter(|variant| variant.name.starts_with("half_split"))
        .map(|variant| variant.metric.metrics.mismatches)
        .min()
        .unwrap_or(usize::MAX);
    let interleaved_best = variants
        .iter()
        .filter(|variant| variant.name.starts_with("interleaved"))
        .map(|variant| variant.metric.metrics.mismatches)
        .min()
        .unwrap_or(usize::MAX);
    if half_split_best < interleaved_best {
        return "swiglu_split_policy_mismatch";
    }
    if best.name.contains("no_clamp") || best.name.contains("clamp_both") {
        return "swiglu_clamp_policy_mismatch";
    }
    if best.name.contains("bf16") || best.name.contains("f32") {
        return "swiglu_dtype_rounding_policy_mismatch";
    }
    "swiglu_policy_unresolved"
}

fn compare_selected_experts(
    actual: &[f32],
    expected: &[f32],
    selected_experts: &[usize],
) -> SelectedExpertsComparisonStatus {
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
            let rank = idx / 2880;
            let diff = SelectedExpertsDiff {
                rank,
                expert: selected_experts[rank],
                hidden_lane: idx % 2880,
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
    SelectedExpertsComparisonStatus {
        metrics: SelectedExpertsMetrics {
            max_abs_diff,
            mean_abs_diff: (sum_abs_diff / len as f64) as f32,
            mismatches,
        },
        first_mismatch,
        worst_mismatch,
    }
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
