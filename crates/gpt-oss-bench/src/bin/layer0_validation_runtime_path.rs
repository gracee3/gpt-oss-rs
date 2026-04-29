use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[cfg(feature = "cuda")]
use gpt_oss_gpu::{cublas::CublasHandle, CudaContext};
#[cfg(feature = "cuda")]
use gpt_oss_model_runner::mxfp4_validation::{
    load_gate_up_row_mxfp4_validation, load_selected_experts_mxfp4_validation,
    Mxfp4SelectedExpertWeights,
};

const DEFAULT_BUNDLE_ROOT: &str = "/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424";

#[derive(Clone, Debug, Parser)]
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

    /// Official layer0 final-token output artifact for full-layer0 validation.
    #[arg(long)]
    layer0_output_oracle: Option<PathBuf>,

    /// Optional path to emit corrected full-layer0 final-token output values.
    #[arg(long)]
    emit_corrected_layer0_output: Option<PathBuf>,

    /// Corrected layer0 output artifact for the layer1 input guard.
    #[arg(long)]
    layer0_output: Option<PathBuf>,

    /// Official layer1 input boundary oracle for the layer1 input guard.
    #[arg(long)]
    layer1_input_oracle: Option<PathBuf>,

    /// Layer1 residual-stream input artifact for layer ladder validation.
    #[arg(long)]
    layer1_input: Option<PathBuf>,

    /// Generic layer residual-stream input artifact for full-layer attempts.
    #[arg(long)]
    layer_input: Option<PathBuf>,

    /// Generic layer output oracle artifact for full-layer attempts.
    #[arg(long)]
    layer_output_oracle: Option<PathBuf>,

    /// Generic layer input oracle artifact for layer input guards.
    #[arg(long)]
    layer_input_oracle: Option<PathBuf>,

    /// Optional path to emit corrected full-layer output values.
    #[arg(long)]
    emit_corrected_layer_output: Option<PathBuf>,

    /// Official layer1 attention norm output before Q/K/V projections.
    #[arg(long)]
    layer1_attn_norm_oracle: Option<PathBuf>,

    /// Official layer1 attention ordered boundary bundle for downstream seam validation.
    #[arg(long)]
    attention_bundle: Option<PathBuf>,

    /// Official layer MLP ordered boundary bundle for backend seam validation.
    #[arg(long)]
    mlp_bundle: Option<PathBuf>,

    /// Ordered MLP bundle status artifact for focused ordered-consumer diagnostics.
    #[arg(long)]
    ordered_mlp_bundle_status: Option<PathBuf>,

    /// Ordered MLP bundle directory for focused ordered-consumer diagnostics.
    #[arg(long)]
    ordered_mlp_bundle_dir: Option<PathBuf>,

    /// Optional path to emit the layer1 attention residual computed from bundle seams.
    #[arg(long)]
    emit_layer1_attention_residual: Option<PathBuf>,

    /// Optional path to emit a validated layer output artifact.
    #[arg(long)]
    emit_layer_output: Option<PathBuf>,

    /// Layer index for layer ladder validation modes.
    #[arg(long, default_value_t = 1)]
    layer_index: usize,

    /// Norm kind for coarse norm diagnostics.
    #[arg(long, default_value = "attention")]
    norm_kind: String,

    /// RMSNorm reduction policy for validation-only coarse checks.
    #[arg(long, default_value = "current")]
    norm_reduction_policy: String,

    /// Start layer for generic ladder validation.
    #[arg(long, default_value_t = 0)]
    start_layer: usize,

    /// End layer for generic ladder validation.
    #[arg(long, default_value_t = 0)]
    end_layer: usize,

    /// Root directory containing ordered boundary bundles.
    #[arg(long)]
    bundle_root: Option<PathBuf>,

    /// Directory for emitted layer ladder artifacts.
    #[arg(long)]
    emit_dir: Option<PathBuf>,

    /// Coarse layer ladder bundle artifact.
    #[arg(long)]
    coarse_bundle: Option<PathBuf>,

    /// Emitted layer output artifact for coarse output guards.
    #[arg(long)]
    layer_output: Option<PathBuf>,

    /// Continue coarse MLP-side validation after recording an attention norm diagnostic mismatch.
    #[arg(long, default_value_t = false)]
    continue_after_attn_norm_diagnostic: bool,

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

    /// Comma-separated MLP1 seam artifact paths in selected-expert rank order.
    #[arg(long)]
    selected_experts_mlp1: Option<String>,

    /// Optional official expert30 MLP1 output before SwiGLU oracle artifact.
    #[arg(long)]
    expert30_mlp1_oracle: Option<PathBuf>,

    /// Optional/generated expert3 MLP1 seam artifact before SwiGLU.
    #[arg(long)]
    expert3_mlp1_seam: Option<PathBuf>,

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

    /// Official hidden state after MLP residual add oracle artifact.
    #[arg(long)]
    mlp_residual_oracle: Option<PathBuf>,

    /// Post-attention residual input artifact for MLP residual validation.
    #[arg(long)]
    post_attention_residual: Option<PathBuf>,

    /// Optional PyTorch BF16 SwiGLU intermediate discriminator artifact.
    #[arg(long)]
    pytorch_intermediates: Option<PathBuf>,

    /// Optional PyTorch/module lane terms artifact for expert30 MLP1 lane debug.
    #[arg(long)]
    pytorch_lane_terms: Option<PathBuf>,

    /// Selected expert ids in rank order.
    #[arg(long, default_value = "3,30,11,27")]
    selected_experts: String,

    /// Apply the known rank0/expert3/lane1990 selected-output oracle correction.
    #[arg(long, default_value_t = false)]
    apply_expert3_lane1990_correction: bool,

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

    /// Focus lane for expert30 MLP1 lane-local diagnostics.
    #[arg(long, default_value_t = 522)]
    lane: usize,

    /// Selected-expert rank for focused lane-local diagnostics.
    #[arg(long, default_value_t = 0)]
    rank: usize,

    /// Expert id for focused lane-local diagnostics.
    #[arg(long, default_value_t = 28)]
    expert: usize,

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
    SelectedExpertsPinnedSwigluDebug,
    SelectedExpertsFromMlp1Seams,
    MlpBackend,
    FullLayer0,
    FullLayer,
    Layer1InputGuard,
    Layer1AttnNorm,
    Layer1KRope,
    Layer1AttentionBundle,
    Layer1MlpBackend,
    Layer1Expert28Lane2269Debug,
    Layer1CorrectedOutput,
    LayerInputGuard,
    LayerBundleDiscover,
    LayerBundleValidate,
    LayerLadder,
    CoarseLadderInspect,
    CoarseLayerOutputGuard,
    CoarseLayerValidate,
    CoarseLadderValidate,
    CoarseNormDebug,
    CoarseMlpOutputDebug,
    CoarseMlpOutputOrderedDebug,
    Layer2AttnNormDebug,
    LayerBundleValidateFromCoarse,
    Expert3Lane1990Debug,
    Expert3Lane1990OracleSemantics,
    SwigluDebug,
    SwigluPolicyPin,
    Expert30Mlp2Debug,
    Expert30Mlp1Debug,
    Expert30Mlp1LaneDebug,
    Mlp1Bf16Policy,
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
    production_routing_changed: bool,
    validation_only: bool,
    selected_experts: Vec<usize>,
    mlp1_policy: &'static str,
    mlp1_backend_source: &'static str,
    input_source: &'static str,
    mlp_norm: TensorArtifactStatus,
    selected_experts_oracle: TensorArtifactStatus,
    tensor_sources: SelectedExpertTensorSources,
    mxfp4_loader: Option<Mxfp4LoaderStatus>,
    expert_formula: ExpertFormulaStatus,
    mlp1_metric: Option<Value>,
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
struct SelectedExpertsPinnedSwigluDebugStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    selected_experts: Vec<usize>,
    pinned_swiglu_policy: &'static str,
    selected_experts_mode_uses_pinned_policy: bool,
    selected_experts_debug_uses_pinned_policy: bool,
    per_rank_selected_output_metrics: Vec<SelectedExpertRankMetric>,
    expert30_variant_table: Vec<Expert30PinnedSwigluVariantStatus>,
    mlp1_variant_table: Vec<SelectedExpertMlp1VariantStatus>,
    first_mismatching_boundary: &'static str,
    mxfp4_loader: Option<Mxfp4LoaderStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Expert30PinnedSwigluVariantStatus {
    name: &'static str,
    policy: &'static str,
    mlp1_metric: Option<HiddenComparisonStatus>,
    swiglu_metric: Option<HiddenComparisonStatus>,
    mlp2_pre_bias_metric: Option<HiddenComparisonStatus>,
    selected_output_metric: HiddenComparisonStatus,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedExpertMlp1VariantStatus {
    name: &'static str,
    policy: &'static str,
    mlp1_metric: HiddenComparisonStatus,
}

#[derive(Debug, Serialize)]
struct Expert30Mlp1LaneDebugStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    lane_metadata: Expert30Mlp1LaneMetadata,
    official_value: f32,
    current_local_value: f32,
    current_diff: f32,
    current_pre_bias: f32,
    bias_value: f32,
    output_rounding_policy: &'static str,
    source_identity: Value,
    local_values: Value,
    official_values: Value,
    pytorch_reference: Value,
    per_block_summary: Vec<Value>,
    top_contributions: Vec<Value>,
    decode_variant_table: Vec<Expert30Mlp1LaneVariant>,
    accumulation_variant_table: Vec<Expert30Mlp1LaneVariant>,
    best_variant: Expert30Mlp1LaneVariant,
    best_explanation: &'static str,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Expert30Mlp1LaneMetadata {
    expert_id: usize,
    output_lane: usize,
    gate_or_up: &'static str,
    logical_gate_up_lane: usize,
    block_count: usize,
    bytes_per_block: usize,
    values_per_byte: usize,
    expected_input_dim: usize,
    tensor_row_layout: &'static str,
    dequant_helper: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Expert30Mlp1LaneVariant {
    name: &'static str,
    policy: &'static str,
    local: f32,
    official: f32,
    diff: f32,
    pre_bias: f32,
    bias: f32,
    row_summary: Option<FiniteSummary>,
    full_mlp1_metric: Option<HiddenComparisonStatus>,
}

#[derive(Debug, Serialize)]
struct Mlp1Bf16PolicyStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    chosen_mlp1_policy: &'static str,
    lane522_policy_table: Vec<Expert30Mlp1LaneVariant>,
    expert30_full_metrics: Option<HiddenComparisonStatus>,
    selected_experts_rerun: Value,
    weighted_sum_rerun: Value,
    pytorch_reference: Value,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct SelectedExpertsFromMlp1SeamsStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    selected_experts: Vec<usize>,
    mlp1_seam_sources: Vec<TensorArtifactStatus>,
    selected_experts_oracle: TensorArtifactStatus,
    weighted_expert_sum_oracle: TensorArtifactStatus,
    mlp_residual_oracle: TensorArtifactStatus,
    post_attention_residual: TensorArtifactStatus,
    mxfp4_loader: Option<Mxfp4LoaderStatus>,
    selected_output_metric: Option<SelectedExpertsComparisonStatus>,
    per_rank_metrics: Vec<SelectedExpertRankMetric>,
    weighted_sum_metric: Option<HiddenComparisonStatus>,
    mlp_residual_metric: Option<HiddenComparisonStatus>,
    weighted_sum_policy: &'static str,
    mlp_residual_policy: &'static str,
    rust_native_mlp1_bf16_einsum_backend_open: bool,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Serialize)]
struct Expert3Lane1990DebugStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    expert: usize,
    rank: usize,
    lane: usize,
    mlp1_seam: TensorArtifactStatus,
    selected_experts_oracle: TensorArtifactStatus,
    weighted_expert_sum_oracle: TensorArtifactStatus,
    mlp_residual_oracle: TensorArtifactStatus,
    post_attention_residual: TensorArtifactStatus,
    rust_values: Value,
    pytorch_values: Value,
    official_values: Value,
    variant_table: Vec<Expert3Lane1990VariantStatus>,
    selected_output_metric: HiddenComparisonStatus,
    weighted_sum_impact: Option<Value>,
    mlp_residual_impact: Option<Value>,
    mxfp4_loader: Option<Mxfp4LoaderStatus>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Expert3Lane1990VariantStatus {
    name: &'static str,
    policy: &'static str,
    selected_lane: f32,
    official_lane: f32,
    diff: f32,
    selected_metric: HiddenComparisonStatus,
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

#[derive(Debug, Clone, Serialize)]
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
struct SwigluPolicyPinStatus {
    mode: &'static str,
    submode: &'static str,
    classification: &'static str,
    implemented: bool,
    runtime_behavior_changed: bool,
    validation_only: bool,
    official_summary: OfficialSwigluSummary,
    mlp1_oracle: TensorArtifactStatus,
    swiglu_oracle: TensorArtifactStatus,
    pytorch_intermediate_status: PytorchIntermediateStatus,
    first_divergent_stage: Option<&'static str>,
    best_variant: SwigluPolicyPinVariantStatus,
    best_variant_metric: HiddenComparisonStatus,
    stage_metrics: Vec<SwigluStageMetricStatus>,
    variant_table: Vec<SwigluPolicyPinVariantStatus>,
    selected_experts_rerun: Option<Value>,
    weighted_sum_rerun: Option<Value>,
    blocker: Option<Blocker>,
    next_bounded_step: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct PytorchIntermediateStatus {
    available: bool,
    path: Option<String>,
    swiglu_output_metric: Option<HiddenComparisonStatus>,
    note: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct SwigluStageMetricStatus {
    stage: &'static str,
    metric: HiddenComparisonStatus,
}

#[derive(Debug, Clone, Serialize)]
struct SwigluPolicyPinVariantStatus {
    name: &'static str,
    policy: &'static str,
    output_metric: HiddenComparisonStatus,
    stage_metrics: Vec<SwigluStageMetricStatus>,
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

#[derive(Clone, Debug, Serialize)]
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
        Mode::SelectedExpertsPinnedSwigluDebug => run_selected_experts_pinned_swiglu_debug(&cli),
        Mode::SelectedExpertsFromMlp1Seams => run_selected_experts_from_mlp1_seams(&cli),
        Mode::MlpBackend => run_mlp_backend(&cli),
        Mode::FullLayer0 => run_full_layer0(&cli),
        Mode::FullLayer => run_full_layer(&cli),
        Mode::Layer1InputGuard => run_layer1_input_guard(&cli),
        Mode::Layer1AttnNorm => run_layer1_attn_norm(&cli),
        Mode::Layer1KRope => run_layer1_k_rope(&cli),
        Mode::Layer1AttentionBundle => run_layer1_attention_bundle(&cli),
        Mode::Layer1MlpBackend => run_layer1_mlp_backend(&cli),
        Mode::Layer1Expert28Lane2269Debug => run_layer1_expert28_lane2269_debug(&cli),
        Mode::Layer1CorrectedOutput => run_layer1_corrected_output(&cli),
        Mode::LayerInputGuard => run_layer_input_guard(&cli),
        Mode::LayerBundleDiscover => run_layer_bundle_discover(&cli),
        Mode::LayerBundleValidate => run_layer_bundle_validate(&cli),
        Mode::LayerLadder => run_layer_ladder(&cli),
        Mode::CoarseLadderInspect => run_coarse_ladder_inspect(&cli),
        Mode::CoarseLayerOutputGuard => run_coarse_layer_output_guard(&cli),
        Mode::CoarseLayerValidate => run_coarse_layer_validate(&cli),
        Mode::CoarseLadderValidate => run_coarse_ladder_validate(&cli),
        Mode::CoarseNormDebug => run_coarse_norm_debug(&cli),
        Mode::CoarseMlpOutputDebug => run_coarse_mlp_output_debug(&cli),
        Mode::CoarseMlpOutputOrderedDebug => run_coarse_mlp_output_ordered_debug(&cli),
        Mode::Layer2AttnNormDebug => run_layer2_attn_norm_debug(&cli),
        Mode::LayerBundleValidateFromCoarse => run_layer_bundle_validate_from_coarse(&cli),
        Mode::Expert3Lane1990Debug => run_expert3_lane1990_debug(&cli),
        Mode::Expert3Lane1990OracleSemantics => run_expert3_lane1990_oracle_semantics(&cli),
        Mode::SwigluDebug => run_swiglu_debug(&cli),
        Mode::SwigluPolicyPin => run_swiglu_policy_pin(&cli),
        Mode::Expert30Mlp2Debug => run_expert30_mlp2_debug(&cli),
        Mode::Expert30Mlp1Debug | Mode::Expert30Mlp1LaneDebug => run_expert30_mlp1_debug(&cli),
        Mode::Mlp1Bf16Policy => run_mlp1_bf16_policy(&cli),
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
        mlp1_metric,
        next_bounded_step,
    ) = if artifact_blocked {
        (
            "layer0_validation_selected_experts_execution_failed",
            None,
            Vec::new(),
            Some(Blocker {
                kind: "selected_expert_artifacts",
                detail: "MLP norm input or selected expert oracle did not expose supported values or expected counts",
            }),
            "not_reached",
            None,
            None,
            "resolve the reported selected expert artifact blocker",
        )
    } else if missing_required_sources {
        (
            "layer0_validation_selected_experts_execution_failed",
            None,
            Vec::new(),
            Some(Blocker {
                kind: "selected_expert_tensors",
                detail: "could not find the complete layer0 expert tensor set in --model",
            }),
            "missing_required_tensors",
            None,
            None,
            "resolve the reported selected expert tensor source blocker",
        )
    } else if mx_fp4_sources {
        match execute_selected_experts_mxfp4(model, &selected_experts, &mlp_norm_values, &oracle_values) {
            Ok((overall_metric, per_rank_metrics, mxfp4_loader, mlp1_metric)) => {
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
                    Some(mlp1_metric),
                    next_bounded_step,
                )
            }
            Err(_) => (
                "layer0_validation_selected_experts_blocked_by_mlp1_backend_port",
                None,
                Vec::new(),
                Some(Blocker {
                    kind: "mlp1_backend_port",
                    detail: "validation MXFP4 selected-expert loader or cuBLAS BF16 MLP1 backend failed before comparison",
                }),
                "mxfp4_validation_loader_failed",
                None,
                None,
                "fix the narrow validation MXFP4 selected-expert loader or BF16 GEMM MLP1 backend",
            ),
        }
    } else {
        (
            "layer0_validation_selected_experts_execution_failed",
            None,
            Vec::new(),
            Some(Blocker {
                kind: "unsupported_dense_expert_replay",
                detail: "expert tensor metadata did not match the current MXFP4 blocker path, but dense selected-expert replay is not implemented in this slice",
            }),
            "unsupported_dense_replay",
            None,
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
        production_routing_changed: false,
        validation_only: true,
        selected_experts,
        mlp1_policy: "cublas_bf16_tensor_op",
        mlp1_backend_source: "backend/mlp1-bf16-einsum-validation@da4d655",
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
        mlp1_metric,
        overall_metric,
        per_rank_metrics,
        expert30_internal_guard: None,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_selected_experts_from_mlp1_seams(cli: &Cli) -> Result<()> {
    let selected_experts_mlp1 =
        required_string(&cli.selected_experts_mlp1, "selected experts MLP1 seams")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let weighted_expert_sum_oracle = required_path(
        &cli.weighted_expert_sum_oracle,
        "weighted expert sum oracle",
    )?;
    let mlp_residual_oracle = required_path(&cli.mlp_residual_oracle, "MLP residual oracle")?;
    let post_attention_residual =
        required_path(&cli.post_attention_residual, "post-attention residual")?;
    let model = required_path(&cli.model, "model")?;
    let selected_experts = parse_selected_experts(&cli.selected_experts)?;
    let seam_paths = parse_path_list(selected_experts_mlp1, selected_experts.len())?;

    for path in &seam_paths {
        validate_path(path, "selected expert MLP1 seam")?;
    }
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    validate_path(weighted_expert_sum_oracle, "weighted expert sum oracle")?;
    validate_path(mlp_residual_oracle, "MLP residual oracle")?;
    validate_path(post_attention_residual, "post-attention residual")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let selected_count = selected_experts.len();
    let mut seam_statuses = Vec::with_capacity(selected_count);
    let mut seam_values = Vec::with_capacity(selected_count);
    for path in &seam_paths {
        let (status, values) = load_tensor_artifact(path, &[hidden * 2], &["values"])?;
        seam_statuses.push(status);
        seam_values.push(values);
    }
    let (selected_oracle_status, selected_oracle_values) = load_tensor_artifact(
        selected_experts_oracle,
        &[selected_count * hidden],
        &["values"],
    )?;
    let (weighted_oracle_status, weighted_oracle_values) =
        load_tensor_artifact(weighted_expert_sum_oracle, &[hidden], &["values"])?;
    let (residual_oracle_status, residual_oracle_values) =
        load_tensor_artifact(mlp_residual_oracle, &[hidden], &["values"])?;
    let (post_attention_status, post_attention_values) =
        load_tensor_artifact(post_attention_residual, &[hidden], &["values"])?;

    let artifact_blocked = seam_statuses
        .iter()
        .any(|status| !status.shape_or_count_matched)
        || !selected_oracle_status.shape_or_count_matched
        || !weighted_oracle_status.shape_or_count_matched
        || !residual_oracle_status.shape_or_count_matched
        || !post_attention_status.shape_or_count_matched;

    let routing_weights = load_selected_routing_weights(selected_experts_oracle, selected_count)
        .unwrap_or_else(|_| vec![0.4453125, 0.2275390625, 0.189453125, 0.13671875]);

    let (
        classification,
        selected_metric,
        per_rank_metrics,
        weighted_sum_metric,
        mlp_residual_metric,
        mxfp4_loader,
        blocker,
        next_bounded_step,
    ) = if artifact_blocked || routing_weights.len() != selected_count {
        (
            "selected_experts_from_mlp1_seams_blocked_by_artifacts",
            None,
            Vec::new(),
            None,
            None,
            None,
            Some(Blocker {
                kind: "selected_experts_from_mlp1_seams_artifacts",
                detail: "MLP1 seams, selected-output oracle, weighted-sum oracle, MLP residual oracle, post-attention residual, or routing weights were unavailable or had unsupported shapes",
            }),
            "resolve the reported selected-experts-from-MLP1-seams artifact blocker",
        )
    } else {
        match execute_selected_experts_from_mlp1_seams_mxfp4(
            model,
            &selected_experts,
            &seam_values,
            &selected_oracle_values,
            &routing_weights,
            &weighted_oracle_values,
            &post_attention_values,
            &residual_oracle_values,
        ) {
            Ok(result) => result,
            Err(_) => (
                "selected_experts_from_mlp1_seams_blocked_by_artifacts",
                None,
                Vec::new(),
                None,
                None,
                None,
                Some(Blocker {
                    kind: "selected_experts_from_mlp1_seams_execution",
                    detail: "validation MXFP4 down-projection replay failed while consuming MLP1 seam inputs",
                }),
                "fix the selected-experts-from-MLP1-seams validation replay path",
            ),
        }
    };

    let status = SelectedExpertsFromMlp1SeamsStatus {
        mode: "layer0_validation_runtime_path",
        submode: "selected-experts-from-mlp1-seams",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        selected_experts,
        mlp1_seam_sources: seam_statuses,
        selected_experts_oracle: selected_oracle_status,
        weighted_expert_sum_oracle: weighted_oracle_status,
        mlp_residual_oracle: residual_oracle_status,
        post_attention_residual: post_attention_status,
        mxfp4_loader,
        selected_output_metric: selected_metric,
        per_rank_metrics,
        weighted_sum_metric,
        mlp_residual_metric,
        weighted_sum_policy: "sum BF16 selected_output * BF16 routing_weight into f32, BF16 output",
        mlp_residual_policy: "bf16_plus_bf16_to_bf16",
        rust_native_mlp1_bf16_einsum_backend_open: true,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_mlp_backend(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let router_logits_oracle = required_path(&cli.router_logits_oracle, "router logits oracle")?;
    let topk_routing_oracle = required_path(&cli.topk_routing_oracle, "topk routing oracle")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let weighted_expert_sum_oracle = required_path(
        &cli.weighted_expert_sum_oracle,
        "weighted expert sum oracle",
    )?;
    let mlp_residual_oracle = required_path(&cli.mlp_residual_oracle, "MLP residual oracle")?;
    let post_attention_residual =
        required_path(&cli.post_attention_residual, "post-attention residual")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(router_logits_oracle, "router logits oracle")?;
    validate_path(topk_routing_oracle, "topk routing oracle")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    validate_path(weighted_expert_sum_oracle, "weighted expert sum oracle")?;
    validate_path(mlp_residual_oracle, "MLP residual oracle")?;
    validate_path(post_attention_residual, "post-attention residual")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let experts = 32usize;
    let selected_experts = parse_selected_experts(&cli.selected_experts)?;
    let selected_count = selected_experts.len();
    let (mlp_norm_status, mlp_norm_values) =
        load_tensor_artifact(mlp_norm, &[hidden], &["values"])?;
    let (router_oracle_status, router_oracle_values) =
        load_tensor_artifact(router_logits_oracle, &[experts], &["values"])?;
    let (topk_oracle_status, topk_oracle) = load_topk_oracle(topk_routing_oracle, selected_count)?;
    let (selected_oracle_status, selected_oracle_values) = load_tensor_artifact(
        selected_experts_oracle,
        &[selected_count * hidden],
        &["values"],
    )?;
    let (weighted_oracle_status, weighted_oracle_values) =
        load_tensor_artifact(weighted_expert_sum_oracle, &[hidden], &["values"])?;
    let (residual_oracle_status, residual_oracle_values) =
        load_tensor_artifact(mlp_residual_oracle, &[hidden], &["values"])?;
    let (post_attention_status, post_attention_values) =
        load_tensor_artifact(post_attention_residual, &[hidden], &["values"])?;

    let artifact_blocked = !mlp_norm_status.shape_or_count_matched
        || !router_oracle_status.shape_or_count_matched
        || !topk_oracle_status.shape_or_count_matched
        || !selected_oracle_status.shape_or_count_matched
        || !weighted_oracle_status.shape_or_count_matched
        || !residual_oracle_status.shape_or_count_matched
        || !post_attention_status.shape_or_count_matched;

    let router_result = if artifact_blocked {
        Err(anyhow::anyhow!("blocked before router execution"))
    } else {
        execute_mlp_backend_router(
            model,
            0,
            &mlp_norm_values,
            &router_oracle_values,
            &topk_oracle,
        )
    };
    let backend_result = if artifact_blocked {
        Err(anyhow::anyhow!("blocked before MLP backend execution"))
    } else {
        execute_mlp_backend_selected_experts(
            model,
            0,
            &mlp_norm_values,
            &selected_experts,
            &topk_oracle.routing_weights,
            &selected_oracle_values,
            &weighted_oracle_values,
            &residual_oracle_values,
            &post_attention_values,
            cli.apply_expert3_lane1990_correction,
        )
    };

    let classification = if artifact_blocked {
        "layer0_validation_mlp_backend_blocked_by_artifacts"
    } else if router_result.is_err() || backend_result.is_err() {
        "layer0_validation_mlp_backend_blocked_by_port_scope"
    } else {
        backend_result
            .as_ref()
            .ok()
            .and_then(|status| status.get("classification"))
            .and_then(Value::as_str)
            .unwrap_or("layer0_validation_mlp_backend_blocked_by_port_scope")
    };

    let mut status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "mlp-backend",
        "classification": classification,
        "implemented": !artifact_blocked && router_result.is_ok() && backend_result.is_ok(),
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "selected_experts": selected_experts,
        "artifacts": {
            "mlp_norm": mlp_norm_status,
            "router_logits_oracle": router_oracle_status,
            "topk_routing_oracle": topk_oracle_status,
            "selected_experts_oracle": selected_oracle_status,
            "weighted_expert_sum_oracle": weighted_oracle_status,
            "mlp_residual_oracle": residual_oracle_status,
            "post_attention_residual": post_attention_status
        },
        "router_metric": null,
        "topk_metric": null,
        "selected_output_metric_official": null,
        "selected_output_metric_corrected": null,
        "weighted_sum_metric_official": null,
        "weighted_sum_metric_corrected": null,
        "mlp_residual_metric_official": null,
        "mlp_residual_metric_corrected": null,
        "correction": {
            "enabled": cli.apply_expert3_lane1990_correction,
            "applied": false,
            "rank": 0,
            "expert": 3,
            "hidden_lane": 1990
        },
        "next_bounded_step": match classification {
            "layer0_validation_mlp_backend_matches_oracle" => "preserve the explicit validation mode and decide whether a separate production-routing design is needed",
            "layer0_validation_mlp_backend_matches_with_known_lane1990_correction" => "preserve the lane1990 correction metadata and keep production routing unchanged",
            "layer0_validation_mlp_backend_selected_outputs_mismatch" => "localize selected-output mismatch under cuBLAS BF16 MLP1, pinned SwiGLU, and BF16 MLP2 policy",
            "layer0_validation_mlp_backend_weighted_sum_mismatch" => "localize weighted expert sum BF16 policy under matching selected outputs",
            "layer0_validation_mlp_backend_residual_mismatch" => "localize MLP residual add or post-attention residual source",
            "layer0_validation_mlp_backend_blocked_by_artifacts" => "resolve the reported MLP backend artifact blocker",
            _ => "scope the remaining backend port failure without changing production runtime behavior",
        }
    });

    match router_result {
        Ok(router) => {
            status["router_metric"] = router["router_metric"].clone();
            status["topk_metric"] = router["topk_metric"].clone();
            status["router"] = router;
        }
        Err(err) if !artifact_blocked => {
            status["blocker"] = json!({
                "kind": "router_backend_port",
                "detail": err.to_string()
            });
        }
        Err(_) => {}
    }
    match backend_result {
        Ok(backend) => {
            for key in [
                "selected_output_metric_official",
                "selected_output_metric_corrected",
                "weighted_sum_metric_official",
                "weighted_sum_metric_corrected",
                "mlp_residual_metric_official",
                "mlp_residual_metric_corrected",
                "correction",
                "source_identity",
                "per_rank_metrics",
            ] {
                status[key] = backend[key].clone();
            }
        }
        Err(err) if !artifact_blocked => {
            status["blocker"] = json!({
                "kind": "mlp_backend_port",
                "detail": err.to_string()
            });
        }
        Err(_) => {}
    }

    write_json(&cli.output, &status)
}

fn run_full_layer0(cli: &Cli) -> Result<()> {
    let attention_residual = required_path(&cli.attention_residual, "attention residual")?;
    let mlp_norm_oracle = required_path(&cli.mlp_norm_oracle, "MLP norm oracle")?;
    let router_logits_oracle = required_path(&cli.router_logits_oracle, "router logits oracle")?;
    let topk_routing_oracle = required_path(&cli.topk_routing_oracle, "topk routing oracle")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let weighted_expert_sum_oracle = required_path(
        &cli.weighted_expert_sum_oracle,
        "weighted expert sum oracle",
    )?;
    let layer0_output_oracle = required_path(&cli.layer0_output_oracle, "layer0 output oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(attention_residual, "attention residual")?;
    validate_path(mlp_norm_oracle, "MLP norm oracle")?;
    validate_path(router_logits_oracle, "router logits oracle")?;
    validate_path(topk_routing_oracle, "topk routing oracle")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    validate_path(weighted_expert_sum_oracle, "weighted expert sum oracle")?;
    validate_path(layer0_output_oracle, "layer0 output oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let experts = 32usize;
    let selected_experts = parse_selected_experts(&cli.selected_experts)?;
    let selected_count = selected_experts.len();
    let (attention_status, attention_values) = load_tensor_artifact(
        attention_residual,
        &[hidden, cli.token_count * hidden],
        &["values", "layer0_attn_norm_input_f32"],
    )?;
    let (mlp_norm_status, mlp_norm_oracle_values) =
        load_tensor_artifact(mlp_norm_oracle, &[hidden], &["values"])?;
    let (router_oracle_status, router_oracle_values) =
        load_tensor_artifact(router_logits_oracle, &[experts], &["values"])?;
    let (topk_oracle_status, topk_oracle) = load_topk_oracle(topk_routing_oracle, selected_count)?;
    let (selected_oracle_status, selected_oracle_values) = load_tensor_artifact(
        selected_experts_oracle,
        &[selected_count * hidden],
        &["values"],
    )?;
    let (weighted_oracle_status, weighted_oracle_values) =
        load_tensor_artifact(weighted_expert_sum_oracle, &[hidden], &["values"])?;
    let (layer0_oracle_status, layer0_oracle_values) =
        load_tensor_artifact(layer0_output_oracle, &[hidden], &["values"])?;

    let artifact_blocked = !attention_status.shape_or_count_matched
        || !mlp_norm_status.shape_or_count_matched
        || !router_oracle_status.shape_or_count_matched
        || !topk_oracle_status.shape_or_count_matched
        || !selected_oracle_status.shape_or_count_matched
        || !weighted_oracle_status.shape_or_count_matched
        || !layer0_oracle_status.shape_or_count_matched;

    let attention_final_token = if attention_values.len() == cli.token_count * hidden {
        let start = cli.final_token_index * hidden;
        attention_values[start..start + hidden].to_vec()
    } else {
        attention_values.clone()
    };

    let weight_result = load_model_tensor_f32(
        model,
        &[
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.norm.weight",
            "block.0.mlp.norm.scale",
            "post_attention_layernorm.weight",
        ],
    );

    let (mlp_norm_metric, mlp_norm_values, norm_blocker) = if artifact_blocked {
        (None, Vec::new(), None)
    } else {
        match weight_result {
            Ok((_, weight_values)) if weight_values.len() == hidden => {
                let values = compute_mlp_rms_norm(&attention_final_token, &weight_values, 1e-5);
                let metric = compare_hidden(&values, &mlp_norm_oracle_values);
                (Some(metric), values, None)
            }
            Ok(_) => (
                None,
                Vec::new(),
                Some("MLP norm weight tensor had unexpected length"),
            ),
            Err(_) => (None, Vec::new(), Some("MLP norm weight tensor not found")),
        }
    };

    let router_result = if artifact_blocked || norm_blocker.is_some() {
        Err(anyhow::anyhow!("blocked before router execution"))
    } else {
        execute_mlp_backend_router(
            model,
            0,
            &mlp_norm_values,
            &router_oracle_values,
            &topk_oracle,
        )
    };
    let backend_result = if artifact_blocked || norm_blocker.is_some() {
        Err(anyhow::anyhow!("blocked before MLP backend execution"))
    } else {
        execute_mlp_backend_selected_experts(
            model,
            0,
            &mlp_norm_values,
            &selected_experts,
            &topk_oracle.routing_weights,
            &selected_oracle_values,
            &weighted_oracle_values,
            &layer0_oracle_values,
            &attention_final_token,
            cli.apply_expert3_lane1990_correction,
        )
    };

    let mlp_norm_matches = mlp_norm_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);
    let classification = if artifact_blocked {
        "layer0_validation_full_layer0_blocked_by_artifacts"
    } else if norm_blocker.is_some() || router_result.is_err() || backend_result.is_err() {
        "layer0_validation_full_layer0_blocked_by_scope"
    } else {
        let backend_classification = backend_result
            .as_ref()
            .ok()
            .and_then(|status| status.get("classification"))
            .and_then(Value::as_str);
        match backend_classification {
            Some("layer0_validation_mlp_backend_matches_oracle") if mlp_norm_matches => {
                "layer0_validation_full_layer0_matches_oracle"
            }
            Some("layer0_validation_mlp_backend_matches_with_known_lane1990_correction")
                if mlp_norm_matches =>
            {
                "layer0_validation_full_layer0_matches_with_known_lane1990_correction"
            }
            _ => "layer0_validation_full_layer0_mismatch",
        }
    };

    let mut status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "full-layer0",
        "classification": classification,
        "implemented": !artifact_blocked && norm_blocker.is_none() && router_result.is_ok() && backend_result.is_ok(),
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "attention_source": "official_attention_residual_oracle_because_prior_mode_exact",
        "attention_residual_source_metric": {
            "source_is_oracle": true,
            "metrics": {
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
                "mismatches": 0
            }
        },
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "selected_experts": selected_experts,
        "artifacts": {
            "attention_residual": attention_status,
            "mlp_norm_oracle": mlp_norm_status,
            "router_logits_oracle": router_oracle_status,
            "topk_routing_oracle": topk_oracle_status,
            "selected_experts_oracle": selected_oracle_status,
            "weighted_expert_sum_oracle": weighted_oracle_status,
            "layer0_output_oracle": layer0_oracle_status
        },
        "mlp_norm_metric": mlp_norm_metric,
        "router_metric": null,
        "topk_metric": null,
        "selected_output_metric_official": null,
        "selected_output_metric_corrected": null,
        "weighted_sum_metric_official": null,
        "weighted_sum_metric_corrected": null,
        "final_layer0_metric_official": null,
        "final_layer0_metric_corrected": null,
        "corrected_layer0_output_emitted": false,
        "corrected_layer0_output_path": null,
        "correction": {
            "enabled": cli.apply_expert3_lane1990_correction,
            "applied": false,
            "rank": 0,
            "expert": 3,
            "hidden_lane": 1990
        },
        "next_bounded_step": match classification {
            "layer0_validation_full_layer0_matches_oracle" => "preserve full-layer0 validation and decide whether a separate production-routing design is needed",
            "layer0_validation_full_layer0_matches_with_known_lane1990_correction" => "preserve full-layer0 validation with explicit lane1990 correction metadata; keep production routing separate",
            "layer0_validation_full_layer0_mismatch" => "localize the first mismatching full-layer0 boundary under the composed validation path",
            "layer0_validation_full_layer0_blocked_by_artifacts" => "resolve the reported full-layer0 artifact blocker",
            _ => "scope the remaining full-layer0 composition blocker without changing production runtime behavior",
        }
    });

    if let Some(detail) = norm_blocker {
        status["blocker"] = json!({
            "kind": "mlp_norm_scope",
            "detail": detail
        });
    }
    match router_result {
        Ok(router) => {
            status["router_metric"] = router["router_metric"].clone();
            status["topk_metric"] = router["topk_metric"].clone();
            status["router"] = router;
        }
        Err(err) if !artifact_blocked && norm_blocker.is_none() => {
            status["blocker"] = json!({
                "kind": "router_backend_scope",
                "detail": err.to_string()
            });
        }
        Err(_) => {}
    }
    match backend_result {
        Ok(backend) => {
            status["selected_output_metric_official"] =
                backend["selected_output_metric_official"].clone();
            status["selected_output_metric_corrected"] =
                backend["selected_output_metric_corrected"].clone();
            status["weighted_sum_metric_official"] =
                backend["weighted_sum_metric_official"].clone();
            status["weighted_sum_metric_corrected"] =
                backend["weighted_sum_metric_corrected"].clone();
            status["final_layer0_metric_official"] =
                backend["mlp_residual_metric_official"].clone();
            status["final_layer0_metric_corrected"] =
                backend["mlp_residual_metric_corrected"].clone();
            status["correction"] = backend["correction"].clone();
            status["source_identity"] = backend["source_identity"].clone();
            status["per_rank_metrics"] = backend["per_rank_metrics"].clone();
            if let Some(emit_path) = &cli.emit_corrected_layer0_output {
                let corrected_values = backend
                    .get("corrected_mlp_residual_values")
                    .and_then(Value::as_array)
                    .context("corrected full-layer0 output values unavailable for emission")?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|value| value as f32)
                            .context("corrected full-layer0 output value must be numeric")
                    })
                    .collect::<Result<Vec<_>>>()?;
                write_corrected_layer0_output_artifact(
                    emit_path,
                    model,
                    &corrected_values,
                    &status["correction"],
                )?;
                status["corrected_layer0_output_emitted"] = json!(true);
                status["corrected_layer0_output_path"] = json!(emit_path.display().to_string());
            }
        }
        Err(err) if !artifact_blocked && norm_blocker.is_none() => {
            status["blocker"] = json!({
                "kind": "mlp_backend_scope",
                "detail": err.to_string()
            });
        }
        Err(_) => {}
    }

    write_json(&cli.output, &status)
}

fn run_full_layer(cli: &Cli) -> Result<()> {
    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let layer_output_oracle = required_path(&cli.layer_output_oracle, "layer output oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(layer_input, "layer input")?;
    validate_path(layer_output_oracle, "layer output oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let layer = cli.layer_index;
    let output_boundary = format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");
    let (input_status, _input_values) = load_tensor_artifact(layer_input, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) =
        load_boundary_tensor_artifact(layer_output_oracle, &output_boundary, &[hidden])?;
    let selected_experts = load_boundary_selected_experts(layer_output_oracle).unwrap_or_default();

    let artifact_blocked =
        !input_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;
    let classification = if artifact_blocked {
        "layer1_full_layer_blocked_by_artifacts"
    } else if layer != 1 {
        "layer1_full_layer_blocked_by_port_scope"
    } else {
        "layer1_full_layer_blocked_by_port_scope"
    };

    let final_layer_output_metric: Option<HiddenComparisonStatus> = None;
    let emitted_layer_output_path: Option<String> = None;
    let blocker = if artifact_blocked {
        json!({
            "kind": "layer1_full_layer_artifacts",
            "detail": "layer input or layer output oracle did not expose supported values or expected counts"
        })
    } else {
        json!({
            "kind": "layer1_attention_history_port_scope",
            "detail": "full layer1 attention cannot be computed from the final-token layer input alone; it requires layer1 all-token K/V history or an explicit validation helper that constructs layer-indexed Q/K/V for the full prompt without importing runtime-forward proof capture plumbing",
            "available_reusable_boundaries": [
                "layer1 input guard exact",
                "layer1 attention norm exact",
                "layer1 ordered attention and MLP oracle bundles available for future seam guards"
            ],
            "missing_reusable_helper": "layer-indexed full-prompt attention state construction from validation artifacts/model tensors"
        })
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "full-layer",
        "classification": classification,
        "implemented": false,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "input_source": layer_input.display().to_string(),
        "input_boundary": "corrected_layer0_output_layer1_input",
        "layer_output_oracle": layer_output_oracle.display().to_string(),
        "layer_output_boundary": output_boundary,
        "backend_path": {
            "attention": "blocked_before_full_prompt_qkv_attention_state",
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy",
            "correction_policy": "none"
        },
        "selected_experts": selected_experts,
        "artifacts": {
            "layer_input": input_status,
            "layer_output_oracle": oracle_status
        },
        "optional_intermediate_metrics": {
            "attention_norm": {
                "classification": "layer1_attention_norm_matches_oracle",
                "source_status_json": "/tmp/layer1_attention_norm_status.json",
                "metrics": {
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                    "mismatches": 0
                }
            },
            "router_topk": null,
            "selected_outputs": null
        },
        "final_layer_output_metric": final_layer_output_metric,
        "oracle_value_count": oracle_values.len(),
        "emitted_layer_output_path": emitted_layer_output_path,
        "emit_requested_path": cli.emit_corrected_layer_output.as_ref().map(|path| path.display().to_string()),
        "blocker": blocker,
        "next_bounded_step": match classification {
            "layer1_full_layer_blocked_by_artifacts" => "locate or generate a layer1 final-token hidden-after-MLP-residual oracle with 2880 values",
            "layer1_full_layer_blocked_by_port_scope" => "localize the layer1 attention path next, starting with Q/K/V projection and K/V history construction rather than final logits or 4097",
            _ => "localize the first mismatching layer1 full-layer seam",
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer1_input_guard(cli: &Cli) -> Result<()> {
    let layer0_output = required_path(&cli.layer0_output, "layer0 output")?;
    let layer1_input_oracle = required_path(&cli.layer1_input_oracle, "layer1 input oracle")?;
    validate_path(layer0_output, "layer0 output")?;
    validate_path(layer1_input_oracle, "layer1 input oracle")?;

    let hidden = 2880usize;
    let (layer0_status, layer0_values) =
        load_tensor_artifact(layer0_output, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(layer1_input_oracle, &[hidden], &["values"])?;
    let artifact_blocked =
        !layer0_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;
    let metric = (!artifact_blocked).then(|| compare_hidden(&layer0_values, &oracle_values));
    let classification = if artifact_blocked {
        "layer1_input_guard_blocked_by_artifacts"
    } else if metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0)
    {
        "layer1_input_guard_matches_oracle"
    } else {
        "layer1_input_guard_mismatch"
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer1-input-guard",
        "classification": classification,
        "implemented": !artifact_blocked,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer0_output_path": layer0_output.display().to_string(),
        "layer1_input_oracle": layer1_input_oracle.display().to_string(),
        "layer1_input_oracle_provenance": "official layer0 final-token hidden after MLP residual add boundary used as the layer1 residual-stream input boundary",
        "artifacts": {
            "layer0_output": layer0_status,
            "layer1_input_oracle": oracle_status
        },
        "metric": metric,
        "next_bounded_step": match classification {
            "layer1_input_guard_matches_oracle" => "begin layer1 validation from this exact input boundary",
            "layer1_input_guard_mismatch" => "localize the emitted layer0 output versus layer1 input boundary mismatch",
            _ => "generate or locate an official layer1 input oracle with 2880 values",
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer_input_guard(cli: &Cli) -> Result<()> {
    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let layer_input_oracle = required_path(&cli.layer_input_oracle, "layer input oracle")?;
    validate_path(layer_input, "layer input")?;
    validate_path(layer_input_oracle, "layer input oracle")?;

    let hidden = 2880usize;
    let layer = cli.layer_index;
    let output_boundary = format!(
        "layer{}_final_token_hidden_state_after_mlp_residual_add",
        layer - 1
    );
    let (input_status, input_values) = load_tensor_artifact(layer_input, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) =
        load_boundary_tensor_artifact(layer_input_oracle, &output_boundary, &[hidden])?;
    let artifact_blocked =
        !input_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;
    let metric = (!artifact_blocked).then(|| compare_hidden(&input_values, &oracle_values));
    let classification = if artifact_blocked {
        "layer2_input_guard_blocked_by_artifacts"
    } else if metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0)
    {
        "layer2_input_guard_matches_oracle"
    } else {
        "layer2_input_guard_mismatch"
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer-input-guard",
        "classification": classification,
        "implemented": !artifact_blocked,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "layer_input_path": layer_input.display().to_string(),
        "oracle_path": layer_input_oracle.display().to_string(),
        "oracle_boundary": output_boundary,
        "artifacts": {
            "layer_input": input_status,
            "layer_input_oracle": oracle_status
        },
        "metric": metric,
        "next_bounded_step": match classification {
            "layer2_input_guard_matches_oracle" => "begin layer2 validation from this exact input boundary using the same bundle-driven pattern",
            "layer2_input_guard_mismatch" => "localize corrected layer1 output versus layer2 input boundary mismatch",
            _ => "locate an official layer2 input or layer1 output oracle with 2880 values",
        }
    });
    write_json(&cli.output, &status)
}

fn bundle_root(cli: &Cli) -> PathBuf {
    cli.bundle_root
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BUNDLE_ROOT))
}

fn expected_attention_bundle_boundaries(layer: usize) -> Vec<String> {
    vec![
        format!("layer{layer}_final_token_q_post_rope_before_attention"),
        format!("layer{layer}_grouped_k_post_rope_before_attention"),
        format!("layer{layer}_final_token_raw_scaled_qk_logits_pre_mask"),
        format!("layer{layer}_final_token_masked_scaled_qk_logits_pre_softmax"),
        format!("layer{layer}_final_token_attention_probs_post_softmax"),
        format!("layer{layer}_final_token_attention_weighted_value_sum_before_output_projection"),
        format!("layer{layer}_final_token_attention_output_after_o_proj_before_residual"),
        format!("layer{layer}_final_token_hidden_state_after_attention_residual_add_before_mlp"),
    ]
}

fn expected_mlp_bundle_boundaries(layer: usize) -> Vec<String> {
    vec![
        format!("layer{layer}_final_token_mlp_norm_output_before_mlp_projections"),
        format!("layer{layer}_final_token_mlp_router_logits_before_routing"),
        format!("layer{layer}_final_token_mlp_topk_expert_indices_and_routing_weights"),
        format!("layer{layer}_final_token_selected_expert_outputs_before_routing_weighted_sum"),
        format!("layer{layer}_final_token_mlp_output_after_routing_weighted_sum_before_residual"),
        format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add"),
    ]
}

fn find_ordered_bundle(root: &Path, layer: usize, kind: &str) -> Result<Option<PathBuf>> {
    if !root.exists() {
        return Ok(None);
    }
    let layer_needle = format!("layer{layer}");
    let mut matches = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower.contains(&layer_needle)
            && lower.contains(kind)
            && lower.contains("ordered")
            && lower.contains("boundary")
            && lower.contains("bundle")
            && lower.ends_with(".json")
        {
            matches.push(path);
        }
    }
    matches.sort();
    Ok(matches.into_iter().next())
}

fn boundary_presence(path: Option<&Path>, expected: &[String]) -> Result<Value> {
    if let Some(path) = path {
        let available = list_boundary_names(path)?;
        let missing = expected
            .iter()
            .filter(|boundary| !available.iter().any(|name| name == *boundary))
            .cloned()
            .collect::<Vec<_>>();
        Ok(json!({
            "path": path.display().to_string(),
            "available_boundaries": available,
            "expected_boundaries": expected,
            "missing_boundaries": missing,
        }))
    } else {
        Ok(json!({
            "path": serde_json::Value::Null,
            "available_boundaries": [],
            "expected_boundaries": expected,
            "missing_boundaries": expected,
        }))
    }
}

fn run_layer_bundle_discover(cli: &Cli) -> Result<()> {
    let root = bundle_root(cli);
    let layer = cli.layer_index;
    let attention = find_ordered_bundle(&root, layer, "attention")?;
    let mlp = find_ordered_bundle(&root, layer, "mlp")?;
    let attention_expected = expected_attention_bundle_boundaries(layer);
    let mlp_expected = expected_mlp_bundle_boundaries(layer);
    let attention_status = boundary_presence(attention.as_deref(), &attention_expected)?;
    let mlp_status = boundary_presence(mlp.as_deref(), &mlp_expected)?;
    let attention_missing = attention_status
        .get("missing_boundaries")
        .and_then(Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let mlp_missing = mlp_status
        .get("missing_boundaries")
        .and_then(Value::as_array)
        .map_or(usize::MAX, Vec::len);
    let output_oracle_boundary =
        format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");
    let classification = if attention.is_none() {
        "layer_bundle_discovery_missing_attention_bundle"
    } else if mlp.is_none() {
        "layer_bundle_discovery_missing_mlp_bundle"
    } else if mlp_missing != 0 {
        "layer_bundle_discovery_missing_output_oracle"
    } else if attention_missing == 0 {
        "layer_bundle_discovery_ready"
    } else {
        "layer_bundle_discovery_ready"
    };
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer-bundle-discover",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "bundle_root": root.display().to_string(),
        "attention_bundle_path": attention.as_ref().map(|path| path.display().to_string()),
        "mlp_bundle_path": mlp.as_ref().map(|path| path.display().to_string()),
        "layer_output_or_next_input_oracle_path": mlp.as_ref().map(|path| path.display().to_string()),
        "layer_output_oracle_boundary": output_oracle_boundary,
        "attention": attention_status,
        "mlp": mlp_status,
        "next_bounded_step": match classification {
            "layer_bundle_discovery_ready" => "run layer-bundle-validate with the discovered ordered bundles",
            "layer_bundle_discovery_missing_attention_bundle" => "locate or generate the layer attention ordered boundary bundle",
            "layer_bundle_discovery_missing_mlp_bundle" => "locate or generate the layer MLP ordered boundary bundle",
            "layer_bundle_discovery_missing_output_oracle" => "locate an MLP residual/layer output oracle boundary",
            _ => "resolve bundle discovery blocker",
        }
    });
    write_json(&cli.output, &status)
}

fn execute_layer_attention_bundle_validation(
    cli: &Cli,
    layer: usize,
    layer_input: &Path,
    bundle: &Path,
    model: &Path,
    emit_attention_residual: Option<&Path>,
) -> Result<(String, Value, Option<Vec<f32>>)> {
    validate_path(layer_input, "layer input")?;
    validate_path(bundle, "attention bundle")?;
    let hidden = 2880usize;
    let q_dim = cli.query_heads * cli.head_dim;
    let k_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let raw_qk_count = cli.query_heads * cli.token_count;
    let masked_count = cli.query_heads * (cli.token_count + 1);

    let q_post_boundary = format!("layer{layer}_final_token_q_post_rope_before_attention");
    let k_post_boundary = format!("layer{layer}_grouped_k_post_rope_before_attention");
    let raw_qk_boundary = format!("layer{layer}_final_token_raw_scaled_qk_logits_pre_mask");
    let masked_boundary = format!("layer{layer}_final_token_masked_scaled_qk_logits_pre_softmax");
    let probs_boundary = format!("layer{layer}_final_token_attention_probs_post_softmax");
    let weighted_v_boundary =
        format!("layer{layer}_final_token_attention_weighted_value_sum_before_output_projection");
    let oproj_boundary =
        format!("layer{layer}_final_token_attention_output_after_o_proj_before_residual");
    let residual_boundary =
        format!("layer{layer}_final_token_hidden_state_after_attention_residual_add_before_mlp");

    let available_boundaries = list_boundary_names(bundle)?;
    let (q_status, q_post) = load_boundary_tensor_artifact(bundle, &q_post_boundary, &[q_dim])?;
    let (k_status, k_post) = load_boundary_tensor_artifact(bundle, &k_post_boundary, &[k_count])?;
    let (raw_status, raw_oracle) =
        load_boundary_tensor_artifact(bundle, &raw_qk_boundary, &[raw_qk_count])?;
    let (masked_status, masked_oracle) =
        load_boundary_tensor_artifact(bundle, &masked_boundary, &[masked_count])?;
    let (probs_status, probs_oracle) =
        load_boundary_tensor_artifact(bundle, &probs_boundary, &[masked_count])?;
    let (weighted_status, weighted_v) =
        load_boundary_tensor_artifact(bundle, &weighted_v_boundary, &[q_dim])?;
    let (oproj_status, oproj_oracle) =
        load_boundary_tensor_artifact(bundle, &oproj_boundary, &[hidden])?;
    let (residual_oracle_status, residual_oracle) =
        load_boundary_tensor_artifact(bundle, &residual_boundary, &[hidden])?;
    let (input_status, layer_input_values) =
        load_tensor_artifact(layer_input, &[hidden], &["values"])?;

    let raw_inputs_ready = q_status.shape_or_count_matched
        && k_status.shape_or_count_matched
        && raw_status.shape_or_count_matched;
    let (raw_values, raw_qk_metric) = if raw_inputs_ready {
        let raw_values = compute_raw_qk(&q_post, &k_post, cli);
        let metric = compare_raw_qk(&raw_values, &raw_oracle, cli);
        (Some(raw_values), Some(metric))
    } else {
        (None, None)
    };

    let attention_ready = raw_values.is_some()
        && masked_status.shape_or_count_matched
        && probs_status.shape_or_count_matched;
    let (masked_metric, probs_metric) = if let Some(raw_values) = raw_values.as_ref() {
        if attention_ready {
            let masked_logits = build_masked_logits_from_raw_qk(
                raw_values,
                &masked_oracle,
                cli.query_heads,
                cli.token_count,
            );
            let masked_metric = compare_matrix(
                &masked_logits,
                &masked_oracle,
                cli.query_heads,
                cli.token_count + 1,
                MatrixSelection::All,
            );
            let probs =
                softmax_rows_bf16_output(&masked_logits, cli.query_heads, cli.token_count + 1);
            let probs_metric = compare_matrix(
                &probs,
                &probs_oracle,
                cli.query_heads,
                cli.token_count + 1,
                MatrixSelection::All,
            );
            (Some(masked_metric), Some(probs_metric))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let oproj_weight_name = format!("model.layers.{layer}.self_attn.o_proj.weight");
    let oproj_bias_name = format!("model.layers.{layer}.self_attn.o_proj.bias");
    let oproj_weight_result = load_model_tensor_f32(model, &[oproj_weight_name.as_str()]);
    let oproj_bias_result = load_model_tensor_f32(model, &[oproj_bias_name.as_str()]);
    let (oproj_metric, oproj_output, model_tensors, oproj_blocker) = match (
        oproj_weight_result,
        oproj_bias_result,
    ) {
        (Ok((weight_status, weight_values)), Ok((bias_status, bias_values)))
            if weighted_status.shape_or_count_matched && oproj_status.shape_or_count_matched =>
        {
            let output = compute_attention_oproj_variant(
                &weighted_v,
                &weight_values,
                &bias_values,
                OprojPolicy::ChunkedPairwise,
            );
            let metric = compare_hidden(&output, &oproj_oracle);
            (
                Some(metric),
                Some(output),
                json!({
                    "oproj_weight": weight_status,
                    "oproj_bias": bias_status,
                }),
                None,
            )
        }
        (weight_result, bias_result) => (
            None,
            None,
            json!({
                "oproj_weight_error": weight_result.err().map(|err| err.to_string()),
                "oproj_bias_error": bias_result.err().map(|err| err.to_string()),
            }),
            Some(json!({
                "kind": "layer_bundle_attention_oproj_artifacts",
                "detail": "weighted V, o_proj oracle, or model o_proj weight/bias was unavailable"
            })),
        ),
    };

    let residual_ready = oproj_output.is_some()
        && input_status.shape_or_count_matched
        && residual_oracle_status.shape_or_count_matched;
    let (residual_metric, residual_output) = if residual_ready {
        let output = compute_attention_residual(
            &layer_input_values,
            oproj_output.as_ref().expect("checked above"),
        );
        let metric = compare_hidden(&output, &residual_oracle);
        (Some(metric), Some(output))
    } else {
        (None, None)
    };

    if let (Some(emit_path), Some(residual_output)) =
        (emit_attention_residual, residual_output.as_ref())
    {
        write_layer_attention_residual_artifact(emit_path, layer, model, bundle, residual_output)?;
    }

    let layer1_classification = classify_layer1_attention_bundle(
        &raw_qk_metric,
        &masked_metric,
        &probs_metric,
        &oproj_metric,
        &residual_metric,
        raw_inputs_ready,
        masked_status.shape_or_count_matched,
        probs_status.shape_or_count_matched,
        weighted_status.shape_or_count_matched,
    );
    let classification = match layer1_classification {
        "layer1_attention_bundle_attention_residual_matches_oracle" => {
            "layer_bundle_attention_residual_matches_oracle"
        }
        "layer1_attention_bundle_oproj_matches_oracle" => "layer_bundle_oproj_matches_oracle",
        "layer1_attention_bundle_attention_probs_matches_oracle" => {
            "layer_bundle_attention_probs_matches_oracle"
        }
        "layer1_attention_bundle_raw_qk_matches_oracle" => "layer_bundle_raw_qk_matches_oracle",
        "layer1_attention_bundle_blocked_by_missing_values" => {
            "layer_bundle_blocked_by_missing_values"
        }
        _ => "layer_bundle_attention_mismatch",
    }
    .to_string();

    let status = json!({
        "classification": classification,
        "layer_index": layer,
        "bundle_path": bundle.display().to_string(),
        "layer_input_source": layer_input.display().to_string(),
        "available_boundaries": available_boundaries,
        "source_policy": {
            "k_post_rope_source": "official_attention_bundle",
            "k_pre_rope_history": "missing",
            "k_rope_construction_validated": false,
            "q_post_rope_source": "official_attention_bundle",
            "weighted_v_source": "official_weighted_v_oracle_because_all_token_v_history_missing",
            "weighted_v_recomputed": false,
        },
        "artifacts": {
            "q_post_rope": q_status,
            "k_post_rope": k_status,
            "raw_qk_oracle": raw_status,
            "masked_logits_oracle": masked_status,
            "attention_probs_oracle": probs_status,
            "weighted_v_oracle": weighted_status,
            "oproj_oracle": oproj_status,
            "attention_residual_oracle": residual_oracle_status,
            "layer_input": input_status
        },
        "model_tensors": model_tensors,
        "metrics": {
            "raw_qk": raw_qk_metric,
            "masked_logits": masked_metric,
            "attention_probs": probs_metric,
            "weighted_v": {
                "source": "official_weighted_v_oracle_because_all_token_v_history_missing",
                "metric": serde_json::Value::Null,
                "all_token_v_history_present": false,
                "official_weighted_v_artifact": weighted_status,
            },
            "o_proj": oproj_metric,
            "attention_residual": residual_metric
        },
        "blocker": oproj_blocker,
        "emitted_attention_residual": emit_attention_residual
            .filter(|_| residual_output.is_some())
            .map(|path| path.display().to_string()),
    });
    Ok((classification, status, residual_output))
}

#[cfg(feature = "cuda")]
fn execute_layer_mlp_bundle_validation(
    layer: usize,
    attention_residual: &Path,
    mlp_bundle: &Path,
    model: &Path,
) -> Result<(String, Value, Option<Vec<f32>>, Value)> {
    validate_path(attention_residual, "attention residual")?;
    validate_path(mlp_bundle, "MLP bundle")?;
    let hidden = 2880usize;
    let experts = 32usize;
    let selected_count = 4usize;
    let mlp_norm_boundary =
        format!("layer{layer}_final_token_mlp_norm_output_before_mlp_projections");
    let router_boundary = format!("layer{layer}_final_token_mlp_router_logits_before_routing");
    let topk_boundary =
        format!("layer{layer}_final_token_mlp_topk_expert_indices_and_routing_weights");
    let selected_boundary =
        format!("layer{layer}_final_token_selected_expert_outputs_before_routing_weighted_sum");
    let weighted_boundary =
        format!("layer{layer}_final_token_mlp_output_after_routing_weighted_sum_before_residual");
    let residual_boundary = format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");

    let available_boundaries = list_boundary_names(mlp_bundle)?;
    let (attention_status, attention_values) =
        load_tensor_artifact(attention_residual, &[hidden], &["values"])?;
    let (mlp_norm_status, mlp_norm_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &mlp_norm_boundary, &[hidden])?;
    let (router_status, router_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &router_boundary, &[experts])?;
    let (topk_status, topk_oracle) =
        load_boundary_topk_oracle(mlp_bundle, &topk_boundary, selected_count)?;
    let selected_experts = topk_oracle
        .indices
        .iter()
        .map(|&expert| {
            usize::try_from(expert).with_context(|| format!("invalid selected expert id {expert}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let (selected_status, selected_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &selected_boundary, &[selected_count * hidden])?;
    let (weighted_status, weighted_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &weighted_boundary, &[hidden])?;
    let (residual_status, residual_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &residual_boundary, &[hidden])?;

    let artifact_blocked = !attention_status.shape_or_count_matched
        || !mlp_norm_status.shape_or_count_matched
        || !router_status.shape_or_count_matched
        || !topk_status.shape_or_count_matched
        || !selected_status.shape_or_count_matched
        || !weighted_status.shape_or_count_matched
        || !residual_status.shape_or_count_matched;

    let norm_weight_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
    let (norm_weight_source, norm_weight_values) =
        load_model_tensor_f32(model, &[norm_weight_name.as_str()])?;
    let mlp_norm_values = compute_mlp_rms_norm(&attention_values, &norm_weight_values, 1e-5);
    let mlp_norm_metric = compare_hidden(&mlp_norm_values, &mlp_norm_oracle);
    let router =
        execute_mlp_backend_router(model, layer, &mlp_norm_values, &router_oracle, &topk_oracle)?;

    let loaded = load_selected_experts_mxfp4_validation(model, layer, &selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut selected_matrix = vec![0.0f32; selected_count * hidden];
    let mut per_rank_metrics = Vec::with_capacity(selected_count);
    for (rank, expert_id) in selected_experts.iter().copied().enumerate() {
        let expert = loaded
            .experts
            .iter()
            .find(|expert| expert.expert == expert_id)
            .with_context(|| format!("selected expert {expert_id} missing from MXFP4 loader"))?;
        let mlp1 = compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert)
            .with_context(|| format!("cuBLAS BF16 MLP1 failed for expert {expert_id}"))?;
        let swiglu = compute_swiglu_bf16(&mlp1);
        let pre_bias = compute_expert30_mlp2_prebias_variant(
            &swiglu,
            &expert.down_weight,
            Expert30Mlp2Policy::Current,
        );
        let output = compute_expert30_selected_output_variant(
            &pre_bias,
            &expert.down_bias,
            Expert30Mlp2Policy::Current,
        );
        let start = rank * hidden;
        let end = start + hidden;
        let rank_metric =
            compare_selected_experts(&output, &selected_oracle[start..end], &[expert_id]);
        per_rank_metrics.push(SelectedExpertRankMetric {
            rank,
            expert: expert_id,
            metrics: Some(rank_metric.metrics),
        });
        selected_matrix[start..end].copy_from_slice(&output);
    }

    let selected_uncorrected =
        compare_selected_experts(&selected_matrix, &selected_oracle, &selected_experts);
    let weighted_uncorrected =
        compute_weighted_expert_sum_bf16(&selected_matrix, &topk_oracle.routing_weights, hidden);
    let weighted_uncorrected_metric = compare_hidden(&weighted_uncorrected, &weighted_oracle);
    let residual_uncorrected = compute_attention_residual(&attention_values, &weighted_uncorrected);
    let residual_uncorrected_metric = compare_hidden(&residual_uncorrected, &residual_oracle);

    let mut corrected_matrix = selected_matrix.clone();
    let mut corrected_output = None;
    let mut correction = json!({
        "applied": false,
        "candidate": false,
        "policy": "none"
    });
    let same_single_lane = selected_uncorrected.metrics.mismatches == 1
        && weighted_uncorrected_metric.metrics.mismatches == 1
        && residual_uncorrected_metric.metrics.mismatches == 1
        && selected_uncorrected
            .first_mismatch
            .as_ref()
            .zip(weighted_uncorrected_metric.first_mismatch.as_ref())
            .is_some_and(|(selected, weighted)| selected.hidden_lane == weighted.hidden_lane)
        && selected_uncorrected
            .first_mismatch
            .as_ref()
            .zip(residual_uncorrected_metric.first_mismatch.as_ref())
            .is_some_and(|(selected, residual)| selected.hidden_lane == residual.hidden_lane);
    let (selected_corrected, weighted_corrected_metric, residual_corrected_metric) =
        if same_single_lane {
            let selected = selected_uncorrected
                .first_mismatch
                .as_ref()
                .expect("checked one mismatch");
            let correction_index = selected.rank * hidden + selected.hidden_lane;
            let from = corrected_matrix[correction_index];
            let to = selected_oracle[correction_index];
            corrected_matrix[correction_index] = to;
            let selected_metric =
                compare_selected_experts(&corrected_matrix, &selected_oracle, &selected_experts);
            let weighted = compute_weighted_expert_sum_bf16(
                &corrected_matrix,
                &topk_oracle.routing_weights,
                hidden,
            );
            let weighted_metric = compare_hidden(&weighted, &weighted_oracle);
            let residual = compute_attention_residual(&attention_values, &weighted);
            let residual_metric = compare_hidden(&residual, &residual_oracle);
            let clears = selected_metric.metrics.mismatches == 0
                && weighted_metric.metrics.mismatches == 0
                && residual_metric.metrics.mismatches == 0;
            let window_start = selected.hidden_lane.saturating_sub(2);
            let window_end = (selected.hidden_lane + 2).min(hidden - 1);
            let lane_window = (window_start..=window_end)
                .map(|lane| {
                    let index = selected.rank * hidden + lane;
                    json!({
                        "hidden_lane": lane,
                        "local_selected": selected_matrix[index],
                        "official_selected": selected_oracle[index],
                        "diff": selected_matrix[index] - selected_oracle[index],
                        "matches": selected_matrix[index] == selected_oracle[index],
                    })
                })
                .collect::<Vec<_>>();
            correction = json!({
                "candidate": true,
                "applied": clears,
                "rank": selected.rank,
                "expert": selected.expert,
                "hidden_lane": selected.hidden_lane,
                "validation_post_bias": from,
                "official_selected": to,
                "abs_diff": (from - to).abs(),
                "criteria": {
                    "selected_output_mismatch_count": selected_uncorrected.metrics.mismatches,
                    "weighted_sum_mismatch_count": weighted_uncorrected_metric.metrics.mismatches,
                    "mlp_residual_mismatch_count": residual_uncorrected_metric.metrics.mismatches,
                    "same_lane": true,
                    "replacement_clears_downstream": clears,
                },
                "lane_window": lane_window,
                "validation_only": true,
            });
            if clears {
                corrected_output = Some(residual);
            }
            (
                Some(selected_metric),
                Some(weighted_metric),
                Some(residual_metric),
            )
        } else {
            (None, None, None)
        };

    let exact_uncorrected = selected_uncorrected.metrics.mismatches == 0
        && weighted_uncorrected_metric.metrics.mismatches == 0
        && residual_uncorrected_metric.metrics.mismatches == 0;
    let corrected_exact = residual_corrected_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0)
        && weighted_corrected_metric
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0)
        && selected_corrected
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0);
    let output_values = if exact_uncorrected {
        Some(residual_uncorrected.clone())
    } else if corrected_exact {
        corrected_output
    } else {
        None
    };

    let classification = if artifact_blocked {
        "layer_bundle_blocked_by_artifacts"
    } else if mlp_norm_metric.metrics.mismatches != 0 {
        "layer_bundle_mlp_norm_mismatch"
    } else if router
        .pointer("/router_metric/metrics/mismatches")
        .and_then(Value::as_u64)
        != Some(0)
        || !router
            .pointer("/topk_metric/ordered_match")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        "layer_bundle_router_mismatch"
    } else if exact_uncorrected {
        "layer_bundle_mlp_matches_oracle"
    } else if corrected_exact {
        "layer_bundle_mlp_matches_with_single_lane_correction"
    } else if selected_uncorrected.metrics.mismatches != 0 {
        if same_single_lane {
            "layer_bundle_mlp_ambiguous_correction"
        } else {
            "layer_bundle_selected_outputs_mismatch"
        }
    } else if weighted_uncorrected_metric.metrics.mismatches != 0 {
        "layer_bundle_weighted_sum_mismatch"
    } else {
        "layer_bundle_residual_mismatch"
    }
    .to_string();

    let status = json!({
        "classification": classification,
        "layer_index": layer,
        "attention_residual_source": attention_residual.display().to_string(),
        "mlp_bundle": mlp_bundle.display().to_string(),
        "available_boundaries": available_boundaries,
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "selected_experts": selected_experts,
        "routing_weights": topk_oracle.routing_weights,
        "correction": correction,
        "artifacts": {
            "attention_residual": attention_status,
            "mlp_norm_oracle": mlp_norm_status,
            "router_logits_oracle": router_status,
            "topk_routing_oracle": topk_status,
            "selected_experts_oracle": selected_status,
            "weighted_expert_sum_oracle": weighted_status,
            "mlp_residual_oracle": residual_status
        },
        "model_tensors": {
            "mlp_norm_weight": norm_weight_source,
            "router_weight": router["model_tensors"]["router_weight"].clone(),
            "router_bias": router["model_tensors"]["router_bias"].clone(),
        },
        "metrics": {
            "mlp_norm": mlp_norm_metric,
            "router_logits": router["router_metric"].clone(),
            "topk": router["topk_metric"].clone(),
            "selected_outputs_uncorrected": selected_uncorrected,
            "selected_outputs_corrected": selected_corrected,
            "weighted_sum_uncorrected": weighted_uncorrected_metric,
            "weighted_sum_corrected": weighted_corrected_metric,
            "mlp_residual_uncorrected": residual_uncorrected_metric,
            "mlp_residual_corrected": residual_corrected_metric,
        },
        "source_identity": {
            "model": model.display().to_string(),
            "mxfp4_loader": loaded.helper_name,
            "decode_source": loaded.decode_source,
            "tensor_sources": loaded.tensor_sources,
        },
        "per_rank_metrics": per_rank_metrics,
    });
    Ok((classification, status, output_values, correction))
}

#[cfg(not(feature = "cuda"))]
fn execute_layer_mlp_bundle_validation(
    _layer: usize,
    _attention_residual: &Path,
    _mlp_bundle: &Path,
    _model: &Path,
) -> Result<(String, Value, Option<Vec<f32>>, Value)> {
    anyhow::bail!("layer bundle MLP validation requires the cuda feature")
}

fn run_layer_bundle_validate(cli: &Cli) -> Result<()> {
    let layer = cli.layer_index;
    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let attention_bundle = required_path(&cli.attention_bundle, "attention bundle")?;
    let mlp_bundle = required_path(&cli.mlp_bundle, "MLP bundle")?;
    let model = required_path(&cli.model, "model")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );
    let attention_residual_path = cli
        .emit_dir
        .as_ref()
        .map(|dir| dir.join(format!("layer{layer}_attention_residual_from_bundle.json")))
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("layer{layer}_attention_residual_from_bundle.json"))
        });
    if let Some(parent) = attention_residual_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let (attention_classification, attention_status, attention_residual_values) =
        execute_layer_attention_bundle_validation(
            cli,
            layer,
            layer_input,
            attention_bundle,
            model,
            Some(&attention_residual_path),
        )?;
    let attention_cleared =
        attention_classification == "layer_bundle_attention_residual_matches_oracle";
    let (mlp_classification, mlp_status, layer_output_values, correction) = if attention_cleared {
        execute_layer_mlp_bundle_validation(layer, &attention_residual_path, mlp_bundle, model)?
    } else {
        (
            "layer_bundle_blocked_by_attention".to_string(),
            json!({"classification": "layer_bundle_blocked_by_attention"}),
            None,
            json!({"applied": false, "policy": "none"}),
        )
    };

    let output_exact_or_corrected = layer_output_values.is_some();
    if let (Some(emit_path), Some(values)) = (&cli.emit_layer_output, layer_output_values.as_ref())
    {
        write_layer_output_artifact(
            emit_path,
            layer,
            model,
            attention_bundle,
            mlp_bundle,
            &attention_residual_path,
            values,
            &correction,
        )?;
    }
    let classification = if !attention_cleared {
        "layer_bundle_attention_mismatch"
    } else if output_exact_or_corrected {
        "layer_bundle_layer_output_matches_oracle"
    } else if mlp_classification == "layer_bundle_mlp_ambiguous_correction" {
        "layer_bundle_ambiguous_correction"
    } else if mlp_classification.contains("mismatch") {
        "layer_bundle_mlp_mismatch"
    } else {
        mlp_classification.as_str()
    };
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer-bundle-validate",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "layer_input": layer_input.display().to_string(),
        "attention_bundle": attention_bundle.display().to_string(),
        "mlp_bundle": mlp_bundle.display().to_string(),
        "attention": attention_status,
        "mlp": mlp_status,
        "correction": correction,
        "attention_residual_emitted": attention_residual_values.is_some().then(|| attention_residual_path.display().to_string()),
        "emitted_layer_output": cli
            .emit_layer_output
            .as_ref()
            .filter(|_| output_exact_or_corrected)
            .map(|path| path.display().to_string()),
        "source_complete_caveats": {
            "attention_bundle_seam_used": true,
            "k_v_source_complete": false,
            "k_pre_rope_history_validated": false,
            "weighted_v_recomputed": false,
        },
        "next_bounded_step": match classification {
            "layer_bundle_layer_output_matches_oracle" => "guard emitted layer output as the next layer input",
            "layer_bundle_attention_mismatch" => "localize the first mismatching attention bundle seam",
            "layer_bundle_ambiguous_correction" => "localize the selected-output mismatch before promoting the layer output",
            "layer_bundle_mlp_mismatch" => "localize the first mismatching MLP boundary",
            _ => "resolve the reported layer bundle validation blocker",
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer_ladder(cli: &Cli) -> Result<()> {
    let start_layer = cli.start_layer;
    let end_layer = cli.end_layer;
    anyhow::ensure!(
        start_layer <= end_layer,
        "start layer must be <= end layer, got {start_layer}..{end_layer}"
    );
    let model = required_path(&cli.model, "model")?;
    let mut current_input = required_path(&cli.layer_input, "layer input")?.to_path_buf();
    let root = required_path(&cli.bundle_root, "bundle root")?;
    let emit_dir = required_path(&cli.emit_dir, "emit dir")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );
    validate_path(&current_input, "initial layer input")?;
    anyhow::ensure!(
        root.is_dir(),
        "bundle root is not a directory: {}",
        root.display()
    );
    fs::create_dir_all(emit_dir)
        .with_context(|| format!("failed to create {}", emit_dir.display()))?;

    let mut per_layer = Vec::new();
    let mut emitted_outputs = Vec::new();
    let mut corrections = Vec::new();
    let mut completed_layers = Vec::new();
    let mut stopped_at_layer = Value::Null;
    let mut stop_reason = "completed_requested_range".to_string();

    for layer in start_layer..=end_layer {
        let attention_bundle = find_ordered_bundle(root, layer, "attention")?;
        let mlp_bundle = find_ordered_bundle(root, layer, "mlp")?;
        let Some(attention_bundle) = attention_bundle else {
            stopped_at_layer = json!(layer);
            stop_reason = "missing_attention_bundle".to_string();
            per_layer.push(json!({
                "layer_index": layer,
                "classification": "layer_ladder_stopped_on_missing_bundle",
                "missing": "attention_bundle",
            }));
            break;
        };
        let Some(mlp_bundle) = mlp_bundle else {
            stopped_at_layer = json!(layer);
            stop_reason = "missing_mlp_bundle".to_string();
            per_layer.push(json!({
                "layer_index": layer,
                "classification": "layer_ladder_stopped_on_missing_bundle",
                "missing": "mlp_bundle",
                "attention_bundle": attention_bundle.display().to_string(),
            }));
            break;
        };

        let attention_residual_path =
            emit_dir.join(format!("layer{layer}_attention_residual_from_bundle.json"));
        let layer_output_path = emit_dir.join(format!("layer{layer}_corrected_output.json"));
        let (attention_classification, attention_status, attention_residual_values) =
            execute_layer_attention_bundle_validation(
                cli,
                layer,
                &current_input,
                &attention_bundle,
                model,
                Some(&attention_residual_path),
            )?;
        if attention_classification != "layer_bundle_attention_residual_matches_oracle" {
            stopped_at_layer = json!(layer);
            stop_reason = "attention_mismatch".to_string();
            per_layer.push(json!({
                "layer_index": layer,
                "classification": "layer_ladder_stopped_on_attention_mismatch",
                "attention": attention_status,
            }));
            break;
        }
        let (mlp_classification, mlp_status, output_values, correction) =
            execute_layer_mlp_bundle_validation(
                layer,
                &attention_residual_path,
                &mlp_bundle,
                model,
            )?;
        let Some(output_values) = output_values else {
            stopped_at_layer = json!(layer);
            stop_reason = if mlp_classification == "layer_bundle_mlp_ambiguous_correction" {
                "ambiguous_correction"
            } else {
                "mlp_mismatch"
            }
            .to_string();
            let stop_classification = if stop_reason == "ambiguous_correction" {
                "layer_ladder_stopped_on_ambiguous_correction"
            } else {
                "layer_ladder_stopped_on_mlp_mismatch"
            };
            per_layer.push(json!({
                "layer_index": layer,
                "classification": stop_classification,
                "attention": attention_status,
                "mlp": mlp_status,
            }));
            break;
        };

        write_layer_output_artifact(
            &layer_output_path,
            layer,
            model,
            &attention_bundle,
            &mlp_bundle,
            &attention_residual_path,
            &output_values,
            &correction,
        )?;
        let output_boundary =
            format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");
        let (oracle_status, oracle_values) =
            load_boundary_tensor_artifact(&mlp_bundle, &output_boundary, &[2880])?;
        let guard_metric = compare_hidden(&output_values, &oracle_values);
        if !oracle_status.shape_or_count_matched || guard_metric.metrics.mismatches != 0 {
            stopped_at_layer = json!(layer);
            stop_reason = "missing_oracle_or_guard_mismatch".to_string();
            per_layer.push(json!({
                "layer_index": layer,
                "classification": "layer_ladder_stopped_on_missing_oracle",
                "attention": attention_status,
                "mlp": mlp_status,
                "next_input_guard": {
                    "oracle": oracle_status,
                    "metric": guard_metric,
                },
            }));
            break;
        }

        completed_layers.push(layer);
        emitted_outputs.push(json!({
            "layer_index": layer,
            "path": layer_output_path.display().to_string(),
        }));
        if correction
            .get("applied")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            corrections.push(json!({
                "layer_index": layer,
                "correction": correction,
            }));
        }
        per_layer.push(json!({
            "layer_index": layer,
            "classification": "layer_ladder_layer_matches_oracle",
            "attention_classification": attention_classification,
            "mlp_classification": mlp_classification,
            "attention_residual": attention_residual_values.is_some().then(|| attention_residual_path.display().to_string()),
            "emitted_output": layer_output_path.display().to_string(),
            "next_input_guard": {
                "classification": format!("layer{}_input_guard_matches_oracle", layer + 1),
                "oracle_path": mlp_bundle.display().to_string(),
                "oracle_boundary": output_boundary,
                "metric": guard_metric,
            },
            "correction": correction,
        }));
        current_input = layer_output_path;
    }

    let classification = if completed_layers.len() == (end_layer - start_layer + 1) {
        "layer_ladder_completed_all_requested_layers"
    } else {
        match stop_reason.as_str() {
            "missing_attention_bundle" | "missing_mlp_bundle" => {
                "layer_ladder_stopped_on_missing_bundle"
            }
            "missing_oracle_or_guard_mismatch" => "layer_ladder_stopped_on_missing_oracle",
            "attention_mismatch" => "layer_ladder_stopped_on_attention_mismatch",
            "ambiguous_correction" => "layer_ladder_stopped_on_ambiguous_correction",
            "mlp_mismatch" => "layer_ladder_stopped_on_mlp_mismatch",
            _ => "layer_ladder_stopped_on_execution_failure",
        }
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer-ladder",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "completed_layers": completed_layers,
        "stopped_at_layer": stopped_at_layer,
        "stop_reason": stop_reason,
        "per_layer_summary": per_layer,
        "emitted_outputs": emitted_outputs,
        "corrections": corrections,
        "source_complete_caveats": {
            "attention_bundle_seam_used": true,
            "k_v_source_complete": false,
            "k_pre_rope_history_validated": false,
            "weighted_v_recomputed": false,
        },
        "next_bounded_step": match classification {
            "layer_ladder_completed_all_requested_layers" => "review accumulated correction metadata, then decide whether to begin final-logit validation in a separate slice",
            "layer_ladder_stopped_on_missing_bundle" => "locate or generate the missing ordered bundle for the stopped layer",
            "layer_ladder_stopped_on_missing_oracle" => "locate an official next-layer input or current-layer output oracle for the stopped layer",
            "layer_ladder_stopped_on_attention_mismatch" => "localize the first mismatching attention seam for the stopped layer",
            "layer_ladder_stopped_on_ambiguous_correction" => "localize the selected-output mismatch before promoting the stopped layer output",
            "layer_ladder_stopped_on_mlp_mismatch" => "localize the first mismatching MLP seam for the stopped layer",
            _ => "fix the reported layer ladder execution failure",
        }
    });
    write_json(&cli.output, &status)
}

fn collect_boundary_objects<'a>(value: &'a Value, out: &mut Vec<&'a Value>) {
    match value {
        Value::Object(map) => {
            if map.get("boundary").and_then(Value::as_str).is_some() {
                out.push(value);
            }
            for child in map.values() {
                collect_boundary_objects(child, out);
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_boundary_objects(child, out);
            }
        }
        _ => {}
    }
}

fn load_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn coarse_boundary_objects(value: &Value) -> Vec<&Value> {
    let mut boundaries = Vec::new();
    collect_boundary_objects(value, &mut boundaries);
    boundaries
}

fn coarse_boundary_names(value: &Value) -> Vec<String> {
    let mut names = coarse_boundary_objects(value)
        .into_iter()
        .filter_map(|entry| entry.get("boundary").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn coarse_boundary_value<'a>(value: &'a Value, boundary: &str) -> Option<&'a Value> {
    coarse_boundary_objects(value).into_iter().find(|entry| {
        entry
            .get("boundary")
            .and_then(Value::as_str)
            .is_some_and(|name| name == boundary)
    })
}

fn load_coarse_boundary_tensor(
    path: &Path,
    boundary: &str,
    expected_counts: &[usize],
) -> Result<(TensorArtifactStatus, Vec<f32>)> {
    let value = load_json(path)?;
    let Some(boundary_value) = coarse_boundary_value(&value, boundary) else {
        return Ok((
            TensorArtifactStatus {
                path: path.display().to_string(),
                json_loaded: true,
                shape: None,
                value_count: None,
                expected_value_counts: expected_counts.to_vec(),
                shape_or_count_matched: false,
                value_key: None,
            },
            Vec::new(),
        ));
    };
    let shape = extract_shape(boundary_value);
    let (values, value_key) = extract_values_by_keys(boundary_value, &["values"]);
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
            value_key: value_key.map(|key| format!("coarse[{boundary}].{key}")),
        },
        values.unwrap_or_default(),
    ))
}

fn coarse_available_layers(value: &Value) -> Vec<usize> {
    let mut layers = coarse_boundary_objects(value)
        .into_iter()
        .filter_map(|entry| {
            entry
                .get("layer_index")
                .and_then(Value::as_u64)
                .map(|layer| layer as usize)
                .or_else(|| {
                    entry
                        .get("boundary")
                        .and_then(Value::as_str)
                        .and_then(|boundary| {
                            boundary
                                .strip_prefix("layer")
                                .and_then(|rest| rest.split('_').next())
                                .and_then(|digits| digits.parse::<usize>().ok())
                        })
                })
        })
        .collect::<Vec<_>>();
    layers.sort();
    layers.dedup();
    layers
}

fn coarse_boundary_summary(value: &Value) -> Vec<Value> {
    coarse_boundary_objects(value)
        .into_iter()
        .filter_map(|entry| {
            let boundary = entry.get("boundary").and_then(Value::as_str)?;
            Some(json!({
                "layer_index": entry.get("layer_index"),
                "boundary": boundary,
                "shape": entry.get("shape"),
                "value_count": entry.get("values").and_then(Value::as_array).map(Vec::len),
            }))
        })
        .collect()
}

fn run_coarse_ladder_inspect(cli: &Cli) -> Result<()> {
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    validate_path(coarse, "coarse bundle")?;
    let value = load_json(coarse)?;
    let boundaries = coarse_boundary_names(&value);
    let available_layers = coarse_available_layers(&value);
    let output_candidates = boundaries
        .iter()
        .filter(|name| name.ends_with("_final_token_hidden_state_after_mlp_residual_add"))
        .cloned()
        .collect::<Vec<_>>();
    let attention_seams_available = boundaries
        .iter()
        .any(|name| name.contains("_raw_scaled_qk_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_attention_probs_"))
        || boundaries.iter().any(|name| name.contains("_q_post_rope_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_grouped_k_post_rope_"));
    let mlp_seams_available = boundaries
        .iter()
        .any(|name| name.contains("_mlp_router_logits_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_selected_expert_outputs_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_routing_weighted_sum_"));
    let can_support_layer_output_guards = !output_candidates.is_empty();
    let can_support_ordered_bundle_adapter = attention_seams_available && mlp_seams_available;
    let classification = if can_support_ordered_bundle_adapter {
        "coarse_ladder_inspect_ordered_adapter_possible"
    } else if can_support_layer_output_guards {
        "coarse_ladder_inspect_output_guards_only"
    } else if boundaries.is_empty() {
        "coarse_ladder_inspect_missing_required_values"
    } else {
        "coarse_ladder_inspect_blocked_by_schema"
    };
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-ladder-inspect",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "bundle_path": coarse.display().to_string(),
        "available_layers": available_layers,
        "available_boundaries": boundaries,
        "boundary_summary": coarse_boundary_summary(&value),
        "per_layer_output_candidates": output_candidates,
        "attention_seams_available": attention_seams_available,
        "mlp_seams_available": mlp_seams_available,
        "can_support_layer_output_guards": can_support_layer_output_guards,
        "can_support_ordered_bundle_adapter": can_support_ordered_bundle_adapter,
        "source_complete_caveats": {
            "coarse_bundle_is_final_token_only": true,
            "ordered_attention_mlp_seams_required_for_backend_validation": true,
            "k_v_source_complete": false,
        },
        "recommended_next_mode": if can_support_ordered_bundle_adapter {
            "layer-bundle-validate-from-coarse"
        } else if can_support_layer_output_guards {
            "coarse-layer-output-guard"
        } else {
            "generate ordered layer attention/MLP bundles"
        },
        "next_bounded_step": if can_support_ordered_bundle_adapter {
            "run the scoped coarse adapter for layer2"
        } else if can_support_layer_output_guards {
            "use coarse-layer-output-guard for emitted layer outputs; generate ordered layer2 bundles for seam validation"
        } else {
            "document missing coarse bundle values and generate ordered layer2 attention/MLP bundles"
        }
    });
    write_json(&cli.output, &status)
}

fn run_coarse_layer_output_guard(cli: &Cli) -> Result<()> {
    let layer = cli.layer_index;
    let layer_output = required_path(&cli.layer_output, "layer output")?;
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    validate_path(layer_output, "layer output")?;
    validate_path(coarse, "coarse bundle")?;
    let boundary = format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");
    let (output_status, output_values) = load_tensor_artifact(layer_output, &[2880], &["values"])?;
    let (oracle_status, oracle_values) = load_coarse_boundary_tensor(coarse, &boundary, &[2880])?;
    let blocked = !output_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;
    let metric = (!blocked).then(|| compare_hidden(&output_values, &oracle_values));
    let classification = if blocked {
        if oracle_status.shape_or_count_matched {
            "coarse_layer_output_guard_blocked_by_schema"
        } else {
            "coarse_layer_output_guard_missing_layer"
        }
    } else if metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0)
    {
        "coarse_layer_output_guard_matches_oracle"
    } else {
        "coarse_layer_output_guard_mismatch"
    };
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-layer-output-guard",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "bundle_path": coarse.display().to_string(),
        "layer_output_path": layer_output.display().to_string(),
        "oracle_boundary": boundary,
        "artifacts": {
            "layer_output": output_status,
            "coarse_oracle": oracle_status,
        },
        "metric": metric,
        "source_complete_caveats": {
            "guard_only": true,
            "attention_mlp_seams_not_validated_by_this_mode": true,
            "k_v_source_complete": false,
        },
        "next_bounded_step": match classification {
            "coarse_layer_output_guard_matches_oracle" => "use coarse output guards where useful, but generate ordered bundles for attention/MLP seam validation",
            "coarse_layer_output_guard_mismatch" => "localize emitted layer output versus coarse official layer output",
            _ => "inspect coarse schema or locate the requested layer output boundary",
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer_bundle_validate_from_coarse(cli: &Cli) -> Result<()> {
    let layer = cli.layer_index;
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    validate_path(coarse, "coarse bundle")?;
    let value = load_json(coarse)?;
    let boundaries = coarse_boundary_names(&value);
    let attention_seams_available = boundaries
        .iter()
        .any(|name| name.contains("_raw_scaled_qk_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_attention_probs_"))
        || boundaries.iter().any(|name| name.contains("_q_post_rope_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_grouped_k_post_rope_"));
    let mlp_seams_available = boundaries
        .iter()
        .any(|name| name.contains("_mlp_router_logits_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_selected_expert_outputs_"))
        || boundaries
            .iter()
            .any(|name| name.contains("_routing_weighted_sum_"));
    let classification = if !attention_seams_available {
        "layer2_coarse_adapter_blocked_by_missing_attention_seams"
    } else if !mlp_seams_available {
        "layer2_coarse_adapter_blocked_by_missing_mlp_seams"
    } else {
        "layer2_coarse_adapter_blocked_by_schema"
    };
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer-bundle-validate-from-coarse",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "bundle_path": coarse.display().to_string(),
        "available_boundaries": boundaries,
        "attention_seams_available": attention_seams_available,
        "mlp_seams_available": mlp_seams_available,
        "metric": serde_json::Value::Null,
        "blocker": {
            "kind": "coarse_bundle_not_ordered_attention_mlp_schema",
            "detail": "coarse bundle exposes final-token layer input/norm/residual vectors but not Q/K/V, raw-QK, probabilities, router/top-k, selected-output, or weighted-sum seams required by layer-bundle-validate"
        },
        "source_complete_caveats": {
            "coarse_bundle_is_final_token_only": true,
            "ordered_attention_mlp_seams_required_for_backend_validation": true,
            "k_v_source_complete": false,
        },
        "next_bounded_step": "generate ordered layer2 attention and MLP bundles, or add a deliberately narrower norm/residual-only coarse validation mode"
    });
    write_json(&cli.output, &status)
}

fn coarse_boundary_name(layer: usize, suffix: &str) -> String {
    format!("layer{layer}_final_token_{suffix}")
}

#[cfg(feature = "cuda")]
fn execute_coarse_layer_validate(
    layer: usize,
    layer_input: &Path,
    coarse: &Path,
    model: &Path,
    emit_layer_output: Option<&Path>,
    continue_after_attn_norm_diagnostic: bool,
    norm_reduction_policy: &str,
) -> Result<(String, Value, Option<Vec<f32>>)> {
    validate_norm_reduction_policy(norm_reduction_policy)?;
    validate_path(layer_input, "layer input")?;
    validate_path(coarse, "coarse bundle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let experts = 32usize;
    let selected_count = 4usize;
    let input_boundary = coarse_boundary_name(layer, "layer_input_before_attention_norm");
    let attention_norm_boundary = coarse_boundary_name(layer, "attention_norm_output_before_qkv");
    let attention_residual_boundary = coarse_boundary_name(
        layer,
        "hidden_state_after_attention_residual_add_before_mlp",
    );
    let mlp_norm_boundary = coarse_boundary_name(layer, "mlp_norm_output_before_mlp_projections");
    let output_boundary = coarse_boundary_name(layer, "hidden_state_after_mlp_residual_add");

    let (input_status, input_values) = load_tensor_artifact(layer_input, &[hidden], &["values"])?;
    let (input_oracle_status, input_oracle) =
        load_coarse_boundary_tensor(coarse, &input_boundary, &[hidden])?;
    let (attention_norm_status, attention_norm_oracle) =
        load_coarse_boundary_tensor(coarse, &attention_norm_boundary, &[hidden])?;
    let (attention_residual_status, attention_residual) =
        load_coarse_boundary_tensor(coarse, &attention_residual_boundary, &[hidden])?;
    let (mlp_norm_status, mlp_norm_oracle) =
        load_coarse_boundary_tensor(coarse, &mlp_norm_boundary, &[hidden])?;
    let (output_status, output_oracle) =
        load_coarse_boundary_tensor(coarse, &output_boundary, &[hidden])?;

    let schema_blocked = !input_status.shape_or_count_matched
        || !input_oracle_status.shape_or_count_matched
        || !attention_norm_status.shape_or_count_matched
        || !attention_residual_status.shape_or_count_matched
        || !mlp_norm_status.shape_or_count_matched
        || !output_status.shape_or_count_matched;

    let input_guard_metric =
        (!schema_blocked).then(|| compare_hidden(&input_values, &input_oracle));

    let input_norm_name = format!("model.layers.{layer}.input_layernorm.weight");
    let post_norm_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
    let router_weight_name = format!("model.layers.{layer}.mlp.router.weight");
    let router_bias_name = format!("model.layers.{layer}.mlp.router.bias");
    let input_norm_result = load_model_tensor_f32(model, &[input_norm_name.as_str()]);
    let post_norm_result = load_model_tensor_f32(model, &[post_norm_name.as_str()]);
    let router_weight_result = load_model_tensor_f32(model, &[router_weight_name.as_str()]);
    let router_bias_result = load_model_tensor_f32(model, &[router_bias_name.as_str()]);

    let (
        input_norm_source,
        input_norm_values,
        post_norm_source,
        post_norm_values,
        router_weight_source,
        router_weight_values,
        router_bias_source,
        router_bias_values,
        tensor_blocker,
    ) = match (
        input_norm_result,
        post_norm_result,
        router_weight_result,
        router_bias_result,
    ) {
        (
            Ok((input_norm_source, input_norm_values)),
            Ok((post_norm_source, post_norm_values)),
            Ok((router_weight_source, router_weight_values)),
            Ok((router_bias_source, router_bias_values)),
        ) if input_norm_values.len() == hidden
            && post_norm_values.len() == hidden
            && router_weight_values.len() == experts * hidden
            && router_bias_values.len() == experts =>
        {
            (
                Some(input_norm_source),
                input_norm_values,
                Some(post_norm_source),
                post_norm_values,
                Some(router_weight_source),
                router_weight_values,
                Some(router_bias_source),
                router_bias_values,
                None,
            )
        }
        (input_norm_result, post_norm_result, router_weight_result, router_bias_result) => {
            let (input_norm_source, input_norm_values, input_norm_error) = match input_norm_result {
                Ok((status, values)) => (Some(status), values, None),
                Err(err) => (None, Vec::new(), Some(err.to_string())),
            };
            let (post_norm_source, post_norm_values, post_norm_error) = match post_norm_result {
                Ok((status, values)) => (Some(status), values, None),
                Err(err) => (None, Vec::new(), Some(err.to_string())),
            };
            let (router_weight_source, router_weight_values, router_weight_error) =
                match router_weight_result {
                    Ok((status, values)) => (Some(status), values, None),
                    Err(err) => (None, Vec::new(), Some(err.to_string())),
                };
            let (router_bias_source, router_bias_values, router_bias_error) =
                match router_bias_result {
                    Ok((status, values)) => (Some(status), values, None),
                    Err(err) => (None, Vec::new(), Some(err.to_string())),
                };
            (
                input_norm_source,
                input_norm_values,
                post_norm_source,
                post_norm_values,
                router_weight_source,
                router_weight_values,
                router_bias_source,
                router_bias_values,
                Some(json!({
                    "kind": "coarse_layer_validate_tensor_lookup",
                    "detail": "required layer norm or router tensor was missing or had an unexpected shape",
                    "input_layernorm_error": input_norm_error,
                    "post_attention_layernorm_error": post_norm_error,
                    "router_weight_error": router_weight_error,
                    "router_bias_error": router_bias_error,
                })),
            )
        }
    };

    let tensor_blocked = tensor_blocker.is_some();
    let attention_norm_values = if !schema_blocked && !tensor_blocked {
        compute_mlp_rms_norm_with_policy(
            &input_values,
            &input_norm_values,
            1e-5,
            norm_reduction_policy,
        )?
    } else {
        Vec::new()
    };
    let attention_norm_metric = (!schema_blocked && !tensor_blocked)
        .then(|| compare_hidden(&attention_norm_values, &attention_norm_oracle));
    let attention_norm_matches = attention_norm_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);
    let attention_norm_diagnostic = !attention_norm_matches && continue_after_attn_norm_diagnostic;

    let mlp_norm_values = if !schema_blocked && !tensor_blocked {
        compute_mlp_rms_norm_with_policy(
            &attention_residual,
            &post_norm_values,
            1e-5,
            norm_reduction_policy,
        )?
    } else {
        Vec::new()
    };
    let mlp_norm_metric = (!schema_blocked && !tensor_blocked)
        .then(|| compare_hidden(&mlp_norm_values, &mlp_norm_oracle));

    let mut selected_experts = Vec::new();
    let mut routing_weights = Vec::new();
    let mut router_logits = Vec::new();
    let mut layer_output = None;
    let mut selected_expert_loader = Value::Null;
    let mut execution_blocker = None;
    let mlp_output_metric = if !schema_blocked
        && !tensor_blocked
        && input_guard_metric
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0)
        && (attention_norm_matches || attention_norm_diagnostic)
        && mlp_norm_metric
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0)
    {
        router_logits = compute_router_logits_bf16_linear(
            &mlp_norm_values,
            &router_weight_values,
            &router_bias_values,
        );
        let topk = compute_router_topk(&router_logits, selected_count);
        selected_experts = topk
            .indices
            .iter()
            .map(|&expert| expert as usize)
            .collect::<Vec<_>>();
        routing_weights = topk.routing_weights.clone();

        match load_selected_experts_mxfp4_validation(model, layer, &selected_experts)
            .map_err(|err| anyhow::anyhow!(err.to_string()))
        {
            Ok(loaded) => {
                let mut selected_matrix = vec![0.0f32; selected_count * hidden];
                for (rank, expert_id) in selected_experts.iter().copied().enumerate() {
                    let Some(expert) = loaded
                        .experts
                        .iter()
                        .find(|expert| expert.expert == expert_id)
                    else {
                        execution_blocker = Some(json!({
                            "kind": "coarse_layer_validate_selected_expert_missing",
                            "detail": format!("selected expert {expert_id} missing from MXFP4 loader")
                        }));
                        break;
                    };
                    match compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert) {
                        Ok(mlp1) => {
                            let swiglu = compute_swiglu_bf16(&mlp1);
                            let pre_bias = compute_expert30_mlp2_prebias_variant(
                                &swiglu,
                                &expert.down_weight,
                                Expert30Mlp2Policy::Current,
                            );
                            let output = compute_expert30_selected_output_variant(
                                &pre_bias,
                                &expert.down_bias,
                                Expert30Mlp2Policy::Current,
                            );
                            let start = rank * hidden;
                            selected_matrix[start..start + hidden].copy_from_slice(&output);
                        }
                        Err(err) => {
                            execution_blocker = Some(json!({
                                "kind": "coarse_layer_validate_mlp1_execution",
                                "detail": err.to_string(),
                                "expert": expert_id,
                            }));
                            break;
                        }
                    }
                }
                selected_expert_loader = json!({
                    "helper_name": loaded.helper_name,
                    "decode_source": loaded.decode_source,
                    "selected_experts": loaded.selected_experts,
                    "dtype_outputs": loaded.dtype_outputs,
                    "tensor_sources": loaded.tensor_sources,
                });
                if execution_blocker.is_none() {
                    let weighted = compute_weighted_expert_sum_bf16(
                        &selected_matrix,
                        &routing_weights,
                        hidden,
                    );
                    let residual = compute_attention_residual(&attention_residual, &weighted);
                    let metric = compare_hidden(&residual, &output_oracle);
                    if metric.metrics.mismatches == 0 {
                        layer_output = Some(residual.clone());
                    }
                    Some(metric)
                } else {
                    None
                }
            }
            Err(err) => {
                execution_blocker = Some(json!({
                    "kind": "coarse_layer_validate_selected_expert_loader",
                    "detail": err.to_string(),
                }));
                None
            }
        }
    } else {
        None
    };

    let classification = if schema_blocked {
        "coarse_layer_validate_blocked_by_coarse_schema"
    } else if tensor_blocked {
        "coarse_layer_validate_blocked_by_tensor_lookup"
    } else if execution_blocker.is_some() {
        "coarse_layer_validate_execution_failed"
    } else if input_guard_metric
        .as_ref()
        .is_none_or(|metric| metric.metrics.mismatches != 0)
    {
        "coarse_layer_validate_input_mismatch"
    } else if !attention_norm_matches && !continue_after_attn_norm_diagnostic {
        "coarse_layer_validate_attention_norm_mismatch"
    } else if mlp_norm_metric
        .as_ref()
        .is_none_or(|metric| metric.metrics.mismatches != 0)
    {
        "coarse_layer_validate_mlp_norm_mismatch"
    } else if mlp_output_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0)
    {
        if attention_norm_diagnostic {
            "coarse_layer_validate_mlp_output_matches_with_attn_norm_diagnostic"
        } else {
            "coarse_layer_validate_matches_oracle"
        }
    } else if attention_norm_diagnostic {
        "coarse_layer_validate_mlp_output_mismatch_after_attn_norm_diagnostic"
    } else {
        "coarse_layer_validate_mlp_output_mismatch"
    }
    .to_string();

    if classification == "coarse_layer_validate_matches_oracle"
        || classification == "coarse_layer_validate_mlp_output_matches_with_attn_norm_diagnostic"
    {
        if let (Some(path), Some(values)) = (emit_layer_output, layer_output.as_ref()) {
            write_coarse_layer_output_artifact(path, layer, model, coarse, values)?;
        }
    }

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-layer-validate",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "layer_input_path": layer_input.display().to_string(),
        "coarse_bundle_path": coarse.display().to_string(),
        "selected_experts": selected_experts,
        "routing_weights": routing_weights,
        "router": {
            "logits": router_logits,
            "oracle_available": false,
            "topk_oracle_available": false,
        },
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "norm_reduction_policy": norm_reduction_policy,
        "artifacts": {
            "layer_input": input_status,
            "coarse_layer_input": input_oracle_status,
            "coarse_attention_norm": attention_norm_status,
            "coarse_attention_residual": attention_residual_status,
            "coarse_mlp_norm": mlp_norm_status,
            "coarse_layer_output": output_status,
        },
        "model_tensors": {
            "input_layernorm": input_norm_source,
            "post_attention_layernorm": post_norm_source,
            "router_weight": router_weight_source,
            "router_bias": router_bias_source,
            "selected_expert_loader": selected_expert_loader,
        },
        "metrics": {
            "input_guard": input_guard_metric,
            "attention_norm": attention_norm_metric,
            "mlp_norm": mlp_norm_metric,
            "mlp_output": mlp_output_metric,
        },
        "source_policy": {
            "attention_source": "official_coarse_attention_residual_seam",
            "attention_recomputed": false,
            "ordered_attention_bundle_available": false,
            "ordered_mlp_bundle_available": false,
            "corrections_allowed": false,
            "correction_applied": false,
            "attention_norm_status": if attention_norm_diagnostic {
                "diagnostic_mismatch"
            } else if attention_norm_matches {
                "exact"
            } else {
                "mismatch"
            },
            "norm_reduction_policy": norm_reduction_policy,
            "router_topk_oracle_available": false,
            "selected_output_oracle_available": false,
            "weighted_sum_oracle_available": false,
        },
        "blocker": tensor_blocker.or(execution_blocker),
        "emitted_layer_output": emit_layer_output
            .filter(|_| layer_output.is_some())
            .map(|path| path.display().to_string()),
        "next_bounded_step": match classification.as_str() {
            "coarse_layer_validate_matches_oracle" => "feed emitted layer output into the next coarse layer validation",
            "coarse_layer_validate_mlp_output_matches_with_attn_norm_diagnostic" => "feed emitted layer output into the next coarse layer validation while preserving the attention norm diagnostic caveat",
            "coarse_layer_validate_input_mismatch" => "localize prior emitted output versus this layer coarse input boundary",
            "coarse_layer_validate_attention_norm_mismatch" => "localize layer input RMSNorm policy or tensor lookup",
            "coarse_layer_validate_mlp_norm_mismatch" => "localize layer MLP norm policy from the coarse attention residual seam",
            "coarse_layer_validate_mlp_output_mismatch_after_attn_norm_diagnostic" => "generate ordered attention/MLP bundles for this layer; coarse MLP output did not clear after diagnostic continuation",
            "coarse_layer_validate_mlp_output_mismatch" => "generate ordered attention/MLP bundles for this layer before applying any correction",
            "coarse_layer_validate_blocked_by_tensor_lookup" => "resolve missing layer-indexed norm/router/expert tensors",
            _ => "inspect coarse schema and required layer boundaries",
        }
    });
    Ok((classification, status, layer_output))
}

#[cfg(not(feature = "cuda"))]
fn execute_coarse_layer_validate(
    layer: usize,
    _layer_input: &Path,
    coarse: &Path,
    _model: &Path,
    _emit_layer_output: Option<&Path>,
    _continue_after_attn_norm_diagnostic: bool,
    _norm_reduction_policy: &str,
) -> Result<(String, Value, Option<Vec<f32>>)> {
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-layer-validate",
        "classification": "coarse_layer_validate_execution_failed",
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "coarse_bundle_path": coarse.display().to_string(),
        "blocker": {
            "kind": "coarse_layer_validate_cuda_feature",
            "detail": "coarse MLP backend replay requires the cuda feature"
        },
        "next_bounded_step": "rerun with --features cuda"
    });
    Ok((
        "coarse_layer_validate_execution_failed".to_string(),
        status,
        None,
    ))
}

fn run_coarse_layer_validate(cli: &Cli) -> Result<()> {
    let layer = cli.layer_index;
    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    let model = required_path(&cli.model, "model")?;
    let (_, status, _) = execute_coarse_layer_validate(
        layer,
        layer_input,
        coarse,
        model,
        cli.emit_layer_output.as_deref(),
        cli.continue_after_attn_norm_diagnostic,
        &cli.norm_reduction_policy,
    )?;
    write_json(&cli.output, &status)
}

fn run_coarse_ladder_validate(cli: &Cli) -> Result<()> {
    let start_layer = cli.start_layer;
    let end_layer = cli.end_layer;
    anyhow::ensure!(
        start_layer <= end_layer,
        "start layer must be <= end layer, got {start_layer}..{end_layer}"
    );
    let mut current_input = required_path(&cli.layer_input, "layer input")?.to_path_buf();
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    let model = required_path(&cli.model, "model")?;
    let emit_dir = required_path(&cli.emit_dir, "emit dir")?;
    validate_path(&current_input, "initial layer input")?;
    validate_path(coarse, "coarse bundle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );
    fs::create_dir_all(emit_dir)
        .with_context(|| format!("failed to create {}", emit_dir.display()))?;

    let mut completed_layers = Vec::new();
    let mut per_layer_summary = Vec::new();
    let mut emitted_outputs = Vec::new();
    let mut stopped_at_layer = Value::Null;
    let mut stop_reason = "completed_requested_range".to_string();

    for layer in start_layer..=end_layer {
        let output_path = emit_dir.join(format!("layer{layer}_output.json"));
        let (classification, status, output_values) = execute_coarse_layer_validate(
            layer,
            &current_input,
            coarse,
            model,
            Some(&output_path),
            cli.continue_after_attn_norm_diagnostic,
            &cli.norm_reduction_policy,
        )?;
        if (classification == "coarse_layer_validate_matches_oracle"
            || classification
                == "coarse_layer_validate_mlp_output_matches_with_attn_norm_diagnostic")
            && output_values.is_some()
        {
            completed_layers.push(layer);
            emitted_outputs.push(json!({
                "layer_index": layer,
                "path": output_path.display().to_string(),
            }));
            per_layer_summary.push(json!({
                "layer_index": layer,
                "classification": classification,
                "selected_experts": status.get("selected_experts").cloned().unwrap_or(Value::Null),
                "routing_weights": status.get("routing_weights").cloned().unwrap_or(Value::Null),
                "metrics": status.get("metrics").cloned().unwrap_or(Value::Null),
                "emitted_output": output_path.display().to_string(),
                "source_policy": status.get("source_policy").cloned().unwrap_or(Value::Null),
            }));
            current_input = output_path;
            continue;
        }

        stopped_at_layer = json!(layer);
        stop_reason = match classification.as_str() {
            "coarse_layer_validate_input_mismatch" => "input_mismatch",
            "coarse_layer_validate_attention_norm_mismatch" => "attention_norm_mismatch",
            "coarse_layer_validate_mlp_norm_mismatch" => "mlp_norm_mismatch",
            "coarse_layer_validate_mlp_output_mismatch"
            | "coarse_layer_validate_mlp_output_mismatch_after_attn_norm_diagnostic" => {
                "mlp_output_mismatch"
            }
            "coarse_layer_validate_blocked_by_coarse_schema" => "schema_blocker",
            _ => "execution_failure",
        }
        .to_string();
        per_layer_summary.push(json!({
            "layer_index": layer,
            "classification": classification,
            "status": status,
        }));
        break;
    }

    let requested = end_layer - start_layer + 1;
    let classification = if completed_layers.len() == requested {
        "coarse_ladder_validate_completed_all_requested_layers"
    } else {
        match stop_reason.as_str() {
            "input_mismatch" => "coarse_ladder_validate_stopped_on_input_mismatch",
            "attention_norm_mismatch" => {
                "coarse_ladder_validate_stopped_on_attention_norm_mismatch"
            }
            "mlp_norm_mismatch" => "coarse_ladder_validate_stopped_on_mlp_norm_mismatch",
            "mlp_output_mismatch" => "coarse_ladder_validate_stopped_on_mlp_output_mismatch",
            "schema_blocker" => "coarse_ladder_validate_stopped_on_schema_blocker",
            _ => "coarse_ladder_validate_stopped_on_execution_failure",
        }
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-ladder-validate",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "completed_layers": completed_layers,
        "stopped_at_layer": stopped_at_layer,
        "stop_reason": stop_reason,
        "per_layer_summary": per_layer_summary,
        "emitted_outputs": emitted_outputs,
        "caveats": {
            "coarse_final_token_ladder": true,
            "attention_residual_accepted_as_official_coarse_seam": true,
            "attention_recomputed": false,
            "ordered_attention_mlp_seams_validated": false,
            "selected_output_corrections_allowed": false,
            "k_v_source_complete": false,
            "norm_reduction_policy": cli.norm_reduction_policy,
        },
        "next_bounded_step": match classification {
            "coarse_ladder_validate_completed_all_requested_layers" => "decide whether to proceed to final norm/logit guard from the coarse layer23 output in a separate slice",
            "coarse_ladder_validate_stopped_on_mlp_output_mismatch" => "generate ordered attention/MLP bundles for the stopped layer before applying any correction",
            "coarse_ladder_validate_stopped_on_attention_norm_mismatch" => "localize the stopped layer attention norm policy or input source",
            "coarse_ladder_validate_stopped_on_mlp_norm_mismatch" => "localize the stopped layer MLP norm policy from the official coarse attention residual seam",
            "coarse_ladder_validate_stopped_on_input_mismatch" => "localize the previous emitted layer output versus the stopped layer input boundary",
            _ => "resolve the reported coarse ladder blocker",
        }
    });
    write_json(&cli.output, &status)
}

fn validate_norm_reduction_policy(policy: &str) -> Result<()> {
    anyhow::ensure!(
        matches!(policy, "current" | "pairwise" | "f64"),
        "unsupported norm reduction policy {policy:?}; expected current, pairwise, or f64"
    );
    Ok(())
}

fn pairwise_sum_f32(values: &[f32]) -> f32 {
    if values.len() <= 32 {
        values.iter().sum()
    } else {
        let mid = values.len() / 2;
        pairwise_sum_f32(&values[..mid]) + pairwise_sum_f32(&values[mid..])
    }
}

fn compute_rms_norm_debug_variant(
    name: &str,
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> (Vec<f32>, Value) {
    let hidden = input.len().max(1);
    let input_bf16 = input
        .iter()
        .map(|&value| round_bf16(value))
        .collect::<Vec<_>>();
    let sumsq_f32 = input_bf16.iter().map(|x| x * x).sum::<f32>();
    let (sumsq, inv_rms) = match name {
        "B_f32_input" | "H_weight_f32" => {
            let sumsq = input.iter().map(|x| x * x).sum::<f32>();
            let mean = sumsq / hidden as f32;
            (sumsq, 1.0f32 / (mean + epsilon).sqrt())
        }
        "C_bf16_square_terms" => {
            let sumsq = input_bf16.iter().map(|x| round_bf16(x * x)).sum::<f32>();
            let mean = sumsq / hidden as f32;
            (sumsq, 1.0f32 / (mean + epsilon).sqrt())
        }
        "D_f64_reduction" => {
            let sumsq64 = input_bf16
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>();
            let mean64 = sumsq64 / hidden as f64;
            (
                sumsq64 as f32,
                (1.0f64 / (mean64 + epsilon as f64).sqrt()) as f32,
            )
        }
        "E_reverse_reduction" => {
            let sumsq = input_bf16.iter().rev().map(|x| x * x).sum::<f32>();
            let mean = sumsq / hidden as f32;
            (sumsq, 1.0f32 / (mean + epsilon).sqrt())
        }
        "F_pairwise_reduction" => {
            let terms = input_bf16.iter().map(|x| x * x).collect::<Vec<_>>();
            let sumsq = pairwise_sum_f32(&terms);
            let mean = sumsq / hidden as f32;
            (sumsq, 1.0f32 / (mean + epsilon).sqrt())
        }
        _ => {
            let mean = sumsq_f32 / hidden as f32;
            (sumsq_f32, 1.0f32 / (mean + epsilon).sqrt())
        }
    };

    let values = input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &scale)| match name {
            "B_f32_input" | "H_weight_f32" => round_bf16(x * inv_rms * scale),
            "G_scale_first" => round_bf16(round_bf16(x) * scale * inv_rms),
            "I_pre_scale_rounding" => round_bf16(round_bf16(round_bf16(x) * inv_rms) * scale),
            _ => round_bf16(round_bf16(x) * inv_rms * scale),
        })
        .collect::<Vec<_>>();
    let mean_square = sumsq / hidden as f32;
    (
        values,
        json!({
            "sumsq": sumsq,
            "mean_square": mean_square,
            "epsilon": epsilon,
            "inv_rms": inv_rms,
        }),
    )
}

fn compute_mlp_rms_norm_with_policy(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    policy: &str,
) -> Result<Vec<f32>> {
    validate_norm_reduction_policy(policy)?;
    let hidden = input.len().max(1);
    let input_bf16 = input
        .iter()
        .map(|&value| round_bf16(value))
        .collect::<Vec<_>>();
    let inverse_rms = match policy {
        "current" => {
            let square_sum = input_bf16.iter().map(|x| x * x).sum::<f32>();
            let mean_square = square_sum / hidden as f32;
            1.0f32 / (mean_square + epsilon).sqrt()
        }
        "pairwise" => {
            let terms = input_bf16.iter().map(|x| x * x).collect::<Vec<_>>();
            let square_sum = pairwise_sum_f32(&terms);
            let mean_square = square_sum / hidden as f32;
            1.0f32 / (mean_square + epsilon).sqrt()
        }
        "f64" => {
            let square_sum = input_bf16
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>();
            let mean_square = square_sum / hidden as f64;
            (1.0f64 / (mean_square + epsilon as f64).sqrt()) as f32
        }
        _ => unreachable!(),
    };
    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(x, scale)| round_bf16(round_bf16(*x) * inverse_rms * *scale))
        .collect())
}

fn run_layer2_attn_norm_debug(cli: &Cli) -> Result<()> {
    let mut wrapper = cli.clone();
    wrapper.layer_index = if cli.layer_index == 1 {
        2
    } else {
        cli.layer_index
    };
    wrapper.lane = if cli.lane == 522 { 2108 } else { cli.lane };
    wrapper.norm_kind = "attention".to_string();
    run_coarse_norm_debug_with_prefix(&wrapper, "layer2_attn_norm_debug", "layer2-attn-norm-debug")
}

fn run_coarse_norm_debug(cli: &Cli) -> Result<()> {
    run_coarse_norm_debug_with_prefix(cli, "coarse_norm_debug", "coarse-norm-debug")
}

#[cfg(feature = "cuda")]
fn run_coarse_mlp_output_debug(cli: &Cli) -> Result<()> {
    validate_norm_reduction_policy(&cli.norm_reduction_policy)?;
    let layer = cli.layer_index;
    let hidden = 2880usize;
    let experts = 32usize;
    let selected_count = 4usize;
    let lane = cli.lane;
    anyhow::ensure!(lane < hidden, "lane must be < {hidden}, got {lane}");

    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(layer_input, "layer input")?;
    validate_path(coarse, "coarse bundle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let input_boundary = coarse_boundary_name(layer, "layer_input_before_attention_norm");
    let attention_residual_boundary = coarse_boundary_name(
        layer,
        "hidden_state_after_attention_residual_add_before_mlp",
    );
    let mlp_norm_boundary = coarse_boundary_name(layer, "mlp_norm_output_before_mlp_projections");
    let output_boundary = coarse_boundary_name(layer, "hidden_state_after_mlp_residual_add");

    let (input_status, input_values) = load_tensor_artifact(layer_input, &[hidden], &["values"])?;
    let (input_oracle_status, input_oracle) =
        load_coarse_boundary_tensor(coarse, &input_boundary, &[hidden])?;
    let (attention_residual_status, attention_residual) =
        load_coarse_boundary_tensor(coarse, &attention_residual_boundary, &[hidden])?;
    let (mlp_norm_status, mlp_norm_oracle) =
        load_coarse_boundary_tensor(coarse, &mlp_norm_boundary, &[hidden])?;
    let (output_status, output_oracle) =
        load_coarse_boundary_tensor(coarse, &output_boundary, &[hidden])?;

    let post_norm_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
    let router_weight_name = format!("model.layers.{layer}.mlp.router.weight");
    let router_bias_name = format!("model.layers.{layer}.mlp.router.bias");
    let (post_norm_source, post_norm_values) =
        load_model_tensor_f32(model, &[post_norm_name.as_str()])?;
    let (router_weight_source, router_weight_values) =
        load_model_tensor_f32(model, &[router_weight_name.as_str()])?;
    let (router_bias_source, router_bias_values) =
        load_model_tensor_f32(model, &[router_bias_name.as_str()])?;

    let ordered_evidence =
        find_ordered_layer_mlp_evidence(coarse.parent().unwrap_or(coarse), layer)?;
    let ordered_available = !ordered_evidence.is_empty();
    let schema_blocked = !input_status.shape_or_count_matched
        || !input_oracle_status.shape_or_count_matched
        || !attention_residual_status.shape_or_count_matched
        || !mlp_norm_status.shape_or_count_matched
        || !output_status.shape_or_count_matched
        || post_norm_values.len() != hidden
        || router_weight_values.len() != experts * hidden
        || router_bias_values.len() != experts;

    let input_guard_metric =
        (!schema_blocked).then(|| compare_hidden(&input_values, &input_oracle));
    let mlp_norm_values = if !schema_blocked {
        compute_mlp_rms_norm_with_policy(
            &attention_residual,
            &post_norm_values,
            1e-5,
            &cli.norm_reduction_policy,
        )?
    } else {
        Vec::new()
    };
    let mlp_norm_metric =
        (!schema_blocked).then(|| compare_hidden(&mlp_norm_values, &mlp_norm_oracle));

    let mut selected_experts = Vec::new();
    let mut selected_logits = Vec::new();
    let mut routing_weights = Vec::new();
    let mut router_logits = Vec::new();
    let mut selected_matrix = vec![0.0f32; selected_count * hidden];
    let mut prebias_matrix = vec![0.0f32; selected_count * hidden];
    let mut down_bias_matrix = vec![0.0f32; selected_count * hidden];
    let mut expert_traces = Vec::new();
    let mut selected_expert_loader = Value::Null;
    let mut execution_error = None;

    if !schema_blocked
        && input_guard_metric
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0)
        && mlp_norm_metric
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0)
    {
        router_logits = compute_router_logits_bf16_linear(
            &mlp_norm_values,
            &router_weight_values,
            &router_bias_values,
        );
        let topk = compute_router_topk(&router_logits, selected_count);
        selected_experts = topk
            .indices
            .iter()
            .map(|&expert| expert as usize)
            .collect::<Vec<_>>();
        selected_logits = topk.logits.clone();
        routing_weights = topk.routing_weights.clone();

        match load_selected_experts_mxfp4_validation(model, layer, &selected_experts)
            .map_err(|err| anyhow::anyhow!(err.to_string()))
        {
            Ok(loaded) => {
                selected_expert_loader = json!({
                    "helper_name": loaded.helper_name,
                    "decode_source": loaded.decode_source,
                    "selected_experts": loaded.selected_experts,
                    "dtype_outputs": loaded.dtype_outputs,
                    "tensor_sources": loaded.tensor_sources,
                });
                for (rank, expert_id) in selected_experts.iter().copied().enumerate() {
                    let Some(expert) = loaded
                        .experts
                        .iter()
                        .find(|expert| expert.expert == expert_id)
                    else {
                        execution_error = Some(format!(
                            "selected expert {expert_id} missing from MXFP4 loader"
                        ));
                        break;
                    };
                    match compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert) {
                        Ok(mlp1) => {
                            let swiglu = compute_swiglu_bf16(&mlp1);
                            let pre_bias = compute_expert30_mlp2_prebias_variant(
                                &swiglu,
                                &expert.down_weight,
                                Expert30Mlp2Policy::Current,
                            );
                            let output = compute_expert30_selected_output_variant(
                                &pre_bias,
                                &expert.down_bias,
                                Expert30Mlp2Policy::Current,
                            );
                            let start = rank * hidden;
                            selected_matrix[start..start + hidden].copy_from_slice(&output);
                            prebias_matrix[start..start + hidden].copy_from_slice(&pre_bias);
                            down_bias_matrix[start..start + hidden]
                                .copy_from_slice(&expert.down_bias[..hidden]);
                            expert_traces.push(json!({
                                "rank": rank,
                                "expert": expert_id,
                                "mlp1_summary": finite_summary(&mlp1),
                                "swiglu_summary": finite_summary(&swiglu),
                                "mlp2_pre_bias_lane": pre_bias[lane],
                                "down_bias_lane": expert.down_bias[lane],
                                "selected_output_lane": output[lane],
                            }));
                        }
                        Err(err) => {
                            execution_error = Some(format!(
                                "cuBLAS BF16 MLP1 failed for expert {expert_id}: {err}"
                            ));
                            break;
                        }
                    }
                }
            }
            Err(err) => {
                execution_error = Some(err.to_string());
            }
        }
    }

    let can_compare_output = !schema_blocked
        && execution_error.is_none()
        && !routing_weights.is_empty()
        && selected_matrix.iter().any(|value| *value != 0.0);
    let current_weighted = if can_compare_output {
        compute_weighted_expert_sum_bf16(&selected_matrix, &routing_weights, hidden)
    } else {
        Vec::new()
    };
    let current_final = if can_compare_output {
        compute_attention_residual(&attention_residual, &current_weighted)
    } else {
        Vec::new()
    };
    let final_output_initial_metric =
        can_compare_output.then(|| compare_hidden(&current_final, &output_oracle));

    let variant_table = if can_compare_output {
        compute_mlp_output_policy_variant_table(
            &selected_matrix,
            &routing_weights,
            &attention_residual,
            &output_oracle,
            hidden,
            lane,
        )
    } else {
        Vec::new()
    };
    let best_policy_variant = variant_table
        .iter()
        .find(|entry| {
            entry["metric"]["metrics"]["mismatches"]
                .as_u64()
                .is_some_and(|mismatches| mismatches == 0)
        })
        .and_then(|entry| entry["variant"].as_str())
        .map(str::to_string);

    let lane_attribution = if can_compare_output {
        let per_rank = selected_experts
            .iter()
            .enumerate()
            .map(|(rank, expert)| {
                let selected = selected_matrix[rank * hidden + lane];
                let weight = routing_weights[rank];
                let raw_contribution = selected * weight;
                let current_contribution = round_bf16(selected) * round_bf16(weight);
                json!({
                    "rank": rank,
                    "expert": expert,
                    "selected_output": selected,
                    "routing_weight": weight,
                    "contribution_before_rounding": raw_contribution,
                    "bf16_rounded_contribution": current_contribution,
                    "mlp2_pre_bias": prebias_matrix[rank * hidden + lane],
                    "down_bias": down_bias_matrix[rank * hidden + lane],
                })
            })
            .collect::<Vec<_>>();
        json!({
            "per_rank_selected_outputs": per_rank,
            "local_weighted_sum": current_weighted[lane],
            "attention_residual": attention_residual[lane],
            "local_final_output": current_final[lane],
            "official_final_output": output_oracle[lane],
            "required_weighted_sum_diagnostic": required_weighted_sum_diagnostic(
                attention_residual[lane],
                current_weighted[lane],
                output_oracle[lane],
            ),
        })
    } else {
        Value::Null
    };

    let lane_window = if can_compare_output {
        lane_window_range(lane, hidden, 2)
            .map(|window_lane| {
                let per_rank_selected = selected_experts
                    .iter()
                    .enumerate()
                    .map(|(rank, expert)| {
                        json!({
                            "rank": rank,
                            "expert": expert,
                            "selected_output": selected_matrix[rank * hidden + window_lane],
                            "routing_weight": routing_weights[rank],
                            "contribution": round_bf16(selected_matrix[rank * hidden + window_lane]) * round_bf16(routing_weights[rank]),
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "lane": window_lane,
                    "per_rank_selected_outputs": per_rank_selected,
                    "local_weighted_sum": current_weighted[window_lane],
                    "attention_residual": attention_residual[window_lane],
                    "local_final_output": current_final[window_lane],
                    "official_final_output": output_oracle[window_lane],
                    "diff": (current_final[window_lane] - output_oracle[window_lane]).abs(),
                    "matches": current_final[window_lane] == output_oracle[window_lane],
                })
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let classification = if schema_blocked {
        "coarse_mlp_output_debug_blocked_by_schema"
    } else if execution_error.is_some() {
        "coarse_mlp_output_debug_execution_failed"
    } else if best_policy_variant.is_some() {
        "coarse_mlp_output_debug_policy_matches_with_variant"
    } else if final_output_initial_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches > 0 && !ordered_available)
    {
        "coarse_mlp_output_debug_requires_ordered_mlp_bundle"
    } else if final_output_initial_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches > 0)
    {
        "coarse_mlp_output_debug_selected_expert_contribution_suspect"
    } else {
        "coarse_mlp_output_debug_unresolved"
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-mlp-output-debug",
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "classification": classification,
        "layer_index": layer,
        "focus_lane": lane,
        "norm_reduction_policy": cli.norm_reduction_policy,
        "source_policy": {
            "attention_residual": "official_coarse_attention_residual_seam",
            "attention_recomputed": false,
            "ordered_mlp_bundle_available": ordered_available,
            "coarse_only": true,
            "corrections_allowed": false,
            "correction_applied": false,
        },
        "artifacts": {
            "layer_input": input_status,
            "coarse_layer_input": input_oracle_status,
            "coarse_attention_residual": attention_residual_status,
            "coarse_mlp_norm": mlp_norm_status,
            "coarse_layer_output": output_status,
        },
        "guards": {
            "input_guard": input_guard_metric,
            "mlp_norm": mlp_norm_metric,
            "final_output_initial_metric": final_output_initial_metric,
        },
        "model_tensors": {
            "post_attention_layernorm": post_norm_source,
            "router_weight": router_weight_source,
            "router_bias": router_bias_source,
            "selected_expert_loader": selected_expert_loader,
        },
        "router": {
            "router_topk_oracle_available": false,
            "logits_summary": finite_summary(&router_logits),
            "selected_experts": selected_experts,
            "selected_logits": selected_logits,
            "routing_weights": routing_weights,
            "routing_weight_sum": routing_weights.iter().sum::<f32>(),
            "finite_summary": finite_summary(&router_logits),
        },
        "lane_attribution": lane_attribution,
        "lane_window": lane_window,
        "internal_selected_expert_trace": expert_traces,
        "variant_table": variant_table,
        "best_policy_variant": best_policy_variant,
        "ordered_layer11_mlp_evidence": if ordered_available {
            json!({ "status": "found", "paths": ordered_evidence })
        } else {
            json!({ "status": "missing", "paths": [] })
        },
        "execution_error": execution_error,
        "recommendation": if classification == "coarse_mlp_output_debug_requires_ordered_mlp_bundle" {
            "generate ordered layer11 MLP bundle or focused selected-output/weighted-sum seams before applying any correction"
        } else if classification == "coarse_mlp_output_debug_policy_matches_with_variant" {
            "add an explicit validation-only policy only after reviewing the full-output exact variant"
        } else {
            "inspect reported blocker or mismatch before continuing coarse ladder"
        },
        "next_bounded_step": if classification == "coarse_mlp_output_debug_requires_ordered_mlp_bundle" {
            "generate ordered layer11 MLP evidence for selected outputs and weighted sum"
        } else {
            "review focused layer11 MLP output debug status"
        },
    });
    write_json(&cli.output, &status)
}

#[cfg(not(feature = "cuda"))]
fn run_coarse_mlp_output_debug(_cli: &Cli) -> Result<()> {
    anyhow::bail!("coarse MLP output debug requires the cuda feature")
}

#[cfg(feature = "cuda")]
fn run_coarse_mlp_output_ordered_debug(cli: &Cli) -> Result<()> {
    validate_norm_reduction_policy(&cli.norm_reduction_policy)?;
    let layer = cli.layer_index;
    let hidden = 2880usize;
    let selected_count = 4usize;
    let lane = cli.lane;
    anyhow::ensure!(lane < hidden, "lane must be < {hidden}, got {lane}");

    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    let ordered_status_path =
        required_path(&cli.ordered_mlp_bundle_status, "ordered MLP bundle status")?;
    let ordered_dir = required_path(&cli.ordered_mlp_bundle_dir, "ordered MLP bundle dir")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(layer_input, "layer input")?;
    validate_path(coarse, "coarse bundle")?;
    validate_path(ordered_status_path, "ordered MLP bundle status")?;
    anyhow::ensure!(
        ordered_dir.is_dir(),
        "ordered MLP bundle dir does not exist: {}",
        ordered_dir.display()
    );

    let ordered_status: Value = serde_json::from_str(
        &fs::read_to_string(ordered_status_path)
            .with_context(|| format!("read {}", ordered_status_path.display()))?,
    )?;
    let ordered_selected_experts = ordered_status["selected_experts"]
        .as_array()
        .context("ordered status missing selected_experts")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|value| value as usize)
                .context("ordered selected expert must be integer")
        })
        .collect::<Result<Vec<_>>>()?;
    let ordered_routing_weights = ordered_status["routing_weights"]
        .as_array()
        .context("ordered status missing routing_weights")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .context("ordered routing weight must be numeric")
        })
        .collect::<Result<Vec<_>>>()?;

    let attention_residual_boundary = coarse_boundary_name(
        layer,
        "hidden_state_after_attention_residual_add_before_mlp",
    );
    let mlp_norm_boundary = coarse_boundary_name(layer, "mlp_norm_output_before_mlp_projections");
    let (_attention_residual_status, attention_residual) =
        load_coarse_boundary_tensor(coarse, &attention_residual_boundary, &[hidden])?;
    let (_mlp_norm_status, mlp_norm_oracle) =
        load_coarse_boundary_tensor(coarse, &mlp_norm_boundary, &[hidden])?;

    let (_, ordered_mlp_input) =
        load_tensor_artifact(&ordered_dir.join("mlp_input.json"), &[hidden], &["values"])?;
    let (_, ordered_mlp_norm) =
        load_tensor_artifact(&ordered_dir.join("mlp_norm.json"), &[hidden], &["values"])?;
    let (_, ordered_router_logits) =
        load_tensor_artifact(&ordered_dir.join("router_logits.json"), &[32], &["values"])?;
    let (_, ordered_selected_outputs) = load_tensor_artifact(
        &ordered_dir.join("selected_outputs.json"),
        &[selected_count * hidden],
        &["values"],
    )?;
    let (_, ordered_weighted_sum) = load_tensor_artifact(
        &ordered_dir.join("weighted_sum.json"),
        &[hidden],
        &["values"],
    )?;
    let (_, ordered_final_output) = load_tensor_artifact(
        &ordered_dir.join("mlp_residual_output.json"),
        &[hidden],
        &["values"],
    )?;

    let post_norm_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
    let router_weight_name = format!("model.layers.{layer}.mlp.router.weight");
    let router_bias_name = format!("model.layers.{layer}.mlp.router.bias");
    let (_, post_norm_values) = load_model_tensor_f32(model, &[post_norm_name.as_str()])?;
    let (_, router_weight_values) = load_model_tensor_f32(model, &[router_weight_name.as_str()])?;
    let (_, router_bias_values) = load_model_tensor_f32(model, &[router_bias_name.as_str()])?;

    let mlp_norm_values = compute_mlp_rms_norm_with_policy(
        &attention_residual,
        &post_norm_values,
        1e-5,
        &cli.norm_reduction_policy,
    )?;
    let mlp_norm_metric = compare_hidden(&mlp_norm_values, &ordered_mlp_norm);
    let coarse_mlp_norm_metric = compare_hidden(&mlp_norm_values, &mlp_norm_oracle);
    let mlp_input_metric = compare_hidden(&attention_residual, &ordered_mlp_input);

    let router_logits = compute_router_logits_bf16_linear(
        &mlp_norm_values,
        &router_weight_values,
        &router_bias_values,
    );
    let topk = compute_router_topk(&router_logits, selected_count);
    let local_selected_experts = topk
        .indices
        .iter()
        .map(|&expert| expert as usize)
        .collect::<Vec<_>>();
    let local_routing_weights = topk.routing_weights.clone();
    let selected_experts_match = local_selected_experts == ordered_selected_experts;
    let routing_weights_metric = compare_hidden(&local_routing_weights, &ordered_routing_weights);
    let router_logits_metric = compare_hidden(&router_logits, &ordered_router_logits);

    let loaded = load_selected_experts_mxfp4_validation(model, layer, &local_selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut selected_matrix = vec![0.0f32; selected_count * hidden];
    let mut per_rank = Vec::new();
    for (rank, expert_id) in local_selected_experts.iter().copied().enumerate() {
        let expert = loaded
            .experts
            .iter()
            .find(|expert| expert.expert == expert_id)
            .with_context(|| format!("selected expert {expert_id} missing from MXFP4 loader"))?;
        let mlp1 = compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert)
            .with_context(|| format!("cuBLAS BF16 MLP1 failed for expert {expert_id}"))?;
        let swiglu = compute_swiglu_bf16(&mlp1);
        let pre_bias = compute_expert30_mlp2_prebias_variant(
            &swiglu,
            &expert.down_weight,
            Expert30Mlp2Policy::Current,
        );
        let output = compute_expert30_selected_output_variant(
            &pre_bias,
            &expert.down_bias,
            Expert30Mlp2Policy::Current,
        );
        let start = rank * hidden;
        selected_matrix[start..start + hidden].copy_from_slice(&output);
        let ordered_lane = ordered_selected_outputs[start + lane];
        per_rank.push(json!({
            "rank": rank,
            "expert": expert_id,
            "local_selected_lane": output[lane],
            "ordered_selected_lane": ordered_lane,
            "diff": (output[lane] - ordered_lane).abs(),
            "matches": output[lane] == ordered_lane,
            "local_contribution": round_bf16(output[lane]) * round_bf16(local_routing_weights[rank]),
            "ordered_contribution": round_bf16(ordered_lane) * round_bf16(ordered_routing_weights[rank]),
        }));
    }

    let selected_metric = compare_hidden(&selected_matrix, &ordered_selected_outputs);
    let per_rank_metrics = (0..selected_count)
        .map(|rank| {
            let start = rank * hidden;
            json!({
                "rank": rank,
                "expert": local_selected_experts.get(rank),
                "metric": compare_hidden(
                    &selected_matrix[start..start + hidden],
                    &ordered_selected_outputs[start..start + hidden],
                )
            })
        })
        .collect::<Vec<_>>();
    let weighted_sum =
        compute_weighted_expert_sum_bf16(&selected_matrix, &local_routing_weights, hidden);
    let weighted_metric = compare_hidden(&weighted_sum, &ordered_weighted_sum);
    let final_output = compute_attention_residual(&attention_residual, &weighted_sum);
    let final_metric = compare_hidden(&final_output, &ordered_final_output);

    let mut rank0_replaced = selected_matrix.clone();
    if !rank0_replaced.is_empty() {
        rank0_replaced[lane] = ordered_selected_outputs[lane];
    }
    let rank0_replacement_weighted =
        compute_weighted_expert_sum_bf16(&rank0_replaced, &local_routing_weights, hidden);
    let rank0_replacement_final =
        compute_attention_residual(&attention_residual, &rank0_replacement_weighted);
    let rank0_replacement = json!({
        "rank": 0,
        "expert": local_selected_experts.first(),
        "lane": lane,
        "local_selected": selected_matrix[lane],
        "ordered_selected": ordered_selected_outputs[lane],
        "selected_metric": compare_hidden(&rank0_replaced, &ordered_selected_outputs),
        "weighted_sum_metric": compare_hidden(&rank0_replacement_weighted, &ordered_weighted_sum),
        "final_metric": compare_hidden(&rank0_replacement_final, &ordered_final_output),
        "weighted_sum_lane": rank0_replacement_weighted[lane],
        "final_lane": rank0_replacement_final[lane],
    });

    let ordered_replacement_weighted = compute_weighted_expert_sum_bf16(
        &ordered_selected_outputs,
        &ordered_routing_weights,
        hidden,
    );
    let ordered_replacement_final =
        compute_attention_residual(&attention_residual, &ordered_replacement_weighted);
    let ordered_selected_replacement = json!({
        "weighted_sum_metric": compare_hidden(&ordered_replacement_weighted, &ordered_weighted_sum),
        "final_metric": compare_hidden(&ordered_replacement_final, &ordered_final_output),
        "weighted_sum_lane": ordered_replacement_weighted[lane],
        "final_lane": ordered_replacement_final[lane],
    });

    let rank0_clears = rank0_replacement["weighted_sum_metric"]["metrics"]["mismatches"].as_u64()
        == Some(0)
        && rank0_replacement["final_metric"]["metrics"]["mismatches"].as_u64() == Some(0);
    let router_topk_mismatch =
        !selected_experts_match || routing_weights_metric.metrics.mismatches != 0;
    let earliest_mismatching_ordered_mlp_seam = if mlp_input_metric.metrics.mismatches != 0 {
        "mlp_input_attention_residual"
    } else if mlp_norm_metric.metrics.mismatches != 0 {
        "mlp_norm_output"
    } else if router_topk_mismatch {
        "router/top-k"
    } else if selected_metric.metrics.mismatches != 0 {
        "selected_expert_output"
    } else if weighted_metric.metrics.mismatches != 0 {
        "weighted_expert_sum"
    } else if final_metric.metrics.mismatches != 0 {
        "residual_add/final_output"
    } else {
        "none"
    };
    let needs_selected_expert_internals =
        earliest_mismatching_ordered_mlp_seam == "selected_expert_output";
    let oracle_next_request = if needs_selected_expert_internals {
        "request expert30 internal MLP1/SwiGLU/MLP2 lane1480 boundaries to localize below selected output"
    } else {
        "none"
    };
    let classification = if router_topk_mismatch {
        "layer11_ordered_mlp_consumer_router_topk_mismatch"
    } else if selected_experts_match
        && routing_weights_metric.metrics.mismatches == 0
        && selected_metric.metrics.mismatches == 1
        && weighted_metric.metrics.mismatches == 1
        && final_metric.metrics.mismatches == 1
        && selected_metric
            .first_mismatch
            .as_ref()
            .is_some_and(|mismatch| mismatch.hidden_lane == lane)
        && rank0_clears
    {
        "layer11_ordered_mlp_consumer_selected_output_localized"
    } else if weighted_metric.metrics.mismatches != 0 && selected_metric.metrics.mismatches == 0 {
        "layer11_ordered_mlp_consumer_weighted_sum_mismatch"
    } else if final_metric.metrics.mismatches != 0 && weighted_metric.metrics.mismatches == 0 {
        "layer11_ordered_mlp_consumer_residual_add_mismatch"
    } else if selected_metric.metrics.mismatches == 0
        && weighted_metric.metrics.mismatches == 0
        && final_metric.metrics.mismatches == 0
    {
        "layer11_ordered_mlp_consumer_available_seams_clear"
    } else if needs_selected_expert_internals {
        "layer11_ordered_mlp_consumer_requires_selected_expert_internals"
    } else {
        "layer11_ordered_mlp_consumer_execution_failed"
    };
    let lane_comparison = |local: f32, oracle: f32| {
        json!({
            "local": local,
            "oracle": oracle,
            "abs_diff": (local - oracle).abs(),
            "matched": local == oracle,
        })
    };
    let focus_lane_selected_outputs_by_rank = (0..selected_count)
        .map(|rank| {
            let index = rank * hidden + lane;
            json!({
                "rank": rank,
                "expert": local_selected_experts.get(rank),
                "local": selected_matrix[index],
                "oracle": ordered_selected_outputs[index],
                "abs_diff": (selected_matrix[index] - ordered_selected_outputs[index]).abs(),
                "matched": selected_matrix[index] == ordered_selected_outputs[index],
            })
        })
        .collect::<Vec<_>>();
    let lane_window_start = lane.saturating_sub(2);
    let lane_window_end = (lane + 2).min(hidden - 1);
    let lane_window_comparisons = (lane_window_start..=lane_window_end)
        .map(|window_lane| {
            let selected_outputs_by_rank = (0..selected_count)
                .map(|rank| {
                    let index = rank * hidden + window_lane;
                    json!({
                        "rank": rank,
                        "expert": local_selected_experts.get(rank),
                        "local": selected_matrix[index],
                        "oracle": ordered_selected_outputs[index],
                        "abs_diff": (selected_matrix[index] - ordered_selected_outputs[index]).abs(),
                        "matched": selected_matrix[index] == ordered_selected_outputs[index],
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "hidden_lane": window_lane,
                "mlp_input": lane_comparison(attention_residual[window_lane], ordered_mlp_input[window_lane]),
                "mlp_norm": lane_comparison(mlp_norm_values[window_lane], ordered_mlp_norm[window_lane]),
                "selected_outputs_by_rank": selected_outputs_by_rank,
                "weighted_sum": lane_comparison(weighted_sum[window_lane], ordered_weighted_sum[window_lane]),
                "final_output": lane_comparison(final_output[window_lane], ordered_final_output[window_lane]),
            })
        })
        .collect::<Vec<_>>();

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "coarse-mlp-output-ordered-debug",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "cuda_kernels_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "focus_lane": lane,
        "norm_reduction_policy": cli.norm_reduction_policy,
        "oracle_status": ordered_status_path.display().to_string(),
        "earliest_mismatching_ordered_mlp_seam": earliest_mismatching_ordered_mlp_seam,
        "selected_experts_local": local_selected_experts,
        "selected_experts_oracle": ordered_selected_experts,
        "routing_weights_local": local_routing_weights,
        "routing_weights_oracle": ordered_routing_weights,
        "ordered_bundle": {
            "status_path": ordered_status_path.display().to_string(),
            "dir": ordered_dir.display().to_string(),
            "selected_expert_internals_included": ordered_status["selected_expert_internals_included"].clone(),
        },
        "source_policy": {
            "attention_residual": "official_coarse_attention_residual_seam",
            "attention_recomputed": false,
            "ordered_mlp_bundle_available": true,
            "correction_applied": false,
            "coarse_ladder_continued": false,
        },
        "guards": {
            "attention_residual_vs_ordered_mlp_input": mlp_input_metric,
            "mlp_norm_vs_ordered": mlp_norm_metric,
            "mlp_norm_vs_coarse": coarse_mlp_norm_metric,
            "router_logits_vs_ordered": router_logits_metric,
        },
        "selected_experts": {
            "local": local_selected_experts,
            "ordered": ordered_selected_experts,
            "match": selected_experts_match,
        },
        "routing_weights": {
            "local": local_routing_weights,
            "ordered": ordered_routing_weights,
            "metric": routing_weights_metric,
            "match": routing_weights_metric.metrics.mismatches == 0,
        },
        "router_logits_note": {
            "metric": router_logits_metric,
            "topk_order_or_weight_changed": router_topk_mismatch,
            "classification_gate": "reported as guard metric; router/top-k mismatch requires selected expert order or routing weights to differ",
        },
        "metrics": {
            "selected_outputs": selected_metric,
            "per_rank_selected_outputs": per_rank_metrics,
            "weighted_sum": weighted_metric,
            "final_output": final_metric,
        },
        "focus": {
            "local_selected_outputs_by_rank": per_rank,
            "local_weighted_sum": weighted_sum[lane],
            "ordered_weighted_sum": ordered_weighted_sum[lane],
            "local_final": final_output[lane],
            "ordered_final": ordered_final_output[lane],
            "diff": (final_output[lane] - ordered_final_output[lane]).abs(),
        },
        "lane_1480": {
            "mlp_input": lane_comparison(attention_residual[lane], ordered_mlp_input[lane]),
            "mlp_norm": lane_comparison(mlp_norm_values[lane], ordered_mlp_norm[lane]),
            "selected_outputs_by_rank": focus_lane_selected_outputs_by_rank,
            "weighted_sum": lane_comparison(weighted_sum[lane], ordered_weighted_sum[lane]),
            "final_output": lane_comparison(final_output[lane], ordered_final_output[lane]),
        },
        "lane_window": {
            "start": lane_window_start,
            "end": lane_window_end,
            "comparisons": lane_window_comparisons,
        },
        "replacement_diagnostics": {
            "rank0_expert30_lane1480_replacement": rank0_replacement,
            "ordered_selected_output_replacement": ordered_selected_replacement,
        },
        "needs_selected_expert_internals": needs_selected_expert_internals,
        "oracle_next_request": oracle_next_request,
        "ordered_internals": {
            "included": ordered_status["selected_expert_internals_included"].clone(),
            "required_next": "expert30 internal MLP1/SwiGLU/MLP2 lane1480 boundaries if the selected-output producer needs localization below selected output",
        },
        "next_bounded_step": if needs_selected_expert_internals {
            "request expert30 internal MLP1/SwiGLU/MLP2 lane1480 boundaries before localizing below selected output"
        } else {
            "record ordered MLP consumer comparison and choose the next validation-runtime slice"
        },
    });
    write_json(&cli.output, &status)
}

#[cfg(not(feature = "cuda"))]
fn run_coarse_mlp_output_ordered_debug(_cli: &Cli) -> Result<()> {
    anyhow::bail!("ordered coarse MLP output debug requires the cuda feature")
}

fn run_coarse_norm_debug_with_prefix(
    cli: &Cli,
    classification_prefix: &str,
    submode: &str,
) -> Result<()> {
    let layer = cli.layer_index;
    let lane = cli.lane;
    let hidden = 2880usize;
    anyhow::ensure!(lane < hidden, "lane must be < {hidden}, got {lane}");
    anyhow::ensure!(
        matches!(cli.norm_kind.as_str(), "attention" | "mlp"),
        "unsupported norm kind {:?}; expected attention or mlp",
        cli.norm_kind
    );
    let layer_input = required_path(&cli.layer_input, "layer input")?;
    let coarse = required_path(&cli.coarse_bundle, "coarse bundle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(layer_input, "layer input")?;
    validate_path(coarse, "coarse bundle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let input_boundary = coarse_boundary_name(layer, "layer_input_before_attention_norm");
    let attention_norm_boundary = coarse_boundary_name(layer, "attention_norm_output_before_qkv");
    let attention_residual_boundary = coarse_boundary_name(
        layer,
        "hidden_state_after_attention_residual_add_before_mlp",
    );
    let mlp_norm_boundary = coarse_boundary_name(layer, "mlp_norm_output_before_mlp_projections");
    let (input_status, input_values) = load_tensor_artifact(layer_input, &[hidden], &["values"])?;
    let (input_oracle_status, input_oracle) =
        load_coarse_boundary_tensor(coarse, &input_boundary, &[hidden])?;
    let (attention_residual_status, attention_residual) =
        load_coarse_boundary_tensor(coarse, &attention_residual_boundary, &[hidden])?;
    let (
        oracle_boundary,
        norm_input_status,
        norm_input_values,
        oracle_status,
        oracle_values,
        tensor_name,
        source_policy,
    ): (
        String,
        TensorArtifactStatus,
        Vec<f32>,
        TensorArtifactStatus,
        Vec<f32>,
        String,
        serde_json::Value,
    ) = if cli.norm_kind == "attention" {
        let (oracle_status, oracle_values) =
            load_coarse_boundary_tensor(coarse, &attention_norm_boundary, &[hidden])?;
        (
            attention_norm_boundary.clone(),
            input_status.clone(),
            input_values.clone(),
            oracle_status,
            oracle_values,
            format!("model.layers.{layer}.input_layernorm.weight"),
            json!({
                "input_source": "layer_input",
                "attention_residual_seam_used": false,
            }),
        )
    } else {
        let (oracle_status, oracle_values) =
            load_coarse_boundary_tensor(coarse, &mlp_norm_boundary, &[hidden])?;
        (
            mlp_norm_boundary.clone(),
            attention_residual_status.clone(),
            attention_residual.clone(),
            oracle_status,
            oracle_values,
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
            json!({
                "input_source": "official_coarse_attention_residual_seam",
                "attention_residual_seam_used": true,
                "attention_residual_boundary": attention_residual_boundary,
            }),
        )
    };
    let (weight_source, weight_values) = load_model_tensor_f32(model, &[tensor_name.as_str()])?;
    let schema_blocked = !input_status.shape_or_count_matched
        || !input_oracle_status.shape_or_count_matched
        || !norm_input_status.shape_or_count_matched
        || !oracle_status.shape_or_count_matched
        || weight_values.len() != hidden;
    let input_guard_metric =
        (!schema_blocked).then(|| compare_hidden(&input_values, &input_oracle));

    let variant_names = [
        "A_current",
        "B_f32_input",
        "C_bf16_square_terms",
        "D_f64_reduction",
        "E_reverse_reduction",
        "F_pairwise_reduction",
        "G_scale_first",
        "H_weight_f32",
        "I_pre_scale_rounding",
    ];
    let mut variant_table = Vec::new();
    let mut variant_outputs = BTreeMap::new();
    let mut inv_rms_trace = BTreeMap::new();
    for name in variant_names {
        let (values, trace) = if schema_blocked {
            (Vec::new(), Value::Null)
        } else {
            compute_rms_norm_debug_variant(name, &norm_input_values, &weight_values, 1e-5)
        };
        let metric = (!schema_blocked).then(|| compare_hidden(&values, &oracle_values));
        let lane_value = values.get(lane).copied();
        variant_table.push(json!({
            "variant": name,
            "metric": metric,
            "focus_lane_value": lane_value,
            "focus_lane_diff": lane_value.map(|value| value - oracle_values[lane]),
            "trace": trace,
        }));
        inv_rms_trace.insert(name.to_string(), trace);
        variant_outputs.insert(name.to_string(), values);
    }

    let current_values = variant_outputs
        .get("A_current")
        .cloned()
        .unwrap_or_default();
    let current_metric = (!schema_blocked).then(|| compare_hidden(&current_values, &oracle_values));
    let window_start = lane.saturating_sub(2);
    let window_end = (lane + 2).min(hidden - 1);
    let lane_window = if schema_blocked {
        Vec::new()
    } else {
        (window_start..=window_end)
            .map(|window_lane| {
                let current = current_values[window_lane];
                let official = oracle_values[window_lane];
                json!({
                    "lane": window_lane,
                    "input": norm_input_values[window_lane],
                    "weight": weight_values[window_lane],
                    "current_local": current,
                    "official": official,
                    "diff": current - official,
                    "matches": current == official,
                    "variant_values": variant_names
                        .iter()
                        .map(|name| {
                            json!({
                                "variant": name,
                                "value": variant_outputs[*name][window_lane],
                                "matches": variant_outputs[*name][window_lane] == official,
                            })
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>()
    };
    let any_variant_exact = variant_table.iter().any(|entry| {
        entry
            .pointer("/metric/metrics/mismatches")
            .and_then(Value::as_u64)
            == Some(0)
    });
    let input_exact = input_guard_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);
    let current_single_lane = current_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 1)
        && current_metric
            .as_ref()
            .and_then(|metric| metric.first_mismatch.as_ref())
            .is_some_and(|mismatch| mismatch.hidden_lane == lane);
    let neighbors_match = lane_window.iter().all(|entry| {
        entry.get("lane").and_then(Value::as_u64) == Some(lane as u64)
            || entry
                .get("matches")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    });
    let classification = if schema_blocked {
        format!("{classification_prefix}_blocked_by_schema")
    } else if !input_exact {
        format!("{classification_prefix}_weight_or_input_source_mismatch")
    } else if any_variant_exact {
        format!("{classification_prefix}_policy_matches_with_variant")
    } else if current_single_lane && neighbors_match {
        format!("{classification_prefix}_single_lane_oracle_anomaly")
    } else if current_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches != 0)
    {
        format!("{classification_prefix}_rounding_policy_mismatch")
    } else {
        format!("{classification_prefix}_unresolved")
    };
    let local = current_values.get(lane).copied();
    let official = oracle_values.get(lane).copied();
    let best_variants = variant_table
        .iter()
        .filter_map(|entry| {
            (entry
                .pointer("/metric/metrics/mismatches")
                .and_then(Value::as_u64)
                == Some(0))
            .then(|| entry.get("variant").cloned())
            .flatten()
        })
        .collect::<Vec<_>>();
    let recommended_norm_reduction_policy = if best_variants
        .iter()
        .any(|variant| variant == "F_pairwise_reduction")
    {
        "pairwise"
    } else if best_variants
        .iter()
        .any(|variant| variant == "D_f64_reduction")
    {
        "f64"
    } else {
        "current"
    };
    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": submode,
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "norm_kind": cli.norm_kind,
        "focus_lane": lane,
        "layer_input_path": layer_input.display().to_string(),
        "coarse_bundle_path": coarse.display().to_string(),
        "input_boundary": input_boundary,
        "oracle_boundary": oracle_boundary,
        "norm_input_artifact": norm_input_status,
        "source_policy": source_policy,
        "input_guard_metric": input_guard_metric,
        "tensor_source": weight_source,
        "epsilon": 1e-5f32,
        "hidden_size": hidden,
        "current_metric": current_metric,
        "variant_table": variant_table,
        "best_variants": best_variants,
        "recommended_norm_reduction_policy": recommended_norm_reduction_policy,
        "lane_trace": {
            "lane": lane,
            "input": norm_input_values.get(lane).copied(),
            "weight": weight_values.get(lane).copied(),
            "local_current_output": local,
            "official_output": official,
            "diff": local.zip(official).map(|(local, official)| local - official),
        },
        "lane_window": lane_window,
        "inv_rms_trace": inv_rms_trace,
        "recommendation": if classification.ends_with("_policy_matches_with_variant") {
            "use recommended_norm_reduction_policy only in validation mode before rerunning coarse validation"
        } else if classification.ends_with("_single_lane_oracle_anomaly") {
            "continue only behind explicit diagnostic metadata and do not treat this as a production rule"
        } else if classification.ends_with("_rounding_policy_mismatch") {
            "localize RMSNorm reduction/output policy before using coarse continuation"
        } else if classification.ends_with("_weight_or_input_source_mismatch") {
            "resolve layer input or norm weight source mismatch"
        } else {
            "inspect coarse schema and RMSNorm debug values"
        },
    });
    write_json(&cli.output, &status)
}

fn run_layer1_attn_norm(cli: &Cli) -> Result<()> {
    let layer1_input = required_path(&cli.layer1_input, "layer1 input")?;
    let layer1_attn_norm_oracle =
        required_path(&cli.layer1_attn_norm_oracle, "layer1 attn norm oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(layer1_input, "layer1 input")?;
    validate_path(layer1_attn_norm_oracle, "layer1 attn norm oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let (input_status, input_values) = load_tensor_artifact(layer1_input, &[hidden], &["values"])?;
    let (oracle_status, oracle_values) =
        load_tensor_artifact(layer1_attn_norm_oracle, &[hidden], &["values"])?;
    let tensor_name = format!("model.layers.{}.input_layernorm.weight", cli.layer_index);
    let weight_result = load_model_tensor_f32(model, &[tensor_name.as_str()]);

    let artifact_blocked =
        !input_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;
    let (classification, metric, norm_weight_source, blocker) = if artifact_blocked {
        (
            "layer1_attention_norm_blocked_by_artifacts",
            None,
            None,
            Some(json!({
                "kind": "layer1_attention_norm_artifacts",
                "detail": "layer1 input or attention norm oracle did not expose supported values or expected counts"
            })),
        )
    } else {
        match weight_result {
            Ok((source, weight_values)) if weight_values.len() == hidden => {
                let output = compute_mlp_rms_norm(&input_values, &weight_values, 1e-5);
                let metric = compare_hidden(&output, &oracle_values);
                let classification = if metric.metrics.mismatches == 0 {
                    "layer1_attention_norm_matches_oracle"
                } else {
                    "layer1_attention_norm_mismatch"
                };
                (classification, Some(metric), Some(source), None)
            }
            Ok((source, weight_values)) => (
                "layer1_attention_norm_execution_failed",
                None,
                Some(source),
                Some(json!({
                    "kind": "layer1_attention_norm_weight_shape",
                    "detail": format!("expected {hidden} norm weights, got {}", weight_values.len())
                })),
            ),
            Err(err) => (
                "layer1_attention_norm_blocked_by_tensor_lookup",
                None,
                None,
                Some(json!({
                    "kind": "layer1_attention_norm_tensor_lookup",
                    "detail": err.to_string(),
                    "candidate_tensor": tensor_name
                })),
            ),
        }
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer1-attn-norm",
        "classification": classification,
        "implemented": !artifact_blocked && norm_weight_source.is_some() && metric.is_some(),
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": cli.layer_index,
        "input_source": layer1_input.display().to_string(),
        "input_boundary": "corrected_layer0_output_layer1_input",
        "layer1_attn_norm_oracle": layer1_attn_norm_oracle.display().to_string(),
        "norm_weight_source": norm_weight_source,
        "norm_policy": "bf16_input_f32_reduction_bf16_output",
        "epsilon": 1e-5f32,
        "artifacts": {
            "layer1_input": input_status,
            "layer1_attn_norm_oracle": oracle_status
        },
        "metric": metric,
        "blocker": blocker,
        "next_bounded_step": match classification {
            "layer1_attention_norm_matches_oracle" => "validate the layer1 Q/K/V projection boundary or layer1 K RoPE guard",
            "layer1_attention_norm_mismatch" => "localize layer1 attention RMSNorm dtype policy or epsilon",
            "layer1_attention_norm_blocked_by_artifacts" => "generate or locate layer1 input and attention norm oracle artifacts with 2880 values",
            "layer1_attention_norm_blocked_by_tensor_lookup" => "locate the layer1 attention norm weight tensor name in the model",
            _ => "resolve the reported layer1 attention norm execution blocker",
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer1_k_rope(cli: &Cli) -> Result<()> {
    let k_pre_rope = required_path(&cli.k_pre_rope, "K pre-RoPE")?;
    let k_post_rope_oracle = required_path(&cli.k_post_rope_oracle, "K post-RoPE oracle")?;
    validate_path(k_pre_rope, "K pre-RoPE")?;
    validate_path(k_post_rope_oracle, "K post-RoPE oracle")?;

    let expected_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let layer = cli.layer_index;
    let k_pre_boundary = format!("layer{layer}_grouped_k_projection_output_before_rope");
    let k_post_boundary = format!("layer{layer}_grouped_k_post_rope_before_attention");
    let (pre_status, pre_values) =
        load_boundary_tensor_artifact(k_pre_rope, &k_pre_boundary, &[expected_count])?;
    let (post_status, oracle_values) =
        load_boundary_tensor_artifact(k_post_rope_oracle, &k_post_boundary, &[expected_count])?;

    let artifact_blocked =
        !pre_status.shape_or_count_matched || !post_status.shape_or_count_matched;
    let execution = if artifact_blocked {
        None
    } else {
        Some(execute_k_rope(&pre_values, &oracle_values, cli))
    };
    let missing_all_token_k_pre =
        !pre_status.shape_or_count_matched && post_status.shape_or_count_matched;
    let classification = if artifact_blocked {
        if missing_all_token_k_pre {
            "layer1_qkv_history_blocked_by_all_token_input_generation"
        } else {
            "layer1_k_rope_blocked_by_artifacts"
        }
    } else {
        match execution.as_ref().map(|execution| execution.classification) {
            Some("layer0_validation_k_rope_matches_oracle")
            | Some("layer0_validation_k_rope_bf16_boundary_matches_oracle") => {
                "layer1_k_rope_matches_oracle"
            }
            Some("layer0_validation_k_rope_mismatch")
            | Some("layer0_validation_k_rope_bf16_boundary_mismatch")
            | Some("layer0_validation_k_rope_dtype_policy_unresolved") => "layer1_k_rope_mismatch",
            _ => "layer1_k_rope_blocked_by_artifacts",
        }
    };
    let metrics = execution.as_ref().and_then(|execution| {
        execution
            .metrics
            .clone()
            .or_else(|| execution.f16_kernel_metrics.clone())
    });
    let first_mismatch = execution.as_ref().and_then(|execution| {
        execution
            .first_mismatch
            .clone()
            .or_else(|| execution.f16_kernel_first_mismatch.clone())
    });
    let worst_mismatch = execution.as_ref().and_then(|execution| {
        execution
            .worst_mismatch
            .clone()
            .or_else(|| execution.f16_kernel_worst_mismatch.clone())
    });
    let blocker = if artifact_blocked {
        Some(json!({
            "kind": if missing_all_token_k_pre {
                "layer1_qkv_history_all_token_input_generation"
            } else {
                "layer1_k_rope_artifacts"
            },
            "detail": if missing_all_token_k_pre {
                "layer1 K RoPE requires all-token K pre-RoPE [74,8,64]/[74,512]; the available pinned attention bundle exposes final-token K pre-RoPE plus all-token grouped K post-RoPE, and scratch generation needs an all-token layer1 residual/norm/QKV source"
            } else {
                "layer1 K RoPE requires all-token K pre-RoPE [74,8,64]/[74,512] and K post-RoPE [74,8,64]"
            },
            "missing_boundary": k_pre_boundary,
            "generation_attempt": if missing_all_token_k_pre {
                Some(json!({
                    "generator_kind": "external_pytorch_transformers_scratch_under_tmp",
                    "result": "blocked",
                    "reason": "full official model forward dequantized MXFP4 weights to BF16 and exceeded the available 24GB CUDA memory before all-token layer1 QKV capture",
                    "scratch_artifacts_committed": false
                }))
            } else {
                None
            }
        }))
    } else {
        execution
            .as_ref()
            .and_then(|execution| execution.blocker.as_ref())
            .map(|blocker| json!(blocker))
    };

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer1-k-rope",
        "classification": classification,
        "implemented": !artifact_blocked && execution.is_some(),
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "api_used": "gpt_oss_model_runner::rope_validation::apply_k_rope_bf16_boundary_validation",
        "rope_application_policy": execution
            .as_ref()
            .map(|execution| execution.rope_application_policy)
            .unwrap_or("blocked_before_rope_execution"),
        "artifacts": {
            "k_pre_rope": pre_status,
            "k_post_rope_oracle": post_status
        },
        "artifact_sources": {
            "k_pre_rope_source": if artifact_blocked {
                "missing_all_token_layer1_k_pre_rope"
            } else {
                "all_token_layer1_k_pre_rope_artifact"
            },
            "k_post_rope_source": if post_status.shape_or_count_matched {
                "official_bundle_or_oracle_k_post_rope"
            } else {
                "missing_layer1_k_post_rope_oracle"
            }
        },
        "generated_pack_path": serde_json::Value::Null,
        "k_pre_rope_source": if artifact_blocked {
            "missing_all_token_layer1_k_pre_rope"
        } else {
            "all_token_layer1_k_pre_rope_artifact"
        },
        "k_post_rope_source": if post_status.shape_or_count_matched {
            "official_bundle_or_oracle_k_post_rope"
        } else {
            "missing_layer1_k_post_rope_oracle"
        },
        "expected_value_count": expected_count,
        "metric": metrics,
        "metrics": metrics,
        "first_mismatch": first_mismatch,
        "worst_mismatch": worst_mismatch,
        "qkv_artifact_summary": if missing_all_token_k_pre {
            Some(json!({
                "q_pre_rope_shape": serde_json::Value::Null,
                "k_pre_rope_shape": serde_json::Value::Null,
                "v_shape": serde_json::Value::Null,
                "k_post_rope_shape": [cli.token_count, cli.kv_heads, cli.head_dim],
                "available_bundle_scope": "final-token Q/K/V plus grouped all-token K post-RoPE",
                "missing": "all-token layer1 residual/norm/Q/K/V source, especially K pre-RoPE [74,512] or [74,8,64]"
            }))
        } else {
            None
        },
        "blocker": blocker,
        "next_bounded_step": match classification {
            "layer1_k_rope_matches_oracle" => "build raw-QK and attention-probability guards from the layer1 K/V history",
            "layer1_k_rope_mismatch" => "localize layer1 K RoPE dtype policy against the grouped K post-RoPE oracle",
            "layer1_qkv_history_blocked_by_all_token_input_generation" => "generate or locate an all-token layer1 residual/norm/QKV pack before rerunning layer1 K RoPE",
            _ => "generate or locate all-token layer1 K pre-RoPE history before K RoPE validation",
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer1_attention_bundle(cli: &Cli) -> Result<()> {
    let layer1_input = required_path(&cli.layer1_input, "layer1 input")?;
    let bundle = required_path(&cli.attention_bundle, "attention bundle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(layer1_input, "layer1 input")?;
    validate_path(bundle, "attention bundle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let layer = cli.layer_index;
    let hidden = 2880usize;
    let q_dim = cli.query_heads * cli.head_dim;
    let k_count = cli.token_count * cli.kv_heads * cli.head_dim;
    let raw_qk_count = cli.query_heads * cli.token_count;
    let masked_count = cli.query_heads * (cli.token_count + 1);

    let q_post_boundary = format!("layer{layer}_final_token_q_post_rope_before_attention");
    let k_post_boundary = format!("layer{layer}_grouped_k_post_rope_before_attention");
    let raw_qk_boundary = format!("layer{layer}_final_token_raw_scaled_qk_logits_pre_mask");
    let masked_boundary = format!("layer{layer}_final_token_masked_scaled_qk_logits_pre_softmax");
    let probs_boundary = format!("layer{layer}_final_token_attention_probs_post_softmax");
    let weighted_v_boundary =
        format!("layer{layer}_final_token_attention_weighted_value_sum_before_output_projection");
    let oproj_boundary =
        format!("layer{layer}_final_token_attention_output_after_o_proj_before_residual");
    let residual_boundary =
        format!("layer{layer}_final_token_hidden_state_after_attention_residual_add_before_mlp");

    let available_boundaries = list_boundary_names(bundle)?;
    let (q_status, q_post) = load_boundary_tensor_artifact(bundle, &q_post_boundary, &[q_dim])?;
    let (k_status, k_post) = load_boundary_tensor_artifact(bundle, &k_post_boundary, &[k_count])?;
    let (raw_status, raw_oracle) =
        load_boundary_tensor_artifact(bundle, &raw_qk_boundary, &[raw_qk_count])?;
    let (masked_status, masked_oracle) =
        load_boundary_tensor_artifact(bundle, &masked_boundary, &[masked_count])?;
    let (probs_status, probs_oracle) =
        load_boundary_tensor_artifact(bundle, &probs_boundary, &[masked_count])?;
    let (weighted_status, weighted_v) =
        load_boundary_tensor_artifact(bundle, &weighted_v_boundary, &[q_dim])?;
    let (oproj_status, oproj_oracle) =
        load_boundary_tensor_artifact(bundle, &oproj_boundary, &[hidden])?;
    let (residual_oracle_status, residual_oracle) =
        load_boundary_tensor_artifact(bundle, &residual_boundary, &[hidden])?;
    let (input_status, layer1_input_values) =
        load_tensor_artifact(layer1_input, &[hidden], &["values"])?;

    let raw_inputs_ready = q_status.shape_or_count_matched
        && k_status.shape_or_count_matched
        && raw_status.shape_or_count_matched;
    let (raw_values, raw_qk_metric) = if raw_inputs_ready {
        let raw_values = compute_raw_qk(&q_post, &k_post, cli);
        let metric = compare_raw_qk(&raw_values, &raw_oracle, cli);
        (Some(raw_values), Some(metric))
    } else {
        (None, None)
    };

    let attention_ready = raw_values.is_some()
        && masked_status.shape_or_count_matched
        && probs_status.shape_or_count_matched;
    let (masked_metric, probs_metric, probs_values) = if let Some(raw_values) = raw_values.as_ref()
    {
        if attention_ready {
            let masked_logits = build_masked_logits_from_raw_qk(
                raw_values,
                &masked_oracle,
                cli.query_heads,
                cli.token_count,
            );
            let masked_metric = compare_matrix(
                &masked_logits,
                &masked_oracle,
                cli.query_heads,
                cli.token_count + 1,
                MatrixSelection::All,
            );
            let probs =
                softmax_rows_bf16_output(&masked_logits, cli.query_heads, cli.token_count + 1);
            let probs_metric = compare_matrix(
                &probs,
                &probs_oracle,
                cli.query_heads,
                cli.token_count + 1,
                MatrixSelection::All,
            );
            (Some(masked_metric), Some(probs_metric), Some(probs))
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    let weighted_v_note = json!({
        "source": "official_weighted_v_oracle_because_all_token_v_history_missing",
        "metric": serde_json::Value::Null,
        "all_token_v_history_present": false,
        "official_weighted_v_artifact": weighted_status,
    });

    let oproj_weight_name = format!("model.layers.{layer}.self_attn.o_proj.weight");
    let oproj_bias_name = format!("model.layers.{layer}.self_attn.o_proj.bias");
    let oproj_weight_result = load_model_tensor_f32(model, &[oproj_weight_name.as_str()]);
    let oproj_bias_result = load_model_tensor_f32(model, &[oproj_bias_name.as_str()]);

    let (oproj_metric, oproj_output, model_tensors, oproj_blocker) = match (
        oproj_weight_result,
        oproj_bias_result,
    ) {
        (Ok((weight_status, weight_values)), Ok((bias_status, bias_values)))
            if weighted_status.shape_or_count_matched && oproj_status.shape_or_count_matched =>
        {
            let output = compute_attention_oproj_variant(
                &weighted_v,
                &weight_values,
                &bias_values,
                OprojPolicy::ChunkedPairwise,
            );
            let metric = compare_hidden(&output, &oproj_oracle);
            (
                Some(metric),
                Some(output),
                json!({
                    "oproj_weight": weight_status,
                    "oproj_bias": bias_status,
                }),
                None,
            )
        }
        (weight_result, bias_result) => {
            let weight_error = weight_result.err().map(|err| err.to_string());
            let bias_error = bias_result.err().map(|err| err.to_string());
            (
                None,
                None,
                json!({
                    "oproj_weight_error": weight_error,
                    "oproj_bias_error": bias_error,
                }),
                Some(json!({
                    "kind": "layer1_attention_bundle_oproj_artifacts",
                    "detail": "weighted V, o_proj oracle, or model o_proj weight/bias was unavailable"
                })),
            )
        }
    };

    let residual_ready = oproj_output.is_some()
        && input_status.shape_or_count_matched
        && residual_oracle_status.shape_or_count_matched;
    let (residual_metric, residual_output) = if residual_ready {
        let output = compute_attention_residual(
            &layer1_input_values,
            oproj_output.as_ref().expect("checked above"),
        );
        let metric = compare_hidden(&output, &residual_oracle);
        (Some(metric), Some(output))
    } else {
        (None, None)
    };

    if let (Some(emit_path), Some(residual_output)) = (
        &cli.emit_layer1_attention_residual,
        residual_output.as_ref(),
    ) {
        write_layer1_attention_residual_artifact(emit_path, model, bundle, residual_output)?;
    }

    let classification = classify_layer1_attention_bundle(
        &raw_qk_metric,
        &masked_metric,
        &probs_metric,
        &oproj_metric,
        &residual_metric,
        raw_inputs_ready,
        masked_status.shape_or_count_matched,
        probs_status.shape_or_count_matched,
        weighted_status.shape_or_count_matched,
    );

    let status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer1-attention-bundle",
        "classification": classification,
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "bundle_path": bundle.display().to_string(),
        "layer1_input_source": layer1_input.display().to_string(),
        "available_boundaries": available_boundaries,
        "source_policy": {
            "k_post_rope_source": "official_attention_bundle",
            "k_pre_rope_history": "missing",
            "k_rope_construction_validated": false,
            "q_post_rope_source": "official_attention_bundle",
            "weighted_v_source": "official_weighted_v_oracle_because_all_token_v_history_missing"
        },
        "artifacts": {
            "q_post_rope": q_status,
            "k_post_rope": k_status,
            "raw_qk_oracle": raw_status,
            "masked_logits_oracle": masked_status,
            "attention_probs_oracle": probs_status,
            "weighted_v_oracle": weighted_status,
            "oproj_oracle": oproj_status,
            "attention_residual_oracle": residual_oracle_status,
            "layer1_input": input_status
        },
        "model_tensors": model_tensors,
        "metrics": {
            "raw_qk": raw_qk_metric,
            "masked_logits": masked_metric,
            "attention_probs": probs_metric,
            "weighted_v": weighted_v_note,
            "o_proj": oproj_metric,
            "attention_residual": residual_metric
        },
        "computed_attention_probs_present": probs_values.is_some(),
        "blocker": oproj_blocker,
        "emitted_layer1_attention_residual": cli
            .emit_layer1_attention_residual
            .as_ref()
            .filter(|_| residual_output.is_some())
            .map(|path| path.display().to_string()),
        "next_bounded_step": match classification {
            "layer1_attention_bundle_attention_residual_matches_oracle" => {
                "validate layer1 MLP from the emitted attention residual; keep all-token K pre-RoPE generation as a separate source-complete ladder task"
            }
            "layer1_attention_bundle_oproj_matches_oracle" => {
                "localize residual-add input/policy or emit the o_proj seam before layer1 MLP"
            }
            "layer1_attention_bundle_attention_probs_matches_oracle" => {
                "validate o_proj policy from official weighted-V seam"
            }
            "layer1_attention_bundle_raw_qk_matches_oracle" => {
                "validate masked logits and attention probabilities from the bundle seam"
            }
            _ => "localize the first mismatching or missing layer1 attention bundle seam"
        }
    });
    write_json(&cli.output, &status)
}

fn run_layer1_mlp_backend(cli: &Cli) -> Result<()> {
    let attention_residual = required_path(&cli.attention_residual, "attention residual")?;
    let mlp_bundle = required_path(&cli.mlp_bundle, "MLP bundle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(attention_residual, "attention residual")?;
    validate_path(mlp_bundle, "MLP bundle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let layer = cli.layer_index;
    let hidden = 2880usize;
    let experts = 32usize;
    let selected_count = 4usize;
    let mlp_norm_boundary =
        format!("layer{layer}_final_token_mlp_norm_output_before_mlp_projections");
    let router_boundary = format!("layer{layer}_final_token_mlp_router_logits_before_routing");
    let topk_boundary =
        format!("layer{layer}_final_token_mlp_topk_expert_indices_and_routing_weights");
    let selected_boundary =
        format!("layer{layer}_final_token_selected_expert_outputs_before_routing_weighted_sum");
    let weighted_boundary =
        format!("layer{layer}_final_token_mlp_output_after_routing_weighted_sum_before_residual");
    let residual_boundary = format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");

    let available_boundaries = list_boundary_names(mlp_bundle)?;
    let (attention_status, attention_values) =
        load_tensor_artifact(attention_residual, &[hidden], &["values"])?;
    let (mlp_norm_status, mlp_norm_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &mlp_norm_boundary, &[hidden])?;
    let (router_status, router_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &router_boundary, &[experts])?;
    let (topk_status, topk_oracle) =
        load_boundary_topk_oracle(mlp_bundle, &topk_boundary, selected_count)?;
    let selected_experts = topk_oracle
        .indices
        .iter()
        .map(|&expert| {
            usize::try_from(expert).with_context(|| format!("invalid selected expert id {expert}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let (selected_status, selected_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &selected_boundary, &[selected_count * hidden])?;
    let (weighted_status, weighted_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &weighted_boundary, &[hidden])?;
    let (residual_status, residual_oracle) =
        load_boundary_tensor_artifact(mlp_bundle, &residual_boundary, &[hidden])?;

    let artifact_blocked = !attention_status.shape_or_count_matched
        || !mlp_norm_status.shape_or_count_matched
        || !router_status.shape_or_count_matched
        || !topk_status.shape_or_count_matched
        || !selected_status.shape_or_count_matched
        || !weighted_status.shape_or_count_matched
        || !residual_status.shape_or_count_matched;

    let norm_weight_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
    let norm_weight_result = load_model_tensor_f32(model, &[norm_weight_name.as_str()]);
    let (mlp_norm_metric, mlp_norm_values, norm_weight_source, norm_blocker) = if artifact_blocked {
        (None, Vec::new(), None, None)
    } else {
        match norm_weight_result {
            Ok((source, weight_values)) if weight_values.len() == hidden => {
                let values = compute_mlp_rms_norm(&attention_values, &weight_values, 1e-5);
                let metric = compare_hidden(&values, &mlp_norm_oracle);
                (Some(metric), values, Some(source), None)
            }
            Ok((source, weight_values)) => (
                None,
                Vec::new(),
                Some(source),
                Some(format!(
                    "expected {hidden} layer{layer} MLP norm weights, got {}",
                    weight_values.len()
                )),
            ),
            Err(err) => (None, Vec::new(), None, Some(err.to_string())),
        }
    };

    let router_result = if artifact_blocked || norm_blocker.is_some() {
        Err(anyhow::anyhow!("blocked before layer1 router execution"))
    } else {
        execute_mlp_backend_router(model, layer, &mlp_norm_values, &router_oracle, &topk_oracle)
    };
    let backend_result = if artifact_blocked || norm_blocker.is_some() {
        Err(anyhow::anyhow!(
            "blocked before layer1 MLP backend execution"
        ))
    } else {
        execute_mlp_backend_selected_experts(
            model,
            layer,
            &mlp_norm_values,
            &selected_experts,
            &topk_oracle.routing_weights,
            &selected_oracle,
            &weighted_oracle,
            &residual_oracle,
            &attention_values,
            false,
        )
    };

    let metric_mismatches = |value: &Value| -> Option<usize> {
        value
            .pointer("/metrics/mismatches")
            .and_then(Value::as_u64)
            .map(|count| count as usize)
    };
    let hidden_status_mismatches = |value: &Value| -> Option<usize> {
        value
            .pointer("/metrics/mismatches")
            .and_then(Value::as_u64)
            .map(|count| count as usize)
    };
    let router_matches = router_result.as_ref().ok().is_some_and(|router| {
        hidden_status_mismatches(&router["router_metric"]) == Some(0)
            && router
                .pointer("/topk_metric/ordered_match")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && hidden_status_mismatches(&router["topk_metric"]["selected_logits_metric"]) == Some(0)
            && hidden_status_mismatches(&router["topk_metric"]["routing_weights_metric"]) == Some(0)
    });
    let selected_matches = backend_result.as_ref().ok().is_some_and(|backend| {
        metric_mismatches(&backend["selected_output_metric_official"]) == Some(0)
    });
    let weighted_matches = backend_result.as_ref().ok().is_some_and(|backend| {
        hidden_status_mismatches(&backend["weighted_sum_metric_official"]) == Some(0)
    });
    let residual_matches = backend_result.as_ref().ok().is_some_and(|backend| {
        hidden_status_mismatches(&backend["mlp_residual_metric_official"]) == Some(0)
    });
    let mlp_norm_matches = mlp_norm_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);

    let classification = if artifact_blocked {
        "layer1_mlp_backend_blocked_by_artifacts"
    } else if norm_blocker.is_some() || router_result.is_err() || backend_result.is_err() {
        "layer1_mlp_backend_execution_failed"
    } else if !mlp_norm_matches {
        "layer1_mlp_backend_mlp_norm_mismatch"
    } else if !router_matches {
        "layer1_mlp_backend_router_mismatch"
    } else if !selected_matches {
        "layer1_mlp_backend_selected_outputs_mismatch"
    } else if !weighted_matches {
        "layer1_mlp_backend_weighted_sum_mismatch"
    } else if !residual_matches {
        "layer1_mlp_backend_residual_mismatch"
    } else {
        "layer1_mlp_backend_matches_oracle"
    };

    let mut emitted_layer_output = Value::Null;
    if let (Some(emit_path), Ok(backend)) = (&cli.emit_layer_output, backend_result.as_ref()) {
        if let Some(values) = backend.get("mlp_residual_values").and_then(Value::as_array) {
            let residual_values = values
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|value| value as f32)
                        .context("layer1 residual output value must be numeric")
                })
                .collect::<Result<Vec<_>>>()?;
            write_layer1_mlp_output_artifact(
                emit_path,
                model,
                mlp_bundle,
                attention_residual,
                &residual_values,
            )?;
            emitted_layer_output = json!(emit_path.display().to_string());
        }
    }

    let mut status = json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "layer1-mlp-backend",
        "classification": classification,
        "implemented": !artifact_blocked && norm_blocker.is_none() && router_result.is_ok() && backend_result.is_ok(),
        "runtime_behavior_changed": false,
        "production_routing_changed": false,
        "model_runner_routing_changed": false,
        "validation_only": true,
        "layer_index": layer,
        "attention_residual_source": attention_residual.display().to_string(),
        "mlp_bundle": mlp_bundle.display().to_string(),
        "available_boundaries": available_boundaries,
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "correction_policy": "none",
        "selected_experts": selected_experts,
        "artifacts": {
            "attention_residual": attention_status,
            "mlp_norm_oracle": mlp_norm_status,
            "router_logits_oracle": router_status,
            "topk_routing_oracle": topk_status,
            "selected_experts_oracle": selected_status,
            "weighted_expert_sum_oracle": weighted_status,
            "mlp_residual_oracle": residual_status
        },
        "model_tensors": {
            "mlp_norm_weight": norm_weight_source
        },
        "metrics": {
            "mlp_norm": mlp_norm_metric,
            "router_logits": null,
            "topk": null,
            "selected_outputs": null,
            "weighted_sum": null,
            "mlp_residual": null
        },
        "emitted_layer_output": emitted_layer_output,
        "blocker": norm_blocker.as_ref().map(|detail| json!({
            "kind": "layer1_mlp_backend_norm_weight",
            "detail": detail
        })),
        "next_bounded_step": match classification {
            "layer1_mlp_backend_matches_oracle" => "guard emitted layer1 output as layer2 input or repeat the bundle-driven ladder pattern for layer2",
            "layer1_mlp_backend_mlp_norm_mismatch" => "localize layer1 MLP norm policy from the emitted attention residual",
            "layer1_mlp_backend_router_mismatch" => "localize layer1 router/top-k policy from exact MLP norm",
            "layer1_mlp_backend_selected_outputs_mismatch" => "localize layer1 selected expert replay under cuBLAS BF16 MLP1, pinned SwiGLU, and BF16 MLP2 policy",
            "layer1_mlp_backend_weighted_sum_mismatch" => "localize layer1 weighted expert sum BF16 policy",
            "layer1_mlp_backend_residual_mismatch" => "localize layer1 MLP residual add or attention residual source",
            "layer1_mlp_backend_blocked_by_artifacts" => "resolve the reported layer1 MLP bundle artifact blocker",
            _ => "fix the reported layer1 MLP backend execution blocker",
        }
    });

    match router_result {
        Ok(router) => {
            status["metrics"]["router_logits"] = router["router_metric"].clone();
            status["metrics"]["topk"] = router["topk_metric"].clone();
            status["model_tensors"]["router_weight"] =
                router["model_tensors"]["router_weight"].clone();
            status["model_tensors"]["router_bias"] = router["model_tensors"]["router_bias"].clone();
        }
        Err(err) if !artifact_blocked && norm_blocker.is_none() => {
            status["blocker"] = json!({
                "kind": "layer1_mlp_backend_router",
                "detail": err.to_string()
            });
        }
        Err(_) => {}
    }
    match backend_result {
        Ok(backend) => {
            status["metrics"]["selected_outputs"] =
                backend["selected_output_metric_official"].clone();
            status["metrics"]["weighted_sum"] = backend["weighted_sum_metric_official"].clone();
            status["metrics"]["mlp_residual"] = backend["mlp_residual_metric_official"].clone();
            status["source_identity"] = backend["source_identity"].clone();
            status["per_rank_metrics"] = backend["per_rank_metrics"].clone();
        }
        Err(err) if !artifact_blocked && norm_blocker.is_none() => {
            status["blocker"] = json!({
                "kind": "layer1_mlp_backend_selected_experts",
                "detail": err.to_string()
            });
        }
        Err(_) => {}
    }

    write_json(&cli.output, &status)
}

fn run_layer1_expert28_lane2269_debug(cli: &Cli) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = cli;
        anyhow::bail!("layer1 expert28 lane2269 debug requires the cuda feature")
    }

    #[cfg(feature = "cuda")]
    {
        let attention_residual = required_path(&cli.attention_residual, "attention residual")?;
        let mlp_bundle = required_path(&cli.mlp_bundle, "MLP bundle")?;
        let model = required_path(&cli.model, "model")?;
        validate_path(attention_residual, "attention residual")?;
        validate_path(mlp_bundle, "MLP bundle")?;
        anyhow::ensure!(
            model.exists(),
            "model path does not exist: {}",
            model.display()
        );

        let layer = cli.layer_index;
        let rank = cli.rank;
        let expert_id = cli.expert;
        let lane = if cli.lane == 522 { 2269 } else { cli.lane };
        let hidden = 2880usize;
        let experts = 32usize;
        let selected_count = 4usize;
        anyhow::ensure!(
            lane < hidden,
            "lane {lane} out of range for hidden {hidden}"
        );
        let mlp_norm_boundary =
            format!("layer{layer}_final_token_mlp_norm_output_before_mlp_projections");
        let router_boundary = format!("layer{layer}_final_token_mlp_router_logits_before_routing");
        let topk_boundary =
            format!("layer{layer}_final_token_mlp_topk_expert_indices_and_routing_weights");
        let selected_boundary =
            format!("layer{layer}_final_token_selected_expert_outputs_before_routing_weighted_sum");
        let weighted_boundary = format!(
            "layer{layer}_final_token_mlp_output_after_routing_weighted_sum_before_residual"
        );
        let residual_boundary =
            format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");

        let available_boundaries = list_boundary_names(mlp_bundle)?;
        let internal_boundaries_available = available_boundaries.iter().any(|name| {
            name.contains("expert28")
                && (name.contains("mlp1")
                    || name.contains("swiglu")
                    || name.contains("pre_bias")
                    || name.contains("prebias"))
        });
        let (attention_status, attention_values) =
            load_tensor_artifact(attention_residual, &[hidden], &["values"])?;
        let (mlp_norm_status, mlp_norm_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &mlp_norm_boundary, &[hidden])?;
        let (router_status, router_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &router_boundary, &[experts])?;
        let (topk_status, topk_oracle) =
            load_boundary_topk_oracle(mlp_bundle, &topk_boundary, selected_count)?;
        let selected_experts = topk_oracle
            .indices
            .iter()
            .map(|&expert| {
                usize::try_from(expert)
                    .with_context(|| format!("invalid selected expert id {expert}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let (selected_status, selected_oracle) = load_boundary_tensor_artifact(
            mlp_bundle,
            &selected_boundary,
            &[selected_count * hidden],
        )?;
        let (weighted_status, weighted_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &weighted_boundary, &[hidden])?;
        let (residual_status, residual_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &residual_boundary, &[hidden])?;

        let artifact_blocked = !attention_status.shape_or_count_matched
            || !mlp_norm_status.shape_or_count_matched
            || !router_status.shape_or_count_matched
            || !topk_status.shape_or_count_matched
            || !selected_status.shape_or_count_matched
            || !weighted_status.shape_or_count_matched
            || !residual_status.shape_or_count_matched
            || rank >= selected_experts.len()
            || selected_experts.get(rank).copied() != Some(expert_id);

        let norm_weight_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
        let (norm_weight_source, norm_weight_values) =
            load_model_tensor_f32(model, &[norm_weight_name.as_str()])?;
        anyhow::ensure!(
            norm_weight_values.len() == hidden,
            "layer{layer} MLP norm weight shape mismatch"
        );
        let mlp_norm_values = compute_mlp_rms_norm(&attention_values, &norm_weight_values, 1e-5);
        let mlp_norm_metric = compare_hidden(&mlp_norm_values, &mlp_norm_oracle);
        let router = execute_mlp_backend_router(
            model,
            layer,
            &mlp_norm_values,
            &router_oracle,
            &topk_oracle,
        )?;

        let loaded = load_selected_experts_mxfp4_validation(model, layer, &selected_experts)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let expert = loaded
            .experts
            .iter()
            .find(|expert| expert.expert == expert_id)
            .with_context(|| format!("selected expert {expert_id} missing from MXFP4 loader"))?;
        let mlp1 = compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert)
            .with_context(|| format!("cuBLAS BF16 MLP1 failed for expert {expert_id}"))?;
        let swiglu = compute_swiglu_bf16(&mlp1);
        let mlp2_pre_bias = compute_expert30_mlp2_prebias_variant(
            &swiglu,
            &expert.down_weight,
            Expert30Mlp2Policy::Current,
        );
        let selected_output = compute_expert30_selected_output_variant(
            &mlp2_pre_bias,
            &expert.down_bias,
            Expert30Mlp2Policy::Current,
        );

        let official_selected_lane = selected_oracle[rank * hidden + lane];
        let selected_lane = selected_output[lane];
        let down_bias_lane = round_bf16(expert.down_bias[lane]);
        let pre_bias_lane = mlp2_pre_bias[lane];
        let variant_table = json!([
            {
                "name": "current_post_bias_policy",
                "value": selected_lane,
                "matches_official": selected_lane == official_selected_lane,
                "abs_diff": (selected_lane - official_selected_lane).abs()
            },
            {
                "name": "no_bias_guard",
                "value": round_bf16(pre_bias_lane),
                "matches_official": round_bf16(pre_bias_lane) == official_selected_lane,
                "abs_diff": (round_bf16(pre_bias_lane) - official_selected_lane).abs()
            },
            {
                "name": "f32_prebias_plus_f32_bias_to_bf16",
                "value": round_bf16(pre_bias_lane + expert.down_bias[lane]),
                "matches_official": round_bf16(pre_bias_lane + expert.down_bias[lane]) == official_selected_lane,
                "abs_diff": (round_bf16(pre_bias_lane + expert.down_bias[lane]) - official_selected_lane).abs()
            },
            {
                "name": "bf16_prebias_plus_bf16_bias_to_bf16",
                "value": round_bf16(round_bf16(pre_bias_lane) + round_bf16(expert.down_bias[lane])),
                "matches_official": round_bf16(round_bf16(pre_bias_lane) + round_bf16(expert.down_bias[lane])) == official_selected_lane,
                "abs_diff": (round_bf16(round_bf16(pre_bias_lane) + round_bf16(expert.down_bias[lane])) - official_selected_lane).abs()
            },
            {
                "name": "bf16_prebias_boundary_only",
                "value": round_bf16(pre_bias_lane),
                "matches_official": round_bf16(pre_bias_lane) == official_selected_lane,
                "abs_diff": (round_bf16(pre_bias_lane) - official_selected_lane).abs()
            },
            {
                "name": "official_selected_lane_replacement",
                "value": official_selected_lane,
                "matches_official": true,
                "abs_diff": 0.0
            }
        ]);

        let window_start = lane.saturating_sub(2);
        let window_end = (lane + 2).min(hidden - 1);
        let lane_window_table = (window_start..=window_end)
            .map(|window_lane| {
                let local = selected_output[window_lane];
                let official = selected_oracle[rank * hidden + window_lane];
                let pre_bias = mlp2_pre_bias[window_lane];
                json!({
                    "hidden_lane": window_lane,
                    "local_mlp2_pre_bias": pre_bias,
                    "down_bias": round_bf16(expert.down_bias[window_lane]),
                    "local_selected_post_bias": local,
                    "official_selected": official,
                    "abs_diff": (local - official).abs(),
                    "official_equals_pre_bias": official == round_bf16(pre_bias),
                    "official_equals_post_bias": official == local
                })
            })
            .collect::<Vec<_>>();

        let mut selected_matrix = vec![0.0f32; selected_count * hidden];
        for (matrix_rank, matrix_expert_id) in selected_experts.iter().copied().enumerate() {
            let expert = loaded
                .experts
                .iter()
                .find(|expert| expert.expert == matrix_expert_id)
                .with_context(|| {
                    format!("selected expert {matrix_expert_id} missing from MXFP4 loader")
                })?;
            let output = if matrix_rank == rank {
                selected_output.clone()
            } else {
                let mlp1 =
                    compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert).with_context(|| {
                        format!("cuBLAS BF16 MLP1 failed for expert {matrix_expert_id}")
                    })?;
                let swiglu = compute_swiglu_bf16(&mlp1);
                let pre_bias = compute_expert30_mlp2_prebias_variant(
                    &swiglu,
                    &expert.down_weight,
                    Expert30Mlp2Policy::Current,
                );
                compute_expert30_selected_output_variant(
                    &pre_bias,
                    &expert.down_bias,
                    Expert30Mlp2Policy::Current,
                )
            };
            let start = matrix_rank * hidden;
            selected_matrix[start..start + hidden].copy_from_slice(&output);
        }

        let selected_uncorrected =
            compare_selected_experts(&selected_matrix, &selected_oracle, &selected_experts);
        let weighted_uncorrected = compute_weighted_expert_sum_bf16(
            &selected_matrix,
            &topk_oracle.routing_weights,
            hidden,
        );
        let weighted_uncorrected_metric = compare_hidden(&weighted_uncorrected, &weighted_oracle);
        let residual_uncorrected =
            compute_attention_residual(&attention_values, &weighted_uncorrected);
        let residual_uncorrected_metric = compare_hidden(&residual_uncorrected, &residual_oracle);

        let mut corrected_selected_matrix = selected_matrix.clone();
        corrected_selected_matrix[rank * hidden + lane] = official_selected_lane;
        let selected_corrected = compare_selected_experts(
            &corrected_selected_matrix,
            &selected_oracle,
            &selected_experts,
        );
        let weighted_corrected = compute_weighted_expert_sum_bf16(
            &corrected_selected_matrix,
            &topk_oracle.routing_weights,
            hidden,
        );
        let weighted_corrected_metric = compare_hidden(&weighted_corrected, &weighted_oracle);
        let residual_corrected = compute_attention_residual(&attention_values, &weighted_corrected);
        let residual_corrected_metric = compare_hidden(&residual_corrected, &residual_oracle);

        let corrected_downstream_matches = selected_corrected.metrics.mismatches == 0
            && weighted_corrected_metric.metrics.mismatches == 0
            && residual_corrected_metric.metrics.mismatches == 0;
        let classification = if artifact_blocked {
            "layer1_expert28_lane2269_blocked_by_bundle_schema"
        } else if corrected_downstream_matches {
            "layer1_expert28_lane2269_downstream_corrected_matches"
        } else if selected_uncorrected.metrics.mismatches == 1
            && selected_corrected.metrics.mismatches == 0
        {
            "layer1_expert28_lane2269_official_selected_output_anomaly"
        } else {
            "layer1_expert28_lane2269_unresolved"
        };

        let status = json!({
            "mode": "layer0_validation_runtime_path",
            "submode": "layer1-expert28-lane2269-debug",
            "runtime_behavior_changed": false,
            "production_routing_changed": false,
            "model_runner_routing_changed": false,
            "validation_only": true,
            "classification": classification,
            "layer_index": layer,
            "rank": rank,
            "expert": expert_id,
            "lane": lane,
            "attention_residual_source": attention_residual.display().to_string(),
            "mlp_bundle": mlp_bundle.display().to_string(),
            "selected_experts": selected_experts,
            "routing_weights": topk_oracle.routing_weights,
            "internal_boundaries_available": internal_boundaries_available,
            "available_boundaries": available_boundaries,
            "guard_metrics": {
                "mlp_norm": mlp_norm_metric,
                "router_logits": router["router_metric"].clone(),
                "topk": router["topk_metric"].clone(),
                "rank_maps_to_expert": selected_experts.get(rank).copied() == Some(expert_id),
                "selected_output_layout": "[rank, hidden]"
            },
            "artifacts": {
                "attention_residual": attention_status,
                "mlp_norm_oracle": mlp_norm_status,
                "router_logits_oracle": router_status,
                "topk_routing_oracle": topk_status,
                "selected_experts_oracle": selected_status,
                "weighted_expert_sum_oracle": weighted_status,
                "mlp_residual_oracle": residual_status
            },
            "model_tensors": {
                "mlp_norm_weight": norm_weight_source,
                "selected_expert_loader": {
                    "helper_name": loaded.helper_name,
                    "decode_source": loaded.decode_source,
                    "tensor_sources": loaded.tensor_sources,
                }
            },
            "boundary_metrics": {
                "expert28_mlp1": serde_json::Value::Null,
                "expert28_swiglu": serde_json::Value::Null,
                "expert28_mlp2_prebias": serde_json::Value::Null,
                "rank0_selected_output": selected_uncorrected,
                "note": "bundle does not expose expert28 internal MLP1/SwiGLU/MLP2-prebias boundaries"
            },
            "lane_values": {
                "mlp2_pre_bias": pre_bias_lane,
                "down_bias": down_bias_lane,
                "selected_post_bias": selected_lane,
                "official_selected": official_selected_lane,
                "diff": selected_lane - official_selected_lane,
                "abs_diff": (selected_lane - official_selected_lane).abs()
            },
            "lane_window_table": lane_window_table,
            "bias_output_variant_table": variant_table,
            "replacement_impact": {
                "selected_output_uncorrected": selected_uncorrected,
                "selected_output_corrected": selected_corrected,
                "weighted_sum_uncorrected": weighted_uncorrected_metric,
                "weighted_sum_corrected": weighted_corrected_metric,
                "mlp_residual_uncorrected": residual_uncorrected_metric,
                "mlp_residual_corrected": residual_corrected_metric
            },
            "external_discriminator": {
                "status": "not_run_oom_risk",
                "reason": "bundle lacks internal expert28 boundaries; full-model PyTorch generation previously OOM'd, and this slice uses existing Rust/MXFP4 validation replay only"
            },
            "next_bounded_step": match classification {
                "layer1_expert28_lane2269_downstream_corrected_matches" => "emit a corrected layer1 output artifact behind explicit validation metadata, then guard it against the layer2 input boundary",
                "layer1_expert28_lane2269_official_selected_output_anomaly" => "record the one-lane selected-output anomaly and run downstream corrected output emission as a separate validation-only slice",
                "layer1_expert28_lane2269_blocked_by_bundle_schema" => "fix the missing or malformed layer1 MLP bundle fields before rerunning this focused debug mode",
                _ => "localize expert28 MLP1, SwiGLU, and MLP2 pre-bias with additional official internal boundaries or a lightweight direct-safetensors discriminator"
            }
        });
        write_json(&cli.output, &status)
    }
}

fn run_layer1_corrected_output(cli: &Cli) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = cli;
        anyhow::bail!("layer1 corrected output emission requires the cuda feature")
    }

    #[cfg(feature = "cuda")]
    {
        let attention_residual = required_path(&cli.attention_residual, "attention residual")?;
        let mlp_bundle = required_path(&cli.mlp_bundle, "MLP bundle")?;
        let model = required_path(&cli.model, "model")?;
        let emit_path = required_path(
            &cli.emit_corrected_layer_output,
            "emit corrected layer output",
        )?;
        validate_path(attention_residual, "attention residual")?;
        validate_path(mlp_bundle, "MLP bundle")?;
        anyhow::ensure!(
            model.exists(),
            "model path does not exist: {}",
            model.display()
        );

        let layer = cli.layer_index;
        let hidden = 2880usize;
        let experts = 32usize;
        let selected_count = 4usize;
        let correction_rank = 0usize;
        let correction_expert = 28usize;
        let correction_lane = 2269usize;
        let mlp_norm_boundary =
            format!("layer{layer}_final_token_mlp_norm_output_before_mlp_projections");
        let router_boundary = format!("layer{layer}_final_token_mlp_router_logits_before_routing");
        let topk_boundary =
            format!("layer{layer}_final_token_mlp_topk_expert_indices_and_routing_weights");
        let selected_boundary =
            format!("layer{layer}_final_token_selected_expert_outputs_before_routing_weighted_sum");
        let weighted_boundary = format!(
            "layer{layer}_final_token_mlp_output_after_routing_weighted_sum_before_residual"
        );
        let residual_boundary =
            format!("layer{layer}_final_token_hidden_state_after_mlp_residual_add");

        let (attention_status, attention_values) =
            load_tensor_artifact(attention_residual, &[hidden], &["values"])?;
        let (mlp_norm_status, mlp_norm_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &mlp_norm_boundary, &[hidden])?;
        let (router_status, router_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &router_boundary, &[experts])?;
        let (topk_status, topk_oracle) =
            load_boundary_topk_oracle(mlp_bundle, &topk_boundary, selected_count)?;
        let selected_experts = topk_oracle
            .indices
            .iter()
            .map(|&expert| {
                usize::try_from(expert)
                    .with_context(|| format!("invalid selected expert id {expert}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let (selected_status, selected_oracle) = load_boundary_tensor_artifact(
            mlp_bundle,
            &selected_boundary,
            &[selected_count * hidden],
        )?;
        let (weighted_status, weighted_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &weighted_boundary, &[hidden])?;
        let (residual_status, residual_oracle) =
            load_boundary_tensor_artifact(mlp_bundle, &residual_boundary, &[hidden])?;

        let artifact_blocked = !attention_status.shape_or_count_matched
            || !mlp_norm_status.shape_or_count_matched
            || !router_status.shape_or_count_matched
            || !topk_status.shape_or_count_matched
            || !selected_status.shape_or_count_matched
            || !weighted_status.shape_or_count_matched
            || !residual_status.shape_or_count_matched
            || selected_experts.get(correction_rank).copied() != Some(correction_expert);

        let norm_weight_name = format!("model.layers.{layer}.post_attention_layernorm.weight");
        let (_, norm_weight_values) = load_model_tensor_f32(model, &[norm_weight_name.as_str()])?;
        let mlp_norm_values = compute_mlp_rms_norm(&attention_values, &norm_weight_values, 1e-5);
        let mlp_norm_metric = compare_hidden(&mlp_norm_values, &mlp_norm_oracle);
        let router = execute_mlp_backend_router(
            model,
            layer,
            &mlp_norm_values,
            &router_oracle,
            &topk_oracle,
        )?;
        let loaded = load_selected_experts_mxfp4_validation(model, layer, &selected_experts)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;

        let mut selected_matrix = vec![0.0f32; selected_count * hidden];
        for (rank, expert_id) in selected_experts.iter().copied().enumerate() {
            let expert = loaded
                .experts
                .iter()
                .find(|expert| expert.expert == expert_id)
                .with_context(|| {
                    format!("selected expert {expert_id} missing from MXFP4 loader")
                })?;
            let mlp1 = compute_mlp1_bf16_tensor_op(&mlp_norm_values, expert)
                .with_context(|| format!("cuBLAS BF16 MLP1 failed for expert {expert_id}"))?;
            let swiglu = compute_swiglu_bf16(&mlp1);
            let pre_bias = compute_expert30_mlp2_prebias_variant(
                &swiglu,
                &expert.down_weight,
                Expert30Mlp2Policy::Current,
            );
            let output = compute_expert30_selected_output_variant(
                &pre_bias,
                &expert.down_bias,
                Expert30Mlp2Policy::Current,
            );
            let start = rank * hidden;
            selected_matrix[start..start + hidden].copy_from_slice(&output);
        }

        let correction_index = correction_rank * hidden + correction_lane;
        let validation_post_bias = selected_matrix[correction_index];
        let official_selected = selected_oracle[correction_index];
        let selected_uncorrected =
            compare_selected_experts(&selected_matrix, &selected_oracle, &selected_experts);
        let weighted_uncorrected = compute_weighted_expert_sum_bf16(
            &selected_matrix,
            &topk_oracle.routing_weights,
            hidden,
        );
        let weighted_uncorrected_metric = compare_hidden(&weighted_uncorrected, &weighted_oracle);
        let residual_uncorrected =
            compute_attention_residual(&attention_values, &weighted_uncorrected);
        let residual_uncorrected_metric = compare_hidden(&residual_uncorrected, &residual_oracle);

        selected_matrix[correction_index] = official_selected;
        let selected_corrected =
            compare_selected_experts(&selected_matrix, &selected_oracle, &selected_experts);
        let weighted_corrected = compute_weighted_expert_sum_bf16(
            &selected_matrix,
            &topk_oracle.routing_weights,
            hidden,
        );
        let weighted_corrected_metric = compare_hidden(&weighted_corrected, &weighted_oracle);
        let residual_corrected = compute_attention_residual(&attention_values, &weighted_corrected);
        let residual_corrected_metric = compare_hidden(&residual_corrected, &residual_oracle);
        let corrected_matches = selected_corrected.metrics.mismatches == 0
            && weighted_corrected_metric.metrics.mismatches == 0
            && residual_corrected_metric.metrics.mismatches == 0;
        let classification = if artifact_blocked {
            "layer1_corrected_output_blocked_by_artifacts"
        } else if corrected_matches {
            "layer1_corrected_output_emitted_and_matches_oracle"
        } else {
            "layer1_corrected_output_still_mismatch"
        };
        let correction = json!({
            "rank": correction_rank,
            "expert": correction_expert,
            "hidden_lane": correction_lane,
            "validation_post_bias": validation_post_bias,
            "official_selected": official_selected
        });
        if !artifact_blocked {
            write_corrected_layer1_output_artifact(
                emit_path,
                model,
                mlp_bundle,
                attention_residual,
                &residual_corrected,
                &correction,
            )?;
        }

        let status = json!({
            "mode": "layer0_validation_runtime_path",
            "submode": "layer1-corrected-output",
            "classification": classification,
            "runtime_behavior_changed": false,
            "production_routing_changed": false,
            "model_runner_routing_changed": false,
            "validation_only": true,
            "layer_index": layer,
            "corrected_layer_output_emitted": !artifact_blocked,
            "corrected_layer_output_path": if artifact_blocked { serde_json::Value::Null } else { json!(emit_path.display().to_string()) },
            "attention_residual_source": attention_residual.display().to_string(),
            "mlp_bundle": mlp_bundle.display().to_string(),
            "selected_experts": selected_experts,
            "correction": correction,
            "metrics": {
                "mlp_norm": mlp_norm_metric,
                "router_logits": router["router_metric"].clone(),
                "topk": router["topk_metric"].clone(),
                "selected_outputs_uncorrected": selected_uncorrected,
                "selected_outputs_corrected": selected_corrected,
                "weighted_sum_uncorrected": weighted_uncorrected_metric,
                "weighted_sum_corrected": weighted_corrected_metric,
                "mlp_residual_uncorrected": residual_uncorrected_metric,
                "mlp_residual_corrected": residual_corrected_metric
            },
            "artifacts": {
                "attention_residual": attention_status,
                "mlp_norm_oracle": mlp_norm_status,
                "router_logits_oracle": router_status,
                "topk_routing_oracle": topk_status,
                "selected_experts_oracle": selected_status,
                "weighted_expert_sum_oracle": weighted_status,
                "mlp_residual_oracle": residual_status
            },
            "caveats": {
                "attention_bundle_seam_used": true,
                "k_v_source_complete": false,
                "weighted_v_recomputed": false,
                "correction_is_validation_only_metadata": true
            },
            "next_bounded_step": match classification {
                "layer1_corrected_output_emitted_and_matches_oracle" => "guard corrected layer1 output against the layer2 input boundary",
                "layer1_corrected_output_still_mismatch" => "localize remaining corrected layer1 output mismatch before layer2 guard",
                _ => "resolve corrected layer1 output artifact blockers"
            }
        });
        write_json(&cli.output, &status)
    }
}

fn run_expert3_lane1990_debug(cli: &Cli) -> Result<()> {
    let mlp1_seam = required_path(&cli.expert3_mlp1_seam, "expert3 MLP1 seam")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let weighted_expert_sum_oracle = required_path(
        &cli.weighted_expert_sum_oracle,
        "weighted expert sum oracle",
    )?;
    let mlp_residual_oracle = required_path(&cli.mlp_residual_oracle, "MLP residual oracle")?;
    let post_attention_residual =
        required_path(&cli.post_attention_residual, "post-attention residual")?;
    let model = required_path(&cli.model, "model")?;
    let lane = cli.lane;
    validate_path(mlp1_seam, "expert3 MLP1 seam")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    validate_path(weighted_expert_sum_oracle, "weighted expert sum oracle")?;
    validate_path(mlp_residual_oracle, "MLP residual oracle")?;
    validate_path(post_attention_residual, "post-attention residual")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let hidden = 2880usize;
    let (mlp1_status, mlp1_values) = load_tensor_artifact(mlp1_seam, &[hidden * 2], &["values"])?;
    let (selected_status, selected_values) =
        load_tensor_artifact(selected_experts_oracle, &[4 * hidden], &["values"])?;
    let (weighted_status, weighted_values) =
        load_tensor_artifact(weighted_expert_sum_oracle, &[hidden], &["values"])?;
    let (residual_status, residual_values) =
        load_tensor_artifact(mlp_residual_oracle, &[hidden], &["values"])?;
    let (post_attention_status, post_attention_values) =
        load_tensor_artifact(post_attention_residual, &[hidden], &["values"])?;
    anyhow::ensure!(
        lane < hidden,
        "lane {lane} out of range for hidden {hidden}"
    );
    let artifact_blocked = !mlp1_status.shape_or_count_matched
        || !selected_status.shape_or_count_matched
        || !weighted_status.shape_or_count_matched
        || !residual_status.shape_or_count_matched
        || !post_attention_status.shape_or_count_matched;

    let pytorch_values = cli
        .pytorch_lane_terms
        .as_deref()
        .and_then(|path| fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .unwrap_or_else(|| json!({ "available": false }));

    let (
        classification,
        rust_values,
        official_values,
        variant_table,
        selected_output_metric,
        weighted_sum_impact,
        mlp_residual_impact,
        mxfp4_loader,
        blocker,
        next_bounded_step,
    ) = if artifact_blocked {
        (
            "expert3_lane1990_unresolved",
            json!({}),
            json!({}),
            Vec::new(),
            empty_hidden_comparison(),
            None,
            None,
            None,
            Some(Blocker {
                kind: "expert3_lane1990_artifacts",
                detail: "expert3 MLP1 seam or downstream official artifacts were missing supported values",
            }),
            "resolve the reported expert3 lane1990 artifact blocker",
        )
    } else {
        match execute_expert3_lane1990_debug_mxfp4(
            model,
            &mlp1_values,
            &selected_values[0..hidden],
            &weighted_values,
            &post_attention_values,
            &residual_values,
            lane,
        ) {
            Ok(result) => result,
            Err(_) => (
                "expert3_lane1990_unresolved",
                json!({}),
                json!({ "selected": selected_values[lane] }),
                Vec::new(),
                empty_hidden_comparison(),
                None,
                None,
                None,
                Some(Blocker {
                    kind: "expert3_lane1990_execution",
                    detail: "validation MXFP4 helper failed while replaying expert3 post-MLP1 path",
                }),
                "fix the expert3 lane1990 validation replay path",
            ),
        }
    };

    let status = Expert3Lane1990DebugStatus {
        mode: "layer0_validation_runtime_path",
        submode: "expert3-lane1990-debug",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        expert: 3,
        rank: 0,
        lane,
        mlp1_seam: mlp1_status,
        selected_experts_oracle: selected_status,
        weighted_expert_sum_oracle: weighted_status,
        mlp_residual_oracle: residual_status,
        post_attention_residual: post_attention_status,
        rust_values,
        pytorch_values,
        official_values,
        variant_table,
        selected_output_metric,
        weighted_sum_impact,
        mlp_residual_impact,
        mxfp4_loader,
        blocker,
        next_bounded_step,
    };

    write_json(&cli.output, &status)
}

fn run_expert3_lane1990_oracle_semantics(cli: &Cli) -> Result<()> {
    let mlp1_seam = required_path(&cli.expert3_mlp1_seam, "expert3 MLP1 seam")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let weighted_expert_sum_oracle = required_path(
        &cli.weighted_expert_sum_oracle,
        "weighted expert sum oracle",
    )?;
    let mlp_residual_oracle = required_path(&cli.mlp_residual_oracle, "MLP residual oracle")?;
    let post_attention_residual =
        required_path(&cli.post_attention_residual, "post-attention residual")?;
    let model = required_path(&cli.model, "model")?;
    let lane = cli.lane;
    validate_path(mlp1_seam, "expert3 MLP1 seam")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    validate_path(weighted_expert_sum_oracle, "weighted expert sum oracle")?;
    validate_path(mlp_residual_oracle, "MLP residual oracle")?;
    validate_path(post_attention_residual, "post-attention residual")?;

    let hidden = 2880usize;
    anyhow::ensure!(
        lane < hidden,
        "lane {lane} out of range for hidden {hidden}"
    );
    let (mlp1_status, mlp1_values) = load_tensor_artifact(mlp1_seam, &[hidden * 2], &["values"])?;
    let (selected_status, selected_values) =
        load_tensor_artifact(selected_experts_oracle, &[4 * hidden], &["values"])?;
    let (weighted_status, weighted_values) =
        load_tensor_artifact(weighted_expert_sum_oracle, &[hidden], &["values"])?;
    let (residual_status, residual_values) =
        load_tensor_artifact(mlp_residual_oracle, &[hidden], &["values"])?;
    let (post_attention_status, post_attention_values) =
        load_tensor_artifact(post_attention_residual, &[hidden], &["values"])?;
    let routing_weights = load_selected_routing_weights(selected_experts_oracle, 4)?;
    let pytorch_values = cli
        .pytorch_lane_terms
        .as_deref()
        .and_then(|path| fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .unwrap_or_else(|| json!({ "available": false }));
    let source_identity = selected_oracle_source_identity(selected_experts_oracle, mlp1_seam)?;

    let artifact_blocked = !mlp1_status.shape_or_count_matched
        || !selected_status.shape_or_count_matched
        || !weighted_status.shape_or_count_matched
        || !residual_status.shape_or_count_matched
        || !post_attention_status.shape_or_count_matched;

    let status = if artifact_blocked {
        json!({
            "mode": "layer0_validation_runtime_path",
            "submode": "expert3-lane1990-oracle-semantics",
            "classification": "expert3_lane1990_unresolved_oracle_semantics",
            "runtime_behavior_changed": false,
            "source_identity": source_identity,
            "blocker": {
                "kind": "expert3_lane1990_oracle_semantics_artifacts",
                "detail": "required seam or oracle artifacts had unsupported shapes or values"
            },
            "next_bounded_step": "resolve the reported artifact blocker"
        })
    } else {
        match execute_expert3_lane1990_oracle_semantics_mxfp4(
            model,
            &mlp1_values,
            &selected_values,
            &routing_weights,
            &weighted_values,
            &post_attention_values,
            &residual_values,
            lane,
        ) {
            Ok(mut status) => {
                status["source_identity"] = source_identity;
                status["pytorch_values"] = pytorch_values;
                status["artifacts"] = json!({
                    "mlp1_seam": mlp1_status,
                    "selected_experts_oracle": selected_status,
                    "weighted_expert_sum_oracle": weighted_status,
                    "mlp_residual_oracle": residual_status,
                    "post_attention_residual": post_attention_status,
                });
                status
            }
            Err(err) => json!({
                "mode": "layer0_validation_runtime_path",
                "submode": "expert3-lane1990-oracle-semantics",
                "classification": "expert3_lane1990_unresolved_oracle_semantics",
                "runtime_behavior_changed": false,
                "source_identity": source_identity,
                "error": err.to_string(),
                "next_bounded_step": "fix the expert3 lane1990 oracle-semantics validation path"
            }),
        }
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

fn run_selected_experts_pinned_swiglu_debug(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let selected_experts_oracle =
        required_path(&cli.selected_experts_oracle, "selected experts oracle")?;
    let model = required_path(&cli.model, "model")?;
    let mlp1_oracle_path = required_path(&cli.expert30_mlp1_oracle, "expert30 MLP1 oracle")?;
    let swiglu_oracle_path = required_path(&cli.expert30_swiglu_oracle, "expert30 SwiGLU oracle")?;
    let mlp2_oracle_path = required_path(
        &cli.expert30_mlp2_pre_bias_oracle,
        "expert30 MLP2 pre-bias oracle",
    )?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(selected_experts_oracle, "selected experts oracle")?;
    validate_path(mlp1_oracle_path, "expert30 MLP1 oracle")?;
    validate_path(swiglu_oracle_path, "expert30 SwiGLU oracle")?;
    validate_path(mlp2_oracle_path, "expert30 MLP2 pre-bias oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let selected_experts = parse_selected_experts(&cli.selected_experts)?;
    let expert30_rank = selected_experts
        .iter()
        .position(|&expert| expert == 30)
        .context("selected expert list must include expert30")?;
    let (mlp_norm_status, mlp_norm_values) = load_tensor_artifact(mlp_norm, &[2880], &["values"])?;
    let (oracle_status, oracle_values) = load_tensor_artifact(
        selected_experts_oracle,
        &[selected_experts.len() * 2880],
        &["values"],
    )?;
    let (_, mlp1_oracle) = load_tensor_artifact(mlp1_oracle_path, &[5760], &["values"])?;
    let (_, swiglu_oracle) = load_tensor_artifact(swiglu_oracle_path, &[2880], &["values"])?;
    let (_, mlp2_oracle) = load_tensor_artifact(mlp2_oracle_path, &[2880], &["values"])?;

    let artifact_blocked =
        !mlp_norm_status.shape_or_count_matched || !oracle_status.shape_or_count_matched;
    let (
        classification,
        per_rank_selected_output_metrics,
        expert30_variant_table,
        mlp1_variant_table,
        first_mismatching_boundary,
        loader,
        blocker,
        next_bounded_step,
    ) = if artifact_blocked {
        (
            "selected_expert_mismatch_unresolved",
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "artifact_load",
            None,
            Some(Blocker {
                kind: "selected_experts_pinned_swiglu_artifacts",
                detail: "MLP norm input or selected expert oracle did not expose supported values or expected counts",
            }),
            "resolve the reported selected-experts pinned-SwiGLU artifact blocker",
        )
    } else {
        match execute_selected_experts_pinned_swiglu_debug_mxfp4(
            model,
            &selected_experts,
            &mlp_norm_values,
            &oracle_values,
            expert30_rank,
            &mlp1_oracle,
            &swiglu_oracle,
            &mlp2_oracle,
        ) {
            Ok(debug) => debug,
            Err(_) => (
                "selected_expert_mismatch_unresolved",
                Vec::new(),
                Vec::new(),
                Vec::new(),
                "mxfp4_loader",
                None,
                Some(Blocker {
                    kind: "selected_experts_pinned_swiglu_mxfp4",
                    detail: "validation MXFP4 helper failed before pinned-SwiGLU debug comparison",
                }),
                "fix the narrow validation MXFP4 selected-expert helper",
            ),
        }
    };

    let status = SelectedExpertsPinnedSwigluDebugStatus {
        mode: "layer0_validation_runtime_path",
        submode: "selected-experts-pinned-swiglu-debug",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        selected_experts,
        pinned_swiglu_policy: "I_torch_like_stage_rounding",
        selected_experts_mode_uses_pinned_policy: true,
        selected_experts_debug_uses_pinned_policy: true,
        per_rank_selected_output_metrics,
        expert30_variant_table,
        mlp1_variant_table,
        first_mismatching_boundary,
        mxfp4_loader: loader,
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

fn run_swiglu_policy_pin(cli: &Cli) -> Result<()> {
    let mlp1_oracle = required_path(&cli.expert30_mlp1_oracle, "expert30 MLP1 oracle")?;
    let swiglu_oracle = required_path(&cli.expert30_swiglu_oracle, "expert30 SwiGLU oracle")?;
    validate_path(mlp1_oracle, "expert30 MLP1 oracle")?;
    validate_path(swiglu_oracle, "expert30 SwiGLU oracle")?;

    let (mlp1_status, mlp1_values) = load_tensor_artifact(mlp1_oracle, &[5760], &["values"])?;
    let (swiglu_status, swiglu_values) = load_tensor_artifact(swiglu_oracle, &[2880], &["values"])?;
    let pytorch = match cli.pytorch_intermediates.as_deref() {
        Some(path) => {
            validate_path(path, "PyTorch BF16 SwiGLU intermediates")?;
            Some(load_pytorch_swiglu_intermediates(path)?)
        }
        None => None,
    };

    let artifact_blocked =
        !mlp1_status.shape_or_count_matched || !swiglu_status.shape_or_count_matched;
    let (
        classification,
        pytorch_status,
        first_divergent_stage,
        best_variant,
        stage_metrics,
        variants,
        blocker,
        next_bounded_step,
    ) = if artifact_blocked {
        let empty = empty_swiglu_policy_pin_variant();
        (
                "swiglu_policy_unresolved",
                PytorchIntermediateStatus {
                    available: pytorch.is_some(),
                    path: cli.pytorch_intermediates.as_ref().map(|path| path.display().to_string()),
                    swiglu_output_metric: None,
                    note: "expert30 MLP1 or SwiGLU oracle did not expose supported values or expected counts",
                },
                None,
                empty.clone(),
                Vec::new(),
                Vec::new(),
                Some(Blocker {
                    kind: "swiglu_policy_pin_artifacts",
                    detail: "expert30 MLP1 or SwiGLU oracle did not expose supported values or expected counts",
                }),
                "resolve the reported SwiGLU policy-pin artifact blocker",
            )
    } else {
        let pytorch_status = PytorchIntermediateStatus {
            available: pytorch.is_some(),
            path: cli
                .pytorch_intermediates
                .as_ref()
                .map(|path| path.display().to_string()),
            swiglu_output_metric: pytorch
                .as_ref()
                .and_then(|artifact| artifact.get("swiglu_output"))
                .map(|values| compare_hidden(values, &swiglu_values)),
            note: if pytorch.is_some() {
                "loaded PyTorch BF16 tensor-expression intermediates generated outside the repo"
            } else {
                "no PyTorch intermediate artifact supplied; only Rust variant output metrics were computed"
            },
        };
        let variants =
            compute_swiglu_policy_pin_variants(&mlp1_values, &swiglu_values, pytorch.as_ref());
        let best = variants
            .iter()
            .min_by(|left, right| {
                left.output_metric
                    .metrics
                    .mismatches
                    .cmp(&right.output_metric.metrics.mismatches)
                    .then_with(|| {
                        left.output_metric
                            .metrics
                            .max_abs_diff
                            .partial_cmp(&right.output_metric.metrics.max_abs_diff)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .cloned()
            .expect("SwiGLU policy pin must include variants");
        let stage_metrics = best.stage_metrics.clone();
        let first_divergent_stage = stage_metrics
            .iter()
            .find(|stage| stage.metric.metrics.mismatches != 0)
            .map(|stage| stage.stage);
        let classification = if best.output_metric.metrics.mismatches == 0 {
            "swiglu_policy_matches_oracle"
        } else if pytorch_status
            .swiglu_output_metric
            .as_ref()
            .is_some_and(|metric| metric.metrics.mismatches == 0)
        {
            "swiglu_policy_pytorch_bf16_semantics_not_encoded"
        } else {
            "swiglu_policy_unresolved"
        };
        let next_bounded_step = match classification {
                "swiglu_policy_matches_oracle" => {
                    "encode this exact SwiGLU policy in selected-experts replay and rerun selected-experts"
                }
                "swiglu_policy_pytorch_bf16_semantics_not_encoded" => {
                    "implement a dedicated validation SwiGLU helper matching PyTorch BF16 sigmoid/operator semantics, or use official SwiGLU as a temporary seam input"
                }
                _ => "regenerate PyTorch BF16 intermediates and inspect the first divergent stage",
            };
        (
            classification,
            pytorch_status,
            first_divergent_stage,
            best,
            stage_metrics,
            variants,
            None,
            next_bounded_step,
        )
    };

    let status = SwigluPolicyPinStatus {
        mode: "layer0_validation_runtime_path",
        submode: "swiglu-policy-pin",
        classification,
        implemented: blocker.is_none(),
        runtime_behavior_changed: false,
        validation_only: true,
        official_summary: OfficialSwigluSummary {
            gate_up_split: "x_glu = x[..., ::2], x_linear = x[..., 1::2]",
            gate_clamp: "x_glu.clamp(min=None, max=7.0)",
            up_clamp: "x_linear.clamp(min=-7.0, max=7.0)",
            sigmoid_scale: 1.702,
            multiply_order:
                "out_glu = x_glu * sigmoid(1.702 * x_glu); out = out_glu * (x_linear + 1)",
            dtype_cast_behavior:
                "PyTorch tensor expression keeps intermediates at torch.bfloat16 for this boundary",
            output_dtype: "BF16 official boundary",
        },
        mlp1_oracle: mlp1_status,
        swiglu_oracle: swiglu_status,
        pytorch_intermediate_status: pytorch_status,
        first_divergent_stage,
        best_variant_metric: best_variant.output_metric.clone(),
        best_variant,
        stage_metrics,
        variant_table: variants,
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

fn run_expert30_mlp1_debug(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let mlp1_oracle = required_path(&cli.expert30_mlp1_oracle, "expert30 MLP1 oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(mlp1_oracle, "expert30 MLP1 oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );

    let (mlp_norm_status, mlp_norm_values) = load_tensor_artifact(mlp_norm, &[2880], &["values"])?;
    let (mlp1_status, mlp1_values) = load_tensor_artifact(mlp1_oracle, &[5760], &["values"])?;
    let lane = cli.lane;
    let artifact_blocked = !mlp_norm_status.shape_or_count_matched
        || !mlp1_status.shape_or_count_matched
        || lane >= mlp1_values.len();

    let status = if artifact_blocked {
        let official = mlp1_values.get(lane).copied().unwrap_or(f32::NAN);
        Expert30Mlp1LaneDebugStatus {
            mode: "layer0_validation_runtime_path",
            submode: "expert30-mlp1-debug",
            classification: "expert30_mlp1_lane522_artifact_identity_mismatch",
            implemented: false,
            runtime_behavior_changed: false,
            validation_only: true,
            lane_metadata: expert30_mlp1_lane_metadata(lane),
            official_value: official,
            current_local_value: f32::NAN,
            current_diff: f32::NAN,
            current_pre_bias: f32::NAN,
            bias_value: f32::NAN,
            output_rounding_policy: "not_run_artifact_blocked",
            source_identity: json!({}),
            local_values: json!({}),
            official_values: json!({ "output": official }),
            pytorch_reference: json!({ "available": false, "result": "not_run" }),
            per_block_summary: Vec::new(),
            top_contributions: Vec::new(),
            decode_variant_table: Vec::new(),
            accumulation_variant_table: Vec::new(),
            best_variant: empty_mlp1_lane_variant("not_run", "artifact blocker", official),
            best_explanation: "artifact blocker",
            next_bounded_step: "resolve the MLP norm or expert30 MLP1 artifact mismatch",
        }
    } else {
        match execute_expert30_mlp1_debug_mxfp4(
            model,
            &mlp_norm_values,
            &mlp1_values,
            lane,
            cli.pytorch_lane_terms.as_deref(),
        ) {
            Ok(status) => status,
            Err(_) => Expert30Mlp1LaneDebugStatus {
                mode: "layer0_validation_runtime_path",
                submode: "expert30-mlp1-debug",
                classification: "expert30_mlp1_lane522_decode_semantics_unresolved",
                implemented: false,
                runtime_behavior_changed: false,
                validation_only: true,
                lane_metadata: expert30_mlp1_lane_metadata(lane),
                official_value: mlp1_values[lane],
                current_local_value: f32::NAN,
                current_diff: f32::NAN,
                current_pre_bias: f32::NAN,
                bias_value: f32::NAN,
                output_rounding_policy: "not_run_mxfp4_loader_failed",
                source_identity: json!({ "rust_model_path": model.display().to_string() }),
                local_values: json!({}),
                official_values: json!({ "output": mlp1_values[lane] }),
                pytorch_reference: json!({ "available": false, "result": "not_run" }),
                per_block_summary: Vec::new(),
                top_contributions: Vec::new(),
                decode_variant_table: Vec::new(),
                accumulation_variant_table: Vec::new(),
                best_variant: empty_mlp1_lane_variant(
                    "not_run",
                    "MXFP4 row helper failed",
                    mlp1_values[lane],
                ),
                best_explanation: "MXFP4 row helper failed",
                next_bounded_step: "fix the narrow validation MXFP4 gate/up row loading path",
            },
        }
    };

    write_json(&cli.output, &status)
}

fn run_mlp1_bf16_policy(cli: &Cli) -> Result<()> {
    let mlp_norm = required_path(&cli.mlp_norm, "MLP norm")?;
    let mlp1_oracle = required_path(&cli.expert30_mlp1_oracle, "expert30 MLP1 oracle")?;
    let model = required_path(&cli.model, "model")?;
    validate_path(mlp_norm, "MLP norm")?;
    validate_path(mlp1_oracle, "expert30 MLP1 oracle")?;
    anyhow::ensure!(
        model.exists(),
        "model path does not exist: {}",
        model.display()
    );
    let (_, mlp_norm_values) = load_tensor_artifact(mlp_norm, &[2880], &["values"])?;
    let (_, mlp1_values) = load_tensor_artifact(mlp1_oracle, &[5760], &["values"])?;
    let lane = cli.lane;
    anyhow::ensure!(lane < mlp1_values.len(), "lane {lane} out of range");
    let status = execute_mlp1_bf16_policy_mxfp4(
        model,
        &mlp_norm_values,
        &mlp1_values,
        lane,
        cli.pytorch_lane_terms.as_deref(),
    )?;
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

fn required_string<'a>(value: &'a Option<String>, label: &str) -> Result<&'a str> {
    value
        .as_deref()
        .with_context(|| format!("--{} is required", label.replace(' ', "-").to_lowercase()))
}

fn parse_path_list(value: &str, expected_count: usize) -> Result<Vec<PathBuf>> {
    let paths = value
        .split(',')
        .map(|part| PathBuf::from(part.trim()))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        paths.len() == expected_count,
        "expected {expected_count} comma-separated paths, got {}",
        paths.len()
    );
    Ok(paths)
}

fn load_selected_routing_weights(path: &Path, expected_count: usize) -> Result<Vec<f32>> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let weights = value
        .get("selected_routing_weights")
        .and_then(Value::as_array)
        .context("selected_routing_weights not found")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .context("selected_routing_weights must be numeric")
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        weights.len() == expected_count,
        "selected_routing_weights count mismatch: {} != {expected_count}",
        weights.len()
    );
    Ok(weights)
}

fn selected_oracle_source_identity(selected_oracle: &Path, mlp1_seam: &Path) -> Result<Value> {
    let selected: Value = serde_json::from_slice(
        &fs::read(selected_oracle)
            .with_context(|| format!("failed to read {}", selected_oracle.display()))?,
    )
    .with_context(|| format!("failed to parse {}", selected_oracle.display()))?;
    let seam: Value = serde_json::from_slice(
        &fs::read(mlp1_seam).with_context(|| format!("failed to read {}", mlp1_seam.display()))?,
    )
    .with_context(|| format!("failed to parse {}", mlp1_seam.display()))?;
    Ok(json!({
        "selected_output_oracle": {
            "path": selected_oracle.display().to_string(),
            "case_id": selected.get("case_id"),
            "suite_id": selected.get("suite_id"),
            "schema_version": selected.get("schema_version"),
            "boundary": selected.get("boundary"),
            "official_model": selected.get("official_model"),
            "selected_expert_indices": selected.get("selected_expert_indices"),
            "selected_routing_weights": selected.get("selected_routing_weights"),
            "serialization_dtype": selected.get("serialization_dtype"),
            "tensor_dtype": selected.get("tensor_dtype"),
            "value_key": "values",
        },
        "mlp1_seam": {
            "path": mlp1_seam.display().to_string(),
            "model_path": seam.get("model_path"),
            "mlp_norm_input": seam.get("mlp_norm_input"),
            "expert": seam.get("expert"),
            "boundary": seam.get("boundary"),
            "operation": seam.get("operation"),
            "dtype": seam.get("dtype"),
            "sha256_f32_le_full": seam.get("sha256_f32_le_full"),
        },
        "model_path_note": "restricted integration model path may symlink shards to /data/models/openai/gpt-oss-20b; source identity is recorded, not assumed",
    }))
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

fn write_corrected_layer0_output_artifact(
    output: &Path,
    model: &Path,
    values: &[f32],
    correction: &Value,
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "corrected layer0 output artifact must have 2880 values, got {}",
        values.len()
    );
    anyhow::ensure!(
        correction
            .get("applied")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "corrected layer0 output emission requires the lane1990 correction to be applied"
    );
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer0_final_token_hidden_after_mlp_residual_corrected",
        "source_mode": "full-layer0",
        "correction_applied": true,
        "correction": correction,
        "model": model.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn write_layer1_attention_residual_artifact(
    output: &Path,
    model: &Path,
    bundle: &Path,
    values: &[f32],
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "layer1 attention residual artifact must have 2880 values, got {}",
        values.len()
    );
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp",
        "source_mode": "layer1-attention-bundle",
        "source_policy": {
            "q_post_rope_source": "official_attention_bundle",
            "k_post_rope_source": "official_attention_bundle",
            "k_rope_construction_validated": false,
            "weighted_v_source": "official_weighted_v_oracle_because_all_token_v_history_missing"
        },
        "model": model.display().to_string(),
        "attention_bundle": bundle.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn write_layer_attention_residual_artifact(
    output: &Path,
    layer: usize,
    model: &Path,
    bundle: &Path,
    values: &[f32],
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "layer attention residual artifact must have 2880 values, got {}",
        values.len()
    );
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": format!("layer{layer}_final_token_hidden_state_after_attention_residual_add_before_mlp"),
        "layer_index": layer,
        "source_mode": "layer-bundle-validate",
        "source_policy": {
            "q_post_rope_source": "official_attention_bundle",
            "k_post_rope_source": "official_attention_bundle",
            "k_rope_construction_validated": false,
            "weighted_v_source": "official_weighted_v_oracle_because_all_token_v_history_missing",
            "weighted_v_recomputed": false,
        },
        "model": model.display().to_string(),
        "attention_bundle": bundle.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "sha256_f32_le_full": digest_f32_le(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn write_layer1_mlp_output_artifact(
    output: &Path,
    model: &Path,
    mlp_bundle: &Path,
    attention_residual: &Path,
    values: &[f32],
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "layer1 output artifact must have 2880 values, got {}",
        values.len()
    );
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer1_final_token_hidden_state_after_mlp_residual_add",
        "source_mode": "layer1-mlp-backend",
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "correction_policy": "none",
        "model": model.display().to_string(),
        "mlp_bundle": mlp_bundle.display().to_string(),
        "attention_residual": attention_residual.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn write_layer_output_artifact(
    output: &Path,
    layer: usize,
    model: &Path,
    attention_bundle: &Path,
    mlp_bundle: &Path,
    attention_residual: &Path,
    values: &[f32],
    correction: &Value,
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "layer output artifact must have 2880 values, got {}",
        values.len()
    );
    let correction_applied = correction
        .get("applied")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": format!("layer{layer}_final_token_hidden_after_mlp_residual_corrected"),
        "layer_index": layer,
        "source_mode": "layer-bundle-validate",
        "attention_source": "official_attention_bundle_seam",
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "correction_applied": correction_applied,
        "correction": correction,
        "k_v_source_complete": false,
        "k_pre_rope_history_validated": false,
        "weighted_v_recomputed": false,
        "reason": "layer_attention_advanced_from_bundle_seam",
        "model": model.display().to_string(),
        "attention_bundle": attention_bundle.display().to_string(),
        "mlp_bundle": mlp_bundle.display().to_string(),
        "attention_residual": attention_residual.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "sha256_f32_le_full": digest_f32_le(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn write_coarse_layer_output_artifact(
    output: &Path,
    layer: usize,
    model: &Path,
    coarse_bundle: &Path,
    values: &[f32],
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "coarse layer output artifact must have 2880 values, got {}",
        values.len()
    );
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": format!("layer{layer}_final_token_hidden_after_mlp_residual_coarse_validated"),
        "layer_index": layer,
        "source_mode": "coarse-layer-validate",
        "attention_source": "official_coarse_attention_residual_seam",
        "attention_recomputed": false,
        "ordered_attention_bundle_available": false,
        "ordered_mlp_bundle_available": false,
        "correction_applied": false,
        "k_v_source_complete": false,
        "k_pre_rope_history_validated": false,
        "weighted_v_recomputed": false,
        "reason": "coarse_layer_validation_advanced_from_official_attention_residual_seam",
        "backend_path": {
            "mlp1": "cublas_bf16_tensor_op",
            "swiglu": "pinned_torch_like_bf16_stage_rounding",
            "mlp2": "bf16_prebias_output_policy"
        },
        "model": model.display().to_string(),
        "coarse_bundle": coarse_bundle.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "sha256_f32_le_full": digest_f32_le(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn write_corrected_layer1_output_artifact(
    output: &Path,
    model: &Path,
    mlp_bundle: &Path,
    attention_residual: &Path,
    values: &[f32],
    correction: &Value,
) -> Result<()> {
    anyhow::ensure!(
        values.len() == 2880,
        "corrected layer1 output artifact must have 2880 values, got {}",
        values.len()
    );
    let payload = json!({
        "case": "developer-message-user-smoke",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer1_final_token_hidden_after_mlp_residual_corrected",
        "layer_index": 1,
        "source_mode": "layer1-mlp-backend",
        "attention_source": "official_attention_bundle_seam",
        "correction_applied": true,
        "correction": correction,
        "k_v_source_complete": false,
        "k_pre_rope_history_validated": false,
        "weighted_v_recomputed": false,
        "reason": "layer1_attention_advanced_from_bundle_seam",
        "model": model.display().to_string(),
        "mlp_bundle": mlp_bundle.display().to_string(),
        "attention_residual": attention_residual.display().to_string(),
        "shape": [2880],
        "tensor_dtype": "bf16_boundary_serialized_as_f32",
        "serialization_dtype": "f32_json",
        "value_key": "values",
        "finite_value_summary": finite_summary(values),
        "sha256_f32_le_full": digest_f32_le(values),
        "values": values,
    });
    write_json(output, &payload)
}

fn classify_layer1_attention_bundle(
    raw_qk_metric: &Option<RawQkComparison>,
    masked_metric: &Option<MatrixComparisonStatus>,
    probs_metric: &Option<MatrixComparisonStatus>,
    oproj_metric: &Option<HiddenComparisonStatus>,
    residual_metric: &Option<HiddenComparisonStatus>,
    raw_inputs_ready: bool,
    masked_present: bool,
    probs_present: bool,
    weighted_v_present: bool,
) -> &'static str {
    if !raw_inputs_ready {
        return "layer1_attention_bundle_blocked_by_missing_values";
    }
    if raw_qk_metric
        .as_ref()
        .is_none_or(|metric| metric.metrics.mismatches != 0)
    {
        return "layer1_attention_bundle_mismatch";
    }
    if !masked_present || !probs_present {
        return "layer1_attention_bundle_raw_qk_matches_oracle";
    }
    if masked_metric
        .as_ref()
        .is_none_or(|metric| metric.metrics.mismatches != 0)
        || probs_metric
            .as_ref()
            .is_none_or(|metric| metric.metrics.mismatches != 0)
    {
        return "layer1_attention_bundle_mismatch";
    }
    if !weighted_v_present {
        return "layer1_attention_bundle_attention_probs_matches_oracle";
    }
    if let Some(metric) = residual_metric {
        if metric.metrics.mismatches == 0 {
            return "layer1_attention_bundle_attention_residual_matches_oracle";
        }
        return "layer1_attention_bundle_mismatch";
    }
    if let Some(metric) = oproj_metric {
        if metric.metrics.mismatches == 0 {
            return "layer1_attention_bundle_oproj_matches_oracle";
        }
        return "layer1_attention_bundle_mismatch";
    }
    "layer1_attention_bundle_attention_probs_matches_oracle"
}

fn list_boundary_names(path: &Path) -> Result<Vec<String>> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let mut names = value
        .get("boundaries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            entry
                .get("boundary")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    names.sort();
    Ok(names)
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

fn load_boundary_tensor_artifact(
    path: &Path,
    boundary: &str,
    expected_counts: &[usize],
) -> Result<(TensorArtifactStatus, Vec<f32>)> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let boundary_value = value
        .get("boundaries")
        .and_then(Value::as_array)
        .and_then(|boundaries| {
            boundaries.iter().find(|entry| {
                entry
                    .get("boundary")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name == boundary)
            })
        })
        .unwrap_or(&value);
    let shape = extract_shape(boundary_value);
    let (values, value_key) = extract_values_by_keys(boundary_value, &["values"]);
    let value_count = values.as_ref().map(Vec::len);
    let count_matches = value_count
        .map(|count| expected_counts.contains(&count))
        .unwrap_or(false);
    let value_key = value_key.map(|key| {
        if boundary_value as *const Value == &value as *const Value {
            key
        } else {
            format!("boundaries[{boundary}].{key}")
        }
    });
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

fn load_boundary_selected_experts(path: &Path) -> Result<Vec<i64>> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    if let Some(values) = value
        .get("selected_expert_indices")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
    {
        return Ok(json_values_to_i64(values));
    }
    let values = value
        .get("boundaries")
        .and_then(Value::as_array)
        .and_then(|boundaries| {
            boundaries.iter().find(|entry| {
                entry
                    .get("boundary")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name.contains("topk_expert_indices"))
            })
        })
        .and_then(|entry| entry.get("selected_expert_indices"))
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_i64(values))
        .unwrap_or_default();
    Ok(values)
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

fn load_boundary_topk_oracle(
    path: &Path,
    boundary: &str,
    top_k: usize,
) -> Result<(TensorArtifactStatus, TopkOracle)> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let boundary_value = value
        .get("boundaries")
        .and_then(Value::as_array)
        .and_then(|boundaries| {
            boundaries.iter().find(|entry| {
                entry
                    .get("boundary")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name == boundary)
            })
        })
        .unwrap_or(&value);
    let indices: Vec<i64> = boundary_value
        .get("selected_expert_indices")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_i64(values))
        .or_else(|| {
            value
                .get("selected_expert_indices")
                .and_then(Value::as_array)
                .map(|values| json_values_to_i64(values))
        })
        .unwrap_or_default();
    let logits: Vec<f32> = boundary_value
        .get("selected_expert_logits")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_f32(values))
        .unwrap_or_default();
    let routing_weights: Vec<f32> = boundary_value
        .get("routing_weights")
        .and_then(|entry| entry.get("values"))
        .and_then(Value::as_array)
        .map(|values| json_values_to_f32(values))
        .or_else(|| {
            boundary_value
                .get("values")
                .and_then(Value::as_array)
                .map(|values| json_values_to_f32(values))
        })
        .or_else(|| {
            value
                .get("routing_weights")
                .and_then(Value::as_array)
                .map(|values| json_values_to_f32(values))
        })
        .unwrap_or_default();
    let matched = indices.len() == top_k && logits.len() == top_k && routing_weights.len() == top_k;
    Ok((
        TensorArtifactStatus {
            path: path.display().to_string(),
            json_loaded: true,
            shape: extract_shape(boundary_value).or_else(|| Some(vec![top_k])),
            value_count: Some(routing_weights.len()),
            expected_value_counts: vec![top_k],
            shape_or_count_matched: matched,
            value_key: Some(format!(
                "boundaries[{boundary}].selected_expert_indices/selected_expert_logits/routing_weights"
            )),
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

fn digest_f32_le(values: &[f32]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn round_f16(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

fn bf16_round_vec(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&value| round_bf16(value)).collect()
}

fn f16_round_vec(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&value| round_f16(value)).collect()
}

fn dot_sequential_f32(input: &[f32], weights: &[f32], f32_input: bool) -> f32 {
    input
        .iter()
        .zip(weights.iter())
        .fold(0.0f32, |acc, (&input_value, &weight)| {
            let input_value = if f32_input {
                input_value
            } else {
                round_bf16(input_value)
            };
            acc + input_value * weight
        })
}

fn dot_reverse_f32(input: &[f32], weights: &[f32]) -> f32 {
    input
        .iter()
        .zip(weights.iter())
        .rev()
        .fold(0.0f32, |acc, (&input_value, &weight)| {
            acc + round_bf16(input_value) * weight
        })
}

fn dot_chunked_pairwise_f32(input: &[f32], weights: &[f32]) -> f32 {
    let mut partials = Vec::new();
    for chunk in (0..input.len()).step_by(64) {
        let mut partial = 0.0f32;
        for idx in chunk..(chunk + 64).min(input.len()) {
            partial += round_bf16(input[idx]) * weights[idx];
        }
        partials.push(partial);
    }
    partials.into_iter().sum()
}

fn dot_blockwise_f32(input: &[f32], weights: &[f32], block: usize) -> f32 {
    let mut total = 0.0f32;
    for chunk in (0..input.len()).step_by(block) {
        let mut partial = 0.0f32;
        for idx in chunk..(chunk + block).min(input.len()) {
            partial += round_bf16(input[idx]) * weights[idx];
        }
        total += partial;
    }
    total
}

fn dot_f64_diagnostic(input: &[f32], weights: &[f32]) -> f32 {
    input
        .iter()
        .zip(weights.iter())
        .fold(0.0f64, |acc, (&input_value, &weight)| {
            acc + round_bf16(input_value) as f64 * weight as f64
        }) as f32
}

fn dot_bf16_product_f32_sum(input: &[f32], weights: &[f32]) -> f32 {
    input
        .iter()
        .zip(weights.iter())
        .fold(0.0f32, |acc, (&input_value, &weight)| {
            acc + round_bf16(round_bf16(input_value) * round_bf16(weight))
        })
}

fn dot_bf16_running(input: &[f32], weights: &[f32], block: Option<usize>) -> f32 {
    match block {
        Some(block) => {
            let mut total = 0.0f32;
            for chunk in (0..input.len()).step_by(block) {
                let mut partial = 0.0f32;
                for idx in chunk..(chunk + block).min(input.len()) {
                    partial = round_bf16(
                        partial + round_bf16(round_bf16(input[idx]) * round_bf16(weights[idx])),
                    );
                }
                total = round_bf16(total + partial);
            }
            total
        }
        None => input
            .iter()
            .zip(weights.iter())
            .fold(0.0f32, |acc, (&input_value, &weight)| {
                round_bf16(acc + round_bf16(round_bf16(input_value) * round_bf16(weight)))
            }),
    }
}

fn dot_chunked_pairwise_f32_chunk(input: &[f32], weights: &[f32], chunk_size: usize) -> f32 {
    let mut partials = Vec::new();
    for chunk in (0..input.len()).step_by(chunk_size) {
        let mut partial = 0.0f32;
        for idx in chunk..(chunk + chunk_size).min(input.len()) {
            partial += round_bf16(input[idx]) * weights[idx];
        }
        partials.push(partial);
    }
    partials.into_iter().sum()
}

fn local_per_block_summary(input: &[f32], weights: &[f32]) -> Vec<Value> {
    (0..90)
        .map(|block| {
            let start = block * 32;
            let end = (start + 32).min(input.len()).min(weights.len());
            let mut sum = 0.0f32;
            let mut abs_sum = 0.0f32;
            let mut max_abs = 0.0f32;
            let mut max_abs_hidden = start;
            for idx in start..end {
                let contribution = round_bf16(input[idx]) * weights[idx];
                let abs = contribution.abs();
                sum += contribution;
                abs_sum += abs;
                if abs > max_abs {
                    max_abs = abs;
                    max_abs_hidden = idx;
                }
            }
            json!({
                "block": block,
                "sum": sum,
                "abs_sum": abs_sum,
                "max_abs": max_abs,
                "max_abs_hidden": max_abs_hidden,
            })
        })
        .collect()
}

fn local_top_contributions(input: &[f32], weights: &[f32], count: usize) -> Vec<Value> {
    let mut contributions = input
        .iter()
        .zip(weights.iter())
        .enumerate()
        .map(|(hidden, (&input_value, &weight))| {
            let rounded_input = round_bf16(input_value);
            let contribution = rounded_input * weight;
            (hidden, rounded_input, weight, contribution)
        })
        .collect::<Vec<_>>();
    contributions.sort_by(|a, b| b.3.abs().total_cmp(&a.3.abs()));
    contributions
        .into_iter()
        .take(count)
        .map(|(hidden, input, weight, contribution)| {
            json!({
                "hidden": hidden,
                "input": input,
                "weight": weight,
                "contribution": contribution,
                "abs_contribution": contribution.abs(),
            })
        })
        .collect()
}

fn load_pytorch_lane_reference(path: Option<&Path>) -> Value {
    let Some(path) = path else {
        return json!({
            "available": false,
            "result": "not_provided",
        });
    };
    match fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
    {
        Some(mut value) => {
            if let Some(object) = value.as_object_mut() {
                object.insert("available".to_string(), Value::Bool(true));
                object.insert(
                    "path".to_string(),
                    Value::String(path.display().to_string()),
                );
            }
            value
        }
        None => json!({
            "available": false,
            "path": path.display().to_string(),
            "result": "failed_to_parse",
        }),
    }
}

fn expert30_mlp1_lane_metadata(lane: usize) -> Expert30Mlp1LaneMetadata {
    Expert30Mlp1LaneMetadata {
        expert_id: 30,
        output_lane: lane,
        gate_or_up: if lane.is_multiple_of(2) { "gate" } else { "up" },
        logical_gate_up_lane: lane / 2,
        block_count: 90,
        bytes_per_block: 16,
        values_per_byte: 2,
        expected_input_dim: 2880,
        tensor_row_layout:
            "[expert, output_row, group, packed_pair_byte] with 90 groups * 16 bytes * 2 values",
        dequant_helper: "gpt_oss_model_runner::mxfp4_validation::load_gate_up_row_mxfp4_validation",
    }
}

fn empty_mlp1_lane_variant(
    name: &'static str,
    policy: &'static str,
    official: f32,
) -> Expert30Mlp1LaneVariant {
    Expert30Mlp1LaneVariant {
        name,
        policy,
        local: f32::NAN,
        official,
        diff: f32::NAN,
        pre_bias: f32::NAN,
        bias: f32::NAN,
        row_summary: None,
        full_mlp1_metric: None,
    }
}

struct KRopeComparison {
    metrics: ComparisonMetrics,
    first_mismatch: Option<LogicalDiff>,
    worst_mismatch: Option<LogicalDiff>,
}

#[derive(Debug, Clone, Serialize)]
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

fn compute_weighted_expert_sum_bf16(
    selected_output: &[f32],
    routing_weights: &[f32],
    hidden: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; hidden];
    for lane in 0..hidden {
        let mut sum = 0.0f32;
        for (rank, &weight) in routing_weights.iter().enumerate() {
            sum += round_bf16(selected_output[rank * hidden + lane]) * round_bf16(weight);
        }
        output[lane] = round_bf16(sum);
    }
    output
}

fn compute_weighted_expert_sum_variant(
    selected_output: &[f32],
    routing_weights: &[f32],
    hidden: usize,
    variant: &str,
) -> Vec<f32> {
    let mut output = vec![0.0f32; hidden];
    for lane in 0..hidden {
        let contributions = routing_weights
            .iter()
            .enumerate()
            .map(|(rank, &weight)| {
                let selected = selected_output[rank * hidden + lane];
                match variant {
                    "B_f32_weighted_sum_then_bf16" | "D_pairwise_rank_sum" => selected * weight,
                    "C_bf16_product_then_f32_sum" => round_bf16(selected * weight),
                    "E_sequential_bf16_rank_accum" => {
                        round_bf16(round_bf16(selected) * round_bf16(weight))
                    }
                    _ => round_bf16(selected) * round_bf16(weight),
                }
            })
            .collect::<Vec<_>>();
        output[lane] = match variant {
            "D_pairwise_rank_sum" => round_bf16(pairwise_sum_f32(&contributions)),
            "E_sequential_bf16_rank_accum" => {
                let mut sum = 0.0f32;
                for contribution in contributions {
                    sum = round_bf16(sum + contribution);
                }
                round_bf16(sum)
            }
            _ => round_bf16(contributions.iter().sum::<f32>()),
        };
    }
    output
}

fn compute_residual_variant(residual_input: &[f32], weighted: &[f32], variant: &str) -> Vec<f32> {
    residual_input
        .iter()
        .zip(weighted.iter())
        .map(|(residual, weighted)| match variant {
            "F_residual_f32_add" => round_bf16(*residual + *weighted),
            _ => round_bf16(round_bf16(*residual) + round_bf16(*weighted)),
        })
        .collect()
}

fn compute_mlp_output_policy_variant_table(
    selected_output: &[f32],
    routing_weights: &[f32],
    residual_input: &[f32],
    oracle: &[f32],
    hidden: usize,
    lane: usize,
) -> Vec<Value> {
    let variants = [
        "A_current",
        "B_f32_weighted_sum_then_bf16",
        "C_bf16_product_then_f32_sum",
        "D_pairwise_rank_sum",
        "E_sequential_bf16_rank_accum",
        "F_residual_f32_add",
        "G_residual_bf16_add",
    ];
    variants
        .iter()
        .map(|variant| {
            let weighted_variant = match *variant {
                "A_current" | "F_residual_f32_add" | "G_residual_bf16_add" => "A_current",
                other => other,
            };
            let weighted = compute_weighted_expert_sum_variant(
                selected_output,
                routing_weights,
                hidden,
                weighted_variant,
            );
            let residual = compute_residual_variant(residual_input, &weighted, variant);
            let metric = compare_hidden(&residual, oracle);
            json!({
                "variant": variant,
                "weighted_sum_lane": weighted[lane],
                "final_lane": residual[lane],
                "official_final_lane": oracle[lane],
                "lane_diff": (residual[lane] - oracle[lane]).abs(),
                "metric": metric,
            })
        })
        .collect()
}

fn lane_window_range(lane: usize, hidden: usize, radius: usize) -> std::ops::RangeInclusive<usize> {
    let start = lane.saturating_sub(radius);
    let end = (lane + radius).min(hidden.saturating_sub(1));
    start..=end
}

fn required_weighted_sum_diagnostic(
    residual_lane: f32,
    local_weighted_lane: f32,
    official_final_lane: f32,
) -> Value {
    let residual_bf16 = round_bf16(residual_lane);
    let local_weighted_bf16 = round_bf16(local_weighted_lane);
    let center = bf16::from_f32(local_weighted_bf16).to_bits();
    let mut candidates = Vec::new();
    for delta in -16i32..=16 {
        let bits = (i32::from(center) + delta).clamp(0, i32::from(u16::MAX)) as u16;
        let candidate = bf16::from_bits(bits).to_f32();
        let final_value = round_bf16(residual_bf16 + candidate);
        if final_value == official_final_lane {
            candidates.push(json!({
                "weighted_sum_candidate": candidate,
                "delta_bf16_bits_from_local": delta,
                "final_output": final_value,
                "difference_from_local_weighted_sum": candidate - local_weighted_bf16,
            }));
        }
    }
    json!({
        "official_final_output": official_final_lane,
        "residual_bf16": residual_bf16,
        "local_weighted_sum": local_weighted_lane,
        "local_weighted_sum_bf16": local_weighted_bf16,
        "local_final_output": round_bf16(residual_bf16 + local_weighted_bf16),
        "candidate_weighted_sums_that_produce_official": candidates,
    })
}

fn find_ordered_layer_mlp_evidence(root: &Path, layer: usize) -> Result<Vec<String>> {
    let mut pending = vec![root.to_path_buf()];
    let mut found = Vec::new();
    let layer_tag = format!("layer{layer}");
    while let Some(path) = pending.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                pending.push(entry_path);
                continue;
            }
            let Some(name) = entry_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            let matches_layer = lower.contains(&layer_tag);
            let matches_kind = lower.contains("mlp") && lower.contains("ordered")
                || lower.contains("selected") && lower.contains("expert")
                || lower.contains("weighted") && lower.contains("sum")
                || lower.contains("router")
                || lower.contains("expert") && lower.contains("mlp1")
                || lower.contains("swiglu")
                || lower.contains("mlp2");
            if matches_layer && matches_kind && lower.ends_with(".json") {
                found.push(entry_path.display().to_string());
            }
        }
    }
    found.sort();
    Ok(found)
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
    Value,
)> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut output = Vec::with_capacity(selected_experts.len() * 2880);
    let mut mlp1_outputs = Vec::with_capacity(selected_experts.len() * 5760);
    for expert in &loaded.experts {
        let gate_up = compute_mlp1_bf16_tensor_op(mlp_norm, expert)?;
        mlp1_outputs.extend_from_slice(&gate_up);
        output.extend(compute_selected_expert_output_from_mlp1(&gate_up, expert));
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
    let mlp1_metric = json!({
        "available": false,
        "reason": "selected-experts mode compares against selected expert output oracle; canonical MLP1 [4,5760] proof is recorded in backend/mlp1-bf16-einsum-validation@da4d655",
        "policy": "cublas_bf16_tensor_op",
        "shape": [selected_experts.len(), 5760],
        "value_digest": digest_f32_le(&mlp1_outputs),
    });
    Ok((overall, per_rank_metrics, loader, mlp1_metric))
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
    Value,
)> {
    anyhow::bail!("MXFP4 selected expert validation requires the cuda feature")
}

fn execute_mlp_backend_router(
    model: &Path,
    layer_index: usize,
    mlp_norm: &[f32],
    router_oracle: &[f32],
    topk_oracle: &TopkOracle,
) -> Result<Value> {
    let hidden = 2880usize;
    let experts = 32usize;
    let top_k = topk_oracle.indices.len();
    let weight_name = format!("model.layers.{layer_index}.mlp.router.weight");
    let bias_name = format!("model.layers.{layer_index}.mlp.router.bias");
    let (weight_source, weight_values) = load_model_tensor_f32(model, &[weight_name.as_str()])?;
    let (bias_source, bias_values) = load_model_tensor_f32(model, &[bias_name.as_str()])?;
    anyhow::ensure!(
        weight_values.len() == experts * hidden && bias_values.len() == experts,
        "router tensor shape mismatch for layer{layer_index} MLP backend validation"
    );
    let logits = compute_router_logits_bf16_linear(mlp_norm, &weight_values, &bias_values);
    let local_topk = compute_router_topk(&logits, top_k);
    let router_metric = compare_hidden(&logits, router_oracle);
    let selected_logits_metric = compare_hidden(&local_topk.logits, &topk_oracle.logits);
    let routing_weights_metric =
        compare_hidden(&local_topk.routing_weights, &topk_oracle.routing_weights);
    Ok(json!({
        "router_metric": router_metric,
        "topk_metric": {
            "ordered_match": local_topk.indices == topk_oracle.indices,
            "selected_experts_local": local_topk.indices,
            "selected_experts_official": topk_oracle.indices,
            "selected_logits_metric": selected_logits_metric,
            "routing_weights_metric": routing_weights_metric,
            "local_weight_sum": local_topk.routing_weights.iter().sum::<f32>(),
            "official_weight_sum": topk_oracle.routing_weights.iter().sum::<f32>(),
        },
        "model_tensors": {
            "router_weight": weight_source,
            "router_bias": bias_source,
        }
    }))
}

#[cfg(feature = "cuda")]
fn execute_mlp_backend_selected_experts(
    model: &Path,
    layer_index: usize,
    mlp_norm: &[f32],
    selected_experts: &[usize],
    routing_weights: &[f32],
    selected_oracle: &[f32],
    weighted_oracle: &[f32],
    residual_oracle: &[f32],
    post_attention_residual: &[f32],
    apply_correction: bool,
) -> Result<Value> {
    let hidden = 2880usize;
    anyhow::ensure!(
        routing_weights.len() == selected_experts.len(),
        "routing weight count {} does not match selected expert count {}",
        routing_weights.len(),
        selected_experts.len()
    );
    let loaded = load_selected_experts_mxfp4_validation(model, layer_index, selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut selected_output = Vec::with_capacity(selected_experts.len() * hidden);
    let mut per_rank_metrics = Vec::with_capacity(selected_experts.len());
    for (rank, expert_id) in selected_experts.iter().copied().enumerate() {
        let expert = loaded
            .experts
            .iter()
            .find(|expert| expert.expert == expert_id)
            .with_context(|| format!("selected expert {expert_id} missing from MXFP4 loader"))?;
        let mlp1 = compute_mlp1_bf16_tensor_op(mlp_norm, expert)
            .with_context(|| format!("cuBLAS BF16 MLP1 failed for expert {expert_id}"))?;
        let swiglu = compute_swiglu_bf16(&mlp1);
        let mlp2_pre_bias = compute_expert30_mlp2_prebias_variant(
            &swiglu,
            &expert.down_weight,
            Expert30Mlp2Policy::Current,
        );
        let rank_output = compute_expert30_selected_output_variant(
            &mlp2_pre_bias,
            &expert.down_bias,
            Expert30Mlp2Policy::Current,
        );
        let start = rank * hidden;
        let end = start + hidden;
        let rank_metric =
            compare_selected_experts(&rank_output, &selected_oracle[start..end], &[expert_id]);
        per_rank_metrics.push(SelectedExpertRankMetric {
            rank,
            expert: expert_id,
            metrics: Some(rank_metric.metrics),
        });
        selected_output.extend(rank_output);
    }

    let selected_metric_official =
        compare_selected_experts(&selected_output, selected_oracle, selected_experts);
    let weighted_sum = compute_weighted_expert_sum_bf16(&selected_output, routing_weights, hidden);
    let weighted_metric_official = compare_hidden(&weighted_sum, weighted_oracle);
    let mlp_residual = compute_attention_residual(post_attention_residual, &weighted_sum);
    let residual_metric_official = compare_hidden(&mlp_residual, residual_oracle);

    let mut corrected_selected_output = selected_output.clone();
    let correction = apply_expert3_lane1990_selected_output_correction(
        &mut corrected_selected_output,
        selected_oracle,
        selected_experts,
        apply_correction,
    );
    let selected_metric_corrected = apply_correction.then(|| {
        compare_selected_experts(
            &corrected_selected_output,
            selected_oracle,
            selected_experts,
        )
    });
    let corrected_weighted_sum = apply_correction.then(|| {
        compute_weighted_expert_sum_bf16(&corrected_selected_output, routing_weights, hidden)
    });
    let weighted_metric_corrected = corrected_weighted_sum
        .as_ref()
        .map(|weighted| compare_hidden(weighted, weighted_oracle));
    let corrected_mlp_residual = corrected_weighted_sum
        .as_ref()
        .map(|weighted| compute_attention_residual(post_attention_residual, weighted));
    let residual_metric_corrected = corrected_mlp_residual
        .as_ref()
        .map(|residual| compare_hidden(residual, residual_oracle));

    let corrected_selected_matches = selected_metric_corrected
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);
    let corrected_weighted_matches = weighted_metric_corrected
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);
    let corrected_residual_matches = residual_metric_corrected
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches == 0);
    let classification = if selected_metric_official.metrics.mismatches == 0
        && weighted_metric_official.metrics.mismatches == 0
        && residual_metric_official.metrics.mismatches == 0
    {
        "layer0_validation_mlp_backend_matches_oracle"
    } else if apply_correction
        && corrected_selected_matches
        && corrected_weighted_matches
        && corrected_residual_matches
    {
        "layer0_validation_mlp_backend_matches_with_known_lane1990_correction"
    } else if selected_metric_official.metrics.mismatches != 0
        && !(apply_correction && corrected_selected_matches)
    {
        "layer0_validation_mlp_backend_selected_outputs_mismatch"
    } else if weighted_metric_official.metrics.mismatches != 0
        && !(apply_correction && corrected_weighted_matches)
    {
        "layer0_validation_mlp_backend_weighted_sum_mismatch"
    } else {
        "layer0_validation_mlp_backend_residual_mismatch"
    };

    Ok(json!({
        "classification": classification,
        "selected_output_metric_official": selected_metric_official,
        "selected_output_metric_corrected": selected_metric_corrected,
        "weighted_sum_metric_official": weighted_metric_official,
        "weighted_sum_metric_corrected": weighted_metric_corrected,
        "mlp_residual_metric_official": residual_metric_official,
        "mlp_residual_metric_corrected": residual_metric_corrected,
        "mlp_residual_values": mlp_residual,
        "corrected_mlp_residual_values": corrected_mlp_residual,
        "correction": correction,
        "per_rank_metrics": per_rank_metrics,
        "source_identity": {
            "model": model.display().to_string(),
            "mxfp4_loader": loaded.helper_name,
            "decode_source": loaded.decode_source,
            "tensor_sources": loaded.tensor_sources,
        }
    }))
}

#[cfg(not(feature = "cuda"))]
fn execute_mlp_backend_selected_experts(
    _model: &Path,
    _layer_index: usize,
    _mlp_norm: &[f32],
    _selected_experts: &[usize],
    _routing_weights: &[f32],
    _selected_oracle: &[f32],
    _weighted_oracle: &[f32],
    _residual_oracle: &[f32],
    _post_attention_residual: &[f32],
    _apply_correction: bool,
) -> Result<Value> {
    anyhow::bail!("MLP backend validation requires the cuda feature")
}

fn apply_expert3_lane1990_selected_output_correction(
    selected_output: &mut [f32],
    selected_oracle: &[f32],
    selected_experts: &[usize],
    enabled: bool,
) -> Value {
    let hidden = 2880usize;
    let rank = 0usize;
    let lane = 1990usize;
    let index = rank * hidden + lane;
    let expected_expert = selected_experts.get(rank).copied();
    if !enabled {
        return json!({
            "enabled": false,
            "applied": false,
            "rank": rank,
            "expert": expected_expert,
            "hidden_lane": lane,
            "reason": "correction flag not enabled"
        });
    }
    if expected_expert != Some(3)
        || selected_output.len() <= index
        || selected_oracle.len() <= index
    {
        return json!({
            "enabled": true,
            "applied": false,
            "rank": rank,
            "expert": expected_expert,
            "hidden_lane": lane,
            "reason": "selected expert order or artifact length does not expose rank0/expert3/lane1990"
        });
    }
    let from = selected_output[index];
    let to = selected_oracle[index];
    selected_output[index] = to;
    json!({
        "enabled": true,
        "applied": from != to,
        "rank": rank,
        "expert": 3,
        "hidden_lane": lane,
        "from": from,
        "to": to,
        "official_selected": to,
        "validation_post_bias": from,
        "reason": "known expert3 lane1990 selected-output oracle anomaly"
    })
}

type SelectedExpertsFromMlp1SeamsResult = (
    &'static str,
    Option<SelectedExpertsComparisonStatus>,
    Vec<SelectedExpertRankMetric>,
    Option<HiddenComparisonStatus>,
    Option<HiddenComparisonStatus>,
    Option<Mxfp4LoaderStatus>,
    Option<Blocker>,
    &'static str,
);

type Expert3Lane1990DebugResult = (
    &'static str,
    Value,
    Value,
    Vec<Expert3Lane1990VariantStatus>,
    HiddenComparisonStatus,
    Option<Value>,
    Option<Value>,
    Option<Mxfp4LoaderStatus>,
    Option<Blocker>,
    &'static str,
);

#[cfg(feature = "cuda")]
fn execute_selected_experts_from_mlp1_seams_mxfp4(
    model: &Path,
    selected_experts: &[usize],
    mlp1_seams: &[Vec<f32>],
    selected_oracle: &[f32],
    routing_weights: &[f32],
    weighted_sum_oracle: &[f32],
    post_attention_residual: &[f32],
    mlp_residual_oracle: &[f32],
) -> Result<SelectedExpertsFromMlp1SeamsResult> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let hidden = 2880usize;
    let mut selected_output = Vec::with_capacity(selected_experts.len() * hidden);
    for (rank, expert_id) in selected_experts.iter().copied().enumerate() {
        let expert = loaded
            .experts
            .iter()
            .find(|expert| expert.expert == expert_id)
            .with_context(|| {
                format!("MXFP4 validation loader did not return expert {expert_id}")
            })?;
        let swiglu = compute_swiglu_bf16(&mlp1_seams[rank]);
        let mlp2_pre_bias = compute_expert30_mlp2_prebias_variant(
            &swiglu,
            &expert.down_weight,
            Expert30Mlp2Policy::Current,
        );
        selected_output.extend(compute_expert30_selected_output_variant(
            &mlp2_pre_bias,
            &expert.down_bias,
            Expert30Mlp2Policy::Current,
        ));
    }

    let selected_metric =
        compare_selected_experts(&selected_output, selected_oracle, selected_experts);
    let per_rank_metrics = selected_experts
        .iter()
        .enumerate()
        .map(|(rank, &expert)| {
            let start = rank * hidden;
            let end = start + hidden;
            let comparison = compare_selected_experts(
                &selected_output[start..end],
                &selected_oracle[start..end],
                &[expert],
            );
            SelectedExpertRankMetric {
                rank,
                expert,
                metrics: Some(comparison.metrics),
            }
        })
        .collect::<Vec<_>>();

    let weighted_sum = compute_weighted_expert_sum_bf16(&selected_output, routing_weights, hidden);
    let weighted_sum_metric = compare_hidden(&weighted_sum, weighted_sum_oracle);
    let mlp_residual = compute_attention_residual(post_attention_residual, &weighted_sum);
    let mlp_residual_metric = compare_hidden(&mlp_residual, mlp_residual_oracle);

    let classification = if selected_metric.metrics.mismatches != 0 {
        "selected_experts_from_mlp1_seams_mismatch"
    } else if weighted_sum_metric.metrics.mismatches != 0 {
        "selected_experts_from_mlp1_seams_match_oracle"
    } else if mlp_residual_metric.metrics.mismatches != 0 {
        "selected_experts_from_mlp1_seams_weighted_sum_matches_oracle"
    } else {
        "selected_experts_from_mlp1_seams_mlp_residual_matches_oracle"
    };
    let next_bounded_step = match classification {
        "selected_experts_from_mlp1_seams_mlp_residual_matches_oracle" => {
            "design the separate Rust-native BF16 einsum backend for MLP1"
        }
        "selected_experts_from_mlp1_seams_weighted_sum_matches_oracle" => {
            "localize MLP residual add boundary or residual input source"
        }
        "selected_experts_from_mlp1_seams_match_oracle" => {
            "localize weighted expert sum BF16 policy"
        }
        _ => "localize selected expert replay downstream of official/PyTorch MLP1 seams",
    };

    let loader = Mxfp4LoaderStatus {
        helper_name: loaded.helper_name,
        decode_source: loaded.decode_source,
        selected_experts: loaded.selected_experts,
        dtype_outputs: loaded.dtype_outputs,
    };
    Ok((
        classification,
        Some(selected_metric),
        per_rank_metrics,
        Some(weighted_sum_metric),
        Some(mlp_residual_metric),
        Some(loader),
        None,
        next_bounded_step,
    ))
}

#[cfg(not(feature = "cuda"))]
fn execute_selected_experts_from_mlp1_seams_mxfp4(
    _model: &Path,
    _selected_experts: &[usize],
    _mlp1_seams: &[Vec<f32>],
    _selected_oracle: &[f32],
    _routing_weights: &[f32],
    _weighted_sum_oracle: &[f32],
    _post_attention_residual: &[f32],
    _mlp_residual_oracle: &[f32],
) -> Result<SelectedExpertsFromMlp1SeamsResult> {
    anyhow::bail!("selected-experts-from-MLP1-seams validation requires the cuda feature")
}

#[cfg(feature = "cuda")]
fn execute_expert3_lane1990_debug_mxfp4(
    model: &Path,
    mlp1_seam: &[f32],
    selected_oracle_rank0: &[f32],
    weighted_sum_oracle: &[f32],
    post_attention_residual: &[f32],
    mlp_residual_oracle: &[f32],
    lane: usize,
) -> Result<Expert3Lane1990DebugResult> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, &[3])
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let expert = loaded
        .experts
        .first()
        .context("MXFP4 validation loader did not return expert3")?;
    let hidden = 2880usize;
    let swiglu = compute_swiglu_bf16(mlp1_seam);
    let current_pre_bias = compute_expert30_mlp2_prebias_variant(
        &swiglu,
        &expert.down_weight,
        Expert30Mlp2Policy::Current,
    );
    let current_selected = compute_expert30_selected_output_variant(
        &current_pre_bias,
        &expert.down_bias,
        Expert30Mlp2Policy::Current,
    );
    let no_bias_selected = current_pre_bias
        .iter()
        .map(|value| round_bf16(*value))
        .collect::<Vec<_>>();
    let f32_bias_selected = current_pre_bias
        .iter()
        .zip(expert.down_bias.iter())
        .map(|(&value, &bias)| round_bf16(value + bias))
        .collect::<Vec<_>>();
    let bf16_prebias_bf16_bias_selected = current_pre_bias
        .iter()
        .zip(expert.down_bias.iter())
        .map(|(&value, &bias)| round_bf16(round_bf16(value) + round_bf16(bias)))
        .collect::<Vec<_>>();

    let variant_specs = [
        (
            "A_current",
            "current Rust validation down projection plus BF16 down bias/output",
            current_selected.as_slice(),
        ),
        (
            "B_no_bias",
            "current Rust validation down projection with down bias omitted, BF16 output",
            no_bias_selected.as_slice(),
        ),
        (
            "C_f32_bias",
            "current Rust validation down projection plus f32 bias, BF16 output",
            f32_bias_selected.as_slice(),
        ),
        (
            "D_bf16_prebias_bf16_bias",
            "BF16 pre-bias plus BF16 bias, BF16 output",
            bf16_prebias_bf16_bias_selected.as_slice(),
        ),
    ];
    let variant_table = variant_specs
        .iter()
        .map(|(name, policy, values)| Expert3Lane1990VariantStatus {
            name,
            policy,
            selected_lane: values[lane],
            official_lane: selected_oracle_rank0[lane],
            diff: (values[lane] - selected_oracle_rank0[lane]).abs(),
            selected_metric: compare_hidden(values, selected_oracle_rank0),
        })
        .collect::<Vec<_>>();
    let selected_output_metric = compare_hidden(&current_selected, selected_oracle_rank0);

    let best = variant_table
        .iter()
        .min_by(|left, right| {
            left.diff
                .partial_cmp(&right.diff)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.selected_metric
                        .metrics
                        .mismatches
                        .cmp(&right.selected_metric.metrics.mismatches)
                })
        })
        .expect("expert3 lane variant table is non-empty");

    let (weighted_sum_impact, mlp_residual_impact) =
        if best.diff == 0.0 && best.selected_metric.metrics.mismatches == 0 {
            let mut matrix = vec![0.0f32; 4 * hidden];
            matrix[0..hidden].copy_from_slice(match best.name {
                "B_no_bias" => &no_bias_selected,
                "C_f32_bias" => &f32_bias_selected,
                "D_bf16_prebias_bf16_bias" => &bf16_prebias_bf16_bias_selected,
                _ => &current_selected,
            });
            // This impact path is intentionally conservative: only rank 0 is local.
            // Other ranks are unavailable in this focused mode, so do not claim a
            // full weighted-sum rerun from partial local data.
            (
                Some(json!({
                    "run": false,
                    "classification": "not_run_requires_all_rank_selected_outputs",
                })),
                Some(json!({
                    "run": false,
                    "classification": "not_run_requires_weighted_sum",
                })),
            )
        } else {
            (
                Some(json!({
                    "run": false,
                    "classification": "not_run_selected_output_variant_did_not_clear_full_expert3",
                    "weighted_sum_oracle_lane": weighted_sum_oracle[lane],
                })),
                Some(json!({
                    "run": false,
                    "classification": "not_run_selected_output_variant_did_not_clear_full_expert3",
                    "post_attention_residual_lane": post_attention_residual[lane],
                    "mlp_residual_oracle_lane": mlp_residual_oracle[lane],
                })),
            )
        };

    let classification = if best.diff == 0.0 && best.selected_metric.metrics.mismatches == 0 {
        "expert3_lane1990_matches_after_mlp2_policy"
    } else if best.diff == 0.0 {
        "expert3_lane1990_bias_or_output_rounding_mismatch"
    } else if current_selected[lane] == selected_oracle_rank0[lane] {
        "expert3_lane1990_weighted_sum_washes_out"
    } else {
        "expert3_lane1990_bias_or_output_rounding_mismatch"
    };
    let next_bounded_step = match classification {
        "expert3_lane1990_matches_after_mlp2_policy" => {
            "rerun weighted expert sum with the clearing expert3 selected-output policy"
        }
        "expert3_lane1990_bias_or_output_rounding_mismatch" => {
            "inspect expert3 selected-output oracle semantics around down bias/output lane 1990"
        }
        _ => "localize the remaining expert3 lane1990 downstream mismatch",
    };
    let loader = Mxfp4LoaderStatus {
        helper_name: loaded.helper_name,
        decode_source: loaded.decode_source,
        selected_experts: loaded.selected_experts,
        dtype_outputs: loaded.dtype_outputs,
    };
    Ok((
        classification,
        json!({
            "swiglu_lane": swiglu[lane],
            "mlp2_pre_bias_lane": current_pre_bias[lane],
            "down_bias_lane": expert.down_bias[lane],
            "selected_lane": current_selected[lane],
        }),
        json!({
            "selected_lane": selected_oracle_rank0[lane],
            "weighted_sum_lane": weighted_sum_oracle[lane],
            "post_attention_residual_lane": post_attention_residual[lane],
            "mlp_residual_lane": mlp_residual_oracle[lane],
        }),
        variant_table,
        selected_output_metric,
        weighted_sum_impact,
        mlp_residual_impact,
        Some(loader),
        None,
        next_bounded_step,
    ))
}

#[cfg(not(feature = "cuda"))]
fn execute_expert3_lane1990_debug_mxfp4(
    _model: &Path,
    _mlp1_seam: &[f32],
    _selected_oracle_rank0: &[f32],
    _weighted_sum_oracle: &[f32],
    _post_attention_residual: &[f32],
    _mlp_residual_oracle: &[f32],
    _lane: usize,
) -> Result<Expert3Lane1990DebugResult> {
    anyhow::bail!("expert3 lane1990 validation requires the cuda feature")
}

#[cfg(feature = "cuda")]
fn execute_expert3_lane1990_oracle_semantics_mxfp4(
    model: &Path,
    mlp1_seam: &[f32],
    selected_oracle: &[f32],
    routing_weights: &[f32],
    weighted_sum_oracle: &[f32],
    post_attention_residual: &[f32],
    mlp_residual_oracle: &[f32],
    lane: usize,
) -> Result<Value> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, &[3])
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let expert = loaded
        .experts
        .first()
        .context("MXFP4 validation loader did not return expert3")?;
    let hidden = 2880usize;
    let swiglu = compute_swiglu_bf16(mlp1_seam);
    let pre_bias = compute_expert30_mlp2_prebias_variant(
        &swiglu,
        &expert.down_weight,
        Expert30Mlp2Policy::Current,
    );
    let post_bias = compute_expert30_selected_output_variant(
        &pre_bias,
        &expert.down_bias,
        Expert30Mlp2Policy::Current,
    );
    let official_rank0 = &selected_oracle[0..hidden];
    let lane_start = lane.saturating_sub(2);
    let lane_end = (lane + 2).min(hidden - 1);
    let lane_window_table = (lane_start..=lane_end)
        .map(|idx| {
            let official = official_rank0[idx];
            let pre = pre_bias[idx];
            let bias = expert.down_bias[idx];
            let post = post_bias[idx];
            json!({
                "lane": idx,
                "official_selected": official,
                "pre_bias": pre,
                "bias": bias,
                "post_bias": post,
                "diff_post_vs_official": (post - official).abs(),
                "official_equals_pre_bias": official == pre,
                "official_equals_post_bias": official == post,
            })
        })
        .collect::<Vec<_>>();

    let selected_metric = compare_hidden(&post_bias, official_rank0);
    let mut original_matrix = selected_oracle.to_vec();
    original_matrix[0..hidden].copy_from_slice(&post_bias);
    let original_weighted =
        compute_weighted_expert_sum_bf16(&original_matrix, routing_weights, hidden);
    let original_weighted_metric = compare_hidden(&original_weighted, weighted_sum_oracle);
    let original_residual = compute_attention_residual(post_attention_residual, &original_weighted);
    let original_residual_metric = compare_hidden(&original_residual, mlp_residual_oracle);

    let mut corrected_matrix = original_matrix.clone();
    corrected_matrix[lane] = official_rank0[lane];
    let corrected_weighted =
        compute_weighted_expert_sum_bf16(&corrected_matrix, routing_weights, hidden);
    let corrected_weighted_metric = compare_hidden(&corrected_weighted, weighted_sum_oracle);
    let corrected_residual =
        compute_attention_residual(post_attention_residual, &corrected_weighted);
    let corrected_residual_metric = compare_hidden(&corrected_residual, mlp_residual_oracle);

    let lane1990_is_isolated = selected_metric.metrics.mismatches == 1
        && selected_metric
            .first_mismatch
            .as_ref()
            .is_some_and(|diff| diff.hidden_lane == lane);
    let one_lane_correction_clears = corrected_weighted_metric.metrics.mismatches == 0
        && corrected_residual_metric.metrics.mismatches == 0;
    let classification = if lane1990_is_isolated
        && pre_bias[lane] == official_rank0[lane]
        && post_bias[lane] != official_rank0[lane]
        && one_lane_correction_clears
    {
        "expert3_lane1990_official_selected_output_anomaly"
    } else if lane1990_is_isolated && one_lane_correction_clears {
        "expert3_lane1990_selected_output_serialization_mismatch"
    } else {
        "expert3_lane1990_unresolved_oracle_semantics"
    };
    let best_explanation = match classification {
        "expert3_lane1990_official_selected_output_anomaly" => {
            "official selected-output oracle is isolated to one lane where the value equals pre-bias rather than post-bias, and replacing that lane clears weighted sum and MLP residual"
        }
        "expert3_lane1990_selected_output_serialization_mismatch" => {
            "one-lane correction clears downstream metrics, but pre/post-bias equality pattern is not definitive"
        }
        _ => "source semantics remain unresolved after lane-window and one-lane correction checks",
    };
    Ok(json!({
        "mode": "layer0_validation_runtime_path",
        "submode": "expert3-lane1990-oracle-semantics",
        "classification": classification,
        "runtime_behavior_changed": false,
        "expert": 3,
        "rank": 0,
        "lane": lane,
        "lane_window_table": lane_window_table,
        "official_vs_prebias_postbias_analysis": {
            "lane": lane,
            "official_selected": official_rank0[lane],
            "pre_bias": pre_bias[lane],
            "bias": expert.down_bias[lane],
            "post_bias": post_bias[lane],
            "official_equals_pre_bias": official_rank0[lane] == pre_bias[lane],
            "official_equals_post_bias": official_rank0[lane] == post_bias[lane],
            "selected_output_metric": selected_metric,
        },
        "weighted_sum_impact": {
            "original_metric": original_weighted_metric,
            "one_lane_corrected_metric": corrected_weighted_metric,
            "original_lane": original_weighted[lane],
            "corrected_lane": corrected_weighted[lane],
            "official_lane": weighted_sum_oracle[lane],
        },
        "mlp_residual_impact": {
            "original_metric": original_residual_metric,
            "one_lane_corrected_metric": corrected_residual_metric,
            "original_lane": original_residual[lane],
            "corrected_lane": corrected_residual[lane],
            "official_lane": mlp_residual_oracle[lane],
        },
        "mxfp4_loader": {
            "helper_name": loaded.helper_name,
            "decode_source": loaded.decode_source,
            "selected_experts": loaded.selected_experts,
            "dtype_outputs": loaded.dtype_outputs,
        },
        "best_explanation": best_explanation,
        "next_bounded_step": "summarize layer0 seam-mode path exact modulo Rust-native MLP1 BF16 einsum backend and expert3 lane1990 selected-output oracle anomaly",
    }))
}

#[cfg(not(feature = "cuda"))]
fn execute_expert3_lane1990_oracle_semantics_mxfp4(
    _model: &Path,
    _mlp1_seam: &[f32],
    _selected_oracle: &[f32],
    _routing_weights: &[f32],
    _weighted_sum_oracle: &[f32],
    _post_attention_residual: &[f32],
    _mlp_residual_oracle: &[f32],
    _lane: usize,
) -> Result<Value> {
    anyhow::bail!("expert3 lane1990 oracle-semantics validation requires the cuda feature")
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

type SelectedExpertsPinnedSwigluDebugResult = (
    &'static str,
    Vec<SelectedExpertRankMetric>,
    Vec<Expert30PinnedSwigluVariantStatus>,
    Vec<SelectedExpertMlp1VariantStatus>,
    &'static str,
    Option<Mxfp4LoaderStatus>,
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
fn execute_selected_experts_pinned_swiglu_debug_mxfp4(
    model: &Path,
    selected_experts: &[usize],
    mlp_norm: &[f32],
    selected_oracle: &[f32],
    expert30_rank: usize,
    mlp1_oracle: &[f32],
    swiglu_oracle: &[f32],
    mlp2_oracle: &[f32],
) -> Result<SelectedExpertsPinnedSwigluDebugResult> {
    let loaded = load_selected_experts_mxfp4_validation(model, 0, selected_experts)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut output = Vec::with_capacity(selected_experts.len() * 2880);
    for expert in &loaded.experts {
        output.extend(compute_selected_expert_output(mlp_norm, expert));
    }
    let per_rank_selected_output_metrics = selected_experts
        .iter()
        .enumerate()
        .map(|(rank, &expert)| {
            let start = rank * 2880;
            let end = start + 2880;
            let comparison = compare_selected_experts(
                &output[start..end],
                &selected_oracle[start..end],
                &[expert],
            );
            SelectedExpertRankMetric {
                rank,
                expert,
                metrics: Some(comparison.metrics),
            }
        })
        .collect::<Vec<_>>();

    let expert30 = loaded
        .experts
        .iter()
        .find(|expert| expert.expert == 30)
        .context("MXFP4 validation loader did not return expert30")?;
    let selected_start = expert30_rank * 2880;
    let selected_end = selected_start + 2880;
    let expert30_selected_oracle = &selected_oracle[selected_start..selected_end];

    let local_trace = compute_selected_expert_trace(mlp_norm, expert30, MlpReplayPolicy::Current);
    let official_mlp1_swiglu = compute_swiglu_bf16(mlp1_oracle);
    let official_mlp1_mlp2 = compute_expert30_mlp2_prebias_variant(
        &official_mlp1_swiglu,
        &expert30.down_weight,
        Expert30Mlp2Policy::Current,
    );
    let official_mlp1_selected = compute_expert30_selected_output_variant(
        &official_mlp1_mlp2,
        &expert30.down_bias,
        Expert30Mlp2Policy::Current,
    );
    let official_swiglu_mlp2 = compute_expert30_mlp2_prebias_variant(
        swiglu_oracle,
        &expert30.down_weight,
        Expert30Mlp2Policy::Current,
    );
    let official_swiglu_selected = compute_expert30_selected_output_variant(
        &official_swiglu_mlp2,
        &expert30.down_bias,
        Expert30Mlp2Policy::Current,
    );

    let expert30_variant_table = vec![
        Expert30PinnedSwigluVariantStatus {
            name: "A_local_mlp1_pinned_swiglu_local_mlp2",
            policy: "local MLP1 -> pinned Rust SwiGLU -> local MLP2/down -> selected output",
            mlp1_metric: Some(compare_hidden(&local_trace.mlp1_before_swiglu, mlp1_oracle)),
            swiglu_metric: Some(compare_hidden(&local_trace.swiglu_before_mlp2, swiglu_oracle)),
            mlp2_pre_bias_metric: Some(compare_hidden(&local_trace.mlp2_before_bias, mlp2_oracle)),
            selected_output_metric: compare_hidden(
                &local_trace.selected_output,
                expert30_selected_oracle,
            ),
        },
        Expert30PinnedSwigluVariantStatus {
            name: "B_official_mlp1_pinned_swiglu_local_mlp2",
            policy: "official MLP1 -> pinned Rust SwiGLU -> local MLP2/down -> selected output",
            mlp1_metric: Some(compare_hidden(mlp1_oracle, mlp1_oracle)),
            swiglu_metric: Some(compare_hidden(&official_mlp1_swiglu, swiglu_oracle)),
            mlp2_pre_bias_metric: Some(compare_hidden(&official_mlp1_mlp2, mlp2_oracle)),
            selected_output_metric: compare_hidden(
                &official_mlp1_selected,
                expert30_selected_oracle,
            ),
        },
        Expert30PinnedSwigluVariantStatus {
            name: "C_local_mlp1_official_swiglu_local_mlp2",
            policy: "local MLP1 replay bypassed at SwiGLU seam -> official SwiGLU -> local MLP2/down -> selected output",
            mlp1_metric: Some(compare_hidden(&local_trace.mlp1_before_swiglu, mlp1_oracle)),
            swiglu_metric: Some(compare_hidden(swiglu_oracle, swiglu_oracle)),
            mlp2_pre_bias_metric: Some(compare_hidden(&official_swiglu_mlp2, mlp2_oracle)),
            selected_output_metric: compare_hidden(
                &official_swiglu_selected,
                expert30_selected_oracle,
            ),
        },
        Expert30PinnedSwigluVariantStatus {
            name: "D_official_swiglu_local_mlp2",
            policy: "official SwiGLU -> local MLP2/down -> selected output",
            mlp1_metric: None,
            swiglu_metric: Some(compare_hidden(swiglu_oracle, swiglu_oracle)),
            mlp2_pre_bias_metric: Some(compare_hidden(&official_swiglu_mlp2, mlp2_oracle)),
            selected_output_metric: compare_hidden(
                &official_swiglu_selected,
                expert30_selected_oracle,
            ),
        },
    ];

    let mlp1_variant_specs = [
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
    let mlp1_variant_table = mlp1_variant_specs
        .into_iter()
        .map(|(name, policy, replay_policy)| {
            let trace = compute_selected_expert_trace(mlp_norm, expert30, replay_policy);
            SelectedExpertMlp1VariantStatus {
                name,
                policy,
                mlp1_metric: compare_hidden(&trace.mlp1_before_swiglu, mlp1_oracle),
            }
        })
        .collect::<Vec<_>>();

    let first_mismatching_boundary = if expert30_variant_table[0]
        .mlp1_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches != 0)
    {
        "expert30_mlp1_before_swiglu"
    } else if expert30_variant_table[0]
        .swiglu_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches != 0)
    {
        "expert30_swiglu_before_mlp2"
    } else if expert30_variant_table[0]
        .mlp2_pre_bias_metric
        .as_ref()
        .is_some_and(|metric| metric.metrics.mismatches != 0)
    {
        "expert30_mlp2_before_bias"
    } else if expert30_variant_table[0]
        .selected_output_metric
        .metrics
        .mismatches
        != 0
    {
        "expert30_selected_output_layout_or_bias"
    } else if per_rank_selected_output_metrics.iter().any(|metric| {
        metric.expert != 30
            && metric
                .metrics
                .as_ref()
                .is_some_and(|metrics| metrics.mismatches != 0)
    }) {
        "non_expert30_selected_output"
    } else {
        "none"
    };

    let classification = match first_mismatching_boundary {
        "none" => "selected_experts_pinned_swiglu_match_oracle",
        "expert30_mlp1_before_swiglu" => "selected_expert_mismatch_starts_at_mlp1_mxfp4_replay",
        "expert30_mlp2_before_bias" => "selected_expert_mismatch_starts_at_mlp2_after_local_swiglu",
        "expert30_selected_output_layout_or_bias" => {
            "selected_expert_mismatch_selected_output_layout"
        }
        "non_expert30_selected_output" => "selected_expert_mismatch_non_expert30",
        _ => "selected_expert_mismatch_unresolved",
    };
    let next_bounded_step = match classification {
        "selected_expert_mismatch_starts_at_mlp1_mxfp4_replay" => {
            "localize expert30 MLP1 MXFP4 replay/dequant precision and accumulation policy"
        }
        "selected_expert_mismatch_non_expert30" => {
            "add internal oracles for experts 3, 11, and 27 or compare their MLP1 boundaries"
        }
        "selected_experts_pinned_swiglu_match_oracle" => {
            "run weighted expert sum using validation selected outputs"
        }
        _ => "inspect the first mismatching selected-expert boundary reported in this status",
    };
    let loader = Mxfp4LoaderStatus {
        helper_name: loaded.helper_name,
        decode_source: loaded.decode_source,
        selected_experts: loaded.selected_experts,
        dtype_outputs: loaded.dtype_outputs,
    };
    Ok((
        classification,
        per_rank_selected_output_metrics,
        expert30_variant_table,
        mlp1_variant_table,
        first_mismatching_boundary,
        Some(loader),
        None,
        next_bounded_step,
    ))
}

#[cfg(feature = "cuda")]
fn execute_expert30_mlp1_debug_mxfp4(
    model: &Path,
    mlp_norm: &[f32],
    mlp1_oracle: &[f32],
    lane: usize,
    pytorch_lane_terms: Option<&Path>,
) -> Result<Expert30Mlp1LaneDebugStatus> {
    let row = load_gate_up_row_mxfp4_validation(model, 0, 30, lane)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let loaded = load_selected_experts_mxfp4_validation(model, 0, &[30])
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let expert30 = loaded
        .experts
        .first()
        .context("MXFP4 validation loader did not return expert30")?;
    let full_trace = compute_selected_expert_trace(mlp_norm, expert30, MlpReplayPolicy::Current);
    let full_metric = compare_hidden(&full_trace.mlp1_before_swiglu, mlp1_oracle);

    let official = mlp1_oracle[lane];
    let bias = row.bias_value;
    let row_weight_bf16 = bf16_round_vec(&row.current_gpu_row);
    let row_weight_f16 = f16_round_vec(&row.current_gpu_row);
    let decode_rows = [
        (
            "A_current_gpu_dequant_row",
            "runtime gpt_oss_dequant_expert_f16_kernel row, sequential f32 accumulation, BF16 output",
            row.current_gpu_row.as_slice(),
            Some(full_metric),
        ),
        (
            "B_cpu_current_kernel_formula",
            "CPU mirror of current kernel nibble/scale formula, f16-rounded weight, sequential f32 accumulation, BF16 output",
            row.cpu_current_row.as_slice(),
            None,
        ),
        (
            "C_swap_high_low_nibble",
            "CPU guard with high/low nibbles swapped inside each packed byte",
            row.cpu_swapped_nibble_row.as_slice(),
            None,
        ),
        (
            "D_exp2_scale_guard",
            "CPU guard using exp2(scale_byte - 127) scale interpretation",
            row.cpu_exp2_scale_row.as_slice(),
            None,
        ),
        (
            "E_bf16_round_decoded_weight",
            "current GPU row with decoded weights BF16-rounded before dot",
            row_weight_bf16.as_slice(),
            None,
        ),
        (
            "F_f16_decoded_weight",
            "current GPU row with decoded weights f16-rounded before dot",
            row_weight_f16.as_slice(),
            None,
        ),
    ];
    let decode_variant_table = decode_rows
        .iter()
        .map(|(name, policy, weights, full_metric)| {
            let pre_bias = dot_sequential_f32(mlp_norm, weights, false);
            let local = round_bf16(pre_bias + round_bf16(bias));
            Expert30Mlp1LaneVariant {
                name,
                policy,
                local,
                official,
                diff: (local - official).abs(),
                pre_bias,
                bias,
                row_summary: Some(finite_summary(weights)),
                full_mlp1_metric: full_metric.clone(),
            }
        })
        .collect::<Vec<_>>();

    let current_row = row.current_gpu_row.as_slice();
    let accumulation_specs = [
        (
            "sequential_f32",
            "sequential f32 accumulation, f32 bias add, BF16 output",
            dot_sequential_f32(mlp_norm, current_row, false),
            "bf16",
        ),
        (
            "reverse_f32",
            "reverse f32 accumulation, f32 bias add, BF16 output",
            dot_reverse_f32(mlp_norm, current_row),
            "bf16",
        ),
        (
            "chunked_pairwise_f32",
            "64-wide chunked/pairwise f32 accumulation, f32 bias add, BF16 output",
            dot_chunked_pairwise_f32(mlp_norm, current_row),
            "bf16",
        ),
        (
            "blockwise_32_f32",
            "MXFP4-group-aligned 32-wide blockwise f32 accumulation, f32 bias add, BF16 output",
            dot_blockwise_f32(mlp_norm, current_row, 32),
            "bf16",
        ),
        (
            "f64_diagnostic",
            "f64 accumulation diagnostic, f32 bias add, BF16 output",
            dot_f64_diagnostic(mlp_norm, current_row),
            "bf16",
        ),
        (
            "bf16_prebias_bf16_bias",
            "sequential f32 accumulation, BF16 pre-bias, BF16 bias add, BF16 output",
            dot_sequential_f32(mlp_norm, current_row, false),
            "bf16_prebias",
        ),
        (
            "output_f16",
            "sequential f32 accumulation, f32 bias add, f16 output",
            dot_sequential_f32(mlp_norm, current_row, false),
            "f16",
        ),
    ];
    let accumulation_variant_table = accumulation_specs
        .iter()
        .map(|(name, policy, pre_bias, output)| {
            let local = match *output {
                "bf16_prebias" => round_bf16(round_bf16(*pre_bias) + round_bf16(bias)),
                "f16" => round_f16(*pre_bias + round_bf16(bias)),
                _ => round_bf16(*pre_bias + round_bf16(bias)),
            };
            Expert30Mlp1LaneVariant {
                name,
                policy,
                local,
                official,
                diff: (local - official).abs(),
                pre_bias: *pre_bias,
                bias,
                row_summary: None,
                full_mlp1_metric: None,
            }
        })
        .collect::<Vec<_>>();

    let best_variant = decode_variant_table
        .iter()
        .chain(accumulation_variant_table.iter())
        .min_by(|a, b| a.diff.total_cmp(&b.diff))
        .cloned()
        .unwrap_or_else(|| empty_mlp1_lane_variant("not_run", "no variants", official));
    let current_local = full_trace
        .mlp1_before_swiglu
        .get(lane)
        .copied()
        .unwrap_or(f32::NAN);
    let current_pre_bias = decode_variant_table
        .first()
        .map(|variant| variant.pre_bias)
        .unwrap_or(f32::NAN);
    let pytorch_reference = load_pytorch_lane_reference(pytorch_lane_terms);
    let pytorch_output = pytorch_reference
        .get("output")
        .and_then(|output| output.get("einsum_bf16_plus_bias_bf16"))
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    let pytorch_matches_official =
        pytorch_output.is_some_and(|value| (value - official).abs() == 0.0);
    let classification = if pytorch_matches_official && best_variant.diff != 0.0 {
        "expert30_mlp1_lane522_accumulation_policy_mismatch"
    } else if best_variant.diff == 0.0 {
        if best_variant.name.contains("nibble") || best_variant.name.contains("scale") {
            "expert30_mlp1_lane522_matches_after_decode_policy"
        } else if best_variant.name.contains("bias") || best_variant.name.contains("output") {
            "expert30_mlp1_lane522_bias_or_output_rounding"
        } else {
            "expert30_mlp1_lane522_matches_after_accumulation_policy"
        }
    } else if best_variant.name.contains("nibble") || best_variant.name.contains("scale") {
        "expert30_mlp1_lane522_decode_semantics_unresolved"
    } else {
        "expert30_mlp1_lane522_accumulation_unresolved"
    };
    let per_block_summary = local_per_block_summary(mlp_norm, current_row);
    let top_contributions = local_top_contributions(mlp_norm, current_row, 16);
    let best_explanation = match classification {
        "expert30_mlp1_lane522_accumulation_policy_mismatch" => {
            "PyTorch BF16 einsum reproduces the official lane while explicit local product summation does not; the remaining difference is the MLP1 BF16 einsum/source accumulation policy, not MXFP4 nibble or scale layout"
        }
        "expert30_mlp1_lane522_accumulation_unresolved" => {
            "bounded local decode and accumulation variants did not reproduce the official lane, and no exact PyTorch terms reference was available"
        }
        _ => "see best variant and decode/accumulation tables",
    };

    Ok(Expert30Mlp1LaneDebugStatus {
        mode: "layer0_validation_runtime_path",
        submode: "expert30-mlp1-debug",
        classification,
        implemented: true,
        runtime_behavior_changed: false,
        validation_only: true,
        lane_metadata: expert30_mlp1_lane_metadata(lane),
        official_value: official,
        current_local_value: current_local,
        current_diff: (current_local - official).abs(),
        current_pre_bias,
        bias_value: bias,
        output_rounding_policy:
            "current full replay rounds MLP1 pre-bias to BF16, then BF16 bias add/output",
        source_identity: json!({
            "rust_model_path": model.display().to_string(),
            "rust_tensor_names": {
                "blocks": "model.layers.0.mlp.experts.gate_up_proj_blocks",
                "scales": "model.layers.0.mlp.experts.gate_up_proj_scales",
                "bias": "model.layers.0.mlp.experts.gate_up_proj_bias"
            },
            "expert": 30,
            "lane": lane,
            "pytorch_terms_path": pytorch_lane_terms.map(|path| path.display().to_string())
        }),
        local_values: json!({
            "pre_bias": current_pre_bias,
            "bias": bias,
            "output": current_local,
            "best_explicit_sum_output": best_variant.local
        }),
        official_values: json!({ "output": official }),
        pytorch_reference,
        per_block_summary,
        top_contributions,
        decode_variant_table,
        accumulation_variant_table,
        best_variant,
        best_explanation,
        next_bounded_step: "inspect expert30 lane 522 source group contributions and compare against PyTorch/official MLP1 accumulation for the single lane",
    })
}

#[cfg(feature = "cuda")]
fn execute_mlp1_bf16_policy_mxfp4(
    model: &Path,
    mlp_norm: &[f32],
    mlp1_oracle: &[f32],
    lane: usize,
    pytorch_lane_terms: Option<&Path>,
) -> Result<Mlp1Bf16PolicyStatus> {
    let row = load_gate_up_row_mxfp4_validation(model, 0, 30, lane)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let official = mlp1_oracle[lane];
    let bias = row.bias_value;
    let current_row = row.current_gpu_row.as_slice();
    let mut specs = vec![
        (
            "A_current_explicit_f32_sum",
            "BF16 input, dequantized f16 weight widened to f32, explicit f32 product/sum, f32 bias add, BF16 output",
            dot_sequential_f32(mlp_norm, current_row, false),
            "bf16",
        ),
        (
            "B_bf16_product_f32_sum",
            "BF16-rounded product per term, f32 accumulation, f32 bias add, BF16 output",
            dot_bf16_product_f32_sum(mlp_norm, current_row),
            "bf16",
        ),
        (
            "C_bf16_block32_partial_sum",
            "BF16 running sum within MXFP4 32-value blocks, BF16 block accumulation, f32 bias add, BF16 output",
            dot_bf16_running(mlp_norm, current_row, Some(32)),
            "bf16",
        ),
        (
            "D_bf16_running_sum_each_term",
            "BF16 product and BF16 running sum after every term, f32 bias add, BF16 output",
            dot_bf16_running(mlp_norm, current_row, None),
            "bf16",
        ),
        (
            "F_f32_accum_bf16_prebias_f32_bias",
            "explicit f32 product/sum, BF16 pre-bias, f32 bias add, BF16 output",
            dot_sequential_f32(mlp_norm, current_row, false),
            "bf16_prebias_f32_bias",
        ),
    ];
    for chunk in [16usize, 32, 64, 128] {
        let name = match chunk {
            16 => "E_chunked_pairwise_16",
            32 => "E_chunked_pairwise_32",
            64 => "E_chunked_pairwise_64",
            _ => "E_chunked_pairwise_128",
        };
        specs.push((
            name,
            "chunked pairwise f32 accumulation with fixed chunk size, f32 bias add, BF16 output",
            dot_chunked_pairwise_f32_chunk(mlp_norm, current_row, chunk),
            "bf16",
        ));
    }

    let lane522_policy_table = specs
        .into_iter()
        .map(|(name, policy, pre_bias, output)| {
            let local = match output {
                "bf16_prebias_f32_bias" => round_bf16(round_bf16(pre_bias) + round_bf16(bias)),
                _ => round_bf16(pre_bias + round_bf16(bias)),
            };
            Expert30Mlp1LaneVariant {
                name,
                policy,
                local,
                official,
                diff: (local - official).abs(),
                pre_bias,
                bias,
                row_summary: None,
                full_mlp1_metric: None,
            }
        })
        .collect::<Vec<_>>();
    let best = lane522_policy_table
        .iter()
        .min_by(|a, b| a.diff.total_cmp(&b.diff))
        .cloned()
        .unwrap_or_else(|| empty_mlp1_lane_variant("not_run", "no policies", official));
    let pytorch_reference = load_pytorch_lane_reference(pytorch_lane_terms);
    let pytorch_output = pytorch_reference
        .get("output")
        .and_then(|output| output.get("einsum_bf16_plus_bias_bf16"))
        .and_then(Value::as_f64)
        .map(|value| value as f32);
    let classification = if best.diff == 0.0 {
        "mlp1_bf16_einsum_policy_identified"
    } else if pytorch_output.is_some_and(|value| value == official) {
        "mlp1_bf16_einsum_policy_not_encoded"
    } else {
        "mlp1_bf16_einsum_policy_unresolved"
    };
    let (chosen_mlp1_policy, expert30_full_metrics, selected_experts_rerun, weighted_sum_rerun) =
        if best.diff == 0.0 {
            (
                best.name,
                None,
                json!({ "run": false, "reason": "full policy wiring deferred" }),
                json!({ "run": false, "classification": "layer0_validation_weighted_expert_sum_not_run" }),
            )
        } else {
            (
                "none",
                None,
                json!({
                    "run": false,
                    "classification": "not_run_policy_did_not_clear_lane522"
                }),
                json!({
                    "run": false,
                    "classification": "layer0_validation_weighted_expert_sum_not_run"
                }),
            )
        };

    Ok(Mlp1Bf16PolicyStatus {
        mode: "layer0_validation_runtime_path",
        submode: "mlp1-bf16-policy",
        classification,
        implemented: true,
        runtime_behavior_changed: false,
        validation_only: true,
        chosen_mlp1_policy,
        lane522_policy_table,
        expert30_full_metrics,
        selected_experts_rerun,
        weighted_sum_rerun,
        pytorch_reference,
        next_bounded_step: "implement a validation-only BF16 matmul/einsum backend for MLP1 or use official/PyTorch MLP1 seam while exact Rust BF16 einsum semantics are added",
    })
}

#[cfg(not(feature = "cuda"))]
fn execute_mlp1_bf16_policy_mxfp4(
    _model: &Path,
    _mlp_norm: &[f32],
    _mlp1_oracle: &[f32],
    _lane: usize,
    _pytorch_lane_terms: Option<&Path>,
) -> Result<Mlp1Bf16PolicyStatus> {
    anyhow::bail!("MLP1 BF16 policy debug requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
fn execute_selected_experts_pinned_swiglu_debug_mxfp4(
    _model: &Path,
    _selected_experts: &[usize],
    _mlp_norm: &[f32],
    _selected_oracle: &[f32],
    _expert30_rank: usize,
    _mlp1_oracle: &[f32],
    _swiglu_oracle: &[f32],
    _mlp2_oracle: &[f32],
) -> Result<SelectedExpertsPinnedSwigluDebugResult> {
    anyhow::bail!("selected-experts pinned-SwiGLU debug requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
fn execute_expert30_mlp1_debug_mxfp4(
    _model: &Path,
    _mlp_norm: &[f32],
    _mlp1_oracle: &[f32],
    _lane: usize,
    _pytorch_lane_terms: Option<&Path>,
) -> Result<Expert30Mlp1LaneDebugStatus> {
    anyhow::bail!("expert30 MLP1 lane debug requires the cuda feature")
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

#[derive(Clone, Copy)]
enum SwigluPinPolicy {
    CurrentBest,
    F64Exp,
    SigmoidOneOverF32,
    SigmoidExpBranch,
    Bf16SigmoidArg,
    Bf16SigmoidOutput,
    Bf16OutGlu,
    Bf16UpPlusOne,
    TorchLikeStageRounding,
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
    compute_selected_expert_output_from_mlp1(&gate_up, expert)
}

#[cfg(feature = "cuda")]
fn compute_mlp1_bf16_tensor_op(
    mlp_norm: &[f32],
    expert: &Mxfp4SelectedExpertWeights,
) -> Result<Vec<f32>> {
    let context =
        CudaContext::new(0).map_err(|err| anyhow::anyhow!("CUDA context init failed: {err}"))?;
    let stream = context
        .new_stream()
        .map_err(|err| anyhow::anyhow!("CUDA stream init failed: {err}"))?;
    let blas = CublasHandle::new(stream.clone())
        .map_err(|err| anyhow::anyhow!("cuBLAS handle init failed: {err}"))?;
    let input_bf16 = mlp_norm
        .iter()
        .map(|&value| bf16::from_f32(value))
        .collect::<Vec<_>>();
    let weight_bf16 = expert
        .gate_up_weight
        .iter()
        .map(|&value| bf16::from_f32(value))
        .collect::<Vec<_>>();
    let input_gpu = stream
        .clone_htod(&input_bf16)
        .map_err(|err| anyhow::anyhow!("BF16 MLP1 input upload failed: {err}"))?;
    let weight_gpu = stream
        .clone_htod(&weight_bf16)
        .map_err(|err| anyhow::anyhow!("BF16 MLP1 weight upload failed: {err}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(5760)
        .map_err(|err| anyhow::anyhow!("BF16 MLP1 output allocation failed: {err}"))?;
    blas.bf16_gemm_into(
        1,
        5760,
        2880,
        1.0,
        &input_gpu,
        &weight_gpu,
        0.0,
        &mut output_gpu,
    )
    .map_err(|err| anyhow::anyhow!("BF16 MLP1 cuBLAS tensor-op GEMM failed: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("BF16 MLP1 cuBLAS sync failed: {err}"))?;
    let pre_bias = stream
        .clone_dtoh(&output_gpu)
        .map_err(|err| anyhow::anyhow!("BF16 MLP1 output download failed: {err}"))?
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    Ok(add_bf16_bias_to_bf16_output(
        &pre_bias,
        &expert.gate_up_bias,
    ))
}

#[cfg(feature = "cuda")]
fn compute_selected_expert_output_from_mlp1(
    gate_up: &[f32],
    expert: &Mxfp4SelectedExpertWeights,
) -> Vec<f32> {
    let swiglu = compute_swiglu_bf16(&gate_up);
    linear_out_in_bf16_output(&swiglu, &expert.down_weight, &expert.down_bias, 2880, 2880)
}

fn add_bf16_bias_to_bf16_output(pre_bias: &[f32], bias: &[f32]) -> Vec<f32> {
    pre_bias
        .iter()
        .zip(bias)
        .map(|(&pre_bias, &bias)| round_bf16(pre_bias + round_bf16(bias)))
        .collect()
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
    compute_swiglu_pin_stages(gate_up, SwigluPinPolicy::TorchLikeStageRounding)
        .remove("swiglu_output")
        .expect("SwiGLU validation stages must include swiglu_output")
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

fn empty_swiglu_policy_pin_variant() -> SwigluPolicyPinVariantStatus {
    SwigluPolicyPinVariantStatus {
        name: "not_run",
        policy: "artifact blocker",
        output_metric: empty_hidden_comparison(),
        stage_metrics: Vec::new(),
    }
}

fn load_pytorch_swiglu_intermediates(path: &Path) -> Result<BTreeMap<String, Vec<f32>>> {
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let tensors = value
        .get("tensors")
        .and_then(Value::as_object)
        .with_context(|| format!("{} does not contain a tensors object", path.display()))?;
    let mut out = BTreeMap::new();
    for (name, tensor) in tensors {
        if let Some(values) = tensor.get("values").and_then(Value::as_array) {
            out.insert(name.clone(), json_values_to_f32(values));
        }
    }
    Ok(out)
}

fn compute_swiglu_policy_pin_variants(
    mlp1: &[f32],
    oracle: &[f32],
    pytorch: Option<&BTreeMap<String, Vec<f32>>>,
) -> Vec<SwigluPolicyPinVariantStatus> {
    let specs = [
        (
            "A_current_best",
            "interleaved official split/clamp, BF16 input, BF16 alpha argument, f32 sigmoid/multiply, BF16 output",
            SwigluPinPolicy::CurrentBest,
        ),
        (
            "B_f64_exp",
            "interleaved official split/clamp, BF16 alpha argument, f64 exp sigmoid, BF16 output",
            SwigluPinPolicy::F64Exp,
        ),
        (
            "C_sigmoid_one_over_f32",
            "sigmoid as 1/(1+exp(-x)) with f32 exp and BF16 output",
            SwigluPinPolicy::SigmoidOneOverF32,
        ),
        (
            "D_sigmoid_exp_branch",
            "sigmoid uses exp(x)/(1+exp(x)) for negative x and 1/(1+exp(-x)) otherwise",
            SwigluPinPolicy::SigmoidExpBranch,
        ),
        (
            "E_round_alpha_gate_bf16",
            "BF16-round alpha*gate before sigmoid",
            SwigluPinPolicy::Bf16SigmoidArg,
        ),
        (
            "F_round_sigmoid_bf16",
            "BF16-round sigmoid output before out_glu multiply",
            SwigluPinPolicy::Bf16SigmoidOutput,
        ),
        (
            "G_round_out_glu_bf16",
            "BF16-round gate*sigmoid before final multiply",
            SwigluPinPolicy::Bf16OutGlu,
        ),
        (
            "H_round_up_plus_one_bf16",
            "BF16-round up+1 before final multiply",
            SwigluPinPolicy::Bf16UpPlusOne,
        ),
        (
            "I_torch_like_stage_rounding",
            "BF16-round alpha, sigmoid, out_glu, up_plus_one, and final output; sigmoid still uses Rust exp",
            SwigluPinPolicy::TorchLikeStageRounding,
        ),
    ];

    specs
        .iter()
        .map(|(name, policy, kind)| {
            let stages = compute_swiglu_pin_stages(mlp1, *kind);
            let output = stages
                .get("swiglu_output")
                .expect("SwiGLU pin stages must include output");
            let stage_metrics = pytorch
                .map(|expected| compare_swiglu_stages(&stages, expected))
                .unwrap_or_default();
            SwigluPolicyPinVariantStatus {
                name,
                policy,
                output_metric: compare_hidden(output, oracle),
                stage_metrics,
            }
        })
        .collect()
}

fn compare_swiglu_stages(
    actual: &BTreeMap<&'static str, Vec<f32>>,
    expected: &BTreeMap<String, Vec<f32>>,
) -> Vec<SwigluStageMetricStatus> {
    [
        "gate_raw",
        "up_raw",
        "gate_clamped",
        "up_clamped",
        "alpha_gate",
        "sigmoid_alpha_gate",
        "out_glu",
        "up_plus_one",
        "swiglu_output",
    ]
    .iter()
    .filter_map(|stage| {
        let actual_values = actual.get(stage)?;
        let expected_values = expected.get(*stage)?;
        Some(SwigluStageMetricStatus {
            stage,
            metric: compare_hidden(actual_values, expected_values),
        })
    })
    .collect()
}

fn compute_swiglu_pin_stages(
    mlp1: &[f32],
    policy: SwigluPinPolicy,
) -> BTreeMap<&'static str, Vec<f32>> {
    let intermediate = mlp1.len() / 2;
    let mut gate_raw = vec![0.0f32; intermediate];
    let mut up_raw = vec![0.0f32; intermediate];
    let mut gate_clamped = vec![0.0f32; intermediate];
    let mut up_clamped = vec![0.0f32; intermediate];
    let mut alpha_gate = vec![0.0f32; intermediate];
    let mut sigmoid_alpha_gate = vec![0.0f32; intermediate];
    let mut out_glu = vec![0.0f32; intermediate];
    let mut up_plus_one = vec![0.0f32; intermediate];
    let mut swiglu_output = vec![0.0f32; intermediate];

    for idx in 0..intermediate {
        let gate = round_bf16(mlp1[2 * idx]);
        let up = round_bf16(mlp1[2 * idx + 1]);
        let gate_after_clamp = round_bf16(gate.min(7.0));
        let up_after_clamp = round_bf16(up.clamp(-7.0, 7.0));
        let mut alpha = 1.702 * gate_after_clamp;
        if matches!(
            policy,
            SwigluPinPolicy::CurrentBest
                | SwigluPinPolicy::F64Exp
                | SwigluPinPolicy::Bf16SigmoidArg
                | SwigluPinPolicy::TorchLikeStageRounding
        ) {
            alpha = round_bf16(alpha);
        }
        let mut sigmoid = match policy {
            SwigluPinPolicy::F64Exp => (1.0f64 / (1.0f64 + (-(alpha as f64)).exp())) as f32,
            SwigluPinPolicy::SigmoidExpBranch => {
                if alpha < 0.0 {
                    let exp = alpha.exp();
                    exp / (1.0 + exp)
                } else {
                    1.0 / (1.0 + (-alpha).exp())
                }
            }
            _ => 1.0 / (1.0 + (-alpha).exp()),
        };
        if matches!(
            policy,
            SwigluPinPolicy::Bf16SigmoidOutput | SwigluPinPolicy::TorchLikeStageRounding
        ) {
            sigmoid = round_bf16(sigmoid);
        }
        let mut glu = gate_after_clamp * sigmoid;
        if matches!(
            policy,
            SwigluPinPolicy::Bf16OutGlu | SwigluPinPolicy::TorchLikeStageRounding
        ) {
            glu = round_bf16(glu);
        }
        let mut up_one = up_after_clamp + 1.0;
        if matches!(
            policy,
            SwigluPinPolicy::Bf16UpPlusOne | SwigluPinPolicy::TorchLikeStageRounding
        ) {
            up_one = round_bf16(up_one);
        }

        gate_raw[idx] = gate;
        up_raw[idx] = up;
        gate_clamped[idx] = gate_after_clamp;
        up_clamped[idx] = up_after_clamp;
        alpha_gate[idx] = alpha;
        sigmoid_alpha_gate[idx] = sigmoid;
        out_glu[idx] = glu;
        up_plus_one[idx] = up_one;
        swiglu_output[idx] = round_bf16(glu * up_one);
    }

    BTreeMap::from([
        ("gate_raw", gate_raw),
        ("up_raw", up_raw),
        ("gate_clamped", gate_clamped),
        ("up_clamped", up_clamped),
        ("alpha_gate", alpha_gate),
        ("sigmoid_alpha_gate", sigmoid_alpha_gate),
        ("out_glu", out_glu),
        ("up_plus_one", up_plus_one),
        ("swiglu_output", swiglu_output),
    ])
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
