use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use gpt_oss_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::worker::gpu_worker::GpuWorker;
use gpt_oss_engine::{RuntimeMode, SequenceData, SequenceGroupMetadata, WorkerConfig};
use gpt_oss_model_runner::gpu_runner::{
    build_bf16_gemm_invocation_record, Bf16GemmInvocationRecord, Layer0KRopeDebugCapture,
    Layer0QGemmTrace, Layer0QkvInternalTrace, Layer0QkvTrace,
};
use gpt_oss_model_runner::model_loader::dtype::DType;
use gpt_oss_model_runner::model_loader::load_model_weights;
use gpt_oss_model_runner::model_loader::weights::{MockGpuAllocator, WeightTensor};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_subscriber::EnvFilter;

const BF16_DENSE_QKV_ENV: &str = "GPT_OSS_LAYER0_BF16_DENSE_QKV";
const DEFAULT_LOCAL_RESIDUAL_INPUT_ARTIFACT: &str =
    ".live/runtime-forward-first-block-20260423/developer-message.runner-layer0-residual-input.json";
const DEFAULT_LOCAL_ATTN_NORM_ARTIFACT: &str =
    ".live/runtime-forward-layer0-attn-bisect-20260423/developer-message.runner-layer0-attn-norm-output.json";
const DEFAULT_OFFICIAL_POST_ATTENTION_RESIDUAL_ARTIFACT: &str =
    "/tmp/pinned-prompt-parity-official-reference-20260423/developer-message.official-layer0-post-attention-residual.cpu.json";
const DEFAULT_OFFICIAL_TRANSFORMER_OUTPUT_ARTIFACT: &str =
    "/tmp/pinned-prompt-parity-official-reference-20260423/developer-message.official-layer0-transformer-output.cpu.json";
const DEFAULT_CANDIDATE_QKV_ARTIFACT: &str =
    ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/developer-message.runner-layer0-qkv-projection-output.json";
const DEFAULT_CANDIDATE_POST_ATTENTION_RESIDUAL_ARTIFACT: &str =
    ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/developer-message.runner-layer0-post-attention-residual.json";
const DEFAULT_CANDIDATE_INTERNAL_TRACE_ARTIFACT: &str =
    ".live/runtime-forward-layer0-qkv-bf16-internal-20260423/developer-message.runner-layer0-qkv-bf16-internal-trace.json";
const DEFAULT_CANDIDATE_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/developer-message.runner-layer0-qkv-bf16-candidate-status.json";
const DEFAULT_Q_GEMM_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-status.json";
const DEFAULT_Q_GEMM_HELPER_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-helper-status.json";
const DEFAULT_Q_GEMM_MICROPROOF_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-microproof-status.json";
const DEFAULT_Q_GEMM_HELPER_CONTRACT_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-helper-contract-status.json";
const DEFAULT_Q_GEMM_LIVE_VS_ISOLATED_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-live-vs-isolated-status.json";
const DEFAULT_Q_GEMM_PRE_HELPER_PLUMBING_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-pre-helper-plumbing-status.json";
const DEFAULT_Q_GEMM_HELPER_INVOCATION_OUTPUT_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-q-gemm-bf16-20260423/developer-message.runner-layer0-q-gemm-bf16-helper-invocation-output-status.json";
const DEFAULT_K_GEMM_PRE_BIAS_STATUS_ARTIFACT: &str =
    ".live/runtime-forward-layer0-k-pre-bias-20260423/developer-message.runner-layer0-k-projection-pre-bias-status.json";
const DEFAULT_K_GEMM_HELPER_PATH_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-gemm-bf16-20260423/developer-message.runner-layer0-k-gemm-bf16-helper-path-status.json";
const DEFAULT_K_GEMM_FUSED_SPLIT_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-gemm-bf16-20260423/developer-message.runner-layer0-k-gemm-bf16-fused-split-status.json";
const DEFAULT_K_GEMM_FUSED_QKV_READOUT_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-gemm-bf16-20260423/developer-message.runner-layer0-k-gemm-bf16-fused-qkv-readout-status.json";
const DEFAULT_K_GEMM_FUSED_K_READOUT_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-gemm-bf16-20260423/developer-message.runner-layer0-k-gemm-bf16-fused-k-readout-status.json";
const DEFAULT_K_CONSUMPTION_ROPE_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-post-rope-pre-cache.json";
const DEFAULT_K_ROPE_CONVENTION_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-rope-convention-status.json";
const DEFAULT_K_ROPE_IMPLEMENTATION_STATUS_OUTPUT: &str =
    ".live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-rope-implementation-status.json";
const DEFAULT_OFFICIAL_K_POST_ROPE_PRE_CACHE_ARTIFACT: &str =
    "/tmp/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-k-post-rope-pre-cache.cpu.json";

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Mode {
    CandidateStatus,
    QGemmStatus,
    QGemmHelperStatus,
    QGemmMicroproofStatus,
    QGemmHelperContractStatus,
    QGemmLiveVsIsolatedStatus,
    QGemmPreHelperPlumbingStatus,
    QGemmHelperInvocationOutputStatus,
    KGemmHelperPathStatus,
    KGemmFusedSplitStatus,
    KGemmFusedQkvReadoutStatus,
    KGemmFusedKReadoutStatus,
    KConsumptionRopeStatus,
    KRopeConventionStatus,
    KRopeImplementationStatus,
}

impl Mode {
    fn default_output(self) -> PathBuf {
        match self {
            Self::CandidateStatus => PathBuf::from(DEFAULT_CANDIDATE_STATUS_OUTPUT),
            Self::QGemmStatus => PathBuf::from(DEFAULT_Q_GEMM_STATUS_OUTPUT),
            Self::QGemmHelperStatus => PathBuf::from(DEFAULT_Q_GEMM_HELPER_STATUS_OUTPUT),
            Self::QGemmMicroproofStatus => PathBuf::from(DEFAULT_Q_GEMM_MICROPROOF_STATUS_OUTPUT),
            Self::QGemmHelperContractStatus => {
                PathBuf::from(DEFAULT_Q_GEMM_HELPER_CONTRACT_STATUS_OUTPUT)
            }
            Self::QGemmLiveVsIsolatedStatus => {
                PathBuf::from(DEFAULT_Q_GEMM_LIVE_VS_ISOLATED_STATUS_OUTPUT)
            }
            Self::QGemmPreHelperPlumbingStatus => {
                PathBuf::from(DEFAULT_Q_GEMM_PRE_HELPER_PLUMBING_STATUS_OUTPUT)
            }
            Self::QGemmHelperInvocationOutputStatus => {
                PathBuf::from(DEFAULT_Q_GEMM_HELPER_INVOCATION_OUTPUT_STATUS_OUTPUT)
            }
            Self::KGemmHelperPathStatus => PathBuf::from(DEFAULT_K_GEMM_HELPER_PATH_STATUS_OUTPUT),
            Self::KGemmFusedSplitStatus => PathBuf::from(DEFAULT_K_GEMM_FUSED_SPLIT_STATUS_OUTPUT),
            Self::KGemmFusedQkvReadoutStatus => {
                PathBuf::from(DEFAULT_K_GEMM_FUSED_QKV_READOUT_STATUS_OUTPUT)
            }
            Self::KGemmFusedKReadoutStatus => {
                PathBuf::from(DEFAULT_K_GEMM_FUSED_K_READOUT_STATUS_OUTPUT)
            }
            Self::KConsumptionRopeStatus => PathBuf::from(DEFAULT_K_CONSUMPTION_ROPE_STATUS_OUTPUT),
            Self::KRopeConventionStatus => PathBuf::from(DEFAULT_K_ROPE_CONVENTION_STATUS_OUTPUT),
            Self::KRopeImplementationStatus => {
                PathBuf::from(DEFAULT_K_ROPE_IMPLEMENTATION_STATUS_OUTPUT)
            }
        }
    }
}

#[derive(Debug, Parser, Clone)]
#[command(about = "Exact-case runtime status for the layer-0 BF16 dense-QKV candidate")]
struct Cli {
    #[arg(long, value_enum, default_value_t = Mode::CandidateStatus)]
    mode: Mode,

    #[arg(
        long,
        default_value = DEFAULT_LOCAL_RESIDUAL_INPUT_ARTIFACT
    )]
    local_residual_input_artifact: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_LOCAL_ATTN_NORM_ARTIFACT
    )]
    local_attn_norm_artifact: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_OFFICIAL_POST_ATTENTION_RESIDUAL_ARTIFACT
    )]
    official_post_attention_residual_artifact: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_OFFICIAL_TRANSFORMER_OUTPUT_ARTIFACT
    )]
    official_transformer_output_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_CANDIDATE_QKV_ARTIFACT)]
    candidate_qkv_artifact: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_CANDIDATE_POST_ATTENTION_RESIDUAL_ARTIFACT
    )]
    candidate_post_attention_residual_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_CANDIDATE_INTERNAL_TRACE_ARTIFACT)]
    candidate_internal_trace_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_Q_GEMM_HELPER_STATUS_OUTPUT)]
    q_gemm_helper_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_Q_GEMM_MICROPROOF_STATUS_OUTPUT)]
    q_gemm_microproof_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_Q_GEMM_HELPER_CONTRACT_STATUS_OUTPUT)]
    q_gemm_helper_contract_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_Q_GEMM_LIVE_VS_ISOLATED_STATUS_OUTPUT)]
    q_gemm_live_vs_isolated_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_GEMM_PRE_BIAS_STATUS_ARTIFACT)]
    k_projection_pre_bias_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_GEMM_HELPER_PATH_STATUS_OUTPUT)]
    k_gemm_helper_path_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_GEMM_FUSED_SPLIT_STATUS_OUTPUT)]
    k_gemm_fused_split_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_GEMM_FUSED_QKV_READOUT_STATUS_OUTPUT)]
    k_gemm_fused_qkv_readout_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_GEMM_FUSED_K_READOUT_STATUS_OUTPUT)]
    k_gemm_fused_k_readout_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_CONSUMPTION_ROPE_STATUS_OUTPUT)]
    k_consumption_rope_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_K_ROPE_CONVENTION_STATUS_OUTPUT)]
    k_rope_convention_status_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_OFFICIAL_K_POST_ROPE_PRE_CACHE_ARTIFACT)]
    official_k_post_rope_pre_cache_artifact: PathBuf,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ArtifactProvenance {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    capture_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    reference_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    authority_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    visible_devices: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    max_model_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    gpu_memory_utilization: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    prompt_renderer: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct ArtifactCase {
    id: String,
    input_token_ids: Vec<TokenId>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    final_token_hidden_f32: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct PinnedArtifact {
    boundary: String,
    #[serde(default)]
    layer_idx: Option<usize>,
    provenance: ArtifactProvenance,
    cases: Vec<ArtifactCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CompareMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    matched: bool,
}

#[derive(Debug, Serialize)]
struct TrialSummary {
    enabled: bool,
    post_attention_residual_vs_official: CompareMetrics,
    transformer_output_vs_official: CompareMetrics,
    materially_improves_post_attention_residual: bool,
    materially_improves_transformer_output: bool,
}

#[derive(Debug, Serialize)]
struct StatusSummary {
    schema_version: String,
    provenance: StatusProvenance,
    exact_case: ExactCaseSummary,
    baseline: TrialSummary,
    candidate: TrialSummary,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct QGemmStageDescriptor {
    stage: &'static str,
    bucket: &'static str,
}

#[derive(Debug, Serialize)]
struct QGemmStatusProvenance {
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    oracle_checkpoint_dir: PathBuf,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct QGemmStatusSummary {
    schema_version: String,
    provenance: QGemmStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    stage_bucket_mapping: Vec<QGemmStageDescriptor>,
    compared_stages: Vec<&'static str>,
    activation_preparation: CompareMetrics,
    weight_preparation: CompareMetrics,
    raw_q_gemm_output: CompareMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    immediate_post_gemm_cast_writeback: Option<CompareMetrics>,
    post_gemm_cast_distinct: bool,
    earliest_q_gemm_divergence_stage: String,
    first_mismatch_consistent_with: String,
    runtime_forward_bf16_q_gemm_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct QGemmHelperStatusProvenance {
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct QGemmHelperStatusSummary {
    schema_version: String,
    provenance: QGemmHelperStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    fused_vs_reference: CompareMetrics,
    standalone_vs_reference: CompareMetrics,
    fused_vs_standalone: CompareMetrics,
    classification: String,
    runtime_forward_bf16_q_gemm_helper_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize)]
struct ExistingQGemmHelperStatusArtifact {
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    classification: String,
}

#[derive(Debug, Deserialize)]
struct ExistingKProjectionPreBiasCase {
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    kv_dim: usize,
}

#[derive(Debug, Deserialize)]
struct ExistingKProjectionPreBiasCombinedQkvContext {
    combined_qkv_conclusion: String,
}

#[derive(Debug, Deserialize)]
struct ExistingKProjectionPreBiasStatusArtifact {
    case: ExistingKProjectionPreBiasCase,
    existing_combined_qkv_context: ExistingKProjectionPreBiasCombinedQkvContext,
    conclusion: String,
}

#[derive(Debug, Deserialize)]
struct ExistingQGemmMicroproofStatusArtifact {
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    classification: String,
    isolated_reproduces_live_standalone_capture: bool,
    helper_vs_reference: CompareMetrics,
}

#[derive(Debug, Serialize)]
struct QGemmMicroproofStatusProvenance {
    q_gemm_helper_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct QGemmMicroproofStatusSummary {
    schema_version: String,
    provenance: QGemmMicroproofStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    isolated_reproduces_live_standalone_capture: bool,
    helper_vs_reference: CompareMetrics,
    activation_row_major_input_hypothesis_vs_reference: CompareMetrics,
    q_weight_column_major_hypothesis_vs_reference: CompareMetrics,
    classification: String,
    runtime_forward_bf16_q_gemm_microproof_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct QGemmHelperContractStatusProvenance {
    q_gemm_microproof_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Clone, Copy)]
enum RawScalarDtype {
    Bf16,
    F32,
}

impl RawScalarDtype {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RawBf16GemmContract {
    trans_weight: cudarc::cublas::sys::cublasOperation_t,
    trans_input: cudarc::cublas::sys::cublasOperation_t,
    lda: i32,
    ldb: i32,
    ldc: i32,
    alpha_beta_dtype: RawScalarDtype,
}

#[derive(Debug, Deserialize, Serialize)]
struct Bf16GemmContractRecord {
    api: String,
    m: usize,
    n: usize,
    k: usize,
    input_dtype: String,
    weight_dtype: String,
    compute_type: String,
    output_dtype: String,
    alpha_value: f32,
    beta_value: f32,
    alpha_beta_dtype: String,
    trans_weight: String,
    trans_input: String,
    lda: i32,
    ldb: i32,
    ldc: i32,
    algo: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct QGemmHelperContractStatusSummary {
    schema_version: String,
    provenance: QGemmHelperContractStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    helper_contract: Bf16GemmContractRecord,
    alternate_contract: Bf16GemmContractRecord,
    raw_exact_contract_reproduces_helper: bool,
    isolated_reproduces_live_standalone_capture: bool,
    helper_vs_reference: CompareMetrics,
    alternate_contract_vs_reference: CompareMetrics,
    best_explained_by: String,
    runtime_forward_bf16_helper_contract_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct QGemmLiveVsIsolatedStatusProvenance {
    q_gemm_helper_contract_status_artifact_path: PathBuf,
    q_gemm_helper_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct QGemmLiveVsIsolatedStatusSummary {
    schema_version: String,
    provenance: QGemmLiveVsIsolatedStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    bf16_close_max_abs_diff: f32,
    isolated_helper_replay_vs_cpu_reference: CompareMetrics,
    live_standalone_vs_isolated_helper_replay: CompareMetrics,
    live_fused_q_vs_live_standalone: CompareMetrics,
    live_fused_q_vs_isolated_helper_replay: CompareMetrics,
    classification: String,
    runtime_forward_live_vs_isolated_bf16_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct QGemmPreHelperPlumbingStatusProvenance {
    q_gemm_live_vs_isolated_status_artifact_path: PathBuf,
    q_gemm_helper_contract_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct QGemmPreHelperPlumbingStatusSummary {
    schema_version: String,
    provenance: QGemmPreHelperPlumbingStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    standalone_activation_vs_isolated_replay: CompareMetrics,
    fused_activation_vs_standalone: CompareMetrics,
    standalone_weight_vs_isolated_replay: CompareMetrics,
    fused_q_slice_weight_vs_standalone: CompareMetrics,
    earliest_input_plumbing_divergence: String,
    runtime_forward_bf16_pre_helper_plumbing_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct QGemmHelperInvocationOutputStatusProvenance {
    q_gemm_helper_status_artifact_path: PathBuf,
    q_gemm_helper_contract_status_artifact_path: PathBuf,
    q_gemm_live_vs_isolated_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct QGemmHelperInvocationOutputStatusSummary {
    schema_version: String,
    provenance: QGemmHelperInvocationOutputStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    candidate_branch_taken: bool,
    live_helper_invocation_state: Bf16GemmInvocationRecord,
    isolated_helper_invocation_state: Bf16GemmInvocationRecord,
    invocation_state_match: bool,
    raw_output_match: bool,
    live_output_vs_isolated_replay: CompareMetrics,
    classification: String,
    runtime_forward_bf16_helper_invocation_output_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmHelperPathStatusProvenance {
    k_projection_pre_bias_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmHelperPathStatusSummary {
    schema_version: String,
    provenance: KGemmHelperPathStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    candidate_branch_taken: bool,
    isolated_helper_replay_vs_cpu_reference: CompareMetrics,
    live_standalone_vs_isolated_helper_replay: CompareMetrics,
    live_fused_k_vs_live_standalone: CompareMetrics,
    live_fused_k_vs_isolated_helper_replay: CompareMetrics,
    classification: String,
    runtime_forward_bf16_k_helper_path_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmFusedSplitStatusProvenance {
    k_projection_pre_bias_status_artifact_path: PathBuf,
    k_gemm_helper_path_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmFusedSplitStatusSummary {
    schema_version: String,
    provenance: KGemmFusedSplitStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    candidate_branch_taken: bool,
    fused_pre_bias_vs_standalone_pre_bias: CompareMetrics,
    fused_post_bias_vs_standalone_post_bias: CompareMetrics,
    downstream_fused_k_vs_standalone_post_bias: CompareMetrics,
    classification: String,
    runtime_forward_bf16_fused_k_split_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmFusedQkvReadoutStatusProvenance {
    k_gemm_helper_path_status_artifact_path: PathBuf,
    k_gemm_fused_split_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmFusedQkvReadoutStatusSummary {
    schema_version: String,
    provenance: KGemmFusedQkvReadoutStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    candidate_branch_taken: bool,
    combined_buffer_match: bool,
    combined_buffer_vs_expected: CompareMetrics,
    downstream_fused_k_vs_expected_k: CompareMetrics,
    classification: String,
    runtime_forward_bf16_fused_qkv_k_readout_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmFusedKReadoutStatusProvenance {
    k_gemm_helper_path_status_artifact_path: PathBuf,
    k_gemm_fused_split_status_artifact_path: PathBuf,
    k_gemm_fused_qkv_readout_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct KGemmFusedKReadoutCandidate {
    candidate_readout_name: String,
    vs_expected_k: CompareMetrics,
}

#[derive(Debug, Deserialize, Serialize)]
struct KGemmFusedKReadoutStatusSummary {
    schema_version: String,
    provenance: KGemmFusedKReadoutStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    candidate_branch_taken: bool,
    candidate_readouts: Vec<KGemmFusedKReadoutCandidate>,
    best_candidate_readout_name: String,
    any_candidate_materially_collapses_mismatch: bool,
    classification: String,
    runtime_forward_bf16_fused_k_readout_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KConsumptionRopeStatusProvenance {
    k_gemm_fused_k_readout_status_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KConsumptionRopeStatusSummary {
    schema_version: String,
    provenance: KConsumptionRopeStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    num_tokens: usize,
    candidate_branch_taken: bool,
    boundary: String,
    pre_rope_k_readout_vs_expected: CompareMetrics,
    k_post_rope_pre_cache_f16_bits: Vec<u16>,
    k_post_rope_pre_cache_f32: Vec<f32>,
    runtime_forward_k_consumption_seam_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KRopeConventionStatusProvenance {
    k_consumption_rope_status_artifact_path: PathBuf,
    official_k_post_rope_pre_cache_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct KRopeConventionVariant {
    variant_name: String,
    vs_official_post_rope_k: CompareMetrics,
    materially_collapses_mismatch: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct KRopeConventionStatusSummary {
    schema_version: String,
    provenance: KRopeConventionStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    num_tokens: usize,
    official_len: usize,
    local_runtime_len: usize,
    pre_rope_k_readout_vs_expected: CompareMetrics,
    baseline_local_runtime_vs_official: CompareMetrics,
    variants: Vec<KRopeConventionVariant>,
    best_variant_name: String,
    any_variant_materially_collapses_mismatch: bool,
    classification: String,
    runtime_forward_k_rope_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct KRopeImplementationStatusProvenance {
    k_rope_convention_status_artifact_path: PathBuf,
    k_consumption_rope_status_artifact_path: PathBuf,
    official_k_post_rope_pre_cache_artifact_path: PathBuf,
    local_attn_norm_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct KRopeImplementationCapturedState {
    final_token_index: usize,
    effective_position_id: i32,
    effective_position_modulo: usize,
    runtime_position_ids: Vec<i32>,
    position_metadata_match: bool,
    head_dim: usize,
    half_dim: usize,
    num_kv_heads: usize,
    final_token_k_pre_rope_len: usize,
    final_token_k_post_rope_len: usize,
    live_runtime_cos_row_f32: Vec<f32>,
    live_runtime_sin_row_f32: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct KRopeImplementationStatusSummary {
    schema_version: String,
    provenance: KRopeImplementationStatusProvenance,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    num_tokens: usize,
    captured_state: KRopeImplementationCapturedState,
    live_runtime_factors_vs_host_generated_factors: CompareMetrics,
    host_recompute_from_live_factors_vs_live_runtime_post_rope: CompareMetrics,
    host_recompute_from_live_factors_vs_official_post_rope: CompareMetrics,
    host_recompute_from_host_generated_factors_vs_live_runtime_post_rope: CompareMetrics,
    host_recompute_from_host_generated_factors_vs_official_post_rope: CompareMetrics,
    classification: String,
    runtime_forward_k_rope_implementation_status_now: String,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct StatusProvenance {
    local_residual_input_artifact_path: PathBuf,
    official_post_attention_residual_artifact_path: PathBuf,
    official_transformer_output_artifact_path: PathBuf,
    model_root: String,
    visible_devices: String,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    candidate_env_var: String,
}

#[derive(Debug, Serialize)]
struct ExactCaseSummary {
    case_id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    boundary: String,
    layer_idx: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct Layer0Capture {
    qkv_projection_output: Vec<f32>,
    post_attention_residual: Vec<f32>,
    transformer_output: Vec<f32>,
    q_dim: usize,
    kv_dim: usize,
    qkv_dim: usize,
    qkv_internal_trace: Option<Layer0QkvInternalTrace>,
}

#[derive(Debug, Clone)]
struct Layer0QGemmShadow {
    activation_row: Vec<f32>,
    q_weight: Vec<f32>,
    raw_q_output: Vec<f32>,
    post_gemm_cast_writeback: Option<Vec<f32>>,
}

struct LoaderModelConfig {
    model_name: String,
    hidden_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    max_model_len: usize,
}

impl gpt_oss_core::config::ModelConfig for LoaderModelConfig {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_model_len(&self) -> usize {
        self.max_model_len
    }
}

struct LoaderParallelConfig;

impl gpt_oss_core::config::ParallelConfig for LoaderParallelConfig {
    fn tensor_parallel_size(&self) -> usize {
        1
    }

    fn pipeline_parallel_size(&self) -> usize {
        1
    }
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateArtifactProvenance {
    model: String,
    capture_source: String,
    reference_kind: String,
    authority_level: String,
    visible_devices: String,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    prompt_renderer: String,
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateQkvCase {
    id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    vector_size: usize,
    q_dim: usize,
    kv_dim: usize,
    qkv_dim: usize,
    final_token_hidden_f32: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateResidualCase {
    id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    final_token_hidden_f32: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateInternalQkvCase {
    id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    vector_size: usize,
    branch_taken: bool,
    q_dim: usize,
    kv_dim: usize,
    qkv_dim: usize,
    fused_qkv_pre_bias_f32: Vec<f32>,
    fused_qkv_post_bias_f32: Vec<f32>,
    packed_qkv_output_f32: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateQkvArtifact {
    schema_version: String,
    suite_id: String,
    boundary: String,
    layer_idx: usize,
    provenance: PinnedPromptIntermediateArtifactProvenance,
    cases: Vec<PinnedPromptIntermediateQkvCase>,
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateResidualArtifact {
    schema_version: String,
    suite_id: String,
    boundary: String,
    layer_idx: usize,
    provenance: PinnedPromptIntermediateArtifactProvenance,
    cases: Vec<PinnedPromptIntermediateResidualCase>,
}

#[derive(Debug, Serialize)]
struct PinnedPromptIntermediateInternalQkvArtifact {
    schema_version: String,
    suite_id: String,
    boundary: String,
    layer_idx: usize,
    provenance: PinnedPromptIntermediateArtifactProvenance,
    cases: Vec<PinnedPromptIntermediateInternalQkvCase>,
}

struct EnvFlagGuard {
    name: &'static str,
    previous: Option<OsString>,
}

impl EnvFlagGuard {
    fn set(name: &'static str, enabled: bool) -> Self {
        let previous = env::var_os(name);
        if enabled {
            env::set_var(name, "1");
        } else {
            env::remove_var(name);
        }
        Self { name, previous }
    }
}

impl Drop for EnvFlagGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            env::set_var(self.name, previous);
        } else {
            env::remove_var(self.name);
        }
    }
}

fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

fn load_json_artifact(path: &Path) -> Result<PinnedArtifact> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read artifact {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse artifact {}", path.display()))
}

fn load_existing_q_gemm_helper_status_artifact(
    path: &Path,
) -> Result<ExistingQGemmHelperStatusArtifact> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read helper-status artifact {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse helper-status artifact {}", path.display()))
}

fn load_existing_q_gemm_microproof_status_artifact(
    path: &Path,
) -> Result<ExistingQGemmMicroproofStatusArtifact> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read microproof-status artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse microproof-status artifact {}",
            path.display()
        )
    })
}

fn load_existing_q_gemm_helper_contract_status_artifact(
    path: &Path,
) -> Result<QGemmHelperContractStatusSummary> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read helper-contract artifact {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse helper-contract artifact {}",
            path.display()
        )
    })
}

fn load_existing_q_gemm_live_vs_isolated_status_artifact(
    path: &Path,
) -> Result<QGemmLiveVsIsolatedStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read live-vs-isolated artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse live-vs-isolated artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_projection_pre_bias_status_artifact(
    path: &Path,
) -> Result<ExistingKProjectionPreBiasStatusArtifact> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-projection-pre-bias artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-projection-pre-bias artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_gemm_helper_path_status_artifact(
    path: &Path,
) -> Result<KGemmHelperPathStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-gemm-helper-path artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-gemm-helper-path artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_gemm_fused_split_status_artifact(
    path: &Path,
) -> Result<KGemmFusedSplitStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-gemm-fused-split artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-gemm-fused-split artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_gemm_fused_qkv_readout_status_artifact(
    path: &Path,
) -> Result<KGemmFusedQkvReadoutStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-gemm-fused-qkv-readout artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-gemm-fused-qkv-readout artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_gemm_fused_k_readout_status_artifact(
    path: &Path,
) -> Result<KGemmFusedKReadoutStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-gemm-fused-k-readout artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-gemm-fused-k-readout artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_consumption_rope_status_artifact(
    path: &Path,
) -> Result<KConsumptionRopeStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-consumption-rope artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-consumption-rope artifact {}",
            path.display()
        )
    })
}

fn load_existing_k_rope_convention_status_artifact(
    path: &Path,
) -> Result<KRopeConventionStatusSummary> {
    let raw = std::fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read k-rope-convention artifact {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse k-rope-convention artifact {}",
            path.display()
        )
    })
}

fn load_single_case_artifact(path: &Path) -> Result<(PinnedArtifact, ArtifactCase)> {
    let artifact = load_json_artifact(path)?;
    if artifact.cases.len() != 1 {
        bail!(
            "{} must contain exactly one case, found {}",
            path.display(),
            artifact.cases.len()
        );
    }
    let case = artifact
        .cases
        .first()
        .cloned()
        .context("missing exact case despite len() check")?;
    Ok((artifact, case))
}

fn validate_exact_case_artifact(
    artifact: &PinnedArtifact,
    case: &ArtifactCase,
    expected_boundary: &str,
    expected_case_id: &str,
    expected_tokens: &[TokenId],
) -> Result<()> {
    if artifact.boundary != expected_boundary {
        bail!(
            "expected boundary {}, found {}",
            expected_boundary,
            artifact.boundary
        );
    }
    if case.id != expected_case_id {
        bail!("expected case id {}, found {}", expected_case_id, case.id);
    }
    if case.input_token_ids != expected_tokens {
        bail!(
            "{} input token ids do not match the exact smoke case",
            artifact.boundary
        );
    }
    Ok(())
}

fn compare_vectors(lhs: &[f32], rhs: &[f32]) -> Result<CompareMetrics> {
    if lhs.len() != rhs.len() {
        bail!(
            "vector length mismatch: left {} vs right {}",
            lhs.len(),
            rhs.len()
        );
    }
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f32;
    for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
        let abs_diff = (lhs_value - rhs_value).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        sum_abs_diff += abs_diff;
    }
    Ok(CompareMetrics {
        max_abs_diff,
        mean_abs_diff: if lhs.is_empty() {
            0.0
        } else {
            sum_abs_diff / lhs.len() as f32
        },
        matched: lhs == rhs,
    })
}

#[derive(Debug, Clone, Copy)]
enum RopeTablePrecision {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, Copy)]
enum RopeOutputPrecision {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, Copy)]
enum RopePairing {
    SplitHalf,
    Interleaved,
}

#[derive(Debug, Clone, Copy)]
struct KRopeVariantSpec {
    name: &'static str,
    table_precision: RopeTablePrecision,
    output_precision: RopeOutputPrecision,
    pairing: RopePairing,
    position_offset: isize,
}

fn round_rope_table_value(value: f32, precision: RopeTablePrecision) -> f32 {
    match precision {
        RopeTablePrecision::F32 => value,
        RopeTablePrecision::F16 => f16::from_f32(value).to_f32(),
        RopeTablePrecision::Bf16 => bf16::from_f32(value).to_f32(),
    }
}

fn round_rope_output_value(value: f32, precision: RopeOutputPrecision) -> f32 {
    match precision {
        RopeOutputPrecision::F32 => value,
        RopeOutputPrecision::F16 => f16::from_f32(value).to_f32(),
        RopeOutputPrecision::Bf16 => bf16::from_f32(value).to_f32(),
    }
}

fn build_host_rope_tables(
    config: &WorkerConfig,
    max_pos: usize,
    precision: RopeTablePrecision,
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.head_dim;
    let half_dim = head_dim / 2;
    let mut inv_freq = vec![0.0f32; half_dim];
    let mut concentration = 1.0f32;
    let use_yarn = matches!(config.rope_scaling_type.as_deref(), Some("yarn"))
        && config.rope_scaling_factor > 1.0;

    if use_yarn {
        concentration = 0.1 * config.rope_scaling_factor.ln() + 1.0;
        let d_half = head_dim as f32 / 2.0;
        let base_ln = config.rope_theta.ln();
        let context_len = config.initial_context_length.max(1) as f32;
        let mut low = d_half
            * (context_len / (config.rope_ntk_beta * 2.0 * std::f32::consts::PI)).ln()
            / base_ln;
        let mut high = d_half
            * (context_len / (config.rope_ntk_alpha * 2.0 * std::f32::consts::PI)).ln()
            / base_ln;
        if config.rope_scaling_truncate {
            low = low.floor();
            high = high.ceil();
        }
        if (high - low).abs() < f32::EPSILON {
            high = low + 0.001;
        }
        for (i, inv) in inv_freq.iter_mut().enumerate() {
            let freq = config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
            let extrapolation = 1.0 / freq;
            let interpolation = 1.0 / (config.rope_scaling_factor * freq);
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            *inv = interpolation * (1.0 - mask) + extrapolation * mask;
        }
    } else {
        for (i, inv) in inv_freq.iter_mut().enumerate() {
            *inv = 1.0 / config.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
        }
    }

    let mut cos_table = vec![0.0f32; max_pos * half_dim];
    let mut sin_table = vec![0.0f32; max_pos * half_dim];
    for pos in 0..max_pos {
        for (i, freq) in inv_freq.iter().enumerate() {
            let theta = pos as f32 * freq;
            cos_table[pos * half_dim + i] =
                round_rope_table_value(theta.cos() * concentration, precision);
            sin_table[pos * half_dim + i] =
                round_rope_table_value(theta.sin() * concentration, precision);
        }
    }

    (cos_table, sin_table)
}

fn position_with_offset(position: usize, offset: isize, max_pos: usize) -> usize {
    if max_pos == 0 {
        return 0;
    }
    let shifted = if offset.is_negative() {
        position.saturating_sub(offset.unsigned_abs())
    } else {
        position.saturating_add(offset as usize)
    };
    shifted.min(max_pos - 1)
}

fn apply_host_k_rope_variant(
    pre_rope_k: &[f32],
    config: &WorkerConfig,
    num_tokens: usize,
    kv_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    spec: KRopeVariantSpec,
) -> Result<Vec<f32>> {
    let head_dim = config.head_dim;
    let half_dim = head_dim / 2;
    if head_dim == 0 || half_dim == 0 {
        bail!("invalid RoPE head_dim {}", head_dim);
    }
    if kv_dim % head_dim != 0 {
        bail!(
            "kv_dim {} is not divisible by head_dim {}",
            kv_dim,
            head_dim
        );
    }
    let num_kv_heads = kv_dim / head_dim;
    let expected_len = num_tokens
        .checked_mul(kv_dim)
        .context("host K RoPE input length overflow")?;
    if pre_rope_k.len() != expected_len {
        bail!(
            "host K RoPE input length mismatch: expected {}, found {}",
            expected_len,
            pre_rope_k.len()
        );
    }
    if cos_table.len() != sin_table.len() || cos_table.len() % half_dim != 0 {
        bail!("invalid host RoPE table lengths");
    }
    let max_pos = cos_table.len() / half_dim;

    let mut output = pre_rope_k.to_vec();
    for token_idx in 0..num_tokens {
        let pos = position_with_offset(token_idx, spec.position_offset, max_pos);
        for head_idx in 0..num_kv_heads {
            let base = (token_idx * num_kv_heads + head_idx) * head_dim;
            for pair_idx in 0..half_dim {
                let cos_val = cos_table[pos * half_dim + pair_idx];
                let sin_val = sin_table[pos * half_dim + pair_idx];
                let (i0, i1) = match spec.pairing {
                    RopePairing::SplitHalf => (base + pair_idx, base + half_dim + pair_idx),
                    RopePairing::Interleaved => (base + 2 * pair_idx, base + 2 * pair_idx + 1),
                };
                let x0 = pre_rope_k[i0];
                let x1 = pre_rope_k[i1];
                output[i0] =
                    round_rope_output_value(x0 * cos_val - x1 * sin_val, spec.output_precision);
                output[i1] =
                    round_rope_output_value(x0 * sin_val + x1 * cos_val, spec.output_precision);
            }
        }
    }
    Ok(output)
}

fn duplicate_rope_half_row(row: &[f32], head_dim: usize) -> Result<Vec<f32>> {
    if row.len() * 2 != head_dim {
        bail!(
            "RoPE half-row length {} does not match head_dim {}",
            row.len(),
            head_dim
        );
    }
    let mut full_row = Vec::with_capacity(head_dim);
    full_row.extend_from_slice(row);
    full_row.extend_from_slice(row);
    Ok(full_row)
}

fn combined_factor_row(cos_row: &[f32], sin_row: &[f32]) -> Result<Vec<f32>> {
    if cos_row.len() != sin_row.len() {
        bail!(
            "RoPE factor row length mismatch: cos {} vs sin {}",
            cos_row.len(),
            sin_row.len()
        );
    }
    let mut combined = Vec::with_capacity(cos_row.len() + sin_row.len());
    combined.extend_from_slice(cos_row);
    combined.extend_from_slice(sin_row);
    Ok(combined)
}

fn host_generated_rope_rows_for_position(
    config: &WorkerConfig,
    position: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let head_dim = config.head_dim;
    let half_dim = head_dim / 2;
    let (cos_table, sin_table) =
        build_host_rope_tables(config, position.saturating_add(1), RopeTablePrecision::F32);
    let start = position
        .checked_mul(half_dim)
        .context("host RoPE row start overflow")?;
    let end = start
        .checked_add(half_dim)
        .context("host RoPE row end overflow")?;
    Ok((
        duplicate_rope_half_row(&cos_table[start..end], head_dim)?,
        duplicate_rope_half_row(&sin_table[start..end], head_dim)?,
    ))
}

fn apply_final_token_k_rope_from_factor_rows(
    pre_rope_k: &[f32],
    cos_row: &[f32],
    sin_row: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if cos_row.len() != head_dim || sin_row.len() != head_dim {
        bail!(
            "factor row length mismatch for head_dim {}: cos {}, sin {}",
            head_dim,
            cos_row.len(),
            sin_row.len()
        );
    }
    let kv_dim = num_kv_heads
        .checked_mul(head_dim)
        .context("final-token K kv_dim overflow")?;
    if pre_rope_k.len() != kv_dim {
        bail!(
            "final-token pre-RoPE K length mismatch: expected {}, found {}",
            kv_dim,
            pre_rope_k.len()
        );
    }
    let half_dim = head_dim / 2;
    let mut output = pre_rope_k.to_vec();
    for head_idx in 0..num_kv_heads {
        let base = head_idx * head_dim;
        for pair_idx in 0..half_dim {
            let i0 = base + pair_idx;
            let i1 = base + half_dim + pair_idx;
            let cos_val = cos_row[pair_idx];
            let sin_val = sin_row[pair_idx];
            let x0 = pre_rope_k[i0];
            let x1 = pre_rope_k[i1];
            output[i0] = f16::from_f32(x0 * cos_val - x1 * sin_val).to_f32();
            output[i1] = f16::from_f32(x0 * sin_val + x1 * cos_val).to_f32();
        }
    }
    Ok(output)
}

fn k_rope_variant_specs() -> Vec<KRopeVariantSpec> {
    vec![
        KRopeVariantSpec {
            name: "runtime_split_half_f32_table_f16_output",
            table_precision: RopeTablePrecision::F32,
            output_precision: RopeOutputPrecision::F16,
            pairing: RopePairing::SplitHalf,
            position_offset: 0,
        },
        KRopeVariantSpec {
            name: "split_half_f32_table_f32_output",
            table_precision: RopeTablePrecision::F32,
            output_precision: RopeOutputPrecision::F32,
            pairing: RopePairing::SplitHalf,
            position_offset: 0,
        },
        KRopeVariantSpec {
            name: "split_half_f16_table_f16_output",
            table_precision: RopeTablePrecision::F16,
            output_precision: RopeOutputPrecision::F16,
            pairing: RopePairing::SplitHalf,
            position_offset: 0,
        },
        KRopeVariantSpec {
            name: "split_half_bf16_table_bf16_output",
            table_precision: RopeTablePrecision::Bf16,
            output_precision: RopeOutputPrecision::Bf16,
            pairing: RopePairing::SplitHalf,
            position_offset: 0,
        },
        KRopeVariantSpec {
            name: "interleaved_f32_table_f16_output",
            table_precision: RopeTablePrecision::F32,
            output_precision: RopeOutputPrecision::F16,
            pairing: RopePairing::Interleaved,
            position_offset: 0,
        },
        KRopeVariantSpec {
            name: "position_plus_one_split_half_f32_table_f16_output",
            table_precision: RopeTablePrecision::F32,
            output_precision: RopeOutputPrecision::F16,
            pairing: RopePairing::SplitHalf,
            position_offset: 1,
        },
        KRopeVariantSpec {
            name: "position_minus_one_split_half_f32_table_f16_output",
            table_precision: RopeTablePrecision::F32,
            output_precision: RopeOutputPrecision::F16,
            pairing: RopePairing::SplitHalf,
            position_offset: -1,
        },
    ]
}

fn bf16_bits_to_f32_slice(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|value| bf16::from_bits(*value).to_f32())
        .collect()
}

fn bf16_bits_to_bf16_slice(values: &[u16]) -> Vec<bf16> {
    values.iter().map(|value| bf16::from_bits(*value)).collect()
}

fn bf16_slice_to_bits(values: &[bf16]) -> Vec<u16> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn compare_bf16_bit_slices(lhs: &[u16], rhs: &[u16]) -> Result<CompareMetrics> {
    if lhs.len() != rhs.len() {
        bail!(
            "bf16 bit-slice length mismatch: left {} vs right {}",
            lhs.len(),
            rhs.len()
        );
    }
    let lhs_f32 = bf16_bits_to_f32_slice(lhs);
    let rhs_f32 = bf16_bits_to_f32_slice(rhs);
    compare_vectors(&lhs_f32, &rhs_f32).map(|mut metrics| {
        metrics.matched = lhs == rhs;
        metrics
    })
}

fn last_token_vector(values: &[f32], token_dim: usize) -> Result<Vec<f32>> {
    if token_dim == 0 {
        bail!("token dimension must be non-zero");
    }
    if values.len() < token_dim {
        bail!(
            "vector length {} is smaller than token dimension {}",
            values.len(),
            token_dim
        );
    }
    if values.len() % token_dim != 0 {
        bail!(
            "vector length {} is not a whole number of tokens with token dimension {}",
            values.len(),
            token_dim
        );
    }
    Ok(values[values.len() - token_dim..].to_vec())
}

fn materially_improves(baseline: &CompareMetrics, candidate: &CompareMetrics) -> bool {
    candidate.mean_abs_diff <= baseline.mean_abs_diff * 0.5
        || candidate.max_abs_diff <= baseline.max_abs_diff * 0.5
}

fn q_gemm_stage_bucket_mapping() -> Vec<QGemmStageDescriptor> {
    vec![
        QGemmStageDescriptor {
            stage: "activation_preparation",
            bucket: "activation cast/preparation",
        },
        QGemmStageDescriptor {
            stage: "weight_preparation",
            bucket: "weight cast/preparation",
        },
        QGemmStageDescriptor {
            stage: "raw_q_gemm_output",
            bucket: "GEMM helper math/output",
        },
        QGemmStageDescriptor {
            stage: "immediate_post_gemm_cast_writeback",
            bucket: "immediate post-GEMM cast/writeback",
        },
    ]
}

fn q_gemm_bucket_for_stage(stage: &str) -> &'static str {
    q_gemm_stage_bucket_mapping()
        .into_iter()
        .find(|entry| entry.stage == stage)
        .map(|entry| entry.bucket)
        .unwrap_or("matched_through_q_gemm_probe")
}

fn q_gemm_next_bounded_step(stage: &str) -> String {
    match stage {
        "activation_preparation" => {
            "re-check the final-token BF16 activation handoff from the exact layer-0 attention-norm artifact into the live fused Q GEMM".into()
        }
        "weight_preparation" => {
            "audit how the live fused BF16 Q slice is packed from the fused QKV weight before changing GEMM math".into()
        }
        "raw_q_gemm_output" => {
            "inspect the live BF16 GEMM helper math/output on the fused Q slice before touching downstream bias or attention code".into()
        }
        "immediate_post_gemm_cast_writeback" => {
            "trace the immediate post-GEMM cast/writeback step on the live BF16 fused QKV branch before touching bias".into()
        }
        _ => {
            "advance the bounded probe past Q GEMM and inspect the next live pre-bias/Q-bias boundary".into()
        }
    }
}

fn resolve_oracle_checkpoint_dir(path: &Path) -> PathBuf {
    let original_dir = path.join("original");
    if original_dir.is_dir() {
        original_dir
    } else {
        path.to_path_buf()
    }
}

fn round_f32_slice_to_bf16(values: &[f32]) -> Vec<bf16> {
    values.iter().map(|value| bf16::from_f32(*value)).collect()
}

fn bf16_slice_to_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn decode_weight_tensor_to_f32(tensor: &WeightTensor) -> Result<Vec<f32>> {
    let bytes = tensor.data().as_bytes();
    match tensor.dtype() {
        DType::F32 => {
            if bytes.len() % 4 != 0 {
                bail!(
                    "tensor {} f32 byte length {} is not divisible by 4",
                    tensor.name(),
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk length checked")))
                .collect())
        }
        DType::F16 => {
            if bytes.len() % 2 != 0 {
                bail!(
                    "tensor {} f16 byte length {} is not divisible by 2",
                    tensor.name(),
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    f16::from_bits(u16::from_le_bytes(
                        chunk.try_into().expect("chunk length checked"),
                    ))
                    .to_f32()
                })
                .collect())
        }
        DType::BF16 => {
            if bytes.len() % 2 != 0 {
                bail!(
                    "tensor {} bf16 byte length {} is not divisible by 2",
                    tensor.name(),
                    bytes.len()
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    bf16::from_bits(u16::from_le_bytes(
                        chunk.try_into().expect("chunk length checked"),
                    ))
                    .to_f32()
                })
                .collect())
        }
        dtype => bail!(
            "unsupported q-weight dtype {} for tensor {}",
            dtype,
            tensor.name()
        ),
    }
}

fn load_layer0_q_weight_shadow_bf16(
    model_root: &Path,
    q_dim: usize,
    hidden_size: usize,
) -> Result<Vec<bf16>> {
    let resolved_root = resolve_oracle_checkpoint_dir(model_root);
    let config = build_worker_config(&resolved_root, usize::MAX, 0.75)?;
    let loader_model_config = LoaderModelConfig {
        model_name: config.model_name.clone(),
        hidden_size: config.hidden_size,
        num_layers: config.num_layers,
        num_attention_heads: config.num_attention_heads,
        num_kv_heads: config.num_kv_heads,
        vocab_size: config.vocab_size,
        max_model_len: config.max_model_len,
    };
    let loader_parallel_config = LoaderParallelConfig;
    let weights = load_model_weights(
        &resolved_root,
        &loader_model_config,
        &loader_parallel_config,
        0,
        &MockGpuAllocator,
    )?;
    let weight_tensor = weights
        .get("model.layers.0.self_attn.q_proj.weight")
        .or_else(|| weights.get("block.0.attn.q_proj.weight"))
        .or_else(|| weights.get("block.0.attn.qkv.weight"))
        .context("missing layer-0 q_proj/qkv weight in oracle checkpoint")?;
    let decoded = decode_weight_tensor_to_f32(weight_tensor)?;
    let q_weight_f32 = match weight_tensor.shape() {
        [rows, cols] if *rows == q_dim && *cols == hidden_size => decoded,
        [rows, cols] if *rows > q_dim && *cols == hidden_size => {
            decoded[..q_dim * hidden_size].to_vec()
        }
        shape => {
            bail!(
                "unexpected oracle attention weight shape {:?}, expected [{}, {}] or [qkv_dim, {}]",
                shape,
                q_dim,
                hidden_size,
                hidden_size
            )
        }
    };
    if q_weight_f32.len() != q_dim * hidden_size {
        bail!(
            "decoded q weight length {} does not match expected {}",
            q_weight_f32.len(),
            q_dim * hidden_size
        );
    }
    Ok(round_f32_slice_to_bf16(&q_weight_f32))
}

fn cpu_bf16_q_matvec(
    activation_row: &[bf16],
    q_weight: &[bf16],
    q_dim: usize,
    hidden_size: usize,
) -> Result<Vec<bf16>> {
    if activation_row.len() != hidden_size {
        bail!(
            "activation row length {} does not match hidden size {}",
            activation_row.len(),
            hidden_size
        );
    }
    if q_weight.len() != q_dim * hidden_size {
        bail!(
            "q weight length {} does not match expected {}",
            q_weight.len(),
            q_dim * hidden_size
        );
    }
    let mut output = Vec::with_capacity(q_dim);
    for row_idx in 0..q_dim {
        let row = &q_weight[row_idx * hidden_size..(row_idx + 1) * hidden_size];
        let mut sum = 0.0f32;
        for (activation, weight) in activation_row.iter().zip(row.iter()) {
            sum += activation.to_f32() * weight.to_f32();
        }
        output.push(bf16::from_f32(sum));
    }
    Ok(output)
}

fn cpu_bf16_row_major_gemm(
    activations: &[bf16],
    weight: &[bf16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<bf16>> {
    if activations.len() != m * k {
        bail!(
            "activation buffer length {} does not match expected {}",
            activations.len(),
            m * k
        );
    }
    if weight.len() != n * k {
        bail!(
            "weight buffer length {} does not match expected {}",
            weight.len(),
            n * k
        );
    }

    let mut output = Vec::with_capacity(m * n);
    for row_idx in 0..m {
        let activation_row = &activations[row_idx * k..(row_idx + 1) * k];
        for col_idx in 0..n {
            let weight_row = &weight[col_idx * k..(col_idx + 1) * k];
            let mut sum = 0.0f32;
            for (activation, weight_value) in activation_row.iter().zip(weight_row.iter()) {
                sum += activation.to_f32() * weight_value.to_f32();
            }
            output.push(bf16::from_f32(sum));
        }
    }
    Ok(output)
}

fn build_worker(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
) -> Result<GpuWorker> {
    let config = build_worker_config(model_path, max_model_len, gpu_memory_utilization)?;
    let mut worker = GpuWorker::new(config)?;
    worker.init_model(model_path)?;
    worker.load_weights(model_path)?;
    let (num_gpu_blocks, num_cpu_blocks) =
        worker.profile_num_available_blocks(gpu_memory_utilization)?;
    worker.init_cache(num_gpu_blocks, num_cpu_blocks)?;
    Ok(worker)
}

fn build_worker_config(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
) -> Result<WorkerConfig> {
    let config_path = model_path.join("config.json");
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let get_usize = |key: &str, default: usize| -> usize {
        value
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        value
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(default)
    };
    let get_bool = |key: &str, default: bool| -> bool {
        value.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    };
    let layer_types = value
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|vals| {
            vals.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let hidden_size = get_usize("hidden_size", 2880);
    let num_attention_heads = get_usize("num_attention_heads", 64);
    let head_dim = get_usize(
        "head_dim",
        hidden_size.checked_div(num_attention_heads).unwrap_or(64),
    );
    let rope_scaling = value.get("rope_scaling");
    let rope_scaling_type = rope_scaling
        .and_then(|v| v.get("rope_type"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let rope_scaling_factor = rope_scaling
        .and_then(|v| v.get("factor"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0);
    let rope_ntk_alpha = rope_scaling
        .and_then(|v| v.get("beta_slow"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0);
    let rope_ntk_beta = rope_scaling
        .and_then(|v| v.get("beta_fast"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(32.0);
    let rope_scaling_truncate = rope_scaling
        .and_then(|v| v.get("truncate"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let initial_context_length = get_usize(
        "initial_context_length",
        rope_scaling
            .and_then(|v| v.get("original_max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(max_model_len),
    );

    Ok(WorkerConfig {
        model_name: model_path.display().to_string(),
        runtime_mode: RuntimeMode::Trusted,
        device_id: 0,
        num_layers: get_usize("num_hidden_layers", 24),
        num_kv_heads: get_usize("num_key_value_heads", 8),
        head_dim,
        hidden_size,
        num_attention_heads,
        intermediate_size: get_usize("intermediate_size", 2880),
        vocab_size: get_usize("vocab_size", 201088),
        max_model_len: max_model_len.min(get_usize("max_position_embeddings", max_model_len)),
        initial_context_length,
        rms_norm_eps: get_f32("rms_norm_eps", 1e-5),
        block_size: 16,
        gpu_memory_utilization,
        rank: 0,
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        architecture: value
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("GptOssForCausalLM")
            .to_string(),
        dtype: Dtype::Float16,
        rope_theta: get_f32("rope_theta", 150000.0),
        rope_scaling_type,
        rope_scaling_factor,
        rope_ntk_alpha,
        rope_ntk_beta,
        rope_scaling_truncate,
        partial_rotary_factor: get_f32("partial_rotary_factor", 1.0),
        attn_logit_softcapping: get_f32("attn_logit_softcapping", 0.0),
        attention_bias: get_bool("attention_bias", false),
        sliding_window: value
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        layer_types,
        num_local_experts: get_usize("num_local_experts", 32),
        num_experts_per_tok: get_usize("num_experts_per_tok", 4),
        kv_cache_dtype: "auto".into(),
        enable_prefix_caching: false,
    })
}

fn build_single_sequence_metadata(prompt_token_ids: &[TokenId]) -> Vec<SequenceGroupMetadata> {
    let seq_id = SequenceId(1);
    let total_len = prompt_token_ids.len();
    let num_blocks = total_len.max(1).div_ceil(16);
    let mut seq_data = HashMap::new();
    seq_data.insert(
        seq_id,
        SequenceData {
            prompt_token_ids: prompt_token_ids.to_vec(),
            output_token_ids: Vec::new(),
            cumulative_logprob: 0.0,
        },
    );
    let mut block_tables = HashMap::new();
    block_tables.insert(seq_id, (0..num_blocks).map(|i| BlockId(i as u32)).collect());

    vec![SequenceGroupMetadata {
        request_id: RequestId(1),
        is_prompt: true,
        seq_data,
        sampling_params: SamplingParams {
            temperature: 0.0,
            max_tokens: 1,
            seed: Some(0),
            ..SamplingParams::default()
        },
        block_tables,
    }]
}

fn capture_layer0_outputs(
    worker: &GpuWorker,
    prompt_token_ids: &[TokenId],
    capture_internal_trace: bool,
) -> Result<Layer0Capture> {
    let metadata = build_single_sequence_metadata(prompt_token_ids);
    let (qkv_projection_output, q_dim, kv_dim, qkv_dim, qkv_internal_trace) =
        if capture_internal_trace {
            let qkv = worker.debug_runner_prefill_layer0_qkv_trace(&metadata)?;
            let qkv_internal_trace =
                Some(worker.debug_runner_prefill_layer0_qkv_internal_trace(&metadata)?);
            (
                qkv.qkv_projection_output,
                qkv.q_dim,
                qkv.kv_dim,
                qkv.qkv_dim,
                qkv_internal_trace,
            )
        } else {
            (Vec::new(), 0, 0, 0, None)
        };
    let layer0 = worker.debug_runner_prefill_layer0_trace(&metadata)?;
    Ok(Layer0Capture {
        qkv_projection_output,
        post_attention_residual: layer0.post_attn_residual.clone(),
        transformer_output: layer0.layer_output.clone(),
        q_dim,
        kv_dim,
        qkv_dim,
        qkv_internal_trace,
    })
}

fn run_trial(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    prompt_token_ids: &[TokenId],
    candidate_enabled: bool,
    capture_internal_trace: bool,
) -> Result<Layer0Capture> {
    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, candidate_enabled);
    let worker = build_worker(model_path, max_model_len, gpu_memory_utilization)?;
    capture_layer0_outputs(&worker, prompt_token_ids, capture_internal_trace)
}

fn capture_layer0_q_gemm_trace(
    worker: &GpuWorker,
    prompt_token_ids: &[TokenId],
) -> Result<Layer0QGemmTrace> {
    let metadata = build_single_sequence_metadata(prompt_token_ids);
    Ok(worker.debug_runner_prefill_layer0_q_gemm_trace(&metadata)?)
}

fn capture_layer0_qkv_trace(
    worker: &GpuWorker,
    prompt_token_ids: &[TokenId],
) -> Result<Layer0QkvTrace> {
    let metadata = build_single_sequence_metadata(prompt_token_ids);
    Ok(worker.debug_runner_prefill_layer0_qkv_trace(&metadata)?)
}

fn run_q_gemm_trial(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    prompt_token_ids: &[TokenId],
) -> Result<Layer0QGemmTrace> {
    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_path, max_model_len, gpu_memory_utilization)?;
    capture_layer0_q_gemm_trace(&worker, prompt_token_ids)
}

fn next_bounded_step(candidate: &TrialSummary) -> String {
    if candidate.materially_improves_post_attention_residual
        && candidate.materially_improves_transformer_output
    {
        "treat this as the active surgical layer-0 BF16 dense-QKV candidate and test the exact final PPP output".into()
    } else {
        "revisit the remaining live-standalone vs isolated discrepancy in the BF16 Q GEMM helper path before broadening beyond layer 0".into()
    }
}

fn build_layer0_q_gemm_shadow(
    model_root: &Path,
    hidden_size: usize,
    q_dim: usize,
    activation_source: &[f32],
    post_gemm_cast_distinct: bool,
) -> Result<Layer0QGemmShadow> {
    let activation_row_bf16 = round_f32_slice_to_bf16(activation_source);
    let q_weight_bf16 = load_layer0_q_weight_shadow_bf16(model_root, q_dim, hidden_size)?;
    let raw_q_output_bf16 =
        cpu_bf16_q_matvec(&activation_row_bf16, &q_weight_bf16, q_dim, hidden_size)?;
    let post_gemm_cast_writeback = if post_gemm_cast_distinct {
        Some(
            raw_q_output_bf16
                .iter()
                .map(|value| f16::from_f32(value.to_f32()).to_f32())
                .collect(),
        )
    } else {
        None
    };
    Ok(Layer0QGemmShadow {
        activation_row: bf16_slice_to_f32(&activation_row_bf16),
        q_weight: bf16_slice_to_f32(&q_weight_bf16),
        raw_q_output: bf16_slice_to_f32(&raw_q_output_bf16),
        post_gemm_cast_writeback,
    })
}

fn build_live_q_gemm_reference(live_trace: &Layer0QGemmTrace) -> Result<Vec<f32>> {
    let activation_row_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_activation_bf16_bits);
    let q_weight_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_q_weight_bf16_bits);
    let reference_output = cpu_bf16_q_matvec(
        &activation_row_bf16,
        &q_weight_bf16,
        live_trace.q_dim,
        live_trace.hidden_size,
    )?;
    Ok(bf16_slice_to_f32(&reference_output))
}

fn load_and_validate_existing_q_gemm_helper_status(
    cli: &Cli,
    case: &ArtifactCase,
) -> Result<ExistingQGemmHelperStatusArtifact> {
    let artifact = load_existing_q_gemm_helper_status_artifact(&cli.q_gemm_helper_status_artifact)?;
    if artifact.case_id != case.id {
        bail!(
            "helper-status artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("helper-status artifact token ids do not match the exact smoke case");
    }
    if artifact.classification != "shared_bf16_gemm_helper" {
        bail!(
            "helper-status artifact classification changed: expected shared_bf16_gemm_helper, found {}",
            artifact.classification
        );
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_projection_pre_bias_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
) -> Result<ExistingKProjectionPreBiasStatusArtifact> {
    let artifact = load_existing_k_projection_pre_bias_status_artifact(
        &cli.k_projection_pre_bias_status_artifact,
    )?;
    if artifact.case.case_id != case.id {
        bail!(
            "k-projection-pre-bias artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case.case_id
        );
    }
    if artifact.case.input_token_ids != case.input_token_ids {
        bail!("k-projection-pre-bias artifact token ids do not match the exact smoke case");
    }
    if artifact.conclusion != "k_pre_bias_already_divergent" {
        bail!(
            "k-projection-pre-bias artifact conclusion changed: expected k_pre_bias_already_divergent, found {}",
            artifact.conclusion
        );
    }
    if artifact
        .existing_combined_qkv_context
        .combined_qkv_conclusion
        != "k_dominant"
    {
        bail!(
            "k-projection-pre-bias artifact combined qkv conclusion changed: expected k_dominant, found {}",
            artifact
                .existing_combined_qkv_context
                .combined_qkv_conclusion
        );
    }
    if artifact.case.hidden_size != exact_hidden_size {
        bail!(
            "k-projection-pre-bias artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.case.hidden_size
        );
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_gemm_helper_path_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<KGemmHelperPathStatusSummary> {
    let artifact =
        load_existing_k_gemm_helper_path_status_artifact(&cli.k_gemm_helper_path_status_artifact)?;
    if artifact.case_id != case.id {
        bail!(
            "k-gemm-helper-path artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("k-gemm-helper-path artifact token ids do not match the exact smoke case");
    }
    if artifact.hidden_size != exact_hidden_size {
        bail!(
            "k-gemm-helper-path artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.hidden_size
        );
    }
    if artifact.q_dim != q_dim {
        bail!(
            "k-gemm-helper-path artifact q_dim mismatch: expected {}, found {}",
            q_dim,
            artifact.q_dim
        );
    }
    if artifact.kv_dim != kv_dim {
        bail!(
            "k-gemm-helper-path artifact kv_dim mismatch: expected {}, found {}",
            kv_dim,
            artifact.kv_dim
        );
    }
    if !artifact.live_standalone_vs_isolated_helper_replay.matched {
        bail!("k-gemm-helper-path artifact no longer shows standalone K matching isolated replay");
    }
    if artifact.live_fused_k_vs_live_standalone.matched {
        bail!("k-gemm-helper-path artifact no longer shows fused K diverging from standalone K");
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_gemm_fused_split_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<KGemmFusedSplitStatusSummary> {
    let artifact =
        load_existing_k_gemm_fused_split_status_artifact(&cli.k_gemm_fused_split_status_artifact)?;
    if artifact.case_id != case.id {
        bail!(
            "k-gemm-fused-split artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("k-gemm-fused-split artifact token ids do not match the exact smoke case");
    }
    if artifact.hidden_size != exact_hidden_size {
        bail!(
            "k-gemm-fused-split artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.hidden_size
        );
    }
    if artifact.q_dim != q_dim {
        bail!(
            "k-gemm-fused-split artifact q_dim mismatch: expected {}, found {}",
            q_dim,
            artifact.q_dim
        );
    }
    if artifact.kv_dim != kv_dim {
        bail!(
            "k-gemm-fused-split artifact kv_dim mismatch: expected {}, found {}",
            kv_dim,
            artifact.kv_dim
        );
    }
    if !artifact.fused_pre_bias_vs_standalone_pre_bias.matched {
        bail!("k-gemm-fused-split artifact no longer shows fused K pre-bias matching standalone K");
    }
    if !artifact.fused_post_bias_vs_standalone_post_bias.matched {
        bail!(
            "k-gemm-fused-split artifact no longer shows fused K post-bias matching standalone K"
        );
    }
    if artifact.downstream_fused_k_vs_standalone_post_bias.matched {
        bail!("k-gemm-fused-split artifact no longer shows downstream fused K diverging");
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_gemm_fused_qkv_readout_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<KGemmFusedQkvReadoutStatusSummary> {
    let artifact = load_existing_k_gemm_fused_qkv_readout_status_artifact(
        &cli.k_gemm_fused_qkv_readout_status_artifact,
    )?;
    if artifact.case_id != case.id {
        bail!(
            "k-gemm-fused-qkv-readout artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("k-gemm-fused-qkv-readout artifact token ids do not match the exact smoke case");
    }
    if artifact.hidden_size != exact_hidden_size {
        bail!(
            "k-gemm-fused-qkv-readout artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.hidden_size
        );
    }
    if artifact.q_dim != q_dim {
        bail!(
            "k-gemm-fused-qkv-readout artifact q_dim mismatch: expected {}, found {}",
            q_dim,
            artifact.q_dim
        );
    }
    if artifact.kv_dim != kv_dim {
        bail!(
            "k-gemm-fused-qkv-readout artifact kv_dim mismatch: expected {}, found {}",
            kv_dim,
            artifact.kv_dim
        );
    }
    if !artifact.combined_buffer_match || !artifact.combined_buffer_vs_expected.matched {
        bail!("k-gemm-fused-qkv-readout artifact no longer shows combined QKV buffer matching");
    }
    if artifact.downstream_fused_k_vs_expected_k.matched {
        bail!("k-gemm-fused-qkv-readout artifact no longer shows downstream fused K diverging");
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_gemm_fused_k_readout_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<KGemmFusedKReadoutStatusSummary> {
    let artifact = load_existing_k_gemm_fused_k_readout_status_artifact(
        &cli.k_gemm_fused_k_readout_status_artifact,
    )?;
    if artifact.case_id != case.id {
        bail!(
            "k-gemm-fused-k-readout artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("k-gemm-fused-k-readout artifact token ids do not match the exact smoke case");
    }
    if artifact.hidden_size != exact_hidden_size {
        bail!(
            "k-gemm-fused-k-readout artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.hidden_size
        );
    }
    if artifact.q_dim != q_dim {
        bail!(
            "k-gemm-fused-k-readout artifact q_dim mismatch: expected {}, found {}",
            q_dim,
            artifact.q_dim
        );
    }
    if artifact.kv_dim != kv_dim {
        bail!(
            "k-gemm-fused-k-readout artifact kv_dim mismatch: expected {}, found {}",
            kv_dim,
            artifact.kv_dim
        );
    }
    if artifact.classification != "k_readout_fixed" {
        bail!(
            "k-gemm-fused-k-readout artifact classification changed: expected k_readout_fixed, found {}",
            artifact.classification
        );
    }
    let current_live = artifact
        .candidate_readouts
        .iter()
        .find(|candidate| candidate.candidate_readout_name == "current_live_fused_k_readout")
        .context("k-gemm-fused-k-readout artifact missing current_live_fused_k_readout")?;
    if !current_live.vs_expected_k.matched {
        bail!("k-gemm-fused-k-readout artifact no longer shows current live fused K matching");
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_consumption_rope_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    num_tokens: usize,
) -> Result<KConsumptionRopeStatusSummary> {
    let artifact =
        load_existing_k_consumption_rope_status_artifact(&cli.k_consumption_rope_status_artifact)?;
    if artifact.case_id != case.id {
        bail!(
            "k-consumption-rope artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("k-consumption-rope artifact token ids do not match the exact smoke case");
    }
    if artifact.hidden_size != exact_hidden_size {
        bail!(
            "k-consumption-rope artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.hidden_size
        );
    }
    if artifact.q_dim != q_dim {
        bail!(
            "k-consumption-rope artifact q_dim mismatch: expected {}, found {}",
            q_dim,
            artifact.q_dim
        );
    }
    if artifact.kv_dim != kv_dim {
        bail!(
            "k-consumption-rope artifact kv_dim mismatch: expected {}, found {}",
            kv_dim,
            artifact.kv_dim
        );
    }
    if artifact.num_tokens != num_tokens {
        bail!(
            "k-consumption-rope artifact num_tokens mismatch: expected {}, found {}",
            num_tokens,
            artifact.num_tokens
        );
    }
    if !artifact.candidate_branch_taken {
        bail!("k-consumption-rope artifact did not take the bf16 fused-qkv branch");
    }
    if artifact.boundary != "layer0_k_post_rope_pre_cache" {
        bail!(
            "k-consumption-rope artifact boundary changed: expected layer0_k_post_rope_pre_cache, found {}",
            artifact.boundary
        );
    }
    if !artifact.pre_rope_k_readout_vs_expected.matched {
        bail!("k-consumption-rope artifact no longer shows exact pre-RoPE K readout");
    }
    let expected_len = num_tokens
        .checked_mul(kv_dim)
        .context("k-consumption-rope expected K length overflow")?;
    if artifact.k_post_rope_pre_cache_f32.len() != expected_len {
        bail!(
            "k-consumption-rope f32 vector length mismatch: expected {}, found {}",
            expected_len,
            artifact.k_post_rope_pre_cache_f32.len()
        );
    }
    if artifact.k_post_rope_pre_cache_f16_bits.len() != expected_len {
        bail!(
            "k-consumption-rope f16 bit vector length mismatch: expected {}, found {}",
            expected_len,
            artifact.k_post_rope_pre_cache_f16_bits.len()
        );
    }
    Ok(artifact)
}

fn load_and_validate_existing_k_rope_convention_status(
    cli: &Cli,
    case: &ArtifactCase,
    exact_hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    num_tokens: usize,
) -> Result<KRopeConventionStatusSummary> {
    let artifact =
        load_existing_k_rope_convention_status_artifact(&cli.k_rope_convention_status_artifact)?;
    if artifact.case_id != case.id {
        bail!(
            "k-rope-convention artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("k-rope-convention artifact token ids do not match the exact smoke case");
    }
    if artifact.hidden_size != exact_hidden_size {
        bail!(
            "k-rope-convention artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            artifact.hidden_size
        );
    }
    if artifact.q_dim != q_dim {
        bail!(
            "k-rope-convention artifact q_dim mismatch: expected {}, found {}",
            q_dim,
            artifact.q_dim
        );
    }
    if artifact.kv_dim != kv_dim {
        bail!(
            "k-rope-convention artifact kv_dim mismatch: expected {}, found {}",
            kv_dim,
            artifact.kv_dim
        );
    }
    if artifact.num_tokens != num_tokens {
        bail!(
            "k-rope-convention artifact num_tokens mismatch: expected {}, found {}",
            num_tokens,
            artifact.num_tokens
        );
    }
    if artifact.classification != "runtime_rope_impl_needed" {
        bail!(
            "k-rope-convention artifact classification changed: expected runtime_rope_impl_needed, found {}",
            artifact.classification
        );
    }
    if artifact.any_variant_materially_collapses_mismatch {
        bail!("k-rope-convention artifact now has a material convention collapse");
    }
    Ok(artifact)
}

fn load_and_validate_existing_q_gemm_microproof_status(
    cli: &Cli,
    case: &ArtifactCase,
) -> Result<ExistingQGemmMicroproofStatusArtifact> {
    let artifact =
        load_existing_q_gemm_microproof_status_artifact(&cli.q_gemm_microproof_status_artifact)?;
    if artifact.case_id != case.id {
        bail!(
            "microproof-status artifact case id mismatch: expected {}, found {}",
            case.id,
            artifact.case_id
        );
    }
    if artifact.input_token_ids != case.input_token_ids {
        bail!("microproof-status artifact token ids do not match the exact smoke case");
    }
    if artifact.classification != "helper_itself_mismatch" {
        bail!(
            "microproof-status artifact classification changed: expected helper_itself_mismatch, found {}",
            artifact.classification
        );
    }
    if artifact.helper_vs_reference.matched {
        bail!("microproof-status artifact helper_vs_reference unexpectedly matches");
    }
    if artifact.isolated_reproduces_live_standalone_capture {
        bail!(
            "microproof-status artifact isolation/live-standalone reproduction unexpectedly changed to true"
        );
    }
    Ok(artifact)
}

fn run_raw_bf16_gemm_contract(
    blas: &CudaBlas,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    activation_row_bf16: &[bf16],
    q_weight_bf16: &[bf16],
    q_dim: usize,
    hidden_size: usize,
    contract: RawBf16GemmContract,
) -> Result<Vec<f32>> {
    use cudarc::cublas::sys::{
        cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        cublasStatus_t::CUBLAS_STATUS_SUCCESS, cudaDataType_t::CUDA_R_16BF,
    };

    if activation_row_bf16.len() != hidden_size {
        bail!(
            "activation row length {} does not match hidden size {}",
            activation_row_bf16.len(),
            hidden_size
        );
    }
    if q_weight_bf16.len() != q_dim * hidden_size {
        bail!(
            "q weight length {} does not match expected {}",
            q_weight_bf16.len(),
            q_dim * hidden_size
        );
    }

    let activation_gpu = stream
        .clone_htod(activation_row_bf16)
        .map_err(|e| anyhow::anyhow!("microproof activation htod failed: {e}"))?;
    let q_weight_gpu = stream
        .clone_htod(q_weight_bf16)
        .map_err(|e| anyhow::anyhow!("microproof q weight htod failed: {e}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(q_dim)
        .map_err(|e| anyhow::anyhow!("microproof output alloc failed: {e}"))?;

    {
        let (weight_ptr, _weight_guard) = DevicePtr::device_ptr(&q_weight_gpu, &stream);
        let (activation_ptr, _activation_guard) = DevicePtr::device_ptr(&activation_gpu, &stream);
        let (output_ptr, _output_guard) = DevicePtrMut::device_ptr_mut(&mut output_gpu, &stream);

        unsafe {
            let status = cudarc::cublas::sys::cublasGemmEx(
                *blas.handle(),
                contract.trans_weight,
                contract.trans_input,
                q_dim as i32,
                1,
                hidden_size as i32,
                match contract.alpha_beta_dtype {
                    RawScalarDtype::Bf16 => &bf16::ONE as *const bf16 as *const c_void,
                    RawScalarDtype::F32 => &1.0f32 as *const f32 as *const c_void,
                },
                weight_ptr as *const c_void,
                CUDA_R_16BF,
                contract.lda,
                activation_ptr as *const c_void,
                CUDA_R_16BF,
                contract.ldb,
                match contract.alpha_beta_dtype {
                    RawScalarDtype::Bf16 => &bf16::ZERO as *const bf16 as *const c_void,
                    RawScalarDtype::F32 => &0.0f32 as *const f32 as *const c_void,
                },
                output_ptr as *mut c_void,
                CUDA_R_16BF,
                contract.ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                bail!("raw helper-contract cublasGemmEx failed: {status:?}");
            }
        }
    }

    let output_host: Vec<bf16> = stream
        .clone_dtoh(&output_gpu)
        .map_err(|e| anyhow::anyhow!("raw helper-contract output dtoh failed: {e}"))?;
    Ok(output_host.iter().map(|value| value.to_f32()).collect())
}

fn cublas_op_name(op: cudarc::cublas::sys::cublasOperation_t) -> &'static str {
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T};

    match op {
        CUBLAS_OP_N => "CUBLAS_OP_N",
        CUBLAS_OP_T => "CUBLAS_OP_T",
        CUBLAS_OP_C => "CUBLAS_OP_C",
        _ => "CUBLAS_OP_UNKNOWN",
    }
}

fn record_bf16_gemm_contract(
    contract: RawBf16GemmContract,
    q_dim: usize,
    hidden_size: usize,
) -> Bf16GemmContractRecord {
    Bf16GemmContractRecord {
        api: "cublasGemmEx".into(),
        m: 1,
        n: q_dim,
        k: hidden_size,
        input_dtype: "CUDA_R_16BF".into(),
        weight_dtype: "CUDA_R_16BF".into(),
        compute_type: "CUBLAS_COMPUTE_32F".into(),
        output_dtype: "CUDA_R_16BF".into(),
        alpha_value: 1.0,
        beta_value: 0.0,
        alpha_beta_dtype: contract.alpha_beta_dtype.as_str().into(),
        trans_weight: cublas_op_name(contract.trans_weight).into(),
        trans_input: cublas_op_name(contract.trans_input).into(),
        lda: contract.lda,
        ldb: contract.ldb,
        ldc: contract.ldc,
        algo: "CUBLAS_GEMM_DEFAULT_TENSOR_OP".into(),
    }
}

fn collapse_rule_satisfied(
    helper_vs_reference: &CompareMetrics,
    hypothesis: &CompareMetrics,
) -> bool {
    if hypothesis.matched {
        return true;
    }
    helper_vs_reference.max_abs_diff > 0.0
        && helper_vs_reference.mean_abs_diff > 0.0
        && hypothesis.max_abs_diff <= helper_vs_reference.max_abs_diff * 0.1
        && hypothesis.mean_abs_diff <= helper_vs_reference.mean_abs_diff * 0.1
}

fn q_gemm_microproof_next_bounded_step(classification: &str) -> String {
    match classification {
        "helper_itself_mismatch" => {
            "inspect hgemm_bf16 math/output behavior itself, not runner integration or QKV packing".into()
        }
        "invocation_layout_mismatch_candidate" => {
            "build a surgical proof around the winning cuBLAS transpose/leading-dimension configuration, not the helper math".into()
        }
        "integration_only" => {
            "return to runner-side integration around how the exact activation and Q-weight inputs are fed into the helper".into()
        }
        other => format!("unexpected microproof classification {other}"),
    }
}

fn q_gemm_helper_contract_next_bounded_step(best_explained_by: &str) -> String {
    match best_explained_by {
        "alpha_beta_or_descriptor_configuration" => {
            "re-run the exact-case candidate status to measure whether layer0_post_attention_residual and layer0_transformer_output materially improve".into()
        }
        "still_unexplained_helper_math_output_behavior" => {
            "inspect whether live hgemm_bf16 still differs from the proved FP32-alpha/beta contract in practice".into()
        }
        other => format!("unexpected helper-contract classification {other}"),
    }
}

const BF16_CLOSE_MAX_ABS_DIFF: f32 = 0.0078125;

fn bf16_close(metrics: &CompareMetrics) -> bool {
    metrics.max_abs_diff <= BF16_CLOSE_MAX_ABS_DIFF
}

fn q_gemm_live_vs_isolated_classification(
    isolated_helper_replay_vs_cpu_reference: &CompareMetrics,
    live_standalone_vs_isolated_helper_replay: &CompareMetrics,
    live_fused_q_vs_live_standalone: &CompareMetrics,
    live_fused_q_vs_isolated_helper_replay: &CompareMetrics,
) -> &'static str {
    if !bf16_close(isolated_helper_replay_vs_cpu_reference) {
        "pre_helper_input_plumbing_mismatch"
    } else if !bf16_close(live_standalone_vs_isolated_helper_replay)
        && !bf16_close(live_fused_q_vs_isolated_helper_replay)
    {
        "pre_helper_input_plumbing_mismatch"
    } else if !bf16_close(live_standalone_vs_isolated_helper_replay) {
        "standalone_integration_mismatch"
    } else if !bf16_close(live_fused_q_vs_live_standalone)
        || !bf16_close(live_fused_q_vs_isolated_helper_replay)
    {
        "fused_path_specific_mismatch"
    } else {
        "helper_fix_propagates_cleanly"
    }
}

fn q_gemm_live_vs_isolated_next_bounded_step(classification: &str) -> String {
    match classification {
        "standalone_integration_mismatch" => {
            "inspect standalone integration around the corrected helper invocation/output path using the exact activation/Q-weight bytes, not fused QKV packing".into()
        }
        "fused_path_specific_mismatch" => {
            "inspect fused-QKV-specific packing/repack/Q-slice integration against the corrected standalone/helper baseline".into()
        }
        "pre_helper_input_plumbing_mismatch" => {
            "inspect the exact live activation/weight/cast bytes fed into both live GEMM lanes before or into helper invocation, not later attention math".into()
        }
        "helper_fix_propagates_cleanly" => {
            "move forward again because the remaining layer-0 issue is later than raw Q GEMM".into()
        }
        other => format!("unexpected live-vs-isolated classification {other}"),
    }
}

fn q_gemm_pre_helper_plumbing_classification(
    standalone_activation_vs_isolated_replay: &CompareMetrics,
    fused_activation_vs_standalone: &CompareMetrics,
    standalone_weight_vs_isolated_replay: &CompareMetrics,
    fused_q_slice_weight_vs_standalone: &CompareMetrics,
) -> &'static str {
    if !standalone_activation_vs_isolated_replay.matched || !fused_activation_vs_standalone.matched
    {
        "activation_preparation"
    } else if !standalone_weight_vs_isolated_replay.matched {
        "weight_preparation"
    } else if !fused_q_slice_weight_vs_standalone.matched {
        "fused_q_slice_plumbing"
    } else {
        "no_pre_helper_difference_found"
    }
}

fn q_gemm_pre_helper_plumbing_next_bounded_step(classification: &str) -> String {
    match classification {
        "activation_preparation" => {
            "inspect the exact BF16 activation capture handed to the live and isolated Q GEMM replays before changing helper math".into()
        }
        "weight_preparation" => {
            "inspect the exact BF16 Q-weight capture handed to the helper replay before touching GEMM math".into()
        }
        "fused_q_slice_plumbing" => {
            "inspect how the fused-path Q slice is extracted and replayed relative to the standalone helper input".into()
        }
        "no_pre_helper_difference_found" => {
            "return to helper invocation/output semantics because the exact pre-helper BF16 buffers now match".into()
        }
        other => format!("unexpected pre-helper plumbing classification {other}"),
    }
}

fn q_gemm_helper_invocation_output_classification(
    invocation_state_match: bool,
    raw_output_match: bool,
) -> &'static str {
    if !invocation_state_match {
        "helper_state_mismatch"
    } else if !raw_output_match {
        "live_output_writeback_or_handle_state_mismatch"
    } else {
        "helper_contract_now_matches_and_issue_moved"
    }
}

fn q_gemm_helper_invocation_output_next_bounded_step(classification: &str) -> String {
    match classification {
        "helper_state_mismatch" => {
            "make the live helper invocation state match the isolated replay contract, then rerun this probe".into()
        }
        "live_output_writeback_or_handle_state_mismatch" => {
            "inspect the immediate post-GEMM writeback or hidden handle-state path, not the already-matched invocation contract".into()
        }
        "helper_contract_now_matches_and_issue_moved" => {
            "rerun the downstream BF16 candidate status now that live and isolated helper output match".into()
        }
        other => format!("unexpected helper invocation/output classification {other}"),
    }
}

fn k_gemm_helper_path_classification(
    isolated_helper_replay_vs_cpu_reference: &CompareMetrics,
    live_standalone_vs_isolated_helper_replay: &CompareMetrics,
    live_fused_k_vs_live_standalone: &CompareMetrics,
    live_fused_k_vs_isolated_helper_replay: &CompareMetrics,
) -> &'static str {
    if !isolated_helper_replay_vs_cpu_reference.matched {
        "shared_bf16_gemm_helper"
    } else if !live_standalone_vs_isolated_helper_replay.matched {
        "standalone_integration_mismatch"
    } else if !live_fused_k_vs_live_standalone.matched
        || !live_fused_k_vs_isolated_helper_replay.matched
    {
        "fused_path_specific_mismatch"
    } else {
        "k_path_cleared"
    }
}

fn k_gemm_helper_path_next_bounded_step(classification: &str) -> String {
    match classification {
        "shared_bf16_gemm_helper" => {
            "return to K helper math/contract and inspect the isolated BF16 helper replay against the CPU reference before touching integration".into()
        }
        "standalone_integration_mismatch" => {
            "inspect the live standalone K integration and immediate call-site/output plumbing, not the shared helper".into()
        }
        "fused_path_specific_mismatch" => {
            "inspect the fused-QKV K slice extraction and repack path, not the standalone K integration".into()
        }
        "k_path_cleared" => {
            "move to V or the combined packed path directly because the layer-0 K path now matches".into()
        }
        other => format!("unexpected K helper-path classification {other}"),
    }
}

fn k_gemm_fused_split_classification(
    fused_pre_bias_vs_standalone_pre_bias: &CompareMetrics,
    fused_post_bias_vs_standalone_post_bias: &CompareMetrics,
    downstream_fused_k_vs_standalone_post_bias: &CompareMetrics,
) -> &'static str {
    if !fused_pre_bias_vs_standalone_pre_bias.matched {
        "fused_pre_bias_mismatch"
    } else if !fused_post_bias_vs_standalone_post_bias.matched {
        "fused_bias_slice_or_application_mismatch"
    } else if !downstream_fused_k_vs_standalone_post_bias.matched {
        "fused_post_bias_repack_or_slice_mismatch"
    } else {
        "k_path_cleared"
    }
}

fn k_gemm_fused_split_next_bounded_step(classification: &str) -> String {
    match classification {
        "fused_pre_bias_mismatch" => {
            "inspect fused combined GEMM K output layout, stride, offset, or order inside the combined QKV output".into()
        }
        "fused_bias_slice_or_application_mismatch" => {
            "inspect fused K bias slice, offset, and application before changing helper math".into()
        }
        "fused_post_bias_repack_or_slice_mismatch" => {
            "inspect later fused-QKV K slice extraction or repack plumbing after the post-bias buffer".into()
        }
        "k_path_cleared" => {
            "rerun the downstream BF16 candidate status and reassess later attention boundaries".into()
        }
        other => format!("unexpected K fused-split classification {other}"),
    }
}

fn k_gemm_fused_qkv_readout_classification(
    combined_buffer_vs_expected: &CompareMetrics,
    downstream_fused_k_vs_expected_k: &CompareMetrics,
) -> &'static str {
    if !combined_buffer_vs_expected.matched {
        "combined_repack_mismatch"
    } else if !downstream_fused_k_vs_expected_k.matched {
        "k_slice_extraction_offset_or_stride_mismatch"
    } else {
        "later_fused_qkv_readout_mismatch"
    }
}

fn k_gemm_fused_qkv_readout_next_bounded_step(classification: &str) -> String {
    match classification {
        "combined_repack_mismatch" => {
            "inspect fused QKV post-bias repack or write ordering before changing later attention inputs".into()
        }
        "k_slice_extraction_offset_or_stride_mismatch" => {
            "inspect K slice extraction offset, stride, and readout semantics against the matched combined QKV buffer".into()
        }
        "later_fused_qkv_readout_mismatch" => {
            "broaden one boundary farther downstream inside fused attention input consumption".into()
        }
        other => format!("unexpected K fused-QKV readout classification {other}"),
    }
}

fn contiguous_k_slice_from_combined_qkv_bits(
    combined_qkv: &[u16],
    num_tokens: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<Vec<u16>> {
    let q_end = num_tokens
        .checked_mul(q_dim)
        .context("q_end overflow while slicing combined QKV")?;
    let k_len = num_tokens
        .checked_mul(kv_dim)
        .context("k length overflow while slicing combined QKV")?;
    let k_end = q_end
        .checked_add(k_len)
        .context("k_end overflow while slicing combined QKV")?;
    if combined_qkv.len() < k_end {
        bail!(
            "combined QKV buffer too short for canonical K slice: len {}, need {}",
            combined_qkv.len(),
            k_end
        );
    }
    Ok(combined_qkv[q_end..k_end].to_vec())
}

fn row_major_k_slice_from_combined_qkv_bits(
    combined_qkv: &[u16],
    num_tokens: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<Vec<u16>> {
    let qkv_dim = q_dim
        .checked_add(kv_dim)
        .and_then(|value| value.checked_add(kv_dim))
        .context("qkv_dim overflow while slicing combined QKV")?;
    let expected_len = num_tokens
        .checked_mul(qkv_dim)
        .context("combined QKV expected length overflow")?;
    if combined_qkv.len() != expected_len {
        bail!(
            "combined QKV length mismatch for row-major K slice: len {}, expected {}",
            combined_qkv.len(),
            expected_len
        );
    }
    Ok(combined_qkv
        .chunks_exact(qkv_dim)
        .flat_map(|row| row[q_dim..q_dim + kv_dim].iter().copied())
        .collect())
}

fn k_gemm_fused_k_readout_classification(
    current_live: &CompareMetrics,
    canonical_contiguous: &CompareMetrics,
    row_major_token_qkv: &CompareMetrics,
    runtime_cache_input_shape: &CompareMetrics,
) -> &'static str {
    if current_live.matched && canonical_contiguous.matched {
        "k_readout_fixed"
    } else if canonical_contiguous.matched && !current_live.matched {
        "k_offset_mismatch"
    } else if row_major_token_qkv.matched || runtime_cache_input_shape.matched {
        "k_stride_or_view_mismatch"
    } else {
        "later_k_readout_transform_mismatch"
    }
}

fn k_gemm_fused_k_readout_next_bounded_step(classification: &str) -> String {
    match classification {
        "k_readout_fixed" => {
            "rerun the downstream BF16 candidate status and reassess layer-0 post-attention and transformer boundaries".into()
        }
        "k_offset_mismatch" => {
            "replace the live fused-K readout with the canonical contiguous K offset, then rerun the K readout and downstream candidate status probes".into()
        }
        "k_stride_or_view_mismatch" => {
            "align the live K stride/view interpretation with the matching readout candidate before touching later attention math".into()
        }
        "later_k_readout_transform_mismatch" => {
            "move one small boundary later into K readout consumption while staying out of broader attention math".into()
        }
        other => format!("unexpected K fused readout classification {other}"),
    }
}

fn best_k_readout_candidate(candidates: &[KGemmFusedKReadoutCandidate]) -> Result<&str> {
    candidates
        .iter()
        .min_by(|lhs, rhs| {
            lhs.vs_expected_k
                .mean_abs_diff
                .total_cmp(&rhs.vs_expected_k.mean_abs_diff)
                .then_with(|| {
                    lhs.vs_expected_k
                        .max_abs_diff
                        .total_cmp(&rhs.vs_expected_k.max_abs_diff)
                })
        })
        .map(|candidate| candidate.candidate_readout_name.as_str())
        .context("no K readout candidates were produced")
}

fn build_k_gemm_helper_path_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    k_projection_pre_bias_artifact: &ExistingKProjectionPreBiasStatusArtifact,
    isolated_helper_replay_vs_cpu_reference: CompareMetrics,
    live_standalone_vs_isolated_helper_replay: CompareMetrics,
    live_fused_k_vs_live_standalone: CompareMetrics,
    live_fused_k_vs_isolated_helper_replay: CompareMetrics,
) -> Result<KGemmHelperPathStatusSummary> {
    if k_projection_pre_bias_artifact.case.case_id != case.id {
        bail!(
            "k-projection-pre-bias artifact case id mismatch: expected {}, found {}",
            case.id,
            k_projection_pre_bias_artifact.case.case_id
        );
    }
    if k_projection_pre_bias_artifact.case.input_token_ids != case.input_token_ids {
        bail!("k-projection-pre-bias artifact token ids do not match the exact smoke case");
    }
    if k_projection_pre_bias_artifact.conclusion != "k_pre_bias_already_divergent" {
        bail!(
            "k-projection-pre-bias artifact conclusion changed: expected k_pre_bias_already_divergent, found {}",
            k_projection_pre_bias_artifact.conclusion
        );
    }
    if k_projection_pre_bias_artifact
        .existing_combined_qkv_context
        .combined_qkv_conclusion
        != "k_dominant"
    {
        bail!(
            "k-projection-pre-bias artifact combined qkv conclusion changed: expected k_dominant, found {}",
            k_projection_pre_bias_artifact
                .existing_combined_qkv_context
                .combined_qkv_conclusion
        );
    }
    if k_projection_pre_bias_artifact.case.hidden_size != exact_hidden_size {
        bail!(
            "k-projection-pre-bias artifact hidden size mismatch: expected {}, found {}",
            exact_hidden_size,
            k_projection_pre_bias_artifact.case.hidden_size
        );
    }
    if k_projection_pre_bias_artifact.case.kv_dim != live_trace.kv_dim {
        bail!(
            "k-projection-pre-bias artifact kv_dim mismatch: expected {}, found {}",
            live_trace.kv_dim,
            k_projection_pre_bias_artifact.case.kv_dim
        );
    }

    let classification = k_gemm_helper_path_classification(
        &isolated_helper_replay_vs_cpu_reference,
        &live_standalone_vs_isolated_helper_replay,
        &live_fused_k_vs_live_standalone,
        &live_fused_k_vs_isolated_helper_replay,
    );
    let next_bounded_step = k_gemm_helper_path_next_bounded_step(classification);
    let runtime_forward_bf16_k_helper_path_status_now = format!(
        "Compared cpu_reference_k_only_gemm vs isolated_live_helper_replay_k_only_gemm, live_standalone_bf16_k_only_gemm vs isolated_live_helper_replay_k_only_gemm, live_fused_bf16_qkv_k_slice vs live_standalone_bf16_k_only_gemm, and live_fused_bf16_qkv_k_slice vs isolated_live_helper_replay_k_only_gemm; classification: {}; next bounded step: {}.",
        classification,
        next_bounded_step
    );

    Ok(KGemmHelperPathStatusSummary {
        schema_version: "runtime_forward_layer0_k_gemm_bf16_helper_path_status/v1".into(),
        provenance: KGemmHelperPathStatusProvenance {
            k_projection_pre_bias_status_artifact_path: cli
                .k_projection_pre_bias_status_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        candidate_branch_taken: live_trace.branch_taken,
        isolated_helper_replay_vs_cpu_reference,
        live_standalone_vs_isolated_helper_replay,
        live_fused_k_vs_live_standalone,
        live_fused_k_vs_isolated_helper_replay,
        classification: classification.to_string(),
        runtime_forward_bf16_k_helper_path_status_now,
        next_bounded_step,
    })
}

fn build_k_gemm_fused_split_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    _k_projection_pre_bias_artifact: &ExistingKProjectionPreBiasStatusArtifact,
    _k_gemm_helper_path_artifact: &KGemmHelperPathStatusSummary,
    fused_pre_bias_vs_standalone_pre_bias: CompareMetrics,
    fused_post_bias_vs_standalone_post_bias: CompareMetrics,
    downstream_fused_k_vs_standalone_post_bias: CompareMetrics,
) -> Result<KGemmFusedSplitStatusSummary> {
    let classification = k_gemm_fused_split_classification(
        &fused_pre_bias_vs_standalone_pre_bias,
        &fused_post_bias_vs_standalone_post_bias,
        &downstream_fused_k_vs_standalone_post_bias,
    );
    let next_bounded_step = k_gemm_fused_split_next_bounded_step(classification);
    let runtime_forward_bf16_fused_k_split_status_now = format!(
        "Compared live fused K pre-bias vs live standalone K pre-bias, live fused K post-bias vs live standalone K post-bias, and downstream live fused K slice vs live standalone K post-bias; classification: {}; next bounded step: {}.",
        classification,
        next_bounded_step
    );

    Ok(KGemmFusedSplitStatusSummary {
        schema_version: "runtime_forward_layer0_k_gemm_bf16_fused_split_status/v1".into(),
        provenance: KGemmFusedSplitStatusProvenance {
            k_projection_pre_bias_status_artifact_path: cli
                .k_projection_pre_bias_status_artifact
                .clone(),
            k_gemm_helper_path_status_artifact_path: cli.k_gemm_helper_path_status_artifact.clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        candidate_branch_taken: live_trace.branch_taken,
        fused_pre_bias_vs_standalone_pre_bias,
        fused_post_bias_vs_standalone_post_bias,
        downstream_fused_k_vs_standalone_post_bias,
        classification: classification.to_string(),
        runtime_forward_bf16_fused_k_split_status_now,
        next_bounded_step,
    })
}

fn build_k_gemm_fused_qkv_readout_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    _k_gemm_helper_path_artifact: &KGemmHelperPathStatusSummary,
    _k_gemm_fused_split_artifact: &KGemmFusedSplitStatusSummary,
    combined_buffer_vs_expected: CompareMetrics,
    downstream_fused_k_vs_expected_k: CompareMetrics,
) -> Result<KGemmFusedQkvReadoutStatusSummary> {
    let classification = k_gemm_fused_qkv_readout_classification(
        &combined_buffer_vs_expected,
        &downstream_fused_k_vs_expected_k,
    );
    let next_bounded_step = k_gemm_fused_qkv_readout_next_bounded_step(classification);
    let combined_buffer_match = combined_buffer_vs_expected.matched;
    let runtime_forward_bf16_fused_qkv_k_readout_status_now = format!(
        "Compared actual fused post-bias combined QKV buffer vs expected standalone post-bias Q||K||V, then compared the current downstream fused K readout vs expected K; classification: {}; next bounded step: {}.",
        classification,
        next_bounded_step
    );

    Ok(KGemmFusedQkvReadoutStatusSummary {
        schema_version: "runtime_forward_layer0_k_gemm_bf16_fused_qkv_readout_status/v1".into(),
        provenance: KGemmFusedQkvReadoutStatusProvenance {
            k_gemm_helper_path_status_artifact_path: cli.k_gemm_helper_path_status_artifact.clone(),
            k_gemm_fused_split_status_artifact_path: cli.k_gemm_fused_split_status_artifact.clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        candidate_branch_taken: live_trace.branch_taken,
        combined_buffer_match,
        combined_buffer_vs_expected,
        downstream_fused_k_vs_expected_k,
        classification: classification.to_string(),
        runtime_forward_bf16_fused_qkv_k_readout_status_now,
        next_bounded_step,
    })
}

fn build_k_gemm_fused_k_readout_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    _k_gemm_helper_path_artifact: &KGemmHelperPathStatusSummary,
    _k_gemm_fused_split_artifact: &KGemmFusedSplitStatusSummary,
    _k_gemm_fused_qkv_readout_artifact: &KGemmFusedQkvReadoutStatusSummary,
    current_live_fused_k_readout: CompareMetrics,
    canonical_contiguous_k_slice: CompareMetrics,
    row_major_token_qkv_slice: CompareMetrics,
    runtime_cache_input_shape: CompareMetrics,
) -> Result<KGemmFusedKReadoutStatusSummary> {
    let classification = k_gemm_fused_k_readout_classification(
        &current_live_fused_k_readout,
        &canonical_contiguous_k_slice,
        &row_major_token_qkv_slice,
        &runtime_cache_input_shape,
    );
    let next_bounded_step = k_gemm_fused_k_readout_next_bounded_step(classification);
    let candidate_readouts = vec![
        KGemmFusedKReadoutCandidate {
            candidate_readout_name: "current_live_fused_k_readout".into(),
            vs_expected_k: current_live_fused_k_readout,
        },
        KGemmFusedKReadoutCandidate {
            candidate_readout_name: "canonical_contiguous_k_slice".into(),
            vs_expected_k: canonical_contiguous_k_slice,
        },
        KGemmFusedKReadoutCandidate {
            candidate_readout_name: "row_major_token_qkv_slice".into(),
            vs_expected_k: row_major_token_qkv_slice,
        },
        KGemmFusedKReadoutCandidate {
            candidate_readout_name: "runtime_cache_input_shape".into(),
            vs_expected_k: runtime_cache_input_shape,
        },
    ];
    let best_candidate_readout_name = best_k_readout_candidate(&candidate_readouts)?.to_string();
    let any_candidate_materially_collapses_mismatch = candidate_readouts
        .iter()
        .any(|candidate| candidate.vs_expected_k.matched);
    let runtime_forward_bf16_fused_k_readout_status_now = format!(
        "Compared current live fused-K readout, canonical contiguous K slice, row-major token QKV slice, and runtime cache-input-shape K readout against expected standalone K; best candidate: {}; classification: {}; next bounded step: {}.",
        best_candidate_readout_name,
        classification,
        next_bounded_step
    );

    Ok(KGemmFusedKReadoutStatusSummary {
        schema_version: "runtime_forward_layer0_k_gemm_bf16_fused_k_readout_status/v1".into(),
        provenance: KGemmFusedKReadoutStatusProvenance {
            k_gemm_helper_path_status_artifact_path: cli.k_gemm_helper_path_status_artifact.clone(),
            k_gemm_fused_split_status_artifact_path: cli.k_gemm_fused_split_status_artifact.clone(),
            k_gemm_fused_qkv_readout_status_artifact_path: cli
                .k_gemm_fused_qkv_readout_status_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        candidate_branch_taken: live_trace.branch_taken,
        candidate_readouts,
        best_candidate_readout_name,
        any_candidate_materially_collapses_mismatch,
        classification: classification.to_string(),
        runtime_forward_bf16_fused_k_readout_status_now,
        next_bounded_step,
    })
}

fn build_k_consumption_rope_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    _k_gemm_fused_k_readout_artifact: &KGemmFusedKReadoutStatusSummary,
    pre_rope_k_readout_vs_expected: CompareMetrics,
) -> Result<KConsumptionRopeStatusSummary> {
    let next_bounded_step = "obtain or generate a matching authoritative PPP layer0_k_post_rope_pre_cache capture, then compare it against this local runner seam before moving to cache write/readback".to_string();
    let runtime_forward_k_consumption_seam_status_now = format!(
        "Captured layer-0 K immediately after f16 RoPE and before cache write for the exact smoke case; pre-RoPE fused-K readout still matches expected K: {}; PPP needs a matching authoritative layer0_k_post_rope_pre_cache capture before classification; next bounded step: {}.",
        pre_rope_k_readout_vs_expected.matched,
        next_bounded_step
    );

    Ok(KConsumptionRopeStatusSummary {
        schema_version: "runtime_forward_layer0_k_consumption_rope_status/v1".into(),
        provenance: KConsumptionRopeStatusProvenance {
            k_gemm_fused_k_readout_status_artifact_path: cli
                .k_gemm_fused_k_readout_status_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        num_tokens: live_trace.num_tokens,
        candidate_branch_taken: live_trace.branch_taken,
        boundary: "layer0_k_post_rope_pre_cache".into(),
        pre_rope_k_readout_vs_expected,
        k_post_rope_pre_cache_f16_bits: live_trace.k_post_rope_pre_cache_f16_bits.clone(),
        k_post_rope_pre_cache_f32: live_trace.k_post_rope_pre_cache_f32.clone(),
        runtime_forward_k_consumption_seam_status_now,
        next_bounded_step,
    })
}

fn k_rope_variant_materially_collapses(
    baseline: &CompareMetrics,
    variant: &CompareMetrics,
) -> bool {
    variant.matched
        || variant.mean_abs_diff <= baseline.mean_abs_diff * 0.25
        || variant.max_abs_diff <= baseline.max_abs_diff * 0.25
}

fn classify_k_rope_convention(best_variant_name: &str, materially_collapsed: bool) -> &'static str {
    if !materially_collapsed {
        return "runtime_rope_impl_needed";
    }
    if best_variant_name.contains("interleaved") {
        "rope_pairing_convention"
    } else if best_variant_name.contains("position_plus")
        || best_variant_name.contains("position_minus")
    {
        "rope_position_or_offset_convention"
    } else if best_variant_name.contains("f16_table")
        || best_variant_name.contains("bf16_table")
        || best_variant_name.contains("f32_output")
        || best_variant_name.contains("bf16_output")
    {
        "rope_dtype_precision_convention"
    } else {
        "runtime_rope_impl_needed"
    }
}

fn build_k_rope_convention_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    local_k_consumption_artifact: &KConsumptionRopeStatusSummary,
    official_post_rope_k: &[f32],
    pre_rope_k_readout_vs_expected: CompareMetrics,
    baseline_local_runtime_vs_official: CompareMetrics,
    variants: Vec<KRopeConventionVariant>,
) -> Result<KRopeConventionStatusSummary> {
    let best_variant = variants
        .iter()
        .min_by(|lhs, rhs| {
            lhs.vs_official_post_rope_k
                .mean_abs_diff
                .partial_cmp(&rhs.vs_official_post_rope_k.mean_abs_diff)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    lhs.vs_official_post_rope_k
                        .max_abs_diff
                        .partial_cmp(&rhs.vs_official_post_rope_k.max_abs_diff)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .context("K RoPE convention sweep produced no variants")?;
    let best_variant_name = best_variant.variant_name.clone();
    let any_variant_materially_collapses_mismatch = variants
        .iter()
        .any(|variant| variant.materially_collapses_mismatch);
    let classification = classify_k_rope_convention(
        &best_variant_name,
        best_variant.materially_collapses_mismatch,
    )
    .to_string();
    let next_bounded_step = if any_variant_materially_collapses_mismatch {
        format!(
            "use {} for a surgical layer-0 K RoPE convention proof/fix, still before cache or later attention work",
            best_variant_name
        )
    } else {
        "inspect the live runtime RoPE implementation and tables directly before cache write/readback or later attention work".to_string()
    };
    let runtime_forward_k_rope_status_now = format!(
        "Host-side K-only RoPE sweep for exact smoke case: best_variant={}, best_max_abs_diff={}, best_mean_abs_diff={}, any_material_collapse={}; classification={}; next bounded step: {}.",
        best_variant_name,
        best_variant.vs_official_post_rope_k.max_abs_diff,
        best_variant.vs_official_post_rope_k.mean_abs_diff,
        any_variant_materially_collapses_mismatch,
        classification,
        next_bounded_step
    );

    Ok(KRopeConventionStatusSummary {
        schema_version: "runtime_forward_layer0_k_rope_convention_status/v1".into(),
        provenance: KRopeConventionStatusProvenance {
            k_consumption_rope_status_artifact_path: cli.k_consumption_rope_status_artifact.clone(),
            official_k_post_rope_pre_cache_artifact_path: cli
                .official_k_post_rope_pre_cache_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        num_tokens: live_trace.num_tokens,
        official_len: official_post_rope_k.len(),
        local_runtime_len: local_k_consumption_artifact.k_post_rope_pre_cache_f32.len(),
        pre_rope_k_readout_vs_expected,
        baseline_local_runtime_vs_official,
        variants,
        best_variant_name,
        any_variant_materially_collapses_mismatch,
        classification,
        runtime_forward_k_rope_status_now,
        next_bounded_step,
    })
}

fn k_rope_position_metadata_matches(capture: &Layer0KRopeDebugCapture) -> bool {
    capture.runtime_position_ids.len() == capture.final_token_index + 1
        && capture
            .runtime_position_ids
            .iter()
            .enumerate()
            .all(|(idx, &position)| position == idx as i32)
        && capture.effective_position_id == capture.final_token_index as i32
        && capture.effective_position_modulo == capture.final_token_index
}

fn classify_k_rope_implementation(
    position_metadata_match: bool,
    factors_vs_host: &CompareMetrics,
    live_factor_recompute_vs_live: &CompareMetrics,
) -> &'static str {
    if !position_metadata_match {
        "runtime_position_metadata_mismatch"
    } else if !factors_vs_host.matched {
        "runtime_table_generation_or_selection_mismatch"
    } else if !live_factor_recompute_vs_live.matched {
        "runtime_rope_application_math_mismatch"
    } else {
        "still_unclear"
    }
}

fn build_k_rope_implementation_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    exact_hidden_size: usize,
    live_trace: &Layer0QkvTrace,
    k_rope_debug: &Layer0KRopeDebugCapture,
    position_metadata_match: bool,
    live_runtime_factors_vs_host_generated_factors: CompareMetrics,
    host_recompute_from_live_factors_vs_live_runtime_post_rope: CompareMetrics,
    host_recompute_from_live_factors_vs_official_post_rope: CompareMetrics,
    host_recompute_from_host_generated_factors_vs_live_runtime_post_rope: CompareMetrics,
    host_recompute_from_host_generated_factors_vs_official_post_rope: CompareMetrics,
) -> KRopeImplementationStatusSummary {
    let classification = classify_k_rope_implementation(
        position_metadata_match,
        &live_runtime_factors_vs_host_generated_factors,
        &host_recompute_from_live_factors_vs_live_runtime_post_rope,
    )
    .to_string();
    let next_bounded_step = match classification.as_str() {
        "runtime_position_metadata_mismatch" => {
            "focus on runtime position metadata construction/packing before RoPE application"
        }
        "runtime_table_generation_or_selection_mismatch" => {
            "focus on runtime RoPE table generation/upload/selection for the captured position"
        }
        "runtime_rope_application_math_mismatch" => {
            "focus on the runtime rotary_embedding_f16 application math against captured factors"
        }
        _ => {
            "obtain an official RoPE factor/position capture or equivalent PPP-side RoPE state before cache or later attention work"
        }
    }
    .to_string();
    let runtime_forward_k_rope_implementation_status_now = format!(
        "Captured live final-token K RoPE state for exact smoke case: factors_match_host={}, host_recompute_matches_live={}, host_recompute_matches_official={}; classification={}; next bounded step: {}.",
        live_runtime_factors_vs_host_generated_factors.matched,
        host_recompute_from_live_factors_vs_live_runtime_post_rope.matched,
        host_recompute_from_live_factors_vs_official_post_rope.matched,
        classification,
        next_bounded_step
    );

    KRopeImplementationStatusSummary {
        schema_version: "runtime_forward_layer0_k_rope_implementation_status/v1".into(),
        provenance: KRopeImplementationStatusProvenance {
            k_rope_convention_status_artifact_path: cli.k_rope_convention_status_artifact.clone(),
            k_consumption_rope_status_artifact_path: cli.k_consumption_rope_status_artifact.clone(),
            official_k_post_rope_pre_cache_artifact_path: cli
                .official_k_post_rope_pre_cache_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: exact_hidden_size,
        q_dim: live_trace.q_dim,
        kv_dim: live_trace.kv_dim,
        num_tokens: live_trace.num_tokens,
        captured_state: KRopeImplementationCapturedState {
            final_token_index: k_rope_debug.final_token_index,
            effective_position_id: k_rope_debug.effective_position_id,
            effective_position_modulo: k_rope_debug.effective_position_modulo,
            runtime_position_ids: k_rope_debug.runtime_position_ids.clone(),
            position_metadata_match,
            head_dim: k_rope_debug.head_dim,
            half_dim: k_rope_debug.half_dim,
            num_kv_heads: k_rope_debug.num_kv_heads,
            final_token_k_pre_rope_len: k_rope_debug.final_token_k_pre_rope_f32.len(),
            final_token_k_post_rope_len: k_rope_debug.final_token_k_post_rope_f32.len(),
            live_runtime_cos_row_f32: k_rope_debug.live_runtime_cos_row_f32.clone(),
            live_runtime_sin_row_f32: k_rope_debug.live_runtime_sin_row_f32.clone(),
        },
        live_runtime_factors_vs_host_generated_factors,
        host_recompute_from_live_factors_vs_live_runtime_post_rope,
        host_recompute_from_live_factors_vs_official_post_rope,
        host_recompute_from_host_generated_factors_vs_live_runtime_post_rope,
        host_recompute_from_host_generated_factors_vs_official_post_rope,
        classification,
        runtime_forward_k_rope_implementation_status_now,
        next_bounded_step,
    }
}

fn q_gemm_live_vs_isolated_first_remaining_mismatch(
    isolated_helper_replay_vs_cpu_reference: &CompareMetrics,
    live_standalone_vs_isolated_helper_replay: &CompareMetrics,
    live_fused_q_vs_live_standalone: &CompareMetrics,
    live_fused_q_vs_isolated_helper_replay: &CompareMetrics,
) -> &'static str {
    if !bf16_close(isolated_helper_replay_vs_cpu_reference) {
        "isolated_helper_replay_vs_cpu_reference"
    } else if !bf16_close(live_standalone_vs_isolated_helper_replay) {
        "live_standalone_vs_isolated_helper_replay"
    } else if !bf16_close(live_fused_q_vs_live_standalone) {
        "live_fused_q_vs_live_standalone"
    } else if !bf16_close(live_fused_q_vs_isolated_helper_replay) {
        "live_fused_q_vs_isolated_helper_replay"
    } else {
        "none"
    }
}

fn build_q_gemm_live_vs_isolated_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
    helper_contract_artifact: &QGemmHelperContractStatusSummary,
    isolated_helper_replay_vs_cpu_reference: CompareMetrics,
    live_standalone_vs_isolated_helper_replay: CompareMetrics,
    live_fused_q_vs_live_standalone: CompareMetrics,
    live_fused_q_vs_isolated_helper_replay: CompareMetrics,
) -> Result<QGemmLiveVsIsolatedStatusSummary> {
    if helper_contract_artifact.case_id != case.id {
        bail!(
            "helper-contract artifact case id mismatch: expected {}, found {}",
            case.id,
            helper_contract_artifact.case_id
        );
    }
    if helper_contract_artifact.input_token_ids != case.input_token_ids {
        bail!("helper-contract artifact token ids do not match the exact smoke case");
    }
    if helper_contract_artifact.helper_contract.alpha_beta_dtype != "f32" {
        bail!(
            "helper-contract artifact alpha/beta dtype changed: expected f32, found {}",
            helper_contract_artifact.helper_contract.alpha_beta_dtype
        );
    }
    if !helper_contract_artifact.raw_exact_contract_reproduces_helper {
        bail!("helper-contract artifact exact contract replay no longer reproduces the helper");
    }
    if helper_contract_artifact.best_explained_by != "alpha_beta_or_descriptor_configuration" {
        bail!(
            "helper-contract artifact classification changed: expected alpha_beta_or_descriptor_configuration, found {}",
            helper_contract_artifact.best_explained_by
        );
    }
    if helper_contract_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "helper-contract artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            helper_contract_artifact.hidden_size
        );
    }
    if helper_contract_artifact.q_dim != live_trace.q_dim {
        bail!(
            "helper-contract artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            helper_contract_artifact.q_dim
        );
    }

    let classification = q_gemm_live_vs_isolated_classification(
        &isolated_helper_replay_vs_cpu_reference,
        &live_standalone_vs_isolated_helper_replay,
        &live_fused_q_vs_live_standalone,
        &live_fused_q_vs_isolated_helper_replay,
    );
    let first_remaining_mismatch = q_gemm_live_vs_isolated_first_remaining_mismatch(
        &isolated_helper_replay_vs_cpu_reference,
        &live_standalone_vs_isolated_helper_replay,
        &live_fused_q_vs_live_standalone,
        &live_fused_q_vs_isolated_helper_replay,
    );
    let next_bounded_step = q_gemm_live_vs_isolated_next_bounded_step(classification);
    let runtime_forward_live_vs_isolated_bf16_status_now = format!(
        "Compared cpu_reference_q_only_gemm vs isolated_live_helper_replay_q_only_gemm, live_standalone_bf16_q_only_gemm vs isolated_live_helper_replay_q_only_gemm, live_fused_bf16_qkv_q_slice vs live_standalone_bf16_q_only_gemm, and live_fused_bf16_qkv_q_slice vs isolated_live_helper_replay_q_only_gemm; first remaining mismatch: {}; classification: {}; next bounded step: {}.",
        first_remaining_mismatch,
        classification,
        next_bounded_step
    );

    Ok(QGemmLiveVsIsolatedStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_live_vs_isolated_status/v1".into(),
        provenance: QGemmLiveVsIsolatedStatusProvenance {
            q_gemm_helper_contract_status_artifact_path: cli
                .q_gemm_helper_contract_status_artifact
                .clone(),
            q_gemm_helper_status_artifact_path: cli.q_gemm_helper_status_artifact.clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: live_trace.hidden_size,
        q_dim: live_trace.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        bf16_close_max_abs_diff: BF16_CLOSE_MAX_ABS_DIFF,
        isolated_helper_replay_vs_cpu_reference,
        live_standalone_vs_isolated_helper_replay,
        live_fused_q_vs_live_standalone,
        live_fused_q_vs_isolated_helper_replay,
        classification: classification.to_string(),
        runtime_forward_live_vs_isolated_bf16_status_now,
        next_bounded_step,
    })
}

fn build_q_gemm_pre_helper_plumbing_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
    live_vs_isolated_artifact: &QGemmLiveVsIsolatedStatusSummary,
    standalone_activation_vs_isolated_replay: CompareMetrics,
    fused_activation_vs_standalone: CompareMetrics,
    standalone_weight_vs_isolated_replay: CompareMetrics,
    fused_q_slice_weight_vs_standalone: CompareMetrics,
) -> Result<QGemmPreHelperPlumbingStatusSummary> {
    if live_vs_isolated_artifact.case_id != case.id {
        bail!(
            "live-vs-isolated artifact case id mismatch: expected {}, found {}",
            case.id,
            live_vs_isolated_artifact.case_id
        );
    }
    if live_vs_isolated_artifact.input_token_ids != case.input_token_ids {
        bail!("live-vs-isolated artifact token ids do not match the exact smoke case");
    }
    if live_vs_isolated_artifact.classification != "pre_helper_input_plumbing_mismatch" {
        bail!(
            "live-vs-isolated artifact classification changed: expected pre_helper_input_plumbing_mismatch, found {}",
            live_vs_isolated_artifact.classification
        );
    }
    if live_vs_isolated_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "live-vs-isolated artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            live_vs_isolated_artifact.hidden_size
        );
    }
    if live_vs_isolated_artifact.q_dim != live_trace.q_dim {
        bail!(
            "live-vs-isolated artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            live_vs_isolated_artifact.q_dim
        );
    }

    let classification = q_gemm_pre_helper_plumbing_classification(
        &standalone_activation_vs_isolated_replay,
        &fused_activation_vs_standalone,
        &standalone_weight_vs_isolated_replay,
        &fused_q_slice_weight_vs_standalone,
    );
    let earliest_input_plumbing_divergence = match classification {
        "activation_preparation" => "activation_preparation",
        "weight_preparation" => "weight_preparation",
        "fused_q_slice_plumbing" => "fused_q_slice_plumbing",
        _ => "no_pre_helper_difference_found",
    }
    .to_string();
    let next_bounded_step = q_gemm_pre_helper_plumbing_next_bounded_step(classification);
    let runtime_forward_bf16_pre_helper_plumbing_status_now = format!(
        "Compared exact BF16 activation, helper-replay Q weight, and fused-path Q-slice buffers directly from the live trace; activation vs isolated replay: {}; fused activation vs standalone: {}; standalone weight vs isolated replay: {}; fused Q-slice weight vs standalone: {}; earliest input plumbing divergence: {}; classification: {}; next bounded step: {}.",
        if standalone_activation_vs_isolated_replay.matched {
            "matched"
        } else {
            "mismatched"
        },
        if fused_activation_vs_standalone.matched {
            "matched"
        } else {
            "mismatched"
        },
        if standalone_weight_vs_isolated_replay.matched {
            "matched"
        } else {
            "mismatched"
        },
        if fused_q_slice_weight_vs_standalone.matched {
            "matched"
        } else {
            "mismatched"
        },
        earliest_input_plumbing_divergence,
        classification,
        next_bounded_step
    );

    Ok(QGemmPreHelperPlumbingStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_pre_helper_plumbing_status/v1".into(),
        provenance: QGemmPreHelperPlumbingStatusProvenance {
            q_gemm_live_vs_isolated_status_artifact_path: cli
                .q_gemm_live_vs_isolated_status_artifact
                .clone(),
            q_gemm_helper_contract_status_artifact_path: cli
                .q_gemm_helper_contract_status_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: live_trace.hidden_size,
        q_dim: live_trace.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        standalone_activation_vs_isolated_replay,
        fused_activation_vs_standalone,
        standalone_weight_vs_isolated_replay,
        fused_q_slice_weight_vs_standalone,
        earliest_input_plumbing_divergence,
        runtime_forward_bf16_pre_helper_plumbing_status_now,
        next_bounded_step,
    })
}

fn build_q_gemm_helper_invocation_output_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    helper_status_artifact: &ExistingQGemmHelperStatusArtifact,
    live_vs_isolated_artifact: &QGemmLiveVsIsolatedStatusSummary,
    helper_contract_artifact: &QGemmHelperContractStatusSummary,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
    live_helper_invocation_state: Bf16GemmInvocationRecord,
    isolated_helper_invocation_state: Bf16GemmInvocationRecord,
    live_output_vs_isolated_replay: CompareMetrics,
) -> Result<QGemmHelperInvocationOutputStatusSummary> {
    if helper_status_artifact.case_id != case.id {
        bail!(
            "helper-status artifact case id mismatch: expected {}, found {}",
            case.id,
            helper_status_artifact.case_id
        );
    }
    if helper_status_artifact.input_token_ids != case.input_token_ids {
        bail!("helper-status artifact token ids do not match the exact smoke case");
    }
    if helper_status_artifact.classification != "shared_bf16_gemm_helper" {
        bail!(
            "helper-status artifact classification changed: expected shared_bf16_gemm_helper, found {}",
            helper_status_artifact.classification
        );
    }
    if live_vs_isolated_artifact.case_id != case.id {
        bail!(
            "live-vs-isolated artifact case id mismatch: expected {}, found {}",
            case.id,
            live_vs_isolated_artifact.case_id
        );
    }
    if live_vs_isolated_artifact.input_token_ids != case.input_token_ids {
        bail!("live-vs-isolated artifact token ids do not match the exact smoke case");
    }
    if live_vs_isolated_artifact.classification != "pre_helper_input_plumbing_mismatch" {
        bail!(
            "live-vs-isolated artifact classification changed: expected pre_helper_input_plumbing_mismatch, found {}",
            live_vs_isolated_artifact.classification
        );
    }
    if helper_contract_artifact.case_id != case.id {
        bail!(
            "helper-contract artifact case id mismatch: expected {}, found {}",
            case.id,
            helper_contract_artifact.case_id
        );
    }
    if helper_contract_artifact.input_token_ids != case.input_token_ids {
        bail!("helper-contract artifact token ids do not match the exact smoke case");
    }
    if helper_contract_artifact.helper_contract.alpha_beta_dtype != "f32" {
        bail!(
            "helper-contract artifact alpha/beta dtype changed: expected f32, found {}",
            helper_contract_artifact.helper_contract.alpha_beta_dtype
        );
    }
    if !helper_contract_artifact.raw_exact_contract_reproduces_helper {
        bail!("helper-contract artifact exact contract replay no longer reproduces the helper");
    }
    if helper_contract_artifact.best_explained_by != "alpha_beta_or_descriptor_configuration" {
        bail!(
            "helper-contract artifact classification changed: expected alpha_beta_or_descriptor_configuration, found {}",
            helper_contract_artifact.best_explained_by
        );
    }
    if helper_contract_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "helper-contract artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            helper_contract_artifact.hidden_size
        );
    }
    if helper_contract_artifact.q_dim != live_trace.q_dim {
        bail!(
            "helper-contract artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            helper_contract_artifact.q_dim
        );
    }
    let invocation_state_match = live_helper_invocation_state == isolated_helper_invocation_state;
    let raw_output_match = live_output_vs_isolated_replay.matched;
    let classification =
        q_gemm_helper_invocation_output_classification(invocation_state_match, raw_output_match);
    let next_bounded_step = q_gemm_helper_invocation_output_next_bounded_step(classification);
    let runtime_forward_bf16_helper_invocation_output_status_now = format!(
        "Captured live standalone helper invocation state (api={}, m={}, n={}, k={}, alpha/beta={}, pointer_mode={}, math_mode={}, atomics_mode={}, graph_workspace_registered={}, graph_workspace_bytes={}); compared against isolated replay state; invocation_state_match={}; raw_output_match={}; classification: {}; next bounded step: {}.",
        live_helper_invocation_state.api,
        live_helper_invocation_state.m,
        live_helper_invocation_state.n,
        live_helper_invocation_state.k,
        live_helper_invocation_state.alpha_beta_dtype,
        live_helper_invocation_state.handle_state.pointer_mode,
        live_helper_invocation_state.handle_state.math_mode,
        live_helper_invocation_state.handle_state.atomics_mode,
        live_helper_invocation_state.handle_state.graph_workspace_registered,
        live_helper_invocation_state.handle_state.graph_workspace_bytes,
        invocation_state_match,
        raw_output_match,
        classification,
        next_bounded_step
    );

    Ok(QGemmHelperInvocationOutputStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_helper_invocation_output_status/v1"
            .into(),
        provenance: QGemmHelperInvocationOutputStatusProvenance {
            q_gemm_helper_status_artifact_path: cli.q_gemm_helper_status_artifact.clone(),
            q_gemm_helper_contract_status_artifact_path: cli
                .q_gemm_helper_contract_status_artifact
                .clone(),
            q_gemm_live_vs_isolated_status_artifact_path: cli
                .q_gemm_live_vs_isolated_status_artifact
                .clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: live_trace.hidden_size,
        q_dim: live_trace.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        live_helper_invocation_state,
        isolated_helper_invocation_state,
        invocation_state_match,
        raw_output_match,
        live_output_vs_isolated_replay,
        classification: classification.to_string(),
        runtime_forward_bf16_helper_invocation_output_status_now,
        next_bounded_step,
    })
}

fn q_gemm_helper_classification(
    fused_vs_reference: &CompareMetrics,
    standalone_vs_reference: &CompareMetrics,
) -> &'static str {
    if standalone_vs_reference.matched && !fused_vs_reference.matched {
        "fused_path_specific"
    } else if !standalone_vs_reference.matched && !fused_vs_reference.matched {
        "shared_bf16_gemm_helper"
    } else {
        "capture/integration_plumbing"
    }
}

fn q_gemm_helper_next_bounded_step(classification: &str) -> String {
    match classification {
        "fused_path_specific" => {
            "inspect fused BF16 QKV path packing and Q-slice/stride semantics around the fused output, not shared GEMM math".into()
        }
        "shared_bf16_gemm_helper" => {
            "inspect the shared hgemm_bf16 math/output path itself, not fused packing/slicing".into()
        }
        "capture/integration_plumbing" => {
            "inspect live capture/emission plumbing around the Q artifact because the narrowed fused path no longer reproduces the divergence".into()
        }
        other => format!("unexpected helper-status classification {other}"),
    }
}

fn build_q_gemm_helper_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
) -> Result<QGemmHelperStatusSummary> {
    let reference_output = build_live_q_gemm_reference(live_trace)?;
    let fused_vs_reference = compare_vectors(&live_trace.raw_q_gemm_output, &reference_output)?;
    let standalone_vs_reference =
        compare_vectors(&live_trace.standalone_q_gemm_output, &reference_output)?;
    let fused_vs_standalone = compare_vectors(
        &live_trace.raw_q_gemm_output,
        &live_trace.standalone_q_gemm_output,
    )?;
    let classification =
        q_gemm_helper_classification(&fused_vs_reference, &standalone_vs_reference);
    let divergence_status = if standalone_vs_reference.matched && !fused_vs_reference.matched {
        "fused only diverges"
    } else if !standalone_vs_reference.matched && !fused_vs_reference.matched {
        "both live paths diverge"
    } else {
        "neither live path diverges"
    };
    let next_bounded_step = q_gemm_helper_next_bounded_step(classification);
    let runtime_forward_bf16_q_gemm_helper_status_now = format!(
        "Compared fused BF16 QKV Q-slice, standalone BF16 Q-only GEMM, and host BF16 reference matvec; {}; classification: {}; next bounded step: {}.",
        divergence_status, classification, next_bounded_step
    );

    Ok(QGemmHelperStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_helper_status/v1".into(),
        provenance: QGemmHelperStatusProvenance {
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: live_trace.hidden_size,
        q_dim: live_trace.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        fused_vs_reference,
        standalone_vs_reference,
        fused_vs_standalone,
        classification: classification.to_string(),
        runtime_forward_bf16_q_gemm_helper_status_now,
        next_bounded_step,
    })
}

fn build_q_gemm_microproof_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    helper_status_artifact: &ExistingQGemmHelperStatusArtifact,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
    helper_vs_reference: CompareMetrics,
    isolated_reproduces_live_standalone_capture: bool,
    activation_row_major_input_hypothesis_vs_reference: CompareMetrics,
    q_weight_column_major_hypothesis_vs_reference: CompareMetrics,
) -> QGemmMicroproofStatusSummary {
    let classification = if helper_vs_reference.matched {
        "integration_only"
    } else if collapse_rule_satisfied(
        &helper_vs_reference,
        &activation_row_major_input_hypothesis_vs_reference,
    ) || collapse_rule_satisfied(
        &helper_vs_reference,
        &q_weight_column_major_hypothesis_vs_reference,
    ) {
        "invocation_layout_mismatch_candidate"
    } else {
        "helper_itself_mismatch"
    };
    let capture_reproduction_status = if isolated_reproduces_live_standalone_capture {
        "the isolated current helper bit-matches the prior standalone live capture"
    } else {
        "the isolated current helper does not bit-match the prior standalone live capture"
    };
    let alt_collapse_status = if collapse_rule_satisfied(
        &helper_vs_reference,
        &activation_row_major_input_hypothesis_vs_reference,
    ) || collapse_rule_satisfied(
        &helper_vs_reference,
        &q_weight_column_major_hypothesis_vs_reference,
    ) {
        "at least one minimal invocation/layout hypothesis materially collapses the mismatch"
    } else {
        "no minimal invocation/layout hypothesis materially collapses the mismatch"
    };
    let current_helper_status = if helper_vs_reference.matched {
        "the isolated current helper matches the host BF16 reference"
    } else {
        "the isolated current helper mismatches the host BF16 reference"
    };
    let next_bounded_step = q_gemm_microproof_next_bounded_step(classification);
    let runtime_forward_bf16_q_gemm_microproof_status_now = format!(
        "Used the live-captured final-token BF16 activation row and layer-0 Q weight reconstructed from Layer0QGemmTrace; {}; {}; {}; classification: {}; next bounded step: {}.",
        current_helper_status,
        capture_reproduction_status,
        alt_collapse_status,
        classification,
        next_bounded_step
    );

    QGemmMicroproofStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_microproof_status/v1".into(),
        provenance: QGemmMicroproofStatusProvenance {
            q_gemm_helper_status_artifact_path: cli.q_gemm_helper_status_artifact.clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: helper_status_artifact.hidden_size,
        q_dim: helper_status_artifact.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        isolated_reproduces_live_standalone_capture,
        helper_vs_reference,
        activation_row_major_input_hypothesis_vs_reference,
        q_weight_column_major_hypothesis_vs_reference,
        classification: classification.to_string(),
        runtime_forward_bf16_q_gemm_microproof_status_now,
        next_bounded_step,
    }
}

fn build_q_gemm_helper_contract_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    microproof_artifact: &ExistingQGemmMicroproofStatusArtifact,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
    helper_contract: RawBf16GemmContract,
    alternate_contract: RawBf16GemmContract,
    raw_exact_contract_reproduces_helper: bool,
    isolated_reproduces_live_standalone_capture: bool,
    helper_vs_reference: CompareMetrics,
    alternate_contract_vs_reference: CompareMetrics,
) -> QGemmHelperContractStatusSummary {
    let helper_gap_collapsed_as_predicted =
        collapse_rule_satisfied(&alternate_contract_vs_reference, &helper_vs_reference);
    let best_explained_by = if helper_gap_collapsed_as_predicted {
        "alpha_beta_or_descriptor_configuration"
    } else {
        "still_unexplained_helper_math_output_behavior"
    };
    let alternate_status = if helper_gap_collapsed_as_predicted {
        "the live FP32-alpha/beta helper materially improves the exact one-row Q case over the legacy BF16-scalar contract"
    } else {
        "the live FP32-alpha/beta helper does not materially improve the exact one-row Q case over the legacy BF16-scalar contract"
    };
    let helper_capture_status = if isolated_reproduces_live_standalone_capture {
        "the isolated helper still bit-matches the prior standalone live capture"
    } else {
        "the isolated helper still does not bit-match the prior standalone live capture"
    };
    let next_bounded_step = q_gemm_helper_contract_next_bounded_step(best_explained_by);
    let runtime_forward_bf16_helper_contract_status_now = format!(
        "Used the live-captured final-token BF16 activation row and layer-0 Q weight reconstructed from Layer0QGemmTrace; observed helper contract: BF16 inputs/weights/output, CUBLAS_COMPUTE_32F, {} alpha/beta, CUBLAS_OP_T weight, CUBLAS_OP_N input, lda={}, ldb={}, ldc={}; {}; {}; classification: {}; next bounded step: {}.",
        helper_contract.alpha_beta_dtype.as_str(),
        helper_contract.lda,
        helper_contract.ldb,
        helper_contract.ldc,
        alternate_status,
        helper_capture_status,
        best_explained_by,
        next_bounded_step
    );

    QGemmHelperContractStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_helper_contract_status/v1".into(),
        provenance: QGemmHelperContractStatusProvenance {
            q_gemm_microproof_status_artifact_path: cli.q_gemm_microproof_status_artifact.clone(),
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: microproof_artifact.hidden_size,
        q_dim: microproof_artifact.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        helper_contract: record_bf16_gemm_contract(
            helper_contract,
            live_trace.q_dim,
            live_trace.hidden_size,
        ),
        alternate_contract: record_bf16_gemm_contract(
            alternate_contract,
            live_trace.q_dim,
            live_trace.hidden_size,
        ),
        raw_exact_contract_reproduces_helper,
        isolated_reproduces_live_standalone_capture,
        helper_vs_reference,
        alternate_contract_vs_reference,
        best_explained_by: best_explained_by.to_string(),
        runtime_forward_bf16_helper_contract_status_now,
        next_bounded_step,
    }
}

fn build_q_gemm_status_summary(
    cli: &Cli,
    case: &ArtifactCase,
    attn_norm_artifact: &PinnedArtifact,
    visible_devices: &str,
    live_trace: &Layer0QGemmTrace,
    shadow: &Layer0QGemmShadow,
) -> Result<QGemmStatusSummary> {
    let activation_preparation =
        compare_vectors(&live_trace.activation_row, &shadow.activation_row)?;
    let weight_preparation = compare_vectors(&live_trace.q_weight, &shadow.q_weight)?;
    let raw_q_gemm_output = compare_vectors(&live_trace.raw_q_gemm_output, &shadow.raw_q_output)?;
    let immediate_post_gemm_cast_writeback = match (
        live_trace.post_gemm_cast_writeback.as_ref(),
        shadow.post_gemm_cast_writeback.as_ref(),
    ) {
        (Some(live), Some(shadow)) => Some(compare_vectors(live, shadow)?),
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            bail!("live/shadow post-GEMM cast applicability diverged")
        }
    };

    let mut compared_stages = vec![
        "activation_preparation",
        "weight_preparation",
        "raw_q_gemm_output",
    ];
    if live_trace.post_gemm_cast_distinct {
        compared_stages.push("immediate_post_gemm_cast_writeback");
    }

    let earliest_q_gemm_divergence_stage = if !activation_preparation.matched {
        "activation_preparation".to_string()
    } else if !weight_preparation.matched {
        "weight_preparation".to_string()
    } else if !raw_q_gemm_output.matched {
        "raw_q_gemm_output".to_string()
    } else if live_trace.post_gemm_cast_distinct
        && immediate_post_gemm_cast_writeback
            .as_ref()
            .is_some_and(|metrics| !metrics.matched)
    {
        "immediate_post_gemm_cast_writeback".to_string()
    } else {
        "matched_through_q_gemm_probe".to_string()
    };
    let first_mismatch_consistent_with =
        q_gemm_bucket_for_stage(&earliest_q_gemm_divergence_stage).to_string();
    let next_bounded_step = q_gemm_next_bounded_step(&earliest_q_gemm_divergence_stage);
    let runtime_forward_bf16_q_gemm_status_now = format!(
        "Compared stages: {}; earliest divergence stage: {}; suspect bucket: {}; next bounded step: {}.",
        compared_stages.join(" -> "),
        earliest_q_gemm_divergence_stage,
        first_mismatch_consistent_with,
        next_bounded_step
    );

    Ok(QGemmStatusSummary {
        schema_version: "runtime_forward_layer0_q_gemm_bf16_status/v1".into(),
        provenance: QGemmStatusProvenance {
            local_attn_norm_artifact_path: cli.local_attn_norm_artifact.clone(),
            model_root: attn_norm_artifact.provenance.model.clone(),
            oracle_checkpoint_dir: resolve_oracle_checkpoint_dir(Path::new(
                &attn_norm_artifact.provenance.model,
            )),
            visible_devices: visible_devices.to_string(),
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        case_id: case.id.clone(),
        input_token_ids: case.input_token_ids.clone(),
        hidden_size: live_trace.hidden_size,
        q_dim: live_trace.q_dim,
        candidate_branch_taken: live_trace.branch_taken,
        stage_bucket_mapping: q_gemm_stage_bucket_mapping(),
        compared_stages,
        activation_preparation,
        weight_preparation,
        raw_q_gemm_output,
        immediate_post_gemm_cast_writeback,
        post_gemm_cast_distinct: live_trace.post_gemm_cast_distinct,
        earliest_q_gemm_divergence_stage,
        first_mismatch_consistent_with,
        runtime_forward_bf16_q_gemm_status_now,
        next_bounded_step,
    })
}

fn build_pinned_prompt_intermediate_provenance(
    artifact: &PinnedArtifact,
    visible_devices: &str,
    max_model_len: usize,
    gpu_memory_utilization: f32,
) -> PinnedPromptIntermediateArtifactProvenance {
    PinnedPromptIntermediateArtifactProvenance {
        model: artifact.provenance.model.clone(),
        capture_source: "runner".to_string(),
        reference_kind: "local_candidate".to_string(),
        authority_level: "scaffold".to_string(),
        visible_devices: visible_devices.to_string(),
        max_model_len,
        gpu_memory_utilization,
        prompt_renderer: artifact
            .provenance
            .prompt_renderer
            .clone()
            .unwrap_or_else(|| "harmony_gpt_oss_rs".to_string()),
    }
}

fn build_pinned_prompt_qkv_artifact(
    local_artifact: &PinnedArtifact,
    local_case: &ArtifactCase,
    visible_devices: &str,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    candidate_capture: &Layer0Capture,
) -> Result<PinnedPromptIntermediateQkvArtifact> {
    let qkv_vector = last_token_vector(
        &candidate_capture.qkv_projection_output,
        candidate_capture.qkv_dim,
    )?;
    Ok(PinnedPromptIntermediateQkvArtifact {
        schema_version: "pinned-prompt-intermediate-artifact/v2".to_string(),
        suite_id: "developer-message".to_string(),
        boundary: "layer0_qkv_projection_output".to_string(),
        layer_idx: 0,
        provenance: build_pinned_prompt_intermediate_provenance(
            local_artifact,
            visible_devices,
            max_model_len,
            gpu_memory_utilization,
        ),
        cases: vec![PinnedPromptIntermediateQkvCase {
            id: local_case.id.clone(),
            input_token_ids: local_case.input_token_ids.clone(),
            hidden_size: local_case
                .hidden_size
                .unwrap_or(candidate_capture.transformer_output.len()),
            vector_size: qkv_vector.len(),
            q_dim: candidate_capture.q_dim,
            kv_dim: candidate_capture.kv_dim,
            qkv_dim: candidate_capture.qkv_dim,
            final_token_hidden_f32: qkv_vector,
        }],
    })
}

fn build_pinned_prompt_post_attention_residual_artifact(
    local_artifact: &PinnedArtifact,
    local_case: &ArtifactCase,
    visible_devices: &str,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    candidate_capture: &Layer0Capture,
) -> PinnedPromptIntermediateResidualArtifact {
    PinnedPromptIntermediateResidualArtifact {
        schema_version: "pinned-prompt-intermediate-artifact/v2".to_string(),
        suite_id: "developer-message".to_string(),
        boundary: "layer0_post_attention_residual".to_string(),
        layer_idx: 0,
        provenance: build_pinned_prompt_intermediate_provenance(
            local_artifact,
            visible_devices,
            max_model_len,
            gpu_memory_utilization,
        ),
        cases: vec![PinnedPromptIntermediateResidualCase {
            id: local_case.id.clone(),
            input_token_ids: local_case.input_token_ids.clone(),
            hidden_size: candidate_capture.post_attention_residual.len(),
            final_token_hidden_f32: candidate_capture.post_attention_residual.clone(),
        }],
    }
}

fn build_pinned_prompt_internal_qkv_artifact(
    local_artifact: &PinnedArtifact,
    local_case: &ArtifactCase,
    visible_devices: &str,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    candidate_capture: &Layer0Capture,
) -> Result<PinnedPromptIntermediateInternalQkvArtifact> {
    let internal_trace = candidate_capture
        .qkv_internal_trace
        .as_ref()
        .context("candidate internal qkv trace was not captured")?;
    let fused_qkv_pre_bias =
        last_token_vector(&internal_trace.fused_qkv_pre_bias, internal_trace.qkv_dim)?;
    let fused_qkv_post_bias =
        last_token_vector(&internal_trace.fused_qkv_post_bias, internal_trace.qkv_dim)?;
    let packed_qkv_output =
        last_token_vector(&internal_trace.packed_qkv_output, internal_trace.qkv_dim)?;
    Ok(PinnedPromptIntermediateInternalQkvArtifact {
        schema_version: "pinned-prompt-intermediate-artifact/v2".to_string(),
        suite_id: "developer-message".to_string(),
        boundary: "layer0_qkv_bf16_internal_trace".to_string(),
        layer_idx: 0,
        provenance: build_pinned_prompt_intermediate_provenance(
            local_artifact,
            visible_devices,
            max_model_len,
            gpu_memory_utilization,
        ),
        cases: vec![PinnedPromptIntermediateInternalQkvCase {
            id: local_case.id.clone(),
            input_token_ids: local_case.input_token_ids.clone(),
            hidden_size: local_case
                .hidden_size
                .unwrap_or(candidate_capture.transformer_output.len()),
            vector_size: packed_qkv_output.len(),
            branch_taken: internal_trace.branch_taken,
            q_dim: internal_trace.q_dim,
            kv_dim: internal_trace.kv_dim,
            qkv_dim: internal_trace.qkv_dim,
            fused_qkv_pre_bias_f32: fused_qkv_pre_bias,
            fused_qkv_post_bias_f32: fused_qkv_post_bias,
            packed_qkv_output_f32: packed_qkv_output,
        }],
    })
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn run_candidate_status(cli: &Cli) -> Result<()> {
    let (local_artifact, local_case) =
        load_single_case_artifact(&cli.local_residual_input_artifact)?;

    let (official_post_attention_artifact, official_post_attention_case) =
        load_single_case_artifact(&cli.official_post_attention_residual_artifact)?;

    let (official_transformer_artifact, official_transformer_case) =
        load_single_case_artifact(&cli.official_transformer_output_artifact)?;
    let expected_tokens = official_post_attention_case.input_token_ids.clone();
    validate_exact_case_artifact(
        &local_artifact,
        &local_case,
        "layer0_residual_input",
        "developer-message-user-smoke",
        &expected_tokens,
    )?;
    validate_exact_case_artifact(
        &official_post_attention_artifact,
        &official_post_attention_case,
        "layer0_post_attention_residual",
        "developer-message-user-smoke",
        &expected_tokens,
    )?;
    validate_exact_case_artifact(
        &official_transformer_artifact,
        &official_transformer_case,
        "transformer_layer_output",
        "developer-message-user-smoke",
        &expected_tokens,
    )?;

    let model_root = Path::new(&local_artifact.provenance.model);
    let max_model_len = local_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = local_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = local_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_artifact = %cli.local_residual_input_artifact.display(),
        "starting exact-case layer0 BF16 dense-QKV status run"
    );

    let baseline_capture = run_trial(
        model_root,
        max_model_len,
        gpu_memory_utilization,
        &local_case.input_token_ids,
        false,
        false,
    )?;
    let candidate_capture = run_trial(
        model_root,
        max_model_len,
        gpu_memory_utilization,
        &local_case.input_token_ids,
        true,
        true,
    )?;

    let baseline_post_attention_residual_vs_official = compare_vectors(
        &baseline_capture.post_attention_residual,
        official_post_attention_case
            .final_token_hidden_f32
            .as_ref()
            .context("official post-attention residual artifact missing final_token_hidden_f32")?,
    )?;
    let baseline_transformer_output_vs_official = compare_vectors(
        &baseline_capture.transformer_output,
        official_transformer_case
            .final_token_hidden_f32
            .as_ref()
            .context("official transformer output artifact missing final_token_hidden_f32")?,
    )?;
    let candidate_post_attention_residual_vs_official = compare_vectors(
        &candidate_capture.post_attention_residual,
        official_post_attention_case
            .final_token_hidden_f32
            .as_ref()
            .context("official post-attention residual artifact missing final_token_hidden_f32")?,
    )?;
    let candidate_transformer_output_vs_official = compare_vectors(
        &candidate_capture.transformer_output,
        official_transformer_case
            .final_token_hidden_f32
            .as_ref()
            .context("official transformer output artifact missing final_token_hidden_f32")?,
    )?;

    let baseline = TrialSummary {
        enabled: false,
        post_attention_residual_vs_official: baseline_post_attention_residual_vs_official,
        transformer_output_vs_official: baseline_transformer_output_vs_official,
        materially_improves_post_attention_residual: false,
        materially_improves_transformer_output: false,
    };
    let candidate = TrialSummary {
        enabled: true,
        materially_improves_post_attention_residual: materially_improves(
            &baseline.post_attention_residual_vs_official,
            &candidate_post_attention_residual_vs_official,
        ),
        materially_improves_transformer_output: materially_improves(
            &baseline.transformer_output_vs_official,
            &candidate_transformer_output_vs_official,
        ),
        post_attention_residual_vs_official: candidate_post_attention_residual_vs_official,
        transformer_output_vs_official: candidate_transformer_output_vs_official,
    };
    let next_bounded_step = next_bounded_step(&candidate);
    let pinned_prompt_qkv_artifact = build_pinned_prompt_qkv_artifact(
        &local_artifact,
        &local_case,
        &visible_devices,
        max_model_len,
        gpu_memory_utilization,
        &candidate_capture,
    )?;
    let pinned_prompt_post_attention_residual_artifact =
        build_pinned_prompt_post_attention_residual_artifact(
            &local_artifact,
            &local_case,
            &visible_devices,
            max_model_len,
            gpu_memory_utilization,
            &candidate_capture,
        );
    let pinned_prompt_internal_qkv_artifact = build_pinned_prompt_internal_qkv_artifact(
        &local_artifact,
        &local_case,
        &visible_devices,
        max_model_len,
        gpu_memory_utilization,
        &candidate_capture,
    )?;
    let summary = StatusSummary {
        schema_version: "runtime_forward_layer0_qkv_bf16_candidate_status/v1".into(),
        provenance: StatusProvenance {
            local_residual_input_artifact_path: cli.local_residual_input_artifact.clone(),
            official_post_attention_residual_artifact_path: cli
                .official_post_attention_residual_artifact
                .clone(),
            official_transformer_output_artifact_path: cli
                .official_transformer_output_artifact
                .clone(),
            model_root: local_artifact.provenance.model.clone(),
            visible_devices,
            max_model_len,
            gpu_memory_utilization,
            candidate_env_var: BF16_DENSE_QKV_ENV.to_string(),
        },
        exact_case: ExactCaseSummary {
            case_id: local_case.id.clone(),
            input_token_ids: local_case.input_token_ids.clone(),
            hidden_size: local_case
                .hidden_size
                .or(official_post_attention_case.hidden_size)
                .or(official_transformer_case.hidden_size)
                .context("exact case missing hidden_size in all artifacts")?,
            boundary: local_artifact.boundary.clone(),
            layer_idx: local_artifact.layer_idx,
        },
        baseline,
        candidate,
        next_bounded_step,
    };
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::CandidateStatus.default_output());

    write_pretty_json(&cli.candidate_qkv_artifact, &pinned_prompt_qkv_artifact)?;
    write_pretty_json(
        &cli.candidate_post_attention_residual_artifact,
        &pinned_prompt_post_attention_residual_artifact,
    )?;
    write_pretty_json(
        &cli.candidate_internal_trace_artifact,
        &pinned_prompt_internal_qkv_artifact,
    )?;
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));
    let activation_source = attn_norm_case
        .final_token_hidden_f32
        .as_ref()
        .context("local attention-norm artifact missing final_token_hidden_f32")?;

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        "starting exact-case layer0 BF16 Q GEMM status run"
    );

    let live_trace = run_q_gemm_trial(
        model_root,
        max_model_len,
        gpu_memory_utilization,
        &attn_norm_case.input_token_ids,
    )?;
    if !live_trace.branch_taken {
        bail!("live layer0 q gemm trace did not take the bf16 fused-qkv branch");
    }
    let shadow = build_layer0_q_gemm_shadow(
        model_root,
        live_trace.hidden_size,
        live_trace.q_dim,
        activation_source,
        live_trace.post_gemm_cast_distinct,
    )?;
    let summary = build_q_gemm_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        &live_trace,
        &shadow,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_helper_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        "starting exact-case layer0 BF16 Q GEMM helper status run"
    );

    let live_trace = run_q_gemm_trial(
        model_root,
        max_model_len,
        gpu_memory_utilization,
        &attn_norm_case.input_token_ids,
    )?;
    if !live_trace.branch_taken {
        bail!("live layer0 q gemm helper trace did not take the bf16 fused-qkv branch");
    }
    let summary = build_q_gemm_helper_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        &live_trace,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmHelperStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_live_vs_isolated_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let helper_status_artifact =
        load_and_validate_existing_q_gemm_helper_status(cli, &attn_norm_case)?;
    let live_vs_isolated_artifact = load_existing_q_gemm_live_vs_isolated_status_artifact(
        &cli.q_gemm_live_vs_isolated_status_artifact,
    )?;
    let helper_contract_artifact = load_existing_q_gemm_helper_contract_status_artifact(
        &cli.q_gemm_helper_contract_status_artifact,
    )?;
    if helper_contract_artifact.case_id != attn_norm_case.id {
        bail!(
            "helper-contract artifact case id mismatch: expected {}, found {}",
            attn_norm_case.id,
            helper_contract_artifact.case_id
        );
    }
    if helper_contract_artifact.input_token_ids != attn_norm_case.input_token_ids {
        bail!("helper-contract artifact token ids do not match the exact smoke case");
    }
    if helper_contract_artifact.helper_contract.alpha_beta_dtype != "f32" {
        bail!(
            "helper-contract artifact alpha/beta dtype changed: expected f32, found {}",
            helper_contract_artifact.helper_contract.alpha_beta_dtype
        );
    }
    if !helper_contract_artifact.raw_exact_contract_reproduces_helper {
        bail!("helper-contract artifact exact contract replay no longer reproduces the helper");
    }
    if helper_contract_artifact.best_explained_by != "alpha_beta_or_descriptor_configuration" {
        bail!(
            "helper-contract artifact classification changed: expected alpha_beta_or_descriptor_configuration, found {}",
            helper_contract_artifact.best_explained_by
        );
    }

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        q_gemm_helper_status_artifact = %cli.q_gemm_helper_status_artifact.display(),
        q_gemm_helper_contract_status_artifact = %cli.q_gemm_helper_contract_status_artifact.display(),
        "starting exact-case layer0 BF16 Q GEMM live-vs-isolated status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_q_gemm_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 q gemm live-vs-isolated trace did not take the bf16 fused-qkv branch");
    }
    if helper_status_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "helper-status artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            helper_status_artifact.hidden_size
        );
    }
    if helper_status_artifact.q_dim != live_trace.q_dim {
        bail!(
            "helper-status artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            helper_status_artifact.q_dim
        );
    }
    if helper_status_artifact.classification != "shared_bf16_gemm_helper" {
        bail!(
            "helper-status artifact classification changed: expected shared_bf16_gemm_helper, found {}",
            helper_status_artifact.classification
        );
    }
    if live_vs_isolated_artifact.case_id != attn_norm_case.id {
        bail!(
            "live-vs-isolated artifact case id mismatch: expected {}, found {}",
            attn_norm_case.id,
            live_vs_isolated_artifact.case_id
        );
    }
    if live_vs_isolated_artifact.input_token_ids != attn_norm_case.input_token_ids {
        bail!("live-vs-isolated artifact token ids do not match the exact smoke case");
    }
    if live_vs_isolated_artifact.classification != "pre_helper_input_plumbing_mismatch" {
        bail!(
            "live-vs-isolated artifact classification changed: expected pre_helper_input_plumbing_mismatch, found {}",
            live_vs_isolated_artifact.classification
        );
    }
    if live_vs_isolated_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "live-vs-isolated artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            live_vs_isolated_artifact.hidden_size
        );
    }
    if live_vs_isolated_artifact.q_dim != live_trace.q_dim {
        bail!(
            "live-vs-isolated artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            live_vs_isolated_artifact.q_dim
        );
    }

    let activation_row_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_activation_bf16_bits);
    let q_weight_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_q_weight_bf16_bits);
    let cpu_reference_q_only_gemm = build_live_q_gemm_reference(&live_trace)?;

    let stream = worker.gpu_stream().clone();
    let activation_gpu = stream
        .clone_htod(&activation_row_bf16)
        .map_err(|e| anyhow::anyhow!("live-vs-isolated activation htod failed: {e}"))?;
    let q_weight_gpu = stream
        .clone_htod(&q_weight_bf16)
        .map_err(|e| anyhow::anyhow!("live-vs-isolated q weight htod failed: {e}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(live_trace.q_dim)
        .map_err(|e| anyhow::anyhow!("live-vs-isolated output alloc failed: {e}"))?;
    worker.blas().hgemm_bf16(
        1,
        live_trace.q_dim,
        live_trace.hidden_size,
        bf16::ONE,
        &activation_gpu,
        &q_weight_gpu,
        bf16::ZERO,
        &mut output_gpu,
    )?;
    let isolated_live_helper_replay_q_only_gemm: Vec<f32> = stream
        .clone_dtoh(&output_gpu)
        .map_err(|e| anyhow::anyhow!("live-vs-isolated output dtoh failed: {e}"))?
        .iter()
        .map(|value| value.to_f32())
        .collect();
    let live_standalone_bf16_q_only_gemm = live_trace.standalone_q_gemm_output.clone();
    let live_fused_bf16_qkv_q_slice = live_trace.raw_q_gemm_output.clone();

    let isolated_helper_replay_vs_cpu_reference = compare_vectors(
        &isolated_live_helper_replay_q_only_gemm,
        &cpu_reference_q_only_gemm,
    )?;
    let live_standalone_vs_isolated_helper_replay = compare_vectors(
        &live_standalone_bf16_q_only_gemm,
        &isolated_live_helper_replay_q_only_gemm,
    )?;
    let live_fused_q_vs_live_standalone = compare_vectors(
        &live_fused_bf16_qkv_q_slice,
        &live_standalone_bf16_q_only_gemm,
    )?;
    let live_fused_q_vs_isolated_helper_replay = compare_vectors(
        &live_fused_bf16_qkv_q_slice,
        &isolated_live_helper_replay_q_only_gemm,
    )?;

    let summary = build_q_gemm_live_vs_isolated_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        &live_trace,
        &helper_contract_artifact,
        isolated_helper_replay_vs_cpu_reference,
        live_standalone_vs_isolated_helper_replay,
        live_fused_q_vs_live_standalone,
        live_fused_q_vs_isolated_helper_replay,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmLiveVsIsolatedStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_pre_helper_plumbing_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let live_vs_isolated_artifact = load_existing_q_gemm_live_vs_isolated_status_artifact(
        &cli.q_gemm_live_vs_isolated_status_artifact,
    )?;
    let helper_contract_artifact = load_existing_q_gemm_helper_contract_status_artifact(
        &cli.q_gemm_helper_contract_status_artifact,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        q_gemm_live_vs_isolated_status_artifact = %cli
            .q_gemm_live_vs_isolated_status_artifact
            .display(),
        q_gemm_helper_contract_status_artifact = %cli.q_gemm_helper_contract_status_artifact.display(),
        "starting exact-case layer0 BF16 Q GEMM pre-helper plumbing status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_q_gemm_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!(
            "live layer0 q gemm pre-helper plumbing trace did not take the bf16 fused-qkv branch"
        );
    }
    if live_vs_isolated_artifact.case_id != attn_norm_case.id {
        bail!(
            "live-vs-isolated artifact case id mismatch: expected {}, found {}",
            attn_norm_case.id,
            live_vs_isolated_artifact.case_id
        );
    }
    if live_vs_isolated_artifact.input_token_ids != attn_norm_case.input_token_ids {
        bail!("live-vs-isolated artifact token ids do not match the exact smoke case");
    }
    if live_vs_isolated_artifact.classification != "pre_helper_input_plumbing_mismatch" {
        bail!(
            "live-vs-isolated artifact classification changed: expected pre_helper_input_plumbing_mismatch, found {}",
            live_vs_isolated_artifact.classification
        );
    }
    if live_vs_isolated_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "live-vs-isolated artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            live_vs_isolated_artifact.hidden_size
        );
    }
    if live_vs_isolated_artifact.q_dim != live_trace.q_dim {
        bail!(
            "live-vs-isolated artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            live_vs_isolated_artifact.q_dim
        );
    }
    if helper_contract_artifact.case_id != attn_norm_case.id {
        bail!(
            "helper-contract artifact case id mismatch: expected {}, found {}",
            attn_norm_case.id,
            helper_contract_artifact.case_id
        );
    }
    if helper_contract_artifact.input_token_ids != attn_norm_case.input_token_ids {
        bail!("helper-contract artifact token ids do not match the exact smoke case");
    }
    if helper_contract_artifact.helper_contract.alpha_beta_dtype != "f32" {
        bail!(
            "helper-contract artifact alpha/beta dtype changed: expected f32, found {}",
            helper_contract_artifact.helper_contract.alpha_beta_dtype
        );
    }
    if !helper_contract_artifact.raw_exact_contract_reproduces_helper {
        bail!("helper-contract artifact exact contract replay no longer reproduces the helper");
    }
    if helper_contract_artifact.best_explained_by != "alpha_beta_or_descriptor_configuration" {
        bail!(
            "helper-contract artifact classification changed: expected alpha_beta_or_descriptor_configuration, found {}",
            helper_contract_artifact.best_explained_by
        );
    }
    if helper_contract_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "helper-contract artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            helper_contract_artifact.hidden_size
        );
    }
    if helper_contract_artifact.q_dim != live_trace.q_dim {
        bail!(
            "helper-contract artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            helper_contract_artifact.q_dim
        );
    }

    let standalone_activation_vs_isolated_replay = compare_bf16_bit_slices(
        &live_trace.standalone_activation_bf16_bits,
        &live_trace.fused_activation_bf16_bits,
    )?;
    let fused_activation_vs_standalone = compare_bf16_bit_slices(
        &live_trace.fused_activation_bf16_bits,
        &live_trace.standalone_activation_bf16_bits,
    )?;
    let standalone_weight_vs_isolated_replay = compare_bf16_bit_slices(
        &live_trace.standalone_q_weight_bf16_bits,
        &live_trace.fused_q_slice_bf16_bits,
    )?;
    let fused_q_slice_weight_vs_standalone = compare_bf16_bit_slices(
        &live_trace.fused_q_slice_bf16_bits,
        &live_trace.standalone_q_weight_bf16_bits,
    )?;

    let summary = build_q_gemm_pre_helper_plumbing_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        &live_trace,
        &live_vs_isolated_artifact,
        standalone_activation_vs_isolated_replay,
        fused_activation_vs_standalone,
        standalone_weight_vs_isolated_replay,
        fused_q_slice_weight_vs_standalone,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmPreHelperPlumbingStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_helper_invocation_output_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let helper_status_artifact =
        load_existing_q_gemm_helper_status_artifact(&cli.q_gemm_helper_status_artifact)?;
    let live_vs_isolated_artifact = load_existing_q_gemm_live_vs_isolated_status_artifact(
        &cli.q_gemm_live_vs_isolated_status_artifact,
    )?;
    let helper_contract_artifact = load_existing_q_gemm_helper_contract_status_artifact(
        &cli.q_gemm_helper_contract_status_artifact,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        q_gemm_helper_status_artifact = %cli.q_gemm_helper_status_artifact.display(),
        q_gemm_live_vs_isolated_status_artifact = %cli
            .q_gemm_live_vs_isolated_status_artifact
            .display(),
        q_gemm_helper_contract_status_artifact = %cli
            .q_gemm_helper_contract_status_artifact
            .display(),
        "starting exact-case layer0 BF16 Q GEMM helper invocation/output status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_q_gemm_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!(
            "live layer0 q gemm helper invocation/output trace did not take the bf16 fused-qkv branch"
        );
    }
    if helper_status_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "helper-status artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            helper_status_artifact.hidden_size
        );
    }
    if helper_status_artifact.q_dim != live_trace.q_dim {
        bail!(
            "helper-status artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            helper_status_artifact.q_dim
        );
    }

    let activation_full_bf16 =
        bf16_bits_to_bf16_slice(&live_trace.standalone_activation_full_bf16_bits);
    let q_weight_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_q_weight_bf16_bits);

    let stream = worker.gpu_stream().clone();
    let activation_gpu = stream
        .clone_htod(&activation_full_bf16)
        .map_err(|e| anyhow::anyhow!("helper invocation activation htod failed: {e}"))?;
    let q_weight_gpu = stream
        .clone_htod(&q_weight_bf16)
        .map_err(|e| anyhow::anyhow!("helper invocation q weight htod failed: {e}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(live_trace.num_tokens * live_trace.q_dim)
        .map_err(|e| anyhow::anyhow!("helper invocation output alloc failed: {e}"))?;
    let isolated_helper_invocation_state = build_bf16_gemm_invocation_record(
        worker.blas(),
        live_trace.num_tokens,
        live_trace.q_dim,
        live_trace.hidden_size,
    )?;
    worker.blas().hgemm_bf16(
        live_trace.num_tokens,
        live_trace.q_dim,
        live_trace.hidden_size,
        bf16::ONE,
        &activation_gpu,
        &q_weight_gpu,
        bf16::ZERO,
        &mut output_gpu,
    )?;
    let helper_output_bf16: Vec<bf16> = stream
        .clone_dtoh(&output_gpu)
        .map_err(|e| anyhow::anyhow!("helper invocation output dtoh failed: {e}"))?;
    let helper_output_bf16_bits = bf16_slice_to_bits(&helper_output_bf16);
    let live_output_vs_isolated_replay = compare_bf16_bit_slices(
        &live_trace.standalone_q_gemm_output_full_bf16_bits,
        &helper_output_bf16_bits,
    )?;

    let summary = build_q_gemm_helper_invocation_output_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &helper_status_artifact,
        &live_vs_isolated_artifact,
        &helper_contract_artifact,
        &visible_devices,
        &live_trace,
        live_trace
            .standalone_helper_invocation_state
            .clone()
            .context("live trace missing standalone helper invocation state")?,
        isolated_helper_invocation_state,
        live_output_vs_isolated_replay,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmHelperInvocationOutputStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_gemm_helper_path_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        k_projection_pre_bias_status_artifact = %cli
            .k_projection_pre_bias_status_artifact
            .display(),
        "starting exact-case layer0 BF16 K helper-path status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 k helper-path trace did not take the bf16 fused-qkv branch");
    }

    let k_projection_pre_bias_artifact = load_and_validate_existing_k_projection_pre_bias_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
    )?;

    let activation_full_bf16 =
        bf16_bits_to_bf16_slice(&live_trace.standalone_activation_full_bf16_bits);
    let standalone_k_weight_bf16 =
        bf16_bits_to_bf16_slice(&live_trace.standalone_k_weight_bf16_bits);
    let cpu_reference_k_only_gemm = cpu_bf16_row_major_gemm(
        &activation_full_bf16,
        &standalone_k_weight_bf16,
        live_trace.num_tokens,
        live_trace.kv_dim,
        exact_hidden_size,
    )?;
    let cpu_reference_k_only_gemm_bits = bf16_slice_to_bits(&cpu_reference_k_only_gemm);

    let stream = worker.gpu_stream().clone();
    let activation_gpu = stream
        .clone_htod(&activation_full_bf16)
        .map_err(|e| anyhow::anyhow!("k helper-path activation htod failed: {e}"))?;
    let k_weight_gpu = stream
        .clone_htod(&standalone_k_weight_bf16)
        .map_err(|e| anyhow::anyhow!("k helper-path k weight htod failed: {e}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(live_trace.num_tokens * live_trace.kv_dim)
        .map_err(|e| anyhow::anyhow!("k helper-path output alloc failed: {e}"))?;
    worker.blas().hgemm_bf16(
        live_trace.num_tokens,
        live_trace.kv_dim,
        exact_hidden_size,
        bf16::ONE,
        &activation_gpu,
        &k_weight_gpu,
        bf16::ZERO,
        &mut output_gpu,
    )?;
    let isolated_helper_replay_output: Vec<bf16> = stream
        .clone_dtoh(&output_gpu)
        .map_err(|e| anyhow::anyhow!("k helper-path output dtoh failed: {e}"))?;
    let isolated_helper_replay_output_bits = bf16_slice_to_bits(&isolated_helper_replay_output);

    let isolated_helper_replay_vs_cpu_reference = compare_bf16_bit_slices(
        &isolated_helper_replay_output_bits,
        &cpu_reference_k_only_gemm_bits,
    )?;
    let live_standalone_vs_isolated_helper_replay = compare_bf16_bit_slices(
        &live_trace.standalone_k_gemm_output_full_bf16_bits,
        &isolated_helper_replay_output_bits,
    )?;
    let live_fused_k_vs_live_standalone = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &live_trace.standalone_k_gemm_output_full_bf16_bits,
    )?;
    let live_fused_k_vs_isolated_helper_replay = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &isolated_helper_replay_output_bits,
    )?;

    let summary = build_k_gemm_helper_path_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        &k_projection_pre_bias_artifact,
        isolated_helper_replay_vs_cpu_reference,
        live_standalone_vs_isolated_helper_replay,
        live_fused_k_vs_live_standalone,
        live_fused_k_vs_isolated_helper_replay,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KGemmHelperPathStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_gemm_fused_split_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        k_projection_pre_bias_status_artifact = %cli
            .k_projection_pre_bias_status_artifact
            .display(),
        k_gemm_helper_path_status_artifact = %cli
            .k_gemm_helper_path_status_artifact
            .display(),
        "starting exact-case layer0 BF16 K fused-split status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 k fused-split trace did not take the bf16 fused-qkv branch");
    }

    let k_projection_pre_bias_artifact = load_and_validate_existing_k_projection_pre_bias_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
    )?;
    let k_gemm_helper_path_artifact = load_and_validate_existing_k_gemm_helper_path_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;

    let fused_pre_bias_vs_standalone_pre_bias = compare_bf16_bit_slices(
        &live_trace.fused_k_pre_bias_bf16_bits,
        &live_trace.standalone_k_pre_bias_bf16_bits,
    )?;
    let fused_post_bias_vs_standalone_post_bias = compare_bf16_bit_slices(
        &live_trace.fused_k_post_bias_bf16_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;
    let downstream_fused_k_vs_standalone_post_bias = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;

    let summary = build_k_gemm_fused_split_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        &k_projection_pre_bias_artifact,
        &k_gemm_helper_path_artifact,
        fused_pre_bias_vs_standalone_pre_bias,
        fused_post_bias_vs_standalone_post_bias,
        downstream_fused_k_vs_standalone_post_bias,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KGemmFusedSplitStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_gemm_fused_qkv_readout_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        k_gemm_helper_path_status_artifact = %cli
            .k_gemm_helper_path_status_artifact
            .display(),
        k_gemm_fused_split_status_artifact = %cli
            .k_gemm_fused_split_status_artifact
            .display(),
        "starting exact-case layer0 BF16 K fused-QKV readout status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 k fused-QKV readout trace did not take the bf16 fused-qkv branch");
    }

    let k_gemm_helper_path_artifact = load_and_validate_existing_k_gemm_helper_path_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;
    let k_gemm_fused_split_artifact = load_and_validate_existing_k_gemm_fused_split_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;

    let combined_buffer_vs_expected = compare_bf16_bit_slices(
        &live_trace.fused_qkv_post_bias_combined_bf16_bits,
        &live_trace.expected_standalone_qkv_post_bias_combined_bf16_bits,
    )?;
    let downstream_fused_k_vs_expected_k = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;

    let summary = build_k_gemm_fused_qkv_readout_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        &k_gemm_helper_path_artifact,
        &k_gemm_fused_split_artifact,
        combined_buffer_vs_expected,
        downstream_fused_k_vs_expected_k,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KGemmFusedQkvReadoutStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_gemm_fused_k_readout_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        k_gemm_helper_path_status_artifact = %cli
            .k_gemm_helper_path_status_artifact
            .display(),
        k_gemm_fused_split_status_artifact = %cli
            .k_gemm_fused_split_status_artifact
            .display(),
        k_gemm_fused_qkv_readout_status_artifact = %cli
            .k_gemm_fused_qkv_readout_status_artifact
            .display(),
        "starting exact-case layer0 BF16 K fused readout status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 k fused readout trace did not take the bf16 fused-qkv branch");
    }

    let k_gemm_helper_path_artifact = load_and_validate_existing_k_gemm_helper_path_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;
    let k_gemm_fused_split_artifact = load_and_validate_existing_k_gemm_fused_split_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;
    let k_gemm_fused_qkv_readout_artifact =
        load_and_validate_existing_k_gemm_fused_qkv_readout_status(
            cli,
            &attn_norm_case,
            exact_hidden_size,
            live_trace.q_dim,
            live_trace.kv_dim,
        )?;

    let canonical_contiguous_k_bits = contiguous_k_slice_from_combined_qkv_bits(
        &live_trace.fused_qkv_post_bias_combined_bf16_bits,
        live_trace.num_tokens,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;
    let row_major_token_qkv_k_bits = row_major_k_slice_from_combined_qkv_bits(
        &live_trace.fused_qkv_post_bias_combined_bf16_bits,
        live_trace.num_tokens,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;
    let runtime_cache_input_shape_k_bits = canonical_contiguous_k_bits.clone();

    let current_live_fused_k_readout = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;
    let canonical_contiguous_k_slice = compare_bf16_bit_slices(
        &canonical_contiguous_k_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;
    let row_major_token_qkv_slice = compare_bf16_bit_slices(
        &row_major_token_qkv_k_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;
    let runtime_cache_input_shape = compare_bf16_bit_slices(
        &runtime_cache_input_shape_k_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;

    let summary = build_k_gemm_fused_k_readout_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        &k_gemm_helper_path_artifact,
        &k_gemm_fused_split_artifact,
        &k_gemm_fused_qkv_readout_artifact,
        current_live_fused_k_readout,
        canonical_contiguous_k_slice,
        row_major_token_qkv_slice,
        runtime_cache_input_shape,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KGemmFusedKReadoutStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_consumption_rope_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        k_gemm_fused_k_readout_status_artifact = %cli
            .k_gemm_fused_k_readout_status_artifact
            .display(),
        "starting exact-case layer0 K post-RoPE pre-cache consumption status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 K consumption trace did not take the bf16 fused-qkv branch");
    }

    let k_gemm_fused_k_readout_artifact = load_and_validate_existing_k_gemm_fused_k_readout_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
    )?;

    let pre_rope_k_readout_vs_expected = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;
    let expected_len = live_trace
        .num_tokens
        .checked_mul(live_trace.kv_dim)
        .context("expected post-rope K length overflow")?;
    if live_trace.k_post_rope_pre_cache_f16_bits.len() != expected_len {
        bail!(
            "post-RoPE K f16 bits length mismatch: expected {}, found {}",
            expected_len,
            live_trace.k_post_rope_pre_cache_f16_bits.len()
        );
    }
    if live_trace.k_post_rope_pre_cache_f32.len() != expected_len {
        bail!(
            "post-RoPE K f32 length mismatch: expected {}, found {}",
            expected_len,
            live_trace.k_post_rope_pre_cache_f32.len()
        );
    }

    let summary = build_k_consumption_rope_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        &k_gemm_fused_k_readout_artifact,
        pre_rope_k_readout_vs_expected,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KConsumptionRopeStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_rope_convention_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let (official_artifact, official_case) =
        load_single_case_artifact(&cli.official_k_post_rope_pre_cache_artifact)?;
    validate_exact_case_artifact(
        &official_artifact,
        &official_case,
        "layer0_k_post_rope_pre_cache",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let official_post_rope_k = official_case
        .final_token_hidden_f32
        .as_ref()
        .context("official K post-RoPE artifact missing final_token_hidden_f32")?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        k_consumption_rope_status_artifact = %cli.k_consumption_rope_status_artifact.display(),
        official_k_post_rope_pre_cache_artifact = %cli.official_k_post_rope_pre_cache_artifact.display(),
        "starting exact-case layer0 K host-side RoPE convention sweep"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker_config = build_worker_config(model_root, max_model_len, gpu_memory_utilization)?;
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 K RoPE convention trace did not take the bf16 fused-qkv branch");
    }

    let local_k_consumption_artifact = load_and_validate_existing_k_consumption_rope_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
        live_trace.num_tokens,
    )?;
    let expected_len = live_trace
        .num_tokens
        .checked_mul(live_trace.kv_dim)
        .context("expected K post-RoPE length overflow")?;
    if official_post_rope_k.len() != expected_len {
        bail!(
            "official post-RoPE K length mismatch: expected {}, found {}",
            expected_len,
            official_post_rope_k.len()
        );
    }

    let pre_rope_k_readout_vs_expected = compare_bf16_bit_slices(
        &live_trace.fused_k_slice_bf16_bits,
        &live_trace.standalone_k_post_bias_bf16_bits,
    )?;
    if !pre_rope_k_readout_vs_expected.matched {
        bail!("current live pre-RoPE K readout no longer matches expected standalone K");
    }

    let q_len = live_trace
        .num_tokens
        .checked_mul(live_trace.q_dim)
        .context("Q length overflow while slicing pre-RoPE K")?;
    let k_end = q_len
        .checked_add(expected_len)
        .context("K end overflow while slicing pre-RoPE K")?;
    if live_trace.qkv_projection_output.len() < k_end {
        bail!(
            "qkv_projection_output too short for contiguous pre-RoPE K slice: need {}, found {}",
            k_end,
            live_trace.qkv_projection_output.len()
        );
    }
    let pre_rope_k = &live_trace.qkv_projection_output[q_len..k_end];
    let baseline_local_runtime_vs_official = compare_vectors(
        &local_k_consumption_artifact.k_post_rope_pre_cache_f32,
        official_post_rope_k,
    )?;
    if baseline_local_runtime_vs_official.matched {
        bail!("local post-RoPE K already matches official; convention sweep is no longer needed");
    }

    let max_table_pos = live_trace
        .num_tokens
        .checked_add(2)
        .context("host RoPE table max_pos overflow")?;
    let mut table_cache: HashMap<&'static str, (Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut variants = Vec::new();
    for spec in k_rope_variant_specs() {
        let table_key = match spec.table_precision {
            RopeTablePrecision::F32 => "f32",
            RopeTablePrecision::F16 => "f16",
            RopeTablePrecision::Bf16 => "bf16",
        };
        if !table_cache.contains_key(table_key) {
            table_cache.insert(
                table_key,
                build_host_rope_tables(&worker_config, max_table_pos, spec.table_precision),
            );
        }
        let (cos_table, sin_table) = table_cache
            .get(table_key)
            .context("missing cached host RoPE table")?;
        let variant_output = apply_host_k_rope_variant(
            pre_rope_k,
            &worker_config,
            live_trace.num_tokens,
            live_trace.kv_dim,
            cos_table,
            sin_table,
            spec,
        )?;
        let metrics = compare_vectors(&variant_output, official_post_rope_k)?;
        let materially_collapses_mismatch =
            k_rope_variant_materially_collapses(&baseline_local_runtime_vs_official, &metrics);
        variants.push(KRopeConventionVariant {
            variant_name: spec.name.to_string(),
            vs_official_post_rope_k: metrics,
            materially_collapses_mismatch,
        });
    }

    let summary = build_k_rope_convention_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        &local_k_consumption_artifact,
        official_post_rope_k,
        pre_rope_k_readout_vs_expected,
        baseline_local_runtime_vs_official,
        variants,
    )?;
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KRopeConventionStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_k_rope_implementation_status(cli: &Cli) -> Result<()> {
    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    let exact_hidden_size = attn_norm_case
        .hidden_size
        .context("exact smoke case missing hidden_size")?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;

    let (official_artifact, official_case) =
        load_single_case_artifact(&cli.official_k_post_rope_pre_cache_artifact)?;
    validate_exact_case_artifact(
        &official_artifact,
        &official_case,
        "layer0_k_post_rope_pre_cache",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let official_post_rope_k = official_case
        .final_token_hidden_f32
        .as_ref()
        .context("official K post-RoPE artifact missing final_token_hidden_f32")?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        k_rope_convention_status_artifact = %cli.k_rope_convention_status_artifact.display(),
        k_consumption_rope_status_artifact = %cli.k_consumption_rope_status_artifact.display(),
        official_k_post_rope_pre_cache_artifact = %cli.official_k_post_rope_pre_cache_artifact.display(),
        "starting exact-case layer0 K runtime RoPE implementation status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker_config = build_worker_config(model_root, max_model_len, gpu_memory_utilization)?;
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_qkv_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 K RoPE implementation trace did not take the bf16 fused-qkv branch");
    }

    let _local_k_consumption_artifact = load_and_validate_existing_k_consumption_rope_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
        live_trace.num_tokens,
    )?;
    let _k_rope_convention_artifact = load_and_validate_existing_k_rope_convention_status(
        cli,
        &attn_norm_case,
        exact_hidden_size,
        live_trace.q_dim,
        live_trace.kv_dim,
        live_trace.num_tokens,
    )?;

    let expected_len = live_trace
        .num_tokens
        .checked_mul(live_trace.kv_dim)
        .context("expected K post-RoPE length overflow")?;
    if official_post_rope_k.len() != expected_len {
        bail!(
            "official post-RoPE K length mismatch: expected {}, found {}",
            expected_len,
            official_post_rope_k.len()
        );
    }
    let k_rope_debug = live_trace
        .k_rope_debug
        .as_ref()
        .context("live layer0 qkv trace missing k_rope_debug capture")?;
    if k_rope_debug.final_token_k_pre_rope_f32.len() != live_trace.kv_dim {
        bail!(
            "final-token pre-RoPE K length mismatch: expected {}, found {}",
            live_trace.kv_dim,
            k_rope_debug.final_token_k_pre_rope_f32.len()
        );
    }
    if k_rope_debug.final_token_k_post_rope_f32.len() != live_trace.kv_dim {
        bail!(
            "final-token post-RoPE K length mismatch: expected {}, found {}",
            live_trace.kv_dim,
            k_rope_debug.final_token_k_post_rope_f32.len()
        );
    }
    if k_rope_debug.live_runtime_cos_row_f32.len() != worker_config.head_dim
        || k_rope_debug.live_runtime_sin_row_f32.len() != worker_config.head_dim
    {
        bail!(
            "live runtime factor row length mismatch for head_dim {}: cos {}, sin {}",
            worker_config.head_dim,
            k_rope_debug.live_runtime_cos_row_f32.len(),
            k_rope_debug.live_runtime_sin_row_f32.len()
        );
    }

    let final_token_start = k_rope_debug
        .final_token_index
        .checked_mul(live_trace.kv_dim)
        .context("official final-token K start overflow")?;
    let final_token_end = final_token_start
        .checked_add(live_trace.kv_dim)
        .context("official final-token K end overflow")?;
    let official_final_token_k = &official_post_rope_k[final_token_start..final_token_end];

    let (host_cos_row, host_sin_row) = host_generated_rope_rows_for_position(
        &worker_config,
        k_rope_debug.effective_position_id.max(0) as usize,
    )?;
    let live_factors = combined_factor_row(
        &k_rope_debug.live_runtime_cos_row_f32,
        &k_rope_debug.live_runtime_sin_row_f32,
    )?;
    let host_factors = combined_factor_row(&host_cos_row, &host_sin_row)?;
    let live_runtime_factors_vs_host_generated_factors =
        compare_vectors(&live_factors, &host_factors)?;

    let host_recompute_from_live_factors = apply_final_token_k_rope_from_factor_rows(
        &k_rope_debug.final_token_k_pre_rope_f32,
        &k_rope_debug.live_runtime_cos_row_f32,
        &k_rope_debug.live_runtime_sin_row_f32,
        k_rope_debug.num_kv_heads,
        k_rope_debug.head_dim,
    )?;
    let host_recompute_from_host_generated_factors = apply_final_token_k_rope_from_factor_rows(
        &k_rope_debug.final_token_k_pre_rope_f32,
        &host_cos_row,
        &host_sin_row,
        k_rope_debug.num_kv_heads,
        k_rope_debug.head_dim,
    )?;
    let host_recompute_from_live_factors_vs_live_runtime_post_rope = compare_vectors(
        &host_recompute_from_live_factors,
        &k_rope_debug.final_token_k_post_rope_f32,
    )?;
    let host_recompute_from_live_factors_vs_official_post_rope =
        compare_vectors(&host_recompute_from_live_factors, official_final_token_k)?;
    let host_recompute_from_host_generated_factors_vs_live_runtime_post_rope = compare_vectors(
        &host_recompute_from_host_generated_factors,
        &k_rope_debug.final_token_k_post_rope_f32,
    )?;
    let host_recompute_from_host_generated_factors_vs_official_post_rope = compare_vectors(
        &host_recompute_from_host_generated_factors,
        official_final_token_k,
    )?;
    let position_metadata_match = k_rope_position_metadata_matches(k_rope_debug);

    let summary = build_k_rope_implementation_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &visible_devices,
        exact_hidden_size,
        &live_trace,
        k_rope_debug,
        position_metadata_match,
        live_runtime_factors_vs_host_generated_factors,
        host_recompute_from_live_factors_vs_live_runtime_post_rope,
        host_recompute_from_live_factors_vs_official_post_rope,
        host_recompute_from_host_generated_factors_vs_live_runtime_post_rope,
        host_recompute_from_host_generated_factors_vs_official_post_rope,
    );
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::KRopeImplementationStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_microproof_status(cli: &Cli) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};

    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let helper_status_artifact =
        load_and_validate_existing_q_gemm_helper_status(cli, &attn_norm_case)?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        q_gemm_helper_status_artifact = %cli.q_gemm_helper_status_artifact.display(),
        "starting exact-case layer0 BF16 Q GEMM microproof status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_q_gemm_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 q gemm microproof trace did not take the bf16 fused-qkv branch");
    }
    if helper_status_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "helper-status artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            helper_status_artifact.hidden_size
        );
    }
    if helper_status_artifact.q_dim != live_trace.q_dim {
        bail!(
            "helper-status artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            helper_status_artifact.q_dim
        );
    }

    let activation_row_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_activation_bf16_bits);
    let q_weight_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_q_weight_bf16_bits);
    let reference_output = build_live_q_gemm_reference(&live_trace)?;

    let stream = worker.gpu_stream().clone();
    let activation_gpu = stream
        .clone_htod(&activation_row_bf16)
        .map_err(|e| anyhow::anyhow!("isolated helper activation htod failed: {e}"))?;
    let q_weight_gpu = stream
        .clone_htod(&q_weight_bf16)
        .map_err(|e| anyhow::anyhow!("isolated helper q weight htod failed: {e}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(live_trace.q_dim)
        .map_err(|e| anyhow::anyhow!("isolated helper output alloc failed: {e}"))?;
    worker.blas().hgemm_bf16(
        1,
        live_trace.q_dim,
        live_trace.hidden_size,
        bf16::ONE,
        &activation_gpu,
        &q_weight_gpu,
        bf16::ZERO,
        &mut output_gpu,
    )?;
    let helper_output: Vec<f32> = stream
        .clone_dtoh(&output_gpu)
        .map_err(|e| anyhow::anyhow!("isolated helper output dtoh failed: {e}"))?
        .iter()
        .map(|value| value.to_f32())
        .collect();

    let helper_vs_live_standalone =
        compare_vectors(&helper_output, &live_trace.standalone_q_gemm_output)?;
    let isolated_reproduces_live_standalone_capture = helper_vs_live_standalone.matched;
    let helper_vs_reference = compare_vectors(&helper_output, &reference_output)?;

    let raw_blas = CudaBlas::new(worker.gpu_stream().clone())
        .map_err(|e| anyhow::anyhow!("failed to create microproof raw cuBLAS handle: {e}"))?;
    let activation_row_major_input_hypothesis = run_raw_bf16_gemm_contract(
        &raw_blas,
        worker.gpu_stream(),
        &activation_row_bf16,
        &q_weight_bf16,
        live_trace.q_dim,
        live_trace.hidden_size,
        RawBf16GemmContract {
            trans_weight: CUBLAS_OP_T,
            trans_input: CUBLAS_OP_T,
            lda: live_trace.hidden_size as i32,
            ldb: 1,
            ldc: live_trace.q_dim as i32,
            alpha_beta_dtype: RawScalarDtype::Bf16,
        },
    )?;
    let q_weight_column_major_hypothesis = run_raw_bf16_gemm_contract(
        &raw_blas,
        worker.gpu_stream(),
        &activation_row_bf16,
        &q_weight_bf16,
        live_trace.q_dim,
        live_trace.hidden_size,
        RawBf16GemmContract {
            trans_weight: CUBLAS_OP_N,
            trans_input: CUBLAS_OP_N,
            lda: live_trace.q_dim as i32,
            ldb: live_trace.hidden_size as i32,
            ldc: live_trace.q_dim as i32,
            alpha_beta_dtype: RawScalarDtype::Bf16,
        },
    )?;

    let summary = build_q_gemm_microproof_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &helper_status_artifact,
        &visible_devices,
        &live_trace,
        helper_vs_reference,
        isolated_reproduces_live_standalone_capture,
        compare_vectors(&activation_row_major_input_hypothesis, &reference_output)?,
        compare_vectors(&q_weight_column_major_hypothesis, &reference_output)?,
    );
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmMicroproofStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_q_gemm_helper_contract_status(cli: &Cli) -> Result<()> {
    use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};

    let (attn_norm_artifact, attn_norm_case) =
        load_single_case_artifact(&cli.local_attn_norm_artifact)?;
    validate_exact_case_artifact(
        &attn_norm_artifact,
        &attn_norm_case,
        "layer0_attn_norm_output",
        "developer-message-user-smoke",
        &attn_norm_case.input_token_ids,
    )?;
    let microproof_artifact =
        load_and_validate_existing_q_gemm_microproof_status(cli, &attn_norm_case)?;

    let model_root = Path::new(&attn_norm_artifact.provenance.model);
    let max_model_len = attn_norm_artifact.provenance.max_model_len.unwrap_or(128);
    let gpu_memory_utilization = attn_norm_artifact
        .provenance
        .gpu_memory_utilization
        .unwrap_or(0.75);
    let visible_devices = attn_norm_artifact
        .provenance
        .visible_devices
        .clone()
        .unwrap_or_else(|| env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()));

    info!(
        model_root = %model_root.display(),
        max_model_len,
        gpu_memory_utilization,
        visible_devices = %visible_devices,
        local_attn_norm_artifact = %cli.local_attn_norm_artifact.display(),
        q_gemm_microproof_status_artifact = %cli.q_gemm_microproof_status_artifact.display(),
        "starting exact-case layer0 BF16 Q GEMM helper-contract status run"
    );

    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, true);
    let worker = build_worker(model_root, max_model_len, gpu_memory_utilization)?;
    let live_trace = capture_layer0_q_gemm_trace(&worker, &attn_norm_case.input_token_ids)?;
    if !live_trace.branch_taken {
        bail!("live layer0 q gemm helper-contract trace did not take the bf16 fused-qkv branch");
    }
    if microproof_artifact.hidden_size != live_trace.hidden_size {
        bail!(
            "microproof-status artifact hidden size mismatch: expected {}, found {}",
            live_trace.hidden_size,
            microproof_artifact.hidden_size
        );
    }
    if microproof_artifact.q_dim != live_trace.q_dim {
        bail!(
            "microproof-status artifact q_dim mismatch: expected {}, found {}",
            live_trace.q_dim,
            microproof_artifact.q_dim
        );
    }

    let activation_row_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_activation_bf16_bits);
    let q_weight_bf16 = bf16_bits_to_bf16_slice(&live_trace.standalone_q_weight_bf16_bits);
    let reference_output = build_live_q_gemm_reference(&live_trace)?;

    let stream = worker.gpu_stream().clone();
    let activation_gpu = stream
        .clone_htod(&activation_row_bf16)
        .map_err(|e| anyhow::anyhow!("helper-contract activation htod failed: {e}"))?;
    let q_weight_gpu = stream
        .clone_htod(&q_weight_bf16)
        .map_err(|e| anyhow::anyhow!("helper-contract q weight htod failed: {e}"))?;
    let mut output_gpu = stream
        .alloc_zeros::<bf16>(live_trace.q_dim)
        .map_err(|e| anyhow::anyhow!("helper-contract output alloc failed: {e}"))?;
    worker.blas().hgemm_bf16(
        1,
        live_trace.q_dim,
        live_trace.hidden_size,
        bf16::ONE,
        &activation_gpu,
        &q_weight_gpu,
        bf16::ZERO,
        &mut output_gpu,
    )?;
    let helper_output: Vec<f32> = stream
        .clone_dtoh(&output_gpu)
        .map_err(|e| anyhow::anyhow!("helper-contract output dtoh failed: {e}"))?
        .iter()
        .map(|value| value.to_f32())
        .collect();

    let helper_vs_live_standalone =
        compare_vectors(&helper_output, &live_trace.standalone_q_gemm_output)?;
    let isolated_reproduces_live_standalone_capture = helper_vs_live_standalone.matched;
    let helper_vs_reference = compare_vectors(&helper_output, &reference_output)?;

    let raw_blas = CudaBlas::new(worker.gpu_stream().clone())
        .map_err(|e| anyhow::anyhow!("failed to create helper-contract raw cuBLAS handle: {e}"))?;
    let helper_contract = RawBf16GemmContract {
        trans_weight: CUBLAS_OP_T,
        trans_input: CUBLAS_OP_N,
        lda: live_trace.hidden_size as i32,
        ldb: live_trace.hidden_size as i32,
        ldc: live_trace.q_dim as i32,
        alpha_beta_dtype: RawScalarDtype::F32,
    };
    let alternate_contract = RawBf16GemmContract {
        alpha_beta_dtype: RawScalarDtype::Bf16,
        ..helper_contract
    };
    let raw_exact_contract_output = run_raw_bf16_gemm_contract(
        &raw_blas,
        worker.gpu_stream(),
        &activation_row_bf16,
        &q_weight_bf16,
        live_trace.q_dim,
        live_trace.hidden_size,
        helper_contract,
    )?;
    let raw_exact_contract_reproduces_helper =
        compare_vectors(&raw_exact_contract_output, &helper_output)?.matched;
    if !raw_exact_contract_reproduces_helper {
        bail!("raw exact helper-contract replay did not reproduce worker.blas().hgemm_bf16 output");
    }
    let alternate_output = run_raw_bf16_gemm_contract(
        &raw_blas,
        worker.gpu_stream(),
        &activation_row_bf16,
        &q_weight_bf16,
        live_trace.q_dim,
        live_trace.hidden_size,
        alternate_contract,
    )?;

    let summary = build_q_gemm_helper_contract_status_summary(
        cli,
        &attn_norm_case,
        &attn_norm_artifact,
        &microproof_artifact,
        &visible_devices,
        &live_trace,
        helper_contract,
        alternate_contract,
        raw_exact_contract_reproduces_helper,
        isolated_reproduces_live_standalone_capture,
        helper_vs_reference,
        compare_vectors(&alternate_output, &reference_output)?,
    );
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| Mode::QGemmHelperContractStatus.default_output());
    write_pretty_json(&output, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    match cli.mode {
        Mode::CandidateStatus => run_candidate_status(&cli),
        Mode::QGemmStatus => run_q_gemm_status(&cli),
        Mode::QGemmHelperStatus => run_q_gemm_helper_status(&cli),
        Mode::QGemmLiveVsIsolatedStatus => run_q_gemm_live_vs_isolated_status(&cli),
        Mode::QGemmPreHelperPlumbingStatus => run_q_gemm_pre_helper_plumbing_status(&cli),
        Mode::QGemmHelperInvocationOutputStatus => run_q_gemm_helper_invocation_output_status(&cli),
        Mode::KGemmHelperPathStatus => run_k_gemm_helper_path_status(&cli),
        Mode::KGemmFusedSplitStatus => run_k_gemm_fused_split_status(&cli),
        Mode::KGemmFusedQkvReadoutStatus => run_k_gemm_fused_qkv_readout_status(&cli),
        Mode::KGemmFusedKReadoutStatus => run_k_gemm_fused_k_readout_status(&cli),
        Mode::KConsumptionRopeStatus => run_k_consumption_rope_status(&cli),
        Mode::KRopeConventionStatus => run_k_rope_convention_status(&cli),
        Mode::KRopeImplementationStatus => run_k_rope_implementation_status(&cli),
        Mode::QGemmMicroproofStatus => run_q_gemm_microproof_status(&cli),
        Mode::QGemmHelperContractStatus => run_q_gemm_helper_contract_status(&cli),
    }
}
