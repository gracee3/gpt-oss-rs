use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::worker::gpu_worker::GpuWorker;
use gpt_oss_engine::{RuntimeMode, SequenceData, SequenceGroupMetadata, WorkerConfig};
use gpt_oss_model_runner::gpu_runner::Layer0QkvTrace;
use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_subscriber::EnvFilter;

const BF16_DENSE_QKV_ENV: &str = "GPT_OSS_LAYER0_BF16_DENSE_QKV";
const DEFAULT_LOCAL_RESIDUAL_INPUT_ARTIFACT: &str =
    ".live/runtime-forward-first-block-20260423/developer-message.runner-layer0-residual-input.json";
const DEFAULT_OFFICIAL_POST_ATTENTION_RESIDUAL_ARTIFACT: &str =
    "/tmp/pinned-prompt-parity-official-reference-20260423/developer-message.official-layer0-post-attention-residual.cpu.json";
const DEFAULT_OUTPUT: &str = ".live/runtime-forward-layer0-qkv-bf16-branch-liveness-20260423/developer-message.runner-layer0-qkv-bf16-branch-liveness-status.json";

#[derive(Debug, Parser, Clone)]
#[command(about = "Exact-case runtime status for the layer-0 BF16 dense-QKV branch liveness check")]
struct Cli {
    #[arg(long, default_value = DEFAULT_LOCAL_RESIDUAL_INPUT_ARTIFACT)]
    local_residual_input_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_OFFICIAL_POST_ATTENTION_RESIDUAL_ARTIFACT)]
    official_post_attention_residual_artifact: PathBuf,

    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Deserialize)]
struct ArtifactProvenance {
    model: String,
    #[serde(default)]
    capture_source: Option<String>,
    #[serde(default)]
    reference_kind: Option<String>,
    #[serde(default)]
    authority_level: Option<String>,
    #[serde(default)]
    visible_devices: Option<String>,
    #[serde(default)]
    max_model_len: Option<usize>,
    #[serde(default)]
    gpu_memory_utilization: Option<f32>,
    #[serde(default)]
    prompt_renderer: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct ArtifactCase {
    id: String,
    input_token_ids: Vec<TokenId>,
    #[serde(default)]
    hidden_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct PinnedArtifact {
    boundary: String,
    #[serde(default)]
    layer_idx: Option<usize>,
    provenance: ArtifactProvenance,
    cases: Vec<ArtifactCase>,
}

#[derive(Debug, Serialize)]
struct CompareMetrics {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    matched: bool,
}

#[derive(Debug, Serialize)]
struct TrialSummary {
    enabled: bool,
    branch_taken: bool,
}

#[derive(Debug, Serialize)]
struct QkvComparison {
    qkv_tensor_equal: bool,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    q_slice: CompareMetrics,
    k_slice: CompareMetrics,
    v_slice: CompareMetrics,
}

#[derive(Debug, Serialize)]
struct StatusSummary {
    schema_version: String,
    provenance: StatusProvenance,
    exact_case: ExactCaseSummary,
    baseline: TrialSummary,
    candidate: TrialSummary,
    qkv_comparison: QkvComparison,
    next_bounded_step: String,
}

#[derive(Debug, Serialize)]
struct StatusProvenance {
    local_residual_input_artifact_path: PathBuf,
    official_post_attention_residual_artifact_path: PathBuf,
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

#[derive(Debug, Clone)]
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

fn capture_layer0_qkv(worker: &GpuWorker, prompt_token_ids: &[TokenId]) -> Result<Layer0QkvTrace> {
    let metadata = build_single_sequence_metadata(prompt_token_ids);
    Ok(worker.debug_runner_prefill_layer0_qkv_trace(&metadata)?)
}

fn run_trial(
    model_path: &Path,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    prompt_token_ids: &[TokenId],
    candidate_enabled: bool,
) -> Result<Layer0QkvTrace> {
    let _guard = EnvFlagGuard::set(BF16_DENSE_QKV_ENV, candidate_enabled);
    let worker = build_worker(model_path, max_model_len, gpu_memory_utilization)?;
    capture_layer0_qkv(&worker, prompt_token_ids)
}

fn next_bounded_step(candidate_branch_taken: bool, qkv_equal: bool) -> String {
    if !candidate_branch_taken {
        "fix layer-0 BF16 dense-QKV wiring and gating only".into()
    } else if qkv_equal {
        "inspect why the BF16 path is numerically identical (output dtype, rounding, or accumulation path)".into()
    } else {
        "move one seam later into attention-side arithmetic after QKV".into()
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    let (local_artifact, local_case) =
        load_single_case_artifact(&cli.local_residual_input_artifact)?;
    let (official_post_attention_artifact, official_post_attention_case) =
        load_single_case_artifact(&cli.official_post_attention_residual_artifact)?;

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
        "starting exact-case layer0 BF16 branch-liveness status run"
    );

    let baseline_trace = run_trial(
        model_root,
        max_model_len,
        gpu_memory_utilization,
        &local_case.input_token_ids,
        false,
    )?;
    let candidate_trace = run_trial(
        model_root,
        max_model_len,
        gpu_memory_utilization,
        &local_case.input_token_ids,
        true,
    )?;

    if baseline_trace.qkv_projection_output.len() != candidate_trace.qkv_projection_output.len() {
        bail!(
            "qkv length mismatch: baseline {} vs candidate {}",
            baseline_trace.qkv_projection_output.len(),
            candidate_trace.qkv_projection_output.len()
        );
    }
    let q_end = baseline_trace.num_tokens * baseline_trace.q_dim;
    let k_end = q_end + baseline_trace.num_tokens * baseline_trace.kv_dim;
    let qkv_compare = compare_vectors(
        &baseline_trace.qkv_projection_output,
        &candidate_trace.qkv_projection_output,
    )?;
    let q_slice = compare_vectors(
        &baseline_trace.qkv_projection_output[..q_end],
        &candidate_trace.qkv_projection_output[..q_end],
    )?;
    let k_slice = compare_vectors(
        &baseline_trace.qkv_projection_output[q_end..k_end],
        &candidate_trace.qkv_projection_output[q_end..k_end],
    )?;
    let v_slice = compare_vectors(
        &baseline_trace.qkv_projection_output[k_end..],
        &candidate_trace.qkv_projection_output[k_end..],
    )?;

    let summary = StatusSummary {
        schema_version: "runtime_forward_layer0_qkv_bf16_branch_liveness_status/v1".into(),
        provenance: StatusProvenance {
            local_residual_input_artifact_path: cli.local_residual_input_artifact.clone(),
            official_post_attention_residual_artifact_path: cli
                .official_post_attention_residual_artifact
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
                .context("exact case missing hidden_size in all artifacts")?,
            boundary: local_artifact.boundary.clone(),
            layer_idx: local_artifact.layer_idx,
        },
        baseline: TrialSummary {
            enabled: false,
            branch_taken: baseline_trace.branch_taken,
        },
        candidate: TrialSummary {
            enabled: true,
            branch_taken: candidate_trace.branch_taken,
        },
        qkv_comparison: QkvComparison {
            qkv_tensor_equal: qkv_compare.matched,
            max_abs_diff: qkv_compare.max_abs_diff,
            mean_abs_diff: qkv_compare.mean_abs_diff,
            q_slice,
            k_slice,
            v_slice,
        },
        next_bounded_step: next_bounded_step(candidate_trace.branch_taken, qkv_compare.matched),
    };

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.output, serde_json::to_vec_pretty(&summary)?)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
