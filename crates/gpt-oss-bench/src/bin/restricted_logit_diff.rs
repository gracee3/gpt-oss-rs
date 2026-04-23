use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::worker::gpu_worker::GpuWorker;
use gpt_oss_engine::{RuntimeMode, SequenceData, SequenceGroupMetadata, WorkerConfig};
use gpt_oss_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser, Clone)]
#[command(about = "Restricted GPT-OSS logit differential diagnosis")]
struct Cli {
    #[arg(long)]
    model: String,

    #[arg(
        long,
        default_value = "Explain tensor parallelism in one short sentence."
    )]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_model_len: usize,

    #[arg(long, default_value_t = 0.75)]
    gpu_memory_utilization: f32,

    #[arg(long, default_value_t = 8)]
    top_k: usize,

    #[arg(long)]
    prefill_last_position_only: bool,

    #[arg(long, value_enum, default_value_t = PrefillCaptureSource::Worker)]
    prefill_capture_source: PrefillCaptureSource,

    #[arg(long)]
    runner_early_stop_layer: Option<usize>,

    #[arg(long)]
    ppp_artifact_json: Option<PathBuf>,

    #[arg(long)]
    ppp_final_norm_intermediate: bool,

    #[arg(long, default_value = ".live/restricted-logit-diff.json")]
    output: PathBuf,

    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, hide = true)]
    child_mode: Option<ChildMode>,

    #[arg(long, value_delimiter = ',', hide = true)]
    forced_output_tokens: Vec<TokenId>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
enum ChildMode {
    Worker,
    Runner,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum PrefillCaptureSource {
    Worker,
    Runner,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepCapture {
    kind: String,
    seq_start_pos: u32,
    input_token_ids: Vec<TokenId>,
    forced_output_token_ids: Vec<TokenId>,
    chosen_token_id: TokenId,
    chosen_token_text: String,
    top_k: Vec<TopLogit>,
    logits: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChildSummary {
    mode: ChildMode,
    model: String,
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    restricted_model_path: String,
    visible_devices: String,
    steps: Vec<StepCapture>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TopLogit {
    token_id: TokenId,
    token_text: String,
    logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepDiff {
    kind: String,
    seq_start_pos: u32,
    matched_within_tolerance: bool,
    max_abs_diff: f32,
    mean_abs_diff: f32,
    chosen_token_match: bool,
    worker_chosen_token_id: TokenId,
    runner_chosen_token_id: TokenId,
    worker_top_k: Vec<TopLogit>,
    runner_top_k: Vec<TopLogit>,
    largest_logit_diffs: Vec<IndexedLogitDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexedLogitDiff {
    token_id: TokenId,
    worker_logit: f32,
    runner_logit: f32,
    abs_diff: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DifferentialReport {
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    restricted_model_path: String,
    visible_devices: String,
    tolerance: f32,
    first_mismatch_boundary: Option<String>,
    conclusion: String,
    worker_steps: Vec<StepCapture>,
    runner_steps: Vec<StepCapture>,
    step_diffs: Vec<StepDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LastPositionCapture {
    artifact_kind: String,
    capture_source: PrefillCaptureSource,
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    input_token_ids: Vec<TokenId>,
    prompt_token_count: usize,
    final_position: usize,
    restricted_model_path: String,
    visible_devices: String,
    argmax: TopLogit,
    argmax_token_id: TokenId,
    top_k: Vec<TopLogit>,
    final_position_top_k: Vec<TopLogit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    official_alignment: Option<OfficialAlignmentSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunnerEarlyStopCapture {
    artifact_kind: String,
    capture_source: PrefillCaptureSource,
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    input_token_ids: Vec<TokenId>,
    prompt_token_count: usize,
    final_position: usize,
    restricted_model_path: String,
    visible_devices: String,
    seam_identifier: String,
    layer_idx: usize,
    hidden_state_dim: usize,
    last_token_hidden_state: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct OfficialArtifactRef {
    prompt: Option<String>,
    input_token_ids: Vec<TokenId>,
    argmax_token_id: TokenId,
    top_k_token_ids: Option<Vec<TokenId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct OfficialAlignmentSummary {
    official_artifact_path: String,
    token_id_agreement: bool,
    official_argmax_token_id: TokenId,
    local_argmax_token_id: TokenId,
    argmax_agreement: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    official_top_k_token_ids: Option<Vec<TokenId>>,
    local_top_k_token_ids: Vec<TokenId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k_identity: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k_overlap_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k_overlap_token_ids: Option<Vec<TokenId>>,
}

#[derive(Debug, Clone)]
struct PromptCaptureInput {
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    official_artifact: Option<(PathBuf, OfficialArtifactRef)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PppTokenArtifactInput {
    suite_id: String,
    case_id: String,
    input_token_ids: Vec<TokenId>,
    prompt_renderer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PppFinalNormIntermediateArtifact {
    schema_version: String,
    suite_id: String,
    boundary: String,
    provenance: PppFinalNormIntermediateProvenance,
    cases: Vec<PppFinalNormIntermediateCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PppFinalNormIntermediateProvenance {
    model: String,
    capture_source: String,
    reference_kind: String,
    authority_level: String,
    visible_devices: String,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_renderer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PppFinalNormIntermediateCase {
    id: String,
    input_token_ids: Vec<TokenId>,
    hidden_size: usize,
    final_token_hidden_f32: Vec<f32>,
}

fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    if let Some(mode) = cli.child_mode {
        return run_child(&cli, mode);
    }

    validate_prefill_capture_args(&cli)?;

    if cli.prefill_last_position_only {
        return run_prefill_last_position_capture(&cli);
    }

    let worker = spawn_child(&cli, ChildMode::Worker, &[])?;
    let forced_output_tokens = worker
        .steps
        .iter()
        .take(2)
        .map(|step| step.chosen_token_id)
        .collect::<Vec<_>>();
    let runner = spawn_child(&cli, ChildMode::Runner, &forced_output_tokens)?;
    let report = build_report(&worker, &runner, cli.top_k)?;

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_prefill_last_position_capture(cli: &Cli) -> Result<()> {
    if cli.ppp_final_norm_intermediate {
        return run_ppp_final_norm_intermediate_capture(cli);
    }

    let tokenizer = Tokenizer::from_pretrained(&cli.model)?;
    let capture_input = resolve_prompt_capture_input(cli, &tokenizer)?;
    let prompt_token_ids = capture_input.prompt_token_ids.clone();
    if prompt_token_ids.is_empty() {
        bail!("prefill-last-position-only requires at least one prompt token");
    }

    let model_path = Path::new(&cli.model);
    let mut worker = build_worker(model_path, cli.max_model_len, cli.gpu_memory_utilization)?;
    let metadata = build_single_sequence_metadata(&prompt_token_ids, &[], true);
    if let Some(stop_after_layer) = cli.runner_early_stop_layer {
        let hidden_capture =
            worker.debug_runner_prefill_last_token_layer_output(&metadata, stop_after_layer)?;
        let capture = RunnerEarlyStopCapture {
            artifact_kind: "runner_prefill_last_token_hidden_state".to_string(),
            capture_source: PrefillCaptureSource::Runner,
            prompt: capture_input.prompt,
            prompt_token_ids: prompt_token_ids.clone(),
            input_token_ids: prompt_token_ids.clone(),
            prompt_token_count: prompt_token_ids.len(),
            final_position: prompt_token_ids.len() - 1,
            restricted_model_path: cli.model.clone(),
            visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
                .unwrap_or_else(|_| "<unset>".into()),
            seam_identifier: hidden_capture.seam_identifier,
            layer_idx: hidden_capture.layer_idx,
            hidden_state_dim: hidden_capture.hidden_state_dim,
            last_token_hidden_state: hidden_capture.hidden_state,
        };

        if let Some(parent) = cli.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&cli.output, serde_json::to_vec_pretty(&capture)?)?;
        println!("{}", serde_json::to_string_pretty(&capture)?);
        return Ok(());
    }

    let last_logits = match cli.prefill_capture_source {
        PrefillCaptureSource::Worker => worker.debug_last_token_logits(&metadata)?,
        PrefillCaptureSource::Runner => worker.debug_runner_last_token_logits(&metadata)?,
    };
    let chosen_token_id = argmax_token(&last_logits)?;
    let top_k = top_k_logits(&last_logits, cli.top_k, &tokenizer)?;
    let local_top_k_token_ids = top_k.iter().map(|entry| entry.token_id).collect::<Vec<_>>();
    let capture = LastPositionCapture {
        artifact_kind: "prefill_last_position_logits_topk".to_string(),
        capture_source: cli.prefill_capture_source,
        prompt: capture_input.prompt,
        prompt_token_ids: prompt_token_ids.clone(),
        input_token_ids: prompt_token_ids.clone(),
        prompt_token_count: prompt_token_ids.len(),
        final_position: prompt_token_ids.len() - 1,
        restricted_model_path: cli.model.clone(),
        visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()),
        argmax: TopLogit {
            token_id: chosen_token_id,
            token_text: tokenizer.decode(&[chosen_token_id]).unwrap_or_default(),
            logit: last_logits[chosen_token_id as usize],
        },
        argmax_token_id: chosen_token_id,
        top_k: top_k.clone(),
        final_position_top_k: top_k,
        official_alignment: capture_input
            .official_artifact
            .as_ref()
            .map(|(path, official)| {
                build_official_alignment_summary(
                    path,
                    official,
                    &prompt_token_ids,
                    chosen_token_id,
                    &local_top_k_token_ids,
                )
            }),
    };

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.output, serde_json::to_vec_pretty(&capture)?)?;
    println!("{}", serde_json::to_string_pretty(&capture)?);
    Ok(())
}

fn run_ppp_final_norm_intermediate_capture(cli: &Cli) -> Result<()> {
    let artifact_path = cli
        .ppp_artifact_json
        .as_ref()
        .context("--ppp-final-norm-intermediate requires --ppp-artifact-json")?;
    let ppp_input = load_ppp_token_artifact(artifact_path)?;
    if ppp_input.input_token_ids.is_empty() {
        bail!("prefill-last-position-only requires at least one prompt token");
    }

    let model_path = Path::new(&cli.model);
    let worker = build_worker(model_path, cli.max_model_len, cli.gpu_memory_utilization)?;
    let metadata = build_single_sequence_metadata(&ppp_input.input_token_ids, &[], true);
    let capture = worker.debug_runner_final_token_post_final_norm_pre_unembedding(&metadata)?;
    let artifact = PppFinalNormIntermediateArtifact {
        schema_version: "pinned-prompt-intermediate-artifact/v1".to_string(),
        suite_id: ppp_input.suite_id,
        boundary: "final_token_post_final_norm_pre_unembedding".to_string(),
        provenance: PppFinalNormIntermediateProvenance {
            model: cli.model.clone(),
            capture_source: "runner".to_string(),
            reference_kind: "local_candidate".to_string(),
            authority_level: "scaffold".to_string(),
            visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
                .unwrap_or_else(|_| "<unset>".into()),
            max_model_len: cli.max_model_len,
            gpu_memory_utilization: cli.gpu_memory_utilization,
            prompt_renderer: ppp_input.prompt_renderer,
        },
        cases: vec![PppFinalNormIntermediateCase {
            id: ppp_input.case_id,
            input_token_ids: ppp_input.input_token_ids,
            hidden_size: capture.hidden_size,
            final_token_hidden_f32: capture.final_token_hidden_f32,
        }],
    };

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.output, serde_json::to_vec_pretty(&artifact)?)?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn validate_prefill_capture_args(cli: &Cli) -> Result<()> {
    if cli.runner_early_stop_layer.is_some()
        && cli.prefill_capture_source != PrefillCaptureSource::Runner
    {
        bail!("--runner-early-stop-layer requires --prefill-capture-source runner");
    }
    if !cli.ppp_final_norm_intermediate {
        return Ok(());
    }
    if !cli.prefill_last_position_only {
        bail!("--ppp-final-norm-intermediate requires --prefill-last-position-only");
    }
    if cli.prefill_capture_source != PrefillCaptureSource::Runner {
        bail!("--ppp-final-norm-intermediate requires --prefill-capture-source runner");
    }
    if cli.ppp_artifact_json.is_none() {
        bail!("--ppp-final-norm-intermediate requires --ppp-artifact-json");
    }
    if cli.runner_early_stop_layer.is_some() {
        bail!("--ppp-final-norm-intermediate cannot be combined with --runner-early-stop-layer");
    }
    Ok(())
}

fn resolve_prompt_capture_input(cli: &Cli, tokenizer: &Tokenizer) -> Result<PromptCaptureInput> {
    if let Some(path) = &cli.ppp_artifact_json {
        let artifact = load_official_artifact(path)?;
        return Ok(PromptCaptureInput {
            prompt: artifact
                .prompt
                .clone()
                .unwrap_or_else(|| "<ppp-rendered-token-ids>".to_string()),
            prompt_token_ids: artifact.input_token_ids.clone(),
            official_artifact: Some((path.clone(), artifact)),
        });
    }

    Ok(PromptCaptureInput {
        prompt: cli.prompt.clone(),
        prompt_token_ids: tokenizer.encode(&cli.prompt)?,
        official_artifact: None,
    })
}

fn run_child(cli: &Cli, mode: ChildMode) -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained(&cli.model)?;
    let prompt_token_ids = tokenizer.encode(&cli.prompt)?;
    let summary = capture_summary(cli, mode, &tokenizer, &prompt_token_ids)?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn spawn_child(
    cli: &Cli,
    mode: ChildMode,
    forced_output_tokens: &[TokenId],
) -> Result<ChildSummary> {
    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    let output = Command::new(current_exe)
        .env(
            "CUDA_VISIBLE_DEVICES",
            std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "1".into()),
        )
        .env(
            "GPT_OSS_DISABLE_CUDA_GRAPHS",
            std::env::var("GPT_OSS_DISABLE_CUDA_GRAPHS").unwrap_or_else(|_| "1".into()),
        )
        .arg("--child-mode")
        .arg(match mode {
            ChildMode::Worker => "worker",
            ChildMode::Runner => "runner",
        })
        .arg("--model")
        .arg(&cli.model)
        .arg("--prompt")
        .arg(&cli.prompt)
        .arg("--max-model-len")
        .arg(cli.max_model_len.to_string())
        .arg("--gpu-memory-utilization")
        .arg(cli.gpu_memory_utilization.to_string())
        .arg("--top-k")
        .arg(cli.top_k.to_string())
        .args(
            forced_output_tokens
                .iter()
                .flat_map(|token| ["--forced-output-tokens".to_string(), token.to_string()]),
        )
        .arg("--log-level")
        .arg(&cli.log_level)
        .output()
        .with_context(|| format!("failed to spawn {mode:?} child"))?;

    if !output.status.success() {
        bail!(
            "{mode:?} child failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let stdout = String::from_utf8(output.stdout).context("child stdout was not utf-8")?;
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .context("child did not emit JSON summary")?;
    Ok(serde_json::from_str(json).context("failed to parse child summary")?)
}

fn capture_summary(
    cli: &Cli,
    mode: ChildMode,
    tokenizer: &Tokenizer,
    prompt_token_ids: &[TokenId],
) -> Result<ChildSummary> {
    let model_path = Path::new(&cli.model);
    let mut worker = build_worker(model_path, cli.max_model_len, cli.gpu_memory_utilization)?;
    let visible_devices =
        std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into());
    let mut forced_output_token_ids = Vec::new();
    let mut steps = Vec::new();

    for (kind, seq_start_pos) in [("prefill", 0u32), ("decode1", 2u32), ("decode2", 3u32)] {
        let metadata = build_single_sequence_metadata(
            prompt_token_ids,
            &forced_output_token_ids,
            kind == "prefill",
        );
        let logits = match mode {
            ChildMode::Worker => worker.debug_logits(&metadata)?,
            ChildMode::Runner => worker.debug_runner_logits(&metadata)?,
        };
        let last_logits = last_token_logits(&logits, tokenizer.vocab_size())?;
        let chosen_token_id = argmax_token(last_logits)?;
        let chosen_token_text = tokenizer.decode(&[chosen_token_id]).unwrap_or_default();
        let top_k = top_k_logits(last_logits, cli.top_k, tokenizer)?;

        steps.push(StepCapture {
            kind: kind.to_string(),
            seq_start_pos,
            input_token_ids: if kind == "prefill" {
                prompt_token_ids.to_vec()
            } else {
                vec![*forced_output_token_ids
                    .last()
                    .expect("decode step must have a forced token")]
            },
            forced_output_token_ids: forced_output_token_ids.clone(),
            chosen_token_id,
            chosen_token_text,
            top_k,
            logits: last_logits.to_vec(),
        });

        if kind != "decode2" {
            let next_forced = cli
                .forced_output_tokens
                .get(forced_output_token_ids.len())
                .copied()
                .unwrap_or(chosen_token_id);
            forced_output_token_ids.push(next_forced);
        }
    }

    Ok(ChildSummary {
        mode,
        model: cli.model.clone(),
        prompt: cli.prompt.clone(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        restricted_model_path: cli.model.clone(),
        visible_devices,
        steps,
    })
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

fn build_single_sequence_metadata(
    prompt_token_ids: &[TokenId],
    output_token_ids: &[TokenId],
    is_prompt: bool,
) -> Vec<SequenceGroupMetadata> {
    let seq_id = SequenceId(1);
    let total_len = prompt_token_ids.len() + output_token_ids.len();
    let num_blocks = total_len.max(1).div_ceil(16);
    let mut seq_data = HashMap::new();
    seq_data.insert(
        seq_id,
        SequenceData {
            prompt_token_ids: prompt_token_ids.to_vec(),
            output_token_ids: output_token_ids.to_vec(),
            cumulative_logprob: 0.0,
        },
    );
    let mut block_tables = HashMap::new();
    block_tables.insert(seq_id, (0..num_blocks).map(|i| BlockId(i as u32)).collect());

    vec![SequenceGroupMetadata {
        request_id: RequestId(1),
        is_prompt,
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

fn last_token_logits<'a>(logits: &'a [f32], vocab_size: usize) -> Result<&'a [f32]> {
    if logits.len() < vocab_size {
        bail!(
            "logits buffer too small: {} < vocab {}",
            logits.len(),
            vocab_size
        );
    }
    let num_tokens = logits.len() / vocab_size;
    let last_offset = (num_tokens - 1) * vocab_size;
    Ok(&logits[last_offset..last_offset + vocab_size])
}

fn argmax_token(logits: &[f32]) -> Result<TokenId> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as TokenId)
        .context("empty logits buffer")
}

fn top_k_logits(logits: &[f32], top_k: usize, tokenizer: &Tokenizer) -> Result<Vec<TopLogit>> {
    let mut indexed = logits
        .iter()
        .enumerate()
        .map(|(idx, logit)| (idx as TokenId, *logit))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed
        .into_iter()
        .take(top_k)
        .map(|(token_id, logit)| TopLogit {
            token_id,
            token_text: tokenizer.decode(&[token_id]).unwrap_or_default(),
            logit,
        })
        .collect())
}

fn load_official_artifact(path: &Path) -> Result<OfficialArtifactRef> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read PPP artifact {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse PPP artifact {}", path.display()))?;
    parse_official_artifact_value(&value).with_context(|| {
        format!(
            "failed to extract input_token_ids/argmax_token_id from PPP artifact {}",
            path.display()
        )
    })
}

fn load_ppp_token_artifact(path: &Path) -> Result<PppTokenArtifactInput> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read PPP token artifact {}", path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse PPP token artifact {}", path.display()))?;
    parse_ppp_token_artifact_value(&value).with_context(|| {
        format!(
            "failed to extract exact PPP token input from {}",
            path.display()
        )
    })
}

fn parse_ppp_token_artifact_value(value: &Value) -> Result<PppTokenArtifactInput> {
    let schema_version = value
        .get("schema_version")
        .and_then(Value::as_str)
        .context("missing schema_version")?;
    if !matches!(
        schema_version,
        "pinned-prompt-artifact/v1" | "pinned-prompt-artifact/v2"
    ) {
        bail!(
            "unsupported schema_version `{schema_version}` for exact PPP mode; expected pinned-prompt-artifact/v1 or v2"
        );
    }

    let suite_id = value
        .get("suite_id")
        .and_then(Value::as_str)
        .context("missing suite_id")?
        .to_string();
    let case_value = extract_single_case_value(value)?
        .context("exact PPP mode requires a single-case `cases` container")?;
    let case_id = case_value
        .get("id")
        .and_then(Value::as_str)
        .context("missing cases[0].id")?
        .to_string();
    let input_token_ids = extract_token_id_list(case_value, &["input_token_ids"])?
        .context("missing cases[0].input_token_ids")?;
    let prompt_renderer = extract_optional_string(
        value
            .get("provenance")
            .and_then(|provenance| provenance.get("prompt_renderer")),
        "provenance.prompt_renderer",
    )?;

    Ok(PppTokenArtifactInput {
        suite_id,
        case_id,
        input_token_ids,
        prompt_renderer,
    })
}

fn parse_official_artifact_value(value: &Value) -> Result<OfficialArtifactRef> {
    if let Some(case_value) = extract_single_case_value(value)? {
        let mut artifact = parse_flat_official_artifact_value(case_value)?;
        if artifact.prompt.is_none() {
            artifact.prompt = value
                .get("prompt")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
        return Ok(artifact);
    }

    parse_flat_official_artifact_value(value)
}

fn extract_single_case_value<'a>(value: &'a Value) -> Result<Option<&'a Value>> {
    let Some(cases) = value.get("cases") else {
        return Ok(None);
    };
    let entries = cases
        .as_array()
        .context("field `cases` must be an array when present")?;
    match entries.len() {
        0 => bail!("field `cases` must contain exactly one case"),
        1 => Ok(entries.first()),
        len => bail!("field `cases` must contain exactly one case, found {len}"),
    }
}

fn parse_flat_official_artifact_value(value: &Value) -> Result<OfficialArtifactRef> {
    let prompt = value
        .get("prompt")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let input_token_ids = extract_token_id_list(value, &["input_token_ids", "prompt_token_ids"])?
        .context("missing input_token_ids")?;
    let argmax_token_id = extract_token_id(value, &["argmax_token_id"])?
        .or_else(|| extract_nested_token_id(value, "argmax"))
        .context("missing argmax_token_id")?;
    let top_k_token_ids = extract_top_k_token_ids(value)?;

    Ok(OfficialArtifactRef {
        prompt,
        input_token_ids,
        argmax_token_id,
        top_k_token_ids,
    })
}

fn extract_token_id(value: &Value, keys: &[&str]) -> Result<Option<TokenId>> {
    for key in keys {
        if let Some(raw) = value.get(key) {
            return Ok(Some(value_to_token_id(raw).with_context(|| {
                format!("field `{key}` must be an integer token id")
            })?));
        }
    }
    Ok(None)
}

fn extract_nested_token_id(value: &Value, key: &str) -> Option<TokenId> {
    value
        .get(key)
        .and_then(|nested| nested.get("token_id"))
        .and_then(Value::as_u64)
        .map(|token_id| token_id as TokenId)
}

fn extract_token_id_list(value: &Value, keys: &[&str]) -> Result<Option<Vec<TokenId>>> {
    for key in keys {
        if let Some(raw) = value.get(key) {
            return Ok(Some(value_to_token_id_list(raw).with_context(|| {
                format!("field `{key}` must be an array of integer token ids")
            })?));
        }
    }
    Ok(None)
}

fn extract_top_k_token_ids(value: &Value) -> Result<Option<Vec<TokenId>>> {
    let raw_top_k = value
        .get("final_position_top_k")
        .or_else(|| value.get("top_k"));
    let Some(raw_top_k) = raw_top_k else {
        return Ok(None);
    };
    let entries = raw_top_k
        .as_array()
        .context("field `top_k`/`final_position_top_k` must be an array when present")?;
    let mut token_ids = Vec::with_capacity(entries.len());
    for (idx, entry) in entries.iter().enumerate() {
        let token_id = if let Some(raw) = entry.get("token_id") {
            value_to_token_id(raw)?
        } else {
            value_to_token_id(entry).with_context(|| {
                format!("top_k[{idx}] must be an integer token id or object with token_id")
            })?
        };
        token_ids.push(token_id);
    }
    Ok(Some(token_ids))
}

fn value_to_token_id(value: &Value) -> Result<TokenId> {
    value
        .as_u64()
        .map(|token_id| token_id as TokenId)
        .context("token id must be a non-negative integer")
}

fn value_to_token_id_list(value: &Value) -> Result<Vec<TokenId>> {
    value
        .as_array()
        .context("token id list must be an array")?
        .iter()
        .map(value_to_token_id)
        .collect()
}

fn extract_optional_string(value: Option<&Value>, field_name: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(raw)) => Ok(Some(raw.clone())),
        Some(_) => bail!("field `{field_name}` must be a string when present"),
    }
}

fn build_official_alignment_summary(
    artifact_path: &Path,
    official: &OfficialArtifactRef,
    local_input_token_ids: &[TokenId],
    local_argmax_token_id: TokenId,
    local_top_k_token_ids: &[TokenId],
) -> OfficialAlignmentSummary {
    let top_k_identity = official
        .top_k_token_ids
        .as_ref()
        .map(|official_top_k| official_top_k == local_top_k_token_ids);
    let (top_k_overlap_count, top_k_overlap_token_ids) =
        if let Some(official_top_k) = official.top_k_token_ids.as_ref() {
            let official_set = official_top_k.iter().copied().collect::<HashSet<_>>();
            let overlap = local_top_k_token_ids
                .iter()
                .copied()
                .filter(|token_id| official_set.contains(token_id))
                .collect::<Vec<_>>();
            (Some(overlap.len()), Some(overlap))
        } else {
            (None, None)
        };

    OfficialAlignmentSummary {
        official_artifact_path: artifact_path.display().to_string(),
        token_id_agreement: official.input_token_ids == local_input_token_ids,
        official_argmax_token_id: official.argmax_token_id,
        local_argmax_token_id,
        argmax_agreement: official.argmax_token_id == local_argmax_token_id,
        official_top_k_token_ids: official.top_k_token_ids.clone(),
        local_top_k_token_ids: local_top_k_token_ids.to_vec(),
        top_k_identity,
        top_k_overlap_count,
        top_k_overlap_token_ids,
    }
}

fn build_report(
    worker: &ChildSummary,
    runner: &ChildSummary,
    top_k: usize,
) -> Result<DifferentialReport> {
    if worker.prompt_token_ids != runner.prompt_token_ids {
        bail!("prompt token ids differ between worker and runner captures");
    }
    if worker.steps.len() != runner.steps.len() {
        bail!("step count differs between worker and runner captures");
    }

    let tolerance = 1e-5f32;
    let mut step_diffs = Vec::with_capacity(worker.steps.len());
    let mut first_mismatch_boundary = None;

    for (worker_step, runner_step) in worker.steps.iter().zip(&runner.steps) {
        let mut largest = worker_step
            .logits
            .iter()
            .zip(&runner_step.logits)
            .enumerate()
            .map(|(idx, (worker_logit, runner_logit))| IndexedLogitDiff {
                token_id: idx as TokenId,
                worker_logit: *worker_logit,
                runner_logit: *runner_logit,
                abs_diff: (*worker_logit - *runner_logit).abs(),
            })
            .collect::<Vec<_>>();
        let max_abs_diff = largest
            .iter()
            .map(|entry| entry.abs_diff)
            .fold(0.0f32, f32::max);
        let mean_abs_diff = if largest.is_empty() {
            0.0
        } else {
            largest.iter().map(|entry| entry.abs_diff).sum::<f32>() / largest.len() as f32
        };
        largest.sort_by(|a, b| {
            b.abs_diff
                .partial_cmp(&a.abs_diff)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        largest.truncate(top_k.min(8));

        let matched = max_abs_diff <= tolerance;
        if !matched && first_mismatch_boundary.is_none() {
            first_mismatch_boundary = Some(worker_step.kind.clone());
        }

        step_diffs.push(StepDiff {
            kind: worker_step.kind.clone(),
            seq_start_pos: worker_step.seq_start_pos,
            matched_within_tolerance: matched,
            max_abs_diff,
            mean_abs_diff,
            chosen_token_match: worker_step.chosen_token_id == runner_step.chosen_token_id,
            worker_chosen_token_id: worker_step.chosen_token_id,
            runner_chosen_token_id: runner_step.chosen_token_id,
            worker_top_k: worker_step.top_k.clone(),
            runner_top_k: runner_step.top_k.clone(),
            largest_logit_diffs: largest,
        });
    }

    let conclusion = if first_mismatch_boundary.is_some() {
        "A: evidence favors implementation-side divergence inside the live worker path".to_string()
    } else {
        "B: evidence favors a semantically invalid restricted reinterpretation or a bug shared by both live and direct CUDA paths, not a worker-path-only divergence".to_string()
    };

    Ok(DifferentialReport {
        prompt: worker.prompt.clone(),
        prompt_token_ids: worker.prompt_token_ids.clone(),
        restricted_model_path: worker.restricted_model_path.clone(),
        visible_devices: worker.visible_devices.clone(),
        tolerance,
        first_mismatch_boundary,
        conclusion,
        worker_steps: worker.steps.clone(),
        runner_steps: runner.steps.clone(),
        step_diffs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cli() -> Cli {
        Cli {
            model: "/tmp/model".to_string(),
            prompt: "hello".to_string(),
            max_model_len: 128,
            gpu_memory_utilization: 0.75,
            top_k: 8,
            prefill_last_position_only: false,
            prefill_capture_source: PrefillCaptureSource::Worker,
            runner_early_stop_layer: None,
            ppp_artifact_json: None,
            ppp_final_norm_intermediate: false,
            output: PathBuf::from(".live/test.json"),
            log_level: "info".to_string(),
            child_mode: None,
            forced_output_tokens: Vec::new(),
        }
    }

    #[test]
    fn parse_official_artifact_minimal_ppp_shape() {
        let value = serde_json::json!({
            "prompt": "shared rendered prompt",
            "input_token_ids": [11, 22, 33],
            "argmax_token_id": 200005,
        });

        let artifact = parse_official_artifact_value(&value).expect("parse PPP artifact");

        assert_eq!(
            artifact,
            OfficialArtifactRef {
                prompt: Some("shared rendered prompt".to_string()),
                input_token_ids: vec![11, 22, 33],
                argmax_token_id: 200005,
                top_k_token_ids: None,
            }
        );
    }

    #[test]
    fn parse_official_artifact_accepts_existing_local_aliases() {
        let value = serde_json::json!({
            "prompt_token_ids": [7, 8, 9],
            "argmax": { "token_id": 78316 },
            "top_k": [
                { "token_id": 78316, "logit": 10.0 },
                { "token_id": 200005, "logit": 9.5 }
            ]
        });

        let artifact =
            parse_official_artifact_value(&value).expect("parse local-compatible artifact");

        assert_eq!(artifact.input_token_ids, vec![7, 8, 9]);
        assert_eq!(artifact.argmax_token_id, 78316);
        assert_eq!(artifact.top_k_token_ids, Some(vec![78316, 200005]));
    }

    #[test]
    fn parse_official_artifact_accepts_single_case_ppp_container() {
        let value = serde_json::json!({
            "schema_version": "pinned-prompt-artifact/v2",
            "suite_id": "developer-message",
            "cases": [
                {
                    "id": "developer-message-user-smoke",
                    "input_token_ids": [11, 22, 33],
                    "argmax_token_id": 200005,
                    "final_position_top_k": [
                        { "token_id": 200005, "logit": 42.5 },
                        { "token_id": 200003, "logit": 17.0 }
                    ]
                }
            ]
        });

        let artifact =
            parse_official_artifact_value(&value).expect("parse authoritative PPP container");

        assert_eq!(
            artifact,
            OfficialArtifactRef {
                prompt: None,
                input_token_ids: vec![11, 22, 33],
                argmax_token_id: 200005,
                top_k_token_ids: Some(vec![200005, 200003]),
            }
        );
    }

    #[test]
    fn parse_ppp_token_artifact_extracts_suite_case_and_tokens() {
        let value = serde_json::json!({
            "schema_version": "pinned-prompt-artifact/v2",
            "suite_id": "developer-message",
            "provenance": {
                "prompt_renderer": "developer-message"
            },
            "cases": [
                {
                    "id": "developer-message-user-smoke",
                    "input_token_ids": [11, 22, 33]
                }
            ]
        });

        let artifact =
            parse_ppp_token_artifact_value(&value).expect("parse exact PPP token artifact");

        assert_eq!(
            artifact,
            PppTokenArtifactInput {
                suite_id: "developer-message".to_string(),
                case_id: "developer-message-user-smoke".to_string(),
                input_token_ids: vec![11, 22, 33],
                prompt_renderer: Some("developer-message".to_string()),
            }
        );
    }

    #[test]
    fn alignment_summary_reports_identity_and_overlap() {
        let official = OfficialArtifactRef {
            prompt: None,
            input_token_ids: vec![1, 2, 3],
            argmax_token_id: 200005,
            top_k_token_ids: Some(vec![200005, 10, 20, 30]),
        };
        let artifact_path = Path::new("/tmp/ppp.json");

        let mismatch = build_official_alignment_summary(
            artifact_path,
            &official,
            &[1, 2, 3],
            78316,
            &[78316, 10, 20, 40],
        );
        assert!(mismatch.token_id_agreement);
        assert!(!mismatch.argmax_agreement);
        assert_eq!(mismatch.top_k_identity, Some(false));
        assert_eq!(mismatch.top_k_overlap_count, Some(2));
        assert_eq!(mismatch.top_k_overlap_token_ids, Some(vec![10, 20]));

        let identical = build_official_alignment_summary(
            artifact_path,
            &official,
            &[1, 2, 3],
            200005,
            &[200005, 10, 20, 30],
        );
        assert!(identical.argmax_agreement);
        assert_eq!(identical.top_k_identity, Some(true));
        assert_eq!(identical.top_k_overlap_count, Some(4));
        assert_eq!(
            identical.top_k_overlap_token_ids,
            Some(vec![200005, 10, 20, 30])
        );
    }

    #[test]
    fn last_position_capture_serializes_ppp_aligned_aliases() {
        let capture = LastPositionCapture {
            artifact_kind: "prefill_last_position_logits_topk".to_string(),
            capture_source: PrefillCaptureSource::Runner,
            prompt: "<ppp-rendered-token-ids>".to_string(),
            prompt_token_ids: vec![1, 2],
            input_token_ids: vec![1, 2],
            prompt_token_count: 2,
            final_position: 1,
            restricted_model_path: "/tmp/model".to_string(),
            visible_devices: "1".to_string(),
            argmax: TopLogit {
                token_id: 200005,
                token_text: String::new(),
                logit: 1.0,
            },
            argmax_token_id: 200005,
            top_k: vec![TopLogit {
                token_id: 200005,
                token_text: String::new(),
                logit: 1.0,
            }],
            final_position_top_k: vec![TopLogit {
                token_id: 200005,
                token_text: String::new(),
                logit: 1.0,
            }],
            official_alignment: Some(OfficialAlignmentSummary {
                official_artifact_path: "/tmp/ppp.json".to_string(),
                token_id_agreement: true,
                official_argmax_token_id: 200005,
                local_argmax_token_id: 200005,
                argmax_agreement: true,
                official_top_k_token_ids: Some(vec![200005]),
                local_top_k_token_ids: vec![200005],
                top_k_identity: Some(true),
                top_k_overlap_count: Some(1),
                top_k_overlap_token_ids: Some(vec![200005]),
            }),
        };

        let json = serde_json::to_value(&capture).expect("serialize capture");
        assert_eq!(json["capture_source"], serde_json::json!("runner"));
        assert_eq!(json["input_token_ids"], serde_json::json!([1, 2]));
        assert_eq!(json["argmax_token_id"], serde_json::json!(200005));
        assert_eq!(
            json["final_position_top_k"][0]["token_id"],
            serde_json::json!(200005)
        );
        assert_eq!(
            json["official_alignment"]["token_id_agreement"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn ppp_final_norm_intermediate_serializes_exact_artifact_shape() {
        let artifact = PppFinalNormIntermediateArtifact {
            schema_version: "pinned-prompt-intermediate-artifact/v1".to_string(),
            suite_id: "developer-message".to_string(),
            boundary: "final_token_post_final_norm_pre_unembedding".to_string(),
            provenance: PppFinalNormIntermediateProvenance {
                model: "/tmp/model".to_string(),
                capture_source: "runner".to_string(),
                reference_kind: "local_candidate".to_string(),
                authority_level: "scaffold".to_string(),
                visible_devices: "1".to_string(),
                max_model_len: 128,
                gpu_memory_utilization: 0.75,
                prompt_renderer: Some("developer-message".to_string()),
            },
            cases: vec![PppFinalNormIntermediateCase {
                id: "developer-message-user-smoke".to_string(),
                input_token_ids: vec![1, 2],
                hidden_size: 3,
                final_token_hidden_f32: vec![0.25, -0.5, 1.0],
            }],
        };

        let json = serde_json::to_value(&artifact).expect("serialize intermediate artifact");
        assert_eq!(
            json["schema_version"],
            serde_json::json!("pinned-prompt-intermediate-artifact/v1")
        );
        assert_eq!(
            json["boundary"],
            serde_json::json!("final_token_post_final_norm_pre_unembedding")
        );
        assert_eq!(
            json["provenance"]["prompt_renderer"],
            serde_json::json!("developer-message")
        );
        assert_eq!(
            json["cases"][0]["final_token_hidden_f32"],
            serde_json::json!([0.25, -0.5, 1.0])
        );
    }

    #[test]
    fn runner_early_stop_capture_serializes_hidden_state_artifact() {
        let capture = RunnerEarlyStopCapture {
            artifact_kind: "runner_prefill_last_token_hidden_state".to_string(),
            capture_source: PrefillCaptureSource::Runner,
            prompt: "<ppp-rendered-token-ids>".to_string(),
            prompt_token_ids: vec![1, 2],
            input_token_ids: vec![1, 2],
            prompt_token_count: 2,
            final_position: 1,
            restricted_model_path: "/tmp/model".to_string(),
            visible_devices: "1".to_string(),
            seam_identifier: "transformer_layer_output".to_string(),
            layer_idx: 0,
            hidden_state_dim: 3,
            last_token_hidden_state: vec![0.25, -0.5, 1.0],
        };

        let json = serde_json::to_value(&capture).expect("serialize early-stop capture");
        assert_eq!(
            json["artifact_kind"],
            serde_json::json!("runner_prefill_last_token_hidden_state")
        );
        assert_eq!(json["capture_source"], serde_json::json!("runner"));
        assert_eq!(json["layer_idx"], serde_json::json!(0));
        assert_eq!(json["hidden_state_dim"], serde_json::json!(3));
        assert_eq!(
            json["last_token_hidden_state"],
            serde_json::json!([0.25, -0.5, 1.0])
        );
    }

    #[test]
    fn runner_early_stop_requires_runner_capture_source() {
        let mut cli = sample_cli();
        cli.runner_early_stop_layer = Some(0);
        let err = validate_prefill_capture_args(&cli)
            .expect_err("worker capture should reject runner early-stop");
        assert!(err
            .to_string()
            .contains("--runner-early-stop-layer requires --prefill-capture-source runner"));

        cli.prefill_capture_source = PrefillCaptureSource::Runner;
        validate_prefill_capture_args(&cli).expect("runner capture should accept early-stop");

        cli.runner_early_stop_layer = None;
        cli.prefill_capture_source = PrefillCaptureSource::Worker;
        validate_prefill_capture_args(&cli)
            .expect("non-early-stop worker capture should remain valid");
    }

    #[test]
    fn ppp_final_norm_intermediate_requires_runner_prefill_ppp_mode() {
        let mut cli = sample_cli();
        cli.ppp_final_norm_intermediate = true;

        let err = validate_prefill_capture_args(&cli)
            .expect_err("exact PPP mode should require prefill capture");
        assert!(err
            .to_string()
            .contains("--ppp-final-norm-intermediate requires --prefill-last-position-only"));

        cli.prefill_last_position_only = true;
        let err = validate_prefill_capture_args(&cli)
            .expect_err("exact PPP mode should require runner capture");
        assert!(err
            .to_string()
            .contains("--ppp-final-norm-intermediate requires --prefill-capture-source runner"));

        cli.prefill_capture_source = PrefillCaptureSource::Runner;
        let err = validate_prefill_capture_args(&cli)
            .expect_err("exact PPP mode should require a PPP artifact input");
        assert!(err
            .to_string()
            .contains("--ppp-final-norm-intermediate requires --ppp-artifact-json"));

        cli.ppp_artifact_json = Some(PathBuf::from("/tmp/ppp.json"));
        validate_prefill_capture_args(&cli).expect("valid exact PPP mode should pass");
    }
}
