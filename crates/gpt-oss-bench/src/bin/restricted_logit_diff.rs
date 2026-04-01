use std::collections::HashMap;
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
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser, Clone)]
#[command(about = "Restricted GPT-OSS logit differential diagnosis")]
struct Cli {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "Explain tensor parallelism in one short sentence.")]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_model_len: usize,

    #[arg(long, default_value_t = 0.75)]
    gpu_memory_utilization: f32,

    #[arg(long, default_value_t = 8)]
    top_k: usize,

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

fn run_child(cli: &Cli, mode: ChildMode) -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained(&cli.model)?;
    let prompt_token_ids = tokenizer.encode(&cli.prompt)?;
    let summary = capture_summary(cli, mode, &tokenizer, &prompt_token_ids)?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn spawn_child(cli: &Cli, mode: ChildMode, forced_output_tokens: &[TokenId]) -> Result<ChildSummary> {
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
                vec![*forced_output_token_ids.last().expect("decode step must have a forced token")]
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
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?).with_context(|| {
            format!("failed to parse {}", config_path.display())
        })?;
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
        bail!("logits buffer too small: {} < vocab {}", logits.len(), vocab_size);
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

fn build_report(worker: &ChildSummary, runner: &ChildSummary, top_k: usize) -> Result<DifferentialReport> {
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
        largest.sort_by(|a, b| b.abs_diff.partial_cmp(&a.abs_diff).unwrap_or(std::cmp::Ordering::Equal));
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
