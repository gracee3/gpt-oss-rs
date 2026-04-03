use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::worker::gpu_worker::GpuWorker;
use gpt_oss_engine::{RuntimeMode, SequenceData, SequenceGroupMetadata, WorkerConfig};
use gpt_oss_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser, Clone)]
#[command(about = "Restricted GPT-OSS final-position top-k prefill capture")]
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

    #[arg(long, default_value = ".live/restricted-prefill-topk.json")]
    output: PathBuf,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TopLogit {
    token_id: TokenId,
    token_text: String,
    logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefillTopKCapture {
    prompt_token_count: usize,
    last_input_token_id: TokenId,
    last_input_token_text: String,
    chosen_token_id: TokenId,
    chosen_token_text: String,
    restricted_model_path: String,
    visible_devices: String,
    wall_clock_seconds: f64,
    final_position_top_k: Vec<TopLogit>,
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

    let started = Instant::now();
    let tokenizer = Tokenizer::from_pretrained(&cli.model)?;
    let prompt_token_ids = tokenizer.encode(&cli.prompt)?;
    let last_input_token_id = *prompt_token_ids
        .last()
        .context("prompt must contain at least one token")?;
    let last_input_token_text = tokenizer.decode(&[last_input_token_id]).unwrap_or_default();
    let model_path = Path::new(&cli.model);
    let mut worker = build_worker(model_path, cli.max_model_len, cli.gpu_memory_utilization)?;
    let metadata = build_single_sequence_metadata(&prompt_token_ids);
    let last_logits = worker.debug_last_token_logits(&metadata)?;
    let chosen_token_id = argmax_token(&last_logits)?;
    let chosen_token_text = tokenizer.decode(&[chosen_token_id]).unwrap_or_default();
    let capture = PrefillTopKCapture {
        prompt_token_count: prompt_token_ids.len(),
        last_input_token_id,
        last_input_token_text,
        chosen_token_id,
        chosen_token_text,
        restricted_model_path: cli.model,
        visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".into()),
        wall_clock_seconds: started.elapsed().as_secs_f64(),
        final_position_top_k: top_k_logits(&last_logits, cli.top_k, &tokenizer)?,
    };

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.output, serde_json::to_vec_pretty(&capture)?)?;
    println!("{}", serde_json::to_string_pretty(&capture)?);
    Ok(())
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
