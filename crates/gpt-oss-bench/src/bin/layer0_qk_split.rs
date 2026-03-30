use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_core::prelude::{BlockId, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::worker::gpu_worker::GpuWorker;
use gpt_oss_engine::{RuntimeMode, SequenceData, SequenceGroupMetadata, WorkerConfig};
use gpt_oss_tokenizer::Tokenizer;
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Serialize;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser, Clone)]
#[command(about = "Layer-0 Q/K projection split: GPU weights vs GEMM path")]
struct Cli {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "/data/models/openai/gpt-oss-20b")]
    original_model: String,

    #[arg(long, default_value = "Explain tensor parallelism in one short sentence.")]
    prompt: String,

    #[arg(long, default_value_t = 128)]
    max_model_len: usize,

    #[arg(long, default_value_t = 0.75)]
    gpu_memory_utilization: f32,

    #[arg(long, default_value = ".live/layer0-qk-split.json")]
    output: PathBuf,

    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffSummary {
    max_abs_diff: f32,
    mean_abs_diff: f32,
}

#[derive(Debug, Clone, Serialize)]
struct WeightDiffSummary {
    restricted: DiffSummary,
    original: DiffSummary,
}

#[derive(Debug, Clone, Serialize)]
struct ProjectionSplitReport {
    uses_fused_qkv: bool,
    q_shape: [usize; 2],
    k_shape: [usize; 2],
    v_shape: [usize; 2],
    q_proj_weight_diff: WeightDiffSummary,
    k_proj_weight_diff: WeightDiffSummary,
    q_proj_pre_bias_output_diff_gpu_vs_cpu: DiffSummary,
    k_proj_pre_bias_output_diff_gpu_vs_cpu: DiffSummary,
    v_proj_pre_bias_output_diff_gpu_vs_cpu: DiffSummary,
    q_proj_output_diff_gpu_vs_cpu: DiffSummary,
    k_proj_output_diff_gpu_vs_cpu: DiffSummary,
    v_proj_output_diff_gpu_vs_cpu: DiffSummary,
    q_proj_post_bias_diff_gpu_vs_expected: DiffSummary,
    k_proj_post_bias_diff_gpu_vs_expected: DiffSummary,
    v_proj_post_bias_diff_gpu_vs_expected: DiffSummary,
}

fn add_bias(values: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut out = values.to_vec();
    if let Some(bias) = bias {
        for (dst, b) in out.iter_mut().zip(bias.iter()) {
            *dst += *b;
        }
    }
    out
}

fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

fn max_mean_diff(lhs: &[f32], rhs: &[f32]) -> Result<DiffSummary> {
    if lhs.len() != rhs.len() {
        bail!("diff length mismatch: {} vs {}", lhs.len(), rhs.len());
    }
    let mut max_abs_diff = 0.0f32;
    let mut sum = 0.0f64;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (a - b).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum += diff as f64;
    }
    Ok(DiffSummary {
        max_abs_diff,
        mean_abs_diff: (sum / lhs.len().max(1) as f64) as f32,
    })
}

fn decode_tensor_to_f32(view: safetensors::tensor::TensorView<'_>, narrow_to_f16: bool) -> Result<Vec<f32>> {
    let data = view.data();
    let values = match view.dtype() {
        SafeDtype::F16 => {
            let words = bytemuck::try_cast_slice::<u8, u16>(data)
                .map_err(|e| anyhow::anyhow!("cast f16 tensor bytes: {e}"))?;
            words
                .iter()
                .map(|bits| f16::from_bits(*bits).to_f32())
                .collect()
        }
        SafeDtype::BF16 => {
            let words = bytemuck::try_cast_slice::<u8, u16>(data)
                .map_err(|e| anyhow::anyhow!("cast bf16 tensor bytes: {e}"))?;
            if narrow_to_f16 {
                words
                    .iter()
                    .map(|bits| f16::from_f32(bf16::from_bits(*bits).to_f32()).to_f32())
                    .collect()
            } else {
                words
                    .iter()
                    .map(|bits| bf16::from_bits(*bits).to_f32())
                    .collect()
            }
        }
        SafeDtype::F32 => {
            let words = bytemuck::try_cast_slice::<u8, f32>(data)
                .map_err(|e| anyhow::anyhow!("cast f32 tensor bytes: {e}"))?;
            if narrow_to_f16 {
                words.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
            } else {
                words.to_vec()
            }
        }
        other => bail!("unsupported tensor dtype {other:?}"),
    };
    Ok(values)
}

fn load_tensor_f32(model_dir: &Path, name: &str, narrow_to_f16: bool) -> Result<Vec<f32>> {
    let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().map(|ext| ext == "safetensors").unwrap_or(false))
        .collect();
    shards.sort();
    for shard in shards {
        let file = File::open(&shard)?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap {}", shard.display()))?;
        let tensors =
            SafeTensors::deserialize(&mmap).with_context(|| format!("parse {}", shard.display()))?;
        if let Ok(view) = tensors.tensor(name) {
            return decode_tensor_to_f32(view, narrow_to_f16);
        }
    }
    bail!("tensor not found: {}", name)
}

fn linear_last_token(input: &[f32], weight: &[f32], out_dim: usize, in_dim: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut acc = bias.and_then(|b| b.get(row)).copied().unwrap_or(0.0);
        let offset = row * in_dim;
        for col in 0..in_dim {
            acc += input[col] * weight[offset + col];
        }
        output[row] = acc;
    }
    output
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    let tokenizer = Tokenizer::from_pretrained(&cli.model)?;
    let prompt_token_ids = tokenizer.encode(&cli.prompt)?;
    let model_path = Path::new(&cli.model);
    let original_model_path = Path::new(&cli.original_model);
    let worker = build_worker(model_path, cli.max_model_len, cli.gpu_memory_utilization)?;
    let metadata = build_single_sequence_metadata(&prompt_token_ids);
    let trace = worker.debug_runner_prefill_trace(&metadata)?;
    let snapshot = worker.debug_runner_layer0_projection_snapshot()?;
    let attention = trace
        .layers
        .first()
        .and_then(|layer| layer.attention.as_ref())
        .context("missing layer0 attention trace")?;

    let q_name = "model.layers.0.self_attn.q_proj.weight";
    let k_name = "model.layers.0.self_attn.k_proj.weight";
    let q_bias_name = "model.layers.0.self_attn.q_proj.bias";
    let k_bias_name = "model.layers.0.self_attn.k_proj.bias";

    let restricted_q = load_tensor_f32(model_path, q_name, true)?;
    let restricted_k = load_tensor_f32(model_path, k_name, true)?;
    let original_q = load_tensor_f32(original_model_path, q_name, true)?;
    let original_k = load_tensor_f32(original_model_path, k_name, true)?;

    let runtime_norm = &attention.attention_norm_output;
    let cpu_q = linear_last_token(
        runtime_norm,
        &snapshot.q_weight,
        snapshot.q_dim,
        snapshot.hidden_size,
        snapshot.q_bias.as_deref(),
    );
    let cpu_k = linear_last_token(
        runtime_norm,
        &snapshot.k_weight,
        snapshot.kv_dim,
        snapshot.hidden_size,
        snapshot.k_bias.as_deref(),
    );
    let cpu_v = linear_last_token(
        runtime_norm,
        &snapshot.v_weight,
        snapshot.kv_dim,
        snapshot.hidden_size,
        snapshot.v_bias.as_deref(),
    );
    let cpu_q_pre_bias = linear_last_token(
        runtime_norm,
        &snapshot.q_weight,
        snapshot.q_dim,
        snapshot.hidden_size,
        None,
    );
    let cpu_k_pre_bias = linear_last_token(
        runtime_norm,
        &snapshot.k_weight,
        snapshot.kv_dim,
        snapshot.hidden_size,
        None,
    );
    let cpu_v_pre_bias = linear_last_token(
        runtime_norm,
        &snapshot.v_weight,
        snapshot.kv_dim,
        snapshot.hidden_size,
        None,
    );
    let expected_q_post_bias = add_bias(&attention.q_proj_pre_bias, snapshot.q_bias.as_deref());
    let expected_k_post_bias = add_bias(&attention.k_proj_pre_bias, snapshot.k_bias.as_deref());
    let expected_v_post_bias = add_bias(&attention.v_proj_pre_bias, snapshot.v_bias.as_deref());

    let report = ProjectionSplitReport {
        uses_fused_qkv: snapshot.uses_fused_qkv,
        q_shape: [snapshot.q_dim, snapshot.hidden_size],
        k_shape: [snapshot.kv_dim, snapshot.hidden_size],
        v_shape: [snapshot.kv_dim, snapshot.hidden_size],
        q_proj_weight_diff: WeightDiffSummary {
            restricted: max_mean_diff(&snapshot.q_weight, &restricted_q)?,
            original: max_mean_diff(&snapshot.q_weight, &original_q)?,
        },
        k_proj_weight_diff: WeightDiffSummary {
            restricted: max_mean_diff(&snapshot.k_weight, &restricted_k)?,
            original: max_mean_diff(&snapshot.k_weight, &original_k)?,
        },
        q_proj_pre_bias_output_diff_gpu_vs_cpu: max_mean_diff(
            &attention.q_proj_pre_bias,
            &cpu_q_pre_bias,
        )?,
        k_proj_pre_bias_output_diff_gpu_vs_cpu: max_mean_diff(
            &attention.k_proj_pre_bias,
            &cpu_k_pre_bias,
        )?,
        v_proj_pre_bias_output_diff_gpu_vs_cpu: max_mean_diff(
            &attention.v_proj_pre_bias,
            &cpu_v_pre_bias,
        )?,
        q_proj_output_diff_gpu_vs_cpu: max_mean_diff(&attention.q_proj, &cpu_q)?,
        k_proj_output_diff_gpu_vs_cpu: max_mean_diff(&attention.k_proj, &cpu_k)?,
        v_proj_output_diff_gpu_vs_cpu: max_mean_diff(&attention.v_proj, &cpu_v)?,
        q_proj_post_bias_diff_gpu_vs_expected: max_mean_diff(&attention.q_proj, &expected_q_post_bias)?,
        k_proj_post_bias_diff_gpu_vs_expected: max_mean_diff(&attention.k_proj, &expected_k_post_bias)?,
        v_proj_post_bias_diff_gpu_vs_expected: max_mean_diff(&attention.v_proj, &expected_v_post_bias)?,
    };

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    let _ = (q_bias_name, k_bias_name);
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
