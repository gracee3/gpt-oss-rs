//! Correctness-first GPU0 owner shell for the real GPT-OSS decode layer.
//!
//! This bounded seam intentionally covers only GPT-OSS BF16 dense attention,
//! private K/V preparation, and the post-attention router input. Expert
//! dispatch and deterministic reduction remain the H2-H5 contracts.

use std::sync::Arc;

use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::device::{list_devices, resolve_stable_device, StableCudaDeviceId};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::kernel_loader::{compiled_ptx_dir, KernelLoader};
use gpt_oss_gpu::pinned_memory::BoundedPinnedLease;

use crate::cpu_runner::{CpuGptOssConfig, CpuKvCacheSnapshot};
use crate::model_loader::owner_selective::{LayerOwnerDenseTensor, OwnerSelectiveModel};

use super::contract::{GptOssPhase, GPT_OSS_HIDDEN_SIZE};
use super::reduction::CudaRankOrderedReducer;
use super::router::{CudaExactRouter, ExactRouterExecution};

const MODULE: &str = "gpt_oss_layer_owner";
const EMBEDDING: &str = "gpt_oss_layer_embedding_kernel";
const RMS_NORM: &str = "gpt_oss_layer_rms_norm_kernel";
const PROJECTION: &str = "gpt_oss_layer_bf16_projection_kernel";
const ROPE: &str = "gpt_oss_layer_rope_kernel";
const APPEND_KV: &str = "gpt_oss_layer_append_kv_kernel";
const ATTENTION: &str = "gpt_oss_layer_attention_kernel";
const RESIDUAL: &str = "gpt_oss_layer_residual_kernel";
const THREADS: u32 = 256;
const NUM_HEADS: usize = 64;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const Q_WIDTH: usize = NUM_HEADS * HEAD_DIM;
const KV_WIDTH: usize = NUM_KV_HEADS * HEAD_DIM;
const MAX_VISIBLE_TOKENS: usize = 128;
const ROPE_PAIRS: usize = HEAD_DIM / 2;

pub const GPT_OSS_LAYER_OWNER_WORK_BYTES: usize = (GPT_OSS_HIDDEN_SIZE * 5
    + Q_WIDTH * 2
    + KV_WIDTH * 2
    + MAX_VISIBLE_TOKENS * KV_WIDTH * 2
    + ROPE_PAIRS * 2)
    * size_of::<u16>();

/// Exact BF16 boundaries downloaded only for the one-layer oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerOwnerShellExecution {
    pub hidden_bf16_bits: Vec<u16>,
    pub input_norm_bf16_bits: Vec<u16>,
    pub query_after_rope_bf16_bits: Vec<u16>,
    pub key_after_rope_bf16_bits: Vec<u16>,
    pub value_projection_bf16_bits: Vec<u16>,
    pub attention_context_bf16_bits: Vec<u16>,
    pub attention_projection_bf16_bits: Vec<u16>,
    pub post_attention_residual_bf16_bits: Vec<u16>,
    pub router_input_bf16_bits: Vec<u16>,
    pub kernel_elapsed_ms: f32,
}

/// Fixed-allocation decode-M=1 GPU0 attention/dense owner shell.
pub struct CudaLayerOwnerShell {
    stable_device: StableCudaDeviceId,
    stream: Arc<CudaStream>,
    loader: KernelLoader,
    hidden: CudaSlice<u16>,
    input_norm: CudaSlice<u16>,
    query: CudaSlice<u16>,
    key: CudaSlice<u16>,
    value: CudaSlice<u16>,
    keys: CudaSlice<u16>,
    values: CudaSlice<u16>,
    cosine: CudaSlice<u16>,
    sine: CudaSlice<u16>,
    attention: CudaSlice<u16>,
    attention_projection: CudaSlice<u16>,
    post_attention_residual: CudaSlice<u16>,
    router_input: CudaSlice<u16>,
}

impl CudaLayerOwnerShell {
    pub fn new(model: &OwnerSelectiveModel, config: &CpuGptOssConfig) -> Result<Self> {
        validate_config(model, config)?;
        let stable_device = model.placement().layer_owner().stable_id.clone();
        let resolved = resolve_stable_device(&stable_device, &list_devices())
            .map_err(|error| LLMError::GpuError(format!("stable layer-owner device: {error}")))?;
        let context = CudaContext::new(resolved.transient_ordinal)
            .map_err(cuda_error("layer-owner context"))?;
        let stream = context
            .new_stream()
            .map_err(cuda_error("layer-owner stream"))?;
        let loader = KernelLoader::new(
            Arc::clone(&context),
            Arc::clone(&stream),
            compiled_ptx_dir(),
        )?;
        for function in [
            EMBEDDING, RMS_NORM, PROJECTION, ROPE, APPEND_KV, ATTENTION, RESIDUAL,
        ] {
            if !loader.has_func(MODULE, function) {
                return Err(LLMError::GpuError(format!(
                    "layer-owner PTX function {function} is unavailable"
                )));
            }
        }
        let alloc = |values, label| stream.alloc_zeros::<u16>(values).map_err(cuda_error(label));
        let shell = Self {
            stable_device,
            hidden: alloc(GPT_OSS_HIDDEN_SIZE, "layer-owner hidden allocation")?,
            input_norm: alloc(GPT_OSS_HIDDEN_SIZE, "layer-owner input-norm allocation")?,
            query: alloc(Q_WIDTH, "layer-owner query allocation")?,
            key: alloc(KV_WIDTH, "layer-owner key allocation")?,
            value: alloc(KV_WIDTH, "layer-owner value allocation")?,
            keys: alloc(
                MAX_VISIBLE_TOKENS * KV_WIDTH,
                "layer-owner key-cache allocation",
            )?,
            values: alloc(
                MAX_VISIBLE_TOKENS * KV_WIDTH,
                "layer-owner value-cache allocation",
            )?,
            cosine: alloc(ROPE_PAIRS, "layer-owner cosine allocation")?,
            sine: alloc(ROPE_PAIRS, "layer-owner sine allocation")?,
            attention: alloc(Q_WIDTH, "layer-owner attention allocation")?,
            attention_projection: alloc(
                GPT_OSS_HIDDEN_SIZE,
                "layer-owner attention-projection allocation",
            )?,
            post_attention_residual: alloc(
                GPT_OSS_HIDDEN_SIZE,
                "layer-owner post-attention allocation",
            )?,
            router_input: alloc(GPT_OSS_HIDDEN_SIZE, "layer-owner router-input allocation")?,
            stream,
            loader,
        };
        shell
            .stream
            .synchronize()
            .map_err(cuda_error("layer-owner construction drain"))?;
        Ok(shell)
    }

    pub fn stable_device(&self) -> &StableCudaDeviceId {
        &self.stable_device
    }

    pub const fn owned_device_bytes(&self) -> usize {
        GPT_OSS_LAYER_OWNER_WORK_BYTES
    }

    pub fn drain(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(cuda_error("layer-owner drain"))
    }

    /// Execute the real layer-0 dense/attention prefix from H3 resident BF16
    /// tensors. The cache image is committed CPU-oracle state; the current K/V
    /// row remains private to this shell and is never published by this call.
    pub fn execute_layer0_decode(
        &mut self,
        model: &OwnerSelectiveModel,
        config: &CpuGptOssConfig,
        token_id: u32,
        position: usize,
        cache: &CpuKvCacheSnapshot,
    ) -> Result<LayerOwnerShellExecution> {
        validate_config(model, config)?;
        if model.placement().layer_owner().stable_id != self.stable_device
            || usize::try_from(token_id).map_or(true, |token| token >= config.vocab_size)
            || cache.token_width != KV_WIDTH
            || cache.capacity != MAX_VISIBLE_TOKENS
            || cache.len >= MAX_VISIBLE_TOKENS
            || cache.start_position.checked_add(cache.len) != Some(position)
            || cache.keys_bf16_bits.len() != cache.len * KV_WIDTH
            || cache.values_bf16_bits.len() != cache.len * KV_WIDTH
        {
            return Err(LLMError::ModelError(
                "invalid layer-owner decode identity/cache shape".into(),
            ));
        }
        let embeddings = dense(
            model,
            "model.embed_tokens.weight",
            config.vocab_size * GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let input_norm_weight = dense(
            model,
            "model.layers.0.input_layernorm.weight",
            GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let q_weight = dense(
            model,
            "model.layers.0.self_attn.q_proj.weight",
            Q_WIDTH * GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let q_bias = dense(model, "model.layers.0.self_attn.q_proj.bias", Q_WIDTH * 2)?;
        let k_weight = dense(
            model,
            "model.layers.0.self_attn.k_proj.weight",
            KV_WIDTH * GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let k_bias = dense(model, "model.layers.0.self_attn.k_proj.bias", KV_WIDTH * 2)?;
        let v_weight = dense(
            model,
            "model.layers.0.self_attn.v_proj.weight",
            KV_WIDTH * GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let v_bias = dense(model, "model.layers.0.self_attn.v_proj.bias", KV_WIDTH * 2)?;
        let sinks = dense(model, "model.layers.0.self_attn.sinks", NUM_HEADS * 2)?;
        let o_weight = dense(
            model,
            "model.layers.0.self_attn.o_proj.weight",
            GPT_OSS_HIDDEN_SIZE * Q_WIDTH * 2,
        )?;
        let o_bias = dense(
            model,
            "model.layers.0.self_attn.o_proj.bias",
            GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let post_norm_weight = dense(
            model,
            "model.layers.0.post_attention_layernorm.weight",
            GPT_OSS_HIDDEN_SIZE * 2,
        )?;
        let (cosine, sine) = rope_tables(config, position)?;

        let submitted = (|| -> Result<_> {
            if cache.len > 0 {
                self.stream
                    .memcpy_htod(
                        &cache.keys_bf16_bits,
                        &mut self.keys.slice_mut(..cache.len * KV_WIDTH),
                    )
                    .map_err(cuda_error("layer-owner prior key H2D"))?;
                self.stream
                    .memcpy_htod(
                        &cache.values_bf16_bits,
                        &mut self.values.slice_mut(..cache.len * KV_WIDTH),
                    )
                    .map_err(cuda_error("layer-owner prior value H2D"))?;
            }
            self.stream
                .memcpy_htod(&cosine, &mut self.cosine)
                .map_err(cuda_error("layer-owner cosine H2D"))?;
            self.stream
                .memcpy_htod(&sine, &mut self.sine)
                .map_err(cuda_error("layer-owner sine H2D"))?;
            let start = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("layer-owner start event"))?;
            launch_embedding(
                &self.stream,
                &self.loader,
                embeddings.allocation(),
                &mut self.hidden,
                token_id,
            )?;
            launch_rms_norm(
                &self.stream,
                &self.loader,
                &self.hidden,
                input_norm_weight.allocation(),
                &mut self.input_norm,
                config.rms_norm_eps,
            )?;
            launch_projection(
                &self.stream,
                &self.loader,
                &self.input_norm,
                q_weight.allocation(),
                q_bias.allocation(),
                &mut self.query,
                Q_WIDTH,
            )?;
            launch_projection(
                &self.stream,
                &self.loader,
                &self.input_norm,
                k_weight.allocation(),
                k_bias.allocation(),
                &mut self.key,
                KV_WIDTH,
            )?;
            launch_projection(
                &self.stream,
                &self.loader,
                &self.input_norm,
                v_weight.allocation(),
                v_bias.allocation(),
                &mut self.value,
                KV_WIDTH,
            )?;
            launch_rope(
                &self.stream,
                &self.loader,
                &mut self.query,
                &self.cosine,
                &self.sine,
                NUM_HEADS,
            )?;
            launch_rope(
                &self.stream,
                &self.loader,
                &mut self.key,
                &self.cosine,
                &self.sine,
                NUM_KV_HEADS,
            )?;
            launch_append_kv(
                &self.stream,
                &self.loader,
                &self.key,
                &self.value,
                &mut self.keys,
                &mut self.values,
                cache.len,
            )?;
            launch_attention(
                &self.stream,
                &self.loader,
                &self.query,
                &self.keys,
                &self.values,
                sinks.allocation(),
                &mut self.attention,
                cache.len + 1,
            )?;
            launch_projection(
                &self.stream,
                &self.loader,
                &self.attention,
                o_weight.allocation(),
                o_bias.allocation(),
                &mut self.attention_projection,
                GPT_OSS_HIDDEN_SIZE,
            )?;
            launch_residual(
                &self.stream,
                &self.loader,
                &self.hidden,
                &self.attention_projection,
                &mut self.post_attention_residual,
            )?;
            launch_rms_norm(
                &self.stream,
                &self.loader,
                &self.post_attention_residual,
                post_norm_weight.allocation(),
                &mut self.router_input,
                config.rms_norm_eps,
            )?;
            let terminal = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("layer-owner terminal event"))?;
            Ok((start, terminal))
        })();
        let (start, terminal) = match submitted {
            Ok(events) => events,
            Err(primary) => {
                return match self.stream.synchronize() {
                    Ok(()) => Err(primary),
                    Err(drain) => Err(LLMError::GpuError(format!(
                        "layer-owner submit failed ({primary}); mandatory drain failed ({drain})"
                    ))),
                };
            }
        };
        terminal
            .synchronize()
            .map_err(cuda_error("layer-owner terminal drain"))?;
        let kernel_elapsed_ms = start
            .elapsed_ms(&terminal)
            .map_err(cuda_error("layer-owner event timing"))?;
        Ok(LayerOwnerShellExecution {
            hidden_bf16_bits: download(&self.stream, &self.hidden, "hidden")?,
            input_norm_bf16_bits: download(&self.stream, &self.input_norm, "input norm")?,
            query_after_rope_bf16_bits: download(&self.stream, &self.query, "query")?,
            key_after_rope_bf16_bits: download(&self.stream, &self.key, "key")?,
            value_projection_bf16_bits: download(&self.stream, &self.value, "value")?,
            attention_context_bf16_bits: download(&self.stream, &self.attention, "attention")?,
            attention_projection_bf16_bits: download(
                &self.stream,
                &self.attention_projection,
                "attention projection",
            )?,
            post_attention_residual_bf16_bits: download(
                &self.stream,
                &self.post_attention_residual,
                "post-attention residual",
            )?,
            router_input_bf16_bits: download(&self.stream, &self.router_input, "router input")?,
            kernel_elapsed_ms,
        })
    }

    /// Hand the resident post-attention norm to the exact GPU0 router through
    /// a same-device D2D copy. Host downloads remain evidence-only.
    #[allow(clippy::too_many_arguments)]
    pub fn route_resident_decode(
        &self,
        router: &mut CudaExactRouter,
        layer: u16,
        placement_epoch: u64,
        source_activation: &mut BoundedPinnedLease<u16>,
        route_records: &mut BoundedPinnedLease<u8>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<ExactRouterExecution> {
        router.execute_device_and_download(
            layer,
            GptOssPhase::Decode,
            placement_epoch,
            1,
            &self.router_input,
            &self.stable_device,
            source_activation,
            route_records,
            timeline,
        )
    }

    /// Apply the final residual from the reducer's still-resident output by a
    /// same-GPU D2D copy. The reducer terminal has drained before this call.
    pub fn finish_layer_residual_resident(
        &mut self,
        reducer: &CudaRankOrderedReducer,
    ) -> Result<Vec<u16>> {
        if reducer.stable_device() != &self.stable_device {
            return Err(LLMError::GpuError(
                "resident reducer/layer-owner device mismatch".into(),
            ));
        }
        let submitted = (|| -> Result<()> {
            self.stream
                .memcpy_dtod(reducer.output_device(), &mut self.attention_projection)
                .map_err(cuda_error("layer-owner reduced update D2D"))?;
            launch_residual(
                &self.stream,
                &self.loader,
                &self.post_attention_residual,
                &self.attention_projection,
                &mut self.router_input,
            )?;
            self.stream
                .synchronize()
                .map_err(cuda_error("layer-owner resident final residual drain"))
        })();
        if let Err(primary) = submitted {
            return match self.stream.synchronize() {
                Ok(()) => Err(primary),
                Err(drain) => Err(LLMError::GpuError(format!(
                    "resident final residual failed ({primary}); mandatory drain failed ({drain})"
                ))),
            };
        }
        download(&self.stream, &self.router_input, "layer output")
    }
}

fn validate_config(model: &OwnerSelectiveModel, config: &CpuGptOssConfig) -> Result<()> {
    let native = model.checkpoint().config();
    if native.num_hidden_layers != config.num_hidden_layers
        || native.num_experts != config.num_local_experts
        || native.vocab_size != config.vocab_size
        || config.hidden_size != GPT_OSS_HIDDEN_SIZE
        || config.num_attention_heads != NUM_HEADS
        || config.num_key_value_heads != NUM_KV_HEADS
        || config.head_dim != HEAD_DIM
        || config.sliding_window != MAX_VISIBLE_TOKENS
        || config.layer_types.first().map(String::as_str) != Some("sliding_attention")
    {
        return Err(LLMError::ModelError(
            "layer-owner shell supports the validated GPT-OSS decode shape only".into(),
        ));
    }
    Ok(())
}

fn dense<'a>(
    model: &'a OwnerSelectiveModel,
    name: &str,
    expected_bytes: usize,
) -> Result<&'a LayerOwnerDenseTensor> {
    let tensor = model
        .layer_owner_dense()
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| LLMError::ModelError(format!("missing layer-owner dense tensor {name}")))?;
    if tensor.logical_bytes != expected_bytes as u64 || tensor.device_bytes() != expected_bytes {
        return Err(LLMError::ModelError(format!(
            "layer-owner dense tensor {name} has {} logical/{} device bytes, expected {expected_bytes}",
            tensor.logical_bytes,
            tensor.device_bytes()
        )));
    }
    Ok(tensor)
}

fn rope_tables(
    config: &CpuGptOssConfig,
    position: usize,
) -> Result<([u16; ROPE_PAIRS], [u16; ROPE_PAIRS])> {
    let scaling = config.rope_scaling.as_ref().ok_or_else(|| {
        LLMError::ModelError("layer-owner proof requires the validated YaRN configuration".into())
    })?;
    let correction = |rotations: f64| {
        (HEAD_DIM as f64
            * (scaling.original_max_position_embeddings as f64
                / (rotations * 2.0 * std::f64::consts::PI))
                .ln())
            / (2.0 * config.rope_theta.ln())
    };
    let mut low = correction(scaling.beta_fast);
    let mut high = correction(scaling.beta_slow);
    if scaling.truncate {
        low = low.floor();
        high = high.ceil();
    }
    low = low.max(0.0);
    high = high.min((HEAD_DIM - 1) as f64);
    let range = (high - low).abs().max(0.001);
    let attention_scale = (0.1 * scaling.factor.ln() + 1.0) as f32;
    let cosine = std::array::from_fn(|index| {
        let frequency = (config.rope_theta as f32).powf((index * 2) as f32 / HEAD_DIM as f32);
        let extrapolation = 1.0_f32 / frequency;
        let interpolation = 1.0_f32 / (scaling.factor as f32 * frequency);
        let ramp = ((index as f32 - low as f32) / range as f32).clamp(0.0, 1.0);
        let inverse_frequency = interpolation * ramp + extrapolation * (1.0 - ramp);
        half::bf16::from_f32((position as f32 * inverse_frequency).cos() * attention_scale)
            .to_bits()
    });
    let sine = std::array::from_fn(|index| {
        let frequency = (config.rope_theta as f32).powf((index * 2) as f32 / HEAD_DIM as f32);
        let extrapolation = 1.0_f32 / frequency;
        let interpolation = 1.0_f32 / (scaling.factor as f32 * frequency);
        let ramp = ((index as f32 - low as f32) / range as f32).clamp(0.0, 1.0);
        let inverse_frequency = interpolation * ramp + extrapolation * (1.0 - ramp);
        half::bf16::from_f32((position as f32 * inverse_frequency).sin() * attention_scale)
            .to_bits()
    });
    Ok((cosine, sine))
}

fn download(
    stream: &Arc<CudaStream>,
    source: &CudaSlice<u16>,
    label: &'static str,
) -> Result<Vec<u16>> {
    stream.clone_dtoh(source).map_err(cuda_error(label))
}

fn grid(values: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((values as u32).div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_embedding(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    embeddings: &CudaSlice<u8>,
    hidden: &mut CudaSlice<u16>,
    token_id: u32,
) -> Result<()> {
    let function = loader.get_func(MODULE, EMBEDDING)?;
    let token = i32::try_from(token_id)
        .map_err(|_| LLMError::ModelError("token id overflows i32".into()))?;
    let width = GPT_OSS_HIDDEN_SIZE as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(embeddings)
            .arg(hidden)
            .arg(&token)
            .arg(&width)
            .launch(grid(GPT_OSS_HIDDEN_SIZE))
            .map_err(cuda_error("layer-owner embedding launch"))?;
    }
    Ok(())
}

fn launch_rms_norm(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    input: &CudaSlice<u16>,
    weight: &CudaSlice<u8>,
    output: &mut CudaSlice<u16>,
    epsilon: f32,
) -> Result<()> {
    let function = loader.get_func(MODULE, RMS_NORM)?;
    let width = GPT_OSS_HIDDEN_SIZE as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(input)
            .arg(weight)
            .arg(output)
            .arg(&width)
            .arg(&epsilon)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(cuda_error("layer-owner RMS launch"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_projection(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    input: &CudaSlice<u16>,
    weight: &CudaSlice<u8>,
    bias: &CudaSlice<u8>,
    output: &mut CudaSlice<u16>,
    rows: usize,
) -> Result<()> {
    let function = loader.get_func(MODULE, PROJECTION)?;
    let rows_i32 = rows as i32;
    let columns = input.len() as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(input)
            .arg(weight)
            .arg(bias)
            .arg(output)
            .arg(&rows_i32)
            .arg(&columns)
            .launch(grid(rows))
            .map_err(cuda_error("layer-owner projection launch"))?;
    }
    Ok(())
}

fn launch_rope(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    values: &mut CudaSlice<u16>,
    cosine: &CudaSlice<u16>,
    sine: &CudaSlice<u16>,
    heads: usize,
) -> Result<()> {
    let function = loader.get_func(MODULE, ROPE)?;
    let heads_i32 = heads as i32;
    let head_dim = HEAD_DIM as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(values)
            .arg(cosine)
            .arg(sine)
            .arg(&heads_i32)
            .arg(&head_dim)
            .launch(grid(heads * ROPE_PAIRS))
            .map_err(cuda_error("layer-owner RoPE launch"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_append_kv(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    key: &CudaSlice<u16>,
    value: &CudaSlice<u16>,
    keys: &mut CudaSlice<u16>,
    values: &mut CudaSlice<u16>,
    token: usize,
) -> Result<()> {
    let function = loader.get_func(MODULE, APPEND_KV)?;
    let token_i32 = token as i32;
    let width = KV_WIDTH as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(key)
            .arg(value)
            .arg(keys)
            .arg(values)
            .arg(&token_i32)
            .arg(&width)
            .launch(grid(KV_WIDTH))
            .map_err(cuda_error("layer-owner K/V append launch"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_attention(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    query: &CudaSlice<u16>,
    keys: &CudaSlice<u16>,
    values: &CudaSlice<u16>,
    sinks: &CudaSlice<u8>,
    output: &mut CudaSlice<u16>,
    visible: usize,
) -> Result<()> {
    let function = loader.get_func(MODULE, ATTENTION)?;
    let visible_i32 = visible as i32;
    let heads = NUM_HEADS as i32;
    let kv_heads = NUM_KV_HEADS as i32;
    let head_dim = HEAD_DIM as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(query)
            .arg(keys)
            .arg(values)
            .arg(sinks)
            .arg(output)
            .arg(&visible_i32)
            .arg(&heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .launch(grid(NUM_HEADS))
            .map_err(cuda_error("layer-owner attention launch"))?;
    }
    Ok(())
}

fn launch_residual(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    residual: &CudaSlice<u16>,
    update: &CudaSlice<u16>,
    output: &mut CudaSlice<u16>,
) -> Result<()> {
    let function = loader.get_func(MODULE, RESIDUAL)?;
    let values = GPT_OSS_HIDDEN_SIZE as i32;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(residual)
            .arg(update)
            .arg(output)
            .arg(&values)
            .launch(grid(GPT_OSS_HIDDEN_SIZE))
            .map_err(cuda_error("layer-owner residual launch"))?;
    }
    Ok(())
}

fn cuda_error(context: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{context}: {error}"))
}
