//! Validation-only helpers for GPT-OSS MXFP4 expert tensors.
//!
//! This module exposes the existing runtime MXFP4 dequant kernel to bench
//! validation code without routing production execution through a new path.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::bridge::{LLMError, Result};
use gpt_oss_gpu::kernel_loader::KernelLoader;

#[derive(Debug, Clone, Serialize)]
pub struct Mxfp4ValidationTensorSource {
    pub model_path: String,
    pub shard_path: String,
    pub tensor_name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub value_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Mxfp4ValidationTensorSources {
    pub gate_up_proj_blocks: Mxfp4ValidationTensorSource,
    pub gate_up_proj_scales: Mxfp4ValidationTensorSource,
    pub gate_up_proj_bias: Mxfp4ValidationTensorSource,
    pub down_proj_blocks: Mxfp4ValidationTensorSource,
    pub down_proj_scales: Mxfp4ValidationTensorSource,
    pub down_proj_bias: Mxfp4ValidationTensorSource,
}

#[derive(Debug, Clone, Serialize)]
pub struct Mxfp4SelectedExpertWeights {
    pub expert: usize,
    pub gate_up_weight: Vec<f32>,
    pub gate_up_bias: Vec<f32>,
    pub down_weight: Vec<f32>,
    pub down_bias: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Mxfp4SelectedExpertsValidationWeights {
    pub helper_name: &'static str,
    pub decode_source: &'static str,
    pub dtype_outputs: &'static str,
    pub selected_experts: Vec<usize>,
    pub tensor_sources: Mxfp4ValidationTensorSources,
    pub experts: Vec<Mxfp4SelectedExpertWeights>,
}

#[derive(Debug, Deserialize)]
struct SafetensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct LoadedTensor {
    source: Mxfp4ValidationTensorSource,
    bytes: Vec<u8>,
}

/// Load and dequantize selected layer expert weights using the runtime MXFP4
/// dequant kernel. This is validation support only; production callers are not
/// routed through this helper.
pub fn load_selected_experts_mxfp4_validation(
    model_path: &Path,
    layer_idx: usize,
    selected_experts: &[usize],
) -> Result<Mxfp4SelectedExpertsValidationWeights> {
    let prefix = format!("model.layers.{layer_idx}.mlp.experts");
    let gate_up_blocks = load_tensor_bytes(model_path, &format!("{prefix}.gate_up_proj_blocks"))?;
    let gate_up_scales = load_tensor_bytes(model_path, &format!("{prefix}.gate_up_proj_scales"))?;
    let gate_up_bias = load_tensor_bytes(model_path, &format!("{prefix}.gate_up_proj_bias"))?;
    let down_blocks = load_tensor_bytes(model_path, &format!("{prefix}.down_proj_blocks"))?;
    let down_scales = load_tensor_bytes(model_path, &format!("{prefix}.down_proj_scales"))?;
    let down_bias = load_tensor_bytes(model_path, &format!("{prefix}.down_proj_bias"))?;

    ensure_shape_dtype(&gate_up_blocks, "U8", &[32, 5760, 90, 16])?;
    ensure_shape_dtype(&gate_up_scales, "U8", &[32, 5760, 90])?;
    ensure_shape_dtype(&gate_up_bias, "BF16", &[32, 5760])?;
    ensure_shape_dtype(&down_blocks, "U8", &[32, 2880, 90, 16])?;
    ensure_shape_dtype(&down_scales, "U8", &[32, 2880, 90])?;
    ensure_shape_dtype(&down_bias, "BF16", &[32, 2880])?;

    let context = CudaContext::new(0)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 validation CUDA context: {e}")))?;
    let stream = context
        .new_stream()
        .map_err(|e| LLMError::GpuError(format!("MXFP4 validation CUDA stream: {e}")))?;
    let loader = KernelLoader::new(
        context.clone(),
        stream.clone(),
        gpt_oss_gpu::kernel_loader::default_ptx_dir(),
    )
    .map_err(|e| LLMError::GpuError(format!("MXFP4 validation kernel loader: {e}")))?;

    let gate_up_blocks_dev = stream
        .clone_htod(&gate_up_blocks.bytes)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 gate_up blocks upload: {e}")))?;
    let gate_up_scales_dev = stream
        .clone_htod(&gate_up_scales.bytes)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 gate_up scales upload: {e}")))?;
    let down_blocks_dev = stream
        .clone_htod(&down_blocks.bytes)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 down blocks upload: {e}")))?;
    let down_scales_dev = stream
        .clone_htod(&down_scales.bytes)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 down scales upload: {e}")))?;

    let gate_up_bias_values = bf16_bytes_to_f32(&gate_up_bias.bytes, "gate_up_proj_bias")?;
    let down_bias_values = bf16_bytes_to_f32(&down_bias.bytes, "down_proj_bias")?;

    let mut experts = Vec::with_capacity(selected_experts.len());
    for &expert in selected_experts {
        if expert >= 32 {
            return Err(LLMError::ModelError(format!(
                "selected expert index out of range for layer {layer_idx}: {expert}"
            )));
        }
        let gate_up_weight = dequant_expert_to_f32(
            &stream,
            &loader,
            &gate_up_blocks_dev,
            &gate_up_scales_dev,
            expert,
            5760,
            2880,
        )?;
        let down_weight = dequant_expert_to_f32(
            &stream,
            &loader,
            &down_blocks_dev,
            &down_scales_dev,
            expert,
            2880,
            2880,
        )?;
        experts.push(Mxfp4SelectedExpertWeights {
            expert,
            gate_up_weight,
            gate_up_bias: slice_bias(&gate_up_bias_values, expert, 5760),
            down_weight,
            down_bias: slice_bias(&down_bias_values, expert, 2880),
        });
    }

    Ok(Mxfp4SelectedExpertsValidationWeights {
        helper_name:
            "gpt_oss_model_runner::mxfp4_validation::load_selected_experts_mxfp4_validation",
        decode_source: "gpt_oss_dequant_expert_f16_kernel",
        dtype_outputs:
            "dequantized_f16_weights_and_bf16_biases_widened_to_f32_for_validation_replay",
        selected_experts: selected_experts.to_vec(),
        tensor_sources: Mxfp4ValidationTensorSources {
            gate_up_proj_blocks: gate_up_blocks.source,
            gate_up_proj_scales: gate_up_scales.source,
            gate_up_proj_bias: gate_up_bias.source,
            down_proj_blocks: down_blocks.source,
            down_proj_scales: down_scales.source,
            down_proj_bias: down_bias.source,
        },
        experts,
    })
}

fn dequant_expert_to_f32(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    blocks: &cudarc::driver::CudaSlice<u8>,
    scales: &cudarc::driver::CudaSlice<u8>,
    expert_idx: usize,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let total = out_features * in_features;
    let zeros = vec![f16::ZERO; total];
    let mut output = stream
        .clone_htod(&zeros)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 output allocation/upload: {e}")))?;
    let kernel = loader.get_func("gpt_oss_moe", "gpt_oss_dequant_expert_f16_kernel")?;
    let threads = 256u32;
    let blocks_grid = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks_grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let expert_idx = expert_idx as i32;
    let out_features = out_features as i32;
    let in_features = in_features as i32;
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&mut output)
            .arg(blocks)
            .arg(scales)
            .arg(&expert_idx)
            .arg(&out_features)
            .arg(&in_features)
            .launch(cfg)
            .map_err(|e| LLMError::GpuError(format!("MXFP4 dequant launch: {e}")))?;
    }
    stream
        .synchronize()
        .map_err(|e| LLMError::GpuError(format!("MXFP4 dequant synchronize: {e}")))?;
    let values = stream
        .clone_dtoh(&output)
        .map_err(|e| LLMError::GpuError(format!("MXFP4 dequant download: {e}")))?;
    Ok(values.iter().map(|value| value.to_f32()).collect())
}

fn load_tensor_bytes(model_path: &Path, tensor_name: &str) -> Result<LoadedTensor> {
    let shards = safetensor_shards(model_path)?;
    for shard in shards {
        if let Some((entry, bytes)) = try_load_tensor_bytes(&shard, tensor_name)? {
            let value_count = entry.shape.iter().product();
            return Ok(LoadedTensor {
                source: Mxfp4ValidationTensorSource {
                    model_path: model_path.display().to_string(),
                    shard_path: shard.display().to_string(),
                    tensor_name: tensor_name.to_string(),
                    dtype: entry.dtype,
                    shape: entry.shape,
                    value_count,
                },
                bytes,
            });
        }
    }
    Err(LLMError::ModelError(format!(
        "tensor {tensor_name} not found under {}",
        model_path.display()
    )))
}

fn safetensor_shards(model_path: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = if model_path.is_file() {
        vec![model_path.to_path_buf()]
    } else {
        let mut paths = fs::read_dir(model_path)
            .map_err(LLMError::IoError)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    };
    if shards.is_empty() {
        return Err(LLMError::ModelError(format!(
            "no safetensors shards found in {}",
            model_path.display()
        )));
    }
    Ok(std::mem::take(&mut shards))
}

fn try_load_tensor_bytes(
    shard: &Path,
    tensor_name: &str,
) -> Result<Option<(SafetensorEntry, Vec<u8>)>> {
    let mut file = fs::File::open(shard).map_err(LLMError::IoError)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(LLMError::IoError)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(LLMError::IoError)?;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&header_bytes)
        .map_err(|e| LLMError::SerializationError(e.to_string()))?;
    let Some(value) = header.get(tensor_name) else {
        return Ok(None);
    };
    let entry: SafetensorEntry = serde_json::from_value(value.clone())
        .map_err(|e| LLMError::SerializationError(e.to_string()))?;
    let data_start = 8u64 + header_len as u64;
    let byte_len = entry.data_offsets[1] - entry.data_offsets[0];
    let mut bytes = vec![0u8; byte_len];
    file.seek(SeekFrom::Start(data_start + entry.data_offsets[0] as u64))
        .map_err(LLMError::IoError)?;
    file.read_exact(&mut bytes).map_err(LLMError::IoError)?;
    Ok(Some((entry, bytes)))
}

fn ensure_shape_dtype(tensor: &LoadedTensor, dtype: &str, shape: &[usize]) -> Result<()> {
    if tensor.source.dtype != dtype || tensor.source.shape != shape {
        return Err(LLMError::ModelError(format!(
            "{} expected dtype {dtype} shape {:?}, got dtype {} shape {:?}",
            tensor.source.tensor_name, shape, tensor.source.dtype, tensor.source.shape
        )));
    }
    Ok(())
}

fn bf16_bytes_to_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return Err(LLMError::ModelError(format!(
            "{name} BF16 byte length is not even"
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect())
}

fn slice_bias(values: &[f32], expert: usize, len: usize) -> Vec<f32> {
    let start = expert * len;
    values[start..start + len].to_vec()
}
