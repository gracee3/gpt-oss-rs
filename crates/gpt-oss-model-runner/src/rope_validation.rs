//! Validation-safe accessors for the runtime RoPE table and kernel path.
//!
//! This module exists so bench validation can reuse the same table construction
//! and kernel symbol as the CUDA runtime without routing production execution
//! through a validation policy.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use half::{bf16, f16};

use crate::bridge::{LLMError, Result};
use crate::runner::ModelRunnerConfig;
use gpt_oss_gpu::kernel_loader::KernelLoader;

/// Build the runtime RoPE cos/sin tables.
///
/// Keep this helper aligned with the runtime table source used by
/// `GpuModelRunner`; callers should not duplicate this math in bench code.
pub fn build_runtime_rope_tables(
    head_dim: usize,
    max_position: usize,
    rope_theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_position * half_dim];
    let mut sin_table = vec![0.0f32; max_position * half_dim];
    for pos in 0..max_position {
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
            let theta = pos as f32 * freq;
            cos_table[pos * half_dim + i] = theta.cos();
            sin_table[pos * half_dim + i] = theta.sin();
        }
    }
    (cos_table, sin_table)
}

/// Build validation RoPE tables from the full runner config.
///
/// This mirrors the proven YaRN-aware table source used by the runtime-forward
/// validation branch while remaining validation-only in this branch. Production
/// callers are not routed through this helper.
pub fn build_validation_rope_tables_from_config(
    config: &ModelRunnerConfig,
    max_position: usize,
) -> (Vec<f32>, Vec<f32>, &'static str) {
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

    let mut cos_table = vec![0.0f32; max_position * half_dim];
    let mut sin_table = vec![0.0f32; max_position * half_dim];
    for pos in 0..max_position {
        for (i, freq) in inv_freq.iter().enumerate() {
            let theta = pos as f32 * freq;
            cos_table[pos * half_dim + i] = theta.cos() * concentration;
            sin_table[pos * half_dim + i] = theta.sin() * concentration;
        }
    }
    let source = if use_yarn {
        "yarn_scaled"
    } else {
        "plain_rope_theta"
    };
    (cos_table, sin_table, source)
}

/// Apply the runtime f16 RoPE kernel to a supplied K tensor.
///
/// Input and output layout is `[token, kv_head, lane]`, flattened row-major.
/// A dummy Q buffer is launched alongside K because the runtime kernel rotates
/// Q and K in one entry point.
pub fn apply_k_rope_f16_validation(
    k_pre_rope: &[f32],
    token_count: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) -> Result<Vec<f32>> {
    let max_position = token_count;
    let (cos_table, sin_table) = build_runtime_rope_tables(head_dim, max_position, rope_theta);
    apply_k_rope_f16_validation_with_tables(
        k_pre_rope,
        token_count,
        kv_heads,
        head_dim,
        &cos_table,
        &sin_table,
    )
}

/// Apply the runtime f16 RoPE kernel using validation tables from config.
pub fn apply_k_rope_f16_validation_with_config(
    k_pre_rope: &[f32],
    token_count: usize,
    kv_heads: usize,
    config: &ModelRunnerConfig,
) -> Result<(Vec<f32>, &'static str)> {
    let max_position = token_count;
    let (cos_table, sin_table, table_source) =
        build_validation_rope_tables_from_config(config, max_position);
    let mut output = apply_k_rope_f16_validation_with_tables(
        k_pre_rope,
        token_count,
        kv_heads,
        config.head_dim,
        &cos_table,
        &sin_table,
    )?;
    for value in &mut output {
        *value = bf16::from_f32(*value).to_f32();
    }
    Ok((output, table_source))
}

/// Apply the official/model BF16 RoPE boundary for K validation.
///
/// This is a CPU validation helper, not a production runtime route. It uses the
/// same validation RoPE tables as the CUDA guard, casts inputs and RoPE factors
/// to BF16, applies the half-split formula with BF16-rounded multiply/add
/// boundaries, and returns BF16 values widened to f32 for artifact comparison.
pub fn apply_k_rope_bf16_boundary_validation(
    k_pre_rope: &[f32],
    token_count: usize,
    kv_heads: usize,
    config: &ModelRunnerConfig,
) -> Result<(Vec<f32>, &'static str)> {
    let (cos_table, sin_table, table_source) =
        build_validation_rope_tables_from_config(config, token_count);
    let output = apply_k_rope_bf16_boundary_validation_with_tables(
        k_pre_rope,
        token_count,
        kv_heads,
        config.head_dim,
        &cos_table,
        &sin_table,
    )?;
    Ok((output, table_source))
}

fn apply_k_rope_bf16_boundary_validation_with_tables(
    k_pre_rope: &[f32],
    token_count: usize,
    kv_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Result<Vec<f32>> {
    let expected = token_count * kv_heads * head_dim;
    if k_pre_rope.len() != expected {
        return Err(LLMError::GpuError(format!(
            "K pre-RoPE value count mismatch: {} != {}",
            k_pre_rope.len(),
            expected
        )));
    }
    let half_dim = head_dim / 2;
    let mut output = vec![0.0f32; expected];
    for token in 0..token_count {
        for kv_head in 0..kv_heads {
            let head_base = (token * kv_heads + kv_head) * head_dim;
            let table_base = token * half_dim;
            for lane in 0..half_dim {
                let x1 = round_bf16(k_pre_rope[head_base + lane]);
                let x2 = round_bf16(k_pre_rope[head_base + half_dim + lane]);
                let cos = round_bf16(cos_table[table_base + lane]);
                let sin = round_bf16(sin_table[table_base + lane]);
                let out1 = round_bf16(round_bf16(x1 * cos) - round_bf16(x2 * sin));
                let out2 = round_bf16(round_bf16(x2 * cos) + round_bf16(x1 * sin));
                output[head_base + lane] = out1;
                output[head_base + half_dim + lane] = out2;
            }
        }
    }
    Ok(output)
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn apply_k_rope_f16_validation_with_tables(
    k_pre_rope: &[f32],
    token_count: usize,
    kv_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Result<Vec<f32>> {
    let expected = token_count * kv_heads * head_dim;
    if k_pre_rope.len() != expected {
        return Err(LLMError::GpuError(format!(
            "K pre-RoPE value count mismatch: {} != {}",
            k_pre_rope.len(),
            expected
        )));
    }

    let context = CudaContext::new(0)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE CUDA context: {e}")))?;
    let stream = context
        .new_stream()
        .map_err(|e| LLMError::GpuError(format!("validation RoPE CUDA stream: {e}")))?;
    let loader = KernelLoader::new(
        context.clone(),
        stream.clone(),
        gpt_oss_gpu::kernel_loader::default_ptx_dir(),
    )
    .map_err(|e| LLMError::GpuError(format!("validation RoPE kernel loader: {e}")))?;
    let rope_cos = stream
        .clone_htod(cos_table)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE cos upload: {e}")))?;
    let rope_sin = stream
        .clone_htod(sin_table)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE sin upload: {e}")))?;
    let positions = (0..token_count as i32).collect::<Vec<_>>();
    let positions_dev = stream
        .clone_htod(&positions)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE positions upload: {e}")))?;

    let dummy_q = vec![f16::ZERO; expected];
    let mut q_dev = stream
        .clone_htod(&dummy_q)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE dummy Q upload: {e}")))?;
    let k_host = k_pre_rope
        .iter()
        .map(|value| f16::from_f32(*value))
        .collect::<Vec<_>>();
    let mut k_dev = stream
        .clone_htod(&k_host)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE K upload: {e}")))?;

    launch_rotary_embedding_f16(
        &stream,
        &loader,
        &mut q_dev,
        &mut k_dev,
        &positions_dev,
        &rope_cos,
        &rope_sin,
        token_count,
        kv_heads,
        kv_heads,
        head_dim,
    )?;
    stream
        .synchronize()
        .map_err(|e| LLMError::GpuError(format!("validation RoPE synchronize: {e}")))?;

    let k_out = stream
        .clone_dtoh(&k_dev)
        .map_err(|e| LLMError::GpuError(format!("validation RoPE K download: {e}")))?;
    Ok(k_out.iter().map(|value| value.to_f32()).collect())
}

fn launch_rotary_embedding_f16(
    stream: &Arc<CudaStream>,
    loader: &KernelLoader,
    q: &mut cudarc::driver::CudaSlice<f16>,
    k: &mut cudarc::driver::CudaSlice<f16>,
    positions: &cudarc::driver::CudaSlice<i32>,
    rope_cos: &cudarc::driver::CudaSlice<f32>,
    rope_sin: &cudarc::driver::CudaSlice<f32>,
    num_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if num_tokens == 0 {
        return Ok(());
    }
    let kernel = loader.get_func("rotary_embedding_f16", "rotary_embedding_f16_kernel")?;
    let half_dim = head_dim / 2;
    let grid_y = num_heads.max(num_kv_heads) as u32;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens as u32, grid_y, 1),
        block_dim: (half_dim.min(1024) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(q)
            .arg(k)
            .arg(rope_cos)
            .arg(rope_sin)
            .arg(positions)
            .arg(&(num_tokens as i32))
            .arg(&(num_heads as i32))
            .arg(&(num_kv_heads as i32))
            .arg(&(head_dim as i32))
            .launch(cfg)
            .map_err(|e| LLMError::GpuError(format!("validation rope_f16 launch: {e}")))?;
    }
    Ok(())
}
