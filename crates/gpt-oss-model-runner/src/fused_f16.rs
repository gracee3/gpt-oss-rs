//! CUDA-free sizing helpers for fused f16 runner allocations.
//!
//! These helpers keep fused QKV, fused gate/up, and f16 scratch sizing outside
//! `GpuModelRunner` so future shard-local allocation can reuse the same shape
//! rules without constructing a runner.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use half::f16;

#[cfg(feature = "cuda")]
use crate::bridge::{LLMError, Result as BridgeResult};

/// Per-buffer element counts for the reusable f16 layer scratch set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16ScratchElementCounts {
    pub qkv: usize,
    pub attn_out: usize,
    pub o_proj: usize,
    pub normed: usize,
    pub residual: usize,
    pub gate_up: usize,
    pub silu_out: usize,
    pub down: usize,
}

pub fn fused_qkv_dim(q_dim: usize, kv_dim: usize) -> usize {
    q_dim + kv_dim + kv_dim
}

pub fn fused_qkv_shape(q_dim: usize, kv_dim: usize, hidden: usize) -> [usize; 2] {
    [fused_qkv_dim(q_dim, kv_dim), hidden]
}

pub fn fused_qkv_num_elements(q_dim: usize, kv_dim: usize, hidden: usize) -> usize {
    fused_qkv_dim(q_dim, kv_dim) * hidden
}

pub fn fused_gate_up_dim(intermediate: usize) -> usize {
    intermediate * 2
}

pub fn fused_gate_up_shape(intermediate: usize, hidden: usize) -> [usize; 2] {
    [fused_gate_up_dim(intermediate), hidden]
}

pub fn fused_gate_up_num_elements(intermediate: usize, hidden: usize) -> usize {
    fused_gate_up_dim(intermediate) * hidden
}

pub fn f16_scratch_element_counts(
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    max_tokens: usize,
) -> Result<F16ScratchElementCounts, String> {
    if max_tokens == 0 {
        return Err("max_tokens must be non-zero for f16 scratch sizing".into());
    }

    let qkv_dim = fused_qkv_dim(q_dim, kv_dim);
    Ok(F16ScratchElementCounts {
        qkv: max_tokens * qkv_dim,
        attn_out: max_tokens * q_dim,
        o_proj: max_tokens * hidden,
        normed: max_tokens * hidden,
        residual: max_tokens * hidden,
        gate_up: max_tokens * fused_gate_up_dim(intermediate),
        silu_out: max_tokens * intermediate,
        down: max_tokens * hidden,
    })
}

impl F16ScratchElementCounts {
    pub fn total_elements(self) -> usize {
        self.qkv
            + self.attn_out
            + self.o_proj
            + self.normed
            + self.residual
            + self.gate_up
            + self.silu_out
            + self.down
    }

    pub fn total_bytes(self) -> usize {
        self.total_elements() * std::mem::size_of::<half::f16>()
    }
}

#[cfg(feature = "cuda")]
pub fn cast_f32_tensor_to_f16(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    element_count: usize,
    kernel: &CudaFunction,
) -> BridgeResult<CudaSlice<f16>> {
    // Safety: cast kernel writes all element_count elements.
    let mut output = unsafe { stream.alloc::<f16>(element_count) }
        .map_err(|e| LLMError::GpuError(format!("cast_f32_f16 alloc: {e}")))?;
    let threads = 256u32;
    let blocks = ((element_count as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(&mut output)
            .arg(input)
            .arg(&(element_count as i32))
            .launch(cfg)
            .map_err(|e| LLMError::GpuError(format!("cast_f32_f16 launch: {e}")))?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_f16_qkv_num_elements_matches_concat_formula() {
        let q_dim = 64;
        let kv_dim = 16;
        let hidden = 128;

        assert_eq!(
            fused_qkv_num_elements(q_dim, kv_dim, hidden),
            (q_dim + kv_dim + kv_dim) * hidden
        );
        assert_eq!(fused_qkv_shape(q_dim, kv_dim, hidden), [96, 128]);
    }

    #[test]
    fn fused_f16_gate_up_num_elements_matches_concat_formula() {
        let intermediate = 256;
        let hidden = 128;

        assert_eq!(
            fused_gate_up_num_elements(intermediate, hidden),
            (intermediate * 2) * hidden
        );
        assert_eq!(fused_gate_up_shape(intermediate, hidden), [512, 128]);
    }

    #[test]
    fn f16_scratch_element_counts_match_eight_buffer_formula() {
        let hidden = 128;
        let q_dim = 64;
        let kv_dim = 16;
        let intermediate = 256;
        let max_tokens = 3;
        let qkv_dim = q_dim + kv_dim + kv_dim;

        let counts =
            f16_scratch_element_counts(hidden, q_dim, kv_dim, intermediate, max_tokens).unwrap();

        assert_eq!(counts.qkv, max_tokens * qkv_dim);
        assert_eq!(counts.attn_out, max_tokens * q_dim);
        assert_eq!(counts.o_proj, max_tokens * hidden);
        assert_eq!(counts.normed, max_tokens * hidden);
        assert_eq!(counts.residual, max_tokens * hidden);
        assert_eq!(counts.gate_up, max_tokens * intermediate * 2);
        assert_eq!(counts.silu_out, max_tokens * intermediate);
        assert_eq!(counts.down, max_tokens * hidden);
        assert_eq!(
            counts.total_elements(),
            max_tokens * (qkv_dim + q_dim + hidden * 4 + intermediate * 3)
        );
    }

    #[test]
    fn f16_scratch_total_bytes_uses_f16_element_size() {
        let counts = f16_scratch_element_counts(4, 3, 2, 5, 7).unwrap();

        assert_eq!(
            counts.total_bytes(),
            counts.total_elements() * std::mem::size_of::<half::f16>()
        );
    }

    #[test]
    fn f16_scratch_sizing_rejects_zero_max_tokens() {
        let err = f16_scratch_element_counts(4, 3, 2, 5, 0).unwrap_err();

        assert!(err.contains("max_tokens"));
    }
}
