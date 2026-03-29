//! Pattern matching engine for kernel fusion in transformer decode/prefill layers.
//!
//! Analyzes a `LayerPattern` (the operation sequence of one transformer block)
//! and returns `FusionGroup`s describing which ops should be fused together,
//! along with resource estimates.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Op representation
// ---------------------------------------------------------------------------

/// A single operation in the transformer layer graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    RMSNorm,
    QKVGemv,      // M=1 GEMV for Q/K/V projection (fused weight)
    QKVGemm,      // M>1 GEMM (prefill)
    BiasAdd,      // QKV bias add
    RoPE,         // rotary position embedding (in-place)
    CacheWrite,   // KV cache scatter
    Attention,    // paged decode attention or prefill flash
    OGemv,        // output projection M=1
    OGemm,        // output projection M>1
    ElemAdd,      // residual add
    ElemAddRMSNorm, // fused residual add + rmsnorm (already exists)
    GateUpGemv,   // gate+up fused projection M=1
    GateUpGemm,   // gate+up fused projection M>1
    SiLU,         // silu activation
    ElemMul,      // gate * up element-wise
    DownGemv,     // down projection M=1
    DownGemm,     // down projection M>1
}

impl Op {
    pub fn is_gemv(self) -> bool {
        matches!(self, Op::QKVGemv | Op::OGemv | Op::GateUpGemv | Op::DownGemv)
    }

    pub fn is_gemm(self) -> bool {
        matches!(self, Op::QKVGemm | Op::OGemm | Op::GateUpGemm | Op::DownGemm)
    }

    pub fn is_elementwise(self) -> bool {
        matches!(
            self,
            Op::RMSNorm
                | Op::BiasAdd
                | Op::RoPE
                | Op::ElemAdd
                | Op::ElemAddRMSNorm
                | Op::SiLU
                | Op::ElemMul
        )
    }
}

// ---------------------------------------------------------------------------
// Layer pattern
// ---------------------------------------------------------------------------

/// Model-specific knobs that affect which fusions are legal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub has_qkv_bias: bool,
    /// Whether a previous layer supplies a fused residual (cross-layer fusion).
    pub cross_layer_residual: bool,
}

impl ModelConfig {
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    pub fn qkv_dim(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// The operation sequence for one transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPattern {
    pub ops: Vec<Op>,
    pub config: ModelConfig,
    /// true = prefill (M>1), false = decode (M=1).
    pub is_prefill: bool,
}

impl LayerPattern {
    /// Build the standard LLaMA-style decode layer pattern (M=1).
    pub fn standard_decode(config: ModelConfig) -> Self {
        let mut ops = Vec::with_capacity(14);

        // Pre-attention
        if config.cross_layer_residual {
            ops.push(Op::ElemAddRMSNorm);
        } else {
            ops.push(Op::RMSNorm);
        }

        // QKV
        ops.push(Op::QKVGemv);
        if config.has_qkv_bias {
            ops.push(Op::BiasAdd);
        }
        ops.push(Op::RoPE);
        ops.push(Op::CacheWrite);
        ops.push(Op::Attention);

        // O proj
        ops.push(Op::OGemv);

        // Post-attention residual + norm (always fused)
        ops.push(Op::ElemAddRMSNorm);

        // MLP
        ops.push(Op::GateUpGemv);
        ops.push(Op::SiLU);
        ops.push(Op::ElemMul);
        ops.push(Op::DownGemv);

        Self {
            ops,
            config,
            is_prefill: false,
        }
    }

    /// Build the standard LLaMA-style prefill layer pattern (M>1).
    pub fn standard_prefill(config: ModelConfig) -> Self {
        let mut ops = Vec::with_capacity(14);

        if config.cross_layer_residual {
            ops.push(Op::ElemAddRMSNorm);
        } else {
            ops.push(Op::RMSNorm);
        }

        ops.push(Op::QKVGemm);
        if config.has_qkv_bias {
            ops.push(Op::BiasAdd);
        }
        ops.push(Op::RoPE);
        ops.push(Op::CacheWrite);
        ops.push(Op::Attention);
        ops.push(Op::OGemm);
        ops.push(Op::ElemAddRMSNorm);
        ops.push(Op::GateUpGemm);
        ops.push(Op::SiLU);
        ops.push(Op::ElemMul);
        ops.push(Op::DownGemm);

        Self {
            ops,
            config,
            is_prefill: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion group
// ---------------------------------------------------------------------------

/// Describes one group of ops to fuse into a single kernel launch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionGroup {
    /// Ops being fused, in execution order.
    pub ops: Vec<Op>,
    /// Human-readable name for the fused kernel.
    pub name: String,
    /// Tensor inputs consumed by the fused kernel.
    pub inputs: Vec<TensorRef>,
    /// Tensor outputs produced by the fused kernel.
    pub outputs: Vec<TensorRef>,
    /// Estimated wall-clock saving in microseconds.
    /// Kernel launch overhead ~5us each; memory round-trip savings on top.
    pub estimated_speedup_us: f64,
    /// Whether this fusion requires shared memory, and how many bytes.
    pub shared_mem_bytes: u32,
    /// Priority: higher = fuse first when groups conflict.
    pub priority: u32,
}

/// A named tensor in the layer graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRef {
    pub name: String,
    /// Element count (not bytes). Multiply by dtype size for bytes.
    pub numel: usize,
}

// ---------------------------------------------------------------------------
// Kernel launch overhead constant
// ---------------------------------------------------------------------------

/// Measured kernel launch overhead on modern NVIDIA GPUs (us).
const LAUNCH_OVERHEAD_US: f64 = 5.0;
/// Cost of one full hidden-state memory round-trip at ~2TB/s HBM (us).
/// For hidden=4096, f16: 4096*2 bytes / 2e12 B/s * 1e6 = ~0.004us per token.
/// Negligible for M=1, but adds up for prefill.
fn mem_roundtrip_us(bytes: usize, _is_prefill: bool) -> f64 {
    // Assume ~2 TB/s HBM bandwidth (A100/H100 class)
    (bytes as f64) / 2e6
}

// ---------------------------------------------------------------------------
// Fusion finder
// ---------------------------------------------------------------------------

/// Analyze a layer pattern and return all legal fusion groups, ordered by priority.
pub fn find_fusions(pattern: &LayerPattern) -> Vec<FusionGroup> {
    let mut groups = Vec::new();
    let cfg = &pattern.config;

    if pattern.is_prefill {
        find_prefill_fusions(pattern, cfg, &mut groups);
    } else {
        find_decode_fusions(pattern, cfg, &mut groups);
    }

    // Sort by priority descending so the caller applies highest-value fusions first.
    groups.sort_by(|a, b| b.priority.cmp(&a.priority));
    groups
}

// ---------------------------------------------------------------------------
// Decode (M=1) fusions
// ---------------------------------------------------------------------------

fn find_decode_fusions(pattern: &LayerPattern, cfg: &ModelConfig, groups: &mut Vec<FusionGroup>) {
    let ops = &pattern.ops;
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;

    // --- Group 1: RMSNorm + QKV_Gemv ---
    // Only when there's no cross-layer residual providing the norm already fused.
    if !cfg.cross_layer_residual && contains_seq(ops, &[Op::RMSNorm, Op::QKVGemv]) {
        // Shared memory: reduction buffer for rmsnorm = hidden * sizeof(f32)
        let smem = (hidden.min(1024) * 4) as u32;
        // Hidden state read once instead of twice (norm output not materialized).
        let mem_saved = hidden * 2; // f16 hidden state
        groups.push(FusionGroup {
            ops: vec![Op::RMSNorm, Op::QKVGemv],
            name: "fused_rmsnorm_qkv_gemv".into(),
            inputs: vec![
                TensorRef { name: "hidden".into(), numel: hidden },
                TensorRef { name: "norm_weight".into(), numel: hidden },
                TensorRef { name: "qkv_weight".into(), numel: cfg.qkv_dim() * hidden },
            ],
            outputs: vec![
                TensorRef { name: "qkv".into(), numel: cfg.qkv_dim() },
            ],
            estimated_speedup_us: LAUNCH_OVERHEAD_US + mem_roundtrip_us(mem_saved, false),
            shared_mem_bytes: smem,
            priority: 20,
        });
    }

    // --- Group 2: ElemAdd + RMSNorm (post-attention, already implemented) ---
    if contains_op(ops, Op::ElemAddRMSNorm) {
        let smem = (hidden.min(1024) * 4) as u32;
        let mem_saved = hidden * 2; // one fewer hidden-state pass
        groups.push(FusionGroup {
            ops: vec![Op::ElemAdd, Op::RMSNorm],
            name: "fused_residual_rmsnorm".into(),
            inputs: vec![
                TensorRef { name: "residual".into(), numel: hidden },
                TensorRef { name: "attn_proj".into(), numel: hidden },
                TensorRef { name: "post_norm_weight".into(), numel: hidden },
            ],
            outputs: vec![
                TensorRef { name: "normed".into(), numel: hidden },
                TensorRef { name: "new_residual".into(), numel: hidden },
            ],
            estimated_speedup_us: LAUNCH_OVERHEAD_US + mem_roundtrip_us(mem_saved, false),
            shared_mem_bytes: smem,
            priority: 30,
        });
    }

    // --- Group 3: ElemAdd + RMSNorm + GateUp_Gemv (three-way) ---
    // The post-attention norm output feeds directly into gate+up GEMV.
    // Fusing all three eliminates the normed intermediate buffer entirely.
    if contains_op(ops, Op::ElemAddRMSNorm) && contains_op(ops, Op::GateUpGemv) {
        let smem = (hidden.min(1024) * 4) as u32;
        // Save: normed intermediate (hidden f16) not written/read + 1 extra launch
        let mem_saved = hidden * 2 * 2; // write + read of normed
        groups.push(FusionGroup {
            ops: vec![Op::ElemAdd, Op::RMSNorm, Op::GateUpGemv],
            name: "fused_residual_rmsnorm_gateup_gemv".into(),
            inputs: vec![
                TensorRef { name: "residual".into(), numel: hidden },
                TensorRef { name: "attn_proj".into(), numel: hidden },
                TensorRef { name: "post_norm_weight".into(), numel: hidden },
                TensorRef {
                    name: "gate_up_weight".into(),
                    numel: intermediate * 2 * hidden,
                },
            ],
            outputs: vec![
                TensorRef { name: "gate_up".into(), numel: intermediate * 2 },
                TensorRef { name: "new_residual".into(), numel: hidden },
            ],
            estimated_speedup_us: 2.0 * LAUNCH_OVERHEAD_US + mem_roundtrip_us(mem_saved, false),
            shared_mem_bytes: smem,
            priority: 40,
        });
    }

    // --- Group 4: SiLU + ElemMul + Down_Gemv (activation + projection) ---
    if contains_seq(ops, &[Op::SiLU, Op::ElemMul]) && contains_op(ops, Op::DownGemv) {
        // No shared memory needed: silu and mul are pure elementwise, GEMV uses registers.
        // Eliminates the activated intermediate buffer.
        let mem_saved = intermediate * 2; // f16 silu output not materialized
        groups.push(FusionGroup {
            ops: vec![Op::SiLU, Op::ElemMul, Op::DownGemv],
            name: "fused_silu_mul_down_gemv".into(),
            inputs: vec![
                TensorRef { name: "gate".into(), numel: intermediate },
                TensorRef { name: "up".into(), numel: intermediate },
                TensorRef { name: "down_weight".into(), numel: hidden * intermediate },
            ],
            outputs: vec![
                TensorRef { name: "mlp_out".into(), numel: hidden },
            ],
            // 2 launches saved (silu+mul collapsed, then fused with gemv)
            estimated_speedup_us: 2.0 * LAUNCH_OVERHEAD_US + mem_roundtrip_us(mem_saved, false),
            shared_mem_bytes: 0,
            priority: 35,
        });
    }

    // --- Bonus: BiasAdd + RoPE (if model has QKV bias) ---
    if cfg.has_qkv_bias && contains_seq(ops, &[Op::BiasAdd, Op::RoPE]) {
        groups.push(FusionGroup {
            ops: vec![Op::BiasAdd, Op::RoPE],
            name: "fused_bias_rope".into(),
            inputs: vec![
                TensorRef { name: "qkv".into(), numel: cfg.qkv_dim() },
                TensorRef { name: "qkv_bias".into(), numel: cfg.qkv_dim() },
                TensorRef { name: "cos".into(), numel: cfg.head_dim / 2 },
                TensorRef { name: "sin".into(), numel: cfg.head_dim / 2 },
                TensorRef { name: "positions".into(), numel: 1 },
            ],
            outputs: vec![
                TensorRef { name: "qkv_roped".into(), numel: cfg.qkv_dim() },
            ],
            estimated_speedup_us: LAUNCH_OVERHEAD_US,
            shared_mem_bytes: 0,
            priority: 10,
        });
    }
}

// ---------------------------------------------------------------------------
// Prefill (M>1) fusions -- only elementwise fusions, no GEMV
// ---------------------------------------------------------------------------

fn find_prefill_fusions(
    pattern: &LayerPattern,
    cfg: &ModelConfig,
    groups: &mut Vec<FusionGroup>,
) {
    let ops = &pattern.ops;
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;

    // ElemAdd + RMSNorm (same kernel as decode, just launched with more blocks)
    if contains_op(ops, Op::ElemAddRMSNorm) {
        let smem = (hidden.min(1024) * 4) as u32;
        groups.push(FusionGroup {
            ops: vec![Op::ElemAdd, Op::RMSNorm],
            name: "fused_residual_rmsnorm".into(),
            inputs: vec![
                TensorRef { name: "residual".into(), numel: hidden },
                TensorRef { name: "attn_proj".into(), numel: hidden },
                TensorRef { name: "post_norm_weight".into(), numel: hidden },
            ],
            outputs: vec![
                TensorRef { name: "normed".into(), numel: hidden },
                TensorRef { name: "new_residual".into(), numel: hidden },
            ],
            estimated_speedup_us: LAUNCH_OVERHEAD_US,
            shared_mem_bytes: smem,
            priority: 30,
        });
    }

    // SiLU + ElemMul (no GEMV fusion for prefill -- cuBLAS GEMM is faster)
    if contains_seq(ops, &[Op::SiLU, Op::ElemMul]) {
        groups.push(FusionGroup {
            ops: vec![Op::SiLU, Op::ElemMul],
            name: "fused_silu_mul".into(),
            inputs: vec![
                TensorRef { name: "gate".into(), numel: intermediate },
                TensorRef { name: "up".into(), numel: intermediate },
            ],
            outputs: vec![
                TensorRef { name: "activated".into(), numel: intermediate },
            ],
            estimated_speedup_us: LAUNCH_OVERHEAD_US,
            shared_mem_bytes: 0,
            priority: 20,
        });
    }

    // BiasAdd + RoPE (elementwise, works at any M)
    if cfg.has_qkv_bias && contains_seq(ops, &[Op::BiasAdd, Op::RoPE]) {
        groups.push(FusionGroup {
            ops: vec![Op::BiasAdd, Op::RoPE],
            name: "fused_bias_rope".into(),
            inputs: vec![
                TensorRef { name: "qkv".into(), numel: cfg.qkv_dim() },
                TensorRef { name: "qkv_bias".into(), numel: cfg.qkv_dim() },
                TensorRef { name: "cos".into(), numel: cfg.head_dim / 2 },
                TensorRef { name: "sin".into(), numel: cfg.head_dim / 2 },
            ],
            outputs: vec![
                TensorRef { name: "qkv_roped".into(), numel: cfg.qkv_dim() },
            ],
            estimated_speedup_us: LAUNCH_OVERHEAD_US,
            shared_mem_bytes: 0,
            priority: 10,
        });
    }
}

// ---------------------------------------------------------------------------
// Pattern helpers
// ---------------------------------------------------------------------------

/// Check that `needle` appears as a contiguous subsequence in `haystack`.
fn contains_seq(haystack: &[Op], needle: &[Op]) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Check that a single op is present anywhere.
fn contains_op(haystack: &[Op], op: Op) -> bool {
    haystack.contains(&op)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_7b_config(cross_layer: bool) -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            has_qkv_bias: false,
            cross_layer_residual: cross_layer,
        }
    }

    fn qwen_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            has_qkv_bias: true,
            cross_layer_residual: false,
        }
    }

    fn llama_gqa_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            has_qkv_bias: false,
            cross_layer_residual: false,
        }
    }

    #[test]
    fn standard_decode_no_cross_layer() {
        let cfg = llama_7b_config(false);
        let pat = LayerPattern::standard_decode(cfg);
        let fusions = find_fusions(&pat);

        // Should have: rmsnorm+qkv, residual+norm, residual+norm+gateup, silu+mul+down
        assert!(fusions.len() >= 4, "got {} fusions", fusions.len());

        let names: Vec<&str> = fusions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"fused_rmsnorm_qkv_gemv"));
        assert!(names.contains(&"fused_residual_rmsnorm"));
        assert!(names.contains(&"fused_residual_rmsnorm_gateup_gemv"));
        assert!(names.contains(&"fused_silu_mul_down_gemv"));
    }

    #[test]
    fn standard_decode_with_cross_layer() {
        let cfg = llama_7b_config(true);
        let pat = LayerPattern::standard_decode(cfg);
        let fusions = find_fusions(&pat);

        // Cross-layer: no standalone rmsnorm+qkv because the first op is ElemAddRMSNorm
        let names: Vec<&str> = fusions.iter().map(|f| f.name.as_str()).collect();
        assert!(!names.contains(&"fused_rmsnorm_qkv_gemv"));
        // But we still get the three-way
        assert!(names.contains(&"fused_residual_rmsnorm_gateup_gemv"));
    }

    #[test]
    fn prefill_no_gemv_fusions() {
        let cfg = llama_7b_config(false);
        let pat = LayerPattern::standard_prefill(cfg);
        let fusions = find_fusions(&pat);

        // Prefill should NOT have any GEMV fusions
        for f in &fusions {
            for op in &f.ops {
                assert!(!op.is_gemv(), "prefill should not fuse GEMV ops: {:?}", f.name);
            }
        }
        // But should still have elementwise fusions
        let names: Vec<&str> = fusions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"fused_residual_rmsnorm"));
        assert!(names.contains(&"fused_silu_mul"));
    }

    #[test]
    fn qkv_bias_model_gets_bias_rope_fusion() {
        let cfg = qwen_config();
        let pat = LayerPattern::standard_decode(cfg);
        let fusions = find_fusions(&pat);

        let names: Vec<&str> = fusions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"fused_bias_rope"));
    }

    #[test]
    fn no_bias_model_skips_bias_rope() {
        let cfg = llama_7b_config(false);
        let pat = LayerPattern::standard_decode(cfg);
        let fusions = find_fusions(&pat);

        let names: Vec<&str> = fusions.iter().map(|f| f.name.as_str()).collect();
        assert!(!names.contains(&"fused_bias_rope"));
    }

    #[test]
    fn gqa_model_config() {
        let cfg = llama_gqa_config();
        assert_eq!(cfg.gqa_ratio(), 4);
        assert_eq!(cfg.qkv_dim(), 32 * 128 + 2 * 8 * 128); // 4096 + 2048 = 6144

        let pat = LayerPattern::standard_decode(cfg);
        let fusions = find_fusions(&pat);
        assert!(!fusions.is_empty());
    }

    #[test]
    fn fusions_sorted_by_priority_desc() {
        let cfg = llama_7b_config(false);
        let pat = LayerPattern::standard_decode(cfg);
        let fusions = find_fusions(&pat);

        for w in fusions.windows(2) {
            assert!(w[0].priority >= w[1].priority);
        }
    }

    #[test]
    fn all_speedups_positive() {
        let cfg = llama_7b_config(false);
        for is_prefill in [false, true] {
            let pat = if is_prefill {
                LayerPattern::standard_prefill(cfg.clone())
            } else {
                LayerPattern::standard_decode(cfg.clone())
            };
            for f in find_fusions(&pat) {
                assert!(f.estimated_speedup_us > 0.0, "{} has non-positive speedup", f.name);
            }
        }
    }
}
