use std::collections::HashMap;
use std::fmt;

use crate::ir::{FusedKernel, FusionGraph, FusionOp};

// ---------------------------------------------------------------------------
// Placeholder types -- will be replaced by cudarc/half imports when integrated
// ---------------------------------------------------------------------------

/// Placeholder for cudarc::driver::CudaSlice<half::f16>.
pub struct GpuBuffer {
    pub len: usize,
    pub ptr: u64,
}

/// Placeholder for cudarc::driver::CudaStream.
pub struct GpuStream;

/// Error type for fusion dispatch.
#[derive(Debug)]
pub enum FusionError {
    KernelNotFound(String),
    ShapeMismatch { expected: (usize, usize), got: (usize, usize) },
    LaunchFailed(String),
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionError::KernelNotFound(name) => write!(f, "fused kernel not found: {name}"),
            FusionError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            FusionError::LaunchFailed(msg) => write!(f, "kernel launch failed: {msg}"),
        }
    }
}

impl std::error::Error for FusionError {}

pub type Result<T> = std::result::Result<T, FusionError>;

// ---------------------------------------------------------------------------
// CompiledKernel
// ---------------------------------------------------------------------------

pub struct CompiledKernel {
    pub name: String,
    pub ptx: Vec<u8>,
    pub func_name: String,
    pub shared_mem_bytes: usize,
    pub block_dim: (u32, u32, u32),
    /// (M, N) -> (grid_x, grid_y, grid_z)
    pub grid_fn: Box<dyn Fn(usize, usize) -> (u32, u32, u32) + Send + Sync>,
}

impl fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("name", &self.name)
            .field("func_name", &self.func_name)
            .field("ptx_len", &self.ptx.len())
            .field("shared_mem_bytes", &self.shared_mem_bytes)
            .field("block_dim", &self.block_dim)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Model shape descriptor (minimal trait-free config for compilation)
// ---------------------------------------------------------------------------

/// Shapes extracted from a model config, enough to pre-compile all fused kernels.
#[derive(Debug, Clone)]
pub struct ModelShapes {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
}

// ---------------------------------------------------------------------------
// FusedLayerExecutor
// ---------------------------------------------------------------------------

/// Replaces individual kernel launches with fused kernel launches when available.
///
/// At model load time, `compile_for_model` inspects the model dimensions and
/// pre-compiles PTX for every fusible pattern. During inference, the `try_fused_*`
/// methods check if a matching compiled kernel exists and dispatch it directly;
/// returning `None` lets the caller fall back to unfused individual kernels.
pub struct FusedLayerExecutor {
    fused_kernels: HashMap<String, CompiledKernel>,
    enabled: bool,
}

impl fmt::Debug for FusedLayerExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedLayerExecutor")
            .field("enabled", &self.enabled)
            .field("num_kernels", &self.fused_kernels.len())
            .field("kernels", &self.fused_kernels.keys().collect::<Vec<_>>())
            .finish()
    }
}

// -- Pattern name constants --------------------------------------------------

const PATTERN_NORM_GEMV: &str = "rmsnorm_gemv";
const PATTERN_SILU_MUL_GEMV: &str = "silu_mul_gemv";
const PATTERN_ADD_NORM_GEMV: &str = "add_rmsnorm_gemv";

impl FusedLayerExecutor {
    /// Create a disabled executor (no fused kernels).
    pub fn disabled() -> Self {
        Self {
            fused_kernels: HashMap::new(),
            enabled: false,
        }
    }

    /// Create an executor with pre-compiled kernels.
    pub fn new(kernels: HashMap<String, CompiledKernel>) -> Self {
        Self {
            fused_kernels: kernels,
            enabled: true,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn num_kernels(&self) -> usize {
        self.fused_kernels.len()
    }

    pub fn has_kernel(&self, pattern: &str) -> bool {
        self.fused_kernels.contains_key(pattern)
    }

    /// Insert or replace a compiled kernel by pattern name.
    pub fn register_kernel(&mut self, kernel: CompiledKernel) {
        self.fused_kernels.insert(kernel.name.clone(), kernel);
    }

    // -----------------------------------------------------------------------
    // Pre-compilation from model config
    // -----------------------------------------------------------------------

    /// Pre-compile all fused kernels for a given model's shapes at startup.
    ///
    /// Accepts anything that implements the `ModelConfig` trait from rvllm-core.
    /// We extract the dimensions we need and generate PTX for each fusion pattern
    /// that applies to this architecture.
    pub fn compile_for_model(shapes: &ModelShapes) -> Self {
        let mut kernels = HashMap::new();

        // Pattern 1: RMSNorm + Gemv (attention pre-norm -> Q/K/V projection)
        // Applies to: hidden_size -> hidden_size (Q), hidden_size -> kv_dim (K, V)
        let norm_gemv_attn = compile_norm_gemv(
            shapes.hidden_size,
            shapes.hidden_size,
            shapes.rms_norm_eps,
        );
        kernels.insert(norm_gemv_attn.name.clone(), norm_gemv_attn);

        let kv_dim = shapes.num_kv_heads * shapes.head_dim;
        if kv_dim != shapes.hidden_size {
            let norm_gemv_kv = compile_norm_gemv(
                shapes.hidden_size,
                kv_dim,
                shapes.rms_norm_eps,
            );
            kernels.insert(norm_gemv_kv.name.clone(), norm_gemv_kv);
        }

        // Pattern 2: SiLU * Mul + Gemv (MLP activation -> down projection)
        let silu_gemv = compile_silu_mul_gemv(
            shapes.intermediate_size,
            shapes.hidden_size,
        );
        kernels.insert(silu_gemv.name.clone(), silu_gemv);

        // Pattern 3: Add + RMSNorm + Gemv (residual add -> post-attn norm -> MLP gate/up)
        let add_norm_gemv = compile_add_norm_gemv(
            shapes.hidden_size,
            shapes.intermediate_size,
            shapes.rms_norm_eps,
        );
        kernels.insert(add_norm_gemv.name.clone(), add_norm_gemv);

        // Also compile for hidden->hidden (post-MLP residual -> next layer pre-attn norm -> Q)
        let add_norm_gemv_attn = compile_add_norm_gemv(
            shapes.hidden_size,
            shapes.hidden_size,
            shapes.rms_norm_eps,
        );
        kernels.insert(add_norm_gemv_attn.name.clone(), add_norm_gemv_attn);

        Self {
            fused_kernels: kernels,
            enabled: true,
        }
    }

    // -----------------------------------------------------------------------
    // Dispatch methods
    // -----------------------------------------------------------------------

    /// Try to run fused RMSNorm + Gemv.
    ///
    /// Fuses: norm(input, weight, eps) -> matmul(normed, proj_weight)
    /// Returns `None` if fusion is disabled or no kernel exists for these shapes.
    pub fn try_fused_norm_gemv(
        &self,
        stream: &GpuStream,
        input: &GpuBuffer,        // [num_tokens, hidden_size]
        norm_weight: &GpuBuffer,   // [hidden_size]
        proj_weight: &GpuBuffer,   // [out_features, hidden_size]
        eps: f32,
        num_tokens: usize,
        hidden_size: usize,
        out_features: usize,
    ) -> Option<Result<GpuBuffer>> {
        if !self.enabled {
            return None;
        }

        let key = norm_gemv_key(hidden_size, out_features);
        let kernel = self.fused_kernels.get(&key)?;

        Some(launch_fused_norm_gemv(
            stream, kernel, input, norm_weight, proj_weight,
            eps, num_tokens, hidden_size, out_features,
        ))
    }

    /// Try to run fused SiLU * Mul + Gemv.
    ///
    /// Fuses: silu(gate) * up -> matmul(activated, down_weight)
    /// Eliminates the intermediate activation tensor from HBM.
    /// Returns `None` if fusion is disabled or no kernel exists for these shapes.
    pub fn try_fused_silu_gemv(
        &self,
        stream: &GpuStream,
        gate: &GpuBuffer,          // [num_tokens, intermediate_size]
        up: &GpuBuffer,            // [num_tokens, intermediate_size]
        down_weight: &GpuBuffer,   // [hidden_size, intermediate_size]
        num_tokens: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Option<Result<GpuBuffer>> {
        if !self.enabled {
            return None;
        }

        let key = silu_mul_gemv_key(intermediate_size, hidden_size);
        let kernel = self.fused_kernels.get(&key)?;

        Some(launch_fused_silu_mul_gemv(
            stream, kernel, gate, up, down_weight,
            num_tokens, intermediate_size, hidden_size,
        ))
    }

    /// Try to run fused Add + RMSNorm + Gemv.
    ///
    /// Fuses: residual = input + add; normed = rmsnorm(residual); out = matmul(normed, weight)
    /// Returns both the residual (for next layer's add) and the projection output.
    /// Returns `None` if fusion is disabled or no kernel exists for these shapes.
    pub fn try_fused_add_norm_gemv(
        &self,
        stream: &GpuStream,
        input: &GpuBuffer,         // [num_tokens, hidden_size]
        add: &GpuBuffer,           // [num_tokens, hidden_size]
        norm_weight: &GpuBuffer,   // [hidden_size]
        proj_weight: &GpuBuffer,   // [out_features, hidden_size]
        eps: f32,
        num_tokens: usize,
        hidden_size: usize,
        out_features: usize,
    ) -> Option<Result<(GpuBuffer, GpuBuffer)>> {
        if !self.enabled {
            return None;
        }

        let key = add_norm_gemv_key(hidden_size, out_features);
        let kernel = self.fused_kernels.get(&key)?;

        Some(launch_fused_add_norm_gemv(
            stream, kernel, input, add, norm_weight, proj_weight,
            eps, num_tokens, hidden_size, out_features,
        ))
    }
}

// ---------------------------------------------------------------------------
// Key generation -- deterministic pattern names incorporating shapes
// ---------------------------------------------------------------------------

fn norm_gemv_key(hidden: usize, out: usize) -> String {
    format!("{PATTERN_NORM_GEMV}_{hidden}x{out}")
}

fn silu_mul_gemv_key(intermediate: usize, hidden: usize) -> String {
    format!("{PATTERN_SILU_MUL_GEMV}_{intermediate}x{hidden}")
}

fn add_norm_gemv_key(hidden: usize, out: usize) -> String {
    format!("{PATTERN_ADD_NORM_GEMV}_{hidden}x{out}")
}

// ---------------------------------------------------------------------------
// Kernel compilation stubs
// ---------------------------------------------------------------------------
// These generate CompiledKernel instances with the correct launch configs.
// The actual PTX emission is delegated to the codegen module (currently empty);
// for now we store an empty ptx vec that will be filled when codegen lands.

fn compile_norm_gemv(hidden: usize, out: usize, _eps: f32) -> CompiledKernel {
    let block_x = hidden.min(1024) as u32;
    // Shared mem: reduction buffer for RMSNorm (one float per thread)
    let shared = block_x as usize * 4;

    CompiledKernel {
        name: norm_gemv_key(hidden, out),
        ptx: Vec::new(), // filled by codegen
        func_name: format!("fused_rmsnorm_gemv_{hidden}x{out}"),
        shared_mem_bytes: shared,
        block_dim: (block_x, 1, 1),
        grid_fn: Box::new(move |m, _n| {
            // One block per token (row of the output)
            (m as u32, 1, 1)
        }),
    }
}

fn compile_silu_mul_gemv(intermediate: usize, hidden: usize) -> CompiledKernel {
    let block_x = intermediate.min(1024) as u32;

    CompiledKernel {
        name: silu_mul_gemv_key(intermediate, hidden),
        ptx: Vec::new(),
        func_name: format!("fused_silu_mul_gemv_{intermediate}x{hidden}"),
        shared_mem_bytes: 0, // elementwise SiLU*mul needs no shared mem; Gemv accumulates in registers
        block_dim: (block_x, 1, 1),
        grid_fn: Box::new(move |m, _n| {
            (m as u32, 1, 1)
        }),
    }
}

fn compile_add_norm_gemv(hidden: usize, out: usize, _eps: f32) -> CompiledKernel {
    let block_x = hidden.min(1024) as u32;
    // Shared mem: reduction buffer for RMSNorm
    let shared = block_x as usize * 4;

    CompiledKernel {
        name: add_norm_gemv_key(hidden, out),
        ptx: Vec::new(),
        func_name: format!("fused_add_rmsnorm_gemv_{hidden}x{out}"),
        shared_mem_bytes: shared,
        block_dim: (block_x, 1, 1),
        grid_fn: Box::new(move |m, _n| {
            (m as u32, 1, 1)
        }),
    }
}

// ---------------------------------------------------------------------------
// Launch stubs -- skeleton for actual CUDA dispatch
// ---------------------------------------------------------------------------
// These will call cudarc launch_builder when integrated. For now they validate
// shapes and return placeholder buffers so the calling code can be written and
// tested against the API without a GPU.

fn launch_fused_norm_gemv(
    _stream: &GpuStream,
    kernel: &CompiledKernel,
    _input: &GpuBuffer,
    _norm_weight: &GpuBuffer,
    _proj_weight: &GpuBuffer,
    _eps: f32,
    num_tokens: usize,
    _hidden_size: usize,
    out_features: usize,
) -> Result<GpuBuffer> {
    if kernel.ptx.is_empty() {
        return Err(FusionError::LaunchFailed(
            format!("kernel {} has no compiled PTX", kernel.name),
        ));
    }

    let _grid = (kernel.grid_fn)(num_tokens, out_features);

    // Actual launch would go here:
    //   stream.launch_builder(&func)
    //     .arg(output).arg(input).arg(norm_weight).arg(proj_weight)
    //     .arg(&eps).arg(&(hidden_size as i32)).arg(&(out_features as i32))
    //     .launch(config)

    Ok(GpuBuffer {
        len: num_tokens * out_features,
        ptr: 0,
    })
}

fn launch_fused_silu_mul_gemv(
    _stream: &GpuStream,
    kernel: &CompiledKernel,
    _gate: &GpuBuffer,
    _up: &GpuBuffer,
    _down_weight: &GpuBuffer,
    num_tokens: usize,
    _intermediate_size: usize,
    hidden_size: usize,
) -> Result<GpuBuffer> {
    if kernel.ptx.is_empty() {
        return Err(FusionError::LaunchFailed(
            format!("kernel {} has no compiled PTX", kernel.name),
        ));
    }

    let _grid = (kernel.grid_fn)(num_tokens, hidden_size);

    Ok(GpuBuffer {
        len: num_tokens * hidden_size,
        ptr: 0,
    })
}

fn launch_fused_add_norm_gemv(
    _stream: &GpuStream,
    kernel: &CompiledKernel,
    _input: &GpuBuffer,
    _add: &GpuBuffer,
    _norm_weight: &GpuBuffer,
    _proj_weight: &GpuBuffer,
    _eps: f32,
    num_tokens: usize,
    hidden_size: usize,
    out_features: usize,
) -> Result<(GpuBuffer, GpuBuffer)> {
    if kernel.ptx.is_empty() {
        return Err(FusionError::LaunchFailed(
            format!("kernel {} has no compiled PTX", kernel.name),
        ));
    }

    let _grid = (kernel.grid_fn)(num_tokens, out_features);

    // Two outputs: residual (for next layer) and projection result
    let residual = GpuBuffer {
        len: num_tokens * hidden_size,
        ptr: 0,
    };
    let output = GpuBuffer {
        len: num_tokens * out_features,
        ptr: 0,
    };
    Ok((output, residual))
}

// ---------------------------------------------------------------------------
// Builder from FusionGraph IR (for dynamic / graph-driven compilation)
// ---------------------------------------------------------------------------

impl FusedLayerExecutor {
    /// Build an executor from an analyzed FusionGraph.
    ///
    /// Inspects each `FusedKernel` chain returned by the graph's fusion analysis
    /// and compiles a CompiledKernel for known patterns. Unknown patterns are
    /// silently skipped (the caller will use unfused fallback).
    pub fn from_graph(graph: &FusionGraph, eps: f32) -> Self {
        let chains = graph.find_fusible_chains();
        let mut kernels = HashMap::new();

        for chain in &chains {
            if let Some(kernel) = try_compile_chain(chain, eps) {
                kernels.insert(kernel.name.clone(), kernel);
            }
        }

        Self {
            fused_kernels: kernels,
            enabled: true,
        }
    }
}

/// Match a FusedKernel chain against known patterns and compile if recognized.
fn try_compile_chain(chain: &FusedKernel, _eps: f32) -> Option<CompiledKernel> {
    let ops = &chain.ops;

    // Pattern: [RMSNorm, Gemv]
    if ops.len() == 2 {
        if let (FusionOp::RMSNorm { eps: e }, FusionOp::Gemv) = (&ops[0], &ops[1]) {
            let out = chain.output_shape.last().copied().unwrap_or(0);
            // Infer hidden from the Gemv input (= RMSNorm output)
            // For a 2-node chain, hidden == out if it's a square projection,
            // otherwise we'd need the full shape info. Use output_shape[1] as best guess.
            let hidden = out; // conservative: caller can override via compile_for_model
            return Some(compile_norm_gemv(hidden, out, *e));
        }
    }

    // Pattern: [SiLU, ElemMul, Gemv]
    if ops.len() == 3 {
        if matches!((&ops[0], &ops[1], &ops[2]), (FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv)) {
            let out = chain.output_shape.last().copied().unwrap_or(0);
            let intermediate = out; // placeholder
            return Some(compile_silu_mul_gemv(intermediate, out));
        }
    }

    // Pattern: [ElemAdd, RMSNorm, Gemv]
    if ops.len() == 3 {
        if let (FusionOp::ElemAdd, FusionOp::RMSNorm { eps: e }, FusionOp::Gemv) =
            (&ops[0], &ops[1], &ops[2])
        {
            let out = chain.output_shape.last().copied().unwrap_or(0);
            let hidden = out;
            return Some(compile_add_norm_gemv(hidden, out, *e));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Dtype;

    fn llama_7b_shapes() -> ModelShapes {
        ModelShapes {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
        }
    }

    fn llama_70b_shapes() -> ModelShapes {
        ModelShapes {
            hidden_size: 8192,
            intermediate_size: 28672,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
        }
    }

    #[test]
    fn disabled_executor_returns_none() {
        let exec = FusedLayerExecutor::disabled();
        assert!(!exec.is_enabled());
        assert_eq!(exec.num_kernels(), 0);

        let stream = GpuStream;
        let buf = GpuBuffer { len: 4096, ptr: 0 };
        let result = exec.try_fused_norm_gemv(
            &stream, &buf, &buf, &buf, 1e-5, 1, 4096, 4096,
        );
        assert!(result.is_none());
    }

    #[test]
    fn compile_for_llama_7b() {
        let exec = FusedLayerExecutor::compile_for_model(&llama_7b_shapes());
        assert!(exec.is_enabled());

        // Should have: norm_gemv for hidden->hidden, silu_mul_gemv, add_norm_gemv x2
        assert!(exec.has_kernel("rmsnorm_gemv_4096x4096"));
        assert!(exec.has_kernel("silu_mul_gemv_11008x4096"));
        assert!(exec.has_kernel("add_rmsnorm_gemv_4096x11008"));
        assert!(exec.has_kernel("add_rmsnorm_gemv_4096x4096"));
    }

    #[test]
    fn compile_for_llama_70b_has_gqa_kernels() {
        let exec = FusedLayerExecutor::compile_for_model(&llama_70b_shapes());

        // GQA: kv_dim = 8 * 128 = 1024, different from hidden_size 8192
        assert!(exec.has_kernel("rmsnorm_gemv_8192x8192"));
        assert!(exec.has_kernel("rmsnorm_gemv_8192x1024"));
        assert!(exec.has_kernel("silu_mul_gemv_28672x8192"));
    }

    #[test]
    fn try_fused_returns_none_for_missing_shape() {
        let exec = FusedLayerExecutor::compile_for_model(&llama_7b_shapes());
        let stream = GpuStream;
        let buf = GpuBuffer { len: 1, ptr: 0 };

        // 4096x999 was never compiled
        let result = exec.try_fused_norm_gemv(
            &stream, &buf, &buf, &buf, 1e-5, 1, 4096, 999,
        );
        assert!(result.is_none());
    }

    #[test]
    fn try_fused_returns_err_when_no_ptx() {
        let exec = FusedLayerExecutor::compile_for_model(&llama_7b_shapes());
        let stream = GpuStream;
        let buf = GpuBuffer { len: 4096, ptr: 0 };

        // Kernel exists but PTX is empty (codegen not implemented yet)
        let result = exec.try_fused_norm_gemv(
            &stream, &buf, &buf, &buf, 1e-5, 1, 4096, 4096,
        );
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn register_and_toggle() {
        let mut exec = FusedLayerExecutor::disabled();
        assert_eq!(exec.num_kernels(), 0);

        exec.register_kernel(compile_norm_gemv(4096, 4096, 1e-5));
        assert_eq!(exec.num_kernels(), 1);
        assert!(exec.has_kernel("rmsnorm_gemv_4096x4096"));

        // Still disabled, so dispatch returns None
        let stream = GpuStream;
        let buf = GpuBuffer { len: 1, ptr: 0 };
        assert!(exec.try_fused_norm_gemv(&stream, &buf, &buf, &buf, 1e-5, 1, 4096, 4096).is_none());

        exec.set_enabled(true);
        assert!(exec.try_fused_norm_gemv(&stream, &buf, &buf, &buf, 1e-5, 1, 4096, 4096).is_some());
    }

    #[test]
    fn grid_fn_produces_correct_dims() {
        let kernel = compile_norm_gemv(4096, 4096, 1e-5);
        assert_eq!((kernel.grid_fn)(1, 4096), (1, 1, 1));
        assert_eq!((kernel.grid_fn)(32, 4096), (32, 1, 1));
        assert_eq!((kernel.grid_fn)(128, 4096), (128, 1, 1));
        assert_eq!(kernel.block_dim, (1024, 1, 1));
        assert_eq!(kernel.shared_mem_bytes, 1024 * 4);
    }

    #[test]
    fn from_graph_recognizes_norm_gemv() {
        let mut g = FusionGraph::new();
        let norm = g.add_node(FusionOp::RMSNorm { eps: 1e-5 }, vec![], vec![1, 4096], Dtype::F16);
        let _gemv = g.add_node(FusionOp::Gemv, vec![norm], vec![1, 4096], Dtype::F16);

        let exec = FusedLayerExecutor::from_graph(&g, 1e-5);
        assert!(exec.is_enabled());
        assert!(exec.has_kernel("rmsnorm_gemv_4096x4096"));
    }

    #[test]
    fn debug_format() {
        let exec = FusedLayerExecutor::compile_for_model(&llama_7b_shapes());
        let dbg = format!("{exec:?}");
        assert!(dbg.contains("FusedLayerExecutor"));
        assert!(dbg.contains("enabled: true"));
    }
}
