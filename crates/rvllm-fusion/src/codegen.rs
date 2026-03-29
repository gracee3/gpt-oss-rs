//! CUDA C code generation for fused kernel patterns.
//!
//! Takes a `FusedKernel` from the fusion IR and emits compilable CUDA C source
//! with `extern "C"` linkage. The generated kernels use f16 I/O, f32
//! accumulation, half2 vectorized loads, and warp-shuffle reductions.

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as FmtWrite;
use std::hash::{Hash, Hasher};
use std::io::Write;

use crate::ir::{FusedKernel, FusionOp};

const THREADS: u32 = 256;
const WARPS: u32 = THREADS / 32;

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusionPattern {
    RMSNormGemv,
    SiLUElemMulGemv,
    ElemAddRMSNorm,
    ElemAddRMSNormGemv,
    Generic,
}

fn classify(kernel: &FusedKernel) -> FusionPattern {
    let ops: Vec<_> = kernel.ops.iter().map(op_tag).collect();
    let s: Vec<&str> = ops.iter().map(|s| s.as_str()).collect();
    match s.as_slice() {
        ["RMSNorm", "Gemv"] => FusionPattern::RMSNormGemv,
        ["SiLU", "ElemMul", "Gemv"] => FusionPattern::SiLUElemMulGemv,
        ["ElemAdd", "RMSNorm"] => FusionPattern::ElemAddRMSNorm,
        ["ElemAdd", "RMSNorm", "Gemv"] => FusionPattern::ElemAddRMSNormGemv,
        _ => FusionPattern::Generic,
    }
}

fn op_tag(op: &FusionOp) -> String {
    match op {
        FusionOp::RMSNorm { .. } => "RMSNorm".into(),
        FusionOp::Gemv => "Gemv".into(),
        FusionOp::SiLU => "SiLU".into(),
        FusionOp::ElemMul => "ElemMul".into(),
        FusionOp::ElemAdd => "ElemAdd".into(),
        FusionOp::BiasAdd => "BiasAdd".into(),
        FusionOp::RoPE => "RoPE".into(),
        FusionOp::Softmax => "Softmax".into(),
        FusionOp::Copy => "Copy".into(),
    }
}

/// Compute a stable name for the generated kernel based on the op sequence.
fn kernel_name(kernel: &FusedKernel) -> String {
    let mut h = DefaultHasher::new();
    for op in &kernel.ops {
        op_tag(op).hash(&mut h);
    }
    let hash = h.finish();
    let tag: String = kernel
        .ops
        .iter()
        .map(|op| match op {
            FusionOp::RMSNorm { .. } => "rn",
            FusionOp::SiLU => "si",
            FusionOp::ElemMul => "em",
            FusionOp::ElemAdd => "ea",
            FusionOp::Gemv => "gv",
            FusionOp::BiasAdd => "ba",
            FusionOp::RoPE => "ro",
            FusionOp::Softmax => "sm",
            FusionOp::Copy => "cp",
        })
        .collect::<Vec<_>>()
        .join("_");
    format!("fused_{hash:016x}_{tag}")
}

// ---------------------------------------------------------------------------
// Common CUDA snippets
// ---------------------------------------------------------------------------

fn emit_header(out: &mut String) {
    out.push_str("#include <cuda_fp16.h>\n\n");
}

fn emit_warp_reduce(out: &mut String, suffix: &str) {
    let _ = write!(
        out,
        "\
__device__ __forceinline__ float warp_reduce_sum_{suffix}(float val) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {{
        val += __shfl_down_sync(0xffffffff, val, offset);
    }}
    return val;
}}

"
    );
}

fn emit_silu_device(out: &mut String) {
    out.push_str(
        "\
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

",
    );
}

// ---------------------------------------------------------------------------
// Pattern-specific generators
// ---------------------------------------------------------------------------

/// RMSNorm -> Gemv
///
/// One kernel: each block loads hidden into smem, computes RMSNorm in-place,
/// then each block computes one output element's dot product with weight row.
///
/// Signature:
///   output[out_dim], hidden[hidden_size], norm_weight[hidden_size],
///   proj_weight[out_dim, hidden_size], eps, out_dim, hidden_size
fn emit_rmsnorm_gemv(out: &mut String, name: &str) {
    let _ = write!(
        out,
        "\
extern \"C\"
__global__ void __launch_bounds__({THREADS})
{name}(
    __half* __restrict__ output,
    const __half* __restrict__ hidden,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    float eps,
    int out_dim,
    int hidden_size
) {{
    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = {THREADS} / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_warp = smem + hidden_size;

    // Phase 1: load hidden into smem, compute sum-of-squares
    float local_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += {THREADS}) {{
        float val = __half2float(hidden[i]);
        s_normed[i] = val;
        local_ss += val * val;
    }}

    local_ss = warp_reduce_sum_gen(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {{
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_gen(val);
        if (lane_id == 0) s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
    }}
    __syncthreads();

    float rms_scale = s_warp[0];

    // Apply norm: s_normed[i] = hidden[i] * norm_weight[i] * rms_scale
    for (int i = tid; i < hidden_size; i += {THREADS}) {{
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }}
    __syncthreads();

    // Phase 2: GEMV dot product with half2 vectorized weight loads
    const __half* w_row = proj_weight + (long long)row * hidden_size;
    float acc = 0.0f;

    const int h2 = hidden_size / 2;
    const half2* w2 = (const half2*)w_row;
    for (int i = tid; i < h2; i += {THREADS}) {{
        half2 w = w2[i];
        int base = i * 2;
        acc += __half2float(w.x) * s_normed[base];
        acc += __half2float(w.y) * s_normed[base + 1];
    }}
    if ((hidden_size & 1) && tid == 0) {{
        int last = hidden_size - 1;
        acc += __half2float(w_row[last]) * s_normed[last];
    }}

    // Warp reduction -> output
    acc = warp_reduce_sum_gen(acc);
    if (lane_id == 0) s_warp[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {{
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_gen(val);
        if (lane_id == 0) output[row] = __float2half(val);
    }}
}}

"
    );
}

/// SiLU -> ElemMul -> Gemv
///
/// One kernel: each block computes silu(gate[i])*up[i]*weight[row,i] on the fly.
///
/// Signature:
///   output[out_dim], gate[intermediate_size], up[intermediate_size],
///   weight[out_dim, intermediate_size], out_dim, intermediate_size
fn emit_silu_elemmul_gemv(out: &mut String, name: &str) {
    let _ = write!(
        out,
        "\
extern \"C\"
__global__ void __launch_bounds__({THREADS})
{name}(
    __half* __restrict__ output,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const __half* __restrict__ weight,
    int out_dim,
    int intermediate_size
) {{
    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = {THREADS} / 32;

    const __half* w_row = weight + (long long)row * intermediate_size;
    float acc = 0.0f;

    // Vectorized half2 loads
    const int k2 = intermediate_size / 2;
    const half2* gate2 = (const half2*)gate;
    const half2* up2 = (const half2*)up;
    const half2* w2 = (const half2*)w_row;

    for (int i = tid; i < k2; i += {THREADS}) {{
        half2 g = gate2[i];
        half2 u = up2[i];
        half2 w = w2[i];

        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);
        float w0 = __half2float(w.x);
        float w1 = __half2float(w.y);

        acc += silu_f32(g0) * u0 * w0;
        acc += silu_f32(g1) * u1 * w1;
    }}

    // Handle odd size
    if ((intermediate_size & 1) && tid == 0) {{
        int last = intermediate_size - 1;
        float g = __half2float(gate[last]);
        acc += silu_f32(g) * __half2float(up[last]) * __half2float(w_row[last]);
    }}

    // Warp reduction
    acc = warp_reduce_sum_gen(acc);

    __shared__ float warp_sums[{WARPS}];
    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {{
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum_gen(val);
        if (lane_id == 0) output[row] = __float2half(val);
    }}
}}

"
    );
}

/// ElemAdd -> RMSNorm
///
/// One kernel: adds two vectors and normalizes. Grid = (num_tokens, 1, 1).
///
/// Signature:
///   output[num_tokens, hidden_size], residual[num_tokens, hidden_size],
///   input[num_tokens, hidden_size], add[num_tokens, hidden_size],
///   norm_weight[hidden_size], eps, hidden_size
fn emit_elemadd_rmsnorm(out: &mut String, name: &str) {
    let _ = write!(
        out,
        "\
extern \"C\"
__global__ void __launch_bounds__({THREADS})
{name}(
    __half* __restrict__ output,
    __half* __restrict__ residual,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    float eps,
    int hidden_size
) {{
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = {THREADS} / 32;
    const int row_off = token * hidden_size;

    extern __shared__ float smem[];
    float* s_data = smem;
    float* s_warp = smem + hidden_size;

    // Phase 1: residual add + sum-of-squares (half2 vectorized loads)
    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)(input + row_off);
    const half2* add2 = (const half2*)(add_vec + row_off);
    half2* res2 = (half2*)(residual + row_off);

    for (int i = tid; i < h2; i += {THREADS}) {{
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        res2[i] = __halves2half2(__float2half(v0), __float2half(v1));
        s_data[i * 2] = v0;
        s_data[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }}
    // Handle odd hidden_size
    if ((hidden_size & 1) && tid == 0) {{
        int last = hidden_size - 1;
        float v = __half2float(input[row_off + last]) + __half2float(add_vec[row_off + last]);
        residual[row_off + last] = __float2half(v);
        s_data[last] = v;
        local_ss += v * v;
    }}

    // Warp reduction of sum-of-squares
    local_ss = warp_reduce_sum_gen(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {{
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_gen(val);
        if (lane_id == 0) s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
    }}
    __syncthreads();

    float rms_scale = s_warp[0];

    // Phase 2: normalize and write output (half2 vectorized stores)
    const half2* nw2 = (const half2*)norm_weight;
    half2* out2 = (half2*)(output + row_off);

    for (int i = tid; i < h2; i += {THREADS}) {{
        half2 nw = nw2[i];
        float n0 = s_data[i * 2] * __half2float(nw.x) * rms_scale;
        float n1 = s_data[i * 2 + 1] * __half2float(nw.y) * rms_scale;
        out2[i] = __halves2half2(__float2half(n0), __float2half(n1));
    }}
    if ((hidden_size & 1) && tid == 0) {{
        int last = hidden_size - 1;
        float val = s_data[last] * __half2float(norm_weight[last]) * rms_scale;
        output[row_off + last] = __float2half(val);
    }}
}}

"
    );
}

/// ElemAdd -> RMSNorm -> Gemv (three-way fusion)
///
/// Phase 1: residual add + RMSNorm into smem (redundant per block, but
///          hidden_size is small and stays in L1/smem).
/// Phase 2: GEMV dot product of normed vector against proj_weight row.
///
/// Grid = (out_dim, 1, 1). Each block redundantly normalizes, then dots.
///
/// Signature:
///   output[out_dim], residual_out[hidden_size],
///   input[hidden_size], add_vec[hidden_size],
///   norm_weight[hidden_size], proj_weight[out_dim, hidden_size],
///   eps, out_dim, hidden_size
fn emit_elemadd_rmsnorm_gemv(out: &mut String, name: &str) {
    let _ = write!(
        out,
        "\
extern \"C\"
__global__ void __launch_bounds__({THREADS})
{name}(
    __half* __restrict__ output,
    __half* __restrict__ residual_out,
    const __half* __restrict__ input,
    const __half* __restrict__ add_vec,
    const __half* __restrict__ norm_weight,
    const __half* __restrict__ proj_weight,
    float eps,
    int out_dim,
    int hidden_size
) {{
    const int row = blockIdx.x;
    if (row >= out_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = {THREADS} / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_warp = smem + hidden_size;

    // Phase 1: residual add -> smem, compute sum-of-squares
    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += {THREADS}) {{
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }}
    if ((hidden_size & 1) && tid == 0) {{
        int last = hidden_size - 1;
        float v = __half2float(input[last]) + __half2float(add_vec[last]);
        s_normed[last] = v;
        local_ss += v * v;
    }}

    local_ss = warp_reduce_sum_gen(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {{
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_gen(val);
        if (lane_id == 0) s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
    }}
    __syncthreads();

    float rms_scale = s_warp[0];

    // Write residual out (only first block to avoid races -- block 0)
    if (row == 0) {{
        for (int i = tid; i < hidden_size; i += {THREADS}) {{
            residual_out[i] = __float2half(s_normed[i]);
        }}
    }}

    // Apply norm weights in-place in smem
    for (int i = tid; i < hidden_size; i += {THREADS}) {{
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }}
    __syncthreads();

    // Phase 2: GEMV dot product
    const __half* w_row = proj_weight + (long long)row * hidden_size;
    float acc = 0.0f;

    const half2* w2 = (const half2*)w_row;
    for (int i = tid; i < h2; i += {THREADS}) {{
        half2 w = w2[i];
        int base = i * 2;
        acc += __half2float(w.x) * s_normed[base];
        acc += __half2float(w.y) * s_normed[base + 1];
    }}
    if ((hidden_size & 1) && tid == 0) {{
        int last = hidden_size - 1;
        acc += __half2float(w_row[last]) * s_normed[last];
    }}

    acc = warp_reduce_sum_gen(acc);
    if (lane_id == 0) s_warp[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {{
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_gen(val);
        if (lane_id == 0) output[row] = __float2half(val);
    }}
}}

"
    );
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate compilable CUDA C source for a fused kernel pattern.
///
/// The returned string includes `#include <cuda_fp16.h>`, device helper
/// functions, and an `extern "C"` kernel with a unique name derived from
/// the pattern hash.
///
/// Returns `None` if the pattern is not yet supported by codegen.
pub fn generate_cuda_source(pattern: &FusedKernel) -> Option<String> {
    let pat = classify(pattern);
    if pat == FusionPattern::Generic {
        return None;
    }

    let name = kernel_name(pattern);
    let mut src = String::with_capacity(4096);

    // Header
    emit_header(&mut src);

    // Device helpers
    emit_warp_reduce(&mut src, "gen");

    match pat {
        FusionPattern::RMSNormGemv => {
            emit_rmsnorm_gemv(&mut src, &name);
        }
        FusionPattern::SiLUElemMulGemv => {
            emit_silu_device(&mut src);
            emit_silu_elemmul_gemv(&mut src, &name);
        }
        FusionPattern::ElemAddRMSNorm => {
            emit_elemadd_rmsnorm(&mut src, &name);
        }
        FusionPattern::ElemAddRMSNormGemv => {
            emit_elemadd_rmsnorm_gemv(&mut src, &name);
        }
        FusionPattern::Generic => unreachable!(),
    }

    Some(src)
}

/// Compile CUDA C source to PTX bytes by shelling out to nvcc.
///
/// `arch` is an sm target like `"sm_80"`. Uses a temp file pair
/// (`.cu` input, `.ptx` output) and cleans up on success or failure.
pub fn compile_to_ptx(cuda_source: &str, arch: &str) -> Result<Vec<u8>, String> {
    use std::process::Command;

    let dir = std::env::temp_dir();
    let id = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = format!("rvllm_fused_{id}_{ts}");
    let cu_path = dir.join(format!("{stem}.cu"));
    let ptx_path = dir.join(format!("{stem}.ptx"));

    // Write source
    {
        let mut f = std::fs::File::create(&cu_path)
            .map_err(|e| format!("failed to create temp .cu: {e}"))?;
        f.write_all(cuda_source.as_bytes())
            .map_err(|e| format!("failed to write temp .cu: {e}"))?;
    }

    let result = Command::new("nvcc")
        .args([
            "--ptx",
            "-arch",
            arch,
            "-O3",
            "--use_fast_math",
            "-o",
            ptx_path.to_str().unwrap(),
            cu_path.to_str().unwrap(),
        ])
        .output();

    // Cleanup source regardless of outcome
    let _ = std::fs::remove_file(&cu_path);

    let output = result.map_err(|e| format!("nvcc exec failed: {e}"))?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&ptx_path);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nvcc compilation failed:\n{stderr}"));
    }

    let ptx_bytes =
        std::fs::read(&ptx_path).map_err(|e| format!("failed to read ptx output: {e}"))?;
    let _ = std::fs::remove_file(&ptx_path);

    Ok(ptx_bytes)
}

/// Convenience: get the kernel function name that will appear in the generated
/// PTX, so the caller can look it up via cuModuleGetFunction.
pub fn kernel_function_name(pattern: &FusedKernel) -> String {
    kernel_name(pattern)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Dtype, FusedKernel, FusionOp};

    fn make_kernel(ops: Vec<FusionOp>) -> FusedKernel {
        FusedKernel {
            node_ids: (0..ops.len()).collect(),
            ops,
            output_shape: vec![1, 4096],
            dtype: Dtype::F16,
        }
    }

    #[test]
    fn test_rmsnorm_gemv_generates() {
        let k = make_kernel(vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv]);
        let src = generate_cuda_source(&k).expect("should generate");
        assert!(src.contains("extern \"C\""));
        assert!(src.contains("cuda_fp16.h"));
        assert!(src.contains("__half"));
        assert!(src.contains("warp_reduce_sum_gen"));
        assert!(src.contains("half2"));
        assert!(src.contains("rsqrtf"));
        let name = kernel_function_name(&k);
        assert!(src.contains(&name));
    }

    #[test]
    fn test_silu_elemmul_gemv_generates() {
        let k = make_kernel(vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv]);
        let src = generate_cuda_source(&k).expect("should generate");
        assert!(src.contains("silu_f32"));
        assert!(src.contains("extern \"C\""));
        assert!(src.contains("half2"));
        let name = kernel_function_name(&k);
        assert!(src.contains(&name));
    }

    #[test]
    fn test_elemadd_rmsnorm_generates() {
        let k = make_kernel(vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-6 }]);
        let src = generate_cuda_source(&k).expect("should generate");
        assert!(src.contains("residual"));
        assert!(src.contains("rsqrtf"));
        assert!(src.contains("extern \"C\""));
    }

    #[test]
    fn test_elemadd_rmsnorm_gemv_generates() {
        let k = make_kernel(vec![
            FusionOp::ElemAdd,
            FusionOp::RMSNorm { eps: 1e-5 },
            FusionOp::Gemv,
        ]);
        let src = generate_cuda_source(&k).expect("should generate");
        assert!(src.contains("proj_weight"));
        assert!(src.contains("residual_out"));
        assert!(src.contains("extern \"C\""));
    }

    #[test]
    fn test_unsupported_returns_none() {
        let k = make_kernel(vec![FusionOp::Softmax]);
        assert!(generate_cuda_source(&k).is_none());
    }

    #[test]
    fn test_kernel_names_unique() {
        let k1 = make_kernel(vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv]);
        let k2 = make_kernel(vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv]);
        assert_ne!(kernel_function_name(&k1), kernel_function_name(&k2));
    }

    #[test]
    fn test_kernel_names_stable() {
        let k1 = make_kernel(vec![FusionOp::RMSNorm { eps: 1e-5 }, FusionOp::Gemv]);
        let k2 = make_kernel(vec![FusionOp::RMSNorm { eps: 1e-6 }, FusionOp::Gemv]);
        // Same op sequence, different eps -> same name (eps doesn't affect hash)
        assert_eq!(kernel_function_name(&k1), kernel_function_name(&k2));
    }
}
