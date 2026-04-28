# CUDA BF16 Projection Policy Implementation Plan

Date: 2026-04-27

Branch: `projection/cuda-bf16-policy-validation`

Base: `integration/mainline-alignment`

Source proof branch: `feature/runtime-forward`

Source proof commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`

## Recommendation

Add a validation-only Q/K/V projection policy comparison harness before any CUDA runtime
routing or cuBLAS policy change.

Do not promote cuBLAS pedantic helpers or oneDNN Q/K/V candidates into production runtime
yet. The next implementation target should be a small `qkv_projection_policy_compare`
harness that compares current CUDA projection behavior, candidate CUDA projection policies,
and the preserved oneDNN/PyTorch oracle outputs for the exact layer0 shapes.

## Current Repo Projection Path Audit

| Path | Role | Dtype | Runtime-affecting? | Backend | Candidate for validation harness? |
| --- | --- | --- | --- | --- | --- |
| `crates/gpt-oss-model-runner/src/gpu_layer.rs` | Main fused f16 layer path: pre-attn norm, Q/K/V, RoPE, attention, o_proj, MLP | f16 storage with f32 accumulation through GEMM APIs | Yes | Custom kernels plus cuBLAS, cublasLt, and `gemv_f16` dispatch | Yes. This is the primary current-runtime comparison target. |
| `crates/gpt-oss-model-runner/src/gpu_runner.rs` | Weight fusion and runner orchestration; fuses Q/K/V and gate/up weights for f16 path | f16 projection weights and f16 fused QKV bias | Yes | Device copies plus model-runner dispatch into layer code | Yes for loading/slicing source tensors, not for policy changes in first harness. |
| `crates/gpt-oss-model-runner/src/layers/linear_cuda.rs` | Generic CUDA linear helpers, LM-head f16/f32 mixed helpers | f32, f16, f16-in/f32-out | Yes | `CublasOps`, `CublasHandle`, optional cublasLt | Useful reference surface for a standalone harness. |
| `crates/gpt-oss-gpu/src/cublas.rs` | Low-level cuBLAS wrapper including `hgemm`, `hgemm_into`, `hgemm_f32_output`, strided batched HGEMM | f16 and f32 | Yes | cuBLAS / `cublasGemmEx` | Yes. Candidate policies must be scoped here or in a harness-only wrapper first. |
| `crates/gpt-oss-gpu/src/cublas_ops.rs` | Higher-level safe GEMM operations used by generic linear layer | f32 and f16 | Yes | cudarc cuBLAS `Gemm` trait | Reference only for harness design unless generic linear needs comparison. |
| `crates/gpt-oss-gpu/src/cublaslt_ops.rs` | cublasLt small-`m` GEMM path | f16 and f32 | Yes when `cublaslt` enabled | cublasLt heuristic matmul | Compare as one current/candidate policy, especially for decode-sized shapes. |
| `kernels/gemv_f16.cu` | Custom `m == 1` decode GEMV path with f32 accumulation and f16 output | f16 input/output | Yes for decode path | Custom CUDA GEMV | Not primary for layer0 prefill `m=74`, but useful for decode-policy guardrails. |
| `crates/gpt-oss-engine/src/worker/gpu_worker.rs` | Older/engine worker path with f32 SGEMM projections and LM head | f32 | Yes | cuBLAS SGEMM plus CPU attention portions | Not first target for GPT-OSS f16 projection parity; useful only as legacy contrast. |
| `kernels/fused_lm_head_argmax*.cu` | Fused LM-head argmax kernels | f32/f16 variants | Yes, but not Q/K/V projection | Custom CUDA | Not a Q/K/V projection-policy target. |
| `kernels/gpt_oss_moe.cu` | GPT-OSS MoE expert kernels | f32/f16 mixed surfaces | Yes, but not Q/K/V projection | Custom CUDA | Out of scope for the projection-policy lane. |

Current integration uses `hgemm_dispatch` in the f16 layer path:

- `m == 1`: custom `gemv_f16_kernel` when available.
- `m <= CUBLASLT_M_THRESHOLD`: cublasLt when enabled.
- Otherwise: cuBLAS `hgemm` / `cublasGemmEx`-style f16 GEMM with f32 compute and f16 output.

Layer0 prefill Q/K/V shapes use `m = 74`, so they are not covered by the custom GEMV path.

## Runtime-Forward Evidence Summary

The runtime-forward proof artifacts establish the target behavior but do not define a CUDA
runtime promotion strategy by themselves.

K projection:

- oneDNN/PyTorch BF16 `linear` matched the official K projection exactly.
- The legacy helper differed on six K lanes.
- A scoped oneDNN K candidate cleared pre-RoPE K and downstream QK checks when paired with
  official/candidate Q.
- The K pedantic cuBLAS experiment showed that `raw_full_m74_pedantic_no_tensor_op` matched
  a CPU reference, while the default tensor-op path did not.

Q projection:

- oneDNN Q candidate matched official pre-RoPE Q exactly.
- Candidate Q cleared post-RoPE and raw scaled QK when paired with candidate K.
- Local runtime Q remained different from official on many lanes, confirming a projection
  policy delta rather than a weight/bias issue.

V projection:

- oneDNN V candidate matched official V projection exactly.
- Candidate V cleared weighted-V sum before `o_proj`.
- Local runtime V variants remained different from candidate/official outputs.

cuBLAS / pedantic:

- Pedantic/no-tensor-op settings helped diagnose an earlier K helper mismatch.
- Later Q/K/V proof target was established by oneDNN/PyTorch behavior.
- A cuBLAS pedantic policy remains performance-sensitive and not yet designed for production
  routing.

Final proof:

- The source branch final-readout direct-module rerun cleared final block output, final norm,
  and LM-head logits with digest
  `67f31845dd24db26cc91954607cfae8ae7ff7b9c8954cb9d3b1610ca9c635209`.

## rvllm Kernel Reference Audit

Reference inspected: `https://github.com/m0at/rvllm/tree/main/kernels`

Local reference clone commit inspected: `537c8b3bdeaf827b534491fe1a4993f7e742f2ae`

License observed: Apache-2.0.

This audit is reference-only. No rvllm code is copied or imported by this plan.

| rvllm file | Operation | Dtype | Shape mode | Backend | Bias | Output dtype | Relevance | Reuse feasibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kernels/cutlass_gemm.cu` | General half GEMM, `output = input @ weight^T` | `cutlass::half_t` | Matrix/prefill | CUTLASS 3.x SM90 tensor op | No explicit bias | Half | Relevant as a CUTLASS wrapper pattern, not an oracle policy by itself | Reference only unless CUTLASS dependency/design is approved. |
| `kernels/cutlass_qkv_bias.cu` | QKV GEMM plus per-column bias add | `cutlass::half_t` | Matrix/prefill | CUTLASS 3.x SM90 tensor op plus bias kernel | Yes | Half | Highly relevant to fused Q/K/V shape and bias handling | Possible adaptation only after license/dependency review and correctness proof. |
| `kernels/cutlass_oproj_residual.cu` | O-projection GEMM fused with residual add | `cutlass::half_t` | Matrix/prefill | CUTLASS 3.x SM90 tensor op | Residual epilogue, not projection bias | Half | Useful for later o_proj performance, not first Q/K/V parity target | Reference only for future fusion/perf work. |
| `kernels/gemv_f16.cu` | Requested file not present in inspected rvllm tree | N/A | N/A | N/A | N/A | N/A | Not available from rvllm at inspected commit | Not applicable. |
| `kernels/fused_norm_qkv_gemv.cu` | Requested file not present in inspected rvllm tree | N/A | N/A | N/A | N/A | N/A | Not available from rvllm at inspected commit | Not applicable. |
| `kernels/fused_add_norm_qkv_gemv.cu` | Requested file not present in inspected rvllm tree | N/A | N/A | N/A | N/A | N/A | Not available from rvllm at inspected commit | Not applicable. |
| `kernels/persistent_gemm.cu` | Requested file not present in inspected rvllm tree | N/A | N/A | N/A | N/A | N/A | Not available from rvllm at inspected commit | Not applicable. |
| `kernels/README.md` | Kernel inventory and PTX build notes | N/A | N/A | CUDA driver PTX loading | N/A | N/A | Confirms reference repository design style | Reference only. |

rvllm’s CUTLASS QKV-bias path is the most relevant reference for a future Q/K/V projection
candidate, but it is not sufficient to claim oneDNN BF16 parity. Tensor-op CUTLASS behavior
may reproduce current CUDA performance characteristics while still differing at oneDNN BF16
rounding boundaries unless explicitly proven.

## Candidate Strategy Matrix

| Option | Correctness likelihood vs oneDNN oracle | Performance risk | Complexity | Integration risk | Dependencies | Validation-only? | Production candidate? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Current CUDA/cuBLAS helper | Known to differ for runtime-forward Q/K/V boundary cases | Low; existing fast path | None | None | Existing cuBLAS/cublasLt/custom GEMV | Yes as baseline | Already production, but not oracle-compatible for the proof case. |
| B. Scoped cuBLAS pedantic/no-tensor-op helper | Proven helpful for K helper vs CPU reference; not yet proven for all Q/K/V oneDNN outputs | Medium to high; may disable tensor-op acceleration | Low to medium | Medium if math/atomics modes leak or scope broadens | cuBLAS only | Yes | Maybe, only after latency and exact Q/K/V oracle proof. |
| C. rvllm/CUTLASS-style projection kernel or wrapper | Unknown until compared; likely fast but not guaranteed oneDNN-equivalent | Medium; may be fast but architecture-specific | High | High due new dependency/build/runtime surface | CUTLASS, architecture-specific kernels | Yes | Possible after design review and license/dependency approval. |
| D. Custom deterministic BF16 projection kernel for validation shapes | Potentially high if reduction/order is designed to match oracle fixtures | High for production; acceptable for validation | High | Medium if kept harness-only | Custom CUDA | Yes | Only if benchmarked and generalized carefully. |
| E. Validation-only path | High for review confidence if it compares multiple policies to oracle artifacts | None for default runtime | Medium | Low when isolated | Explicit local artifacts and optional CUDA | Yes | Not production routing. It informs later options. |

## Minimal Harness Design

Name: `qkv_projection_policy_compare`

Purpose:

- Compare current CUDA helper behavior against oneDNN/PyTorch oracle outputs.
- Compare one or more candidate CUDA policies against the same oracle.
- Measure latency for baseline and candidate policies.
- Avoid changing default runtime behavior.

Inputs:

- Authoritative norm input tensor for the exact case.
- Q/K/V weight slices.
- Q/V/K bias where applicable.
- Official oneDNN/PyTorch oracle outputs.
- Explicit artifact paths supplied by CLI flags.

Required shapes:

- Q: `[74, 2880] x [4096, 2880]`
- K: `[74, 2880] x [512, 2880]`
- V: `[74, 2880] x [512, 2880]`

Outputs:

- Current CUDA helper metrics vs oracle.
- Candidate CUDA policy metrics vs oracle.
- Latency for current helper.
- Latency for candidate helper.
- Known boundary-lane table.
- Optional downstream checks:
  - Q/K post-RoPE.
  - Raw QK scores.
  - Weighted V before `o_proj`.

Requirements:

- No dependency on `runtime_forward_layer0_qkv_bf16_candidate_status.rs`.
- No `.live` raw artifact committed to the repo.
- Artifact paths are explicit and local-only.
- No oneDNN/CPU dependency in the runtime CUDA path.
- Default runtime behavior remains unchanged.
- The harness must clearly separate baseline runtime behavior from candidate validation behavior.

Suggested CLI shape:

```bash
cargo run -p gpt-oss-bench --features cuda --bin qkv_projection_policy_compare -- \
  --norm-input /path/to/layer0-attn-norm-output.json \
  --q-weight /path/to/q-weight.json \
  --k-weight /path/to/k-weight.json \
  --v-weight /path/to/v-weight.json \
  --q-oracle /path/to/q-onednn-oracle.json \
  --k-oracle /path/to/k-onednn-oracle.json \
  --v-oracle /path/to/v-onednn-oracle.json \
  --policy current,cublas-pedantic \
  --output /tmp/qkv_projection_policy_compare.json
```

The exact artifact format should be deliberately small. Prefer manifest-addressed external
artifacts or local paths over committed raw full-value tensors.

## Correctness Guardrails

- Validate Q, K, and V independently against oneDNN oracle outputs.
- Include known K six-lane boundary cases from the runtime-forward K oracle artifact.
- Include Q post-RoPE and raw QK checks when both Q and K candidate outputs match.
- Include V weighted-sum check when V candidate output matches.
- Require exact BF16 output parity for the scoped oracle fixtures before any routing change.
- Include a final-token top-20/logit digest check only if an integration-safe smoke harness is
  available.

## Performance Guardrails

- Compare latency for current vs candidate policy on Q/K/V shapes.
- Compare tensor-op default vs pedantic/no-tensor-op settings.
- Measure batch-size sensitivity beyond `m=74` before production consideration.
- Estimate cumulative layer-wide cost if a policy would route all Q/K/V layers.
- Ensure no global cuBLAS math-mode or atomics-mode leakage.
- Restore cuBLAS math and atomics modes after scoped candidate calls.
- Record workspace and GPU memory overhead.

## Integration Strategy

Commit 1:

- Add this design/audit document only.

Commit 2:

- Add `qkv_projection_policy_compare` correctness/latency harness only.
- No runtime routing.
- No proof harness import.
- No `.live` raw artifact commit.

## Harness Skeleton Status

The `qkv_projection_policy_compare` skeleton exists at
`crates/gpt-oss-bench/src/bin/qkv_projection_policy_compare.rs`.

Current status:

- Validates explicit local artifact paths for norm input, Q/K/V weights, and Q/K/V oracle
  outputs.
- Loads JSON artifact metadata and validates expected norm, Q/K/V weight, and Q/K/V oracle
  output shapes when shape metadata is present.
- Records discovered dtype, digest/checksum, and value-count metadata when available.
- Emits the planned JSON status/schema with expected Q/K/V shapes and per-artifact metadata
  status.
- Accepts comma-separated policy names such as `current` and future names like
  `current,cublas-pedantic`.
- Does not perform numerical tensor comparison yet.
- Does not execute CUDA projection comparison yet.
- Does not route or change runtime projection behavior.

Next implementation step:

- Add baseline current CUDA helper comparison against the loaded artifacts while keeping default
  runtime behavior unchanged.

Commit 3:

- Add a guarded projection-policy implementation behind an explicit validation flag.
- Keep default runtime path unchanged.

Commit 4:

- Optional runtime routing only after Q/K/V correctness and performance evidence.

## Attribution and License Notes

- rvllm was inspected as a reference only.
- The inspected rvllm checkout includes an Apache-2.0 license.
- Any future code adaptation must preserve required license notices and should be reviewed as
  a dependency/design decision, especially if adding CUTLASS build requirements.
- Do not copy rvllm code into this repository without a separate implementation prompt and
  review.

## Explicit Do-Not-Do List

- Do not promote oneDNN CPU projection candidates into the CUDA runtime.
- Do not globally force cuBLAS pedantic math or atomics modes.
- Do not import the 45k-line runtime-forward proof binary.
- Do not import selected expert readout correction.
- Do not import debug/status capture plumbing.
- Do not commit raw `.live` full-value artifacts.
- Do not start 4097-boundary work in this lane.
- Do not route production Q/K/V projections until correctness and performance guardrails pass.

## Next Bounded Step

Implement baseline current CUDA helper comparison against explicit local oracle artifacts. Keep
the harness validation-only, with no production routing and no CUDA kernel changes unless a
candidate policy is explicitly approved later.
