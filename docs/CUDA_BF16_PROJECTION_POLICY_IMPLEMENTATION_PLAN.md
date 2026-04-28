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

## Harness CUDA Dependency/API Decision

Blocked classification observed:
`qkv_projection_policy_compare_k_current_blocked_by_helper_api_gap`.

Reason:

- `gpt-oss-model-runner::gpu_layer::hgemm_dispatch` is private layer internals and should not
  be used by a validation harness.
- `gpt-oss-engine::GpuWorker` can execute model paths, but it does not expose a standalone
  artifact-tensor GEMM comparison surface.
- The public current CUDA GEMM surface is in `gpt-oss-gpu::cublas::CublasHandle`.

Chosen option: `qkv_projection_dependency_plan_ready_direct_gpu_dep`.

Dependency/API boundary:

- `gpt-oss-bench` has an optional direct `gpt-oss-gpu` dependency enabled only by the existing
  `cuda` feature.
- The default non-CUDA bench build remains unchanged.
- Validation binaries can call public `gpt-oss-gpu` APIs such as `CublasHandle::hgemm` or
  `CublasHandle::hgemm_into`.
- No new `gpt-oss-gpu` runtime API is introduced for this decision.
- No private `gpu_layer` helpers are exposed or used.
- Default runtime behavior remains unchanged.

Next implementation slice expected files:

- `crates/gpt-oss-bench/src/bin/qkv_projection_policy_compare.rs`
- `docs/CUDA_BF16_PROJECTION_POLICY_IMPLEMENTATION_PLAN.md`

The next slice should implement K-only current CUDA helper comparison using the public
`gpt-oss-gpu::cublas::CublasHandle` API. It should not add a candidate/pedantic policy yet.

## K Current CUDA Baseline Status

`qkv_projection_policy_compare` now has an explicit validation-only execution path for
`--execute --projection k --policy current`.

Current behavior:

- Loads norm input, K weight, and K oracle JSON values only for explicit K execution.
- Runs `K_out [74, 512] = norm_input [74, 2880] x K_weight [512, 2880]^T` through the public
  `gpt-oss-gpu::cublas::CublasHandle::hgemm` path.
- Emits max/mean absolute error, exact mismatch count, first/worst mismatch metadata, a compact
  mismatch table, a local output checksum, and a small latency sample.
- Leaves metadata-only mode as the default when `--execute` is absent.
- Does not add a candidate policy, pedantic/no-tensor-op path, runtime routing, CUDA kernel change,
  or proof-harness dependency.

The first K current CUDA baseline produced a broad mismatch against the selected K oracle, much
larger than the six-lane runtime-forward K projection delta. Candidate policies should wait until
the baseline contract is calibrated.

The harness now emits K contract reconciliation diagnostics:

- Explicit activation/weight/oracle shapes, dtype conversions, hgemm transposition, leading
  dimensions, bias policy, and output/oracle layout.
- CPU f32, f16-rounded, BF16-rounded, and transposed-flat sanity replays against the same oracle.
- CUDA current output comparisons against CPU f32/f16/BF16 replays.
- Runtime-forward reference metrics showing the known six-lane helper-vs-oneDNN K delta and the
  scoped proof-only candidate result when local artifacts are available.

Next step depends on the K result:

- If current K matches the oracle, extend the same baseline to Q/V.
- If current K mismatches broadly, reconcile dtype/layout/artifact identity before considering any
  candidate policy.
- If the mismatch is isolated to a calibrated dtype policy difference, design a K-only candidate
  comparison path before considering any runtime projection policy.

## K BF16 CUDA Baseline Blocker

Requested BF16 CUDA baseline:
`--execute --projection k --policy current --storage-dtype bf16`.

Current finding: `qkv_projection_policy_compare_k_bf16_cuda_blocked_by_public_api_gap`.

Public API audit:

- `gpt-oss-gpu::cublas::CublasHandle` currently exposes f32/f16 GEMM helpers such as `sgemm`,
  `hgemm`, `hgemm_into`, and `hgemm_f32_output`.
- No public `CublasHandle` method currently accepts `half::bf16` device buffers.
- No public `gpt-oss-gpu` GEMM wrapper currently uses `CUDA_R_16BF` input/output with
  `CUBLAS_COMPUTE_32F`.
- The bench harness should not add raw cuBLAS calls or expose private model-runner projection
  plumbing to bypass that API boundary.

Decision:

- Do not implement the BF16 CUDA baseline in this slice.
- Do not add a pedantic/no-tensor-op candidate policy yet.
- Treat BF16 CUDA execution as a separate, reviewable `gpt-oss-gpu` API decision.

Required next API shape before implementation:

- A narrow public validation-oriented BF16 GEMM helper in `gpt-oss-gpu`, or a clearly production-
  intended BF16 GEMM API if broader runtime use is approved.
- Inputs: BF16 activation and BF16 weight buffers in the existing row-major projection contract.
- Output: BF16 buffer, downloaded as f32 by the harness for comparison.
- Compute: FP32 accumulation with explicit cuBLAS data types and algorithm policy recorded in the
  status artifact.
- No default runtime routing change and no global cuBLAS math/atomics-mode change.

The CPU BF16 replay remains the calibrated local contract check for now: it reproduces the
runtime-forward six-lane K legacy/helper-vs-oneDNN oracle pattern. Candidate policies should wait
until the BF16 CUDA baseline API exists and is measured against that CPU BF16 replay.

## BF16 GEMM Public API Status

Prior blocker classification:
`qkv_projection_policy_compare_k_bf16_cuda_blocked_by_public_api_gap`.

API status: `CublasHandle::bf16_gemm_into` has been added as a narrow public BF16 GEMM surface
for validation and projection-policy experiments.

Policy:

- Input storage: `CUDA_R_16BF`.
- Output storage: `CUDA_R_16BF`.
- Compute type: `CUBLAS_COMPUTE_32F`.
- Algorithm: `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, matching the baseline tensor-op family rather than a
  pedantic/no-tensor-op candidate.
- Layout: same row-major projection convention as `hgemm_into`, computing
  `C[m,n] = A[m,k] @ B[n,k]^T`.

Boundaries:

- No default runtime routing changed.
- No CUDA kernels changed.
- No cuBLAS math-mode or atomics-mode changes were added.
- No pedantic/no-tensor-op behavior was introduced.
- `qkv_projection_policy_compare` now wires this API only for explicit K baseline validation.

## K BF16 CUDA Baseline Status

`qkv_projection_policy_compare` now supports:

```bash
--execute --projection k --policy current --storage-dtype bf16
```

Current behavior:

- Loads norm input, K weight, and K oracle JSON values only for explicit K execution.
- Converts norm input and K weight values to `half::bf16` before CUDA upload.
- Runs `K_out [74, 512] = norm_input [74, 2880] x K_weight [512, 2880]^T` through
  `CublasHandle::bf16_gemm_into`.
- Downloads BF16 output and widens to f32 for comparison.
- Compares BF16 CUDA output against the K oracle and against the CPU BF16 replay.
- Includes CPU BF16 replay vs oracle metrics in the same status artifact.

Boundaries:

- Validation-only; no runtime projection routing changed.
- No CUDA kernels changed.
- No pedantic/no-tensor-op candidate policy was added.
- Q and V CUDA BF16 comparisons remain unimplemented.

The result determines whether current BF16 tensor-op GEMM matches the oneDNN oracle, matches the
legacy BF16 helper contract, or needs more contract reconciliation before any candidate policy work.

Next bounded step:

- Use the BF16 K baseline result to decide whether to extend current-baseline comparison to Q/V or
  design a K-only candidate policy comparison.

## K cuBLAS Pedantic BF16 Discriminator Status

`qkv_projection_policy_compare` now supports the K-only validation command:

```bash
--execute --projection k --storage-dtype bf16 --policy current,cublas-pedantic
```

The `cublas-pedantic` policy is harness-only and uses `CublasHandle::bf16_gemm_pedantic_into`.

Policy:

- Input/output storage: `CUDA_R_16BF`.
- Compute type: `CUBLAS_COMPUTE_32F_PEDANTIC`.
- Algorithm: `CUBLAS_GEMM_DFALT`.
- Scoped math mode: `CUBLAS_PEDANTIC_MATH`.
- Scoped atomics mode: `CUBLAS_ATOMICS_NOT_ALLOWED`.
- The previous cuBLAS math and atomics modes are restored after each scoped call.

Boundaries:

- Validation-only; no runtime projection routing changed.
- Existing `bf16_gemm_into` tensor-op behavior is unchanged.
- No CUDA kernels changed.
- No Q/V comparison or production projection policy was added.

The discriminator reports current BF16 tensor-op metrics, cublas-pedantic metrics, CPU BF16 replay
metrics, current-vs-pedantic delta, latency per policy, and math/atomics restore status. The next
step depends on whether pedantic matches the oracle, matches the CPU BF16 replay, improves but
remains unmodeled, or is not sufficient.

## V BF16 Projection Policy Comparison Status

`qkv_projection_policy_compare` now accepts optional projection bias artifacts:

```text
--q-bias <PATH>
--k-bias <PATH>
--v-bias <PATH>
```

V execution requires `--v-bias` because the official V projection path is bias-bearing:

```text
--execute --projection v --storage-dtype bf16 --policy current,cublas-pedantic
```

The V comparison is still validation-only. It runs the same current BF16 tensor-op and scoped
`cublas-pedantic` policies used by the K discriminator, then applies the V bias using a BF16
round-trip policy before comparison:

- Input/output storage: `CUDA_R_16BF`.
- Current compute: `CUBLAS_COMPUTE_32F` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
- Pedantic compute: `CUBLAS_COMPUTE_32F_PEDANTIC` with `CUBLAS_GEMM_DFALT`.
- Bias policy: BF16 GEMM output plus BF16-rounded bias, rounded to BF16 output.
- Production routing remains unchanged.
- No CUDA kernels are changed.
- Q projection comparison remains deferred because Q is larger and bias-bearing.

The V status artifact reports current/pedantic metrics versus the V oracle, pedantic/current
metrics versus the CPU BF16 replay, current-vs-pedantic delta, latency per policy, bias metadata,
and cuBLAS math/atomics restore status.

## Real V Full-Value Comparison Status

Standalone full-value V artifacts were not already present as an obvious checked-in or `.live`
artifact pack. A local scratch pack was generated outside the repository at:

```text
/tmp/qkv_projection_v_artifacts-20260428-090001/
```

Scratch provenance:

- Norm input source:
  `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-attn-norm-full-input.cpu.json`
- Model source:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- Tensor sources:
  `model.layers.0.self_attn.v_proj.weight` and `model.layers.0.self_attn.v_proj.bias`
- Oracle generation:
  CPU BF16 `torch.nn.functional.linear(norm, v_weight, v_bias)` with MKLDNN enabled.
- Runtime-forward source commit:
  `5bcba1d2edcb9c15b1ed567700976dad03e12300`
- Raw scratch artifacts are local-only and not committed.

Generated artifact digests:

- `norm_input`: `sha256-float-hex:c87d0bddb117b34f04bf18e0c94dcc0da8177ec5a99550a6566b807d55406fcc`
- `v_weight`: `sha256-float-hex:af233ce814d4d9ebdb1c9c31fa52f8a289648969011c8f12ef5b53563e953161`
- `v_bias`: `sha256-float-hex:57f9ec9658aad995521c27fdd1c50fb8bce562f5e27a5e859008b237a472ac89`
- `v_oracle`: `sha256-float-hex:3afedc5eb920470d2cd2330761043b05188075b7818e9df8d829c8f32414debc`

Real V result:

- Classification: `qkv_projection_policy_compare_v_pedantic_bf16_improves_but_unmodeled`.
- Current BF16 tensor-op vs oracle:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0018099253065884113`, `mismatches=17446`.
- Pedantic BF16 vs oracle:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012496220879256725`, `mismatches=11366`.
- Pedantic BF16 vs CPU BF16 replay:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012498591095209122`, `mismatches=11368`.
- CPU BF16 replay vs oracle:
  `max_abs_diff=0.015625`, `mean_abs_diff=7.029975108707731e-7`, `mismatches=15`.
- Latency:
  current BF16 `0.015809 ms`; pedantic BF16 `0.3494485 ms`.
- cuBLAS math and atomics modes restored after the pedantic scoped call.

Interpretation:

- V does not follow the K pattern exactly.
- Pedantic improves the real V output but does not match the CPU BF16 replay or the generated
  oneDNN/PyTorch oracle.
- Production routing remains inappropriate.
- The next projection-policy step should reconcile V bias/reduction details or move to Q only as
  another validation-only comparison, not as runtime extraction.

## V Contract Reconciliation Status

`qkv_projection_policy_compare` now emits V-specific contract reconciliation diagnostics for the
real local scratch artifacts above.

Additional diagnostics:

- Separates `V_pre_bias = norm_input @ v_weight.T` from the post-bias output.
- Compares current and pedantic BF16 CUDA pre-bias outputs against the CPU BF16 pre-bias replay.
- Evaluates bounded bias-add variants:
  - CUDA BF16 pre-bias plus BF16 bias, BF16 output.
  - CUDA BF16 pre-bias without bias as a guard.
  - CPU BF16 pre-bias plus BF16 bias, BF16 output.
  - CPU f16 pre-bias plus f16 bias, f16 output.
  - CPU f32 pre-bias plus f32 bias, BF16 output.
  - CPU BF16 pre-bias without bias as a guard.
- Emits a focused worst-mismatch trace with pre-bias value, bias value, post-bias value, oracle
  value, current value, and whether the mismatch exists before bias.
- Reports source runtime-forward status digests when available.

Real V reconciliation result:

- Classification: `qkv_projection_policy_compare_v_oracle_differs_from_cpu_bf16_replay`.
- Artifact identity finding: scratch artifacts use `sha256-float-hex` digests while the available
  runtime-forward status evidence records `sha256-prefix4096` digests, so direct digest equality is
  unavailable rather than proven mismatched.
- Pedantic BF16 pre-bias vs CPU BF16 pre-bias replay:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.
- Pedantic BF16 post-bias vs generated oracle:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012496220879256725`, `mismatches=11366`.
- CPU BF16 post-bias replay vs generated oracle:
  `max_abs_diff=0.015625`, `mean_abs_diff=7.029975108707731e-7`, `mismatches=15`.
- Best bounded bias-add variant: `cpu_f32_pre_bias_plus_f32_bias_bf16_output`, still with
  `max_abs_diff=0.015625` and `mismatches=15`.
- Focused worst pedantic mismatch: token `1`, feature `155`, kv head `2`, lane `27`; pedantic
  pre-bias and CPU BF16 pre-bias both equal `-4.3125`, so the mismatch is not a pre-bias GEMM
  issue. The pedantic BF16 post-bias output is `-4.78125`, while CPU BF16 post-bias replay and the
  generated oracle are both `-4.75` at that coordinate.

Interpretation:

- V pedantic pre-bias GEMM is calibrated to the CPU BF16 replay, matching the K pedantic
  pre-bias behavior.
- The remaining real-V discrepancy is a post-bias/oracle contract issue for the generated scratch
  artifact path, not evidence for production routing.
- Candidate policy work should remain validation-only until V oracle generation and bias-add
  semantics are pinned against runtime-forward source evidence with comparable digests.

## V Oracle Pinning Status

A new pinned local scratch pack was generated outside the repository:

```text
/tmp/qkv_projection_v_artifacts-pinned-20260428-092126/
```

Pinned artifact conventions:

- `sha256_f32_le_full`: SHA-256 over every row-major value encoded as little-endian f32.
- `sha256_prefix4096_f32_le`: SHA-256 over the first 4096 row-major values encoded as
  little-endian f32.
- Existing `digest` remains `sha256-float-hex` for compatibility with the earlier scratch pack.
- Raw pinned artifacts are local-only and are not committed.

Pinned source details:

- Exact case: `developer-message-user-smoke`.
- Source runtime-forward commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`.
- Norm input source: runtime-forward layer0 attention norm artifact.
- V weight/bias source: separate safetensors tensors
  `model.layers.0.self_attn.v_proj.weight` and `model.layers.0.self_attn.v_proj.bias`.
- Torch version: `2.10.0+cu128`; MKLDNN enabled; thread count `8`.

Oracle variant comparison:

- `torch.nn.Linear` module call vs `torch.nn.functional.linear`: exact match.
- Module/F.linear vs explicit f32 matmul plus f32 bias then BF16 output:
  `max_abs_diff=0.00048828125`, `mismatches=16`.
- CPU BF16 replay post-bias vs module/F.linear oracle:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012496220942183926`, `mismatches=11366`.

Pinned digest identity:

- The pinned artifacts now carry full and prefix little-endian f32 digests.
- Available runtime-forward status evidence still records `sha256-prefix4096` digests with a
  proof-specific convention that does not equal the pinned `sha256_prefix4096_f32_le` values.
- Artifact identity is therefore classified as `incomparable_digest_scheme_or_source`, not as a
  proven value mismatch.

Pinned V harness result using the module/F.linear oracle:

- Classification: `qkv_projection_policy_compare_v_oracle_differs_from_cpu_bf16_replay`.
- Current BF16 tensor-op vs oracle:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0018099253065884113`, `mismatches=17446`.
- Pedantic BF16 vs oracle:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012496220879256725`, `mismatches=11366`.
- Pedantic BF16 vs CPU BF16 replay:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012498591095209122`, `mismatches=11368`.
- CPU BF16 replay vs oracle:
  `max_abs_diff=0.015625`, `mean_abs_diff=7.029975108707731e-7`, `mismatches=15`.

Conclusion:

- V pre-bias GEMM remains calibrated for the pedantic path.
- The V oracle is now pinned to module/F.linear behavior, and explicit matmul-plus-bias is not an
  equivalent oracle.
- The remaining issue is the exact oneDNN/module BF16 bias/reduction contract, not CUDA runtime
  routing.
- Treat the V ambiguity as a pinned validation finding. It should not block Q validation, but it
  does block production V routing.

## Q Projection Policy Comparison Status

A pinned local Q scratch pack was generated outside the repository:

```text
/tmp/qkv_projection_q_artifacts-20260428-093932/
```

Q artifact source:

- Exact case: `developer-message-user-smoke`.
- Source runtime-forward commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`.
- Norm input source: runtime-forward layer0 attention norm artifact.
- Q weight/bias source: separate safetensors tensors
  `model.layers.0.self_attn.q_proj.weight` and `model.layers.0.self_attn.q_proj.bias`.
- Torch version: `2.10.0+cu128`; MKLDNN enabled; thread count `8`.
- Raw Q artifacts are local-only and are not committed.

Oracle variant comparison:

- `torch.nn.Linear` module call vs `torch.nn.functional.linear`: exact match.
- Module/F.linear vs explicit f32 matmul plus f32 bias then BF16 output:
  `max_abs_diff=0.00390625`, `mismatches=53`.
- CPU BF16 replay post-bias vs module/F.linear oracle:
  `max_abs_diff=0.0625`, `mean_abs_diff=0.000886685973283248`, `mismatches=88225`.

Harness changes:

- `qkv_projection_policy_compare` now supports
  `--execute --projection q --storage-dtype bf16 --policy current,cublas-pedantic`.
- Q execution loads explicit Q weight, Q bias, and Q oracle artifacts.
- Q output uses Q-specific logical indices: `q_head = feature / 64`, `lane = feature % 64`.
- CPU replay helper is parallelized across token rows to keep full Q f32/f16/BF16 replay
  diagnostics bounded.
- Default runtime behavior remains unchanged.

Pinned Q harness result using the module/F.linear oracle:

- Classification: `qkv_projection_policy_compare_q_oracle_differs_from_cpu_bf16_replay`.
- Current BF16 tensor-op vs oracle:
  `max_abs_diff=0.0625`, `mean_abs_diff=0.0010602121474221349`, `mismatches=107138`.
- Pedantic BF16 vs oracle:
  `max_abs_diff=0.0625`, `mean_abs_diff=0.0008867621072567999`, `mismatches=88227`.
- Pedantic BF16 vs CPU BF16 replay:
  `max_abs_diff=0.0625`, `mean_abs_diff=0.0008866449352353811`, `mismatches=88214`.
- Pedantic BF16 pre-bias vs CPU BF16 pre-bias replay:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.
- CPU BF16 replay vs oracle:
  `max_abs_diff=0.00390625`, `mean_abs_diff=2.185926462061616e-7`, `mismatches=92`.
- Latency:
  current BF16 `0.069368 ms`; pedantic BF16 `0.3861728 ms`.

## Q/K/V Validation Matrix Summary

The validation-only harness now covers K, V, and Q with the current BF16 tensor-op policy and the
scoped `cublas-pedantic` BF16 policy. All results below use explicit local artifact paths; no raw
artifacts are committed.

| Projection | Bias? | Current BF16 vs oracle | Pedantic BF16 vs oracle | Pedantic vs CPU BF16 replay | CPU BF16 replay vs oracle | Current latency | Pedantic latency | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K | No | `max=0.5`, `mismatches=12729` | `max=0.0078125`, `mismatches=6` | Exact match | Six BF16 replay-vs-oracle lanes remain | `~0.0158 ms` | `~0.29-0.33 ms` | Pedantic calibrates K to CPU BF16 replay, but not exactly to the oneDNN/PyTorch oracle. |
| V | Yes | `max=0.03125`, `mean=0.0018099253065884113`, `mismatches=17446` | `max=0.03125`, `mean=0.0012496220879256725`, `mismatches=11366` | Pre-bias exact; post-bias `max=0.03125`, `mismatches=11368` | `max=0.015625`, `mean=7.029975108707731e-7`, `mismatches=15` | `~0.0158 ms` | `~0.35 ms` | V exposes a module/F.linear fused-backend bias/output contract gap after calibrated pre-bias GEMM. |
| Q | Yes | `max=0.0625`, `mean=0.0010602121474221349`, `mismatches=107138` | `max=0.0625`, `mean=0.0008867621072567999`, `mismatches=88227` | Pre-bias exact; post-bias `max=0.0625`, `mismatches=88214` | `max=0.00390625`, `mean=2.185926462061616e-7`, `mismatches=92` | `0.069368 ms` | `0.3861728 ms` | Q follows the V pattern at larger scale: calibrated pre-bias GEMM, unresolved module/F.linear post-bias oracle contract. |

Design conclusion:

- Pedantic BF16 is useful as a validation/pre-bias replay proxy for Q/K/V GEMM behavior.
- Pedantic BF16 is much slower than current tensor-op BF16 for these shapes.
- Pedantic BF16 does not reproduce the official module/F.linear post-bias oracle for Q/V.
- The remaining oracle gap appears to be the fused backend/module linear contract, especially
  post-bias/output rounding for biased projections.
- Pedantic BF16 should not be production-routed.
- No production Q/K/V projection policy should be promoted from the current validation findings.

## Custom BF16 Biased Linear Validation Kernel Status

A correctness-first validation-only CUDA kernel was added for the V projection experiment:

- Kernel file: `kernels/bf16_linear_bias_validation.cu`.
- Symbol: `bf16_linear_bias_validation_kernel`.
- Harness policy: `custom-bf16-linear`.
- Supported invocation:
  `--execute --projection v --storage-dtype bf16 --policy custom-bf16-linear`.
- Layout: input row-major `[m,k]`, weight row-major `[n,k]`, output row-major `[m,n]`.
- Dtypes: BF16 input, BF16 weight, BF16 bias, FP32 accumulation, BF16 output.
- Mode `0`: f32 accumulation plus f32 bias add, rounded to BF16 output.
- The kernel is launched only by `qkv_projection_policy_compare` through a narrow
  `KernelLoader` validation helper. It is not production routing.

Pinned V result using `/tmp/qkv_projection_v_artifacts-pinned-20260428-092126/`:

- Classification:
  `qkv_projection_policy_compare_v_custom_bf16_linear_matches_cpu_bf16_replay`.
- Custom BF16 linear vs module/F.linear oracle:
  `max_abs_diff=0.015625`, `mean_abs_diff=7.029975108707731e-7`, `mismatches=15`.
- Custom BF16 linear vs CPU BF16 replay:
  exact match.
- Custom BF16 linear vs cuBLAS pedantic:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0012498591095209122`, `mismatches=11368`.
- Latency: `~1.021 ms` for the one-output-element-per-thread correctness kernel.

Conclusion:

- The custom kernel confirms the decomposed BF16 replay contract, including bias, but it still does
  not reproduce the official module/F.linear fused-backend output for V.
- This narrows the remaining gap to module/F.linear epilogue/output semantics, not basic BF16
  input, weight, bias, or deterministic FP32 accumulation plumbing.
- The custom kernel is intentionally slow and validation-only. It should not be routed into
  production runtime.

## Downstream Impact Question

There are now two possible correctness standards for CUDA/Rust projection work:

- Strict oneDNN/F.linear epilogue parity: reproduce the official PyTorch/oneDNN module output at
  the projection boundary, including fused bias/output rounding details.
- Rust/CUDA decomposed BF16 policy: define a deterministic BF16 projection contract that matches
  the custom kernel/CPU BF16 replay, then require downstream validation to show projection-level
  differences do not affect the proof-relevant seams.

The current custom V kernel matches decomposed CPU BF16 replay exactly, but it remains 15 lanes
from the module/F.linear V projection oracle. Before chasing exact oneDNN epilogue behavior, the
harness now supports an opt-in weighted-V downstream check:

```text
--attention-probs <PATH>
--weighted-v-oracle <PATH>
```

For V, the check uses official final-token attention probabilities shaped `[64,75]`, drops the
sink probability column before the value sum, applies GQA mapping `kv_head = q_head / 8`, and
compares the custom V weighted sum against the official pre-`o_proj` weighted-value oracle shaped
`[4096]`.

Next evidence needed:

- Weighted V sum impact for custom/decomposed V output.
- Q/K raw QK impact for custom/decomposed Q/K outputs.
- Final logits and top-k impact if future harness extraction makes this integration-safe.

Recommended compact artifact pack for downstream checks:

- Attention probabilities `[64,75]` with sink column.
- Weighted V oracle `[4096]` before `o_proj`.
- Custom V output `[74,512]` or inputs sufficient to regenerate it.

V downstream result using the pinned official reference artifacts:

- Attention probabilities:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-probs-post-softmax-status.json`.
- Weighted V oracle:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-weighted-value-sum-before-output-projection-status.json`.
- Classification: `v_custom_projection_downstream_weighted_sum_matches_oracle`.
- Custom V projection vs module/F.linear V oracle still has 15 mismatching projection lanes.
- Weighted V with f32 exploratory output vs oracle:
  `max_abs_diff=0.013309478759765625`, `mean_abs_diff=0.0006673983880318701`,
  `mismatches=4096`.
- Weighted V rounded to BF16 output vs oracle:
  exact match.

Conclusion:

- For the exact layer0 final-token V downstream boundary, the 15 projection-level custom/decomposed
  BF16 differences disappear after the official sink-dropping weighted V sum when the weighted sum
  is compared at the BF16 output boundary.
- This does not prove production suitability, but it is evidence that strict module/F.linear V
  projection epilogue parity may be stronger than needed for this downstream seam.
- Equivalent downstream-impact checks are still needed for Q/K raw QK and final logits before any
  production projection routing decision.

## Q/K Downstream Raw QK Impact Check

The next downstream seam for the decomposed/custom projection policy is
`layer0_final_token_raw_scaled_qk_logits_pre_mask`, shaped `[64,74]` with scale `0.125`.

Artifact availability inspection found the official pinned full-value artifacts:

- Official Q post-RoPE before attention:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-q-post-rope-before-attention.cpu.json`
  with shape `[74,4096]`.
- Official grouped K post-RoPE:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-k-post-rope-grouped.cpu.json`
  with shape `[74,8,64]`.
- Official final-token raw scaled QK logits before mask:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-final-token-raw-scaled-qk-logits-pre-mask.cpu.json`
  with shape `[64,74]`.

Additional runtime-forward status evidence exists, but it is not a pinned
custom/decomposed Q/K policy artifact:

- `developer-message.runner-layer0-q-pre-post-rope-runtime-localization.local-q-capture.json`
  contains local runtime Q pre/post-RoPE captures shaped `[74,4096]`. Its metrics line up with the
  current/runtime Q projection delta, not with a validated custom/decomposed Q policy.
- `developer-message.runner-layer0-k-post-rope-and-score-after-onednn-k-candidate-status.json`
  records that scoped oneDNN/candidate K post-RoPE can match official K and that official Q plus
  candidate K clears raw QK, but local runtime Q plus candidate K still mismatches raw QK.

Route chosen: Option C, blocked pending a compact custom/decomposed Q/K post-RoPE artifact pack.

Why this slice did not add a raw-QK harness run:

- Running official Q post-RoPE plus official K post-RoPE would validate the dot-product formula,
  but it would not answer whether custom/decomposed Q/K projection differences matter downstream.
- Running the available local runtime Q capture would answer the current-runtime Q path, not the
  validation-only custom/decomposed projection policy.
- Generating custom/decomposed Q/K post-RoPE inside this slice would require adding Q custom biased
  projection support and a RoPE transform path, which is larger than a downstream-impact probe.

Required artifact pack for the next Q/K downstream check:

- Custom/decomposed Q post-RoPE final token `[4096]` or all tokens `[74,4096]`.
- Custom/decomposed K post-RoPE all tokens `[74,8,64]`.
- Official raw scaled QK oracle `[64,74]`.
- Provenance recording projection policy, RoPE implementation, source commit, exact case,
  final-token index `73`, scale `0.125`, and comparable digests.

Planned comparison once the pack exists:

- For each `q_head in 0..64`, use `kv_head = q_head / 8`.
- For each key token in `0..74`, compute
  `dot(Q_final[q_head,:], K[token,kv_head,:]) * 0.125`.
- Compare both f32 exploratory scores and BF16-rounded scores against the official raw-QK oracle.

Current classification: `qk_downstream_blocked_by_missing_artifacts`.

## Q/K Downstream Raw QK Artifact-Pack Status

A local scratch artifact pack was generated for the custom/decomposed Q/K downstream check:

- Scratch directory:
  `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/`.
- Norm input source:
  `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-attn-norm-full-input.cpu.json`.
- Q tensor source:
  `model.layers.0.self_attn.q_proj.weight` and `model.layers.0.self_attn.q_proj.bias`
  from `/data/models/openai/gpt-oss-20b`.
- K tensor source:
  `model.layers.0.self_attn.k_proj.weight` from `/data/models/openai/gpt-oss-20b`.
- K bias policy:
  absent/not applied for the decomposed policy.
- Projection policy:
  BF16 input, BF16 weight, BF16 Q bias, FP32 accumulation, BF16 projection output.
- RoPE policy:
  local Python half-split RoPE implementation matching the promoted runtime convention, with BF16
  post-RoPE output.
- Raw-QK policy:
  final query token index `73`, scale `0.125`, output compared after BF16 rounding.

Generated scratch artifacts:

- `q_pre_rope_custom.json`: `[74,4096]`.
- `k_pre_rope_custom.json`: `[74,512]`.
- `q_post_rope_custom_final_token.json`: `[4096]`.
- `k_post_rope_custom_all_tokens.json`: `[74,8,64]`.
- `raw_qk_custom_bf16.json`: `[64,74]`.
- `raw_qk_oracle.json`: `[64,74]`.
- `provenance.json` and `comparison_summary.json` with comparable `sha256_f32_le_full` and
  `sha256_prefix4096_f32_le` digests.

Official sanity check:

- Recomputed raw QK from official Q post-RoPE `[74,4096]` and official K post-RoPE `[74,8,64]`
  matches the official raw scaled QK oracle exactly after BF16 rounding:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.
- This validates the raw-QK dot-product formula and BF16 output comparison boundary used by this
  scratch check.

Custom/decomposed Q/K downstream result:

- Classification: `qk_downstream_custom_decomposed_raw_qk_mismatch_large`.
- Custom BF16-rounded raw QK vs official oracle:
  `max_abs_diff=1.0`, `mean_abs_diff=0.06418905407190323`, `mismatches=2847`.
- Custom f32 exploratory raw QK vs official BF16 oracle:
  `max_abs_diff=0.76666259765625`, `mean_abs_diff=0.06887843459844589`,
  `mismatches=4736`.
- First BF16-rounded mismatch:
  `q_head=0`, `token=0`, custom `-3.578125`, oracle `-3.5625`, abs diff `0.015625`.
- Worst BF16-rounded mismatch:
  `q_head=22`, `token=1`, custom `-72.5`, oracle `-71.5`, abs diff `1.0`.

Projection/RoPE boundary comparison:

- Custom Q post-RoPE final token vs official Q final token:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0021520100999623537`, `mismatches=1707`.
- Custom K post-RoPE all tokens vs official K post-RoPE:
  `max_abs_diff=0.5`, `mean_abs_diff=0.0066139730624854565`, `mismatches=15417`.
- A bounded variant check found that applying the model K bias did not change the raw-QK metric,
  and using FP16 RoPE output reduced the mean error slightly but still left a large mismatch.

Conclusion:

- Unlike the V weighted-sum seam, the custom/decomposed Q/K projection differences do matter at the
  exact layer0 final-token raw-QK BF16 boundary.
- Strict module/F.linear projection-boundary parity may be stronger than needed for V, but Q/K
  require either a closer projection/RoPE policy or an additional downstream-correctness candidate.
- No production Q/K/V projection routing should be added from the current custom/decomposed policy.
- The scratch artifacts are local-only and are not committed.

## Q/K Raw-QK Mismatch Attribution

A local mixed-source attribution check was run against the same scratch pack:

- Artifact pack:
  `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/`.
- Summary JSON:
  `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/qk_mixed_attribution_summary.json`.
- Official Q/K/raw-QK artifacts:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/`.
- Comparison boundary:
  raw scores rounded to BF16 before comparison against the official raw-QK oracle.

| Q source | K source | max abs diff | mean abs diff | mismatches | Conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| official | official | `0.0` | `0.0` | `0` | Guard passes; formula and BF16 boundary are correct. |
| custom/decomposed | official | `0.5` | `0.03450170159339905` | `1947` | Q-side custom/decomposed differences alone are large. |
| official | custom/decomposed | `0.5` | `0.04469071328639984` | `2338` | K-side custom/decomposed differences alone are large. |
| custom/decomposed | custom/decomposed | `1.0` | `0.06418905407190323` | `2847` | Q and K differences compound. |

Classification: `qk_downstream_mismatch_q_and_k_contribute`.

Focused locations:

- Custom Q + official K first mismatch:
  `q_head=0`, `token=13`, custom `-4.6875`, oracle `-4.65625`, abs diff `0.03125`.
- Custom Q + official K worst mismatch:
  `q_head=16`, `token=4`, custom `-68.0`, oracle `-67.5`, abs diff `0.5`.
- Official Q + custom K first mismatch:
  `q_head=0`, `token=0`, custom `-3.578125`, oracle `-3.5625`, abs diff `0.015625`.
- Official Q + custom K worst mismatch:
  `q_head=16`, `token=6`, custom `-78.5`, oracle `-78.0`, abs diff `0.5`.

Conclusion:

- The raw-QK downstream failure is not Q-only or K-only; both custom/decomposed Q and
  custom/decomposed K independently contribute large mismatches.
- The next projection-policy work must focus on both Q and K projection/RoPE boundary parity, not
  just the Q bias epilogue or the K six-lane projection boundary.
- The V weighted-sum result should remain scoped to V; it does not generalize to Q/K raw-QK.
- No runtime behavior changed and no raw artifacts are committed.

## Q/K Projection vs RoPE Localization

A follow-up localization check was run against the same scratch pack:

- Summary JSON:
  `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/qk_projection_rope_localization_summary.json`.
- Classification: `qk_projection_mismatch_rope_transform_dominant`.

Official pre-RoPE artifact availability:

- Q full-value official pre-RoPE artifact:
  not available as a standalone value tensor in the inspected `.live` artifacts.
- Q official/candidate status metrics:
  available. Runtime-forward status records candidate Q pre-RoPE vs official as exact, candidate Q
  post-RoPE vs official as exact, and local/runtime Q post-RoPE as broad mismatch.
- K full-value official pre-RoPE artifact:
  available in
  `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-k-projection-weight-arithmetic.cpu.json`.

Metrics from the local scratch pack and available official artifacts:

- Custom Q final-token post-RoPE vs official final-token Q post-RoPE:
  `max_abs_diff=0.03125`, `mean_abs_diff=0.0021520100999623537`, `mismatches=1707`.
- Custom K pre-RoPE vs official module K pre-RoPE:
  `max_abs_diff=0.001953125`, `mean_abs_diff=5.2416783802300415e-08`, `mismatches=5`.
- Custom K post-RoPE vs official K post-RoPE:
  `max_abs_diff=0.5`, `mean_abs_diff=0.0066139730624854565`, `mismatches=15417`.

Concentration findings:

- Q final-token post-RoPE mismatch is spread across many heads; top mean-diff heads include
  `q_head=27`, `51`, `14`, `22`, and `62`, all with `max_abs_diff=0.03125`.
- K post-RoPE mismatch is broad across all KV heads. Top mean-diff KV heads are `2`, `5`, `4`,
  `1`, and `0`; KV heads `2` and `5` reach `max_abs_diff=0.5`.

Critical guard:

- Feeding official module K pre-RoPE through the local Python half-split RoPE generator did not
  reproduce official K post-RoPE.
- Best tested local variant was BF16 output with the promoted sign convention, but it still had
  `max_abs_diff=0.5`, `mean_abs_diff=0.006613972131162882`, `mismatches=15416`.
- This means the generated scratch custom Q/K post-RoPE pack is not suitable to attribute the
  raw-QK mismatch to projection alone until RoPE generation is pinned against the official/model
  RoPE path.

Conclusion:

- The raw-QK formula and BF16 comparison boundary are still guarded by official Q + official K
  recomputing the official raw-QK oracle exactly.
- The current scratch custom/decomposed Q/K post-RoPE artifacts include a local RoPE generation
  mismatch, so the previous mixed attribution is qualified: both custom scratch Q and K damage
  raw-QK, but the immediate root to fix is RoPE transform/table parity in the artifact generator.
- Next focus should be local RoPE generation parity against official/model RoPE using official K
  pre/post artifacts, then regenerate custom/decomposed Q/K post-RoPE and rerun raw-QK attribution.
- No runtime behavior changed and no raw artifacts are committed.

## RoPE Pinning For Q/K Downstream Artifacts

A bounded RoPE pinning sweep was run against official K pre/post full-value artifacts:

- Summary JSON:
  `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/rope_pinning_summary.json`.
- Official K pre-RoPE source:
  `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-k-projection-weight-arithmetic.cpu.json`.
- Official K post-RoPE source:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-k-post-rope-grouped.cpu.json`.

Variants tested:

- Lane pairing:
  half-split and adjacent.
- Sign convention:
  standard promoted sign and inverse sign.
- Position indexing:
  token index `0..73`, plus `+1` and `-1` sanity offsets.
- Table source:
  YaRN-scaled config table, plain config `rope_theta`, and base `10000`.
- Input/output dtype:
  f32, BF16, and F16 combinations.

Classification: `rope_pinning_unresolved`.

Best variant:

- Table source: YaRN-scaled config table.
- Pairing: half-split.
- Sign: promoted sign convention.
- Position offset: `0`.
- Input dtype: f32/BF16/F16 all tied.
- Output dtype: BF16.
- Metrics vs official K post-RoPE:
  `max_abs_diff=0.5`, `mean_abs_diff=0.006613972131162882`, `mismatches=15416`.

Important contrast:

- The current integration/runtime RoPE table code path observed in `gpu_runner.rs` and
  `rotary_cuda.rs` uses plain `rope_theta` table generation, not the YaRN-scaled table used by the
  best local scratch variant.
- Plain config `rope_theta` with half-split, promoted sign, and BF16 output was much worse:
  `max_abs_diff=31.875`, `mean_abs_diff=0.8329102993011475`, `mismatches=37845`.
- Therefore the exact official/model RoPE path is not captured by the bounded local generator, and
  the integration/runtime table source needs explicit follow-up before using scratch-generated Q/K
  post-RoPE artifacts for downstream policy decisions.

Decision:

- No exact RoPE variant was found.
- Custom/decomposed Q/K post-RoPE artifacts were not regenerated in this slice.
- Raw-QK and mixed attribution were not rerun because doing so would keep using unpinned RoPE.

Next bounded step:

- Pin the official/model RoPE table source and transform more directly, preferably by extracting a
  compact official RoPE table or by using the model/module RoPE call to generate local scratch
  Q/K post-RoPE artifacts.
- Only after that, regenerate custom/decomposed Q/K post-RoPE and rerun raw-QK attribution.
- Keep this validation-only; do not route production Q/K/V projections.

## Official/Model RoPE Pinning Status

The official/model RoPE path was pinned directly after the bounded local table sweep failed:

- Summary JSON:
  `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/qk_official_rope_attribution_summary.json`.
- Approach:
  use the GPT-OSS torch model `AttentionBlock.rope` / `RotaryEmbedding.forward` call directly.
- Official capture paths inspected:
  `/home/emmy/openai/worktrees/runtime-forward/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py`,
  `/home/emmy/openai/worktrees/runtime-forward/crates/gpt-oss-bench/tools/layer0_k_post_rope_and_score_after_onednn_k_candidate.py`,
  and
  `/home/emmy/openai/worktrees/runtime-forward/crates/gpt-oss-bench/tools/layer0_q_projection_onednn_oracle_scoped_candidate.py`.
- Official model implementation:
  `/home/emmy/openai/gpt-oss/gpt_oss/torch/model.py`.
- Classification:
  `rope_pinning_official_model_call_guard_passed_qk_rerun_complete`.

Guard result:

- Official module K pre-RoPE passed through `AttentionBlock.rope` reproduced official K post-RoPE
  exactly:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.

Pinned raw-QK rerun:

- Custom/decomposed Q and K pre-RoPE artifacts were passed through the official/model RoPE call.
- The regenerated custom final-token Q post-RoPE matched official final-token Q exactly:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.
- The regenerated custom K post-RoPE differed from official K in only two very small lanes:
  `max_abs_diff=0.000030517578125`, `mean_abs_diff=9.061517092234794e-10`,
  `mismatches=2`.
- BF16-rounded custom Q + custom K raw-QK matched the official raw-QK oracle exactly:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.

Mixed attribution rerun with official/model RoPE:

| Q source | K source | max abs diff | mean abs diff | mismatches |
| --- | --- | ---: | ---: | ---: |
| official | official | `0.0` | `0.0` | `0` |
| custom/decomposed | official | `0.0` | `0.0` | `0` |
| official | custom/decomposed | `0.0` | `0.0` | `0` |
| custom/decomposed | custom/decomposed | `0.0` | `0.0` | `0` |

Conclusion:

- The earlier large Q/K raw-QK drift was caused by the scratch RoPE generator, not by the
  custom/decomposed Q/K projection policy.
- With official/model RoPE pinned, the exact layer0 final-token raw-QK BF16 boundary matches the
  official oracle.
- This remains a validation-only finding. No runtime behavior changed, no runtime Q/K/V routing was
  added, and no raw `/tmp` or `.live` artifacts are committed.

## Downstream Projection-Policy Decision Summary

The downstream attention seams now give a stronger validation picture than the projection-boundary
module/F.linear comparisons alone:

- Q/K:
  custom/decomposed Q and K pre-RoPE passed through the official/model RoPE call produce exact
  BF16-rounded final-token raw-QK scores against the official oracle.
- V:
  custom/decomposed V projection still differs from module/F.linear at 15 projection lanes, but the
  downstream weighted-V sum matched the official BF16 boundary exactly.
- RoPE:
  the earlier large Q/K raw-QK mismatch was caused by an unpinned scratch RoPE generator. It is not a
  projection-policy verdict.

This supports the decomposed BF16 projection policy as a validation candidate for layer0 attention
downstream seams, while keeping strict projection-boundary oneDNN/module epilogue parity as a
separate open question.

### Attention O-Proj Seam Check

A bounded scratch check was also run for the next layer0 attention seam:

- Summary JSON:
  `/tmp/qkv_projection_attention_oproj_artifacts-20260428-validated/attention_oproj_custom_policy_summary.json`.
- Boundary:
  `layer0_final_token_attention_output_after_o_proj_before_residual`.
- Input:
  official weighted-V BF16 boundary `[4096]`, which is downstream-equivalent to the custom V path
  because the custom V weighted-sum check matched exactly.
- Weights:
  layer0 attention output projection weight/bias from the local safetensors model.
- Oracle:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-output-after-o-proj-before-residual-status.json`.

Classification: `attention_oproj_custom_policy_matches_oracle`.

Best checked policy:

- `torch.nn.functional.linear` with BF16 input, weight, and bias.
- Metrics vs official o-proj-before-residual oracle:
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, `mismatches=0`.

Context:

- The decomposed CPU matmul variants differed by only three tiny lanes:
  `max_abs_diff=0.000003814697265625`, `mean_abs_diff=1.6556845894299954e-9`,
  `mismatches=3`.
- This confirms that the attention o-proj seam can be reproduced exactly with the official BF16
  linear backend when fed the downstream-equivalent weighted-V boundary, but it does not yet justify
  production routing through any custom projection path.

Current decision:

- Do not route production Q/K/V projections yet.
- Continue treating decomposed BF16 projection as a validation candidate rather than a production
  policy.
- Production promotion still requires a guarded runtime validation path, CUDA performance evidence,
  more layers/cases, a final-token/logit smoke, and later 4097-boundary testing.
- No runtime behavior changed, no production CUDA kernels changed, and no raw `/tmp` or `.live`
  artifacts are committed.

Recommended next design step:

- Investigate a module/F.linear-equivalent biased projection validation policy, including epilogue
  and output-rounding semantics, before any runtime projection routing.
- In parallel, define production acceptance criteria: whether top-k/logit-order parity is
  sufficient for production while exact full-logit oracle parity remains a validation-only target.

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

Choose the next design question before adding any new candidate policy: either investigate a
module/F.linear-equivalent biased projection validation policy, or define whether top-k/logit-order
parity is an acceptable production criterion while exact full-logit oracle parity remains
validation-only. Keep the harness validation-only, with no production routing and no CUDA kernel
changes unless a candidate policy is explicitly approved later.
