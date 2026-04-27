# Runtime-Forward Runtime Candidate Promotion Plan

Classification: `runtime_candidate_rope_fix_extracted`

This plan starts the runtime/CUDA promotion track for the
`feature/runtime-forward` final-token oracle parity milestone without merging
the feature branch wholesale.

## Source

- Source branch: `feature/runtime-forward`
- Source worktree: `/home/emmy/openai/worktrees/runtime-forward`
- Source milestone commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`
- Current promotion branch: `promotion/runtime-forward-runtime-candidates`
- Proof artifact:
  `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json`
- Proof classification: `final_readout_direct_module_logits_cleared`

The final-token proof path matched official through the transformer stack,
final norm, and LM-head logits for `developer-message-user-smoke`. That proof
used bench/proof-only mechanisms that must not be promoted blindly.

## Runtime Diff Inventory

Runtime-affecting paths changed on `feature/runtime-forward` include:

- `kernels/rotary_embedding.cu`
- `kernels/rotary_embedding_f16.cu`
- `kernels/rms_norm_f16.cu`
- `crates/gpt-oss-model-runner/src/gpu_layer.rs`
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
- `crates/gpt-oss-model-runner/src/runner.rs`
- `crates/gpt-oss-model-runner/src/architectures/*.rs`
- `crates/gpt-oss-gpu/src/cublas.rs`
- `crates/gpt-oss-gpu/src/kernel_loader.rs`
- `crates/gpt-oss-gpu/src/lib.rs`
- `crates/gpt-oss-engine/src/gpu_engine.rs`
- `crates/gpt-oss-engine/src/worker/*.rs`
- `crates/gpt-oss-server/src/*`
- `crates/gpt-oss-tokenizer/src/protocol.rs`
- `crates/gpt-oss-bench/**`

Only the RoPE kernel pair is extracted in this slice. All broader runtime,
server/protocol, and bench/proof changes remain deferred.

## Candidate Matrix

| Candidate | Changed files | Runtime-affecting | Performance-sensitive | Default path affected | Proof artifacts | Validation available | Dependencies | Recommendation | Risk | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoPE half-split pairing fix | `kernels/rotary_embedding.cu`, `kernels/rotary_embedding_f16.cu` | Yes | Low | Yes, CUDA RoPE Q/K path | K RoPE/grouped compare statuses and final readout direct-module artifact | `git diff --check`, cargo checks, future CUDA smoke | None beyond existing kernels | Extract first | Medium semantic risk because default RoPE behavior changes, but patch is isolated and matches GPT-OSS half-split convention | Extracted in this slice |
| BF16 RMSNorm policy/scalar fix | `kernels/rms_norm_f16.cu`, `crates/gpt-oss-gpu/src/kernel_loader.rs`, `crates/gpt-oss-model-runner/src/gpu_layer.rs`, related runner/debug paths | Yes | Yes | Yes, layer0 f16 RMSNorm path in source branch | Layer0 attention RMSNorm scalar/runtime-fix statuses and final readout artifact | Needs focused CUDA validation and perf guardrails | Entangled with debug/proof QKV helpers in source diff | Defer | Higher, due to new kernel, loader symbols, shared memory policy, and default-path numeric change | Prepare separate scoped extraction with perf notes |
| cuBLAS BF16/pedantic helpers | `crates/gpt-oss-gpu/src/cublas.rs`, `crates/gpt-oss-gpu/src/lib.rs`, `crates/gpt-oss-gpu/Cargo.toml`, model-runner callers | Partly | Yes | Not as normal default GEMM in source proof path; used by K helper/proof paths | All-token K pedantic/runtime-fix status | Needs isolated helper tests and performance guardrails | `cudarc`, BF16 types, proof-only callers | Defer | High if pedantic mode leaks into default GEMM | Review after RMSNorm/RoPE |
| Q/K/V projection oneDNN candidates | `crates/gpt-oss-model-runner/src/gpu_layer.rs`, bench tools | No for default runtime, proof-only candidate path | Yes | No, env/proof-gated | Q/K/V oneDNN scoped candidate statuses | Proof-only Python/Torch checks | Torch/MKLDNN policy, large artifacts, debug traces | Do not promote | High and not CUDA runtime implementation | Keep in feature/proof lane |
| Selected expert output readout correction | bench tools and diagnostic readout paths | No unless a real runtime staging bug is separately proven | No for default runtime | No | Selected expert output readout fix/localization statuses | Proof-only | Torch artifacts, selected-output readout assumptions | Do not promote | Medium diagnostic risk; likely status readout correction, not runtime fix | Keep proof-only |
| Debug capture plumbing | `gpu_runner.rs`, `gpu_worker.rs`, bench bins | Partly | Medium | Should remain off default paths | Ordered bundle/final readout statuses | Needs API design review | Broad runtime API surface | Do not promote broadly | High surface-area risk | Keep gated or redesign narrowly |
| Server/Harmony/protocol changes | `crates/gpt-oss-server/**`, `crates/gpt-oss-tokenizer/src/protocol.rs` | Yes | No | Yes | Not part of final-token CUDA proof | Separate route/protocol validation | Server/Harmony lanes | Do not include | Unrelated behavior risk | Leave to owning lane |

## Candidate Extracted

Extracted candidate: RoPE half-split pairing fix.

Files changed:

- `kernels/rotary_embedding.cu`
- `kernels/rotary_embedding_f16.cu`

The extraction changes RoPE pair indexing from adjacent `(2i, 2i + 1)` lanes
to GPT-OSS half-split `(i, half_dim + i)` lanes for both query and key paths.
No debug capture plumbing, oneDNN candidate code, BF16 RMSNorm policy, cuBLAS
helper change, server/protocol change, `.live` artifact, or Rust harness import
is included.

## Performance Guardrails

- Treat RoPE as a semantic fix, not a performance refactor.
- Keep launch shape and memory access count unchanged.
- Run the smallest available cargo checks on CPU branches now.
- Before broader promotion, run a CUDA smoke that exercises both f32 and f16
  RoPE kernels and checks a known GPT-OSS final-token boundary artifact.
- Do not combine future BF16 RMSNorm or cuBLAS helper changes with this RoPE
  commit.

## Do Not Promote Yet

- BF16 RMSNorm policy/scalar-reduction fix.
- cuBLAS BF16/pedantic helpers.
- oneDNN Q/K/V projection-policy candidates.
- selected expert output readout correction.
- broad debug capture plumbing.
- server/Harmony/protocol behavior.
- raw `.live` PPP/full-value artifacts.
- the full `runtime_forward_layer0_qkv_bf16_candidate_status.rs` bench file.

## Next Bounded Step

Review this RoPE-only runtime candidate. If accepted, prepare a separate BF16
RMSNorm extraction plan that isolates the kernel and loader changes from debug
capture and Q/K/V proof-only plumbing, with explicit performance guardrails.

## BF16 RMSNorm Candidate Extraction Plan

Classification: `rmsnorm_candidate_plan_ready_for_scoped_extraction`

This is a plan only. No BF16 RMSNorm kernel, loader, model-runner, engine,
debug, or proof-bench code is extracted in this commit.

### Source Files Inspected

Compared from current `promotion/runtime-forward-runtime-candidates` to
`feature/runtime-forward` milestone
`5bcba1d2edcb9c15b1ed567700976dad03e12300`:

- `kernels/rms_norm_f16.cu`
- `crates/gpt-oss-gpu/src/kernel_loader.rs`
- `crates/gpt-oss-model-runner/src/gpu_layer.rs`
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
- `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
- `crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs`

The source diff across those paths is broad: six files, roughly 49k insertions,
and large debug/proof additions. It must not be copied wholesale.

### Runtime, Debug, and Proof Split

Required runtime behavior:

- Add BF16 input, BF16 weight, f32 reduction, BF16 output-rounding semantics
  for the scoped GPT-OSS layer-0 pre-attention RMSNorm path.
- Use lane-order pairwise/tree f32 sum over BF16-expanded `x * x`.
- Compute inverse RMS as `1.0f / sqrtf(mean_square + eps)`.
- Multiply as `(x_bf16 * inverse_rms) * weight_bf16`.
- Round the output through BF16 before storing into the existing half buffer.

Required kernel-loader/plumbing:

- Register only the production BF16 policy kernel symbol.
- Route only the intended layer-0 attention RMSNorm call to the BF16 policy
  path.
- Keep later attention RMSNorms, MLP RMSNorms, final RMSNorm, fused residual
  RMSNorm, and non-BF16/non-GPT-OSS paths on the existing kernels.

Debug/capture-only plumbing to exclude:

- `rms_norm_f16_bf16_policy_debug_kernel`.
- Scalar capture buffers and `GPT_OSS_LAYER0_RMSNORM_SCALAR_DEBUG` env handling.
- `GpuModelRunner` debug trace structs and debug prefill APIs.
- `GpuWorker` debug logits, direct-runner, and trace APIs.
- Progress/timing debug plumbing unrelated to the production RMSNorm call.

Bench/proof-only code to exclude:

- `runtime_forward_layer0_qkv_bf16_candidate_status.rs`.
- Replay sweep modes and `.live` status writers.
- Python/Torch helper scripts and large PPP/full-value artifacts.
- Q/K/V oneDNN candidate paths and selected expert readout correction modes.

### Proposed Extraction Scope

Recommended strategy: split into two small commits.

1. Add a production-only BF16 RMSNorm kernel variant and loader registration.
   Proposed files:
   - `kernels/rms_norm_f16.cu`
   - `crates/gpt-oss-gpu/src/kernel_loader.rs`

2. Add the scoped model-runner call-site selection.
   Proposed file:
   - `crates/gpt-oss-model-runner/src/gpu_layer.rs`

Do not modify:

- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
- `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
- `crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs`
- server, Harmony, protocol, tokenizer, or runtime API surfaces

The intended promotion is a new scoped BF16-policy RMSNorm kernel variant, not
a direct modification of the existing `rms_norm_f16_kernel`. The existing
kernel remains the default for ordinary f16 RMSNorm calls. The scoped variant
should be selected only for GPT-OSS BF16 layer-0 pre-attention RMSNorm.

The source branch routes `cfg.layer_idx == 0` pre-attention RMSNorm through the
BF16 policy path. The scoped extraction should preserve that narrow behavior
and avoid touching MLP RMSNorm, final RMSNorm, fused residual RMSNorm, and
non-layer-0 attention RMSNorm. If a config/dtype gate is available at the call
site, use it so non-GPT-OSS or non-BF16 paths remain unchanged.

Implementation caution: the production kernel uses two shared-memory buffers
over `hidden_size`. The extracted launcher must allocate
`2 * hidden_size * sizeof(float)` shared memory, not only
`block_threads * sizeof(float)`, before any promotion review.

### Proof Artifacts

The supporting source artifacts are local `.live` proof artifacts and should
not be committed:

- `.live/runtime-forward-layer0-attn-norm-conventions-20260423/developer-message.runner-layer0-attn-rmsnorm-bf16-runtime-fix-status.json`
  records the policy change from `f16_input_f16_weight_f32_reduction_f16_output`
  to `bf16_input_bf16_weight_f32_reduction_bf16_output`, scoped to layer-0
  attention RMSNorm. It is partial: live-vs-official max diff improved but did
  not fully clear at that stage.
- `.live/runtime-forward-layer0-attn-norm-conventions-20260423/developer-message.runner-layer0-attn-rmsnorm-live-cuda-vs-authoritative-replay-status.json`
  shows final-token layer-0 attention RMSNorm matched, while earlier-token
  residual differences remained before the scalar fix.
- `.live/runtime-forward-layer0-attn-norm-conventions-20260423/developer-message.runner-layer0-attn-rmsnorm-cuda-single-lane-scalar-capture-status.json`
  is diagnostic-only evidence identifying the scalar delta at token 18, lane
  92. It justifies the scalar/reduction change but depends on debug capture
  plumbing that must not be promoted.
- `.live/runtime-forward-layer0-attn-norm-conventions-20260423/developer-message.runner-layer0-attn-rmsnorm-scalar-runtime-fix-status.json`
  supports the production candidate: the scoped scalar/reduction behavior
  reduced live-vs-official RMSNorm differences to tiny residuals with max
  `1.1920928955078125e-7`; token 18 lane 92 matched authoritative replay and
  official after the fix.
- `.live/runtime-forward-layer0-attn-norm-conventions-20260423/developer-message.runner-layer0-attn-rmsnorm-residual-lanes-k-causality-status.json`
  is diagnostic. It shows remaining grouped-K mismatch was not explained by
  the tiny RMSNorm residual lanes and involves separate K-helper policy work,
  which remains deferred.
- `.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json`
  is the milestone-level proof that the final-token proof path ultimately
  cleared through final norm and LM-head logits, but it does not by itself
  justify promoting every proof-only mechanism.

### Performance Risks

This candidate is performance-sensitive:

- It changes a default runtime CUDA path if routed into layer-0 attention
  RMSNorm.
- It replaces the existing block-strided local-sum plus halving reduction with
  a lane-order two-buffer pairwise/tree reduction over `hidden_size`.
- It uses `sqrtf` plus reciprocal instead of `rsqrtf`.
- It increases shared memory from `block_threads * sizeof(float)` to
  `2 * hidden_size * sizeof(float)` for the scoped variant.
- It adds BF16 round-trip conversion for inputs, weights, and output values.
- It touches every token for the scoped layer-0 attention RMSNorm call.

The scope is narrow enough to review, but promotion should require performance
guardrails before push or PR expansion.

### Required Validation Before Runtime Promotion

Minimum validation for the kernel/loader commit:

- `git diff --check`
- `cargo check -p gpt-oss-gpu --features cuda`
- PTX symbol confirmation for `rms_norm_f16_bf16_policy_kernel`
- A small deterministic CPU/GPU comparison for the new BF16 policy kernel, if a
  suitable harness exists or can be added without debug capture plumbing

Minimum validation after call-site selection:

- `git diff --check`
- `cargo check -p gpt-oss-bench --lib`
- `cargo check -p gpt-oss-gpu --features cuda`
- A targeted runtime smoke for `developer-message-user-smoke` if available
- A microbenchmark or before/after timing for the scoped layer-0 attention
  RMSNorm path; if no benchmark exists, add or run one before pushing this
  candidate for review

Do not use the 45k-line runtime-forward proof bench as the promotion harness.
Do not require long GPU proof modes unless explicitly approved.

### Go/No-Go Recommendation

Go for scoped extraction only if the next slice keeps the change to the
production kernel symbol, loader registration, and layer-0 model-runner call
site. Stop if extraction pulls in debug runner APIs, engine worker debug APIs,
the full proof bench, Q/K/V projection candidates, selected expert readout
corrections, or cuBLAS/pedantic helper changes.

The next bounded prompt should extract only commit 1: production BF16 RMSNorm
kernel variant plus loader registration, then run CUDA compile validation and
inspect symbols. The model-runner call-site selection should remain a separate
follow-up commit.

### Commit 1 Status

Commit 1 extracts only the production `rms_norm_f16_bf16_policy_kernel` symbol
and registers it with the kernel loader. No model-runner call site routes to the
new kernel yet. The debug scalar-capture kernel and all debug runner/worker
plumbing remain excluded.

The next commit, if approved, should route only the GPT-OSS layer-0 attention
RMSNorm call site to this BF16 policy variant. Performance guardrails remain
required before broader promotion.
