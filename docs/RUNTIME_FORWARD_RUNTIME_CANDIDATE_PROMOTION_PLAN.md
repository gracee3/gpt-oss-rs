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
