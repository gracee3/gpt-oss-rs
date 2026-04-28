# Layer0 Validation Runtime Path Plan

This plan turns the scratch layer0 downstream-equivalence result into a clean,
validation-only Rust/CUDA path. It does not promote a production route and does
not change default model-runner behavior.

## Current Evidence

The source scratch branch proved the exact `developer-message-user-smoke` layer0
final-token path using local artifact generation and oracle/model calls:

| Seam | Result | max abs diff | mismatches |
| --- | --- | ---: | ---: |
| Q/K raw QK | exact | `0.0` | `0` |
| V weighted sum | exact | `0.0` | `0` |
| Attention o-proj before residual | exact | `0.0` | `0` |
| Attention residual add before MLP | exact | `0.0` | `0` |
| MLP norm | exact | `0.0` | `0` |
| Router logits | exact | `0.0` | `0` |
| Top-k/routing | exact | `0.0` | `0` |
| Selected expert outputs | exact after fresh expert30 reconstruction | `0.0` | `0` |
| Weighted expert sum | exact | `0.0` | `0` |
| MLP residual / layer0 final-token output | exact | `0.0` | `0` |

The important correction was that the earlier Q/K raw-QK drift came from the
scratch RoPE generator. Once the official/model RoPE call was used, custom
decomposed Q/K reached the official raw-QK BF16 boundary exactly.

## Target Validation Runtime Path

The target is a validation-only bench path that runs layer0 for one exact case
under Rust-controlled CUDA execution and compares the final-token layer0 output
to the official oracle.

Input:

- Exact layer0 input/residual or token hidden states for
  `developer-message-user-smoke`.
- Layer0 model tensors needed for attention, MLP norm, router, selected experts,
  and output projections.
- Official layer0 final-token hidden state after MLP residual add, shape
  `[2880]`.

Operations:

1. Layer0 attention norm.
2. Q/K/V projections under the validation decomposed BF16 policy.
3. Runtime/kernel RoPE for Q/K, not scratch Python RoPE.
4. Raw QK, mask, and softmax for the final query token.
5. Weighted V.
6. Attention o-proj.
7. Attention residual add.
8. MLP norm.
9. Router logits, top-k, and routing weights.
10. Selected expert replay for the selected experts.
11. Weighted expert sum.
12. MLP residual add.

Output:

- Layer0 final-token hidden state after MLP residual add, shape `[2880]`.

Comparison:

- Compare against the official layer0 final-token hidden state after MLP
  residual add with max/mean absolute difference, mismatch count, and first/worst
  hidden lane.

## Reusable Pieces

- `crates/gpt-oss-gpu/src/cublas.rs`
  - Public BF16 GEMM APIs exist for current tensor-op and pedantic validation
    policies.
- `kernels/bf16_linear_bias_validation.cu`
  - Validation-only BF16 biased linear kernel for decomposed projection checks.
- `crates/gpt-oss-gpu/src/kernel_loader.rs`
  - Registers CUDA kernels, including RoPE kernels and the BF16 biased linear
    validation kernel.
- `crates/gpt-oss-bench/src/bin/qkv_projection_policy_compare.rs`
  - Contains flexible JSON artifact loading, BF16 conversion, comparison
    reporting, and policy-classification patterns that can be factored later.
- `crates/gpt-oss-model-runner/src/gpu_layer.rs`
  - Contains runtime layer pieces for RMSNorm, RoPE application, attention, and
    MoE execution.
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
  - Contains model-level tensor loading and RoPE table setup that should be
    reused carefully rather than duplicated.
- `docs/CUDA_BF16_LAYER0_DOWNSTREAM_EQUIVALENCE_SUMMARY.md`
  - Records the scratch equivalence result and caveats.

## Gaps

- RoPE table/source parity:
  scratch Python RoPE was wrong. The validation runtime path must use the
  runtime/kernel RoPE setup and explicitly guard it against official K pre/post
  RoPE before treating Q/K downstream evidence as valid.
- Model loading APIs:
  the bench path needs a narrow way to load only layer0 tensors without routing
  production model-runner execution through a validation policy.
- Custom projection policy API:
  the decomposed BF16 policy needs an explicit validation-only execution surface,
  separate from production projection routes.
- Artifact schema:
  official input/output artifacts and intermediate optional guards should use a
  small, stable JSON schema with comparable digests and no raw `.live` commits.
- Selected expert fresh replay:
  stale selected-output artifacts caused the expert30 false mismatch. The runtime
  path should recompute selected expert outputs directly.
- Scope control:
  layer ladder, final logits, performance work, and 4097 remain later milestones.

## Staged Implementation

1. Skeleton/status binary.
   - Parse explicit layer0 input and official output paths.
   - Emit the planned JSON schema and `runtime_behavior_changed = false`.
   - No CUDA execution and no model-runner routing.
2. Layer0 attention-only validation.
   - Load exact layer0 input and attention tensors.
   - Run attention norm, validation Q/K/V projections, runtime/kernel RoPE,
     raw-QK guard, weighted V, o-proj, and attention residual.
3. Layer0 MLP validation.
   - Run MLP norm, router/top-k, selected expert replay, and weighted expert sum
     from the attention residual boundary.
4. Full layer0 final-token validation.
   - Join attention and MLP stages and compare final layer0 output `[2880]` to
     official.
5. Layer ladder and final logits later.
   - Extend only after layer0 is clean, benchmarked, and integration-safe.

## Initial Skeleton Binary

`crates/gpt-oss-bench/src/bin/layer0_validation_runtime_path.rs` is a
non-executing status binary for this first slice. It validates that the explicit
input and official-output paths exist, then emits a JSON plan artifact.

It intentionally does not:

- launch CUDA kernels,
- load a model,
- call model-runner layer execution,
- alter production routing,
- import Torch/oneDNN/runtime-forward proof code, or
- write raw tensor artifacts.

## Validation Commands

For the skeleton slice:

```bash
cargo fmt --package gpt-oss-bench
cargo check -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda
cargo run -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda -- --help
git diff --check
```

For a future status run:

```bash
cargo run -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda -- \
  --layer0-input <layer0-input-json> \
  --official-layer0-output <official-layer0-output-json> \
  --output /tmp/layer0_validation_runtime_path_status.json
```

## Do-Not-Promote Boundaries

- No production runtime routing.
- No default model-runner behavior changes.
- No production CUDA kernel replacement.
- No oneDNN/Torch runtime dependency.
- No runtime-forward proof binary import.
- No debug capture plumbing import.
- No raw `.live` or `/tmp` artifacts committed.
- No all-layer, final-logit, or 4097 claims from this layer0-only path.
