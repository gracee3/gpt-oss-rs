# CUDA BF16 Projection Policy Design

Date: 2026-04-27

This document defines the design boundary for CUDA BF16 dense projection
promotion after the runtime-forward final-token oracle parity milestone.

Source proof branch: `feature/runtime-forward`

Source proof commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`

Current integration promotion scope: RoPE half-split pairing and scoped layer0
pre-attention BF16 RMSNorm only. Projection-policy mechanisms remain proof-only
until a separate CUDA strategy is designed and validated.

## Problem Statement

The official CPU oracle captures for GPT-OSS dense projections are governed by
PyTorch BF16 linear behavior, which on the observed proof machine used oneDNN /
MKLDNN BF16 matmul behavior. Runtime CUDA projection helpers use CUDA/cuBLAS
arithmetic and can differ from the official oracle at BF16 rounding-boundary
lanes.

The runtime-forward branch proved that oneDNN Q/K/V projection candidates match
the official target for the exact `developer-message-user-smoke` proof case.
Those candidates are CPU oracle references and proof tools, not CUDA runtime
fixes.

cuBLAS pedantic/no-tensor-op experiments were useful for localizing one K helper
mismatch, but they are not yet a production promotion strategy. They are
performance-sensitive and must not be applied globally to unrelated GEMMs.

The unresolved runtime design question is:

How should CUDA/runtime BF16 dense projections match the official PyTorch/oneDNN
BF16 projection oracle without globally slowing or perturbing unrelated GEMMs?

## Evidence Summary

Evidence was inspected from the runtime-forward `.live` status artifacts. These
paths are evidence references only; raw artifacts are not integration assets.

### K Projection

Relevant artifacts:

- `.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-onednn-oracle-helper-proof-status.json`
- `.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-onednn-oracle-scoped-helper-fix-status.json`
- `.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-pytorch-bf16-linear-backend-policy-status.json`
- `.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-onednn-primitive-reproduction-status.json`

Findings:

- Weight and bias arithmetic were cleared as the source of the K mismatch.
- The oneDNN K oracle matched official module/captured K with zero mismatching
  tokens and lanes.
- The legacy helper differed from the oneDNN oracle on six known
  rounding-boundary lanes.
- The scoped oneDNN K candidate was bench/proof-only but cleared the K
  pre-RoPE target and provided the K candidate used by downstream RoPE and raw
  QK score checks.

### Q Projection

Relevant artifacts:

- `.live/runtime-forward-layer0-q-provenance-20260423/developer-message.runner-layer0-q-projection-weight-bias-arithmetic-policy-status.json`
- `.live/runtime-forward-layer0-q-provenance-20260423/developer-message.runner-layer0-q-projection-onednn-oracle-scoped-candidate-status.json`

Findings:

- Weight and bias arithmetic were cleared before the Q policy candidate.
- The oneDNN Q oracle matched the official Q projection.
- The candidate Q projection cleared pre-RoPE Q, post-RoPE Q, and raw QK score
  when paired with the candidate K projection.
- The local runtime helper still differed from candidate/official Q in the
  proof artifact, confirming a projection arithmetic policy delta rather than a
  RoPE-only issue.

### V Projection

Relevant artifacts:

- `.live/runtime-forward-layer0-v-provenance-20260423/developer-message.runner-layer0-v-projection-weight-bias-arithmetic-policy-status.json`
- `.live/runtime-forward-layer0-v-provenance-20260423/developer-message.runner-layer0-v-projection-onednn-oracle-scoped-candidate-status.json`

Findings:

- Weight and bias arithmetic were cleared before the V policy candidate.
- The oneDNN V oracle matched the official V projection.
- The candidate V projection cleared the weighted V sum before `o_proj` when
  paired with already-cleared attention probabilities.
- The artifact explicitly marks the oneDNN V candidate as bench/proof-only.

### cuBLAS / Pedantic Helper Experiments

Relevant artifacts:

- `.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-all-token-k-gemm-math-mode-status.json`
- `.live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-all-token-k-pedantic-runtime-fix-status.json`

Findings:

- Default tensor-op BF16 GEMM for the all-token K helper differed from the CPU
  reference in the K helper experiment.
- A pedantic/no-tensor-op variant using `CUBLAS_PEDANTIC_MATH`,
  `CUBLAS_COMPUTE_32F_PEDANTIC`, and `CUBLAS_GEMM_DFALT` cleared the earlier
  K helper comparison.
- Later projection proof established the broader target as PyTorch/oneDNN BF16
  projection behavior, not "pedantic cuBLAS everywhere".
- cuBLAS promotion remains performance-sensitive and unresolved.

### Final Proof

Relevant artifact:

- `.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json`

The final-readout direct-module rerun classified as
`final_readout_direct_module_logits_cleared`, with final block output, final
norm, and LM-head logits all matched exactly. This proves the full proof path on
`feature/runtime-forward`; it does not by itself select a production CUDA BF16
projection policy.

## Runtime Requirements

A CUDA BF16 projection policy must satisfy:

- Exact BF16 output parity against the official oracle for relevant GPT-OSS
  dense projections.
- No broad slowdown of unrelated GEMMs.
- No global cuBLAS math-mode or atomics-mode changes by default.
- Deterministic behavior for restricted trace validation.
- Separate correctness and performance gates.
- Clear scope before implementation:
  - Q/K/V only or all attention projections?
  - Attention projections only or all dense BF16 projections?
  - Layer0 only, all layers, or validation-case-only?
  - Runtime production mode or validation-only mode?
- No CPU/oneDNN dependency in the CUDA runtime path.
- No proof-harness or debug-capture dependency in the production path.

## Non-Goals

- Do not ship oneDNN CPU projection as the runtime GPU path.
- Do not change Harmony, server, protocol, or tokenizer behavior.
- Do not import the runtime-forward proof harness into runtime.
- Do not commit raw `.live` full-value artifacts.
- Do not optimize or start 4097-boundary work.
- Do not globally force pedantic cuBLAS without correctness and performance
  proof.
- Do not route unrelated GEMMs through an oracle-sensitive policy by default.

## Candidate Designs

### A. Status Quo Plus Proof-Only Oracle

Keep CUDA runtime projection behavior unchanged. Use oneDNN/PyTorch proof paths
only as external oracle validation.

Pros:

- Zero runtime performance risk.
- Preserves current production GEMM behavior.
- Keeps proof-only CPU dependencies out of runtime.

Cons:

- Integration cannot natively reproduce the full final-token proof.
- Runtime CUDA projections may remain different from the official oracle on
  BF16 boundary lanes.
- Does not provide a CUDA path for validation or future production parity.

### B. Guarded cuBLAS Pedantic Projection Path

Add a scoped projection helper that uses `CUBLAS_PEDANTIC_MATH`,
`CUBLAS_COMPUTE_32F_PEDANTIC`, and `CUBLAS_GEMM_DFALT` only for explicitly
selected GPT-OSS BF16 projection calls.

Pros:

- Builds directly on the K helper experiment that cleared a local all-token K
  comparison.
- Can be guarded behind an explicit validation flag.
- Keeps the default tensor-op path available.

Cons:

- Not yet proven to reproduce full oneDNN/PyTorch Q/K/V policy in all relevant
  lanes.
- Likely slower than tensor-op GEMM.
- Requires strict cuBLAS math/atomics save/restore to avoid mode leakage.
- Risky if accidentally applied to all BF16 GEMMs.

### C. Custom CUDA BF16 Projection Kernel for Oracle-Sensitive Shapes

Implement a deterministic CUDA kernel with a reduction/blocking and BF16
rounding policy intended to match the official oneDNN/PyTorch oracle for known
projection shapes.

Pros:

- Full control over reduction order, rounding, bias application, and output
  conversion.
- Can be scoped to exact Q/K/V shapes and validation mode.
- Avoids global cuBLAS mode changes.

Cons:

- Highest implementation complexity.
- Significant performance risk versus cuBLAS tensor-op paths.
- Exact oneDNN boundary behavior may depend on blocking/thread partition
  details that are difficult to reproduce generally.
- Requires dedicated microbenchmarks and correctness fixtures before runtime
  use.

### D. Hybrid Validation Mode Only

Keep production runtime fast. Add an explicit validation-only policy path that
can route selected projections to an oracle-compatible CUDA policy for restricted
proof/smoke runs.

Pros:

- Preserves production performance by default.
- Provides integration-safe validation without importing oneDNN proof
  candidates or the full proof harness.
- Can accumulate correctness/performance evidence before any production route.

Cons:

- Does not immediately make default runtime projections oracle-identical.
- Requires a new validation flag/interface and tests.
- Still needs CUDA policy selection and benchmarking.

### E. Layer/Case-Scoped Candidate Path

Add a narrowly scoped candidate path for specific layer/case validation, similar
to the runtime-forward proof candidates, and explicitly keep it out of default
production routing.

Pros:

- Minimal blast radius.
- Useful for targeted layer0 validation and regression investigation.

Cons:

- Easy to overfit to `developer-message-user-smoke`.
- Not a general production design.
- Must be clearly labeled proof/validation-only to avoid accidental promotion.

## Recommended Design Direction

Start with a guarded validation-only CUDA projection policy path, not default
production routing.

The first implementation should be a validation feature or explicit runtime flag
that can select a candidate CUDA BF16 projection policy only for scoped GPT-OSS
Q/K/V projection calls. It should not change default GEMM behavior.

This direction is preferred because:

- The proof evidence identifies oneDNN/PyTorch BF16 projection behavior as the
  oracle target, but does not yet prove a production CUDA policy.
- cuBLAS pedantic/no-tensor-op cleared one K helper comparison but remains too
  broad and performance-sensitive for default routing.
- A validation-only path can collect correctness and latency evidence before
  any production decision.
- It keeps CPU oneDNN proof code, debug plumbing, and raw artifacts out of the
  runtime.

Production routing should be considered only after the guarded path proves Q/K/V
correctness and acceptable performance on the relevant shapes.

## Required Correctness Guardrails

Before any cuBLAS/projection promotion, add deterministic correctness checks for
the exact projection shapes:

- Q: `[74, 2880] x [4096, 2880]`
- K: `[74, 2880] x [512, 2880]`
- V: `[74, 2880] x [512, 2880]`

Each check should compare:

- Existing CUDA helper output.
- Candidate CUDA projection-policy output.
- oneDNN/PyTorch oracle output.

The guardrails must include:

- Known layer0 rounding-boundary lanes from the K proof.
- Q pre-RoPE and post-RoPE comparisons.
- K pre-RoPE and post-RoPE comparisons.
- Raw QK score comparison using candidate Q and K.
- Weighted V sum before `o_proj`.
- Final-readout digest comparison if the proof harness is available.
- Top-20/logit guard if full final-token proof replay is available.
- Explicit pass/fail status artifacts that are small enough for docs or
  manifest-only policy.

## Required Performance Guardrails

Before any production route, measure:

- Old CUDA projection latency versus candidate latency for Q/K/V shapes.
- Tensor-op cuBLAS versus pedantic/no-tensor-op cuBLAS.
- Batch/token-count sensitivity, including smaller and larger token counts than
  74.
- Layer-wide cumulative runtime estimate if applied beyond layer0.
- GPU memory overhead and temporary allocation pressure.
- cuBLAS math-mode and atomics-mode restoration after scoped calls.
- No global math-mode leakage into unrelated GEMMs.

Any benchmark must report whether the candidate affects only explicit validation
calls or default production calls.

## Integration Strategy

Commit 1: design doc only.

- Add this document.
- No runtime code.
- No proof harness.
- No `.live` artifacts.

Commit 2: microbench/correctness harness only.

- Add a small integration-safe harness for Q/K/V projection policy comparison.
- No runtime routing.
- No full runtime-forward proof binary import.
- No CPU oneDNN dependency in production runtime.

Commit 3: guarded projection-policy implementation.

- Add CUDA/cuBLAS candidate implementation behind an explicit validation flag.
- Scope to selected GPT-OSS BF16 Q/K/V projection calls.
- Include math/atomics save/restore if cuBLAS modes are changed.
- Keep default runtime behavior unchanged.

Commit 4: optional runtime routing.

- Only after correctness and performance evidence exists.
- Scope narrowly, document blast radius, and keep rollback simple.

## Open Questions

- Which CUDA policy can reproduce the oneDNN/PyTorch BF16 boundary cases across
  Q/K/V without overfitting to layer0?
- Is exact oneDNN parity required for all logits, or is top-k/logit-order parity
  sufficient for production?
- What performance cost is acceptable for validation mode and for production
  mode?
- Is validation-only projection policy sufficient before any 4097-boundary work?
- How should validation tooling avoid importing the runtime-forward proof
  harness and debug capture plumbing?
- Should the first candidate be cuBLAS pedantic/no-tensor-op or a custom CUDA
  deterministic kernel?
- Should scope begin at layer0 Q/K/V only, all attention projections, or all BF16
  dense projections?
