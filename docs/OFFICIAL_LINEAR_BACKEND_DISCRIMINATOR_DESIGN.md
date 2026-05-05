# Official Linear Backend Discriminator Design

## Classification

`official_linear_backend_discriminator_design_recorded`

## Scope

This is a validation-only design focused on BF16 linear boundaries such as
attention o-proj, `attn.out`, and `torch.nn.functional.linear`. It is
motivated by the layer6 o-proj evidence where official live `attn.out` and
producer-side `F.linear` match the ordered o-proj artifact, while bounded
consumer accumulation variants and simpler matmul/einsum replay do not clear
the full vector.

This is not a production runtime design, not a default routing proposal, not a
CUDA kernel proposal, and not a Torch runtime dependency proposal. It does not
authorize layer output emission, ladder continuation, correction metadata,
tolerance-based pass criteria, or any final-logit, all-layer, server, or
4097-token claim.

## Motivation

Layer6 introduced an attention o-proj boundary that differs from the earlier
layer4/layer5 reverse-accumulation evidence.

Consumer statuses:

```text
/tmp/layer6_ordered_bundle_validate_status.json
/tmp/layer6_attention_oproj_policy_sweep_status.json
```

Producer status:

```text
/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json
```

Findings:

- Weighted V is exact.
- Raw-QK, masked logits, and attention probabilities are exact.
- MLP validation is exact.
- Strict/default o-proj has 2 mismatches.
- Focus lane `22`: local `9.125`, official `9.0625`, diff `0.0625`.
- Bounded consumer sweeps did not clear the full o-proj vector.
- Producer official `attn.out(weighted_flat)` matches the prior o-proj
  artifact.
- Producer `torch.nn.functional.linear` matches the prior o-proj artifact.
- Producer matmul/einsum full-vector replay has 826 mismatches.
- Weighted-V live versus prior artifact has 0 mismatches.

The evidence suggests the relevant boundary is the official BF16 linear
backend used by `attn.out` / `F.linear`, not a simple global accumulation-order
switch.

## Prior Related Evidence

- Layer0 o-proj previously cleared under a chunked-pairwise validation
  discriminator.
- Layer4 o-proj cleared under reverse f32 accumulation.
- Layer5 o-proj cleared under reverse f32 accumulation after the explicit
  weighted-V policy.
- Layer6 did not clear under current, reverse, pairwise, chunked-pairwise,
  f64 diagnostic, or bias variants without collateral mismatches.

Together, these surfaces rule out a single global o-proj accumulation switch.

## Discriminator Goal

The discriminator should answer:

1. Is official BF16 `F.linear` parity reproducible by an existing Rust/CUDA
   validation backend?
2. Does cuBLAS BF16 tensor-op reproduce the official layer6 o-proj artifact?
3. Does pedantic cuBLAS BF16 reproduce it?
4. Is the mismatch CPU-vs-GPU BF16 backend behavior?
5. Is the mismatch due to operator API choice: `F.linear` / module call vs
   matmul/einsum?
6. Is the issue full-vector or lane-local?
7. Which policy, if any, clears without collateral mismatches?

## Proposed Evidence Inputs

Layer6 primary inputs:

- Weighted-V artifact from the layer6 ordered attention bundle, or weighted-V
  recomputed from audit all-token V.
- Layer6 o-proj weight.
- Layer6 o-proj bias.
- Official layer6 o-proj artifact.
- Producer dtype probe status:
  `/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json`.

Optional comparison surfaces:

- Layer0 o-proj, where chunked-pairwise cleared.
- Layer4 o-proj, where reverse cleared.
- Layer5 o-proj, where reverse cleared after weighted-V policy.

## Proposed Candidate Backends

These are conceptual only:

- Current sequential Rust f32 accumulation.
- Reverse f32 accumulation.
- Pairwise f32 accumulation.
- Chunked-pairwise f32 accumulation.
- Existing cuBLAS BF16 tensor-op validation helper.
- Existing cuBLAS pedantic BF16 validation helper, if available.
- CPU BF16 reference replay, diagnostic only.
- Producer-side PyTorch BF16 `F.linear`, source proof only, not a Rust
  dependency.

BF16-product remains evidence-only/rejected unless used only as a negative
guard.

## Required Metrics

For every backend, report:

- o-proj full-vector max_abs_diff.
- mean_abs_diff.
- mismatch count.
- first mismatch.
- worst mismatch.
- focus lane `22` value.
- residual recompute result.
- attention-to-MLP bridge result.
- collateral mismatch count.
- whether the full vector clears.
- whether the result is diagnostic-only.

## Required Status JSON Contract

Minimum fields:

```text
classification
validation_only
runtime_behavior_changed
production_routing_changed
cuda_kernels_changed
layer_index
operator
selected_backend
source_statuses
official_backend_reference
backend_results
full_vector_metrics
focus_lane_metrics
collateral_mismatches
output_emitted
ladder_continued
final_logit_claim
all_layer_claim
server_claim
context_length_claim
next_bounded_step
```

## Expected Future Classifications

- `official_linear_backend_discriminator_design_recorded`
- `layer6_oproj_official_linear_backend_discriminator_ready`
- `layer6_oproj_cublas_tensor_op_matches_official`
- `layer6_oproj_cublas_pedantic_matches_official`
- `layer6_oproj_no_validation_backend_matches_official`
- `layer6_oproj_backend_discriminator_collateral_mismatches`
- `layer6_oproj_backend_discriminator_blocked_by_schema`
- `layer6_oproj_backend_discriminator_execution_failed`

## Guardrails

- Default behavior must remain current.
- No production runtime routing.
- No default model-runner routing.
- No CUDA kernel replacement.
- No correction metadata.
- No tolerance pass.
- No layer output emission.
- No ladder continuation.
- No raw `/tmp` or `.live` commits.
- No Torch runtime dependency in Rust.
- No final-logit, all-layer, server, or 4097-token claim.

## Proof Gates Before Code

1. Accepted design doc.
2. Exact command examples.
3. Schema for status JSON.
4. Source proof status for the layer6 producer probe.
5. Clear list of candidate backends.
6. Existing helper availability check.
7. Full-vector collateral metrics.
8. No-default-routing guarantee.
9. Rollback story.
10. Performance/release gate if a backend becomes a candidate.

## Suggested Future Implementation Order

1. Status-schema-only mode.
2. Artifact/source loader validation.
3. Current/reverse/pairwise/chunked replay reuse.
4. cuBLAS BF16 tensor-op validation backend probe.
5. cuBLAS pedantic validation backend probe.
6. Optional layer0/layer4/layer5 comparison mode.
7. Matrix summary update.

This design does not authorize implementation by itself.

## Recommendation

Do not generate layer7 yet unless the goal is pure evidence collection. Do not
open a runtime implementation branch. The next best step after this doc is a
small validation-only discriminator implementation branch, but only if it is
accepted separately.
