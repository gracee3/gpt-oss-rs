# Selected MLP Down Policy Runtime Design

## Purpose

This document evaluates whether the validation-only selected MLP
down-projection reduction/output-cast convention could become a scoped runtime
candidate later.

This is design only. It does not implement runtime behavior, change CUDA
kernels, change default model-runner routing, apply correction metadata, or make
final-logit, all-layer, server, or 4097-token claims.

## Evidence Summary

Three ordered MLP proof surfaces currently support discussing a scoped candidate
for selected MLP down-projection replay:

| Layer | Classification | Result |
| --- | --- | --- |
| 11 | `layer11_selected_mlp_down_policy_replay_full_mlp_cleared` | `deterministic_f32_abs_ascending_sum_then_bf16_output` clears selected outputs, weighted sum, and final output. |
| 1 | `layer1_selected_mlp_down_policy_replay_full_mlp_cleared` | `deterministic_f32_abs_ascending_sum_then_bf16_output` clears selected outputs, weighted sum, and final output. |
| 2 | `layer2_selected_mlp_down_policy_replay_full_mlp_cleared` | `deterministic_f32_abs_ascending_sum_then_bf16_output` clears selected outputs, weighted sum, and final output. |

Layer2 caveat: the layer2 ordered MLP bundle is MLP-only. Its seed is the
official/coarse attention residual seam, so this is not source-complete layer2
attention validation.

## Candidate Convention

Candidate:

```text
deterministic_f32_abs_ascending_sum_then_bf16_output
```

Precise validation convention:

1. Use the selected expert SwiGLU/down-projection source values at the BF16
   boundary used by the validation replay.
2. Use the BF16/dequantized down weights as currently represented by the
   validation loader.
3. Form products in `f32`.
4. Sort the products by ascending absolute product magnitude using a
   deterministic order.
5. Accumulate the sorted products in `f32`.
6. Round the down-projection output to the final BF16 output boundary.
7. Continue selected-output, weighted expert sum, and MLP residual comparisons
   against ordered MLP oracle evidence.

This is a validation candidate, not a production runtime policy. The reduction
order is deliberately deterministic and evidence-oriented; it may have
performance and implementation costs that are unacceptable for production.

## Weaker Candidates

The current sequential `f32` accumulator reproduces the known one-lane selected
MLP mismatches in the ordered replay surfaces.

These candidates clear layer11 but do not clear layer1 or layer2:

```text
naive_f64_sum_then_bf16_output
pairwise_f64_sum_then_bf16_output
pairwise_f32_sum_then_bf16_output
```

Rejected evidence-only policy:

```text
bf16_product_then_f32_sum_then_bf16_output
```

Reason rejected: it repeatedly clears some focus lanes but introduces broad
collateral mismatches.

| Replay surface | Selected-output mismatches | Weighted-sum mismatches | Final-output mismatches |
| --- | ---: | ---: | ---: |
| Layer11 selected MLP replay | 3473 | 1089 | 444 |
| Layer1 selected MLP replay | 3133 | 952 | 475 |
| Layer2 selected MLP replay | 3371 | 919 | 416 |

The BF16-product policy must remain evidence-only and must not be promoted as a
correction candidate.

## Scope Boundaries

The candidate applies only to selected MLP down-projection replay.

It does not:

- solve attention
- solve source-complete K/V history
- prove all layers
- prove final logits
- prove server/runtime parity
- change production routing
- change default model-runner behavior
- change CUDA kernels
- add a Torch dependency to Rust
- apply correction metadata

## Runtime-Design Options

### Option A: Keep CPU/Rust Validation-Only Harness

Keep the convention as a validation-only proof harness.

Pros:

- Safest option.
- Preserves the evidence without changing runtime behavior.
- Keeps the candidate available for additional ordered MLP proof surfaces.

Cons:

- No production benefit.
- Does not answer runtime performance or implementation questions.

### Option B: Add Validation-Only CUDA/Rust Helper Behind Bench Flags

Introduce a validation-only helper behind explicit benchmark flags.

Pros:

- Still not production behavior.
- Can test implementation shape and performance.
- Useful for broader selected-expert validation across more layers and lanes.

Cons:

- Adds implementation surface.
- Still cannot be treated as a runtime fix.
- Must avoid default model-runner routing changes.

### Option C: Production Candidate Behind Disabled Feature Flag

Introduce a production runtime candidate behind an explicit feature flag that is
disabled by default.

Requirements:

- Must not be enabled by default.
- Must not change default model-runner behavior.
- Requires broader proof gates before use.
- Requires performance assessment.
- Requires multi-layer non-regression.

Risk:

- The deterministic absolute-ascending reduction order may be expensive.
- The current evidence is final-token ordered MLP evidence, not source-complete
  end-to-end runtime parity.

### Option D: Wait for More Source-Complete Ordered Bundles

Do nothing until source-complete ordered attention and MLP bundles exist for
more layers.

Pros:

- Most conservative.
- Avoids overfitting to final-token ordered MLP surfaces.
- Keeps attention/KV gaps explicit.

Cons:

- Delays any runtime candidate work.
- Requires more oracle generation.

## Required Proof Gates Before Implementation

Before any implementation branch promotes this beyond design, require:

- replay on at least one more ordered MLP layer if available
- preferably source-complete layer2 attention plus MLP ordered bundles
- multiple focus lanes, not only lane 1480
- full selected-output comparison across all selected experts
- weighted expert sum comparison
- MLP residual comparison
- explicit no-collateral-mismatch checks
- preservation of the rejected BF16-product policy as evidence-only
- no raw `/tmp` or `.live` artifacts committed to git
- no production/default routing changes during validation
- performance assessment before any production candidate is considered

## Proposed Future Branch

If approved later, a separate implementation branch could be:

```text
feature/selected-mlp-down-policy-validation
```

That branch should begin as validation-only. Runtime policy discussion should
remain explicitly disabled until the proof gates above are satisfied.

## Non-Goals

- No runtime code change in this design slice.
- No CUDA kernel change.
- No default model-runner behavior change.
- No Torch dependency in Rust.
- No correction metadata.
- No final-logit claim.
- No all-layer claim.
- No server/runtime parity claim.
- No 4097-token work.
