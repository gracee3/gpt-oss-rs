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

Layer2 note: the layer2 ordered MLP replay was originally MLP-only and seeded
from an official/coarse attention residual seam. A later ordered attention audit
proof gate now validates the layer2 final-token attention path through weighted
V and residual add; see the next section. This strengthens the layer2 proof
surface but does not turn the selected MLP down convention into a production
policy.

## Layer2 Ordered Attention Audit Proof Gate

The prior preferred layer2 proof gate has now been partially satisfied for the
final-token ordered validation surface. The validation-runtime consumer has
source-complete layer2 ordered attention evidence for the final-token path,
including all-token K post-RoPE history and a supplemental all-token V audit
bundle.

Relevant statuses:

```text
/tmp/layer2_ordered_attention_bundle_status.json
/tmp/layer2_ordered_attention_audit_bundle_status.json
/tmp/layer2_ordered_attention_audit_validate_status.json
/tmp/layer2_ordered_bundle_validate_status.json
/tmp/layer2_ordered_mlp_bundle_status.json
/tmp/layer2_selected_mlp_down_policy_replay_status.json
```

Classifications:

```text
layer2_ordered_bundle_validate_attention_cleared_mlp_cleared
layer2_ordered_attention_audit_weighted_v_and_residual_cleared
layer2_selected_mlp_down_policy_replay_full_mlp_cleared
```

The audit bundle emits all-token V as:

```text
shape = [74, 8, 64]
layout = all-real-token V projection tensor [token, kv_head, head_dim]
```

The consumer recomputed:

- weighted V from attention probabilities plus all-token V, matching exactly
- attention residual from layer input plus o_proj, matching exactly
- attention residual to ordered MLP input bridge, still exact

This removes the earlier layer2 consumer caveats that weighted V was only an
official seam and that attention residual was only bridge-checked. Remaining
limits are still material: this is final-token ordered validation only; it does
not continue the ladder, prove final logits, prove all layers, prove
server/runtime parity, or make any 4097-token claim.

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
- The layer2 ordered attention audit strengthens the case for doing this in a
  separate validation branch, because layer2 now has ordered attention plus
  ordered MLP evidence through the final-token MLP output.

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

- completed: ordered MLP replay on layer11, layer1, and layer2
- completed for layer2 final-token ordered validation: ordered MLP surface
- completed for layer2 final-token ordered validation: source-complete ordered
  attention capture
- completed for layer2 final-token ordered validation: all-token V audit
- completed for layer2 final-token ordered validation: weighted-V recompute
- completed for layer2 final-token ordered validation: residual-add recompute
- completed for layer2 final-token ordered validation: attention-to-MLP bridge
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
remain explicitly disabled until the remaining proof gates above are satisfied.

## Implementation Branch Scaffold

The first validation-only implementation branch is:

```text
feature/selected-mlp-down-policy-validation
```

The first slice on that branch is intentionally status/scaffold-only. It
centralizes the candidate policy registry for validation reporting and adds:

```text
--mode selected-mlp-down-policy-candidate-status
```

This mode records the candidate policy, the rejected evidence-only policy, and
the current proof-gate status. It may inspect local `/tmp` proof statuses when
they are present, but missing local artifacts do not fail the mode because the
committed design and handoff docs remain the provenance source.

This scaffold does not implement production runtime behavior, change default
model-runner routing, change CUDA kernels, apply correction metadata, continue
the ladder, or make final-logit, all-layer, server, or 4097-token claims.
Runtime performance remains unassessed.

The next validation-only slice adds a reusable replay/status mode:

```text
--mode selected-mlp-down-policy-candidate-replay-status
```

This mode runs the centralized selected MLP down-policy registry against
available ordered MLP surfaces, currently layer1 and layer2, with layer11 used
only when the local ordered bundle is present. It compares selected expert
outputs, weighted expert sum, and final MLP residual output for the baseline,
the deterministic absolute-ascending candidate, and the rejected BF16-product
evidence policy. The mode is still a proof harness only: runtime policy
discussion remains disabled, production/default routing and CUDA kernels remain
unchanged, no correction metadata is applied, and performance is still
unassessed.

The same replay mode also supports a deterministic multi-lane smoke summary
using:

```text
--focus-lanes 0,1,2,127,248,522,1024,1480,1990,2108,2269,2879
```

The lane list includes edge lanes, historical localization lanes, and a small
mid-vector checkpoint set. The smoke report is intentionally readable evidence
only: the full-vector selected-output, weighted-sum, and final-output metrics
remain authoritative. On the available local surfaces, layer1 and layer2 clear
with the deterministic absolute-ascending candidate across both the full vector
and the requested focus lanes. Layer11 remains docs-only provenance unless its
local ordered bundle is present. Runtime policy discussion remains disabled and
performance remains unassessed.

## Cost/Performance Characterization

The validation branch also records a bounded cost report at:

```text
/tmp/selected_mlp_down_policy_cost_status.json
```

The report measures the same available ordered MLP surfaces, layer1 and layer2,
with layer11 included only when its local artifact exists. It keeps correctness
and cost separate: the deterministic absolute-ascending candidate still clears
selected outputs, weighted sum, and final output on the available surfaces, but
the cost profile is materially different from current sequential f32 replay.

The current sequential replay is O(selected_experts * outputs * inputs). The
deterministic absolute-ascending candidate is O(selected_experts * outputs *
inputs log inputs) when implemented as a per-output product sort. In the current
validation helper this also allocates a temporary product buffer per output
lane. The timings are debug-mode, per-policy replay timings and are only
directional; they exclude cargo compile time and are not a production
performance benchmark.

The BF16-product policy remains evidence-only and rejected regardless of speed
because it introduces broad collateral mismatches. The cost report does not
authorize runtime implementation, production/default routing changes, CUDA
kernel changes, correction metadata, ladder continuation, or any final-logit,
all-layer, server, or 4097-token claim.

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
