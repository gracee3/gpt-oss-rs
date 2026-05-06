# Selected-MLP-Down Bundle Revalidation Design

Classification: `selected_mlp_down_bundle_revalidation_design_recorded`

## Scope

- Docs-only design.
- Validation-only full-bundle revalidation support.
- Target operator: selected expert MLP2/down projection output and downstream
  weighted sum/final MLP residual.
- Target layers: 11, 20, and 19.
- No implementation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission is authorized.
- No ladder continuation is authorized.

## Source Evidence

Docs:

- `docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md`
- `docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md`
- `docs/LAYER0_VALIDATION_RUNTIME_PATH_PLAN.md`
- `docs/LAYER0_VALIDATION_RUNTIME_HANDOFF_PLAN.md`

Layer11 statuses:

- `/tmp/layer11_ordered_consumer_surface_status.json`
- `/tmp/layer11_ordered_bundle_validate_status.json`
- `/tmp/layer11_selected_mlp_down_policy_replay_status.json`
- `/tmp/layer11_ordered_consumer_probe_status.json`

Layer11 historical focused chain, where present:

- `/tmp/layer11_ordered_mlp_consumer_compare_status.json`
- `/tmp/layer11_expert30_internal_consumer_compare_status.json`
- `/tmp/layer11_expert30_down_terms_consumer_compare_status.json`
- `/tmp/layer11_expert30_down_lane1480_einsum_dtype_probe_status.json`
- `/tmp/layer11_expert30_down_cast_policy_sweep_status.json`

Layer20 statuses:

- `/tmp/layer20_ordered_consumer_surface_status.json`
- `/tmp/layer20_attention_oproj_policy_sweep_status.json`
- `/tmp/layer20_ordered_bundle_validate_oproj_policy_status.json`
- `/tmp/layer20_selected_mlp_down_policy_replay_status.json`
- `/tmp/layer20_ordered_consumer_probe_status.json`

Layer19 statuses:

- `/tmp/layer19_ordered_consumer_surface_status.json`
- `/tmp/layer19_selected_mlp_down_policy_replay_status.json`
- `/tmp/layer19_ordered_consumer_probe_status.json`

Batch statuses:

- `/tmp/ordered_surface_batch_consumer_10_15_status.json`
- `/tmp/ordered_surface_batch_probe_10_15_status.json`
- `/tmp/ordered_surface_batch_consumer_16_23_status.json`
- `/tmp/ordered_surface_batch_probe_16_23_oproj_status.json`

## Evidence Matrix

| Layer | Upstream status | Strict/default first failure | Replay result | Full-bundle revalidation support | Classification |
| --- | --- | --- | --- | --- | --- |
| 11 | attention/bridge exact | selected MLP down lane 1480 | full MLP replay clears; selected policy recorded as replay evidence | missing selected-MLP-down bundle policy flag | support gap / likely clearable |
| 20 | o-proj pairwise clears first | selected MLP down after o-proj clear | full MLP replay clears; baseline has one selected-output mismatch | missing selected-MLP-down bundle policy flag | support gap after o-proj |
| 19 | attention/bridge exact | selected MLP down lane 3005 | replay has collateral mismatches; baseline has four selected-output mismatches | should not be forced | blocked/collateral |

Negative and contrast rows:

| Layer | Result | Note |
| --- | --- | --- |
| 12 | strict/default clear | no selected-MLP-down policy needed |
| 14 | strict/default clear | no selected-MLP-down policy needed |
| 15 | strict/default clear | no selected-MLP-down policy needed |
| 22 | strict/default clear | no selected-MLP-down policy needed |
| 4 | baseline exact and abs-ascending regresses | policy must not be global |

## Problem Statement

The replay mode can evaluate selected-MLP-down policies at the ordered MLP
level, but `layer-bundle-validate` cannot currently apply an explicit
selected-MLP-down policy during full bundle revalidation. That prevents
layer11 and layer20 from being resolved at the full-bundle level, while
layer19 shows why the future flag must be guarded against collateral
mismatches.

## Key Questions

1. Which replay-cleared selected-MLP-down policy should be allowed in full
   bundle validation?
2. Should the bundle validator allow multiple replay-proven policies, or only
   one selected policy per source status?
3. How should status JSON distinguish replay-cleared, full-bundle-cleared,
   support-gap, and collateral-blocked cases?
4. How should layer19 prevent accidental policy promotion?
5. Should the flag be validation-only and disabled by default?
6. Should the future mode consume replay status as provenance before accepting
   a policy override?

## Candidate Future CLI Design

Do not implement in this branch.

Loose form:

```text
--selected-mlp-down-policy current|naive-f64|pairwise-f64|pairwise-f32|deterministic-abs-ascending
--selected-mlp-down-policy-source-status /tmp/layerN_selected_mlp_down_policy_replay_status.json
```

Stricter form:

```text
--selected-mlp-down-policy-from-replay /tmp/layerN_selected_mlp_down_policy_replay_status.json
```

Recommended stricter design: prefer
`--selected-mlp-down-policy-from-replay` so the bundle validator can only use
a policy proven by that layer's replay status.

Required behavior:

- Default remains current/sequential.
- Explicit flag is required for any selected-MLP-down policy.
- Fail closed if replay status is missing.
- Fail closed if replay status has collateral mismatches for the requested
  policy.
- Emit selected policy and source status in JSON.
- Do not apply policy to attention, o-proj, raw-QK, weighted-V, or norm.
- Do not emit layer output unless full bundle validates exactly.

## Future Status Contract

Future full-bundle status fields:

```json
{
  "selected_mlp_down_policy": "naive_f64_sum_then_bf16_output",
  "selected_mlp_down_policy_source_status": "/tmp/layer11_selected_mlp_down_policy_replay_status.json",
  "selected_mlp_down_policy_applied": true,
  "selected_mlp_down_replay_classification": "layer11_selected_mlp_down_policy_replay_full_mlp_cleared",
  "selected_mlp_down_replay_had_collateral": false,
  "selected_outputs_mismatches": 0,
  "weighted_sum_mismatches": 0,
  "final_output_mismatches": 0,
  "full_bundle_cleared": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "output_emitted": false,
  "ladder_continued": false,
  "correction_metadata_applied": false,
  "tolerance_pass": false
}
```

Future batch/status classifications:

- `selected_mlp_down_bundle_revalidation_design_recorded`
- `selected_mlp_down_bundle_revalidation_ready`
- `selected_mlp_down_bundle_revalidation_blocked_by_missing_replay_status`
- `selected_mlp_down_bundle_revalidation_blocked_by_collateral_mismatches`
- `selected_mlp_down_bundle_revalidation_execution_failed`

Future per-layer classifications:

- `layer11_selected_mlp_down_bundle_revalidation_ready`
- `layer11_selected_mlp_down_bundle_revalidation_expected_clear`
- `layer20_selected_mlp_down_bundle_revalidation_ready_after_oproj_policy`
- `layer19_selected_mlp_down_bundle_revalidation_blocked_by_collateral`

## Layer-Specific Design Notes

### Layer11

- First failure: selected MLP down lane 1480.
- Replay clears full MLP.
- Full bundle revalidation is missing only because the flag is unsupported.
- Future design should revalidate layer11 first because it is the cleanest
  support-gap case.
- No output emission is authorized in this design branch.

Policy note: use whatever policy the current layer11 replay status declares as
best full-vector/full-MLP clearing. Do not hardcode a global policy without
reading the status.

### Layer20

- First failure was attention o-proj lane 2212.
- O-proj pairwise cleared the full o-proj seam.
- Full bundle revalidation then stopped at selected MLP down.
- Future revalidation must compose raw/default attention seams, o-proj
  pairwise policy source, and selected-MLP-down replay-proven policy source.
- This should be second after layer11 because it composes two explicit
  policies.

### Layer19

- First failure: selected MLP down lane 3005.
- Replay has collateral mismatches.
- No policy should be selected.
- Future implementation should classify this as collateral-blocked and should
  not revalidate with a policy.
- Layer19 is the negative control preventing overbroad selected-MLP-down policy
  application.

## Proof Gates Before Implementation

1. Accepted docs design.
2. Exact replay status inputs for target layers.
3. Status parser can identify full-MLP-clearing policy and collateral counts.
4. Layer11 revalidation clears full bundle under replay-proven policy.
5. Layer20 revalidation composes o-proj policy plus selected-MLP-down replay
   policy and either clears or localizes the next seam.
6. Layer19 remains blocked unless a no-collateral full-vector replay policy
   exists.
7. BF16-product remains evidence-only/rejected if collateral appears.
8. No default policy change.
9. No runtime/default/CUDA behavior change.
10. No output emission or ladder continuation.

## Proposed Future Implementation Slice

Only describe; do not authorize.

Future branch:

```text
validation/selected-mlp-down-bundle-revalidation
```

Scope:

- Add validation-only bundle revalidation flag.
- Consume replay status JSON.
- Apply selected-MLP-down policy only when replay status proves no collateral.
- Test layer11 first.
- Optionally test layer20 composition second.
- Verify layer19 remains blocked.
- No runtime/default/CUDA changes.
- No output/ladders/corrections/tolerance.

Validation for future implementation:

```text
cargo fmt --package gpt-oss-bench
cargo check -p gpt-oss-bench --features cuda
layer11 revalidation command
layer20 revalidation command if supported
layer19 negative guard/status
jq guard
git diff --check
```

## Recommended Next Step

Implement `validation/selected-mlp-down-bundle-revalidation` only after the
user explicitly approves code.

Alternative if avoiding code: create
`docs/raw-qk-source-boundary-analysis-design` for Workstream C.

Preferred next step: if the goal is to retire the support-gap class, authorize
the selected-MLP-down bundle revalidation code slice next.

## Non-Goals

- No runtime implementation.
- No production routing.
- No default behavior changes.
- No CUDA changes.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency.

## Implementation Status

Mode/flag:

```text
--mode layer-bundle-validate
--selected-mlp-down-policy-from-replay
```

Batch status:

```text
/tmp/selected_mlp_down_bundle_revalidation_status.json
```

Classification:

```text
selected_mlp_down_bundle_revalidation_partial
```

Recorded behavior:

- Layer11 consumes `/tmp/layer11_selected_mlp_down_policy_replay_status.json`
  and accepts `naive_f64_sum_then_bf16_output` from replay provenance. The
  selected-down policy clears selected outputs, weighted sum, and final output,
  but full bundle revalidation remains blocked by a one-lane router-logit seam.
- Layer20 composes the explicit attention o-proj pairwise policy with the
  replay-proven selected-MLP-down policy from
  `/tmp/layer20_selected_mlp_down_policy_replay_status.json`; the composed
  validation clears the full bundle with
  `pairwise_f32_sum_then_bf16_output`.
- Layer19 consumes `/tmp/layer19_selected_mlp_down_policy_replay_status.json`
  as a negative guard and rejects policy application when replay collateral is
  present.
- Selected policies are read from replay status provenance, not manually named.
- No output emission, ladder continuation, correction metadata, tolerance pass,
  runtime/default/CUDA change, final-logit, all-layer, server, or 4097-token
  claim is authorized.

## Layer11 Router-Logit Follow-Up

Source selected-MLP-down revalidation:

```text
/tmp/selected_mlp_down_bundle_revalidation_status.json
/tmp/layer11_ordered_bundle_validate_selected_mlp_down_policy_status.json
```

Router localization statuses:

```text
/tmp/layer11_router_logit_inspect_status.json
/tmp/layer11_router_logit_policy_debug_status.json
/tmp/layer11_router_logit_localization_status.json
```

Classification:

```text
layer11_router_logit_localization_recorded
```

Result:

- The remaining layer11 seam is one router-logit mismatch at expert 7:
  local `0.0693359375`, official `0.06884765625`, diff `0.00048828125`.
- The selected experts remain `[30, 13, 4, 20]`; selected logits and routing
  weights remain exact for the selected experts.
- The selected-MLP-down policy context remains
  `naive_f64_sum_then_bf16_output`; selected outputs, weighted sum, and final
  output clear under that policy.
- Router-logit debug classifies
  `pairwise_f32_bf16_bias_bf16_output` as a full-vector clear. Reverse and
  chunked-pairwise also clear the full router-logit vector; BF16 prebias and
  BF16-product variants have collateral mismatches.
- Full bundle revalidation with a router policy was not run because no narrow
  router-logit bundle policy flag exists in this slice.

Guardrails: no output emission, ladder continuation, correction metadata,
tolerance pass, runtime/default/CUDA change, final-logit, all-layer, server, or
4097-token claim.

## Layer11 Router-Logit Bundle Revalidation

Source statuses:

```text
/tmp/layer11_router_logit_localization_status.json
/tmp/layer11_router_logit_policy_debug_status.json
/tmp/layer11_selected_mlp_down_policy_replay_status.json
```

Revalidation status:

```text
/tmp/layer11_ordered_bundle_validate_router_selected_mlp_down_policy_status.json
```

Final status:

```text
/tmp/layer11_router_logit_bundle_revalidation_status.json
```

Classification:

```text
layer11_router_logit_bundle_revalidation_full_bundle_cleared
```

Result:

- The narrow validation-only router-logit policy flag consumes
  `/tmp/layer11_router_logit_policy_debug_status.json`.
- Selected router policy:
  `pairwise_f32_bf16_bias_bf16_output`.
- Selected-MLP-down policy:
  `naive_f64_sum_then_bf16_output`.
- Full bundle revalidation classifies
  `layer11_ordered_bundle_validate_attention_cleared_mlp_cleared_with_router_logit_selected_mlp_down_policy`.
- Router logits, selected logits, routing weights, selected outputs, weighted
  sum, and final output all report zero mismatches.
- Remaining blocker: none.

Guardrails: no output emission, ladder continuation, correction metadata,
tolerance pass, runtime/default/CUDA change, final-logit, all-layer, server, or
4097-token claim.
