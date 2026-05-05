# Ordered Surface Batch Orchestration Design

## Classification

`ordered_surface_batch_orchestration_design_recorded`

## Scope

This design covers final-token ordered surface validation only for the
`developer-message-user-smoke` prompt/case. The intended first batch range is
layers 7..9 or 7..10, with a possible later expansion to layers 6..23 after
pilot review.

This is batch workflow orchestration only. It is not layer ladder
continuation, not layer output emission, not policy promotion, not a
runtime/default/CUDA behavior change, and not a final-logit, all-layer,
server, or 4097-token claim.

## Core Principle

Batch the workflow.
Do not batch the math.

The orchestration may batch artifact generation, validation commands, status
normalization, first-failure classification, and reporting. It must never
assume that raw-QK, weighted-V, o-proj, MLP down, RMSNorm, or linear backend
policy is globally stable. Every explicit policy must be proven by layer,
operator, and status before it is used for validation.

## Motivation

Layer2:

- Source-complete attention audit cleared.
- MLP down required deterministic abs-ascending.

Layer3:

- Raw-QK required pairwise/reverse.
- MLP baseline exact.

Layer4:

- O-proj required reverse.
- MLP baseline exact and abs-ascending regressed.

Layer5:

- Weighted-V required pairwise.
- O-proj required reverse.
- MLP down required deterministic abs-ascending.

Layer6:

- Audit, raw-QK, weighted-V, and MLP were exact.
- O-proj remained blocked.
- Local Rust/cuBLAS backend probes did not clear the full vector.
- The producer API probe shows fused-linear/addmm semantics plus
  layout/fused-bias sensitivity.

Conclusion: the workflow shape is stable, but numerical policy is not globally
stable.

## Non-Goals

- No production runtime routing.
- No default model-runner behavior changes.
- No CUDA kernel changes.
- No correction metadata.
- No tolerance pass.
- No layer output emission.
- No ladder continuation.
- No raw `/tmp` or `.live` commits.
- No Torch runtime dependency in Rust.
- No final-logit, all-layer, server, or 4097-token claim.

## Batch Pipeline Phases

The pipeline below is conceptual only. It records the workflow shape and status
contracts; it does not authorize implementation by itself.

### Phase 1 - Oracle Surface Generation

For each layer N:

- Generate ordered attention bundle.
- Generate attention audit bundle with all-token V.
- Generate ordered MLP bundle.
- Generate compact oracle surface status.

Expected paths:

```text
/tmp/layerN_ordered_attention_bundle_status.json
/tmp/layerN_ordered_attention_audit_bundle_status.json
/tmp/layerN_ordered_mlp_bundle_status.json
/tmp/layerN_ordered_surface_pilot_status.json
```

Failure handling:

- Missing schema: `layerN_oracle_surface_blocked_by_schema`.
- OOM/model-load: `layerN_oracle_surface_blocked_by_memory_or_model_load`.
- Generation failure: `layerN_oracle_surface_generation_failed`.
- Continue to the next layer only if infrastructure is stable and the failure
  is layer-local.

### Phase 2 - Consumer Strict Validation

For each layer N:

- Run attention audit with default/current policies first.
- Run strict/default ordered bundle validation only if audit clears or the
  validation is explicitly scoped.
- Run selected MLP down replay only if bridge/MLP input is exact.
- Emit compact consumer surface status.

Expected paths:

```text
/tmp/layerN_ordered_attention_audit_validate_status.json
/tmp/layerN_ordered_bundle_validate_status.json
/tmp/layerN_selected_mlp_down_policy_replay_status.json
/tmp/layerN_ordered_consumer_surface_status.json
```

### Phase 3 - First-Failure Classification

Normalize the first failing seam to one of:

- `none_strict_clear`
- `attention_audit_weighted_v`
- `attention_audit_residual`
- `attention_bridge`
- `raw_qk`
- `masked_logits`
- `attention_probs`
- `weighted_v`
- `attention_o_proj`
- `mlp_norm`
- `router_topk`
- `selected_mlp_down`
- `weighted_expert_sum`
- `final_mlp_residual`
- `schema_or_artifact`
- `model_load_or_memory`
- `unknown`

### Phase 4 - Conditional Probe Dispatch

Only run probes for the seam that actually fails.

Weighted-V audit mismatch:

- Run weighted-v-single-mismatch-debug.
- If a full-vector candidate clears, rerun audit with explicit weighted-V
  policy.
- Then rerun full bundle validation.

Raw-QK or masked-logit mismatch:

- Run raw-qk-policy-sweep.
- If the full matrix clears, rerun bundle with explicit raw-QK policy.

O-proj mismatch:

- Run attention-oproj-policy-sweep.
- If the full vector clears, rerun bundle with explicit o-proj policy.
- If no bounded backend clears, queue official-linear producer/API probe.

MLP down mismatch:

- Run selected-mlp-down-policy-replay.
- Record baseline, deterministic abs-ascending, BF16-product guard, and any
  collateral.

Official-linear blocked:

- Do not run blind backend sweeps repeatedly.
- Queue producer/API/source probe or fused-linear/addmm discriminator design.

### Phase 5 - Matrix/Report Generation

Emit:

```text
/tmp/ordered_surface_batch_status.json
```

Optional docs:

```text
docs/CROSS_LAYER_VALIDATION_POLICY_MATRIX_AUTO.md
```

Do not auto-commit raw artifacts.

## Normalized Per-Layer Status Contract

Minimum fields:

```json
{
  "layer_index": 7,
  "classification": "...",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "oracle_generation": {
    "attention_status": "...",
    "attention_audit_status": "...",
    "mlp_status": "...",
    "surface_status": "..."
  },
  "consumer_validation": {
    "attention_audit_status": "...",
    "ordered_bundle_validate_status": "...",
    "selected_mlp_down_replay_status": "..."
  },
  "first_failing_operator": "...",
  "first_failing_lane": null,
  "recommended_probe": "...",
  "explicit_policies_used": {},
  "strict_default_cleared": false,
  "explicit_policy_cleared": false,
  "output_emitted": false,
  "ladder_continued": false,
  "correction_metadata_applied": false,
  "tolerance_pass": false,
  "final_logit_claim": false,
  "all_layer_claim": false,
  "server_claim": false,
  "context_length_claim": false,
  "next_bounded_step": "..."
}
```

## Batch Summary Matrix Columns

Required columns:

- layer
- oracle attention generated
- audit generated
- MLP generated
- attention audit result
- strict/default bundle result
- selected MLP down replay result
- first failing operator
- raw-QK policy
- weighted-V policy
- o-proj policy
- MLP down policy
- producer/API probe needed
- strict/default cleared
- explicit policy cleared
- output emitted
- ladder continued
- next action

## CLI Design, Conceptual Only

Oracle:

```text
ordered-surface-batch-generate --layers 7..9 --model ... --case developer-message-user-smoke
```

Consumer classify-only:

```text
ordered-surface-batch-validate --layers 7..9 --strict-first --classify-only
```

Consumer bounded probes:

```text
ordered-surface-batch-validate --layers 7..9 --strict-first --run-bounded-probes
```

Producer API probes:

```text
ordered-surface-batch-validate --layers 7..9 --queue-producer-api-probes
```

Implementation is not authorized by this doc alone.

## Pilot Recommendation

- Do not start with 6..23.
- First pilot layers 7..9 or 7..10.
- Classify-only first.
- Add bounded probes only after classify-only status is stable.
- Producer API probes remain manual/explicit until enough o-proj failures
  justify batching them.

## Parallelism and Failure Policy

- Independent oracle generation can be parallelized only if GPU/memory budget
  is explicit.
- Consumer validation should default to sequential per layer until artifacts
  and statuses are stable.
- One layer-local failure should not stop the full batch.
- Infrastructure failure should stop the batch.
- Every per-layer failure must preserve a status JSON when possible.

Infrastructure failures:

- Model load failure.
- OOM.
- Corrupted artifact.
- Schema version mismatch.
- Missing common prompt/case source.

Layer-local failures:

- Weighted-V mismatch.
- Raw-QK mismatch.
- O-proj mismatch.
- MLP down mismatch.

## Guardrails

Every status must record:

```text
runtime_behavior_changed = false
production_routing_changed = false
cuda_kernels_changed = false
output_emitted = false
ladder_continued = false
correction_metadata_applied = false
tolerance_pass = false
final_logit_claim = false
all_layer_claim = false
server_claim = false
context_length_claim = false
```

## Recommended Branch Sequence

1. `design/official-linear-backend-discriminator`

   - Update with Stage 3 producer/API result.

2. `design/ordered-surface-batch-orchestration`

   - This doc.

3. `validation/ordered-surface-batch-status`

   - Implementation Stage 1: consume existing layer2..6 statuses and emit
     normalized matrix only.

4. `oracle/ordered-surface-batch-generation`

   - Implementation Stage 2: generate artifacts for pilot layer range 7..9.

5. `validation/ordered-surface-batch-consumer`

   - Implementation Stage 3: strict/default validate pilot range and classify
     first failing seam.

6. Expand layer range only after pilot review.

## Caveats

- Final-token only.
- One prompt/case.
- Ordered bundle/audit surfaces only.
- No server path.
- No generated outputs promoted.
- No final logits.
- No all-layer parity.
- No 4097.
- No production/server parity.
