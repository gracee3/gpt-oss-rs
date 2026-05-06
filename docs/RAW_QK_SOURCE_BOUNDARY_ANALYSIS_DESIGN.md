# Raw-QK Source Boundary Analysis Design

Classification: `raw_qk_source_boundary_analysis_design_recorded`

## Scope

- Docs-only design.
- Validation-only source/artifact analysis.
- Target operator: final-token attention raw scaled QK before mask.
- Target layers: 7, 17, and 23.
- Partial-success contrast: layer21.
- No implementation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission or ladder continuation is authorized.

## Source Evidence

Docs:

- `docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md`
- `docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md`
- `docs/LAYER0_VALIDATION_RUNTIME_HANDOFF_PLAN.md`
- `docs/LAYER0_VALIDATION_RUNTIME_PATH_PLAN.md`

Layer7 statuses:

- `/tmp/layer7_ordered_consumer_surface_status.json`
- `/tmp/layer7_raw_qk_qhead50_col57_dtype_probe_status.json`
- `/tmp/layer7_raw_qk_policy_sweep_status.json`

Layer17 statuses:

- `/tmp/layer17_ordered_consumer_surface_status.json`
- `/tmp/layer17_raw_qk_qhead35_col65_dtype_probe_status.json`
- `/tmp/layer17_raw_qk_policy_sweep_status.json`

Layer21 statuses:

- `/tmp/layer21_ordered_consumer_surface_status.json`
- `/tmp/layer21_raw_qk_qhead52_col55_dtype_probe_status.json`
- `/tmp/layer21_raw_qk_policy_sweep_status.json`
- `/tmp/layer21_ordered_bundle_validate_raw_qk_policy_status.json`

Layer23 statuses:

- `/tmp/layer23_ordered_consumer_surface_status.json`
- `/tmp/layer23_raw_qk_qhead33_col27_dtype_probe_status.json`
- `/tmp/layer23_raw_qk_policy_sweep_status.json`

Batch statuses:

- `/tmp/ordered_surface_batch_probe_status.json`
- `/tmp/raw_qk_dtype_probes_17_21_23_status.json`
- `/tmp/ordered_surface_batch_probe_17_21_23_raw_qk_status.json`

## Evidence Matrix

| Layer | Focus | Dtype probe classification | Sweep result | Revalidation result | Classification |
| --- | --- | --- | --- | --- | --- |
| 7 | q_head 50 / key col 57 | artifact precision/source boundary | collateral mismatches; no full-matrix clear | not run | blocked: artifact/source boundary |
| 17 | q_head 35 / key col 65 | accumulation boundary | collateral mismatches; no full-matrix clear | not run | blocked: accumulation boundary with collateral |
| 21 | q_head 52 / key col 55 | accumulation boundary | reverse clears full raw-QK and masked logits | revalidation stops at o-proj | partial: raw-QK solved, next seam o-proj |
| 23 | q_head 33 / key col 27 | artifact precision/source boundary | no candidate clears | not run | blocked: artifact/source boundary |

Negative and contrast rows:

| Layer | Result | Note |
| --- | --- | --- |
| 3 | raw-QK pairwise/reverse full-matrix clear | known valid accumulation-policy contrast |
| 4 | raw-QK strict/default exact | negative control |
| 5 | raw-QK strict/default exact | negative control |
| 6 | raw-QK strict/default exact | negative control |
| 12/14/15/22 | strict/default cleared layers | preserve as negative controls |

## Problem Statement

Raw-QK no longer looks like a single accumulation-order issue. It splits into:

- accumulation-boundary cases where a policy may clear;
- collateral cases where focus-entry or dtype evidence does not become a safe
  full-matrix policy;
- artifact/source precision cases where bounded policies do not explain the
  official artifact.

Layer21 proves the pipeline works when a safe full-matrix raw-QK policy exists.
Layers7/17/23 show why focus-lane evidence is insufficient.

## Key Questions

1. What distinguishes layer21 from layer17, since both have
   accumulation-boundary dtype probes but only layer21 gets a safe full-matrix
   policy?
2. Do layer7 and layer23 share a source/artifact generation pattern distinct
   from accumulation-boundary cases?
3. Are the official raw-QK artifacts for layer7/layer23 produced by a different
   API path, dtype boundary, layout, or source tensor path?
4. Is there a producer/API matrix analogous to the o-proj fused-linear/addmm
   probe that can explain raw-QK artifact precision cases?
5. Can we design a raw-QK source discriminator that avoids blind sweeps and
   rejects focus-lane-only policies?
6. Should layer17 be treated as accumulation-policy collateral or as a
   source/producer mismatch that only appears as accumulation-boundary at the
   focus entry?

## Hypotheses

### Hypothesis A - Accumulation policy class

Layer21 is the clean positive case: reverse f32 scale-after-sum BF16 output
clears the full raw-QK/masked-logit matrix, then exposes o-proj.

### Hypothesis B - Focus-entry accumulation evidence is insufficient

Layer17's dtype probe may show pairwise/f64 focus agreement, but the full matrix
has collateral mismatches. Therefore dtype-probe focus evidence must be only a
prerequisite, not a policy-selection criterion.

### Hypothesis C - Artifact/source precision boundary

Layers7 and 23 may reflect source-artifact precision or producer-expression
differences rather than a correctable local accumulation policy.

### Hypothesis D - Official producer raw-QK API class

The official raw-QK artifact may be produced by a batched/einsum/matmul
expression whose full-matrix behavior differs from isolated dot products or
local replay variants. This should be tested by a producer/API design before
more consumer sweeps.

## Proposed Discriminator Design

This is conceptual only. Do not implement it in this branch.

The discriminator should compare per target layer:

1. Source metadata:
   - Q post-RoPE dtype, shape, stride, and contiguity.
   - Grouped K post-RoPE dtype, shape, stride, and contiguity.
   - Raw-QK output dtype and shape.
   - Scale.
   - q_head / kv_head mapping.
   - Key column source: real token vs sink.
   - Source status paths.
   - Full-vector digests where available.
2. Producer/API variants:
   - official full expression.
   - isolated dot for focus entry.
   - matmul/einsum equivalent.
   - batched raw-QK expression if available.
   - elementwise product sum.
   - scale-before vs scale-after variants.
   - BF16 output-cast boundary.
   - repeated-expression determinism.
3. Consumer variants:
   - current sequential f32 scale-after-sum BF16 output.
   - reverse f32 scale-after-sum BF16 output.
   - pairwise f32 scale-after-sum BF16 output.
   - f64 diagnostic.
   - scale-per-term.
   - deterministic abs-ascending.
   - BF16-product evidence guard.
4. Full-matrix classification:
   - full raw-QK clear.
   - full masked-logit clear.
   - attention probabilities unchanged.
   - focus-only clear.
   - collateral mismatches.
   - artifact/source boundary suspected.
   - producer/API mismatch suspected.

## Target Layers for Future Producer/API Probe

Recommended minimal future probe set:

1. Layer23 artifact precision boundary.
2. Layer17 accumulation-boundary collateral case.
3. Layer21 raw-QK clear positive control.

Optional: layer7 artifact precision boundary, if old artifacts are still local
and complete.

Reason:

- Layer23 represents the current artifact/source boundary in 16..23.
- Layer17 represents accumulation-boundary-but-collateral.
- Layer21 is the positive policy-clear control.
- Layer7 can be historical context but may not need rerun if provenance is
  sufficient.

Suggested future branch:

```text
oracle/raw-qk-producer-api-probes-23-17-21
```

This branch does not authorize that probe.

## Status JSON Contract

Future status shape:

```json
{
  "classification": "raw_qk_source_boundary_analysis_recorded",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "operator": "attention_raw_qk",
  "source_matrix": "docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md",
  "layers": [
    {
      "layer_index": 23,
      "class": "artifact_precision_boundary",
      "q_head": 33,
      "key_column": 27,
      "dtype_probe_status": "/tmp/layer23_raw_qk_qhead33_col27_dtype_probe_status.json",
      "sweep_status": "/tmp/layer23_raw_qk_policy_sweep_status.json",
      "producer_api_probe_status": null,
      "full_raw_qk_cleared": false,
      "full_masked_logits_cleared": false,
      "focus_lane_only": false,
      "collateral_mismatches": true,
      "recommended_next_step": "producer_api_probe"
    }
  ],
  "output_emitted": false,
  "ladder_continued": false,
  "correction_metadata_applied": false,
  "tolerance_pass": false,
  "final_logit_claim": false,
  "all_layer_claim": false,
  "server_claim": false,
  "context_length_claim": false
}
```

## Classification Vocabulary

Design classifications:

- `raw_qk_source_boundary_analysis_design_recorded`
- `raw_qk_source_boundary_analysis_ready`
- `raw_qk_source_boundary_analysis_blocked_by_missing_status`
- `raw_qk_source_boundary_analysis_execution_failed`

Future per-layer classifications:

- `layerN_raw_qk_source_boundary_artifact_precision`
- `layerN_raw_qk_source_boundary_accumulation_collateral`
- `layerN_raw_qk_source_boundary_full_matrix_clear`
- `layerN_raw_qk_source_boundary_focus_only_rejected`
- `layerN_raw_qk_source_boundary_producer_api_recommended`
- `layerN_raw_qk_source_boundary_source_access_blocked`

## Proof Gates Before Any Implementation Or Policy Promotion

Require:

1. Source/dtype probe status exists.
2. Full raw-QK matrix clears, not just focus entry.
3. Full masked-logit matrix clears.
4. Attention probabilities have no collateral.
5. BF16-product remains evidence-only/rejected if broad mismatches appear.
6. f64 diagnostic remains diagnostic.
7. No focus-lane-only promotion.
8. No runtime/default/CUDA behavior change.
9. No output emission or ladder continuation.
10. Negative controls preserved.

## Recommended Immediate Next Step

Docs-only raw-QK source boundary design is enough for now.

Next executable branch only if separately approved:

```text
oracle/raw-qk-producer-api-probes-23-17-21
```

This should mirror the successful o-proj approach: use a minimal producer/API
probe set with a blocked artifact/source case, a collateral accumulation case,
and a positive clear control before any implementation or more sweeps.

## Non-Goals

- No runtime implementation.
- No default routing change.
- No CUDA kernel change.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency in Rust.
- No raw artifact commits.
