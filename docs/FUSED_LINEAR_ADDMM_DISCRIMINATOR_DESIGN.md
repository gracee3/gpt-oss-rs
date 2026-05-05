# Fused Linear AddMM Discriminator Design

## Classification

`fused_linear_addmm_discriminator_design_recorded`

## Scope

This is a docs-only design for a future validation-only fused-linear/addmm
discriminator for BF16 linear boundaries such as layer6 attention o-proj.

The design is motivated by producer-side evidence that official PyTorch
`attn.out` / `F.linear` parity is explained by fused linear/addmm semantics
with the original contiguous BF16 input, weight, and bias layout.

This document does not authorize Rust code changes, Python producer changes,
runtime/default routing changes, CUDA kernel changes, artifact generation,
probe execution, output emission, layer ladder continuation, correction
metadata, tolerance-based pass criteria, a Torch runtime dependency in Rust,
or final-logit, all-layer, server, or 4097-token claims.

## Stage 3 Evidence

Source status:

```text
/tmp/layer6_attention_oproj_api_probe_status.json
```

Source artifact directory:

```text
/tmp/layer6_attention_oproj_api_probe/
```

Classification:

```text
layer6_oproj_producer_api_probe_layout_sensitive
```

Stage 3 established that `F.linear` is not unique. These producer-side
operators clear the full layer6 o-proj vector:

- Module call: `attn.out(weighted_flat)`.
- `torch.nn.functional.linear`.
- `torch._C._nn.linear`.
- Fused `torch.addmm`.

These producer-side variants reproduce the 826-mismatch pattern:

- `weight @ input + bias`.
- `input @ weight.T + bias`.
- `torch.matmul`.
- `torch.einsum`.
- `F.linear(..., bias=None) + bias`.
- Stride-2 input view into `F.linear`.

MKLDNN enabled/disabled variants both cleared, and default-thread versus
single-thread variants both cleared. Stage 3 therefore did not identify
MKLDNN or thread count as the differentiator.

The key conclusion is that official layer6 o-proj is explained by fused
linear/addmm semantics plus original layout and fused-bias handling, not by a
plain explicit matmul/einsum replay.

## Why Explicit Matmul/Einsum Is Not Enough

The explicit matmul/einsum family clears focus lane 22, but it leaves 826
full-vector mismatches. Focus-lane clearing is not a sufficient parity signal
when collateral mismatches remain.

The same 826-mismatch pattern appears when bias is unfused:

```text
F.linear(..., bias=None) + bias
```

The same 826-mismatch pattern also appears when the input is a stride-2 view
passed through `F.linear`. This makes both fused-bias handling and input
layout part of the validation boundary.

The discriminator should therefore compare candidate validation backends
against fused linear/addmm semantics, not against simpler expression-level
matmul/einsum equivalence.

## Required Layout/Fused-Bias Metadata

Any future status or schema must record enough metadata to distinguish
official fused-linear/addmm behavior from unfused or layout-altered replay.

Required input metadata:

- Weighted-V source path or source status.
- Input dtype, shape, stride, storage offset, contiguity, and layout label.
- Whether the input is the original contiguous BF16 tensor.
- Whether any clone, view, transpose, or stride-altering operation was used.

Required weight metadata:

- Weight dtype, shape, stride, storage offset, and contiguity.
- Orientation, for example `output_lane_by_input_dim`.
- Whether a transposed view or cloned contiguous copy was used.

Required bias metadata:

- Bias dtype, shape, stride, storage offset, and contiguity.
- Whether bias is fused into the linear/addmm call.
- Whether bias is added after a pre-bias output is materialized.

Required output metadata:

- Output dtype, shape, stride, and contiguity.
- Full-vector digest or finite summary.
- Focus-lane value.
- Full-vector mismatch count, max_abs_diff, mean_abs_diff, first mismatch,
  and worst mismatch against the official artifact.

Required provenance:

- Producer API probe status.
- Producer dtype probe status.
- Ordered attention bundle status.
- Attention audit status.
- Consumer validation or sweep status used as comparison context.

## Candidate Implementation Hypotheses

These are conceptual only and do not authorize implementation.

1. A validation-only fused-addmm-like helper may be needed if an existing
   helper cannot preserve fused bias and original BF16 layout semantics.
2. Existing Rust/CUDA helpers may be candidates only if they can be mapped to
   fused addmm semantics and report layout/fused-bias metadata explicitly.
3. Explicit matmul/einsum replays should remain negative guards, because Stage
   3 showed they reproduce the 826-mismatch pattern.
4. Unfused-bias variants should remain negative guards, because they reproduce
   the 826-mismatch pattern.
5. Stride/layout-altered variants should remain negative guards, because a
   stride-2 input view reproduces the 826-mismatch pattern.
6. Producer-side PyTorch remains source proof only. It must not become a Rust
   runtime dependency.

## Status JSON Contract

Minimum fields for a future fused-linear/addmm discriminator status:

```json
{
  "classification": "...",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "layer_index": 6,
  "operator": "attention_o_proj",
  "source_statuses": {
    "producer_api_probe_status": "/tmp/layer6_attention_oproj_api_probe_status.json",
    "producer_dtype_probe_status": "/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json",
    "ordered_attention_status": "/tmp/layer6_ordered_attention_bundle_status.json",
    "attention_audit_status": "/tmp/layer6_ordered_attention_audit_bundle_status.json",
    "consumer_validate_status": "/tmp/layer6_ordered_bundle_validate_status.json"
  },
  "official_reference": {
    "operator_family": "fused_linear_addmm",
    "fused_bias": true,
    "original_contiguous_bf16_layout": true
  },
  "layout_metadata": {
    "input": {},
    "weight": {},
    "bias": {},
    "output": {}
  },
  "candidate_results": [
    {
      "candidate": "...",
      "diagnostic_only": false,
      "fused_bias": true,
      "preserves_original_input_layout": true,
      "full_vector_metrics": {},
      "focus_lane_metrics": {},
      "collateral_mismatches": 0,
      "clears_full_vector": false,
      "selected": false
    }
  ],
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

Expected future classifications should distinguish at least:

- Schema-only readiness.
- Candidate backend full-vector match.
- Candidate backend collateral mismatch.
- Layout metadata missing.
- Fused-bias metadata missing.
- No candidate backend explains official fused addmm semantics.
- Execution failure.

## Guardrails

Every discriminator status must preserve:

```text
validation_only = true
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

Additional guardrails:

- Default behavior must remain current.
- No production runtime routing.
- No default model-runner behavior changes.
- No CUDA kernel replacement.
- No raw `/tmp` or `.live` commits.
- No Torch runtime dependency in Rust.
- No policy promotion from focus-lane-only clearing.
- No candidate may be selected unless the full vector clears without
  collateral mismatches.

## Proof Gates Before Code

Before any implementation branch is authorized, require:

1. Accepted fused-linear/addmm design doc.
2. Exact source statuses for the Stage 3 producer API and dtype probes.
3. Status schema for layout and fused-bias metadata.
4. Clear definition of the official fused-linear/addmm reference.
5. Inventory of existing validation helpers that might map to fused-addmm
   semantics.
6. Full-vector comparison metrics for every candidate.
7. Negative guards for explicit matmul/einsum, unfused bias, and layout-altered
   input.
8. No-default-routing guarantee.
9. Rollback story.
10. Performance/release gate if any backend becomes a candidate.

## Suggested Implementation Order

This document records order only; it does not authorize implementation.

1. Add schema-only status support for fused-linear/addmm metadata.
2. Add artifact/source loader validation against the schema.
3. Add producer API and dtype probe provenance fields.
4. Add negative-guard reporting for explicit matmul/einsum, unfused bias, and
   stride/layout-altered input.
5. Compare existing validation helpers only after each helper is mapped to
   fused bias and layout semantics.
6. Select no backend unless full-vector parity clears without collateral
   mismatches.
7. Optionally re-probe layer4/layer5 producer APIs to determine whether their
   reverse o-proj policy was coincidentally matching fused-linear/addmm
   behavior.

## Recommendation

The next bounded step is a validation-only schema/source-proof branch for
fused-linear/addmm discrimination. It should consume the Stage 3 producer API
probe as source proof and treat fused `torch.addmm`, `_C._nn.linear`,
`F.linear`, and module `attn.out` as the official-equivalent operator family.

Do not implement runtime behavior, select a production backend, emit/promote
layer6 output, continue the ladder, apply correction metadata, or infer a
global o-proj policy from this design.
