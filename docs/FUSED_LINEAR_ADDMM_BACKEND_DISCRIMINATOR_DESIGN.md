# Fused Linear/AddMM Backend Discriminator Design

Classification: `fused_linear_addmm_backend_discriminator_design_update_recorded`

## Scope

- Docs-only design update.
- Validation-only future discriminator.
- Target operator: attention o-proj BF16 linear with bias.
- Prompt/case: `developer-message-user-smoke`.
- Final-token ordered-surface evidence only.
- Sampled layers:
  - Layer6 historical context.
  - Layer10 pairwise-clear control.
  - Layers 13, 16, and 18 blocked-family.
  - Layer21 raw-QK-solved / o-proj-blocked.
- No implementation is authorized.
- No backend is selected.
- No consumer revalidation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission or ladder continuation is authorized.

## Source Evidence

Docs:

- `docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md`
- `docs/ORDERED_SURFACE_BATCH_MILESTONE_SUMMARY.md`
- `docs/ORDERED_SURFACE_BATCH_POST_WORKSTREAM_TAXONOMY.md`
- `docs/ORDERED_SURFACE_BATCH_FINAL_CLAIMS_SUMMARY.md`
- `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- `docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md`
- `docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md`

Statuses:

- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`
- `/tmp/fused_linear_addmm_status_scaffold.json`
- `/tmp/layer6_attention_oproj_api_probe_status.json`
- `/tmp/layer6_official_linear_backend_discriminator_probe_status.json`

## Final Producer/API Matrix Summary

| Layer | Role | Producer/API reference | Negative controls | Interpretation |
| --- | --- | --- | --- | --- |
| 6 | historical blocker | module/F.linear/_C/addmm full-vector clear | matmul/einsum/unfused-bias mismatch class | original fused-linear/addmm pattern |
| 10 | pairwise-clear control | module/F.linear/_C/addmm full-vector clear | 822 mismatches | local pairwise clear is not backend identity |
| 13 | blocked-family | module/F.linear/_C/addmm full-vector clear | 819 mismatches | fused-linear/addmm pattern |
| 16 | blocked-family | module/F.linear/_C/addmm full-vector clear | 763 mismatches | fused-linear/addmm pattern |
| 18 | blocked-family | module/F.linear/_C/addmm full-vector clear | 764/765 mismatches | fused-linear/addmm pattern |
| 21 | raw-QK-solved / o-proj-blocked | module/F.linear/_C/addmm full-vector clear | 757 mismatches | fused-linear/addmm pattern |

Every sampled blocked/control layer matches the same producer/API reference
class. Explicit matmul/einsum/unfused-bias are negative controls. Local
pairwise, reverse, and current policies may clear or fail, but they do not
establish official backend identity. A future discriminator must compare
against the producer/API full-vector reference, not against focus-lane-only
local policies.

## Problem Statement

The project now needs a validation-only discriminator that can compare
Rust/CUDA candidate o-proj implementations against the official producer/API
fused-linear/addmm reference without changing production runtime behavior.

The discriminator must answer:

- Does any available Rust/CUDA backend reproduce producer/API fused-linear/addmm
  semantics full-vector?
- Does it do so across blocked layers and controls?
- Does it preserve negative controls?
- Does it avoid focus-lane-only promotion?
- Does it avoid collateral mismatch promotion?
- Does it record layout/fused-bias/source metadata precisely enough to explain
  failures?

## Future Discriminator Input Contract

Design only. Do not implement.

Future mode name suggestion:

```text
--mode fused-linear-addmm-backend-discriminator
```

Required inputs:

```text
--fused-linear-addmm-status-scaffold /tmp/fused_linear_addmm_status_scaffold.json
--producer-api-status-13-16-10 /tmp/o_proj_producer_api_probes_13_16_10_status.json
--producer-api-status-18-21 /tmp/o_proj_producer_api_probes_18_21_status.json
--layer6-api-probe-status /tmp/layer6_attention_oproj_api_probe_status.json
```

Optional per-layer bundle/provenance statuses:

```text
--layerN-attention-bundle-status /tmp/layerN_ordered_attention_bundle_status.json
--layerN-oproj-policy-sweep-status /tmp/layerN_attention_oproj_policy_sweep_status.json
--layer21-raw-qk-revalidation-status /tmp/layer21_ordered_bundle_validate_raw_qk_policy_status.json
```

Output:

```text
--status-output /tmp/fused_linear_addmm_backend_discriminator_status.json
```

Required input behavior:

- Fail closed on missing producer/API reference status.
- Record missing optional layer provenance explicitly.
- Parse JSON metadata and metrics only unless a future implementation slice
  explicitly approves candidate execution.
- Do not import Torch.
- Do not execute model code unless separately approved.
- Do not change runtime/default/CUDA behavior.

## Required Source Metadata

Each candidate comparison must record:

- Layer index.
- Role/class.
- Focus lane.
- Weighted-V dtype/device/shape/stride/contiguity.
- O-proj weight dtype/device/shape/stride/contiguity.
- O-proj bias dtype/device/shape/stride/contiguity.
- Official output dtype/device/shape.
- Fused-bias status.
- Input-layout sensitivity status.
- Producer/API source class.
- Source status paths.
- Full-vector mismatch counts.
- Max/mean abs diff.
- First/worst mismatch.
- Focus-lane result as secondary metadata only.

## Candidate Backend Families

Design candidate families, but do not implement:

1. Current Rust sequential f32 accumulation + BF16 output.
2. Reverse f32 accumulation + BF16 output.
3. Pairwise f32 accumulation + BF16 output.
4. Chunked pairwise f32 accumulation variants.
5. f64 diagnostic, diagnostic only.
6. BF16 prebias/BF16 product evidence guards, rejected if collateral.
7. Existing cuBLAS BF16 tensor-op helper, if already available.
8. Existing cuBLAS pedantic helper, if already available.
9. Future fused-addmm-like validation helper, only if separately designed.

Candidate names must be explicit in status JSON. Diagnostic/evidence-only
candidates cannot be selected. cuBLAS availability must be recorded without
assuming parity. Candidate execution is future work and is not authorized by
this docs branch.

## Decision Rules

A candidate may only be marked full-vector-clearing if:

- It matches the producer/API full vector exactly.
- Full-vector mismatches = 0.
- `max_abs_diff = 0`.
- It clears at least one blocked-family layer.
- It preserves at least one control/negative-control layer.
- It does not create collateral mismatches.
- It records fused-bias and layout metadata.
- It is not diagnostic-only.
- It is not BF16-product evidence-only.
- It is not selected from focus lane alone.

A candidate must be rejected if:

- It clears only the focus lane.
- It clears one layer but introduces collateral on another required layer.
- It depends on unfused-bias semantics.
- It matches explicit matmul/einsum negative controls instead of producer/API
  reference.
- It lacks layout/fused-bias metadata.
- It would require production/default routing changes in the validation branch.

## Layer Set For Future Discriminator

Recommended minimum set:

- Layer13 blocked-family.
- Layer16 blocked-family.
- Layer18 blocked-family.
- Layer21 raw-QK-solved / o-proj-blocked.
- Layer10 pairwise-clear control.
- Layer6 historical context, if status-compatible.

Optional regression guards:

- Strict/default cleared layers 12, 14, 15, and 22 should remain untouched.
- Explicit-policy cleared layers 8, 9, and 10 should not be reclassified as
  runtime-cleared.
- Layer4/layer5 reverse-clear historical context may be referenced but should
  not drive backend identity.

## Future Status Contract

Future status JSON:

```json
{
  "classification": "fused_linear_addmm_backend_discriminator_design_ready",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "operator": "attention_o_proj",
  "reference": {
    "api": "module/F.linear/_C/addmm",
    "dtype": "torch.bfloat16",
    "fused_bias": true,
    "layout_sensitive": true,
    "full_vector_required": true,
    "focus_lane_only_accepted": false
  },
  "source_statuses": {
    "final_matrix": "docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md",
    "status_scaffold": "/tmp/fused_linear_addmm_status_scaffold.json",
    "producer_api_13_16_10": "/tmp/o_proj_producer_api_probes_13_16_10_status.json",
    "producer_api_18_21": "/tmp/o_proj_producer_api_probes_18_21_status.json",
    "layer6_api_probe": "/tmp/layer6_attention_oproj_api_probe_status.json"
  },
  "layers": [
    {
      "layer_index": 13,
      "role": "blocked_family",
      "producer_api_reference_available": true,
      "candidate_results": []
    }
  ],
  "candidate_backends": [],
  "selected_backend": null,
  "backend_selected": false,
  "implementation_authorized": false,
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

Allowed future classifications:

- `fused_linear_addmm_backend_discriminator_design_ready`
- `fused_linear_addmm_backend_discriminator_status_recorded`
- `fused_linear_addmm_backend_discriminator_blocked_by_missing_reference`
- `fused_linear_addmm_backend_discriminator_no_backend_selected`
- `fused_linear_addmm_backend_discriminator_backend_candidate_full_vector_cleared`
- `fused_linear_addmm_backend_discriminator_candidate_collateral_mismatches`
- `fused_linear_addmm_backend_discriminator_execution_failed`

## Proof Gates Before Any Code Slice

1. This docs design accepted.
2. Producer/API final matrix present.
3. Status scaffold present.
4. Full-vector-only decision rule accepted.
5. Focus-lane-only selection explicitly prohibited.
6. Negative controls preserved.
7. Candidate status schema defined.
8. No backend selected by design.
9. No runtime/default/CUDA behavior change.
10. No output emission or ladder continuation.

## Proposed Future Branches

### Option 1 - Status-Only Scaffold Update

Branch:

```text
validation/fused-linear-addmm-backend-discriminator-status
```

Scope:

- Read existing producer/API statuses and status scaffold.
- Emit discriminator-readiness JSON.
- No candidate execution.
- No backend selection.
- No consumer revalidation.
- No runtime/default/CUDA changes.

### Option 2 - Validation-Only Candidate Comparator

Branch:

```text
validation/fused-linear-addmm-backend-discriminator
```

Scope:

- Execute explicitly selected candidate helper families.
- Compare each candidate against producer/API full-vector references.
- Record full-vector and collateral metrics.
- Select no backend unless proof gates pass.
- No runtime/default/CUDA changes.
- No output/ladders/corrections/tolerance.

Recommended next after this docs branch: Option 1 first, unless the user
explicitly approves candidate execution.

## Status-Only Readiness Implementation

Mode:

```text
--mode fused-linear-addmm-backend-discriminator-status
```

Status path:

```text
/tmp/fused_linear_addmm_backend_discriminator_status.json
```

Classification:

```text
fused_linear_addmm_backend_discriminator_status_recorded
```

Consumed statuses:

- `/tmp/fused_linear_addmm_status_scaffold.json`
- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`
- `/tmp/layer6_attention_oproj_api_probe_status.json`
- `/tmp/layer6_official_linear_backend_discriminator_probe_status.json`

Sampled layers emitted:

- Layer6 historical blocker/context.
- Layer10 pairwise-clear control.
- Layers13/16/18 blocked-family.
- Layer21 raw-QK-solved / o-proj-blocked.

Result:

- Candidate execution: false.
- Backend selected: false.
- Implementation authorized: false.
- Consumer revalidation authorized: false.
- Next bounded step: review readiness status before authorizing candidate
  execution.

Guardrails: no runtime/default/CUDA behavior change, output emission, ladder
continuation, correction metadata, tolerance pass, final-logit, all-layer,
server, or 4097-token claim.

## Candidate Comparator Implementation

Mode:

```text
--mode fused-linear-addmm-backend-discriminator
```

Status path:

```text
/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json
```

Classification:

```text
fused_linear_addmm_backend_discriminator_no_candidate_selected
```

Layers evaluated:

- Layer6 historical blocker/context.
- Layer10 pairwise-clear control.
- Layers13/16/18 blocked-family.
- Layer21 raw-QK-solved / o-proj-blocked.

Candidates evaluated where helpers exist:

- `current_sequential_f32_bf16_output`
- `reverse_f32_bf16_output`
- `pairwise_f32_bf16_output`
- `chunked_pairwise_f32_bf16_output`
- `f64_diagnostic`
- `bf16_prebias_evidence_guard`

Unavailable helpers recorded, not failed:

- `bf16_product_evidence_guard`
- `cublas_bf16_tensor_op_if_available`
- `cublas_bf16_pedantic_if_available`

Outcome:

- No existing selectable helper clears the full sampled set.
- Best partial local candidate by blocked layers is `pairwise_f32_bf16_output`:
  it clears layer10 and layer21, but still has collateral mismatches on
  layer6, layer13, layer16, and layer18.
- `f64_diagnostic` remains diagnostic-only.
- `bf16_prebias_evidence_guard` remains evidence-only and has broad collateral
  mismatches.
- Backend selected: false.
- Implementation authorized: false.
- Consumer revalidation authorized: false.

Next bounded step: review the candidate matrix before any fused-addmm helper
design, backend design, or consumer revalidation.

## Helper Design Follow-Up

The no-candidate-selected result is converted into a docs-only helper design:

```text
docs/FUSED_LINEAR_ADDMM_HELPER_DESIGN.md
```

Classification:

```text
fused_linear_addmm_helper_design_recorded
```

The helper design records why existing local policies are exhausted, why
`pairwise_f32_bf16_output` remains partial evidence, and what a future
validation-only helper candidate slice would need to prove. The next executable
branch, only if separately approved, is:

```text
validation/fused-linear-addmm-helper-candidate
```

It does not authorize implementation, backend selection, consumer
revalidation, runtime/default/CUDA behavior changes, output emission, or ladder
continuation.

## Helper Candidate Implementation

The follow-up validation-only helper candidate mode is recorded as:

```text
--mode fused-linear-addmm-helper-candidate
```

Status path:

```text
/tmp/fused_linear_addmm_helper_candidate_status.json
```

Classification:

```text
fused_linear_addmm_helper_candidate_no_candidate_selected
```

Outcome:

- Existing local helpers were rerun as regression baselines.
- cuBLAS BF16 tensor-op and pedantic helpers were executed where available
  through validation-only wrappers.
- No candidate cleared the full sampled set.
- `pairwise_f32_bf16_output` remains partial evidence: it clears layer10 and
  layer21 but not layer6/13/16/18.
- `cublas_bf16_pedantic_or_deterministic` is the best partial by total
  mismatch count, but it clears only layer16 and is not selectable.
- `cublas_bf16_tensor_op` has broad collateral mismatches.
- Backend selected: false.
- Implementation authorized: false.
- Consumer revalidation authorized: false.

Next bounded step: docs/design for a fused-addmm-like validation helper if the
sampled-set exactness requirement still matters. No runtime/default/CUDA
behavior change, output emission, or ladder continuation follows from this
status.

## Fused-AddMM-Like Helper Implementation Design

The follow-up design is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_LIKE_HELPER_IMPLEMENTATION_DESIGN.md
```

Classification:

```text
fused_linear_addmm_like_helper_implementation_design_recorded
```

It records that existing local helpers plus available cuBLAS BF16 tensor-op and
pedantic candidates did not clear the sampled set. Future work should model the
producer/API fused-addmm-like boundary directly, with plausible paths including
cuBLASLt fused-bias epilogue or a custom validation-only CUDA fused linear+bias
helper. The next executable branch, only if explicitly approved, is
`validation/fused-linear-addmm-like-helper-prototype`.

Backend selected: false. Consumer revalidation authorized: false. No
runtime/default/CUDA behavior change, output emission, or ladder continuation
is authorized.

## Fused-AddMM-Like Helper Prototype

The validation-only cuBLASLt fused-bias prototype is recorded in:

```text
/tmp/fused_linear_addmm_like_helper_prototype_status.json
```

Classification:

```text
fused_linear_addmm_like_helper_candidate_no_candidate_selected
```

Mode:

```text
--mode fused-linear-addmm-like-helper-prototype
```

The prototype added a narrow validation-only candidate,
`cublaslt_bf16_matmul_bias_epilogue`, using BF16 input/weight/bias, cuBLASLt
bias epilogue, and BF16 output. cuBLASLt was available and the candidate
executed for layers 6/10/13/16/18/21.

Result:

- Full sampled set clear: false.
- Full-vector clear on any sampled layer: false.
- Total sampled mismatches: 8432.
- Focus-lane-only clears on layer6 and layer13 remain rejected.
- Backend selected: false.
- Implementation authorized: false.
- Consumer revalidation authorized: false.

The backend-discriminator conclusion is unchanged: existing helpers plus the
cuBLASLt fused-bias epilogue prototype do not reproduce the producer/API
module/F.linear/_C/addmm reference across the sampled set. No
runtime/default/CUDA behavior change, output emission, ladder continuation, or
final-logit/all-layer/server/4097 claim follows from this result.

## CPU Producer Attribution Plan

The next Workstream A plan is CPU-first producer attribution:

```text
docs/FUSED_LINEAR_ADDMM_CPU_PRODUCER_ATTRIBUTION_PLAN.md
```

Reusable oracle seam pipeline/scaling guidance is recorded in:

```text
docs/ORACLE_PRODUCER_SEAM_PIPELINE_AND_SCALING_PLAN.md
```

Classifications:

```text
fused_linear_addmm_cpu_producer_attribution_plan_recorded
oracle_producer_seam_pipeline_and_scaling_plan_recorded
```

The cuBLASLt prototype result remains validation evidence only. Plain cuBLASLt
fused-bias epilogue did not reproduce the producer/API reference. The next step
is to inspect the CPU Torch producer path for the official fused-linear/addmm
seam, not continue GPU/cuBLAS backend guessing.

No backend is selected. No implementation is authorized by the docs branch. No
consumer revalidation, runtime/default/CUDA behavior change, output emission,
or ladder continuation is authorized.

## CPU Producer Attribution Probe Results

The CPU-first producer attribution probe is recorded in:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_recorded
```

Branch:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-probes
```

Result:

- Layers evaluated: 6, 10, 13, 16, 18, 21.
- API paths tested: module call, F.linear, _C linear, addmm, ATen addmm
  profiler attribution, explicit matmul, explicit einsum, and unfused bias.
- Module/F.linear/_C/addmm/addmm clear full-vector across the sampled set.
- Explicit matmul/einsum/unfused-bias variants remain negative controls.
- MKLDNN and thread toggles are covered by source producer/API traces.
- AVX2 contract consistency: true for all sampled layers.
- Source-level dispatch proven: false.
- Backend identity proven: false.

The result narrows the producer attribution lane, but it does not authorize a
Rust helper implementation or backend selection.

## CPU Producer Attribution Result Update

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_result_update_recorded
```

The backend discriminator and candidate comparator found no existing selectable
backend. The CPU attribution result confirms the official API family and the
AVX2 contract consistency, but it does not prove backend identity or
source-level dispatch.

Recorded source:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-probes
2e5e5791a9c353a07ba40929a216056364af164c
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Next design path:

```text
docs/fused-linear-addmm-source-stepthrough-plan
```

This source step-through plan should come before any Rust fused-addmm helper
implementation. Backend selected: false. Implementation authorized: false.
Consumer revalidation authorized: false. Runtime/default/CUDA behavior changes:
false.

## Source Step-Through Plan

The source step-through plan is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_SOURCE_STEPTHROUGH_PLAN.md
```

Classification:

```text
fused_linear_addmm_source_stepthrough_plan_recorded
```

The candidate comparator plus CPU attribution still do not select a backend.
The next gate before helper implementation is source-level step-through,
starting with read-only dispatch table attribution:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

No PyTorch patch/rebuild, Rust helper implementation, backend selection,
consumer revalidation, runtime/default/CUDA behavior change, output emission,
or ladder continuation is authorized by this plan.

## Source Dispatch Table Attribution Result

The read-only dispatch table/profiler attribution lane is recorded in:

```text
/tmp/fused_linear_addmm_source_dispatch_table_status.json
```

Classification:

```text
fused_linear_addmm_source_dispatch_table_recorded
```

Result:

- Dispatch tables inspected: `aten::linear`, `aten::addmm`, `aten::mm`, and
  `aten::matmul`.
- Dispatch table labels show CPU registrations and MkldnnCPU labels, but not
  the exact lower-level BF16 CPU kernel path.
- CPU profiler toggles covered default, MKLDNN disabled, MKLDNN enabled,
  single thread, and default thread count.
- Profiler operators observed include `aten::linear`, `aten::addmm`,
  `aten::matmul`, `aten::mm`, `aten::einsum`, and `aten::bmm`.
- No deeper MKLDNN/oneDNN/DNNL/MKL backend event name was visible.
- Source-level dispatch proven: false.
- Backend identity proven: false.

The candidate comparator plus CPU attribution plus read-only dispatch table
attribution still do not select a backend. The next bounded decision is review
before any source instrumentation, not Rust helper implementation or consumer
revalidation.

## Non-Goals

- No runtime implementation.
- No backend selection.
- No default routing change.
- No CUDA kernel change.
- No Torch runtime dependency.
- No consumer revalidation.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
