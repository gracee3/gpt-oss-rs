# Ordered Surface Batch Milestone Summary

Classification: `ordered_surface_batch_milestone_summary_recorded`

## Scope

- Docs-only milestone summary.
- Prompt/case: `developer-message-user-smoke`.
- Final-token ordered-surface evidence only.
- Layers covered: 7..23, plus historical layer6 context.
- No implementation is authorized.
- No backend is selected.
- No runtime/default/CUDA behavior change is authorized.
- No output emission or ladder continuation is authorized.

## Source Documents

- `docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md`
- `docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md`
- `docs/ORDERED_SURFACE_BATCH_FINAL_CLAIMS_SUMMARY.md`
- `docs/ORDERED_SURFACE_BATCH_POST_WORKSTREAM_TAXONOMY.md`
- `docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md`
- `docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md`
- `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- `docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md`
- `docs/SELECTED_MLP_DOWN_BUNDLE_REVALIDATION_DESIGN.md`
- `docs/RAW_QK_SOURCE_BOUNDARY_ANALYSIS_DESIGN.md`

Key statuses:

- `/tmp/ordered_surface_batch_consumer_status.json`
- `/tmp/ordered_surface_batch_probe_status.json`
- `/tmp/ordered_surface_batch_consumer_10_15_status.json`
- `/tmp/ordered_surface_batch_probe_10_15_status.json`
- `/tmp/ordered_surface_batch_consumer_16_23_status.json`
- `/tmp/ordered_surface_batch_probe_16_23_oproj_status.json`
- `/tmp/ordered_surface_batch_probe_17_21_23_raw_qk_status.json`
- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`
- `/tmp/fused_linear_addmm_status_scaffold.json`
- `/tmp/raw_qk_producer_api_probes_23_17_21_status.json`
- `/tmp/selected_mlp_down_bundle_revalidation_status.json`
- `/tmp/layer11_router_logit_bundle_revalidation_status.json`

## Milestone Claim

The ordered-surface batch pivot produced a bounded, validation-only taxonomy
for the `developer-message-user-smoke` final-token case across layers 7..23. It
moved the project from a broad mismatch hunt into operator-specific evidence
classes. The strongest current conclusion is that no single global policy
switch is justified.

## Final Taxonomy

| Class | Layers | Status |
| --- | --- | --- |
| strict/default cleared | 12, 14, 15, 22 | preserve as negative controls |
| explicit o-proj policy full-bundle cleared | 8, 9, 10 | validation-only pairwise o-proj |
| composed validation-policy full-bundle cleared | 11, 20 | layer11 router+selected-MLP; layer20 o-proj+selected-MLP |
| selected-MLP collateral negative control | 19 | policy rejected |
| raw-QK artifact/source boundary | 7, 23 | layer23 explained by producer/API; layer7 historical |
| raw-QK accumulation collateral | 17 | focus-only clears rejected |
| raw-QK positive control now o-proj blocked | 21 | raw-QK reverse clears; remaining blocker is o-proj |
| o-proj fused-linear/addmm evidence class | 6, 10, 13, 16, 18, 21 | sampled cases match producer/API fused-linear/addmm pattern |

## Workstream A - O-Proj / Fused Linear AddMM

Status: evidence matrix complete for sampled cases.

Matrix doc:

```text
docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md
```

Sampled coverage:

- Layer6 historical context.
- Layer10 pairwise-clear control.
- Layers 13, 16, 18, and 21 blocked-family or o-proj-blocked cases.

All sampled cases match module/F.linear/_C/addmm fused-bias
original-layout producer/API semantics. Explicit matmul/einsum/unfused-bias
forms are negative controls. Local pairwise, reverse, and current policies are
not backend identity. No backend is selected, and no runtime implementation is
authorized.

| Layer | Role | Result |
| --- | --- | --- |
| 6 | historical blocker | fused-linear/addmm producer/API pattern |
| 10 | pairwise-clear control | same fused-linear/addmm pattern |
| 13 | blocked-family | same fused-linear/addmm pattern |
| 16 | blocked-family | same fused-linear/addmm pattern |
| 18 | blocked-family | same fused-linear/addmm pattern |
| 21 | raw-QK-solved / o-proj-blocked | same fused-linear/addmm pattern |

Conclusion: Workstream A is ready for a docs-only fused-linear/addmm
backend-discriminator design update, not runtime implementation.

## Workstream B - Selected-MLP-Down / Router Support

Status: support-gap class retired for layer11/layer20.

- Layer20 clears the full bundle with o-proj pairwise plus replay-proven
  selected-MLP-down policy.
- Layer11 clears the full bundle with router-logit pairwise plus replay-proven
  selected-MLP-down policy.
- Layer19 remains the collateral negative control.
- No output was emitted.
- No runtime/default/CUDA behavior changed.

Conclusion: Workstream B no longer blocks the milestone; keep layer19 as the
negative control.

## Workstream C - Raw-QK Source Boundary

Status: minimal producer/API evidence set complete.

- Layer23 artifact/source boundary is explained by the official
  full/einsum/batched producer expression.
- Layer17 focus-only clears are rejected because full-matrix collateral
  persists.
- Layer21 positive raw-QK full-matrix clear is confirmed; the remaining blocker
  is o-proj.
- Layer7 remains historical artifact/source-boundary context.
- No global raw-QK policy is justified.

Conclusion: Workstream C does not require more raw-QK sweeps for this
prompt/case. Layer21 belongs under Workstream A.

## What We Can Claim

- A layer7..23 validation-only ordered-surface taxonomy exists for this one
  prompt/case.
- Strict/default clears exist for layers 12/14/15/22.
- Explicit validation-only full-bundle clears exist for layers 8/9/10.
- Layer11 and layer20 support-gap cases are retired under validation-only
  composed policies.
- Layer19 remains a negative control.
- Workstream A's sampled o-proj class coheres around fused-linear/addmm
  producer/API semantics.
- Workstream C's raw-QK source-boundary evidence is recorded.
- No single global policy switch is justified.

## What We Cannot Claim

- Production runtime parity.
- Default model-runner parity.
- CUDA kernel correctness.
- Runtime policy promotion.
- Backend identity for local policies.
- Final-logit parity.
- All-layer parity.
- Server parity.
- 4097-token or long-context behavior.
- Output promotion.
- Ladder continuation.
- Correction metadata as production semantics.
- Tolerance-based parity.
- Global policy safety.

## Recommended Next Decision

### Option A - Pause and Preserve Milestone

Use this doc as the operator-facing milestone summary before implementation
design.

### Option B - Docs-Only Backend-Discriminator Design Update

Next branch:

```text
docs/fused-linear-addmm-backend-discriminator-design-update
```

Purpose:

- Turn Workstream A producer/API matrix into backend-discriminator design
  requirements.
- Define full-vector metrics.
- Define source/layout/fused-bias metadata.
- Define candidate backend comparison contract.
- Explicitly prohibit backend selection from focus-lane clears.

### Option C - Validation-Only Backend-Discriminator/Status Plan

Only if explicitly approved after Option B.

At milestone time, Option B was recommended if preparing for implementation
design. This branch records that docs-only design update below. Do not jump
directly to runtime implementation.

## Backend-Discriminator Design Update

Option B is completed as a docs-only design update:

```text
docs/FUSED_LINEAR_ADDMM_BACKEND_DISCRIMINATOR_DESIGN.md
```

Classification:

```text
fused_linear_addmm_backend_discriminator_design_update_recorded
```

Recommended next branch, only if explicitly approved:

```text
validation/fused-linear-addmm-backend-discriminator-status
```

This is status-only readiness work, not runtime implementation, backend
selection, consumer revalidation, output emission, or ladder continuation.

## Backend-Discriminator Status Readiness

Option C status-only readiness is recorded in:

```text
/tmp/fused_linear_addmm_backend_discriminator_status.json
```

Classification:

```text
fused_linear_addmm_backend_discriminator_status_recorded
```

The status consumes existing producer/API evidence and the scaffold for layers
6, 10, 13, 16, 18, and 21. Candidate execution remains unapproved; no backend is
selected, no consumer revalidation is authorized, and no runtime/default/CUDA
behavior changes are made.

## Backend Candidate Comparator

The validation-only candidate comparator phase is recorded in:

```text
/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json
```

Classification:

```text
fused_linear_addmm_backend_discriminator_no_candidate_selected
```

Result:

- Candidate execution ran only existing validation helpers.
- No selectable candidate clears the full sampled fused-linear/addmm reference
  set.
- `pairwise_f32_bf16_output` is the best partial candidate, preserving layer10
  and clearing layer21, but retaining collateral on layer6/13/16/18.
- Milestone claims are unchanged: no production backend, final-logit,
  all-layer, server, or 4097-token claim is made.

## Fused Linear/AddMM Helper Design

The docs-only helper design is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_HELPER_DESIGN.md
```

Classification:

```text
fused_linear_addmm_helper_design_recorded
```

It defines the missing validation-only helper work after the comparator found
no selectable candidate. It keeps pairwise as partial evidence, requires
full-vector exactness across layers 6/10/13/16/18/21, and points to
`validation/fused-linear-addmm-helper-candidate` only if a future helper
candidate slice is explicitly approved.

Milestone claims remain unchanged: no backend selection, runtime/default/CUDA
behavior change, consumer revalidation, output emission, ladder continuation,
final-logit, all-layer, server, or 4097-token claim.

## Fused Linear/AddMM Helper Candidate

The validation-only helper candidate run is recorded in:

```text
/tmp/fused_linear_addmm_helper_candidate_status.json
```

Classification:

```text
fused_linear_addmm_helper_candidate_no_candidate_selected
```

Result:

- No helper cleared the full sampled set for layers 6/10/13/16/18/21.
- Pairwise remains a partial/local candidate, clearing layer10 and layer21
  only.
- cuBLAS BF16 pedantic is the best partial by total mismatch count, clearing
  layer16 only.
- cuBLAS BF16 tensor-op was available but produced broad collateral mismatches.
- No backend is selected and no consumer revalidation is authorized.

Milestone claims remain conservative: no runtime/default/CUDA behavior change,
output emission, ladder continuation, final-logit, all-layer, server, or
4097-token claim.

## Fused-AddMM-Like Helper Implementation Design

The docs-only implementation design for the next Workstream A step is recorded
in:

```text
docs/FUSED_LINEAR_ADDMM_LIKE_HELPER_IMPLEMENTATION_DESIGN.md
```

Classification:

```text
fused_linear_addmm_like_helper_implementation_design_recorded
```

The design records that existing helpers and the available cuBLAS candidates
are exhausted for layers 6/10/13/16/18/21. A future candidate must model the
producer/API module/F.linear/_C/addmm BF16 fused-bias original-layout reference
more directly. The proposed executable branch, only if explicitly approved, is
`validation/fused-linear-addmm-like-helper-prototype`.

Milestone claims remain unchanged: no backend selection, runtime/default/CUDA
behavior change, consumer revalidation, output emission, ladder continuation,
final-logit, all-layer, server, or 4097-token claim.

## Fused-AddMM-Like Helper Prototype

The validation-only cuBLASLt fused-bias prototype is recorded in:

```text
/tmp/fused_linear_addmm_like_helper_prototype_status.json
```

Classification:

```text
fused_linear_addmm_like_helper_candidate_no_candidate_selected
```

Result:

- cuBLASLt was available through the validation binary.
- `cublaslt_bf16_matmul_bias_epilogue` executed for layers
  6/10/13/16/18/21.
- No layer cleared full-vector exactly.
- Total sampled mismatches: 8432.
- Full sampled set cleared: false.
- Backend selected: false.
- Consumer revalidation authorized: false.

Milestone claims remain unchanged: no runtime/default/CUDA behavior change,
output emission, ladder continuation, final-logit, all-layer, server, or
4097-token claim.

## CPU Producer Attribution Plans

CPU-first producer attribution for the fused-linear/addmm o-proj seam is
recorded in:

```text
docs/FUSED_LINEAR_ADDMM_CPU_PRODUCER_ATTRIBUTION_PLAN.md
```

Reusable oracle producer-seam pipeline and scaling guidance is recorded in:

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
is CPU-first producer attribution, not another CUDA/helper sweep.

Future implementation branch, only after review:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-impl
```

Milestone claims remain unchanged: no backend is selected, no implementation is
authorized by this docs branch, no consumer revalidation is authorized, and no
runtime/default/CUDA behavior change, output emission, ladder continuation,
final-logit, all-layer, server, or 4097-token claim is made.

## CPU Producer Attribution Probe Results

The CPU-first producer attribution probe is recorded in:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_recorded
```

Milestone update:

- Layers evaluated: 6, 10, 13, 16, 18, 21.
- Module/F.linear/_C/addmm/addmm clear all sampled full-vector references.
- Explicit matmul/einsum/unfused-bias variants remain negative controls.
- AVX2 contract consistency is recorded for all sampled layers.
- Source-level dispatch is not proven.
- Backend identity is not proven.
- Backend selected: false.
- Implementation authorized: false.
- Consumer revalidation authorized: false.

Milestone claims remain conservative: no runtime/default/CUDA behavior change,
output emission, ladder continuation, final-logit, all-layer, server, or
4097-token claim.

## CPU Producer Attribution Result Update

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_result_update_recorded
```

Source:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-probes
2e5e5791a9c353a07ba40929a216056364af164c
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Milestone decision:

- CPU producer attribution result recorded.
- Module/F.linear/_C/addmm/addmm clear all sampled layers.
- Explicit matmul/einsum/unfused-bias remain negative controls.
- AVX2 contract consistency is true across the sampled set.
- Source-level dispatch remains unproven.
- Backend identity remains unproven.
- No backend is selected.
- No runtime/default/CUDA behavior change is authorized.

Recommended next branch:

```text
docs/fused-linear-addmm-source-stepthrough-plan
```

Milestone claims remain unchanged: no implementation authorization, no consumer
revalidation, no output emission, no ladder continuation, and no
final-logit/all-layer/server/4097 claim.

## Fused Linear/AddMM Source Step-Through Plan

The source step-through plan is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_SOURCE_STEPTHROUGH_PLAN.md
```

Classification:

```text
fused_linear_addmm_source_stepthrough_plan_recorded
```

Milestone decision:

- AVX2 contract consistency is plausible but source-level dispatch remains
  unproven.
- Backend selected: false.
- Implementation authorized: false.
- Runtime/default/CUDA behavior changes authorized: false.
- Consumer revalidation authorized: false.

Next branch:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

Milestone claims remain unchanged: no output emission, no ladder continuation,
no correction/tolerance, and no final-logit/all-layer/server/4097 claim.

## Fused Linear/AddMM Source Dispatch Table Attribution

The read-only dispatch table/profiler attribution lane is recorded in:

```text
/tmp/fused_linear_addmm_source_dispatch_table_status.json
```

Classification:

```text
fused_linear_addmm_source_dispatch_table_recorded
```

Milestone decision:

- Dispatch tables for `aten::linear`, `aten::addmm`, `aten::mm`, and
  `aten::matmul` were collected from the installed Torch wheel.
- CPU profiler toggles ran for default, MKLDNN disabled, MKLDNN enabled,
  single-thread, and default-thread-count settings.
- Profiler output observed ATen-level `linear`, `addmm`, `matmul`, `mm`,
  `einsum`, and `bmm`.
- No deeper MKLDNN/oneDNN/DNNL/MKL backend event name was visible.
- Source-level dispatch proven: false.
- Backend identity proven: false.
- Source instrumentation is recommended only after review.

Milestone claims remain unchanged: no PyTorch patch/rebuild, backend
selection, implementation authorization, consumer revalidation,
runtime/default/CUDA behavior change, output emission, ladder continuation,
correction/tolerance, or final-logit/all-layer/server/4097 claim.

## Fused Linear/AddMM Source Walk Attribution

The read-only PyTorch source-walk attribution lane is recorded in:

```text
/tmp/fused_linear_addmm_source_walk_attribution_status.json
```

Classification:

```text
fused_linear_addmm_source_walk_attribution_recorded
```

Milestone decision:

- Source tree `/home/emmy/openai/pytorch` is available and matches the
  installed Torch git version.
- The source tree is dirty from existing local edits in relevant ATen files;
  this branch did not modify it.
- Candidate path graph recorded:
  `linear` -> `addmm` -> `addmm_out_cpu` -> `addmm_impl_cpu_` ->
  `cpublas::gemm` -> BF16 cpublas/gemm_stub candidates.
- AVX2 contract source candidates were found, but source-level dispatch and
  backend identity remain unproven.
- Source instrumentation is recommended only after review.

Milestone claims remain unchanged: no PyTorch patch/rebuild, backend
selection, implementation authorization, consumer revalidation,
runtime/default/CUDA behavior change, output emission, ladder continuation,
correction/tolerance, or final-logit/all-layer/server/4097 claim.

## Guardrails

- Validation-only.
- No implementation authorization.
- No backend selected.
- No production runtime routing.
- No default model-runner behavior changes.
- No CUDA kernel changes.
- No Torch runtime dependency.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
