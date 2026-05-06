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

## CPU Producer Attribution Result

Status:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_backend_attribution_inconclusive
```

The CPU-first attribution probe reproduces the official o-proj full vector for
all sampled Workstream A layers through module/F.linear/_C/addmm CPU Torch
paths, while explicit matmul/einsum/unfused-bias remain negative controls.
Profiler evidence is informative but inconclusive for backend identity.

Milestone claims remain unchanged: no backend is selected, no implementation is
authorized, no consumer revalidation is authorized, and no runtime/default/CUDA
behavior change, output emission, ladder continuation, final-logit, all-layer,
server, or 4097-token claim is made.

## AddMM Boundary Localization Result

Status:

```text
/tmp/fused_linear_addmm_addmm_boundary_localization_status.json
```

Classification:

```text
fused_linear_addmm_addmm_boundary_inconclusive
```

The CPU-only localization probe shows that fused addmm-with-bias clears every
sampled o-proj vector, while zero-bias addmm plus a separate bias add,
explicit matmul/einsum plus bias, and explicit unfused-bias do not. The result
supports the milestone taxonomy's fused-linear/addmm evidence class, but it
does not select a backend because core/einsum and layout guard signals are also
present.

Milestone claims remain unchanged: no backend is selected, no implementation is
authorized, no consumer revalidation is authorized, and no runtime/default/CUDA
behavior change, output emission, ladder continuation, final-logit, all-layer,
server, or 4097-token claim is made.

## Fused-Bias Arithmetic Contract Result

Status:

```text
/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json
```

Classification:

```text
fused_linear_addmm_fused_bias_arithmetic_contract_inconclusive
```

The CPU-only arithmetic-contract probe supports the milestone's fused-bias
interpretation: addmm with bias clears all sampled o-proj vectors, while
zero-bias addmm plus a separate bias add and explicit matmul/einsum/unfused
bias remain negative controls. Explicit pre-round-bias arithmetic variants
clear selected lanes on most sampled layers and full vectors on layers 10, 13,
and 16, but no variant clears the entire sampled set.

Milestone claims remain unchanged: bias-before-output-rounding is supported,
the exact accumulation/product policy remains unresolved, no backend is
selected, no implementation is authorized, no consumer revalidation is
authorized, and no runtime/default/CUDA behavior change, output emission,
ladder continuation, final-logit, all-layer, server, or 4097-token claim is
made.

## Official API Seam Synthesis

Decision record:

```text
docs/FUSED_LINEAR_ADDMM_OFFICIAL_API_SEAM_SYNTHESIS.md
```

Classification:

```text
fused_linear_addmm_official_api_seam_synthesis_recorded
```

Milestone decision: preserve Workstream A as an official CPU Torch API seam for
the sampled attention o-proj boundary. The seam is module/F.linear/_C/addmm with
BF16 input, BF16 weight, BF16 bias, fused bias before final observable BF16
output, and full-vector exactness required. Explicit matmul/einsum/unfused-bias
remain negative controls.

No further blind sweeps are recommended from this evidence. Future validation,
if needed, should use producer/API artifacts as oracle seams. No backend is
selected, no implementation is authorized, no consumer revalidation is
authorized, and no runtime/default/CUDA behavior change, output emission,
ladder continuation, final-logit, all-layer, server, or 4097-token claim is
made.

## Rust/CUDA Policy Feasibility Plan

Follow-up plan:

```text
docs/FUSED_LINEAR_ADDMM_RUST_CUDA_POLICY_FEASIBILITY_PLAN.md
```

Classification:

```text
fused_linear_addmm_rust_cuda_policy_feasibility_plan_recorded
```

The milestone now has a staged feasibility path if Workstream A ever proceeds:
CPU Torch dispatch-stability, then bounded Rust CPU policy synthesis, then CUDA
mirror only after one global CPU policy clears, then a separate promotion-proof
plan. This does not authorize implementation and does not change milestone
claims.

## CPU Dispatch-Stability Result

Status:

```text
/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json
```

Classification:

```text
fused_linear_addmm_cpu_dispatch_stability_stable
```

Gate A passed for the sampled official seam. All required CPU thread/backend
settings reproduced the baseline and official full vectors exactly for layers
6, 10, 13, 16, 18, and 21. No addmm output changed under tested settings, and
the diagnostic negative controls remained negative.

Milestone claims remain unchanged: no backend is selected, no implementation is
authorized, no consumer revalidation is authorized, and no runtime/default/CUDA
behavior change, output emission, ladder continuation, final-logit, all-layer,
server, or 4097-token claim is made.

## Rust CPU Policy Synthesis Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_synthesis_partial_only
```

Gate B did not find one global Rust CPU arithmetic policy for the sampled
Workstream A o-proj set. Layers 10, 13, and 21 have per-layer full-vector
clears, but layers 6, 16, and 18 retain mismatches under the best selectable
replays. The milestone taxonomy therefore remains anchored to the official CPU
Torch fused-linear/addmm API seam, and no CUDA mirror, backend selection,
consumer revalidation, runtime/default/CUDA behavior change, output emission,
or ladder continuation is authorized.

## Rust CPU Policy Closure Audit Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_closure_no_global_policy
```

The closure audit replayed all 238 selectable focus-clearing candidates that
had not yet received full-vector replay in Gate B. All replays executed, no
single Rust CPU policy cleared layers 6, 10, 13, 16, 18, and 21 full-vector
exactly, and no validation policy was selected.

The top near-global candidates remain residual-only, with remaining mismatches
on layers 6, 16, and 18. The simple residual analysis found one-BF16-ULP-or-less
differences but no shared rounding/tie rule that justifies a new narrow design.
The recommended next state is
`stop_policy_lane_preserve_official_api_seam`.

Milestone conclusion: Workstream A should remain recorded as an official CPU
Torch API seam unless a new design review authorizes a different lane. CUDA
mirror work, consumer revalidation, backend selection, runtime/default/CUDA
behavior changes, output emission, and ladder continuation remain unauthorized.

## PyTorch Source Attribution Plan

Plan:

```text
docs/FUSED_LINEAR_ADDMM_PYTORCH_SOURCE_ATTRIBUTION_PLAN.md
```

Classification:

```text
fused_linear_addmm_pytorch_source_attribution_plan_recorded
```

The milestone now has a planning-only option for external PyTorch source and
dispatch attribution. The plan isolates future installed-wheel introspection
and optional source builds under `/home/emmy/openai/pytorch*` and dedicated
virtual environments, preserving the current `gpt-oss-rs` worktrees. No
PyTorch clone/build, implementation, consumer revalidation, CUDA mirror work,
runtime/default/CUDA behavior change, output emission, or ladder continuation
is authorized by this planning branch.

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
