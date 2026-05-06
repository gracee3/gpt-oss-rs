# Ordered Surface Batch Post-Workstream Taxonomy

Classification: `ordered_surface_batch_post_workstream_taxonomy_recorded`

## Scope

- Docs-only taxonomy refresh.
- Prompt/case: `developer-message-user-smoke`.
- Final-token ordered-surface evidence only.
- Layers covered: 7..23, plus historical layer6 context.
- Workstreams A/B/C have current bounded conclusions.
- No implementation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission is authorized.
- No ladder continuation is authorized.

## Source Evidence

Docs:

- `docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md`
- `docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md`
- `docs/ORDERED_SURFACE_BATCH_FINAL_CLAIMS_SUMMARY.md`
- `docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md`
- `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- `docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md`
- `docs/SELECTED_MLP_DOWN_BUNDLE_REVALIDATION_DESIGN.md`
- `docs/RAW_QK_SOURCE_BOUNDARY_ANALYSIS_DESIGN.md`

Key statuses:

- `/tmp/fused_linear_addmm_status_scaffold.json`
- `/tmp/selected_mlp_down_bundle_revalidation_status.json`
- `/tmp/layer11_router_logit_bundle_revalidation_status.json`
- `/tmp/raw_qk_producer_api_probes_23_17_21_status.json`
- `/tmp/ordered_surface_batch_probe_17_21_23_raw_qk_status.json`
- `/tmp/ordered_surface_batch_probe_16_23_oproj_status.json`
- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`

## Final Workstream Status

### Workstream A — O-Proj / Fused Linear AddMM

Status: active follow-up class, not runtime implementation.

- Layer13 and layer16 blocked-family cases match the producer/API
  fused-linear/addmm pattern.
- Layer10 pairwise-clear control also matches the producer/API
  fused-linear/addmm pattern.
- Local pairwise clearing is validation evidence, not backend identity.
- Layer18 blocked-family also matches the producer/API fused-linear/addmm
  pattern.
- Layer21 joins Workstream A after raw-QK reverse clears and also matches the
  producer/API fused-linear/addmm pattern.
- Layer6 remains historical fused-linear/addmm context.
- No backend is selected.

## O-Proj Producer/API Final Matrix

The final o-proj producer/API evidence matrix is recorded in:

```text
docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md
```

Classification:

```text
o_proj_producer_api_final_matrix_recorded
```

Layer18 and layer21 probe status:

```text
/tmp/o_proj_producer_api_probes_18_21_status.json
```

Both layer18 and layer21 match the fused-linear/addmm producer/API pattern.
Workstream A's sampled evidence matrix now covers layer6 historical context,
layer10 control, and layers 13/16/18/21. Workstream A is ready for a design
decision, not implementation.

Next choice: pause for a milestone summary, or design a validation/backend
discriminator. This taxonomy does not authorize consumer revalidation, backend
selection, or runtime/default/CUDA changes.

### Workstream B — Selected-MLP-Down / Router Support

Status: retired for support-gap cases.

- Layer11 full bundle clears under router-logit pairwise plus
  selected-MLP-down replay-proven policy.
- Layer20 full bundle clears under o-proj pairwise plus selected-MLP-down
  replay-proven policy.
- Layer19 remains the collateral negative control.
- No output is promoted.
- No runtime/default/CUDA behavior changed.

### Workstream C — Raw-QK Source Boundary

Status: minimal producer/API evidence set complete.

- Layer23 artifact/source boundary is explained by the official
  full/einsum/batched producer expression.
- Layer17 focus-entry clears are rejected because full-matrix collateral
  persists.
- Layer21 positive raw-QK full-matrix clear is confirmed; raw-QK is no longer
  the remaining blocker for layer21.
- Layer7 remains historical artifact/source-boundary context.
- No global raw-QK policy is justified.
- No implementation is authorized.

## Updated Final Taxonomy

| Class | Layers | Status |
| --- | --- | --- |
| strict/default cleared | 12, 14, 15, 22 | preserve as negative controls |
| explicit o-proj policy full-bundle cleared | 8, 9, 10 | validation-only pairwise o-proj |
| composed validation-policy full-bundle cleared | 11, 20 | layer11 router+selected-MLP; layer20 o-proj+selected-MLP |
| selected-MLP collateral negative control | 19 | policy rejected |
| raw-QK artifact/source boundary | 7, 23 | layer23 explained by producer/API; layer7 historical |
| raw-QK accumulation collateral | 17 | focus-only clears rejected |
| raw-QK positive control now o-proj blocked | 21 | raw-QK reverse clears; remaining blocker is o-proj |
| o-proj blocked-family | 13, 16, 18, 21 | producer/API pattern known for 13/16; 18/21 pending if needed |
| historical o-proj fused-linear/addmm context | 6 | non-row context |

## Layer21 Decision

Layer21 should now be tracked under Workstream A, not Workstream C.

Reason:

- Raw-QK reverse clears the full raw-QK and masked-logit matrices.
- The producer/API probe confirms layer21 as a positive raw-QK full-matrix
  clear control.
- Full bundle revalidation stops later at o-proj.
- Therefore any next layer21 work should be o-proj/fused-linear/addmm
  classification, not more raw-QK policy sweeping.

Do not:

- Run another raw-QK sweep for layer21.
- Treat reverse as a global raw-QK policy.
- Emit layer21 output.
- Continue the ladder.
- Implement runtime policy.

## Claims After Workstream C

Can claim:

- Workstream C's minimal producer/API probe set is complete.
- Layer23 artifact/source boundary is explained by official full/einsum/batched
  producer expression.
- Layer17 focus-only policies are rejected due full-matrix collateral.
- Layer21 is raw-QK-solved for this prompt/case and remains o-proj-blocked.
- No global raw-QK policy follows.

Cannot claim:

- Runtime/default raw-QK parity.
- CUDA correctness.
- Global raw-QK policy safety.
- Final-logit parity.
- All-layer parity.
- Server parity.
- 4097-token behavior.
- Output promotion or ladder continuation.

## Recommended Next Decision

Option A - pause and preserve taxonomy:

- Stop here.
- Use this taxonomy as the operator-facing summary before any implementation
  discussion.

Option B - continue Workstream A evidence coverage:

- Run producer/API o-proj probes for layer18 and layer21.
- Purpose: determine whether the remaining o-proj blocked candidates match the
  fused-linear/addmm pattern already observed for layer6/13/16/10.
- Still oracle evidence only.
- No consumer revalidation or implementation.

Recommended: Option A if preparing a milestone summary. Option B only if the
next goal is to close the o-proj evidence matrix before implementation design.

Suggested future branch if Option B is selected:

```text
oracle/o-proj-producer-api-probes-18-21
```

This branch does not authorize it.

## Guardrails

- Validation-only.
- No implementation authorization.
- No production runtime routing.
- No default model-runner behavior changes.
- No CUDA kernel changes.
- No Torch runtime dependency.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
