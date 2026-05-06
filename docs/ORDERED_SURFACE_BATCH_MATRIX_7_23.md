# Ordered Surface Batch Matrix 7..23

Classification:

```text
ordered_surface_batch_matrix_7_23_recorded
```

This docs-only matrix consolidates the ordered-surface batch pivot for layers
7..23. It records validation evidence and blocked classes only. It does not
authorize runtime/default routing/CUDA changes, output emission, ladder
continuation, correction metadata, tolerance passes, or final-logit,
all-layer, server, or 4097-token claims.

## Source Statuses

```text
/tmp/ordered_surface_batch_consumer_status.json
/tmp/ordered_surface_batch_probe_status.json
/tmp/ordered_surface_batch_generation_10_15_status.json
/tmp/ordered_surface_batch_consumer_10_15_status.json
/tmp/ordered_surface_batch_probe_10_15_status.json
/tmp/ordered_surface_batch_generation_16_23_status.json
/tmp/ordered_surface_batch_consumer_16_23_status.json
/tmp/ordered_surface_batch_probe_16_23_oproj_status.json
/tmp/raw_qk_dtype_probes_17_21_23_status.json
/tmp/ordered_surface_batch_probe_17_21_23_raw_qk_status.json
/tmp/raw_qk_producer_api_probes_23_17_21_status.json
```

Historically relevant layer6 context lives outside this 7..23 matrix:
the official-linear / fused-addmm discriminator design and layer6 producer API
result explain an o-proj backend boundary. Layer6 is not included as a row.

## Matrix

| Layer | Strict/default result | First failing seam | Probe result | Selected validation-only policy | Final status |
| --- | --- | --- | --- | --- | --- |
| 7 | failed | raw-QK q_head 50 / key col 57 | dtype probe artifact precision; raw-QK sweep collateral mismatches | none | blocked: raw-QK artifact/source boundary |
| 8 | failed | o-proj lane 2578 | full-vector o-proj clear; bundle revalidation cleared | `pairwise_f32_accum_f32_bias_bf16_output` for o-proj | explicit-policy full-bundle cleared |
| 9 | failed | o-proj lane 446 | full-vector o-proj clear; bundle revalidation cleared | `pairwise_f32_accum_f32_bias_bf16_output` for o-proj | explicit-policy full-bundle cleared |
| 10 | failed | o-proj lane 915 | full-vector o-proj clear; bundle revalidation cleared | `pairwise_f32_accum_f32_bias_bf16_output` for o-proj | explicit-policy full-bundle cleared |
| 11 | failed | selected MLP down lane 1480 | replay full MLP clear; no bundle revalidation flag | `naive_f64_sum_then_bf16_output` replay evidence | revalidation support missing |
| 12 | cleared | none | not needed | none | strict/default cleared |
| 13 | failed | o-proj lane 151 | no non-diagnostic full-vector o-proj clear | none | blocked: o-proj bounded-family |
| 14 | cleared | none | not needed | none | strict/default cleared |
| 15 | cleared | none | not needed | none | strict/default cleared |
| 16 | failed | o-proj lane 2666 | no non-diagnostic full-vector o-proj clear | none | blocked: o-proj bounded-family |
| 17 | failed | raw-QK q_head 35 / key col 65 | dtype accumulation boundary; sweep collateral mismatches | none | blocked: raw-QK collateral/no full-matrix policy |
| 18 | failed | o-proj lane 63 | no non-diagnostic full-vector o-proj clear | none | blocked: o-proj bounded-family |
| 19 | failed | selected MLP down lane 3005 | replay collateral mismatches; baseline has selected-output mismatches | none | blocked: selected-MLP-down collateral |
| 20 | failed | o-proj lane 2212 | o-proj full-vector clear; revalidation stopped at selected MLP down | `pairwise_f32_accum_f32_bias_bf16_output` for o-proj only | partial: o-proj cleared, MLP-down support gap |
| 21 | failed | raw-QK q_head 52 / key col 55 | reverse clears raw-QK/masked logits; revalidation stopped at o-proj | `reverse_f32_scale_after_sum_bf16_output` for raw-QK only | partial: raw-QK cleared, o-proj blocked |
| 22 | cleared | none | not needed | none | strict/default cleared |
| 23 | failed | raw-QK q_head 33 / key col 27 | dtype artifact precision; no candidate clears | none | blocked: raw-QK artifact/source boundary |

## Final Taxonomy

### Strict/default cleared

- Layers 12, 14, 15, and 22.

### Explicit-policy full-bundle cleared

- Layers 8, 9, and 10.
- Policy: `pairwise_f32_accum_f32_bias_bf16_output` for attention o-proj.

### Partial seam-cleared but full-bundle-blocked

- Layer20: o-proj pairwise clears; full bundle stops at selected MLP down.
- Layer21: raw-QK reverse clears; full bundle stops at attention o-proj.

### Raw-QK blocked

- Layer7: artifact precision / source boundary.
- Layer17: accumulation-boundary dtype probe, but sweep collateral mismatches.
- Layer23: artifact precision / source boundary.

Workstream C raw-QK source boundary analysis design is recorded in:

```text
docs/RAW_QK_SOURCE_BOUNDARY_ANALYSIS_DESIGN.md
```

It preserves layer7/23 as artifact/source-boundary cases, layer17 as an
accumulation-boundary collateral case, and layer21 as the positive raw-QK
full-matrix clear control.

Post-matrix update: Workstream C producer/API probe results are now recorded in
`/tmp/raw_qk_producer_api_probes_23_17_21_status.json`. Layer23's
artifact/source boundary is explained by the official full/einsum/batched
producer expression. Layer17 focus-only clears are rejected because full-matrix
collateral persists. Layer21 remains the positive raw-QK clear control, but its
full bundle still stops at o-proj. The result does not authorize
implementation, runtime/default/CUDA changes, output emission, ladder
continuation, correction metadata, tolerance, or any final-logit/all-layer/
server/4097 claim.

### O-proj bounded-family blocked

- Layer13.
- Layer16.
- Layer18.
- Layer21 after raw-QK revalidation.
- Layer6 as historical non-row context with producer API / fused-addmm
  evidence.

Producer/API probes for layers 13/16/10 are recorded in:

```text
/tmp/o_proj_producer_api_probes_13_16_10_status.json
```

Result: blocked layers 13/16 and pairwise-clear control layer10 all match the
same fused-linear/addmm producer pattern. Consequence: pairwise local clearing
remains validation-only evidence, not official backend identity.

The validation modeling follow-up from this evidence is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md
```

### Selected-MLP-down blocked or support gap

- Layer11: replay clears but bundle revalidation flag missing.
- Layer19: replay collateral mismatches.
- Layer20: selected-MLP-down support gap after o-proj clears.

Workstream B selected-MLP-down bundle revalidation design is recorded in:

```text
docs/SELECTED_MLP_DOWN_BUNDLE_REVALIDATION_DESIGN.md
```

It keeps layer11/layer20 as support-gap targets, layer19 as the collateral
negative control, and does not authorize implementation.

### Evidence-only / rejected policies

- BF16-product remains rejected wherever it introduces broad collateral
  mismatches.
- F64 diagnostic remains diagnostic only.
- Deterministic abs-ascending is not globally safe.
- Reverse/pairwise raw-QK and o-proj policies are layer/operator-specific, not
  global defaults.

## Interpretation

- The pivot worked: the project now has a broad layer7..23 ordered-surface
  taxonomy without ladder continuation.
- No single global policy switch is justified.
- Attention o-proj is the largest remaining class, but not uniform:
  pairwise clears layers 8/9/10/20 o-proj; no bounded family clears layers
  13/16/18; layer21 now needs o-proj work after raw-QK clears; layer6
  separately points to fused-linear/addmm producer semantics.
- Raw-QK also splits into at least two classes: accumulation-boundary cases
  where a policy may clear, and artifact/source precision cases where bounded
  policies do not clear.
- Selected-MLP-down requires separate support: bundle revalidation support is
  missing for layer11 and likely relevant after layer20, while layer19 has
  collateral mismatches and should not be forced into a global policy.
- Strict/default clears exist and should be preserved as negative controls.

## Recommended Next Steps

1. Stop broad layer collection for this prompt/case.
2. Create a docs/design backlog from the matrix before more probes.
3. Prioritize o-proj blocked-family design/API review for layers 13/16/18/21
   plus layer6 context.
4. Prioritize selected-MLP-down bundle revalidation support for layer11/layer20.
5. Prioritize raw-QK artifact/source analysis for layers 7/23 and collateral
   case layer17.
6. Do not emit layer outputs or continue the ladder until a scoped promotion
   decision is made.
7. Do not implement runtime/default/CUDA policy changes from this matrix.

## Backlog / Design Follow-Up

The docs-only backlog/design follow-up from this matrix is recorded in:

```text
docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md
```

The Workstream A o-proj blocked-family discriminator design is recorded in:

```text
docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md
```

The follow-up producer/API matrix for layers 13/16/10 is recorded in:

```text
/tmp/o_proj_producer_api_probes_13_16_10_status.json
```

The fused-linear/addmm validation plan update is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md
```

The conservative final claims summary for the ordered-surface batch pivot is
recorded in:

```text
docs/ORDERED_SURFACE_BATCH_FINAL_CLAIMS_SUMMARY.md
```

The post-workstream taxonomy refresh is recorded in:

```text
docs/ORDERED_SURFACE_BATCH_POST_WORKSTREAM_TAXONOMY.md
```

Post-workstream pointer: layer21's remaining blocker is o-proj, not raw-QK.
Raw-QK reverse clears the full raw-QK/masked-logit matrices for layer21, and
the producer/API probe confirms it as the positive raw-QK clear control. Any
next layer21 evidence work belongs under Workstream A o-proj/fused-linear/addmm
classification, not another raw-QK sweep. No implementation or runtime/default/
CUDA behavior change is authorized.

The final o-proj producer/API matrix is recorded in:

```text
docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md
```

It adds layer18 and layer21 to the fused-linear/addmm producer/API evidence:
both match the module/F.linear/_C/addmm full-vector-clear pattern, while
explicit matmul/einsum and unfused-bias forms remain negative controls. Layer21
therefore remains o-proj blocked after raw-QK clears. This is evidence for a
future validation/backend discriminator decision, not a backend selection or
runtime implementation authorization.

## Guardrails

- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No runtime/default/CUDA changes.
- No final-logit/all-layer/server/4097 claim.
