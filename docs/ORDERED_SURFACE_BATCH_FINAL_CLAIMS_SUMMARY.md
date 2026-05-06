# Ordered Surface Batch Final Claims Summary

Classification: `ordered_surface_batch_final_claims_summary_recorded`

## Scope

- Docs-only claims summary.
- Prompt/case: `developer-message-user-smoke`.
- Final-token ordered-surface evidence only.
- Layers covered: 7..23, plus historical layer6 context.
- Current status: broad collection stopped; Workstreams A/B/C triaged.
- No implementation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission is authorized.
- No ladder continuation is authorized.

## Evidence Base

Primary docs:

- `docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md`
- `docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md`
- `docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md`
- `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- `docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md`
- `docs/SELECTED_MLP_DOWN_BUNDLE_REVALIDATION_DESIGN.md`
- `docs/RAW_QK_SOURCE_BOUNDARY_ANALYSIS_DESIGN.md`
- `docs/ORDERED_SURFACE_BATCH_POST_WORKSTREAM_TAXONOMY.md`

Key statuses:

- `/tmp/ordered_surface_batch_consumer_status.json`
- `/tmp/ordered_surface_batch_probe_status.json`
- `/tmp/ordered_surface_batch_generation_10_15_status.json`
- `/tmp/ordered_surface_batch_consumer_10_15_status.json`
- `/tmp/ordered_surface_batch_probe_10_15_status.json`
- `/tmp/ordered_surface_batch_generation_16_23_status.json`
- `/tmp/ordered_surface_batch_consumer_16_23_status.json`
- `/tmp/ordered_surface_batch_probe_16_23_oproj_status.json`
- `/tmp/raw_qk_dtype_probes_17_21_23_status.json`
- `/tmp/ordered_surface_batch_probe_17_21_23_raw_qk_status.json`
- `/tmp/raw_qk_producer_api_probes_23_17_21_status.json`
- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/fused_linear_addmm_status_scaffold.json`
- `/tmp/selected_mlp_down_bundle_revalidation_status.json`
- `/tmp/layer11_router_logit_bundle_revalidation_status.json`

## Claims We Can Make

1. The ordered-surface batch pivot produced a useful layer7..23 taxonomy for
   this one prompt/case.
2. Strict/default clears exist for layers 12, 14, 15, and 22.
3. Explicit validation-only full-bundle clears exist for layers 8, 9, and 10
   under attention o-proj pairwise policy.
4. Workstream A narrowed the o-proj class:
   - blocked layers 13 and 16 and pairwise-clear control layer10 all match the
     producer-side module/F.linear/_C/addmm BF16 fused-bias original-layout
     pattern;
   - local pairwise clearing is validation evidence, not backend identity.
5. Workstream B retired the selected-MLP-down support-gap class for layer11 and
   layer20:
   - layer20 full bundle clears with o-proj pairwise plus replay-proven
     selected-MLP-down policy;
   - layer11 full bundle clears with router-logit pairwise plus replay-proven
     selected-MLP-down policy;
   - layer19 remains a collateral negative control.
6. Workstream C's minimal producer/API probe set executed:
   - layer23's artifact/source-boundary case is explained by the official
     full/einsum/batched producer expression, not isolated dot or local
     accumulation variants;
   - layer17 rejects focus-only policy selection because full-matrix collateral
     persists under focus-clearing policies;
   - layer21 remains the positive raw-QK control where reverse clears full
     raw-QK/masked logits before o-proj blocks.
7. No single global policy switch is justified.
8. The project has moved from unknown mismatch to an operator-specific backlog:
   - fused-linear/addmm o-proj validation modeling;
   - raw-QK source/artifact boundary analysis;
   - preservation of negative controls and strict/default clears.

## Claims We Cannot Make

We cannot claim:

- production runtime parity;
- default model-runner parity;
- CUDA kernel correctness;
- runtime policy promotion;
- final-logit parity;
- all-layer parity;
- server parity;
- 4097-token or long-context behavior;
- all prompts;
- output promotion;
- ladder continuation;
- correction metadata as production semantics;
- tolerance-based parity;
- that pairwise/reverse/current local policies are official backend identities;
- that any policy is globally safe.

## Workstream Status

### Workstream A - O-Proj / Fused Linear AddMM

Status: status scaffold recorded; not implemented as a runtime backend.

- Layer13/layer16 blocked-family and layer10 pairwise-clear control all follow
  the same producer/API fused-linear/addmm pattern.
- Normalized scaffold: `/tmp/fused_linear_addmm_status_scaffold.json`.
- No backend selected.
- No runtime/default/CUDA change.
- Future work would be validation modeling or backend discriminator only after
  separate approval.

### Workstream B - Selected-MLP-Down / Router Support

Status: support-gap cases retired.

- Layer20 full-bundle cleared.
- Layer11 full-bundle cleared.
- Layer19 remains negative control.
- Router-logit support was needed for layer11 after selected-MLP-down cleared.
- No output emitted.
- No ladder continuation.
- No runtime/default/CUDA change.

### Workstream C - Raw-QK Source Boundary

Status: producer/API probe set recorded; no implementation authorized.

- Design doc: `docs/RAW_QK_SOURCE_BOUNDARY_ANALYSIS_DESIGN.md`.
- Result status: `/tmp/raw_qk_producer_api_probes_23_17_21_status.json`.
- Oracle branch: `oracle/raw-qk-producer-api-probes-23-17-21`.
- Oracle commit: `17e69f43fdec02a794c1f437c19cf5f033df55d6`.
- Result classification: `raw_qk_producer_api_probes_23_17_21_generated`.
- Layer23 artifact/source boundary explained by official full/einsum/batched
  producer expression.
- Layer17 focus-only policy rejected because full-matrix collateral persists.
- Layer21 positive raw-QK full-matrix clear control confirmed; full bundle
  still stops later at o-proj.
- Layer7 remains historical artifact/source-boundary context.
- No global raw-QK policy is justified.

## Current Layer Taxonomy

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

## Post-Workstream Taxonomy

The final post-workstream taxonomy refresh is recorded in:

```text
docs/ORDERED_SURFACE_BATCH_POST_WORKSTREAM_TAXONOMY.md
```

Classification:

```text
ordered_surface_batch_post_workstream_taxonomy_recorded
```

Workstream C's result update is complete. Layer21 is now classified as
raw-QK-solved for this prompt/case and o-proj-blocked for any next bounded
work. No global raw-QK policy is justified, and no implementation or
runtime/default/CUDA behavior change is authorized.

## Recommended Next Step

Next bounded decision: pause and preserve this taxonomy as the milestone
summary, or continue Workstream A evidence coverage with o-proj producer/API
probes for layers 18 and 21. Do not start implementation or select a runtime
raw-QK policy from this evidence.

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
