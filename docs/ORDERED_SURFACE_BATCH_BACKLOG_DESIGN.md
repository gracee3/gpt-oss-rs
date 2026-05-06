# Ordered Surface Batch Backlog Design

Classification:

```text
ordered_surface_batch_backlog_design_recorded
```

## Scope

- Docs-only backlog/design.
- Final-token ordered-surface evidence only.
- One prompt/case: `developer-message-user-smoke`.
- Layers 7..23 plus historical layer6 context.
- No broad layer collection next.
- No implementation authorized.
- No output emission or ladder continuation.

## Source Matrix

Primary source:

```text
docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md
```

Source statuses:

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
```

## Ranked Backlog

### Workstream A - O-Proj Blocked-Family Design

Priority: highest.

Layers:

- 13.
- 16.
- 18.
- 21 after raw-QK reverse clears.
- Layer6 historical context with fused-linear/addmm producer/API evidence.

Known cleared contrast:

- Layers 8, 9, 10, and 20 clear o-proj with
  `pairwise_f32_accum_f32_bias_bf16_output`.
- Layer4/layer5 older evidence cleared with reverse o-proj.
- Layer6 did not clear with bounded local backend family and has producer
  fused-linear/addmm evidence.

Problem statement: o-proj is the largest remaining class, but it is not
uniform. Some layers clear with pairwise, some with reverse, and some do not
clear under bounded local sweeps.

Next design question: is the blocked class an official
fused-linear/addmm/API/layout/fused-bias boundary, a missing backend
discriminator, or a source/artifact mismatch?

Do not:

- Assume pairwise globally.
- Assume reverse globally.
- Run blind backend sweeps repeatedly.
- Implement runtime o-proj changes.

Potential next docs/design:

```text
docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md
```

Potential next evidence lane: producer/API probes for layer13/16/18/21 only if
design approves.

Proof gates:

- Compare blocked layers against layer6 fused-linear/addmm design.
- Preserve full-vector metrics.
- Require no collateral mismatches.
- Require source/layout/fused-bias metadata.
- Do not promote focus-lane-only clears.

Producer/API probe update:

- The layer13/layer16/layer10 producer/API probe set is complete:
  `/tmp/o_proj_producer_api_probes_13_16_10_status.json`.
- Blocked layers 13 and 16 match the layer6 fused-linear/addmm pattern.
- Pairwise-clear control layer10 also matches the same fused-linear/addmm
  pattern.
- Local policy classes are therefore not official backend classes: pairwise
  local clearing is validation-only evidence, not backend identity proof.
- Workstream A's next design target is fused-linear/addmm validation modeling,
  not another blind local accumulation sweep.
- The fused-linear/addmm validation plan is recorded in
  `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`.
- The fused-linear/addmm status scaffold design is recorded in
  `docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md`.
- Recommended next branch is
  `validation/fused-linear-addmm-status-scaffold` only if explicitly approved;
  otherwise use `docs/fused-linear-addmm-status-scaffold-design`.

### Workstream B - Selected-MLP-Down Revalidation Support

Priority: high.

Layers:

- 11.
- 20.
- 19.

Known results:

- Layer11 replay clears full MLP under
  `naive_f64_sum_then_bf16_output`, but bundle revalidation flag is missing.
- Layer20 o-proj clears, then full bundle stops at selected MLP down support
  gap.
- Layer19 selected-MLP replay has collateral mismatches and should not be
  forced into the layer11/layer20 pattern.

Problem statement: the validator has replay evidence for selected-MLP-down
policy behavior, but full ordered bundle revalidation cannot yet apply an
explicit selected-MLP-down policy. That prevents layer11 and layer20 from being
resolved at full-bundle level.

Potential next implementation design: selected-MLP-down bundle revalidation
flag/support.

Do not implement yet.

Required design questions:

- Which selected-MLP-down policies are already supported in replay?
- Which policy should be allowed in full bundle validation?
- How should status JSON distinguish replay-cleared vs full-bundle-cleared?
- How do we prevent layer19 collateral cases from being swept in accidentally?
- Should support be validation-only CLI flag only?

Proof gates:

- Layer11 full-bundle revalidation clears under explicit replay-proven policy.
- Layer20 full-bundle revalidation clears or cleanly localizes the next seam
  after o-proj pairwise.
- Layer19 remains blocked unless a full-vector no-collateral policy exists.
- BF16-product remains rejected if collateral appears.

Design status:

- Workstream B design is recorded in
  `docs/SELECTED_MLP_DOWN_BUNDLE_REVALIDATION_DESIGN.md`.
- Classification:
  `selected_mlp_down_bundle_revalidation_design_recorded`.
- Implementation slice is recorded on
  `validation/selected-mlp-down-bundle-revalidation`.
- Batch status:
  `/tmp/selected_mlp_down_bundle_revalidation_status.json`.
- Classification:
  `selected_mlp_down_bundle_revalidation_partial`.
- Layer11 accepts the replay-proven `naive_f64_sum_then_bf16_output` policy for
  selected-down outputs, but full bundle revalidation remains blocked by one
  router-logit mismatch.
- Layer20 composes o-proj pairwise with replay-proven
  `pairwise_f32_sum_then_bf16_output` and clears the full bundle.
- Layer19 is the collateral negative control and must not be forced into the
  layer11/layer20 pattern.
- Layer11 follow-up localization is recorded in
  `/tmp/layer11_router_logit_localization_status.json` with classification
  `layer11_router_logit_localization_recorded`.
- Layer11 selected-down support works; the remaining seam is router logit
  expert 7, local `0.0693359375` vs official `0.06884765625`.
- Router-logit localization found full-vector clears under pairwise, reverse,
  and chunked-pairwise policies; the localization slice stopped before bundle
  router-policy revalidation.
- Layer11 router-logit bundle revalidation is recorded in
  `/tmp/layer11_router_logit_bundle_revalidation_status.json` with
  classification `layer11_router_logit_bundle_revalidation_full_bundle_cleared`.
- Layer11 now clears full bundle with router policy
  `pairwise_f32_bf16_bias_bf16_output` and selected-MLP-down policy
  `naive_f64_sum_then_bf16_output`.
- Workstream B support-gap cases are retired for layer11 and layer20; layer19
  remains the collateral negative control.
- Runtime/default/CUDA behavior remains unchanged; no outputs are emitted and
  the ladder is not continued.

### Workstream C - Raw-QK Artifact/Source Boundary Analysis

Priority: medium.

Layers:

- 7.
- 17.
- 23.
- Layer21 as partial success: reverse clears raw-QK but exposes o-proj.

Known results:

- Layer7 artifact precision/source boundary; no valid full-matrix policy.
- Layer17 accumulation-boundary dtype probe, but sweep has collateral
  mismatches.
- Layer23 artifact precision/source boundary; no candidate clears.
- Layer21 reverse clears raw-QK/masked logits, then o-proj blocks.

Problem statement: raw-QK splits into accumulation-boundary cases and
artifact/source precision cases. Some focus-entry evidence does not become a
safe full-matrix policy.

Producer/API result status: the minimal Workstream C probe set is complete for
layers 23, 17, and 21.

Design status:

- Workstream C raw-QK source boundary analysis design is recorded in
  `docs/RAW_QK_SOURCE_BOUNDARY_ANALYSIS_DESIGN.md`.
- Classification: `raw_qk_source_boundary_analysis_design_recorded`.
- Producer/API probe set completed on branch
  `oracle/raw-qk-producer-api-probes-23-17-21`.
- Oracle commit: `17e69f43fdec02a794c1f437c19cf5f033df55d6`.
- Status: `/tmp/raw_qk_producer_api_probes_23_17_21_status.json`.
- Classification: `raw_qk_producer_api_probes_23_17_21_generated`.
- Layer23 artifact/source boundary is explained by the official
  full/einsum/batched producer expression; isolated dot and local variants do
  not explain it.
- Layer17 focus-entry clears exist, but full-matrix collateral persists, so
  focus-only policy selection is rejected.
- Layer21 positive raw-QK full-matrix clear control is confirmed; full bundle
  still stops later at o-proj.
- No global raw-QK policy is justified.
- Implementation remains unauthorized.
- Layer7 remains historical artifact/source-boundary context.

Do not:

- Use BF16-product as correction.
- Use f64 diagnostic as policy.
- Treat focus-lane clearing as enough.
- Run raw-QK sweeps without dtype/source prerequisites.

Proof gates:

- Full raw-QK matrix clear.
- Full masked-logit clear.
- No attention-probability collateral.
- Source status and dtype probe present.
- Evidence-only policies remain rejected.

## Negative Controls / Preserve As Is

- Strict/default cleared layers: 12, 14, 15, and 22.
- Explicit-policy full-bundle cleared layers: 8, 9, and 10.
- Do not rerun or perturb these unless a later design specifically requires
  regression guards.
- Strict/default clears are important negative controls against global
  policies.

## Cross-Cutting Guardrails

- No single global policy switch.
- No runtime/default routing changes.
- No CUDA kernel changes.
- No output emission.
- No ladder continuation.
- No tolerance pass.
- No correction metadata.
- No final-logit claim.
- No all-layer claim.
- No server claim.
- No 4097-token claim.

## Proposed Branch Sequence

1. `docs/o-proj-blocked-family-discriminator-design`
   - Docs-only.
   - Compare layer13/16/18/21 with layer6 fused-linear/addmm evidence.
   - Decide whether producer/API probes are needed.
2. `validation/selected-mlp-down-bundle-policy-design` or
   `docs/selected-mlp-down-bundle-revalidation-design`
   - First docs-only unless the user explicitly authorizes implementation.
   - Define CLI/status contract for applying replay-proven selected-MLP-down
     policy inside full bundle validation.
3. `oracle/raw-qk-source-boundary-analysis`
   - Only after o-proj and selected-MLP-down backlog priorities are triaged.
   - Focus layer7/17/23 source/artifact precision differences.

## Recommended Immediate Next Step

Update the final taxonomy with the Workstream C producer/API result and decide
whether layer21's post-raw-QK o-proj blocker belongs under Workstream A.

Reason:

- Layer23 is now explained as an artifact/source-boundary case.
- Layer17 is now a clear focus-only rejection case.
- Layer21 is now the positive raw-QK full-matrix clear control, but the full
  bundle still stops later at o-proj.
- No raw-QK runtime/default policy follows from this evidence.

## Workstream A Design Pointer

The docs-only o-proj blocked-family discriminator design is recorded in:

```text
docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md
```

It targets layers 13, 16, 18, and 21. The first minimal producer/API probe set
for layer13, layer16, and the layer10 pairwise-clear control has now confirmed
that all three sampled layers follow the fused-linear/addmm producer pattern.
The fused-linear/addmm validation modeling plan is recorded in
`docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`, and the status scaffold design is
recorded in `docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md`. No
implementation is authorized by the design, probe result, plan update, or
scaffold design.

## Final Claims Summary

The conservative final claims summary for the ordered-surface batch pivot is
recorded in:

```text
docs/ORDERED_SURFACE_BATCH_FINAL_CLAIMS_SUMMARY.md
```

Classification: `ordered_surface_batch_final_claims_summary_recorded`.

Status:

- Workstream A: fused-linear/addmm o-proj evidence normalized in the status
  scaffold; no backend selected.
- Workstream B: layer11 and layer20 support-gap cases retired under
  validation-only composed policies; layer19 remains the collateral negative
  control.
- Workstream C: raw-QK producer/API result recorded for layers 23/17/21; no
  global policy or implementation authorized.
- Next bounded decision: update final taxonomy and decide whether layer21's
  o-proj blocker belongs under Workstream A.

Guardrails: no implementation authorization, runtime/default/CUDA change,
output emission, ladder continuation, correction metadata promotion, tolerance
pass, final-logit, all-layer, server, or 4097-token claim.

## Post-Workstream Taxonomy Pointer

The final post-workstream taxonomy refresh is recorded in:

```text
docs/ORDERED_SURFACE_BATCH_POST_WORKSTREAM_TAXONOMY.md
```

Classification: `ordered_surface_batch_post_workstream_taxonomy_recorded`.

Current status:

- Workstream A remains active only as an o-proj blocked-family evidence class.
  Layer18 remains unprobed, and layer21 joins this backlog after raw-QK reverse
  clears.
- Workstream B's selected-MLP-down support-gap class is retired for layer11
  and layer20; layer19 remains the collateral negative control.
- Workstream C's minimal producer/API evidence set is complete. Layer23 is
  explained as artifact/source boundary, layer17 rejects focus-only policy, and
  layer21 is no longer a raw-QK blocker.
- No implementation, runtime/default/CUDA change, output emission, ladder
  continuation, correction metadata, tolerance pass, or global raw-QK policy is
  authorized.

## O-Proj Producer/API Final Matrix

The final sampled o-proj producer/API matrix is recorded in:

```text
docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md
```

Classification: `o_proj_producer_api_final_matrix_recorded`.

Status paths:

- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`
- `/tmp/fused_linear_addmm_status_scaffold.json`

Workstream A update:

- Layer18 and layer21 producer/API probes are complete.
- All sampled o-proj blocked/control cases match fused-linear/addmm
  producer/API semantics: layer6 context, layer10 control, and layers
  13/16/18/21.
- Explicit matmul/einsum/unfused-bias remain negative controls.
- The next design target is a validation/backend discriminator or milestone
  preservation, not more blind sweeps.
- No backend is selected and no implementation is authorized.

## Milestone Summary

The ordered-surface batch milestone summary is recorded in:

```text
docs/ORDERED_SURFACE_BATCH_MILESTONE_SUMMARY.md
```

Classification: `ordered_surface_batch_milestone_summary_recorded`.

Milestone status:

- Workstream A evidence matrix is complete for sampled cases.
- Workstream B support-gap cases are retired for layer11/layer20, with layer19
  preserved as the collateral negative control.
- Workstream C minimal producer/API evidence is complete and does not justify a
  global raw-QK policy.
- Next design branch, if preparing for implementation design:
  `docs/fused-linear-addmm-backend-discriminator-design-update`.

## Fused Linear/AddMM Backend Discriminator Design Update

Workstream A backend-discriminator design update is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_BACKEND_DISCRIMINATOR_DESIGN.md
```

Classification: `fused_linear_addmm_backend_discriminator_design_update_recorded`.

Next branch, only if explicitly approved:

```text
validation/fused-linear-addmm-backend-discriminator-status
```

This next step is status-only readiness over existing producer/API statuses and
the status scaffold. It is not candidate execution, backend selection,
consumer revalidation, runtime/default/CUDA behavior change, output emission,
or ladder continuation.

## Non-Goals

- No implementation authorization.
- No production routing.
- No default behavior changes.
- No CUDA changes.
- No new probes in this branch.
- No raw artifact commits.
