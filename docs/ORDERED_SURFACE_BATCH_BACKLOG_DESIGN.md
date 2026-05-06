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

Potential next design: raw-QK source/artifact precision discriminator.

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

Start with Workstream A docs-only o-proj blocked-family discriminator design.

Reason:

- O-proj is the largest remaining class.
- Layer6 already has fused-linear/addmm evidence.
- Layer13/16/18/21 may reveal whether the blocked class shares layer6
  semantics.
- This is more likely to shape a useful discriminator than another blind
  sweep.

## Workstream A Design Pointer

The docs-only o-proj blocked-family discriminator design is recorded in:

```text
docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md
```

It targets layers 13, 16, 18, and 21. The first minimal producer/API probe set
for layer13, layer16, and the layer10 pairwise-clear control has now confirmed
that all three sampled layers follow the fused-linear/addmm producer pattern.
The fused-linear/addmm validation modeling plan is recorded in
`docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`. No implementation is authorized
by the design, probe result, or plan update.

## Non-Goals

- No implementation authorization.
- No production routing.
- No default behavior changes.
- No CUDA changes.
- No new probes in this branch.
- No raw artifact commits.
