# Frontier Proof Status

This file tracks which integration-held subsets are ready for `main`, which frontier candidates remain paused, and what proof is still required before any further runtime-forward promotion.

## Ready For Main

### Safe Harness Wrapper Chain Already Mirrored On Integration

- `402924a` Improve Tier-2 harness operator workflow
- `ae4ef08` Add opt-in warm oracle listener scaffold
- `746fe7c` Wire warm oracle into probe validation wrapper
- `645b6eb` Add artifact provenance guardrails for trace reuse
- `480cd08` Version wrapper-owned trace capture metadata
- `ae3c9bd` Add dry-run trace artifact inspection mode
- `dd050b0` Add wrapper regression script for harness metadata paths
- `ae5bf7d` Add strict current-contract mode for trace reuse

### Safe Harness Proof-Tooling Chain Already Mirrored On Integration

- `7c115dc` Add long-context proof setup scaffold
- `def25f2` Add bounded YaRN proof execution summary
- `63f0ecc` Document bounded YaRN proof seam

### Safe Runtime-Forward Extraction Chain Already Mirrored On Integration

- `7ba163c` conservative semantic cache projection helpers
- `e174797` rope-scaling config parsing hardening
- `38f692e` worker-config rope field carry-through
- Coupled default/support-plumbing subset:
  - `5c30e56` Add `ModelRunnerConfig` default
  - `7bdf268` Add `ModelRunnerConfig` default for architecture tests
  - `dfc6abe` Carry `ModelRunnerConfig` rope fields through test fixtures

These subsets are the current integration-held runtime-forward ceiling. No additional runtime-forward cherry-picks should be promoted from integration until a new candidate clears the proof bar below.

## Paused / Deferred

- Pause additional runtime-forward cherry-picks until a proof-backed candidate exists beyond the already-mirrored safe chain.
- `838d3f8`: closed and deferred on the current runtime seam.
- `bd49d35`: partially supported only at the YaRN table-construction boundary; not yet proven by bounded long-context runtime evidence above 4096 tokens.
- `bd49d35` remains the only live semantic frontier.

## Reopen Criteria For `838d3f8`

Current same-input CUDA seam evidence argues against `838d3f8` on the current runtime path: when sink counts were nonzero, the observed GPU boundary still showed no extra visible KV token offsets. That closes the proposed sink-aware semantic cache claim on the current seam rather than merely leaving it unproven.

Do not reopen `838d3f8` unless all of the following are observed:

- A new runtime implementation seam that explicitly carries extra visible KV offsets for sink tokens, or equally direct contrary evidence at the same boundary.
- Same-input evidence comparing conservative vs sink-aware projection against observed runtime/cache behavior, not just config metadata.
- At least one bounded case with nonzero sink tensors where the sink-aware projection better matches the effective runtime path.
- At least one bounded control case with zero or absent sink tensors where the conservative path remains the honest predictor.

`838d3f8` is closed/deferred on the current runtime seam and is not a live promotion target. Do not reopen it without a new runtime implementation seam or equally direct contradictory evidence.

Until then, no sub-slice beyond `992a741` is honest to promote.

## Advancement Criteria For `bd49d35`

Current evidence supports `bd49d35` only at the YaRN table-construction boundary. That is not yet enough to promote the remaining runtime-forward slice because the missing proof is still end-to-end runtime behavior under active long-context scaling.

Evidence already present:

- Positive support at the YaRN table-construction boundary.
- Safe additive parser/config/support-plumbing sub-slices are already mirrored on integration.

Evidence still required before promotion:

- A completed bounded same-input runtime check above 4096 tokens where YaRN scaling is active on GPU0.
- A matched control case where YaRN scaling is inactive.
- Localized observation that the implemented GPU-runner YaRN table usage reproduces expected GPT-OSS behavior on the same prompt/input data.
- Evidence strong enough to distinguish "table construction looks plausible" from "runtime long-context behavior is correct."

Do not treat restricted bench plumbing or compile/test success alone as proof of `bd49d35`.

## Promotion Gate For `bd49d35`

Reopen promotion discussion for `bd49d35` only when the submitted proof set includes all of the following:

- the same prompt/input exercised in both the safe and variant runs
- a sink-free restricted model for the compared long-context case
- at least one case above 4096 tokens
- a safe-vs-variant comparison on the same bounded observation seam
- at least one localized runtime artifact or value above the table-construction boundary
- evidence that the observed difference or parity propagates through the runtime path, not just through YaRN table construction

If any of those conditions are missing, the result is still useful harness evidence but not yet promotion-gating proof.

## Proof-Backed Promotion Bar

A frontier runtime-forward candidate is promotion-eligible only when all of the following are true:

- The candidate is bounded enough to extract without dragging in speculative neighboring work.
- The proof is same-input and localized to the claimed runtime boundary.
- The proof includes a control case that would fail or stay conservative if the claim were wrong.
- The evidence supports the promoted behavior directly, not just metadata carry-through or parser acceptance.

## Next Accepted Artifact Set

The next frontier handoff from harness/feature to integration is accepted only if it contains:

- the exact prompt/input identity used for the safe and variant runs
- sink-free model confirmation for the restricted comparison case
- an explicit `>4096` token count record
- the paired safe and variant command plans or exact invoked commands
- at least one localized runtime artifact or value above the table boundary
- a short operator summary stating whether the evidence shows runtime-path propagation or stops at table construction
- a clear pass/fail statement on whether the result is promotion-gating proof for `bd49d35`

## GPU0 Live-Test Readiness

Already ready:

- Tier-2 harness workflow and wrapper chain on integration.
- Current-contract trace reuse guardrails.
- Dry-run inspection and wrapper regression coverage.
- The safe runtime-forward extraction chain already mirrored on integration.

Frontier result required to unlock the next live promotion decision:

- For `838d3f8`: no live promotion target on the current seam. Reopen only if a new runtime seam explicitly carries extra visible KV offsets, or equally direct contrary evidence appears.
- For `bd49d35`: a completed same-input runtime proof above 4096 tokens on GPU0 showing that YaRN-active behavior matches expectation and localizes correctly against a control case.

Until the `bd49d35` frontier clears that bar, integration stays paused on additional runtime-forward promotion.
