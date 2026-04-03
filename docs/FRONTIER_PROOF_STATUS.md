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
- `bd49d35`: supported through localized post-RoPE, post-attention context, and post-attention residual runtime seams above 4096 tokens on GPU0, but not yet proven downstream of those boundaries.
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

Current evidence supports `bd49d35` through the localized post-RoPE, post-attention context, and post-attention residual runtime seams above 4096 tokens on GPU0. That is still not enough to promote the remaining runtime-forward slice because downstream propagation has not yet been shown.

Evidence already present:

- Positive support at the YaRN table-construction boundary.
- Same-input `>4096` sink-free GPU0 case with real post-RoPE runtime artifacts.
- Safe-vs-variant divergence localized at the post-RoPE q/k last-token seam above the activation boundary.
- Same-input `>4096` sink-free GPU0 case with localized post-attention context last-token artifacts.
- Safe-vs-variant divergence preserved at the post-attention context last-token seam.
- Same-input `>4096` sink-free GPU0 case with localized post-attention residual last-token artifacts.
- Safe-vs-variant divergence preserved at the post-attention residual seam, with attenuation relative to post-attention context but not disappearance.
- The residual result is strong enough to support bounded live-test planning discussion.
- Safe additive parser/config/support-plumbing sub-slices are already mirrored on integration.

Evidence still required before promotion:

- A true retained-state 4096-token prefill on the same bounded sink-free case.
- One continuation/decode step on that retained state.
- A localized layer-0 continuation-token artifact, preferably post-attention residual.
- If the preferred continuation-token target is not honestly reachable, a fallback localized continuation-token artifact at post-attention context or post-RoPE q/k.
- Evidence strong enough to show propagation through the runtime path beyond post-attention residual, not just divergence at earlier seams.

Proof-method boundary:

- The full 4100-token `restricted_prefill_trace` path is no longer the preferred way to push one hop later downstream.
- The next honest proof path is a true retained-state continuation seam, not prompt replay.
- Failure to emit even the safe-side artifact for post-MLP or fallback early layer-output on that broader path should be treated as a proof-method boundary, not as evidence against `bd49d35`.

Current retained-state blocker:

- The retained-state `restricted_logit_diff` decode1 seam is confirmed as the next honest proof path.
- The current blocker is no longer metadata upload or artifact-exit handling.
- The retained seam has already established meaningful runtime propagation for `bd49d35` through post-RoPE, post-attention context, post-attention residual, retained layer-0 completion, retained layer-1 attention and post-attention residual handoff, and retained layer-1 MLP entry through `RETAINED_MLP_STAGE layer=1 stage=router_input_ready`.
- On the exact `4096 + 1` case with honest `--max-model-len 4608`, the retained child still does not emit `RETAINED_LAYER_BOUNDARY layer=1 stage=mlp_done`, `RETAINED_STEP_FORWARD_DONE`, `DECODE1_BEGIN`, `RETAINED_PROOF_ENTER`, or `RETAINED_PROOF_CAPTURED`.
- The next preferred action is now a bounded GPU1 live-smoke on a clean candidate tree rather than more retained-seam chasing by default.
- Retained seam chasing is paused unless the live-smoke results indicate a specific need to resume it.
- The retained seam findings above are part of the decision record and should be preserved for future follow-up even if the workflow pivots to live-smoke.
- Promotion remains paused until bounded GPU1 live-smoke results are in hand, or until a real continuation-token artifact is emitted by resuming the retained seam deliberately.

Do not treat restricted bench plumbing or compile/test success alone as proof of `bd49d35`.

## Promotion Gate For `bd49d35`

Reopen promotion discussion for `bd49d35` only when the submitted proof set includes all of the following:

- the same prompt/input exercised in both the safe and variant runs
- a sink-free restricted model for the compared long-context case
- at least one case above 4096 tokens
- a safe-vs-variant comparison on the same bounded observation seam
- localized post-RoPE runtime evidence above the activation boundary
- a localized post-attention context last-token artifact
- a localized post-attention residual last-token artifact
- a true retained-state continuation comparison on the same case
- a localized continuation-token artifact, preferably layer-0 post-attention residual
- evidence that the observed difference or parity propagates through the runtime path, not just through YaRN table construction, the post-RoPE boundary, the post-attention context boundary, or the post-attention residual boundary

If any of those conditions are missing, the result is still useful harness evidence but not yet promotion-gating proof.

Post-RoPE runtime evidence is now a satisfied checklist item for `bd49d35`. It is necessary but not sufficient for promotion discussion.
Post-attention context runtime evidence is now also a satisfied checklist item for `bd49d35`. It is necessary but not sufficient for promotion discussion.
Post-attention residual runtime evidence is now also a satisfied checklist item for `bd49d35`. It is necessary but not sufficient for promotion discussion.

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
- the post-RoPE last-token artifact or value already observed
- the post-attention context last-token artifact or value already observed
- the post-attention residual last-token artifact or value already observed
- retained-state 4096-token prefill on the same bounded case
- one continuation/decode step on that retained state
- the next preferred localized continuation-token artifact: layer-0 post-attention residual
- the fallback localized continuation-token artifact only if needed: layer-0 post-attention context or post-RoPE q/k
- a short operator summary stating whether the evidence shows runtime-path propagation beyond post-attention residual or still stops at that boundary
- a short note if the broader 4100-token `restricted_prefill_trace` path failed to emit safe-side artifacts, explicitly marking that as a proof-method boundary rather than negative semantic evidence
- a clear pass/fail statement on whether the result is promotion-gating proof for `bd49d35`

## Bounded Live-Test Planning Gate

The current residual result is strong enough to reopen bounded live-test planning discussion.

That still does not justify a claim of full runtime correctness or promotion readiness. It only clears the bar for planning the next bounded live-test step.

Code promotion remains paused pending either one of the narrower downstream proofs above or an explicit decision to proceed with a bounded live-smoke anyway.
Bounded live-test planning can stay open, but it is still blocked by missing continuation-token artifact emission on the retained-state seam.

## GPU0 Live-Test Readiness

Already ready:

- Tier-2 harness workflow and wrapper chain on integration.
- Current-contract trace reuse guardrails.
- Dry-run inspection and wrapper regression coverage.
- The safe runtime-forward extraction chain already mirrored on integration.

Frontier result required to unlock the next live promotion decision:

- For `838d3f8`: no live promotion target on the current seam. Reopen only if a new runtime seam explicitly carries extra visible KV offsets, or equally direct contrary evidence appears.
- For `bd49d35`: the next accepted proof path is a true retained-state continuation seam above 4096 tokens on GPU0: retained-state 4096-token prefill, one continuation/decode step on the same case, and a localized layer-0 continuation-token artifact, preferably post-attention residual, with post-attention context or post-RoPE q/k only as fallback localized targets if needed.

Until the `bd49d35` frontier clears that bar, integration stays paused on additional runtime-forward promotion.
