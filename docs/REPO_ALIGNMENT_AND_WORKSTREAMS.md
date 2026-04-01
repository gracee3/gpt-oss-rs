# Repo Alignment And Workstreams

## Alignment Summary

The repo is now organized around exactly three active workstreams:

1. `integration/mainline-alignment`
2. `harness/tier2-workflow`
3. `feature/runtime-forward`

The policy is:

- land safe docs and workflow clarification on `main`
- stage broader harness/tooling extraction on `integration/mainline-alignment`
- keep unfinished runtime/semantic work isolated from mainline hygiene

## What Was Merged To Main

`main` now carries the canonical Tier-2 docs:

- `docs/TIER2_FP16_CUDA_WORKFLOW.md`
- `docs/TIER2_RESULTS_AND_STATUS.md`
- README/doc index pointers to the Tier-2 workflow and status docs

This is an intentional documentation landing, not a claim that the runtime root cause is settled.

## What Was Staged On Integration

`integration/mainline-alignment` is the staging lane for the safe extracted harness/tooling stack that is broader than a docs-only mainline landing.

Current staged content on that branch includes:

- restricted prefill activation trace probe plumbing
- restricted model-view and oracle compare helpers
- tiered validation script
- representative parity/supporting cases already used by the investigation
- Tier-2 seed-capture and same-input local-replay workflow plumbing

The integration lane exists so these changes can stay coherent and testable without pretending they all belong on `main` yet.

## What Was Archived Or Preserved Only

The following branches were preserved and pushed as historical or reference lanes, but are not the new canonical active workstreams:

- `debug/probe-enablement-k-rope-preserve-20260330`
- `debug/replay-k-seam-audit-20260331`
- `replay/probe-enablement-proven-fixes`
- `codex/cleanup-2026-04-01`
- `gpt-oss/deferred-frontier-recon`
- `gpt-oss/deferred-frontier-shape-audit`
- `gpt-oss/full-attention-next-case`
- `gpt-oss/graph-first-case`
- `gpt-oss/sink-first-case`
- `gpt-oss/sink-gating-fix`
- `gpt-oss/sliding-attention-first-case`
- `gpt-oss/tier2-warm-oracle-core`
- `integration/tier01-lane`

These branches contain useful history, but they should be mined selectively rather than merged wholesale.

## Active Workstreams

### Mainline Hygiene / Integration Alignment

- Branch: `integration/mainline-alignment`
- Worktree: canonical active checkout
- Scope:
  - safe extraction/cherry-picks
  - merge-risk reduction
  - integration validation batches
  - branch/worktree hygiene and docs upkeep

### Harness / Live Testing / Contract Follow-Up

- Branch: `harness/tier2-workflow`
- Worktree: canonical active checkout
- Scope:
  - Tier-2 contract follow-up
  - seed-capture and local-replay ergonomics
  - live-testing workflow polish
  - representative sentinel-layer reruns

### Forward Implementation / Feature / Integration

- Branch: `feature/runtime-forward`
- Worktree: canonical active checkout
- Scope:
  - unfinished runtime/semantic work
  - future selective extraction from preserved branches
  - integration candidates that are not yet ready for mainline promotion

## Worktree Policy

- Keep only the minimum active worktree set.
- Push important branch state before pruning local worktrees.
- Do not delete dirty or unpushed work without an explicit preservation step.
- Prefer extracted commits and docs over long-lived noisy debug merges.
