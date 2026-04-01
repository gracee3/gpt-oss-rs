# Repo Alignment And Workstreams

## Alignment Summary

This repo is organized around three active workstreams only:

1. `integration/mainline-alignment`
2. `harness/tier2-workflow`
3. `feature/runtime-forward`

The operating rule is simple:

- land safe harness, docs, validation plumbing, and narrow correctness fixes aggressively
- keep speculative runtime/semantic work isolated
- preserve exploratory history, but do not merge it wholesale

Current checkpoint:

- `main`: `b4c2efa`
- `integration/mainline-alignment`: `b4c2efa`
- `harness/tier2-workflow`: `b4c2efa`
- `feature/runtime-forward`: `bd49d35`

## What Landed On Main

The current mainline landing batch is the safe probe/harness stack:

- restricted prefill trace probe and restricted logit diff entrypoints
- restricted model-view generator and oracle compare helpers
- tiered validation script and Tier-2 workflow docs
- bounded decode and representative parity cases that support harness validation
- Tier-2 compare-mode, seed-capture, and same-input local replay workflow

These changes are intended to make live testing and future extraction disciplined. They do not claim a settled runtime root cause.

## What Stays On Integration

`integration/mainline-alignment` is the branch reserved for:

- follow-up cherry-picks that are coherent but not yet ready for direct mainline landing
- post-merge validation batches
- cleanup needed to keep archived branches and active worktrees understandable

Current held work on integration:

- ready for main after narrow validation: `integration/tier01-lane` commit `391a975` (`scripts/probe_validation_tier.sh` Tier-1 trace-reuse indentation fix)
- doc updates that keep workstream ownership and branch disposition current

Not ready for mainline from preserved/runtime history:

- YaRN RoPE runtime changes
- fp16 QKV / RoPE kernel fixes
- restricted probe/oracle helper stack that still sits on a wider noisy history and needs fresh extraction against the aligned checkpoint
- the `feature/runtime-forward` branch tip at `bd49d35` and its ancestor stack

## What Was Archived Or Preserved Only

The following branches were preserved and pushed as historical or reference lanes, but are not the canonical active workstreams:

- `archive/replay-probe-enablement-proven-fixes`
- `codex/cleanup-2026-04-01`
- `debug/probe-enablement-k-rope-preserve-20260330`
- `debug/replay-k-seam-audit-20260331`
- `gpt-oss/deferred-frontier-recon`
- `gpt-oss/deferred-frontier-shape-audit`
- `gpt-oss/full-attention-next-case`
- `gpt-oss/graph-first-case`
- `gpt-oss/sink-first-case`
- `gpt-oss/sink-gating-fix`
- `gpt-oss/sliding-attention-first-case`
- `gpt-oss/tier2-warm-oracle-core`
- `integration/tier01-lane`

## What Was Archived Or Left As History

The following branch families are historical reference unless explicitly reopened:

- `debug/*`
- `gpt-oss/*first-case`
- `gpt-oss/sink-*`
- `gpt-oss/deferred-*`

They contain useful investigation history, but their durable knowledge should live in docs and extracted commits rather than wholesale merges.

## What Was Pruned

The following local worktrees were pruned after their branch state was merged, pushed, or otherwise preserved:

- `/home/emmy/openai/worktrees/codex-cleanup-ops`
- `/home/emmy/openai/worktrees/deferred-doc-curation`
- `/home/emmy/openai/worktrees/deferred-frontier-recon`
- `/home/emmy/openai/worktrees/deferred-frontier-shape-audit`
- `/home/emmy/openai/worktrees/full-attention-next-case`
- `/home/emmy/openai/worktrees/graph-first-case`
- `/home/emmy/openai/worktrees/integration-tier01-lane-repro`
- `/home/emmy/openai/worktrees/replay-k-seam-audit-20260331`
- `/home/emmy/openai/worktrees/sink-first-case`
- `/home/emmy/openai/worktrees/sink-gating-fix`
- `/home/emmy/openai/worktrees/sliding-attention-first-case`
- `/home/emmy/openai/worktrees/tier2-warm-oracle-core`

## Active Workstreams

### 1. Mainline Hygiene / Integration Alignment

- Branch: `integration/mainline-alignment`
- Worktree: `~/openai/gpt-oss-rs`
- TODO file: `docs/WORKSTREAM_INTEGRATION_TODO.md`
- Current tip: `b4c2efa`
- Scope:
  - post-merge hygiene
  - remaining safe extraction and validation batches
  - branch/archive cleanup

### 2. Harness / Live Testing / Contract Follow-Up

- Branch: `harness/tier2-workflow`
- Worktree: `~/openai/worktrees/tier2-workflow`
- TODO file: `docs/WORKSTREAM_HARNESS_TODO.md`
- Current tip: `b4c2efa`
- Scope:
  - seed-capture and local-replay ergonomics
  - compare-mode and live-testing workflow polish
  - representative sentinel-layer reruns and harness-only improvements

### 3. Forward Implementation / Runtime Work

- Branch: `feature/runtime-forward`
- Worktree: `~/openai/worktrees/runtime-forward`
- TODO file: `docs/WORKSTREAM_RUNTIME_TODO.md`
- Current tip: `bd49d35`
- Scope:
  - runtime/semantic implementation work that is still incomplete
  - selective extraction from deeper investigative branches
  - future integration candidates that are not yet mainline-safe

## Worktree Policy

- Keep only the minimum active set of named worktrees.
- The root checkout serves as the active `integration/mainline-alignment` worktree so the repo ends with exactly three active worktrees total.
- Push or otherwise preserve important branches before pruning worktrees.
- Do not delete dirty or unpushed work without an explicit preservation step.
- Prefer archive branches or untouched historical refs over merging noisy debug stacks.
