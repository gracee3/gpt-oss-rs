# Repo Alignment And Workstreams

## Alignment Summary

This repo is organized around three canonical active workstreams:

1. `integration/mainline-alignment`
2. `harness/tier2-workflow`
3. `feature/runtime-forward`

The operating rule is simple:

- land safe harness, docs, validation plumbing, and narrow correctness fixes aggressively
- keep speculative runtime/semantic work isolated
- preserve exploratory history, but do not merge it wholesale

## Integration Checkpoint

Committed on `integration/mainline-alignment` now:

- safe harness wrapper chain:
  `402924a`, `ae4ef08`, `746fe7c`, `645b6eb`, `480cd08`, `ae3c9bd`, `dd050b0`, `ae5bf7d`
- safe runtime-forward extraction chain already mirrored on integration:
  `7ba163c`, `e174797`, `38f692e`
- support-plumbing/default-chain subset already mirrored on integration:
  `5c30e56`, `7bdf268`, `dfc6abe`
- branch-local integration validation/proof stack through the current tip:
  retained-boundary proof runners/docs, frontier-status/Harmony/no-trace notes,
  and `restricted_prefill_topk` capture/wiring/boundary-result commits through `3edeac8`

Deferred from this lane:

- pause additional runtime-forward cherry-picks until there is a proof-backed candidate beyond the already-mirrored safe chain
- `838d3f8`: explicitly deferred; still experimental and unjustified for promotion
- `feature/runtime-forward` still carries unresolved runtime semantics beyond the mirrored safe subset; the older YaRN runtime claim at `bd49d35` remains unproven by bounded same-input end-to-end evidence
- auxiliary proof refs/worktrees such as `proof/sink-visibility` and `runtime/bd49d35-live-smoke` are evidence-capture lanes, not merge targets

Coupling note to preserve:

- do not treat `5c30e56` as standalone; on this branch the honest mirrored subset is `5c30e56` + `7bdf268` + `dfc6abe`

Current checkpoint:

- `main`: `b4c2efa`
- `integration/mainline-alignment`: `3edeac8` (local and origin aligned)
- `harness/tier2-workflow`: `7fd5174`
- `feature/runtime-forward`: `a8248b7`

## What Landed On Main

The current shared mainline checkpoint at `b4c2efa` is the coordination/docs batch:

- workstream milestone/TODO coordination
- Tier-2 workflow and status documentation
- branch disposition tracking for the three active workstreams

These changes keep the repo operable and aligned. They do not claim that the broader restricted probe/tooling/runtime stack is already settled on `main`.

## Integration Hold State

`integration/mainline-alignment` is currently in a no-further-landings-right-now hold state at `3edeac8`.

Previously evaluated preserved clean refs do not justify a new landing by themselves; on this branch they are already represented or superseded:

- `extraction/conservative-semantic-cache` (`992a741`)
- `extraction/rope-scaling-parser-hardening` (`17396e2`)
- `extraction/rope-scaling-carry-forward` (`2c195e3`)
- `extraction/modelrunner-default` (`6ae2c95`)
- `extraction/modelrunner-architecture-fixture-cleanup` (`1b56c4b`)
- `extraction/modelrunner-conformance-fixture-carry` (`ade3bec`)

Reopen this lane for new landings only when:

- branch/docs drift requires a small coordination refresh
- a genuinely new bounded validated extraction appears that is not already represented or superseded on this branch

Until one of those conditions is true:

- require a fresh extraction decision before promoting newer retained-boundary, Harmony, or `restricted_prefill_topk` work
- do not present the full integration tip as already ready for `main`

## What Stays On Integration

`integration/mainline-alignment` is the branch reserved for:

- small validated batches queued for promotion to `main`
- follow-up cherry-picks that are coherent but not yet ready for direct mainline landing
- post-merge validation batches
- cleanup needed to keep archived branches and active worktrees understandable

Current held work on integration:

- hold/preservation mode for the current tip unless real drift or a genuinely new clean extraction appears
- branch/disposition doc maintenance and coordination refreshes
- retained-boundary, frontier-proof, Harmony, and no-trace validation docs/scripts kept for integration evidence capture
- `restricted_prefill_topk` capture, worker-config wiring, and recorded boundary result through `3edeac8`

Not ready for mainline from preserved/runtime history:

- `archive/harness-tier2-workflow-prealign-2026-04-01` as a whole branch; it mixes useful workflow work with code changes and needs fresh extraction
- YaRN RoPE runtime changes
- fp16 QKV / RoPE kernel fixes
- restricted probe/oracle helper stack that still sits on a wider noisy history and needs fresh extraction against the aligned checkpoint
- the `feature/runtime-forward` branch tip at `a8248b7` and the unresolved semantic/runtime stack behind `838d3f8` and `bd49d35`

## What Was Archived Or Preserved Only

The following branches were preserved and pushed as historical or reference lanes, but are not the canonical active workstreams:

- `archive/harness-tier2-workflow-prealign-2026-04-01`
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
- `integration/probe-enablement`
- `integration/tier01-lane`

## What Was Archived Or Left As History

The following branch families are historical reference unless explicitly reopened:

- `debug/*`
- `gpt-oss/*first-case`
- `gpt-oss/sink-*`
- `gpt-oss/deferred-*`

They contain useful investigation history, but their durable knowledge should live in docs and extracted commits rather than wholesale merges.

## Auxiliary Local Proof Lanes

The following refs/worktrees exist for bounded evidence capture only and are not canonical active workstreams:

- `proof/sink-visibility` at `891a487`
- `runtime/bd49d35-live-smoke` at `fa089a7`
- detached `/tmp/runtime-*` proof worktrees exist locally; use the prune-later inventory below to distinguish branch-backed later-prune copies from still-ambiguous unique SHAs

## Prune-Later Inventory

Must remain preserved:

- the three canonical active workstreams
- `integration/probe-enablement` as a historical integration breadcrumb; details live in `docs/integration_plan.md`
- the preserved clean `extraction/*` refs listed in the integration hold-state docs
- `proof/sink-visibility` and `runtime/bd49d35-live-smoke` as branch-backed auxiliary evidence lanes
- canonical preserved pointer for the `bd49d35` replay/runtime stack: `archive/replay-probe-enablement-proven-fixes`

Candidate for later pruning, but not from this pass:

- detached `/tmp/runtime-defaultsplit-*`, `/tmp/runtime-modelrunner-base-*`, `/tmp/runtime-rope-base-*`, and `/tmp/runtime-rope-integbase-*` worktrees because their tips are already backed by named `extraction/*` refs
- detached `/tmp/runtime-yarn-long-safe-*`, `/tmp/runtime-yarn-proofsafe-*`, and `/tmp/runtime-yarn-variant-*` worktrees because their tips are already backed by `extraction/modelrunner-conformance-fixture-carry` (`ade3bec`)
- detached `/tmp/runtime-rope-carry-*` because `838d3f8` is already preserved in branch-backed proof/runtime lanes
- the branch-backed proof worktrees for `proof/sink-visibility` and `runtime/bd49d35-live-smoke` once their doc breadcrumbs remain intact
- duplicate `bd49d35` aliases such as `debug/replay-k-seam-audit-20260331`, `replay/probe-enablement-proven-fixes`, and `safety/pre-cleanup-2026-04-01` once the canonical preserved pointer and docs breadcrumb remain intact

Ambiguous and should not be touched yet:

- local `harmony` at `ef7d7ff`; it is a local-only alias and is not yet dispositioned in coordination docs
- detached `/tmp/runtime-yarn-proofsafe-gpu0` at `e7d767e`
- detached `/tmp/runtime-yarn-proofvar-gpu0` at `4e3ddb4`
- detached `/tmp/runtime-yarn-safeproof-base-*` at `01126e9`
- any other detached `/tmp/runtime-*` worktree whose tip is not already backed by a named branch or explicitly breadcrumbed in docs

## Cleanup Pass Order

1. Reconfirm prerequisites before any prune:
   the hold state still stands on `integration/mainline-alignment`,
   this doc still names `archive/replay-probe-enablement-proven-fixes` as the canonical `bd49d35` pointer,
   `integration/probe-enablement` still has its breadcrumb in `docs/integration_plan.md`,
   and the preserved `extraction/*`, `proof/sink-visibility`, and `runtime/bd49d35-live-smoke` refs still exist.
   Operational note:
   if one of the listed `/tmp/runtime-*` paths is already missing on disk but still appears in `git worktree list` as `prunable`,
   remove that stale registration selectively with `git worktree remove -f <path>`.
   Do not start with broad `git worktree prune` while ambiguous missing `/tmp/runtime-*` entries still exist,
   because it would also sweep items that are still blocked on manual triage.

2. Prune detached extraction-backed copies first:
   `/tmp/runtime-defaultsplit-*`,
   `/tmp/runtime-modelrunner-base-*`,
   `/tmp/runtime-rope-base-*`,
   `/tmp/runtime-rope-integbase-*`,
   `/tmp/runtime-yarn-long-safe-*`,
   `/tmp/runtime-yarn-proofsafe-*`,
   and `/tmp/runtime-yarn-variant-*`.
   Preconditions:
   the corresponding named `extraction/*` refs remain in this ledger and still resolve,
   and each candidate path is removed one at a time so only the intended stale registration or worktree disappears.

3. Prune detached branch-backed experimental copies next:
   `/tmp/runtime-rope-carry-*`.
   Preconditions:
   `838d3f8` is still reachable from at least one branch-backed preserved lane,
   currently `feature/runtime-forward`, `proof/sink-visibility`, or `runtime/bd49d35-live-smoke`.

4. Prune branch-backed proof worktrees after the detached copies:
   the local worktrees for `proof/sink-visibility` and `runtime/bd49d35-live-smoke`.
   Preconditions:
   their branch refs still exist,
   this ledger still names them as preserved auxiliary evidence lanes,
   and `docs/WORKSTREAM_RUNTIME_TODO.md` still lists them as auxiliary proof refs.

5. Collapse duplicate `bd49d35` aliases only after the canonical pointer is unchanged:
   `debug/replay-k-seam-audit-20260331`,
   `replay/probe-enablement-proven-fixes`,
   and `safety/pre-cleanup-2026-04-01`.
   Preconditions:
   `archive/replay-probe-enablement-proven-fixes` remains the canonical preserved pointer in this doc,
   and no remaining breadcrumb depends on one of the duplicate aliases by name.

6. Stop and manually triage anything still ambiguous:
   local `harmony`,
   `e7d767e`,
   `4e3ddb4`,
   `01126e9`,
   and any other detached `/tmp/runtime-*` tip without a named ref or docs breadcrumb.
   No cleanup pass should touch those items until they are either breadcrumbed or intentionally discarded.

## What Was Pruned

The following local worktrees were pruned after their branch state was merged, pushed, or otherwise preserved:

- stale registration `/tmp/runtime-defaultsplit-eVErYU` removed after confirming `extraction/modelrunner-architecture-fixture-cleanup` still preserved `1b56c4b`
- stale registration `/tmp/runtime-modelrunner-base-hJm1gb` removed after confirming `extraction/modelrunner-conformance-fixture-carry` still preserved `ade3bec`
- stale registration `/tmp/runtime-rope-base-IAJvP9` removed after confirming `extraction/rope-scaling-parser-hardening` still preserved `17396e2`
- stale registration `/tmp/runtime-rope-integbase-iZXsQZ` removed after confirming `extraction/rope-scaling-carry-forward` still preserved `2c195e3`
- stale registration `/tmp/runtime-yarn-long-safe-7738` removed after confirming `extraction/modelrunner-conformance-fixture-carry` still preserved `ade3bec`
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
- Current tip: `3edeac8`
- Scope:
  - post-merge hygiene
  - remaining safe extraction and validation batches
  - branch/archive cleanup

### 2. Harness / Live Testing / Contract Follow-Up

- Branch: `harness/tier2-workflow`
- Worktree: `~/openai/worktrees/tier2-workflow`
- TODO file: `docs/WORKSTREAM_HARNESS_TODO.md`
- Current tip: `7fd5174`
- Scope:
  - seed-capture and local-replay ergonomics
  - compare-mode and live-testing workflow polish
  - representative sentinel-layer reruns and harness-only improvements

### 3. Forward Implementation / Runtime Work

- Branch: `feature/runtime-forward`
- Worktree: `~/openai/worktrees/runtime-forward`
- TODO file: `docs/WORKSTREAM_RUNTIME_TODO.md`
- Current tip: `a8248b7`
- Scope:
  - runtime/semantic implementation work that is still incomplete
  - selective extraction from deeper investigative branches
  - future integration candidates that are not yet mainline-safe

## Worktree Policy

- Keep only the minimum active set of named canonical worktrees.
- The root checkout serves as the canonical `integration/mainline-alignment` worktree; auxiliary proof/scratch worktrees may exist temporarily, but they are not canonical lanes and should stay clearly labeled/prunable.
- Push or otherwise preserve important branches before pruning worktrees.
- Do not delete dirty or unpushed work without an explicit preservation step.
- Prefer archive branches or untouched historical refs over merging noisy debug stacks.
- `stash@{0}` is preserved local state for coordination-doc alternatives; do not drop it unless that draft is intentionally reconciled or discarded.
