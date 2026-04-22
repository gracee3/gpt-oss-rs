# Workstream Integration TODO

Branch/worktree:

- `integration/mainline-alignment`
- `~/openai/gpt-oss-rs`

Purpose:

- safe landings
- selective extraction
- repo hygiene
- docs alignment
- integration validation

Checkpoint:

- lane state:
  no further landings right now; treat this branch as a hold/preservation lane unless real drift or a genuinely new bounded extraction appears
- committed here now:
  mirrored harness/operator chain through strict current-contract mode
  mirrored conservative runtime-forward extraction chain plus the coupled `5c30e56` + `7bdf268` + `dfc6abe` support subset
  integration validation/proof stack through retained-boundary, frontier/Harmony, and `restricted_prefill_topk` commits at `3edeac8`
- intentionally deferred from this lane:
  no additional runtime-forward cherry-picks until a proof-backed candidate exists beyond the already-mirrored safe chain
  `838d3f8` stays explicitly experimental and deferred
  `feature/runtime-forward` beyond the mirrored safe subset, including the older `bd49d35` YaRN runtime claim, stays isolated
- current proof note:
  `feature/runtime-forward` tip is now `a8248b7`, but the unresolved runtime semantic boundary still traces back through `838d3f8` and `bd49d35`

Current status:

- `main`: `b4c2efa`
- `integration/mainline-alignment`: `3edeac8` (local and origin aligned)
- `harness/tier2-workflow`: `7fd5174`
- `feature/runtime-forward`: `a8248b7`

Immediate next steps:

- keep the root checkout on `integration/mainline-alignment`
- keep the lane in hold/preservation mode unless real drift or a genuinely new bounded validated extraction appears
- if reopened, promote only narrow validated batches from active lanes
- keep new runtime-forward intake paused except for explicitly justified clean extractions
- do not manufacture new promotion slices from preserved refs that are already represented or superseded on this branch
- keep the current integration validation/proof stack branch-local until a smaller extraction is justified
- land only small integration-safe harness/tooling fixes before reopening extraction from preserved stacks
- refresh branch disposition notes when `main`, `integration`, `harness`, or `runtime-forward` tips move
- re-extract any future restricted probe/oracle tooling from preserved history onto the current aligned checkpoint instead of reviving the old stack
- prune any future stale worktrees only after state is pushed or otherwise preserved
- use `docs/REPO_ALIGNMENT_AND_WORKSTREAMS.md` as the canonical preservation ledger and cleanup playbook; the canonical `bd49d35` preserved pointer is `archive/replay-probe-enablement-proven-fixes`, branch-backed scratch state is a later-prune candidate, and local `harmony` plus detached unique proof SHAs stay untouched until breadcrumbed or otherwise preserved
- keep `docs/REPO_ALIGNMENT_AND_WORKSTREAMS.md` and `docs/NEXT_MILESTONES.md` current when branch tips move
- avoid reopening archived debug branches as active integration work
- treat `archive/harness-tier2-workflow-prealign-2026-04-01` as a commit-mining source, not a merge target

Reopen conditions for new landings:

- real branch/docs drift that requires a small coordination refresh
- a genuinely new bounded validated extraction that is not already represented or superseded on this branch

Previously evaluated preserved refs that do not justify reopening by themselves:

- `992a741`
- `17396e2`
- `2c195e3`
- `6ae2c95`
- `1b56c4b`
- `ade3bec`

Still isolated, not ready for main:

- `feature/runtime-forward` runtime semantics and branch-local proof/status stack through `a8248b7`, especially `838d3f8` and `bd49d35`
- `archive/harness-tier2-workflow-prealign-2026-04-01` as a whole branch
- `proof/sink-visibility`, `runtime/bd49d35-live-smoke`, and detached `/tmp/runtime-*` proof worktrees
- preserved restricted probe/oracle helper history that still needs clean re-extraction

Guardrails:

- do not start speculative runtime/model work here
- do not take new runtime-forward cherry-picks until a proof-backed candidate exists beyond the already-mirrored safe chain
- do not merge noisy historical debug branches wholesale
- preserve branch history first, prune second
- do not present preserved probe/runtime history as already landed on `main`
- preserve the coupling note that `5c30e56` is not standalone on this branch without `7bdf268` + `dfc6abe`
- do not treat auxiliary proof branches/worktrees as integration merge targets
