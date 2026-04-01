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

Immediate next steps:

- keep the root checkout on `integration/mainline-alignment`
- promote only narrow validated batches from active lanes
- prune any future stale worktrees only after state is pushed or otherwise preserved
- keep `docs/REPO_ALIGNMENT_AND_WORKSTREAMS.md` and `docs/NEXT_MILESTONES.md` current when branch tips move
- avoid reopening archived debug branches as active integration work

Guardrails:

- do not start speculative runtime/model work here
- do not merge noisy historical debug branches wholesale
- preserve branch history first, prune second
