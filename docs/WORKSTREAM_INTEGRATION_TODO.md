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
- land only small integration-safe harness/tooling fixes before reopening extraction from preserved stacks
- refresh branch disposition notes when `main`, `integration`, `harness`, or `runtime-forward` tips move
- re-extract any future restricted probe/oracle tooling from preserved history onto the current aligned checkpoint instead of reviving the old stack
- prune any future stale worktrees only after state is pushed or otherwise preserved
- keep `docs/REPO_ALIGNMENT_AND_WORKSTREAMS.md` and `docs/NEXT_MILESTONES.md` current when branch tips move
- avoid reopening archived debug branches as active integration work

Ready or nearly ready on integration:

- `integration/tier01-lane` commit `391a975` (`scripts/probe_validation_tier.sh` Tier-1 trace-reuse indentation fix)
- workstream/disposition doc maintenance

Still isolated, not ready for main:

- `feature/runtime-forward` runtime semantics and kernel work (`bd49d35`)
- preserved restricted probe/oracle helper history that still needs clean re-extraction

Guardrails:

- do not start speculative runtime/model work here
- do not merge noisy historical debug branches wholesale
- preserve branch history first, prune second
