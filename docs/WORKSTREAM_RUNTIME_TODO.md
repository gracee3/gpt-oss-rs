# Workstream Runtime TODO

Branch/worktree:

- `feature/runtime-forward`
- `~/openai/worktrees/runtime-forward`

Purpose:

- isolated runtime/semantic implementation work
- forward engineering not yet ready for mainline

Immediate next steps:

- keep runtime-forward intentionally separate from aligned mainline branches
- validate any candidate fix against the Tier-2 contract before proposing promotion
- mine archived replay/debug branches selectively when specific prior evidence is needed
- record any new runtime hypotheses as explicit TODOs rather than implied settled behavior

Guardrails:

- no default compare behavior changes
- no promotion to aligned branches without validation
- no “real runtime bug” claim without same-input local replay surviving
