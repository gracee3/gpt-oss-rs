# Workstream Harness TODO

Branch/worktree:

- `harness/tier2-workflow`
- `~/openai/worktrees/tier2-workflow`

Purpose:

- Tier-2 harness workflow
- seed capture
- same-input local replay
- live-testing workflow and operator ergonomics

Completed in this pass:

- tightened `scripts/probe_validation_tier.sh` help text with operator-first examples
- added bounded validation for local replay flag combinations in the wrapper script
- documented a representative sentinel-layer operator flow and a short live-testing checklist

Immediate next steps:

- consider a bounded performance-only short-circuit for explicit local replay requests if it stays opt-in and does not change default behavior
- keep script output and docs aligned if additional operator-facing flags are added
- add any further example commands only where they reduce real operator ambiguity
- keep the Tier-2 contract docs aligned with actual harness flags and outputs

Guardrails:

- do not change default compare behavior silently
- do not re-open broad runtime semantic work here
- treat raw compare as telemetry and same-input replay as ownership proof
