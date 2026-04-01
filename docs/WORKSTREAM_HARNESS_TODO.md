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
- added an opt-in warm-oracle listener scaffold so multiple compare requests can reuse one Python oracle session without changing one-shot defaults

Immediate next steps:

- decide whether the warm-oracle listener should stay as a direct tool entrypoint or gain a thin shell wrapper after real operator use
- consider a bounded performance-only short-circuit for explicit local replay requests only if it stays opt-in and does not change default behavior
- keep script output and docs aligned if additional operator-facing flags are added
- add any further example commands only where they reduce real operator ambiguity
- keep the Tier-2 contract docs aligned with actual harness flags and outputs

Guardrails:

- do not change default compare behavior silently
- do not re-open broad runtime semantic work here
- treat raw compare as telemetry and same-input replay as ownership proof
