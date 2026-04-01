# Workstream Harness TODO

Branch/worktree:

- `harness/tier2-workflow`
- `~/openai/worktrees/tier2-workflow`

Purpose:

- Tier-2 harness workflow
- seed capture
- same-input local replay
- live-testing workflow and operator ergonomics

Immediate next steps:

- document exact live restricted-fp16 CUDA run commands and expected artifacts
- tighten `scripts/probe_validation_tier.sh` help/output around `--seed-layers`, `--compare-mode`, and local replay
- add a short operator checklist for when to run Tier 0, Tier 1, and Tier 2
- identify any harness-only cleanup that reduces rerun cost without changing default compare behavior
- keep the Tier-2 contract docs aligned with actual harness flags and outputs

Guardrails:

- do not change default compare behavior silently
- do not re-open broad runtime semantic work here
- treat raw compare as telemetry and same-input replay as ownership proof
