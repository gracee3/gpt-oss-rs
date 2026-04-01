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
- wired an opt-in `--warm-oracle` shell path that exercises two compare requests in one listener session and emits a reuse-check artifact
- added metadata sidecars plus fail-closed provenance checks so trace reuse only happens when key capture/replay inputs still match
- distinguished current wrapper-captured trace sidecars from legacy-unversioned artifacts with an explicit wrapper-owned capture contract marker
- added a dry-run trace inspection mode so operators can see reuse classification and exact provenance reasons before running compare-only or recapture paths
- added `scripts/test_probe_validation_wrapper.sh` so the wrapper-only provenance/inspection/warm-oracle paths can be rerun without ad hoc temp-shell setup
- added an opt-in strict mode that requires the current wrapper-owned trace capture contract instead of allowing legacy artifact reuse

Immediate next steps:

- decide whether `--warm-oracle` should stay as a bounded reuse-check path or grow into a more general multi-request operator flow after real use
- consider a bounded performance-only short-circuit for explicit local replay requests only if it stays opt-in and does not change default behavior
- keep trace/oracle sidecar metadata narrow and honest; do not turn it into a broad batching protocol
- keep script output and docs aligned if additional operator-facing flags are added
- add any further example commands only where they reduce real operator ambiguity
- keep the Tier-2 contract docs aligned with actual harness flags and outputs

Guardrails:

- do not change default compare behavior silently
- do not re-open broad runtime semantic work here
- treat raw compare as telemetry and same-input replay as ownership proof
