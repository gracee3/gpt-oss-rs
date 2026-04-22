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
- added `scripts/test_run_gpu0_live_smoke.sh` so the bounded GPU0 live-smoke runner and `restricted_prefill_topk` operator path have a no-GPU regression check for setup/run summaries
- scoped `run_staged_boundary_smoke.sh` proof-artifact env plumbing to the continuation stage and added `scripts/test_run_staged_boundary_smoke.sh` as a no-GPU regression for the staged setup/run summaries
- added `scripts/test_run_yarn_long_context_proof.sh` so the YaRN long-context proof runner has a no-GPU regression for prompt/setup outputs, per-side proof env plans, and compact proof-artifact comparison summaries
- aligned `run_retained_continuation_proof.sh` so `restricted_logit_diff` run invocations actually receive the forced-output tokens already advertised in the generated run plans, and added `scripts/test_run_retained_continuation_proof.sh` for a no-GPU regression over tokenization, progress markers, and proof-artifact summaries

Immediate next steps:

- treat this lane as quiet maintenance unless an operator-facing flag, help surface, artifact contract, or new harness seam changes
- keep the current docs/help/examples aligned if any operator-facing flags or artifact sidecars change
- keep `--warm-oracle` bounded unless real operator use justifies a broader multi-request flow
- keep trace/oracle sidecar metadata narrow and honest; do not turn it into a broad batching protocol
- use `scripts/test_probe_validation_wrapper.sh` as the lightweight wrapper regression check after future harness-only polish
- use `scripts/test_run_gpu0_live_smoke.sh` after live-smoke runner or `restricted_prefill_topk` operator-surface edits
- use `scripts/test_run_staged_boundary_smoke.sh` after staged boundary runner or continuation proof-artifact surface edits
- use `scripts/test_run_yarn_long_context_proof.sh` after long-context proof runner or compact proof-artifact summary edits
- use `scripts/test_run_retained_continuation_proof.sh` after retained continuation runner, forced-output-token plumbing, or progress-marker summary edits

Bounded coverage ledger:

- `scripts/probe_validation_tier.sh` -> `scripts/test_probe_validation_wrapper.sh`
  verifies trace metadata/provenance inspection, legacy-vs-current capture classification, strict current-contract rejection, and bounded warm-oracle reuse in test mode
- `scripts/run_gpu0_live_smoke.sh` -> `scripts/test_run_gpu0_live_smoke.sh`
  verifies setup/run summaries, env and command plans, and the `restricted_prefill_topk` operator path on a fake tree
- `scripts/run_staged_boundary_smoke.sh` -> `scripts/test_run_staged_boundary_smoke.sh`
  verifies staged setup/run summaries and keeps continuation proof-artifact env plumbing scoped off the prefix stage
- `scripts/run_yarn_long_context_proof.sh` -> `scripts/test_run_yarn_long_context_proof.sh`
  verifies prompt/setup outputs, per-side proof env/command plans, compact proof-artifact selection, and `vector_diff` summary generation
- `scripts/run_retained_continuation_proof.sh` -> `scripts/test_run_retained_continuation_proof.sh`
  verifies a bounded retained slice: tokenization summary, forced-output-token plumbing for `restricted_logit_diff`, one marker-profile progress summary, and compact proof-artifact comparison

Guardrails:

- do not change default compare behavior silently
- do not re-open broad runtime semantic work here
- keep `scripts/probe_validation_tier.sh` as the only same-input local replay ownership gate before any runtime-defect claim
- treat raw compare as telemetry and same-input replay as ownership proof
