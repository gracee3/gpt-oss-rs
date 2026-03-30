# tier2-warm-oracle-core Status

## Current Worktree / Branch
- Repo: `/home/emmy/openai/worktrees/tier2-warm-oracle-core`
- Branch: `gpt-oss/tier2-warm-oracle-core`
- Base branch: `integration/probe-enablement`

## Scope Implemented in this Slice
- Added persistent warm-session runner path in
  `crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py`
  (`--listen`, in-memory model/session reuse, batch mode, explicit compare-mode metadata).
- Added strict trace reuse-keying and explicit reuse decision output in
  `scripts/probe_validation_tier.sh`:
  - key now includes: model path, artifact hash, prompt/token identity, max-model-len,
    compare mode, trace schema marker, oracle schema marker.
  - decision now prints `reuse_trace=<true|false> reason=<...>` and blocks mismatches.
- Added operator entrypoint
  `scripts/oracle_debug_loop.sh`
  with:
  - `--warm-oracle` for repeated compare loops in a single persistent process
  - `--reuse-trace` and `--mode {full,fast}`
  - benchmark output with per-run timings and mean/median.
- Added compare-mode flag plumbing (`full|fast`) to preserve default behavior while enabling
  debug-only fast mode.

## Validation captured so far
- Warm repeated compare (`--warm-oracle`) 5x on same trace (same process):
  - run ms: `74482, 7012, 7398, 7149, 7201`
  - mean: `20648.400 ms`
  - median: `7201.000 ms`
- Non-warm repeated compare (5x separate Python runs) is in progress/partly observed:
  `77714, 82241, 88009, 96830, 95287 ms` (mean `88016.200`, median `88009.000`).
- Match/mismatch reuse checks:
  - Full-key match path has been exercised in earlier runs and showed `reason=full_key_match`.
  - Mismatch path was being validated when previous command was interrupted; this remains to be
    re-run now to capture a fresh terminal log line in this continuation.

## What’s not yet committed
- A final command-level confirmation record for:
  - explicit reuse mismatch reason from `probe_validation_tier.sh` in the same run.
  - default `full` mode parity output message for the canonical path in this exact environment.

## Pending Next Actions (where left off)
1. Re-run `probe_validation_tier.sh --compare-only` mismatch case with a deliberately altered
   prompt/len flag and capture the `reuse_trace=false reason=...` output to close the strict reuse
   acceptance criterion.
2. Optionally re-run full 5x non-warm baseline immediately before final handoff if reproducibility is
   desired with the same environment/inputs.
3. After those two confirmations, produce final structured report output requested by the workstream plan
   and prepare the branch for handoff.
