# Integration Plan

## Baseline
- Commit: `917b680`
- Status: `main` is the known-good conformance baseline. `cargo test -p gpt-oss-conformance` passes `43/43`.
- Oracle/traced baseline:
  - `restricted_prefill_trace`: unavailable on baseline `main` because the binary is not present there
  - oracle diff artifacts: unavailable on baseline `main` for the same reason
  - consequence: code branches that require the restricted prefill/oracle probe to validate safely were not promoted in this pass

## Branch Inventory

| Branch | Classification | Purpose | Action | Notes |
|--------|---------------|--------|--------|-------|
| `gpt-oss/deferred-doc-curation` | `HIGH_VALUE_MERGE` | narrow deferred-frontier note | merged | doc-only, low risk, validated neutral on `main` |
| `gpt-oss/deferred-frontier-recon` | `REFERENCE_ONLY` | longer deferred-frontier recon note | keep branch | useful context, not needed on `main` |
| `gpt-oss/deferred-frontier-shape-audit` | `REFERENCE_ONLY` | deferred frontier shape audit | keep branch | reference material, not a promotion target |
| `gpt-oss/full-attention-next-case` | `CHERRY_PICK` | validated probe + parity + deep oracle debugging lane | do not merge whole branch | contains high-value commits mixed with large traced-lane/runtime edits; needs selective extraction against a probe-enabled integration branch |
| `gpt-oss/graph-first-case` | `DO_NOT_MERGE` | first graph replay case | do not merge | deferred frontier; out of current trusted scope |
| `gpt-oss/sink-first-case` | `DO_NOT_MERGE` | sliding/sink first-case lane | do not merge | deferred frontier, superseded by later sink draft branch |
| `gpt-oss/sink-gating-fix` | `REFERENCE_ONLY` | wider sink/sliding draft + server/reference edits | keep branch | too wide for blind promotion; needs selective extraction if reopened |
| `gpt-oss/sliding-attention-first-case` | `REFERENCE_ONLY` | sliding attention vertical slice + tests/docs | keep branch | useful deferred reference, not safe for current `main` |

## Validation Baseline

### Pre-merge
- `main` HEAD before integration: `917b680`
- `cargo test -p gpt-oss-conformance`: pass (`43/43`)
- `restricted_prefill_trace`: not runnable on baseline `main`
- oracle diff: not runnable on baseline `main`
- Earliest divergence stage: not measurable on baseline `main` without the restricted probe surface

## Merge Log

### `gpt-oss/deferred-doc-curation`
- Action: merged through `integration/deferred-doc-curation`, then merged into `main`
- Result: neutral
- Validation:
  - `cargo test -p gpt-oss-conformance`: pass (`43/43`) on integration branch and on final `main`
- Notes: doc-only branch; safe to promote without widening runtime scope

### `gpt-oss/deferred-frontier-recon`
- Action: not merged
- Result: held as `REFERENCE_ONLY`
- Notes: reference note, not needed on `main`

### `gpt-oss/deferred-frontier-shape-audit`
- Action: not merged
- Result: held as `REFERENCE_ONLY`
- Notes: shape audit is useful context but not a promotion target

### `gpt-oss/full-attention-next-case`
- Action: not merged
- Result: reclassified as `CHERRY_PICK`
- Notes: contains valuable narrow fixes and probes, but the branch is too wide to merge safely without first standing up the same restricted trace/oracle validation surface on an integration branch

### `gpt-oss/graph-first-case`
- Action: not merged
- Result: `DO_NOT_MERGE`
- Notes: deferred graph frontier remains out of current scope

### `gpt-oss/sink-first-case`
- Action: not merged
- Result: `DO_NOT_MERGE`
- Notes: deferred frontier and superseded by broader sink draft work

### `gpt-oss/sink-gating-fix`
- Action: not merged
- Result: `REFERENCE_ONLY`
- Notes: wide mixed draft; would require selective extraction

### `gpt-oss/sliding-attention-first-case`
- Action: not merged
- Result: `REFERENCE_ONLY`
- Notes: useful deferred vertical slice, not safe to promote into current trusted baseline

## Final State
- Final `main` after integration: `8313225`
- Promoted work:
  - `DEFERRED_FRONTIER.md`
- Post-integration validation:
  - `cargo test -p gpt-oss-conformance`: pass (`43/43`)
- Regressions observed: none
- Current oracle/traced frontier on `main`: still not directly measurable because `main` does not yet contain the restricted prefill/oracle probe toolchain

## Probe Enablement Plan

| Item | Needed? | Source branch/commit | Reason | Notes |
|------|---------|----------------------|--------|-------|
| `restricted_prefill_trace` bench binary | yes | `gpt-oss/full-attention-next-case` `ea3bc93` | restores smallest useful restricted trace entrypoint | highest-priority extraction target |
| `GpuWorker::debug_runner_prefill_trace` plumbing | yes | `gpt-oss/full-attention-next-case` `ea3bc93` | allows bench binary to invoke trace surface through existing worker setup | probe-only entrypoint |
| `PrefillActivationTrace` / `PrefillLayerTrace` runner structs and `debug_prefill_trace` | yes | `gpt-oss/full-attention-next-case` `ea3bc93` | minimal useful trace surface for embedding + per-layer outputs | broadest dependency in first extraction |
| oracle helper script updates | maybe | `gpt-oss/full-attention-next-case` `0eef46e` | improves later oracle-side comparison | defer unless trace surface is working |
| `restricted_logit_diff` | no | `gpt-oss/full-attention-next-case` `ae9a5bd` | lower-priority differential tool | not needed for first viable milestone |
| `restricted_oracle_prefill.py` | no | `gpt-oss/full-attention-next-case` `f4ac565` | direct oracle comparator | optional follow-up after trace runs |
| deeper layer/substage tracing | no | later `gpt-oss/full-attention-next-case` commits | parity debugging, not probe enablement | explicitly deferred |

## Probe Enablement Log

### Step 1
- Action: created `integration/probe-enablement` from `main`
- Files/commits brought over: none yet
- Result: isolated integration branch ready
- Remaining blocker: `main` still lacks the restricted prefill trace binary and trace plumbing

### Step 2
- Action: inventoried minimal probe dependencies from `gpt-oss/full-attention-next-case`
- Files/commits brought over: none yet
- Result: identified `ea3bc93` as the smallest likely viable extraction target for trace enablement
- Remaining blocker: need to cherry-pick or manually extract the probe surface and test whether it builds/runs without semantic-parity commits

### Step 3
- Action: cherry-picked the core restricted prefill trace probe surface from `gpt-oss/full-attention-next-case`
- Files/commits brought over:
  - `ea3bc93` (manually resolved to keep only probe surface)
  - retained files:
    - `crates/gpt-oss-bench/src/bin/restricted_prefill_trace.rs`
    - `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
    - `crates/gpt-oss-model-runner/src/gpu_runner.rs`
  - deliberately excluded:
    - `crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py`
- Result: restricted trace plumbing landed on `integration/probe-enablement`
- Remaining blocker: validate that the binary builds cleanly on top of `main`-based dependencies

### Step 4
- Action: added the missing bench dependency and validated the minimal probe path
- Files/commits brought over:
  - local follow-up: `crates/gpt-oss-bench/Cargo.toml` adds `gpt-oss-tokenizer.workspace = true`
- Result:
  - `cargo build --release --features cuda -p gpt-oss-bench --bin restricted_prefill_trace`: passes
  - `cargo build --release --features cuda -p gpt-oss-engine`: passes
  - live probe startup on GPU1 works through tokenizer load, worker creation, model init, and weight load
- Remaining blocker:
  - running against stock `/data/models/openai/gpt-oss-20b` fails at trusted admission with:
    - `config error: trusted GPT-OSS mode rejects attention sinks until runtime support and parity are proven: model.layers.6.self_attn.sinks`
  - so the current branch is probe-ready at the code level, but still needs a restricted/admissible model view (or equivalent narrow model-view preparation step) before future semantic cherry-picks can be safely validated from `main`

## Restricted Model View Enablement

### Step 5
- Action: added a tiny local restricted-model-view generator for probe use only
- Files/commits added:
  - `crates/gpt-oss-bench/tools/create_restricted_probe_model_view.py`
- Result:
  - generator now creates a derived model directory that:
    - symlinks the real checkpoint/tokenizer assets
    - rewrites `config.json` to full-attention-only with sliding disabled
    - adds `zzzz-sinks-override.safetensors` so all `self_attn.sinks` tensors load as zeros
- Remaining blocker: none at the model-view preparation stage

### Step 6
- Action: generated a restricted probe model view and reran the trace on GPU1
- Files/commits added:
  - generated local model view (not committed): `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
  - generated trace artifact (not committed): `/home/emmy/openai/gpt-oss-rs/.live/restricted-cuda-prefill-trace.integration.json`
- Result:
  - `restricted_prefill_trace` now runs end-to-end on GPU1 against the restricted model view
  - it progresses through:
    - trusted admission
    - f32 + f16 weight load
    - KV cache initialization
    - prefill execution
    - trace export
- Remaining blocker:
  - probe enablement itself is no longer blocked
  - next safe integration step is still separate: bring over only the smallest oracle-side comparison helper needed for measured semantic cherry-picks

## Oracle Comparison Enablement

### Step 7
- Action: restored the smallest oracle-side comparison helper from `gpt-oss/full-attention-next-case`
- Files/commits added:
  - `crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py`
- Result:
  - helper now:
    - reads the integration-branch restricted prefill trace artifact
    - uses the restricted model view for config
    - uses the original monolithic GPT-OSS checkpoint for oracle weights
    - tolerates the minimal integration trace shape without deeper attention subtraces
- Remaining blocker:
  - helper execution depends on an existing Python environment with `torch`
  - current successful run used `/home/emmy/openai/worktrees/full-attention-next-case/.venv-oracle/bin/python`

### Step 8
- Action: ran the trace-vs-oracle comparison on the integration branch artifacts
- Files/commits added:
  - generated trace artifact (not committed): `/home/emmy/openai/gpt-oss-rs/.live/restricted-cuda-prefill-trace.integration.json`
  - generated oracle diff artifact (not committed): `/home/emmy/openai/gpt-oss-rs/.live/restricted-prefill-trace-diff.integration.json`
- Result:
  - trace-vs-oracle comparison now runs successfully
  - earliest divergence boundary reported by the minimal compare surface:
    - `layer0.post_attn_residual`
    - `max_abs_diff = 26.2617188`
    - `mean_abs_diff = 0.31712404078090667`
- Remaining blocker:
  - safe semantic cherry-picks deeper than this still need additional selective trace/detail promotion if they must be measured below the coarse per-layer surfaces now available

## Probe Validation Tiers

### Objective
Reduce rerun cost by running only what is informative for each edit.

Use `scripts/probe_validation_tier.sh` for the smallest useful tiered check.

### Tier 0 — Compile-only
- Scope: tiny edits or infrastructure/CLI changes
- Command:
  - `./scripts/probe_validation_tier.sh --tier 0`
- Mandatory steps:
  - compile `gpt-oss-engine`
  - compile `restricted_prefill_trace`

### Tier 1 — Targeted local/stage check
- Scope: narrow frontier edits where full oracle rerun is not yet justified
- Command:
  - `./scripts/probe_validation_tier.sh --tier 1`
- Mandatory steps:
  - Tier 0 compile
  - trace capture on the fixed prompt/model (or reuse existing artifact)
- Artifact reuse policy:
  - safe to reuse when trace metadata matches:
    - same `restricted_model_path`
    - same `prompt`
  - if either changes, or if trace-generation code changed, rerun trace
  - if uncertain, disable reuse with `--no-reuse`
- Oracle compare: skipped unless explicitly requested

### Tier 2 — Full restricted trace + oracle compare
- Scope: merge candidates, frontier moves, newly added trace depth
- Command:
  - `./scripts/probe_validation_tier.sh --tier 2`
- Mandatory steps:
  - Tier 1 checks
  - oracle compare step

### Minimal compare-only path
- Command:
  - `./scripts/probe_validation_tier.sh --compare-only`
- Use when:
  - trace artifact is known-good and unchanged
  - only oracle-side helper logic changed

### Mandatory full rerun rules
- Always run Tier 2 for:
  - semantic cherry-pick certification
  - first time a frontier appears to move
  - when trace surfaces are modified beyond prompt/model/pure formatting
- Tier 2 can remain not required only for tooling-only edits that do not change runtime outputs

## Probe Validation Log

### Step 9
- Action: added tiered validation helper
- Files/commits added:
  - `scripts/probe_validation_tier.sh`
- Result:
  - enabled smallest viable compile/trace/compare decision workflow
  - added explicit artifact reuse checks keyed on model/prompt metadata
  - added one-click compare-only rerun path
- Remaining blocker:
  - not a replacement for full semantic confidence; deeper trace surfaces still need separate selective promotions before broad frontier-level certainty

## Fast Merge Log

### `gpt-oss/full-attention-next-case` (selected probe-only extract)
- Action: cherry-picked `f4ac565`
- Validation: Tier 0 pass
- Notes: added `crates/gpt-oss-bench/tools/restricted_oracle_prefill.py` (standalone prefill logit comparator helper)

### `gpt-oss/full-attention-next-case` (selected parity-case enrichments)
- Action: cherry-picked `ae9a5bd`, `adb0174`, `8d974de`, `7be518b`
- Validation: Tier 0 pass
- Notes: added `restricted_logit_diff` binary plus three conformance MoE parity tests/cases for biased routed middle-layer 3-layer coverage without changing runtime behavior

### `gpt-oss/full-attention-next-case` (remaining high-risk/value commits)
- Action: skipped (`e269212` and downstream semantic commits)
- Validation: N/A (not merged)
- Notes: kept for later integration; remaining commits are parity diagnostics plus runtime semantic edits (`e269212` and later) that would expand scope past probe-enablement
