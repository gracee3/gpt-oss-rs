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
- Final `main` after integration: `f5e6b6d`
- Promoted work:
  - `DEFERRED_FRONTIER.md`
- Post-integration validation:
  - `cargo test -p gpt-oss-conformance`: pass (`43/43`)
- Regressions observed: none
- Current oracle/traced frontier on `main`: still not directly measurable because `main` does not yet contain the restricted prefill/oracle probe toolchain
