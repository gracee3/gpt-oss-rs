# M3 Plan — Immutable Model and Per-Sequence State

- Research: [`../cpu-runtime-research/03-model-sequence-state.md`](../cpu-runtime-research/03-model-sequence-state.md)
- Compatibility: batch-one `CpuModelRunner` remains supported
- Execution concurrency: one execution owner initially

## Entry reconciliation

Before implementation, inspect all `CpuModelRunner` fields and public methods,
tensor/repack ownership, KV cache mutation points, traces, engine `CpuWorker`
sampling state, and fixture construction. Record any public API that cannot be
migrated atomically and introduce adapters before moving ownership.

## Interfaces

### Shared model

`Arc<CpuModel>` owns configuration, tensor store/mappings, repacked weights,
layer descriptors, norms, RoPE, kernel plan, attention policy, and shared
thread resources. It never owns or infers a current sequence.

### Persistent sequence state

`CpuSequenceModelState` owns per-layer KV, absolute next position, context cap,
token history, and a monotonic `revision`. Construction, reset, and inspection
are explicit model APIs. No RNG, stream, or response data belongs here.

### Common batch

`CpuStepBatch` contains ordered rows. Each row carries sequence ID, input token,
absolute position, and `logits_required`. Validation rejects duplicate mutable
sequence aliases, position mismatches, unsupported sizes, and stale revisions
before staging.

### Transactional execution

`CpuExecutionContext` owns reusable activation, staged-KV, logits, and later
matrix scratch. `CpuModel::prepare_step` executes against committed state plus
per-sequence staged overlays and returns `PreparedCpuStep`. The prepared value
contains expected revisions, staged KV rows/positions/token history, requested
logits, and staged sampled-generation changes where the compatibility facade
needs them.

`PreparedCpuStep::commit` validates every expected revision before changing
any sequence. Commit is all-or-nothing, increments revisions, and applies
sliding eviction only then. Rejecting, explicitly discarding, or dropping a
prepared value leaves committed model state, RNG, sampling history, and output
unchanged.

### Lifecycle and compatibility

Expose create, reset, abort/discard, remove, and shutdown operations at the
appropriate model/engine owners. Retain `CpuModelRunner` as a batch-one facade
holding `Arc<CpuModel>`, one sequence state, and one execution context. Preserve
existing prefill, token, logits, trace, reset, and diagnostics behavior.

## Commit slices

1. Introduce shared types, IDs, row/batch validation, revisions, and ownership
   tests without changing execution.
2. Move immutable runner fields to `CpuModel`/`Arc` and keep the facade green.
3. Move KV/history/position to `CpuSequenceModelState` and implement staged KV
   overlay plus prepare/discard/commit.
4. Stage sampling/RNG/output in the CPU worker boundary, add lifecycle APIs,
   failure injection, and stale-commit rejection.
5. Close out ownership docs, compatibility tests, and evidence.

## Focused gate

- multiple states share model mappings/repack ownership without byte copies;
- independent and interleaved sequences match isolated batch-one runs;
- full and sliding caches at and across capacity retain correct history;
- injected failure before staging, at early/late layers, logits, and sampling;
- discard-on-drop and explicit discard are byte-for-byte no-ops;
- stale and duplicate commits are rejected atomically;
- reset/abort/remove/shutdown affect only targeted state;
- existing trace fixtures and `CpuModelRunner` callers remain green;
- formatting, locked model-runner/engine tests, and affected workspace checks.

## Documentation updates

- New or updated ownership section in CPU runtime/model-runner docs.
- `docs/cpu-runtime-research/03-model-sequence-state.md`: implementation
  status and deviations.
- Public type documentation for batch, revisions, preparation, and lifecycle.
- this file: commands, fixture coverage, commits, and results.

## Deviations and decisions

- None recorded yet.

## Completion evidence

- Implementation commits: pending
- Commands/results: pending
- Fixture and rollback evidence: pending
- Closeout commit/workflow: pending

