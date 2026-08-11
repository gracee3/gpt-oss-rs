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

- Staged KV rows and requested logits are owned by `PreparedCpuStep`, while
  `CpuExecutionContext` enforces one active preparation and records reusable
  execution metadata. This makes drop/discard self-contained and avoids a
  borrow across the sampling boundary. M2 adds reusable matrix panels and
  scratch to the context.
- Sampling/RNG changes are staged by `CpuWorker` beside the prepared model
  step rather than stored inside `PreparedCpuStep`. This preserves the model
  layer's independence from sampling policy while retaining one publish point
  after both model execution and sampling succeed.
- The canonical multi-sequence engine table remains in M4. M3 exposes the
  shared-model and per-sequence types and proves interleaved ownership with
  direct model fixtures; the compatibility worker intentionally remains
  batch-one until scheduling is replaced.

## Completion evidence

- Implementation commits: `c9ca550` (rows/batches/revisions), `4c63de1`
  (shared immutable model), `94a01a6` (transactional KV/state), and `2247df9`
  (transactional sampling and lifecycle).
- `cargo fmt --all --check` and `cargo check -p gpt-oss-engine --locked`:
  passed.
- `cargo test -p gpt-oss-model-runner -p gpt-oss-engine --lib --locked`:
  353 model-runner tests and 249 engine tests passed.
- Shared-mapping, independent/interleaved sequence, cache-boundary,
  stale/drop/discard, injected model failure, sampling rollback, and lifecycle
  fixtures are included in those suites. State equality assertions cover KV
  content, position, token history, revision, and abort state.
- Full-model capture:
  `/data/models/openai/gpt-oss-rs-cpu-work/results/m3-harmony_122-auto.json`.
  The 122-token prompt completed one-token generation with token `200005`,
  finite recorded durations, and exit status zero.
- Closeout commit/workflow: this documentation checkpoint; remote CPU workflow
  verification follows publication.
