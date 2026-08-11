# M4 Plan — CPU Batching and Scheduling

- Research: [`../cpu-runtime-research/04-cpu-batching-scheduling.md`](../cpu-runtime-research/04-cpu-batching-scheduling.md)
- Dependency: transactional multi-row model execution from M3/M2
- Default CPU profile: one sequence
- Experimental activation: explicit `--max-num-seqs > 1`

## Entry reconciliation

Before implementation, map scheduler-owned groups, CPU worker request state,
cancellation paths, output/stream owners, limits/config defaults, server startup
branches, and CUDA dependencies. Preserve CUDA scheduling unchanged. Refine the
plan before code if one canonical table cannot be introduced without an
adapter checkpoint.

### Reconciled repository seam

The existing synchronous `LLMEngine` scheduler/executor trait passes cloned
`SequenceGroup` values into an executor that has already combined model
execution and sampling. It cannot express a post-execution cancellation check
or table-owned per-sequence model state without retaining two authorities.
The unused `scheduler::Scheduler` has another local group type and advances
prompt progress during `schedule`; CUDA serving does not use it.

M4 therefore introduces a dedicated `CpuBatchEngine`/`AsyncCpuBatchEngine`
path. That path owns the one `SequenceTable`, shared `CpuModel`, execution
context, sampler, tokenizer, and ID-only CPU scheduler. Its iteration is split
into reserve, blocking prepare/sample, cancellation recheck, and commit so the
async owner can drain aborts and observe disconnected output receivers after
kernels return but before state is published. Server CPU startup moves to this
path. `AsyncGpuLLMEngine`, `GpuLLMEngine`, GPU block management, and the generic
mock/test `AsyncLLMEngine` remain unchanged.

The unused cloned-group scheduler implementation is replaced by the CPU ID
scheduler rather than adapted. The existing batch-one `CpuWorker` may remain a
compatibility/test facade, but it is not an authority in the server CPU path.
Canonical records own model, generation, scheduler-progress, and output state;
the async delivery map carries only request IDs and channels.

`PreparedCpuStep` needs one narrow M4 extension: filter cancelled sequence
deltas and commit the retained states from an owned ID/state map. Validation
still completes for every retained state before any is mutated, allowing one
in-flight cancellation to discard its rows while adjacent sequences commit.

## Interfaces

### Canonical state and queues

Replace `SingleRequestScheduler` and cloned scheduler-local sequence groups
with one authoritative `SequenceTable`. Records own request metadata,
`CpuSequenceModelState`, generation/RNG/output state, lifecycle, prompt/decode
progress, and revision. Waiting/runnable/in-flight queues carry IDs only.

### Reservation protocol

Scheduling creates a reservation containing record revisions, ordered prompt
or decode rows, `logits_required`, and reserved budget. Reservation does not
advance tokens, KV, RNG, output, or scheduler counters. Execution prepares a
model/generation result. Before commit, recheck cancellation and revisions;
then atomically commit model state, generation state, progress, and ordered
output. Failed, stale, or cancelled outcomes discard all staged work.

### Budgets and policy

Honor `max_num_seqs`, `max_num_batched_tokens`, and `max_prefill_chunk`.
`max_prefill_chunk = 0` means remaining token budget is the only prompt-chunk
bound. Build mixed batches of decode rows and prompt chunks. Use decode-first
selection while guaranteeing progress for the oldest prefill. With a token
budget of one, deterministically alternate mixed-class turns.

Reject best-of and beam search for the experimental CPU scheduler. Preserve
per-request streaming order, finish/stop behavior, context limits, and KV
isolation. Cancellation while waiting/runnable removes future work; in-flight
cancellation invalidates commit without disturbing adjacent successful rows.

### Topology

Add a read-only descriptor for allowed logical CPUs, physical-core sibling
relationships, observed NUMA nodes, and effective thread count. It reports
facts only: no affinity, placement, memory-binding, or automatic tuning policy.

### Server integration

Move CPU startup to the unified sequence table/scheduler/model path. Keep the
`gpt-oss-cpu` default at one sequence and require explicit greater-than-one
configuration for experimental multi-request scheduling. Leave CUDA engines
and their configuration paths unchanged.

## Commit slices

1. Introduce canonical records/table, ID-only queues, revisions, and lifecycle
   tests behind existing batch-one behavior.
2. Add reservation/execute/commit and cancellation/failure atomicity.
3. Add mixed decode/chunked-prefill budgets, fairness, and CLI/config limits.
4. Add topology descriptor and diagnostics.
5. Replace CPU server startup/worker path and add concurrent streaming and
   non-streaming fixtures without changing CUDA.
6. Close out scheduler/server/runtime docs and evidence.

## Focused gate

- one canonical identity observed by scheduler, execution, output, and cancel;
- schedule-only no-op on tokens, KV, RNG, output, and computed progress;
- budget and chunk boundaries, FCFS ties, decode-first behavior, and
  deterministic no-starvation at token budget one;
- mixed prompt/decode batches and exact logits flags;
- waiting, runnable, and in-flight cancellation beside successful rows;
- injected model/sampling failures and stale reservations are atomic;
- streaming/non-streaming order, EOS/stop/max-token/disconnect/shutdown;
- best-of/beam rejection and batch-one defaults;
- topology parsing under synthetic allowed-CPU/NUMA fixtures;
- CPU concurrent server smoke; unchanged CUDA package checks.

## Documentation updates

- scheduler design and reservation state machine;
- server/runtime CLI docs for both new limits and experimental opt-in;
- topology diagnostic contract and explicit non-policy status;
- research M4 status and this command/evidence ledger.

## Deviations and decisions

- CPU uses a dedicated engine rather than extending the generic cloned-group
  scheduler/executor traits. This is the adapter boundary required to preserve
  one canonical state owner and a real pre-commit cancellation recheck.
- The generic mock engine and CUDA engines remain separate compatibility
  systems. “Unified” here means all native CPU startup, scheduling, execution,
  output, and cancellation use the one CPU sequence table; it does not mean
  replacing CUDA block scheduling with CPU policy.
- Output channels remain in the async delivery owner and are keyed only by
  request ID. Token/text/logprob order and finish state live in the canonical
  record, so dropping or replacing a channel cannot create a second progress
  authority.

## Completion evidence

- Implementation commits: pending
- Commands/results: pending
- Concurrent server fixture/smoke: pending
- Closeout commit/workflow: pending
