# Native CPU Scheduler

Native CPU serving has one authoritative `SequenceTable`. Each
`CpuSequenceRecord` owns request metadata, model/KV state, prompt and decode
progress, generation/RNG state, accumulated output, lifecycle, and revision.
Scheduler queues and asynchronous output delivery carry IDs only; neither
holds a cloned mutable sequence group.

The generic mock engine and CUDA engines retain their existing scheduling
paths. `CpuWorker` remains a batch-one compatibility fixture, but the server
does not use it.

## Reserve, execute, commit

One CPU iteration has three phases:

1. `CpuScheduler::reserve` selects ordered rows under the configured sequence,
   token, and prompt-chunk budgets. It records sequence revisions and marks
   records in flight but does not advance prompt positions, KV, generation,
   RNG, sampled tokens, or output.
2. `CpuBatchEngine::execute` constructs a `CpuStepBatch`, prepares the complete
   model step, and stages sampling and output against cloned generation state.
   Model and sampling errors release the reservation without computed
   progress.
3. `CpuBatchEngine::commit` rechecks cancellation and revisions. It validates
   all retained model deltas before mutating any state, commits successful
   adjacent sequences together, publishes ordered output, and discards
   cancelled rows. A stale reservation commits nothing.

The asynchronous owner drains abort commands and checks for disconnected
receivers after blocking kernels return and before commit. Waiting and runnable
cancellation removes future work. In-flight cancellation tombstones the record
so its prepared delta is discarded without disturbing retained rows. Shutdown
also discards prepared work before releasing the table.

## Budgets and fairness

- `max_num_seqs` limits distinct sequences in an iteration. The `gpt-oss-cpu`
  profile defaults to one; an explicit value greater than one enables the
  experimental multi-request path.
- `max_num_batched_tokens` limits total prompt and decode rows in an iteration.
- `max_prefill_chunk` limits prompt rows from one sequence in an iteration.
  Zero means the remaining token budget is the only chunk bound.

Decode rows are chosen first, while the oldest prefill receives reserved
progress whenever the budget permits. If the total token budget is one and
both classes are runnable, turns alternate deterministically so neither class
can starve. Prefill order uses stable arrival order, and row order remains
stable through model execution and output publication.

The first implementation deliberately rejects best-of and beam search. It
uses contiguous per-sequence KV, retains row-wise causal attention, and does
not add CPU swap preemption or paged KV.

## Topology diagnostics

`CpuTopology` observes the process `Cpus_allowed_list` and
`Mems_allowed_list`, Linux package/core relationships for allowed logical
CPUs, NUMA-node CPU lists, available parallelism, and the configured Rayon
worker count. Startup logs a compact summary and the process masks.

This descriptor is informational. It does not set affinity, bind memory,
choose NUMA placement, or tune the worker count.
