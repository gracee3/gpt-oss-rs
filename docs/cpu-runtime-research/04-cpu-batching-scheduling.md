# Step 4 — CPU Batching and Scheduling Integration

- Status: research complete; ready for implementation planning
- Initial scope: several independent single-candidate requests
- Excluded initially: best-of, beam search, GPU block swap, tuned fairness

## Objective

Replace the server's CPU-only batch-one queue with iteration-level scheduling
over several independent sequences. The scheduler must feed the common
`CpuStepBatch` contract, preserve per-sequence KV isolation and streaming
order, and advance progress only after successful execution.

This is a CPU integration, not a rename or deletion of accelerator systems.
CUDA executors, block managers, and feature flags remain available for their
explicit paths.

## Current repository baseline

There are presently two scheduling worlds:

1. `crates/gpt-oss-engine/src/engine.rs` defines the synchronous scheduler used
   by the server. `crates/gpt-oss-server/src/server.rs` installs
   `SingleRequestScheduler`, which exposes only the FIFO front request.
2. `crates/gpt-oss-engine/src/scheduler/scheduler.rs` contains waiting,
   running, and swapped queues, token budgets, chunked prefill, preemption, and
   a GPU-oriented block manager. It defines a second local `SequenceGroup`
   because its fields differ from the canonical engine group.

The second scheduler clones groups into outputs and back into `running`. It
increments `num_prompt_tokens_processed` while scheduling, before execution is
known to have succeeded. Separately, the engine output path owns sampled-token
state that is not fed back into those scheduler clones. Batch-one CPU serving
works because `CpuWorker` ignores this broader state and uses request-ID
continuity plus its own last-generated token.

Simply replacing `SingleRequestScheduler` with the existing continuous
scheduler would therefore create multiple authorities for progress and output.

## Canonical ownership and interfaces

### One sequence table

The engine owns one authoritative table keyed by `SequenceId`. Each record
contains:

- request ID, arrival ordering, sampling parameters, and lifecycle status;
- prompt and generated token IDs;
- committed/computed token counts and current phase;
- `CpuSequenceModelState` and `CpuGenerationState` for CPU execution;
- cancellation/finished state and response routing metadata.

Schedulers and workers carry IDs or immutable row descriptors, not cloned
mutable sequence groups. A request may map to one sequence in the initial CPU
scope. The table shape may retain a future group relationship for best-of or
beam search, but those modes remain rejected by CPU validation.

### Schedule reservation

The scheduler returns a value equivalent to:

```text
ScheduledCpuStep
  step_id
  rows: [ScheduledCpuRow]
  per-sequence scheduled token counts

ScheduledCpuRow
  stable batch_row
  sequence_id
  token_id
  absolute_position
  phase: prefill | decode
  logits_required
```

Creating this value reserves token/sequence budget and marks work in flight. It
does not advance prompt progress, append output, modify KV, or consume sampling
RNG.

### Explicit outcome and commit

After model execution and sampling, the engine provides an outcome equivalent
to:

```text
CpuStepOutcome
  step_id
  per-sequence: success | cancelled | stopped | failed
  sampled token/logprob for each logits row
  prepared model/generation commit handles
```

Commit verifies the step ID and scheduled row counts, applies each successful
sequence transaction, advances computed-token counters, appends sampled
outputs, evaluates stop conditions, releases reservations, and produces
response events. A whole-model execution error discards all prepared sequence
transactions for the step and commits no progress.

If one request is cancelled while an otherwise successful batch is in flight,
its per-sequence prepared commit is discarded while unaffected sequences may
commit. Execution is not asynchronously interrupted inside unsafe kernels.

## Source findings

### SCHED-E001 — iteration-level scheduling needs a post-execution update

- **PRIMARY-SOURCE FACT:** ORCA schedules one model iteration at a time and
  applies batching selectively across transformer operations.
- **LOCAL-SOURCE OBSERVATION:** vLLM
  `vllm/v1/core/sched/scheduler.py` represents work as the gap between
  `num_computed_tokens` and available tokens, applies a token budget, and has a
  separate output-update path.
- **CURRENT-REPO FACT:** the existing internal scheduler advances prefill while
  constructing cloned schedule output.
- **PROVISIONAL DECISION:** preserve the useful token-budget/chunk concepts but
  replace pre-advance and clones with reserve/commit against canonical IDs.

### SCHED-E002 — mixed prefill/decode batches require bounded prefill

- **LOCAL-SOURCE OBSERVATION:** Sarathi-Serve
  `sarathi/core/scheduler/sarathi_scheduler.py` first accounts for active
  decodes and bounds prompt chunks by remaining batch budget.
- **LOCAL-SOURCE OBSERVATION:** mistral.rs
  `mistralrs-core/src/paged_attention/scheduler.rs` has explicit completion
  turns and wait counters to prevent incompatible or waiting work from being
  ignored indefinitely.
- **INFERENCE:** the CPU scheduler needs stable FCFS admission, a maximum
  active-sequence count, a per-step token budget, a prompt-chunk cap, and an
  explicit starvation invariant. Exact default sizes and latency tradeoffs are
  tuning/implementation-plan choices.

The required invariant is: while a runnable prefill exists and resources can
execute at least one token, sustained decode traffic cannot defer the oldest
prefill forever. A correctness-first implementation can reserve a minimum
prefill token or alternate a bounded prefill turn when decode consumes the
entire budget. The later implementation plan will choose the explicit constant
and configuration shape; the schedule/commit API supports either policy.

### SCHED-E003 — server slots demonstrate stable per-request batching

- **LOCAL-SOURCE OBSERVATION:** llama.cpp
  `tools/server/server-context.cpp` has a dedicated scheduling/update loop,
  explicit slot states, batch rows carrying slot ID/position/output selection,
  prompt chunks, and per-slot sampling/output.
- **LIMITATION:** llama.cpp context memory and graph execution differ from this
  Rust engine. Its value is the explicit slot lifecycle and stable row mapping,
  not code to transplant.

### SCHED-E004 — MoE batching crosses requests only after routing

- **LOCAL-SOURCE OBSERVATION:** official gpt-oss and mistral.rs flatten token
  rows, route them to experts, gather expert inputs, and restore outputs.
- **PROVISIONAL DECISION:** dense projections batch all compatible rows;
  attention remains sequence-ragged; MoE builds stable expert buckets across
  the scheduled rows; sampling and streaming return to each row's sequence.

## Scheduling lifecycle

One engine iteration is:

1. Drain new requests and cancellations into the canonical table and queues.
2. Retire already-finished records and release their CPU state.
3. Select active decode rows and bounded prompt chunks under
   `max_active_sequences`, `max_batched_tokens`, and `max_prefill_chunk`.
4. Assign stable batch-row indices. Prompt rows request logits only on the row
   that will produce a sample; decode rows request logits.
5. Resolve disjoint mutable sequence-state views and execute one transactional
   `CpuStepBatch` through the shared model/execution context.
6. Sample all logits rows into staged generation states. A sampling failure
   fails the step before commit.
7. Recheck cancellation tombstones, commit successful per-sequence model and
   generation state, and advance scheduler progress.
8. Emit response events in each request's token order, independently of batch
   row or expert execution order.
9. Remove finished/cancelled state or keep runnable IDs for the next iteration.

The scheduler may combine:

- several decode rows, one token per sequence;
- one or more bounded prompt chunks;
- prompt and decode rows in a mixed layer-major model step.

It may not batch incompatible model instances, layouts, context policies, or
execution backends. This milestone has one shared GPT-OSS CPU model, so such
incompatibility is validated at admission rather than split dynamically.

## Cancellation and failure semantics

- **Waiting cancellation:** remove the queue ID and sequence record
  immediately; no model state exists or no work is in flight.
- **Runnable cancellation:** mark cancelled, remove from future scheduling,
  and release committed CPU state when not in flight.
- **In-flight cancellation:** set a tombstone. Finish the current safe batch,
  discard that sequence's prepared commit/output, then remove it.
- **Model/kernel failure:** discard every prepared commit in the step, release
  reservations, report affected request errors, and leave committed KV and
  progress unchanged.
- **Sampling failure:** same no-progress rule; staged RNG/output is discarded.
- **Client disconnect:** route through cancellation; never drop only the stream
  sender while leaving an uncollectable sequence running.
- **Shutdown:** stop admission, cancel waiting records, finish or discard the
  one in-flight step, release all sequence state, then drop model resources.

## Topology seam

The execution configuration may expose a read-only descriptor containing:

- allowed logical CPU IDs and physical-core relationships;
- NUMA node IDs and process CPU/memory binding observations;
- worker thread count and optional future placement hints.

Discovery must respect the process's allowed CPU/memory set. It does not pin
threads, move mappings, or select first-touch policy in this feature slice.
Those actions affect performance and deployment behavior and require later
multi-socket measurements. The current development host has one NUMA node.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| Install the current internal scheduler unchanged | Rejected: duplicate group types, cloned mutable state, pre-execution progress, and GPU block/swap assumptions. |
| Add a third CPU-specific sequence model | Rejected. CPU policy may differ, but identity, lifecycle, and commit types must be canonical. |
| Put every sequence behind `Arc<Mutex<_>>` | Rejected initially. One engine execution owner can borrow disjoint table entries and avoids lock ordering in model code. |
| Decode-only batching first, no prompt progress guarantee | Rejected as an end state because sustained traffic can starve new requests. It may be a short-lived internal refactor step only. |
| Best-of/beam support in the first multi-request slice | Deferred. Candidate grouping, copy-on-write KV, and selection semantics are separate features. |
| CPU paged KV and swap preemption | Deferred. Contiguous per-sequence CPU KV is sufficient to establish correct iteration-level scheduling. |

## Focused correctness plan

- canonical sequence identity tests proving scheduler, executor, output, and
  cancellation observe the same mutable record;
- reservation tests showing schedule alone changes no computed tokens, KV,
  RNG, or output;
- several independent prompt/decode requests with stable streaming and
  non-streaming token order;
- mixed prompt-chunk/decode batches and logits flags for only eligible rows;
- token/sequence/chunk budget boundaries, FCFS ties, and a deterministic
  starvation-protection fixture;
- cancellation while waiting, runnable, and in flight, including one cancelled
  row beside successful rows;
- injected model and sampling failures with no scheduler or sequence progress;
- different context lengths and sliding/full layers without cross-sequence KV
  visibility;
- finish by EOS, stop string, maximum tokens, client disconnect, and shutdown;
- explicit rejection of best-of, beam, non-GPT-OSS, and incompatible backend
  inputs;
- server smoke coverage with concurrent streaming and non-streaming requests
  after integration.

Throughput, latency percentiles, maximum batch sizes, affinity, and NUMA
placement are informational or deferred until the runtime feature set is
complete.

## Planning handoff

The authoritative sequence table, reservation/commit boundary, descriptor
shape, batch lifecycle, cancellation semantics, mixed-operation split, and
topology boundary are ready for implementation planning. The next plan must
choose concrete configuration names/defaults and a migration sequence that
keeps batch-one server tests passing throughout the refactor.
