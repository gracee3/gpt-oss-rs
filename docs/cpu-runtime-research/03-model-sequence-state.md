# Step 3 — Immutable Model and Per-Sequence CPU State

- Status: implemented and verified in M3
- Compatibility goal: preserve existing batch-one results and CLI behavior
- Concurrency goal: one execution owner, many independent sequence states

## Objective

Make one loaded GPT-OSS CPU model safely reusable by multiple sequences without
duplicating mappings, repack caches, layer metadata, kernel dispatch, or thread
resources. Mutable model history, generation history, and temporary execution
memory must have explicit and different owners.

This step does not introduce concurrent model execution. A single CPU engine
thread may own all sequence mutation initially. The split creates the seam for
batching and later concurrency without using shared mutable runner state.

## Current ownership graph

`CpuModelRunner` currently owns all of the following:

```text
immutable/model-scale
  CpuGptOssConfig
  CpuTensorStore and mapped SafeTensors shards
  CpuLayer descriptors and RepackedMxfp4 mappings
  final norm, YarnRope, Kernels dispatch
  Rayon ThreadPool

mutable/single-sequence
  one CpuKvCache per layer
  position
  token_history
```

`CpuWorker` then owns a second mutable layer: one optional request-ID-keyed
state containing last sampled token, sampling history, and RNG. A new request
causes `prefill` to reset the model runner. The worker rejects more than one
scheduled request or sequence.

`CpuTensorStore` and `RepackedMxfp4` already expose immutable mmap-backed data;
they do not need redesign. `Kernels` is copyable, RoPE is immutable after
construction, and the Rayon pool can be shared by non-overlapping executions.

## Current failure hazard

`forward_token_with_trace` advances layers sequentially. Each layer appends its
K/V entry before attention and before later layers execute; the model-wide
position advances only after the token finishes. An error after layer N can
therefore leave caches 0 through N advanced while position and later caches
remain unchanged.

The existing sliding cache also evicts by shifting its `Vec` before appending.
A length-only rollback cannot restore the evicted entry. Multi-row execution
must not amplify this into partially advanced sequences.

## Proposed ownership model

### `Arc<CpuModel>` — immutable and shared

Own or reference:

- validated model configuration and tensor identity;
- `CpuTensorStore`, layer descriptors, final norm, and all repacked MXFP4
  mappings;
- RoPE tables/configuration and kernel dispatch plan;
- immutable context limits and attention policy metadata;
- the shared Rayon pool and read-only topology descriptor.

Loading and repacking happen once before the model is placed in an `Arc`. Model
methods never infer or retain a current sequence.

### `CpuSequenceModelState` — one per sequence

Own:

- one KV cache per transformer layer;
- absolute next position and context cap;
- model token history needed for traces or model semantics;
- sequence-local full/sliding attention metadata.

Construction is explicit through the model so every layer receives the correct
capacity. It contains no sampling RNG, response text, stream channel, or model
weights.

### `CpuGenerationState` — one per sequence/request candidate

Own:

- sampling RNG and sampling history;
- sampled token IDs, cumulative log probability, and stop state;
- output/detokenization state owned by the engine layer.

This separation lets model conformance tests run without a sampler and lets a
sampling failure roll back independently from model execution.

### `CpuExecutionContext` — worker-local and reusable

Own:

- bounded activation matrices and Q8/residual-Q8 panels;
- matrix scratch queried from the selected backend;
- staged KV transaction storage and output/logit buffers;
- no state that identifies the sequence after a step completes.

One execution context is used by one model step at a time. It may grow to a
configured maximum and reuse allocations; reported scratch bounds include all
per-batch staging.

### Engine sequence table — authoritative mutable owner

The engine owns a map from `SequenceId` to a record containing request
metadata, `CpuSequenceModelState`, and `CpuGenerationState`. The initial design
mutates this table on one engine thread, so scheduler descriptors can carry IDs
and the execution path can borrow disjoint entries without `Arc<Mutex<_>>`.

## Source findings

### STATE-E001 — mature runtimes separate model and context

- **LOCAL-SOURCE OBSERVATION:** llama.cpp `src/llama-model.h` holds immutable
  model weights and metadata. `src/llama-context.h` references the model and
  owns mutable memory/KV, outputs, schedulers, and execution resources.
- **LOCAL-SOURCE OBSERVATION:** mistral.rs
  `mistralrs-core/src/sequence.rs::Sequence` owns tokens, output, sampling,
  cache/recurrent state, and lifecycle status while the pipeline/model is
  shared separately.
- **INFERENCE:** model sharing with explicit contexts is the established
  ownership boundary. This repository should go further by separating
  sequence-persistent state from worker-local scratch, because it intends to
  batch several contexts in one CPU execution.

### STATE-E002 — mmap-backed weights are already suitable for sharing

- **CURRENT-REPO FACT:** `CpuTensorStore` owns immutable shard mmaps and tensor
  views; `CpuRepackCache` opens immutable repacked files; layer descriptors
  store names, small vectors, and repacked mappings.
- **PROVISIONAL DECISION:** move these values into `CpuModel` without cloning
  tensor bytes. Use `Arc<CpuModel>`, not one `CpuModelRunner` per sequence.

### STATE-E003 — batch mutation must have a success boundary

- **LOCAL-SOURCE OBSERVATION:** llama.cpp memory implementations expose
  `init_batch` contexts, and `llama_context::decode` prepares batch allocation
  and output capacity before applying successful work.
- **LOCAL-SOURCE OBSERVATION:** current vLLM scheduler separates scheduling
  from `update_from_output` and explicitly adjusts computed-token state on
  rejected or failed work.
- **INFERENCE:** scheduling a row is not proof it executed. Sequence state and
  scheduler progress need an explicit commit after the entire CPU step and
  sampling result are valid.

## Transactional KV design

The first implementation should use an overlay transaction rather than append
and undo:

1. Validate every participating sequence and allocate bounded staging for all
   affected layers and rows before mutation.
2. For each layer, write new K/V values into a transaction-owned segment tagged
   by `SequenceId` and absolute position.
3. Attention reads a logical view consisting of the committed cache plus
   earlier staged rows for the same sequence. Rows from other sequences are
   never visible.
4. Keep staged entries for every affected layer until the whole model step and
   downstream sampling validation succeed.
5. Commit layers and positions together. Sliding-window eviction occurs only
   during commit, so no evicted entry must be reconstructed on failure.
6. On any model, logits, or sampling error, discard staged KV and staged
   generation changes; committed sequence states and scheduler counters remain
   unchanged.

For chunked prefill, each successfully executed chunk is a transaction. A
later cancellation or failed chunk does not erase earlier committed chunks.
The transaction bound is therefore the scheduled step, not an entire prompt.

The initial committed `CpuKvCache` may retain its contiguous vectors and shift
on sliding-window commit. A ring buffer would remove the O(window × width)
shift and is a likely later improvement, but it is not required to establish
correct ownership or rollback. GPU-oriented paged swap/preemption is also not
required for the first CPU state split.

## Proposed model methods

Exact Rust names remain an implementation-plan detail, but the interfaces must
have these properties:

```text
CpuModel::new_sequence_state(context_cap) -> CpuSequenceModelState

CpuModel::execute_step(
    execution: &mut CpuExecutionContext,
    batch: &CpuStepBatch,
    sequences: disjoint mutable sequence-state views,
) -> PreparedCpuStep

PreparedCpuStep::logits() -> per-request row results
PreparedCpuStep::commit(...)
PreparedCpuStep::discard()
```

`execute_step` may not create, look up, or remove engine sequences. The engine
resolves IDs and provides the exact state views. A prepared result either owns
the staged data or holds a scoped transaction; dropping it without commit must
be equivalent to discard.

Explicit engine lifecycle operations are required:

- create sequence with validated context capacity and sampling seed;
- reset a sequence for an intentional new prompt;
- abort/cancel without affecting other sequences;
- remove finished sequence and free KV/generation memory;
- shut down the model only after all execution contexts and sequences are gone.

## Invariants

- Immutable model bytes and repack mappings are loaded once and shared.
- A `SequenceId` has at most one mutable model state and one generation state.
- Position equals the number of committed model input tokens, subject to the
  explicit absolute-position contract; every layer agrees on that position.
- KV cache entries contain only their owning sequence and layer.
- Failed or discarded steps do not change cache length/content, position,
  token history, RNG, sampled tokens, or scheduler progress.
- Scratch and staged rows cannot outlive their execution context or be observed
  by another simultaneous call.
- Reset/remove is explicit; changing request ID never implicitly resets model
  state.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| One `CpuModelRunner` per sequence | Rejected because it duplicates model descriptors/mappings and prevents natural dense batching. |
| Shared runner protected by a mutex | Rejected because the runner still contains one sequence's caches and creates a global mutable-state bottleneck. |
| Append each layer and roll back lengths | Rejected because sliding eviction destroys data and because partial-layer state is externally inconsistent. |
| Clone complete KV state before each step | Correct but unbounded and prohibitively large; use bounded staged deltas. |
| Introduce CPU paging during the split | Deferred. Paging changes allocation/preemption policy and is not needed to prove safe sharing. |

## Focused correctness plan

- load one model and create several sequence states while asserting mappings
  and repacked storage are shared, not copied;
- run two sequences with different prompts interleaved and compare each with an
  isolated batch-one reference;
- inject failures before staging, after an early layer, after a late layer,
  during logits, and during sampling; assert byte-for-byte committed-state and
  RNG stability;
- exercise full and sliding caches at capacity, including a failure that would
  otherwise evict the oldest entry;
- reset, abort, remove, recreate, and context-limit failures without affecting
  neighboring sequences;
- drop a prepared result without commit and verify discard semantics;
- run successful multi-row prompt chunks and decode rows, verifying every
  layer position and token history advances together;
- keep existing batch-one prefill/decode and trace fixtures working through a
  compatibility adapter.

No concurrent execution stress test, paged-cache benchmark, or NUMA policy is
required until a later feature slice introduces those behaviors.

## Planning handoff

The ownership classes, authoritative table, transaction boundary, staged-KV
strategy, explicit lifecycle, and initial contiguous cache choice are settled
for implementation planning. The later plan must order the refactor so the
batch-one adapter stays usable after each commit.

## Implementation status

M3 implemented the ownership split as researched:

- `CpuModel` contains the immutable model-scale resources and is shared through
  `Arc`; multiple compatibility runners share the same mapping ownership.
- `CpuSequenceModelState` contains per-layer KV, position, context cap, token
  history, abort state, and a monotonic revision.
- `CpuStepRow`/`CpuStepBatch` carry sequence identity, token, absolute position,
  and `logits_required`, including consecutive same-sequence prompt rows.
- `CpuModel::prepare_step` evaluates against committed state plus a bounded
  staged-KV overlay. `PreparedCpuStep::commit` validates every revision and
  delta before any mutation and applies sliding eviction only during commit.
- `CpuWorker` stages its RNG and generation history until sampling succeeds,
  then commits model and generation progress together at the worker boundary.
  Reset, abort, remove, and shutdown are explicit operations.
- `CpuModelRunner` remains the batch-one compatibility facade used by existing
  parity and trace fixtures.

The implementation keeps staged KV and requested logits in the self-contained
`PreparedCpuStep` rather than borrowing storage from `CpuExecutionContext`.
This preserves simple drop/discard semantics without cloning committed KV;
M2 extends the execution context with caller-owned matrix scratch. The
authoritative multi-sequence engine `SequenceTable` remains an M4 deliverable,
as planned.

Focused synthetic tests cover shared mappings, isolated versus interleaved
sequences, full/sliding boundaries, failures before staging and after early or
late layers/logits, dropped and stale prepared work, lifecycle operations, and
sampling rollback. A one-token 20B `harmony_122` compatibility smoke retained
the expected first token `200005` on automatic dispatch.
