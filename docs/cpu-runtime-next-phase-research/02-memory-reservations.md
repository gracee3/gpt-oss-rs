# C2: CPU Memory Inventory and Reservation

- Outcome: **planning-ready**
- Scope: accounting and reservation contracts; contiguous KV remains current
- Source budget used: current repository, llama.cpp, vLLM, and official Linux
  `/proc` memory-accounting documentation

## Objective and non-questions

C2 asks which bytes have which owner; how to avoid adding mapped, resident,
allocator, and logical quantities; what can be estimated statically; and how a
request receives, expands, refunds, and releases a conservative grant.

It does not choose paged KV, prefix reuse, swap, NUMA policy, an allocator, a
physical-memory percentage, or an automatic capacity. Those require measured
host evidence. The first contract must work with contiguous sequence-local KV.

## Current memory inventory

| Class | Current owner and source path | Accounting obligation |
| --- | --- | --- |
| SafeTensors mappings | `CpuTensorStore::{shards,tensors}` maps every shard read-only | Record virtual file bytes and resident file-backed pages separately; charge once globally |
| MXFP4 repacks | Each `RepackedMxfp4` maps one selected cache file; the cache directory can contain several layouts | Charge only mapped selected files to runtime; disk cache bytes are not RAM |
| Owned model metadata | `CpuModel` owns config, layer names, norms, biases, sinks, RoPE tables, thread pool | Global private persistent bytes; include worker stacks separately |
| Sequence KV | 24 `CpuKvCache` pairs of growing `Vec<bf16>`; 12 sliding, 12 full for the inspected 20B config | Per-request logical length and allocator capacity; never add both as independent consumption |
| Sequence/generation | prompt `String`, prompt tokens, model token history, sampler past tokens/RNG, cumulative output text/tokens/logprobs | Per-request, with explicit duplicate ownership |
| Prepared state | `PreparedSequenceDelta` retains K/V `Vec<bf16>` per row/layer; generation/output are cloned before commit | Per-iteration request-attributed staged bytes, released on commit/discard |
| Operator scratch | one worker-local `CpuExecutionContext::matrix_scratch`; transient hidden, logits, expert and attention vectors | Global worker scratch high-water plus in-flight transient estimate |
| MoE routing | routes, per-expert buckets, exact/Q8 activation panels, expert outputs and traces | Per-iteration; diagnostics separately charged |
| Command/delivery | 256 command items, 64 cumulative `RequestOutput` items/request, plus HTTP 16/32 string-item channels | Item counts do not bound bytes; C1 byte grants required |
| Response/batch stores | process-local hash maps; batch output may also persist to `/tmp` | Separate RAM and disk quotas; history can duplicate prompt/output/tool data |
| Allocator/OS | capacities, size classes, fragmentation, thread stacks, metadata, file page cache | Global headroom/observed facts, not exact per-request logical ownership |

- **C2-E-001 / CURRENT-REPO FACT:** `CpuTensorStore::open` maps every file with a
  `.safetensors` extension, not only paths in the index. The inspected snapshot
  contains three shards totaling 13,761,316,904 raw shard bytes and 13,789,246,222
  directory bytes. Mapping size does not mean all pages are resident.
- **C2-E-002 / CURRENT-REPO FACT:** the inspected repack root occupies
  20,304,746,600 disk bytes, but `CpuModel` maps only gate/up and down files for
  the selected layout. Adding the directory size to RSS would be false.
- **C2-E-003 / CURRENT-REPO FACT:** `CpuKvCache` starts with empty vectors and
  grows on append; sliding caches copy/truncate at capacity. `token_history`
  preallocates `context_cap`, while generation past tokens and output tokens
  are separate vectors.
- **C2-E-004 / CURRENT-REPO FACT:** the default CPU profile has
  `max_model_len=8192`, `max_num_seqs=1`, and
  `max_num_batched_tokens=2048`; explicit overrides can raise concurrency.
  None is a global memory budget.
- **C2-E-005 / CURRENT-REPO FACT:** current output construction clones the
  cumulative prompt and completion into each channel item. A 64-item channel
  has no stable byte ceiling and may contain many copies of the same prefixes.

The raw shard total is 13,761,316,904 bytes only as an explanatory sum; the
authoritative model identity uses config/index/file hashes from the E1
manifest, not an inferred directory total. A later inventory tool should use
checked arithmetic over the exact mapped paths reported by `CpuTensorStore`.

## Static formulas and inspected 20B example

Let `b` be KV scalar bytes, `Hkv` KV heads, `D` head dimension, `F` full
layers, `S` sliding layers, `W` sliding window, and `L` committed context:

```text
kv_logical(L) = 2 * b * Hkv * D * (F * L + S * min(L, W))
staged_kv(rows) = rows * 2 * b * Hkv * D * (F + S)
token_vectors(cap, prompt, generated) =
    sizeof(TokenId) * (cap + duplicated_prompt_and_generation_capacities)
```

For the inspected config (`b=2`, `Hkv=8`, `D=64`, `F=S=12`, `W=128`):

- one row across all layers stages 49,152 K/V payload bytes;
- after the sliding window is full, each additional committed token adds
  24,576 logical full-layer bytes;
- at context 8192, sequence KV payload is 204,472,320 bytes (195 MiB), before
  vector capacity/metadata/fragmentation;
- 2048 staged rows would hold 100,663,296 K/V payload bytes (96 MiB), before
  row/layer vector metadata and other model intermediates.

The staged maximum is a structural upper bound from the configured row budget,
not evidence that the current scheduler reaches it on this host. All formulas
use checked `u128` intermediate arithmetic; overflow is an admission failure,
never saturation or wraparound.

Additional formulas belong to their descriptors:

- C4-A route records, bucket offsets, grouped inputs/outputs, and trace bytes;
- C4-B activation/output/scratch from `M,N,K,dtype,strides,backend`;
- C4-C staged/committed attention views and scratch from row-specific bounds;
- C1 delivery from encoded event capacity, plus response-store serialized and
  owned-object estimates.

## Accounting dimensions: never sum unlike facts

```text
VirtualMapped = address ranges mapped from immutable files
ResidentFile  = resident file-backed pages (shared/private reported explicitly)
ResidentAnon  = resident anonymous/private pages
AllocatorActive = requested live allocation payload where allocator exposes it
AllocatorRetained = arenas/caches retained by allocator
LogicalCommitted = bytes promised to request/resource owners
LogicalUsed = conservative current ownership estimate
DiskCache = repack/batch artifacts on storage
```

`RSS` is a process observation, not a sum target. `PSS` apportions shared pages
and cannot be derived from logical ownership. File-backed residency can be in
both process RSS and the system page cache; it is not added twice. Allocator
capacity includes active elements and spare capacity; logical `Vec` length plus
capacity is double-counting. Peak RSS is useful validation but not a grant
ledger.

### C2-E-006 / PRIMARY-SOURCE FACT

- Source: Linux `/proc/<pid>/smaps`, `smaps_rollup`, and process status
  documentation, accessed 2026-08-11
- Observation: mappings report size, RSS, PSS, shared/private clean/dirty, and
  anonymous facts with different meanings.
- Implication: E1 snapshots must name the field and sampling instant. One
  generic `memory_bytes` value is invalid.
- Limitation: kernel accounting is sampled and not request-attributable.
- Confidence: high.

### C2-E-007 / LOCAL-SOURCE OBSERVATION

- Source: NX-SRC-002, llama.cpp `2468576...`
- Path/symbol: `src/llama-model-loader.cpp::{init_mappings,load_all_data}` and
  `llama_model::memory_breakdown`
- Observation: mapped file ranges, backend allocations, used fragments, and a
  memory breakdown are handled as distinct facts.
- Implication: a useful inventory names source/owner and can exclude unused
  mapped fragments without equating mappings with resident bytes.
- Limitation: ggml backend buffers and llama's NUMA/offload topology are not a
  local implementation model.
- Confidence: high.

### C2-E-008 / LOCAL-SOURCE OBSERVATION

- Source: NX-SRC-004, vLLM `52be12c...`
- Path/symbol: `vllm/v1/core/kv_cache_manager.py::KVCacheManager::{allocate_slots,free}`
  and `block_pool.py::BlockPool::{get_new_blocks,free_blocks}`
- Observation: capacity allocation and release are explicit scheduler-visible
  operations; lack of free capacity is a scheduling result.
- Implication: memory grant failure should precede mutation and have a stable
  rejection reason.
- Limitation: block hashing, GPU paging, eviction, and production-scale policy
  are explicitly not transferred to contiguous laptop CPU KV.
- Confidence: high for lifecycle, low as evidence for paging here.

## Candidate reservation protocol

```rust
struct MemoryEstimate {
    request_floor: u128,
    max_growth: u128,
    delivery: u128,
    response_store: u128,
    by_class: BTreeMap<MemoryClass, u128>,
}

struct MemoryGrant {
    id: GrantId,
    request_id: RequestId,
    granted: u128,
    used_estimate: u128,
    phase: GrantPhase,
}

enum GrantPhase { Granted, Active, Released }
enum GrantFailure {
    EstimateOverflow,
    PerRequestLimit,
    GlobalRequestLimit,
    GlobalMemoryLimit,
    DeliveryLimit,
    StoreLimit,
    ExpansionDenied,
    ManagerUnavailable,
}

trait MemoryReservations {
    fn grant(&mut self, request: RequestId, estimate: MemoryEstimate)
        -> Result<MemoryGrant, GrantFailure>;
    fn expand(&mut self, id: GrantId, delta: MemoryEstimate)
        -> Result<(), GrantFailure>;
    fn refund(&mut self, id: GrantId, class: MemoryClass, bytes: u128)
        -> Result<(), GrantFailure>;
    fn release(&mut self, id: GrantId) -> Result<(), GrantFailure>;
}
```

Lifecycle:

```text
Estimate -> Denied
    |
    +-> Granted -> Active -> ExpandPending -> Active
                     |             |
                     |             +-> ExpansionDenied -> controlled terminal
                     +-> Refund -> Active
                     +-> Release -> Released
```

Grant is atomic and occurs before the tokenized request enters canonical
waiting queues. The envelope has a separate hard body/tool-schema limit and a
bounded tokenization-work permit so memory is bounded before full admission.
The grant includes already-owned prompt/token bytes, maximum requested KV
growth through `prompt + max_tokens`, staged high-water attribution, delivery,
and requested storage. Global immutable model/mapping and worker scratch floors
are reserved before readiness, not recharged to each request.

`expand` succeeds before an allocation or ownership promise. Expansion is for
facts not conservatively known at initial grant, such as encoded tool/trace
payload or a larger actual store object. An ordinary generation request must
not rely on best-effort KV expansion after admission; its requested context
growth is granted up front. `refund` reduces a named logical class after
coalescing, commit/discard, or a reduced final store estimate. `release` is
idempotent and runs on every rejection-after-grant, cancellation, delivery
abandonment, owner failure, response eviction, and shutdown path.

The ledger invariant is:

```text
sum(active grants by class) <= configured logical budget by class
0 <= used_estimate <= granted for each active grant
released grants contribute zero and cannot expand/refund
```

Reservations prevent admitted work from exceeding configured logical promises;
they cannot guarantee physical allocation success because resident mappings,
allocator fragmentation, other processes, and the kernel can change. Readiness
therefore also reserves global headroom and records allocation failures as
execution failures without corrupting the ledger.

## Contiguous KV candidate policy

Keep `CpuKvCache` sequence-local. At admission, reserve the formula at the
maximum reachable requested context. Allocation may remain lazy or be grown in
bounded chunks, but allocator capacity is reported separately. Sliding layers
reserve at most `W`; full layers reserve requested context. Commit adds logical
used bytes; discard adds none. Sequence cancellation/release drops all its KV
ownership. Paging remains a future implementation behind a storage-neutral C4-C
read seam, not part of this grant interface.

## Alternatives and decisions

- Request-count-only admission: rejected because prompt length, logprobs,
  context growth, delivery, and storage differ by orders of magnitude.
- RSS threshold polling: rejected as a race-prone sampled symptom with no
  ownership/refund semantics.
- Immediate paged KV: deferred because no measured fragmentation/capacity or
  sharing requirement outweighs current contiguous simplicity.
- Up-front physical allocation of every possible byte: not selected; it can
  inflate resident/private memory and latency. Logical hard reservation plus
  explicit growth is sufficient for planning.
- **C2-D-001:** use named logical grants with separate process observations.
- **C2-D-002:** retain contiguous KV and reserve maximum promised context
  growth at admission.

## Failure modes and focused tests

Later tests must cover arithmetic overflow in every multiplication/addition;
per-request/global denial; grant atomicity; duplicate request/grant IDs;
expansion success/denial before allocation; refund underflow; double release;
release after queued, in-flight, prepared, committed, delivery-abandoned,
stored, evicted, owner-failed, and shutdown states; staged commit/discard
accounting; sliding-window plateau; full-layer growth; delivery coalescing
refund; store rejection; and invariant checks after injected failures.

A model-free allocator probe should compare logical grant transitions with
allocator active/retained and `/proc/self/smaps_rollup` snapshots. A static
SafeTensors/config probe should enumerate exact mapped paths and checked
formulas. Neither requires model execution. Any future capacity policy needs
resident/private and cold/warm observations on named hosts.

## Risks, open questions, and conclusion

- **C2-Q-001:** allocator introspection support and global headroom percentage
  cannot be chosen without a target allocator/host study.
- **C2-Q-002:** route-specific maximum encoded tool/logprob/store expansion
  needs the C1 delivery representation.
- **C2-Q-003:** worker stacks and Rayon nested parallelism need an observed
  high-water bound when C4 parallelism is considered.
- Disk exhaustion and RAM exhaustion need separate quotas and failure codes.
- Conservative per-request staged charging can overreserve when requests share
  an iteration; a later planner may reserve a global iteration floor plus
  request-attributed deltas, but must preserve the invariant.

The inventory, non-double-counting dimensions, checked formulas, contiguous-KV
policy, grant lifecycle, failures, and tests are explicit enough for later
implementation planning. Capacity numbers and allocator policy remain measured
decisions. C2 is **planning-ready**.
