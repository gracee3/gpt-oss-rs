# C6: Long-Horizon Seam Pressure

- Outcome: **planning-ready**
- Scope: stress analysis of C1/C2/E1/C3/C4 seams; no feature design
- Source budget used: PagedAttention/vLLM (NX-SRC-004), the original
  speculative-decoding paper, Sarathi-Serve (NX-SRC-008), and official Linux
  NUMA policy documentation

## Objective, questions, and non-questions

C6 asks whether paging, prefix reuse, speculative and branching generation,
preemption, limited execution plans, NUMA, or distribution would break the
candidate lifecycle, memory, evidence, numerical, or operator seams. Each
stressor receives at most one primary reference and one counterpoint, then one
of: a bounded future-charter recommendation, deferral, or rejection.

C6 does not design or implement any of these features, generalize current APIs
for hypothetical compatibility, choose a scheduler/cache representation, or
claim future performance. A pressure finding changes a seam only when current
needs already justify that change.

## Current seam baseline

- **C6-E-001 / CURRENT-REPO FACT:** C1 assigns each request and sequence one
  canonical owner, separates compute commit from byte-bounded delivery, and
  requires explicit cancellation and terminal cleanup.
- **C6-E-002 / CURRENT-REPO FACT:** C2 accounts mapped/resident/allocator and
  logical ownership separately and defines grant/expand/refund/release without
  selecting paging. Current KV is contiguous and sequence-local.
- **C6-E-003 / CURRENT-REPO FACT:** E1 separates production metrics,
  diagnostics, and offline artifacts and records requested/effective runtime,
  lifecycle phases, resource pressure, fallbacks, and negative results.
- **C6-E-004 / CURRENT-REPO FACT:** C3 attaches trust to a finite effective
  configuration and all reachable fallbacks; C4-C identifies KV by sequence,
  layer, absolute position, committed revision, and staged visibility behind a
  storage-neutral borrowed-span seam.

These are candidate research contracts, not implemented features. Pressure is
evaluated against their semantics, not against assumed future code.

## Bounded external evidence cards

### C6-E-005 / PRIMARY AND LOCAL SOURCE OBSERVATION

- Question: which ownership pressure comes from paged KV, reuse, and
  preemption?
- Source: Kwon et al., “Efficient Memory Management for Large Language Model
  Serving with PagedAttention,” arXiv:2309.06180; and NX-SRC-004 vLLM
- Pin/path: paper as published; vLLM `52be12c...`,
  `vllm/v1/core/{kv_cache_manager.py,single_type_kv_cache_manager.py}` and
  `vllm/v1/core/sched/scheduler.py::{schedule,_preempt_request}`
- Observation: block tables decouple logical token positions from physical KV;
  cached blocks introduce shared/reclaimable ownership; scheduling may free
  request blocks and restart/preempt requests under pressure. The pinned source
  separately tracks cached-block hits, allocation, preemption, and stale output.
- Implication: physical allocation, logical visibility, refcounts, eviction,
  reservation ownership, and delivery generation must remain distinct.
- Counterpoint/limitation: GPU block allocators and vLLM's broad scheduler are
  not a CPU requirement. Current contiguous KV can satisfy the same absolute
  read and reservation seams with less machinery.
- Confidence: high for pressure, low for transfer.

### C6-E-006 / PRIMARY-SOURCE FACT

- Question: what state pressure comes from speculative decoding?
- Source: Leviathan, Kalman, and Matias, “Fast Inference from Transformers via
  Speculative Decoding,” ICML 2023 / arXiv:2211.17192
- Observation: a draft proposes multiple tokens and the target verifies them
  while preserving the target distribution; only an accepted prefix becomes
  output and rejected suffix work is discarded.
- Implication: staged token/KV/RNG work needs one atomic accepted-prefix commit,
  discarded-work accounting, and client usage semantics that do not report
  proposals as committed tokens.
- Counterpoint: current sampling commits one prepared target-model step and has
  no draft model or acceptance transaction. General tree state is unnecessary
  until a dedicated charter exists.
- Confidence: high for the transaction pressure.

### C6-E-007 / PRIMARY AND LOCAL SOURCE OBSERVATION

- Question: what scheduling seam is reusable without adopting a scheduler?
- Source: Agrawal et al., “Sarathi-Serve,” OSDI 2024; NX-SRC-008
- Pin/path: `96f99117...`; `sarathi/core/scheduler/
  simple_chunking_scheduler.py::_schedule`, scheduler output datatypes, and
  `sarathi/metrics/README.md`
- Observation: iterations carry a bounded token budget, explicit prompt chunk
  lengths, decode work, ignored/preempted IDs, and distinct arrival/schedule/
  execution/preemption timing.
- Implication: a limited per-iteration work list and separate phase metrics can
  remain internal; no general graph or distributed protocol is implied.
- Counterpoint: the source targets GPU/pipeline serving and multi-replica
  experiments. Current CPU ownership and commit rules are stricter and its
  replica machinery is not transferred.
- Confidence: high for descriptor pressure, low for architecture transfer.

### C6-E-008 / PRIMARY-SOURCE FACT

- Question: what does NUMA add beyond a thread count?
- Source: Linux kernel, “NUMA Memory Policy,”
  `docs.kernel.org/admin-guide/mm/numa_memory_policy.html`, accessed 2026-08-11
- Observation: task and VMA policies influence which nodes satisfy memory
  allocations; policy and allowed node sets can change effective placement.
- Implication: capability, affinity, memory placement, and node-specific
  pressure are independent evidence fields on multi-node hosts.
- Counterpoint: the research host reports one NUMA node, so it supplies no
  placement or cross-node evidence.
- Confidence: high for the seam, unavailable for local validation.

## Stressor decisions

### 1. Paged KV

**Pressure.** Pages would split one logical sequence range across allocations,
make expansion incremental, and introduce page-table metadata, free lists, and
fragmentation. Cancellation must free uncommitted pages while committed pages
remain owned until sequence release. Attention must not infer causality from
page order.

**Seam result.** C2 already separates logical KV, allocator bytes, resident
pages, and reservation lifecycle. C4-C's `KvRead` can yield several ordered
absolute spans. C1 owns terminal release. E1 needs bounded page-allocation,
eviction, fragmentation, and span-count facts, all already expressible as
resource/diagnostic fields.

**C6-D-001: deferred.** No seam change and no paging charter now. Reconsider
only after contiguous-KV pressure is measured under E1 and C2's grant protocol
is implemented and verified. Paging remains an implementation choice, not a
memory-contract premise.

### 2. Prefix reuse

**Pressure.** Reuse makes KV physical ownership shared across requests and
requires model/tokenizer/template/numerical identity, exact token-prefix
identity, tenant/privacy boundaries, immutability, refcounts, eviction, and
invalidations. A hit changes physical work but not a request's logical context,
usage, or served-model identity.

**Seam result.** C2's distinct physical/logical dimensions can charge shared
objects once physically and attribute logical references separately. C4-C's
immutable absolute reads can represent a shared committed prefix. C1 still
requires a request-local owner and cancellation. E1 must record hit length and
identity hash offline without exposing prompts as labels.

**C6-D-002: deferred with a future security/provenance charter.** Do not add
shared ownership to current C1/C2 merely for reuse. Any later charter must
start with isolation, identity, invalidation, refcount, and side-channel policy;
hit-rate potential alone is insufficient.

### 3. Speculative decoding

**Pressure.** A target verification step can accept zero through several draft
tokens. Target KV, RNG, generated text, finish state, and usage must commit the
same accepted prefix. Draft scratch/model state is temporary and rejected work
must be refunded without appearing as client tokens. Cancellation during
verification is pre-commit; disconnect after an accepted commit follows C1's
normal post-commit rule.

**Seam result.** C1's prepared/committed/delivered separation and C2's
expand/refund protocol survive if commit accepts a variable-length prefix.
C4-C's staged-visibility ordinal naturally caps verified rows. E1 needs draft,
verified, accepted, rejected, and target-work counters offline/diagnostically;
production labels remain bounded. C3 trust must cover draft-disabled target
fallback and every effective verification length.

**C6-D-003: recommend a later narrow charter**, only after ordinary target
commit/cancellation and C3 numerical evidence are planning-ready. It may add a
variable-length accepted-prefix transaction, not a generic branch graph.

### 4. Branching generation

**Pressure.** Multiple live children share an immutable prefix but own distinct
RNG, staged/committed suffixes, finish states, delivery streams, cancellations,
and reservations. Parent cancellation semantics, partial child delivery, and
response ordering become public API choices.

**Seam result.** Stable row/sequence IDs and shared-prefix-compatible memory
could represent the state, but C1 intentionally defines one result stream per
request and the current routes do not establish a branching product contract.
Adding a tree would expand public lifecycle and storage scope rather than
protect a current seam.

**C6-D-004: rejected for the next-phase charter.** Do not add branch IDs,
parent/child states, or tree reservations without an explicit product/API
requirement. Speculative private proposals do not justify public branches.

### 5. Preemption

**Pressure.** A running request may surrender compute or KV, later resume,
recompute, or fail. Already committed output must not be duplicated; in-flight
staged work and queued delivery have different lifetimes. Releasing KV changes
the reservation, while retaining it may not relieve memory pressure.

**Seam result.** A scheduling pause before execute needs no new request state;
model-state preemption would need an explicit suspended/restart generation,
stale-output rejection, and reservation transition. C1 cleanup plus C2
refund/expand can express those later, but current contiguous KV provides no
cheap reclaim-and-resume mechanism.

**C6-D-005: deferred.** Prefer admission and bounded scheduling before state
preemption. Charter preemption only after evidence shows admission cannot meet
the objective and define whether the operation retains, swaps, or recomputes
KV. Do not add `Suspended` to C1 in anticipation.

### 6. Limited execution plans

**Pressure.** Batching already coordinates prompt/decode rows and C4 exposes
operator work. A plan could name an ordered, bounded set of row groups and
resource requirements, but a general DAG would blur ownership, allow hidden
allocations, and couple kernels to service scheduling.

**Seam result.** An internal immutable iteration descriptor containing stable
row IDs, operation problem references, declared scratch, and one thread budget
fits C1/C2/C4. Commit remains a separate owner action after the whole plan
succeeds. E1 can time plan stages without creating per-operation metric labels.

**C6-D-006: recommend a later bounded internal charter** after the C4
descriptors are validated. Explicitly reject a user-visible plan API, arbitrary
DAG, allocator ownership, or commit nodes. The charter question is descriptor
reuse and validation, not scheduling policy.

### 7. NUMA

**Pressure.** Weight mappings, repacks, KV, scratch, worker affinity, and first
touch can reside on different nodes. Aggregate free/RSS values can hide one
node's pressure; changing affinity changes effective dispatch evidence.

**Seam result.** E1's capability snapshot has NUMA/affinity fields and C3 keys
affinity/NUMA policy. C2 can extend each physical allocation with an observed
node/policy dimension without changing grant ownership. No kernel or service
API needs a NUMA parameter.

**C6-D-007: deferred.** Require a multi-node host plus read-only placement and
pressure evidence before a dedicated charter. The one-node host cannot choose
placement, replication, or thread policy.

### 8. Distribution

**Pressure.** Remote execution adds partial failure, leases, retries,
idempotence, clocks, topology, network backpressure, cross-node model identity,
distributed reservation, and possibly tensor transfer. A process-local commit
and shutdown contract is not automatically a distributed transaction.

**Seam result.** E1 manifests and stable identities remain useful, but C1's
canonical owner, C2's grants, and failure/delivery rules would require a new
system boundary and consistency model. Sarathi's replica vocabulary does not
answer those product choices.

**C6-D-008: rejected for the next-phase charter.** Do not add remote-owner,
lease, rank, transport, or distributed-cache fields to present contracts.
Reopen only under a separately authorized distribution objective.

## Cross-stressor failure matrix

| Failure | Seam that must remain authoritative | Future focused assertion |
| --- | --- | --- |
| Cancel after staged work but before commit | C1 transaction + C2 refund | No KV/token/RNG publication; temporary grant refunded |
| Disconnect after commit | C1 canonical owner/delivery split | Committed state remains consistent; delivery storage stays bounded |
| Shared/reused storage evicted | C2 physical owner/refcount + C4 absolute read | Live logical reference cannot observe reclaimed or wrong-identity KV |
| Preempt/restart races old output | C1 request generation and delivery identity | Stale generation cannot commit or deliver |
| Speculative suffix rejected | C1 accepted-prefix commit | Only accepted token/KV/RNG/usage becomes visible |
| Placement or fallback changes | E1 effective snapshot + C3 tuple | Uncovered NUMA/path/shape rejects before execution |
| Remote/worker owner disappears | C1 owner-failure taxonomy | Deterministic terminal state and reservation release; no retry inference |

## Risks, conclusion, and outcome

The main risk is “future-proofing” present contracts until they become an
unimplementable generic runtime. The counter-risk is coupling semantic
positions or ownership to contiguous addresses. The current seams strike the
right boundary: stable identities, explicit transactions and grants, and
storage-neutral reads, without scheduler/paging/distribution policy.

C6 is **planning-ready as a seam review**. It recommends only two later narrow
charters—private speculative accepted-prefix commits and internal limited
execution descriptors—both behind nearer correctness gates. Paging, prefix
reuse, preemption, and NUMA are deferred to measured need; branching and
distribution are rejected from the next-phase scope. No implementation plan or
current contract expansion follows from this outcome.
