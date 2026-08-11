# Cross-Track Decision and Research Closeout

- Outcome: **planning-ready**
- Readiness boundary: foundation authorization decisions only; no
  portfolio-wide implementation plan
- Research state: complete
- Implementation-planning authorization: none
- Corpus limitation: `E1-NEG-001: unavailable`

## Meaning of this outcome

The research program is complete even though several technical candidates are
deferred. “Planning-ready for the foundation only” means E1, C1, and C2 have
sufficiently bounded candidate contracts for an owner to authorize separate
implementation-planning work later. It does not authorize that work now and
does not make C3, C4, or C5 ready by association.

There is no portfolio-wide implementation plan. C3 requires one narrow
diagnostic; C4 requires the missing owner workload corpus before candidate
ranking; C5 requires an AMX host; C6 recommends only later research charters;
and C7 is a maintenance inventory whose slices retain their own gates.

## Integrated dependency result

```text
E1 evidence identity and effective-runtime facts
  ├─> C1 request/owner/commit/delivery lifecycle
  ├─> C2 reservation and physical/logical memory accounting
  ├─> C3 configuration-specific correctness/trust cells
  ├─> C4 measured operator problem distributions
  └─> C5 host/build/kernel/permission/worker evidence

C1 commit boundaries + C2 grants
  ├─> C4 scratch/output ownership
  └─> C6 stressors without premature feature APIs

C3 trust cells
  └─> every C4/C5 main path, tail, and reachable fallback
```

E1 is not “telemetry work after implementation”; it is the prerequisite for
every promotion, negative result, or threshold claim. C1 and C2 are peers:
the request owner cannot infer a resource grant from queue admission, and the
reservation manager cannot infer request terminal state from memory use. C3
evaluates the effective paths produced by C4/C5 rather than requested backend
names.

## Cross-track decisions

### NX-D-001: preserve one canonical owner and separate delivery

The canonical sequence/request owner alone commits model state, RNG, generated
tokens, finish state, and usage. Delivery consumes immutable committed events
through a byte-bounded/coalescing surface. Slow or disconnected consumers do
not become model-state owners and must not block unrelated execution.

This resolves the present ambiguity where a bounded message count can still
hold unbounded cumulative bytes and the async owner can await a slow output
channel after commit.

### NX-D-002: reserve before runnable admission

Validation computes checked static/dynamic bounds; a request becomes runnable
only after C2 grants its initial reservation. Expansion precedes additional KV,
staging, scratch, or delivery ownership. Refund handles unused estimates;
release follows the C1 terminal owner action. Virtual size, resident pages,
allocator capacity, and logical attribution remain separate measurements.

Contiguous sequence-local KV is retained. Paging is neither required nor
forbidden by the reservation contract.

### NX-D-003: split evidence by audience and trust by effective tuple

Always-on bounded metrics, opt-in diagnostic traces, and offline manifests are
independent evidence surfaces. Requested and effective configurations are
never collapsed. Trust belongs to finite cells including model/build hashes,
host permissions, thread/affinity policy, exact operation path, preparation,
shape/tail region, oracle, assertion set, and every reachable fallback.

A path, model, or program does not receive a global trusted bit. Trusted CPU
serving remains rejected.

### NX-D-004: keep operator problems separate

MoE orchestration, dense BF16 matrix work, and attention have different
semantic identities and ownership:

- MoE owns stable route/rank/expert identity and rank-order unrouting.
- Dense BF16 owns M/N/K, views/strides, FP32 accumulation, explicit BF16 output
  boundary, preparation identity, and scratch.
- Attention owns sequence/row identity, absolute position, committed/staged
  visibility, causal/sliding/GQA/sink semantics, and storage-neutral KV reads.

They may share thread-budget, scratch, dispatch, and evidence vocabulary, but
not one general operator descriptor. No automatic threshold or preferred
backend was selected.

### NX-D-005: hardware and future features cannot broaden current claims

AMX compile/emulation evidence does not certify native execution. Paging,
prefix reuse, speculation, branching, preemption, execution plans, NUMA, and
distribution do not justify adding hypothetical states or fields to C1/C2.
C6's only positive later-research recommendations are a private
accepted-prefix transaction for speculation and a bounded internal iteration
descriptor. They are not current implementation candidates.

### NX-D-006: maintenance stays mechanically reviewable

C7 slices cannot carry lifecycle, memory, numerical, dispatch, or API semantic
changes. The two current workspace warnings and documentation status mismatch
are concrete small candidates; route/facade removal, MoE-helper deduplication,
workflow pinning, and module movement retain explicit consumer or evidence
gates.

## Planning-readiness matrix

| Track | Outcome | May an owner authorize implementation planning now? | Exact next gate |
| --- | --- | --- | --- |
| E1 | planning-ready | Yes, independently | Preserve three surfaces and measured overhead/privacy bounds |
| C1 | planning-ready | Yes, with E1 identifiers | Choose bounded delivery/storage defaults without widening route compatibility |
| C2 | planning-ready | Yes, with C1 terminal ownership | Choose initial policy values using checked formulas and host measurements |
| C3 | narrow experiment warranted | No for trust promotion | Supply an E1-complete first-mismatch capture and execute only C3-X-001 |
| C4-A | deferred | No | Supply recoverable prefill/decode expert-bucket distributions and memory/thread facts |
| C4-B | deferred | No | Supply operation/shape/tail/preparation distributions and matched repetitions |
| C4-C | deferred | No | Supply visible-context/staged-row/full-vs-sliding distributions and scratch facts |
| C5 | deferred | No | Complete the native AMX host matrix; performance remains a later separate gate |
| C6 | planning-ready seam review | No feature implementation | Obtain separate owner authorization for either recommended future research charter |
| C7 | planning-ready audit | Only per bounded slice | Respect each slice's consumer, upstream, or semantic evidence gate |

Planning readiness is not priority. This matrix prevents the absence of one
late corpus from blocking independent foundation decisions while also
preventing those decisions from laundering unsupported operator or hardware
claims.

## Required evidence still unavailable

### NX-Q-001: owner benchmark/oracle corpus

The results directory did not contain the promised new corpus. A usable intake
must recover repository/build, command/environment allowlist, host/capability,
model and repack hashes, requested/effective configuration, workload, timer
inclusions, repetitions, raw outputs, and artifact hashes. Partial results are
advisory and cannot select dispatch, thresholds, or trust.

### NX-Q-002: first BF16 arithmetic boundary

Repository history localizes the remaining difference to layer-0 K/V dense
projection outputs feeding attention, but retained raw data cannot choose RMS
normalization, a reduction lane/order, or BF16 conversion. C3-X-001 is the
complete bounded follow-up; no new model-scale run is required or authorized.

### NX-Q-003: native AMX lifecycle

This host lacks AMX. Required future evidence includes real CPUID/XSTATE/
permission, per-worker first use, signal/alternate-stack and fork/exec cases,
tile coexistence/release, native/emulator raw tile equality, and covered scalar
fallbacks. Portable CI is not a substitute.

### NX-Q-004: representative memory and delivery pressure

Static formulas bound logical KV/staging and mapped checkpoints, but policy
defaults need observed allocator/RSS/fragmentation and slow-client byte
pressure under an E1 manifest. Such measurements can be model-free or use a
small fixture; this research did not invent policy numbers.

## Rejected inferences

- Seven-prompt greedy parity does not confer tensor, fallback, host, or service
  trust.
- A bounded channel item count is not a byte bound.
- Mapped bytes, resident pages, allocator capacity, and request ownership are
  not interchangeable memory totals.
- A PyTorch model function does not specify the installed operator's reduction
  order.
- Grouped work, matrix batching, tiled attention, and AMX are not faster by
  definition.
- Portable compilation does not validate an ISA.
- Future paging/distribution possibilities do not require present public APIs.
- A name such as legacy, compatibility, placeholder, or experimental does not
  prove dead code.

## Verification and scope attestation

The research used source inspection, static model/configuration arithmetic,
read-only host probes, and existing locked checks. It launched no new 20B run,
benchmark campaign, oracle capture, or tuning. Raw existing artifacts remain
outside Git under the registered paths and hashes. Upstream checkouts stayed
clean and distinct by source role. Tiger Lake/Xe work was not inspected beyond
recording the shared-seam boundary.

The final repository diff must contain Markdown documentation only. Required
closeout checks are `git diff --check`, relative-link verification,
`cargo fmt --all --check`, `cargo check --workspace --locked`, and focused
engine/server/model-runner tests. The two pre-existing model-runner warnings
are retained as C7 evidence rather than silently fixed.

## Closeout

The research corpus is complete and independently reviewable. Foundation
contracts are ready for a later authorization decision; numerical, operator,
and AMX work remain behind explicit evidence gates. No implementation plan,
source change, dispatch promotion, trusted-mode change, or future-feature
charter was created.
