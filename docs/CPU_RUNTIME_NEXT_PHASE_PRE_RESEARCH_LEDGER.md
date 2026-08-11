# CPU Runtime Next-Phase Pre-Research Intake Ledger

- Status: intake consolidated; awaiting approval of bounded research charters
- Created: 2026-08-11
- Repository baseline: `main` at
  `a090bb0e81457e4302deb36d6e52a0847c14bfb0`
- Predecessor program: M1-M5 experimental CPU runtime feature set
- Research state: not started
- Pre-planning state: not started
- Implementation authorization: none

This is the durable pre-research, pre-pre-planning ledger for the CPU runtime
phase after M1-M5. It preserves the post-implementation observations,
candidate work areas, possible source-review lanes, and unanswered questions
raised during the program closeout. It now also incorporates the owner-supplied
next-phase handoff covering the service envelope, commit-versus-delivery
boundary, evidence spine, operator-contract questions, and long-horizon seam
stressors. The active scope of this ledger is the CPU runtime.

This document is not a research synthesis, roadmap, implementation plan,
priority commitment, benchmark interpretation, or authorization to change
code. The observations below come only from the completed implementation
record, existing repository text/source, retained final-smoke evidence,
user-supplied considerations, and explicitly pinned source-intake checkouts.
The separate T14 Tiger Lake/Iris Xe investigation retains its own source and
host records and is not a candidate workstream in this CPU ledger.

The completed program and its validation remain documented in the
[`final integration record`](cpu-runtime-plans/FINAL_INTEGRATION.md). The
format and evidence discipline for any later research are inherited from
[`CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md`](CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md).

## Phase boundary

This intake phase may:

- preserve current-repository facts and already-recorded experiment results;
- register candidate questions, source families, constraints, and risks;
- accept additional user-provided sources and considerations;
- identify where two candidate tracks may interact;
- record what evidence a later research phase would need;
- perform explicitly requested source-intake checkouts and inspect only enough
  high-level structure to establish relevance, pin, license, and a future
  bounded source-review lane;
- reconcile planning/status documentation without changing runtime behavior.

This intake phase must not:

- perform open-ended source tours, feasibility web research, or treat an
  unpinned source as evidence;
- run new benchmarks, oracle campaigns, hardware experiments, or audits;
- choose an implementation design, public API, persistent layout, or policy;
- create a detailed implementation sequence, commit plan, or delivery gate;
- implement or refactor any runtime, kernel, engine, server, CI, or test code;
- promote a candidate item into automatic dispatch or trusted mode.

Research begins only after the user has added the intended sources and
considerations and explicitly approves a bounded research scope.

## Staged authorization workflow

The next phase retains five explicit gates:

1. **Pre-research intake and framing:** this document and conceptual provenance
   may be updated. This is the currently authorized gate.
2. **Research planning:** after owner confirmation, write bounded charters with
   source pins, questions, artifacts, budgets, and stopping criteria.
3. **Research and documentation:** after separate confirmation, execute the
   approved charters and record evidence and conclusions.
4. **Implementation planning:** after accepted research conclusions, prepare a
   decision-complete plan with sequencing, gates, and fallback rules.
5. **Implementation:** begin only under a separately authorized implementation
   goal.

A source candidate is not an adopted design. A favorable research conclusion
is not implementation authorization, and an implemented experimental path is
not automatically eligible for trusted mode or automatic dispatch.

## Intake vocabulary

Later research must use the established evidence vocabulary: **CURRENT-REPO
FACT**, **LOCAL-SOURCE OBSERVATION**, **PRIMARY-SOURCE FACT**, **EXPERIMENT**,
**INFERENCE**, **PROVISIONAL DECISION**, and **OPEN QUESTION**.

This earlier intake ledger additionally uses five non-evidence labels:

- **INTAKE ITEM**: something to preserve for possible later investigation;
- **SOURCE CANDIDATE**: a source or source family proposed for later review but
  not inspected for this ledger;
- **CANDIDATE TRAJECTORY**: a possible grouping or ordering retained for
  discussion, not a decision or plan.
- **USER CONSIDERATION**: hardware, constraint, preference, hypothesis, or
  desired outcome supplied by the user and not independently verified here.

No row in this document is a provisional decision. Statements that go beyond
directly cited repository facts are questions or hypotheses.

## Project posture and initial service envelope

The next phase remains governed by [`PROJECT_INTENT.md`](PROJECT_INTENT.md): a
narrow, inspectable, Rust-first, GPT-OSS-specific learning and evidence
project. Mature systems are sources of questions, contracts, and failure
patterns; they are not compatibility targets or architecture authorities.

Treat the following as a safe starting envelope to research, not a public
product promise:

| Dimension | Intake position |
| --- | --- |
| Purpose | Personal learning, reproducible systems research, and possible narrowly upstreamable findings or components |
| Process/model | One Linux process serving one explicitly selected GPT-OSS checkpoint |
| Clients | Trusted local or trusted-LAN clients; malformed and oversized input still require bounds |
| Workload | Interactive use with low, explicitly bounded concurrency |
| Security | No hostile multi-tenant isolation; authentication and TLS remain outside the process |
| Compatibility | No broad OpenAI API, model, platform, or client-compatibility promise |
| Durability | No recovery of active requests across process restart |
| Availability | No HA, replication, zero-downtime, or distributed-service promise |
| Disconnect | Stops future work; resumable generation is not promised |
| Backends | Native CPU is the active experimental mainline; other backends are separately selected and qualified |
| Correctness/trust | Evidence attaches to a configuration tuple, never to the entire program without qualification |

Research should not expand this envelope merely because a reference project
supports more. Safe request/body limits, deterministic cleanup, and clear
failure reporting remain necessary inside a trusted-client envelope.

## Architectural principles to preserve

1. One canonical execution authority owns sequence truth. HTTP handlers,
   delivery workers, and backend helpers must not become competing owners.
2. Reserve, execute, and commit remain distinct. KV, RNG, token history,
   sampling, and output progress become authoritative only at commit.
3. Commit is distinct from delivery. Delivery can lag, fail, or be abandoned
   without rolling back or independently advancing sequence truth.
4. Resource bounds use explicit units and ownership. Reservations require
   grant, expansion, refund, release, and cleanup semantics.
5. Scalar/reference, forced, and automatic paths remain distinguishable, with
   machine-readable eligibility, selection, fallback, and rejection reasons.
6. Persistent layout, transient scratch, alignment, reuse, allocation, and
   thread ownership remain visible at kernel/operator boundaries.
7. Effective configuration, model/source identity, work shapes, and resolved
   behavior must be observable without making telemetry an architecture center.
8. Add a typed seam only when multiple real implementations or workflows need
   it. Do not create a general framework for hypothetical reuse.
9. Rejection and negative results are useful evidence when their boundary and
   reasoning are recorded.
10. Inspiration, repository-native synthesis, and code adaptation retain the
    distinctions in [`BORROWED_CONCEPTS.md`](BORROWED_CONCEPTS.md).

## Commit-versus-delivery starting contract

The lifecycle to test during C1 research is:

```text
envelope accepted -> tokenized -> admitted/queued -> reserved
  -> executing/staged -> committed -> pending delivery
  -> delivered or abandoned -> terminal
```

This is a starting contract, not a decided implementation:

- before commit, staged KV, RNG, sampled tokens, histories, and progress may be
  discarded;
- commit atomically publishes authoritative sequence progress;
- a pre-commit disconnect tombstones the request; a non-preemptible execution
  slice may finish, but its staged result is discarded;
- a post-commit disconnect stops future generation and records committed output
  as undelivered rather than rolling model state back;
- streaming carries deltas rather than repeatedly serializing cumulative text;
- delivery is byte-bounded; ordinary text deltas may be coalesced, while tool,
  usage, error, and terminal events preserve required ordering;
- computed, committed, queued-for-delivery, delivered, and abandoned
  tokens/bytes are distinct where the distinction is operationally useful;
- engine-owner exit or panic makes readiness false and terminates queued/waiting
  requests instead of leaving waiters hanging;
- initial stored-response semantics retain only successfully completed
  responses; `store` does not imply continued generation after disconnect.

C1 research must locate exactly where KV writes, sampling, RNG advancement,
usage accounting, response storage, finish reasons, and terminal events cross
the commit boundary. It must also define stable failure reasons for rejection,
cancellation, slow consumer, delivery abandonment, and engine failure.

## Completed-program outcome retained as context

M1-M5 established these implemented seams:

- specification-correct shared E8M0 handling and a forced AVX-512/VNNI x8
  MXFP4 GEMV over `InterleavedSplitX8V2`;
- immutable shared CPU model resources, per-sequence state, transactional
  preparation/commit, revisions, and explicit lifecycle operations;
- a typed MXFP4 matrix contract, scalar reference, explicit AVX2 4x8 backend,
  caller-owned scratch, and layer-major multi-row execution;
- one canonical CPU `SequenceTable` with reserve/execute/commit scheduling,
  bounded prompt chunks, mixed decode/prefill rows, cancellation rechecks,
  opt-in multi-request serving, and read-only topology diagnostics;
- a forced-only, feature-gated AMX-INT8 prototype with portable panel packing,
  scalar tile emulation, independent runtime gates, and a guarded native shim.

The architectural result was cleaner in several ways worth retaining during
later review:

- transactional state did not require shared mutable model ownership or an
  `Arc<Mutex<_>>` graph;
- sampling could remain outside `PreparedCpuStep` while sharing the final
  publication boundary with model/KV state;
- native CPU serving was simpler as a dedicated canonical engine than as an
  extension of the generic cloned-group scheduler;
- the AVX-512 x8 body itself did not require AVX-512VL, although the complete
  forced compatibility path still does because of its canonical-row tail;
- the available C++17 compiler accepted the required AMX intrinsics, avoiding
  a nightly Rust or handwritten-assembly requirement;
- the surveyed 20B scale bytes contained neither E8M0 `0x00` nor `0xff`, so the
  semantic correction did not change existing checkpoint scale values;
- the program integrated without `origin/main` drift or a CI-specific code
  correction.

These are context, not proof that the resulting paths are tuned, generally
portable, production-ready, or trusted-mode eligible.

## Observation register

### API and delivery

- **PPR-API-E001 / EXPERIMENT:** the retained final text-completion stream
  emitted `" with"` followed by `" with a"`, then `[DONE]`. The two events
  used one stable request ID and completed in order, but their text was
  cumulative rather than a pair of deltas. Artifact:
  `/data/models/openai/gpt-oss-rs-cpu-work/results/final-concurrent-stream.sse`.
- **PPR-API-F001 / CURRENT-REPO FACT:** the text-completion streaming route
  passes accumulated `co.text` directly to `CompletionStreamChunk::new`, whose
  type documentation calls the text incremental. GPT-OSS Chat and Responses
  use separate token/diff state; the exact affected endpoint set has not been
  audited. Path: `crates/gpt-oss-server/src/routes/completions.rs`.
- **PPR-API-E002 / EXPERIMENT:** a server launched with the local model path
  `/data/models/openai/gpt-oss-20b` rejected the request model
  `openai/gpt-oss-20b` with `model_not_found`; the absolute configured path
  succeeded.
- **PPR-API-F002 / CURRENT-REPO FACT:** Completions, Chat, Responses, tools,
  and Batch routes compare the request model string exactly with
  `AppState::model_name`. No separate served-model alias is represented.

### Scheduling, admission, and output backpressure

- **PPR-SCH-F001 / CURRENT-REPO FACT:** `AsyncCpuBatchEngine` uses a single
  canonical background owner, a 256-entry command channel, and 64-entry
  per-request output channels.
- **PPR-SCH-F002 / CURRENT-REPO FACT:** after a commit, the canonical owner
  awaits `channel.send(output)` for each request. Disconnected receivers are
  cancelled, but a connected receiver that stops draining can fill its channel
  and hold the owner at the send point. Whether HTTP-body buffering prevents
  or merely delays this condition has not been tested.
- **PPR-SCH-F003 / CURRENT-REPO FACT:** `max_num_seqs` limits active runnable
  sequences in an iteration. It does not establish a maximum waiting-request,
  queued-prompt-token, queued-byte, or total CPU-runtime memory budget.
- **PPR-SCH-Q001 / OPEN QUESTION:** what delivery and cancellation policy
  prevents one slow consumer from delaying unrelated scheduling and command
  processing without losing ordered per-request output?
- **PPR-SCH-Q002 / OPEN QUESTION:** what explicit admission limits and overload
  response should bound waiting requests, prompt tokens, and memory?

### Operations and lifecycle

- **PPR-OPS-F001 / CURRENT-REPO FACT:** `/metrics` returns a placeholder string
  even though telemetry metric names and recorders exist elsewhere in the
  engine.
- **PPR-OPS-F002 / CURRENT-REPO FACT:** `/health` is unconditional liveness and
  does not query engine state, queue state, or readiness.
- **PPR-OPS-F003 / CURRENT-REPO FACT:** `AsyncCpuBatchEngine::shutdown` signals
  cancellation but does not expose an awaitable background-task join through
  the server's `InferenceEngine` trait.
- **PPR-OPS-Q001 / OPEN QUESTION:** which CPU runtime metrics and readiness
  states are required before multi-request serving can be described as
  operationally supportable?
- **PPR-OPS-Q002 / OPEN QUESTION:** what shutdown contract should guarantee for
  admitted, in-flight, prepared, committed-but-undelivered, and stored API
  state has not been audited.

### CPU memory and KV ownership

- **PPR-MEM-F001 / CURRENT-REPO FACT:** CPU KV storage remains contiguous and
  sequence-local. The CPU scheduler does not use paged KV, swap preemption, or
  CPU prefix reuse.
- **PPR-MEM-F002 / CURRENT-REPO FACT:** the default CPU profile keeps one active
  sequence, while explicit `max_num_seqs > 1` enables the experimental
  multi-request path.
- **PPR-MEM-Q001 / OPEN QUESTION:** what exact KV, staged-row, generation,
  prompt, and delivery memory must be budgeted per request and globally?
- **PPR-MEM-Q002 / OPEN QUESTION:** should the first bounded-memory step retain
  contiguous per-sequence KV with explicit byte admission, introduce segmented
  growth, or define a paged/block-managed CPU KV contract?
- **PPR-MEM-Q003 / OPEN QUESTION:** when, if ever, should CPU prefix reuse,
  eviction, or swap be coupled to that ownership model?

### Numerical and trusted-mode closure

- **PPR-NUM-F001 / CURRENT-REPO FACT:** `docs/CPU_RUNTIME.md` retains a rare
  stricter-trace BF16 reduction-order difference before expert projection.
  The diagnostic exact-BF16 expert mode does not by itself explain a
  difference that occurs earlier.
- **PPR-NUM-F002 / CURRENT-REPO FACT:** CPU trusted mode remains rejected. New
  AVX-512, matrix, scheduling, and AMX paths remain experimental or
  explicitly selected as documented.
- **PPR-NUM-Q001 / OPEN QUESTION:** what is the smallest reproducible operator
  boundary for the pre-expert BF16 difference, and is it an implementation
  defect, a permitted reduction-order variance, or a missing reference
  boundary?
- **PPR-NUM-Q002 / OPEN QUESTION:** what correctness and operational evidence,
  separate from performance, is required for a later trusted-mode review?

### Batched compute coverage

- **PPR-CMP-F001 / CURRENT-REPO FACT:** multi-row model execution is genuinely
  layer-major, and MXFP4 expert buckets use `Mxfp4MatmulProblem`.
- **PPR-CMP-F002 / CURRENT-REPO FACT:** dense BF16 projections still invoke the
  dispatched matvec primitive per row. Ragged causal attention also remains
  row-wise. Exact-BF16 expert projection remains a row fallback.
- **PPR-CMP-F003 / CURRENT-REPO FACT:** automatic M=1 expert execution retains
  established GEMV dispatch; automatic M>1 uses the scalar matrix reference.
  Optimized AVX2 matrix execution is explicit.
- **PPR-CMP-Q001 / OPEN QUESTION:** should a later batched-compute phase begin
  with a BF16 dense matrix contract, a ragged attention contract, further
  grouped-MoE work, or a combination? Workload evidence is pending.
- **PPR-CMP-Q002 / OPEN QUESTION:** which semantic metadata belongs in a future
  ragged-attention problem without coupling scheduler policy to its kernel?

### AMX hardware closure

- **PPR-AMX-F001 / CURRENT-REPO FACT:** portable AMX packing, tile emulation,
  status injection, feature compilation, and forced-unavailable behavior are
  covered. The development host exposes no AMX CPUID support.
- **PPR-AMX-F002 / CURRENT-REPO FACT:** native AMX equality, permission and
  XSTATE behavior under real workers, signal/error paths, sustained execution,
  and performance have not been exercised.
- **PPR-AMX-Q001 / OPEN QUESTION:** what AMX-capable hosts, kernel versions,
  compilers, worker counts, and failure injection are required for a hardware
  bring-up gate?
- **PPR-AMX-Q002 / OPEN QUESTION:** no evidence yet selects persistent B-panel
  caching, different M/N tiling, AMX-BF16, or automatic AMX dispatch.

### Optional serving features, topology, and maintenance

- **PPR-OPT-F001 / CURRENT-REPO FACT:** native CPU serving rejects best-of,
  beam search, tensor/pipeline parallelism, CUDA graphs, and trusted mode. The
  CPU compatibility worker remains batch-one and is not used by the server.
- **PPR-OPT-F002 / CURRENT-REPO FACT:** CPU topology is diagnostic only; no
  affinity, NUMA memory binding, placement, or worker-count policy is applied.
- **PPR-MNT-F001 / CURRENT-REPO FACT:** CPU workflows currently use
  `actions/checkout@v4`; GitHub emitted a Node 20 deprecation annotation while
  successfully forcing the action onto Node 24.
- **PPR-MNT-F002 / CURRENT-REPO FACT:** locked checks retain the pre-existing
  unused `semantic_spec` field/accessor warnings in the generic GPT-OSS model
  runner. CPU-kernel Clippy passes with warnings denied.
- **PPR-OPT-Q001 / OPEN QUESTION:** product requirements have not established
  the relative value of best-of, beam search, speculative decoding, CPU prefix
  reuse, or topology policy.

### External Tiger Lake/Iris Xe lane boundary

Tiger Lake/Iris Xe, OpenCL, Level Zero, and SPIR-V investigation is owned by
the separate T14 lane and is intentionally outside this CPU intake ledger.
Its existing documentation and source checkouts are preserved; no C8 CPU
workstream, host setup action, dependency decision, or offload assumption is
carried into the active portfolio here. Before a future implementation changes
a shared descriptor, execution context, cache identity, dispatch surface,
Cargo feature, or build tool, the CPU and T14 lanes must exchange a short
integration note and assign one owner for that seam. CPU research does not wait
for the external lane.

## Candidate workstream intake register

The grouping and order below preserve the closeout discussion. They are not a
roadmap and may be split, combined, reordered, or rejected after user intake
and research.

| Candidate ID | Intake area | Questions retained for later research | Research state | Planning state |
| --- | --- | --- | --- | --- |
| C1 | Service, lifecycle, and API hardening | Request/engine/delivery state machines, delta streaming, served-model aliases, slow-consumer isolation, explicit admission, readiness/draining/failure, shutdown, response storage, deterministic fault tests | Not started | Not started |
| E1 | Cross-cutting evidence and observability spine | Run manifest, capability snapshot, effective configuration, cheap work descriptors, production metrics, diagnostic traces, offline evidence, dispatch/fallback reasons | Not started | Not started |
| C2 | CPU memory inventory and reservation | Exact ownership and byte accounting, mapped versus resident memory, global/per-request budgets, queued prompt bounds, reservation expansion/refund/release, contiguous versus future storage | Not started | Not started |
| C3-N / C3-T | Numerical closure and trusted-mode policy | Pre-expert BF16 localization, minimal regression, accepted semantic boundaries, configuration-specific evidence tuples, uncovered fallback rejection | Not started | Not started |
| C4 | MoE, dense BF16, and attention operator architecture | Separate typed contracts, numerical semantics, work shapes, persistent/transient ownership, scratch, threading, tracing, and future backend eligibility | Not started | Not started |
| C5 | AMX hardware bring-up | Native/emulator equality, Linux permission inheritance, tile lifecycle, worker/error/signal behavior, hardware matrix | Not started | Not started |
| C6 | Long-horizon seam stressors | Paging, prefix reuse, speculation, branching generation, preemption, limited execution plans, NUMA, and distributed boundaries; none are active feature commitments | Not started | Not started |
| C7 | Bounded repository and CI maintenance | Workflow action refresh, generic warnings, naming, compatibility-facade and duplicate-path audit, kept separate from semantic work | Not started | Not started |

## Candidate trajectory retained for discussion

The consolidated intake suggests this possible dependency shape:

```text
C1 service/lifecycle contract ----+
                                  +--> C2 memory reservation seam
E1 evidence/observability spine ---+
             |                    +--> C3-N/C3-T closure policy
             +--------------------+--> C4 operator research charters

AMX host availability -------------> C5 AMX hardware bring-up
measured need and owner demand -----> selected C6 seam-stressor charter

C7 maintenance may be handled in bounded, behavior-preserving slices
```

The reason to retain C1 as a possible first tranche is that cumulative text
streaming, slow-consumer coupling, explicit overload behavior, and operational
visibility can be investigated without selecting a kernel or dispatch policy.
That is an intake rationale only. The sequence is not locked until the added
considerations and later evidence are reviewed.

### C1 research intake: service, lifecycle, and API hardening

The later C1 charter should define, without coupling the result to a particular
HTTP library or optimized kernel:

- request, engine-owner, and delivery state machines;
- envelope validation versus token/resource admission;
- commit, disconnect, cancellation, and longest non-preemptible-slice behavior;
- byte-bounded delivery, delta coalescing, ordered non-text events, and stable
  slow-consumer failure;
- model identity, usage accounting, finish reasons, and stable error classes;
- liveness, readiness, draining, failed, stopped, and owner-panic behavior;
- shutdown ordering, join/wait semantics, queued waiter cleanup, and response
  storage;
- deterministic synthetic lifecycle/fault tests that do not repeatedly execute
  the 20B model.

Research is sufficient when these contracts can be reviewed independently of
Axum, the tokenizer implementation, and any particular matrix backend.

### E1 research intake: evidence and observability spine

E1 is cross-cutting rather than a competing runtime milestone. It should define
one machine-readable run manifest and one effective-runtime/capability snapshot
covering:

- repository commit, dirty state, branch role, build identity, toolchain,
  lockfile, features, and compile profile;
- model revision and file hashes, tokenizer/template identity, cache/layout
  versions, and source roles;
- host, kernel, CPU capabilities, backend permissions, thread count, affinity,
  and requested versus effective configuration;
- operation classes, representative work shapes, dispatch decisions, and
  eligibility/fallback/rejection reasons;
- command, workload identity, prompts or privacy-preserving prompt hashes,
  seeds, artifact hashes, limitations, and evidence status.

It must keep three instrumentation layers distinct:

1. **Production metrics:** bounded-cardinality, low-overhead counters, gauges,
   and histograms for request/queue/batch/resource health.
2. **Diagnostic traces:** opt-in named boundaries with targeted tensor
   summaries or payloads for first-divergence and lifecycle debugging.
3. **Offline benchmark/oracle evidence:** raw per-run records, environment,
   output, correctness status, and statistical summaries.

The charter must budget measurement overhead; distinguish wall-clock from
monotonic timestamps; prevent request IDs, prompt contents, secrets, or
unbounded model names from entering metric labels; define redaction of commands
and environment variables; and say what is omitted from every timer.

### C2 research intake: memory inventory and reservation

The initial goal is an ownership/accounting seam, not a paged-KV decision.
Inventory canonical and repacked mappings, page-cache and RSS behavior,
full/sliding KV, prompts and tool schemas, staged state, operator scratch,
route/group buffers, delivery queues, stored responses, allocator headroom,
and fragmentation. Avoid double-counting virtual mapped bytes, file-backed
resident pages, allocator reservations, and logical ownership.

A future reservation contract should express request, grant, conservative
estimate, measured use, expansion, refund, release, rejection reason, and
cleanup. It should serve contiguous sequence-local KV now without preventing a
later block/indexed experiment.

### C3 research intake: numerical closure and trust

- **C3-N** localizes numerical differences and defines the appropriate evidence
  boundary: scalar equality, operator tolerance, trace localization, logits,
  token parity, or distributional correctness.
- **C3-T** defines configuration-specific trusted-mode tuples and startup or
  request rejection for uncovered paths. Model hashes, cache/layout versions,
  host ISA, resolved operations, shapes/fallbacks, threads, lifecycle, and API
  behavior may all be part of the tuple.

Passing one automatic path must not silently confer trust on an untested tail,
fallback, thread mode, or matrix shape.

### C4 research intake: three operator charters

Do not research or tune C4 as one undifferentiated campaign.

**MoE orchestration:** preserve top-k semantics and stable source-row/rank
reduction; define bounded bucket offsets and route/group/compute/unroute
ownership; record bucket distributions and actual expert `M`; and investigate
parallelism across disjoint experts without nested oversubscription.

**Dense BF16:** define a sibling problem contract rather than forcing BF16 into
the MXFP4 type; include M/N/K, strides, layouts, accumulation and exact BF16
storage boundaries, immutable preparation, scratch/alignment, output ownership,
decode versus prefill classes, and explicit reference/forced/automatic modes.

**Attention:** keep row-wise attention as the semantic baseline; define row and
sequence mapping, positions, committed and same-step staged KV, causal/sliding
bounds, GQA, sinks, numerical boundaries, output/scratch, and a read interface
that does not require paged storage. Tiled or online-softmax algorithms are
candidate implementations only after this contract is established.

### C5 and C7 boundaries

AMX remains blocked on a real AMX host for native validation and tuning;
portable emulation and compilation do not establish tile lifecycle or speed.
C7 may reconcile documentation, warnings, names, and CI in small independent
slices, but must not hide semantic changes in maintenance refactors.

## Long-horizon seam-stressor register

These candidates test whether C1/C2/E1 seams are honest. Inclusion is not a
feature commitment or permission for source tours.

| Candidate | Question retained now | Gate before a bounded research charter |
| --- | --- | --- |
| Paged/block-managed KV | Can memory, sharing, or lifetime flexibility outweigh CPU indexing/locality and complexity costs? | Concrete fragmentation, capacity, or feature-enablement problem on named hardware |
| Prefix reuse | Can immutable shared KV be keyed, bounded, invalidated, and isolated correctly across model/tokenizer/template/tool configuration? | Repeated-prefix workload plus equivalence, eviction, privacy, and measurable-benefit criteria |
| Speculative decoding | Can proposal, verification, lookahead KV, RNG, acceptance, cancellation, and multi-token commit remain exact? | Concrete draft source and distribution-preserving correctness strategy plus memory/host-benefit hypothesis |
| Beam/best-of | What fork/share/copy-on-write, scoring, streaming, cancellation, and memory semantics would branching require? | Owner-established need; compatibility alone is insufficient |
| Request preemption | Would suspend/recompute/swap improve fairness beyond simpler admission and chunking? | Measured queueing/fairness harm and bounded deterministic recovery cost |
| Limited execution plan | Would typed plans improve scratch liveness, dispatch evidence, or tracing without becoming a tensor framework? | At least two real consumers or a measured maintenance/debugging problem |
| NUMA/topology policy | How do affinity, first touch, mapped weights, KV locality, SMT, and page migration behave? | Real multi-node host; diagnostics and manual experiments precede automation |
| Distributed serving | Which process, sharding, collective, and failure boundaries are educationally valuable? | Concrete model-fit, throughput, learning, or upstream-contribution objective justifying major complexity |

These items were previously held back because each multiplies ownership:
paging and prefix reuse change KV identity; speculation changes multi-token
commit and RNG; branching multiplies sequence state; preemption changes refund
and recovery; graph execution may compete with the direct model path; NUMA
automation can make behavior less predictable; and distribution introduces
collectives, sharding, and failure domains. Recording them now is seam review,
not reversal of that caution.

## Mature-project study matrix

The later research plan should select only the rows needed by an approved
charter and pin exact paths/symbols before drawing conclusions.

| Source | Candidate lesson | Boundary to preserve |
| --- | --- | --- |
| llama.cpp server and tests | Separate HTTP work, canonical inference ownership, per-sequence state, response delivery, cancellation, and low-cost lifecycle tests | Do not import router breadth, resumable streaming, model generality, or its graph/runtime architecture by default |
| vLLM scheduler and KV manager | Token-budget work representation, reservation, memory-manager ownership, cache lifecycle, and preemption pressures | GPU paging, recomputation, and production-scale policy are hypotheses rather than CPU-laptop defaults |
| TGI router/launcher | Independent validation/admission, token limits, overload, streaming, telemetry, readiness, and stable errors | TGI is now maintenance-mode; do not import its sharded deployment topology or compatibility surface |
| oneDNN primitives and diagnostics | Immutable typed problems, explicit layouts/scratch, implementation eligibility, grouped variable-size work, and selection/rejection evidence | No oneDNN dependency or general primitive/graph runtime without a concrete experiment |
| ORCA | Iteration-level scheduling and selective operation batching | Do not carry distributed assumptions into the CPU owner |
| Sarathi-Serve | Bounded prompt chunks, decode progress, and queue/execution/batch measurements | GPU utilization results do not establish CPU policy or thresholds |
| MegaBlocks | Route/group/compute/unroute and expert-bucket accounting | Exclude training, distributed, and GPU infrastructure |
| ggml within llama.cpp | Graph/memory/backend diagnostics and evidence for limited plan representations | Do not turn `gpt-oss-rs` into a tensor framework without two real consumers |
| Official GPT-OSS | Numerical semantics, precision boundaries, attention sinks, routing, and stable operator phase names | Remains semantic authority; its GPU runtime architecture is not the local runtime design |
| mistral.rs | Rust lifecycle/observability patterns and production-path CPU benchmark discipline | General model/runtime compatibility is not a goal |
| Linux NUMA primary material | First touch, affinity, placement observation, and migration vocabulary | No automatic policy before evidence on a real multi-node host |
| Transformers generation references | Greedy, sampling, beam, scoring, and generation-state comparison vocabulary | No `generate()` compatibility goal and no checkout before a generation charter |

## Cross-track questions retained for research planning

1. What is the smallest stable request/sequence identity across queueing,
   execution, delivery failure, and possible future branching?
2. Which state is staged, committed, shared, cloneable, recomputable,
   deliverable, discardable, or terminal?
3. Can a KV read/storage seam support contiguous and sliding state now without
   imposing the cost of hypothetical paging, prefix sharing, beams, or
   speculative lookahead?
4. Where should resource estimates be conservative, measured, or dynamically
   expanded, and how are overestimates refunded?
5. What does cancellation mean during a long CPU kernel, and which boundaries
   would need shortening before preemption is useful?
6. Which operator fields are stable GPT-OSS semantics and which are backend or
   execution-plan details?
7. How do request batching, Rayon/model threads, expert parallelism, and
   external thread pools avoid nested oversubscription?
8. How are fallback, tail rows, partially covered shapes, and runtime failures
   represented in trusted evidence?
9. Which runtime facts make a run reproducible without making observability a
   platform or leaking prompts, credentials, or high-cardinality identity?
10. Which findings could become small upstream tests, typed Rust interfaces,
    kernels, documentation, or benchmark methods without requiring adoption of
    the complete runtime?

Later research outcomes use: **adopt contract now**, **authorize narrow
experiment later**, **implementation candidate**, **defer pending
evidence/hardware/demand**, **reject for this project**, or **inconclusive**.
Each still requires the next authorization gate before code work.

## Pending benchmark and oracle input

Benchmark/oracle work is external to this intake request and has not been run
or interpreted here. When results are available, register their exact commit,
commands, host, model, prompts, configurations, artifacts, and limitations
before using them.

Possible later questions for that evidence include:

- whether any forced AVX-512 or matrix path merits automatic selection;
- which M crossover, prefill chunk, and concurrency shapes matter;
- whether the observed workload is attention-, dense-projection-, MoE-,
  scheduling-, or memory-bound;
- whether the recorded BF16 trace difference changes output behavior;
- which representative hosts are missing from the evidence.

No answer to those questions is assumed in this ledger.

The later benchmark charter should separate microkernel, single-sequence,
batched-engine, and end-to-end server evidence. It should distinguish prefill
from decode at context depth; warm from cold-weight behavior; raw latency from
throughput and goodput; and correctness failures from unsupported or
insufficiently trusted validation. Raw repetitions, peak RSS, thermal/frequency
context, topology/affinity, work descriptors, and exclusions belong in the
record before tuning claims are made.

The conformance fixture and general research corpus use different source roles.
Dedicated clean checkouts now exist at:

- `/home/emmy/src/cpu-runtime-research/openai-gpt-oss-oracle-7802bf263` for the
  official blocking oracle revision `7802bf263f902efd4c7d18fcceff3ba72f941e80`;
- `/home/emmy/src/cpu-runtime-research/llama.cpp-oracle-030ebb558` for the
  advisory llama.cpp revision `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`.

Newer research checkouts must not be substituted silently for these fixture
pins.

## Candidate source-review lanes

These are proposed source families only. Their current revisions, licenses,
relevant paths, and findings must be established afresh if research is later
authorized.

| Source candidate | Possible question lane | Intake status |
| --- | --- | --- |
| Current repository and M1-M5 diff | Unsafe/FFI boundaries, scheduler state machine, delivery backpressure, shutdown, memory accounting, API behavior | Candidate; no new audit performed |
| llama.cpp | Server slots, output/backpressure, model naming, CPU KV ownership, batching, CPU kernels | Candidate; prior pin exists but no refresh performed |
| vLLM | Admission, continuous scheduling, output commit/delivery, paged KV, overload behavior | Candidate; prior pin exists but no refresh performed |
| mistral.rs | Rust ownership, request lifecycle, cancellation, ragged batches, delivery | Candidate; prior pin exists but no refresh performed |
| Sarathi-Serve | Chunked prefill, decode progress, queue fairness and work bounds | Candidate; prior pin exists but no refresh performed |
| ik_llama.cpp | MXFP4 multi-row kernels, BF16/matrix organization, CPU batching | Candidate; prior pin exists but no refresh performed |
| oneDNN | BRGEMM interfaces, caller-owned scratch, AMX context and lifecycle | Candidate; prior pin exists but no refresh performed |
| Linux and Intel primary material | Dynamic XSTATE, AMX permissions, signal behavior, tile configuration/release | Candidate; no updated document review performed |
| Official gpt-oss/reference framework | BF16 operator boundaries, Harmony/API semantics, model numerical behavior | Candidate; prior pin exists but no refresh performed |
| API protocol specifications and client behavior | Incremental streaming, model identifiers, overload/error and usage contracts | Candidate; exact authoritative sources pending |
| TGI | Envelope/token admission, overload, delivery, readiness, and stable failure surfaces | Pinned sparse checkout added; maintenance-mode historical reference, not authority |
| ggml within pinned llama.cpp | Computation graph, memory planning, backend boundaries, and diagnostics | Use embedded pinned source initially; no standalone checkout until a limited-plan charter proves need |
| Transformers generation semantics | Greedy, sampling, beam, scoring, and branching-state comparison baseline | Long-horizon candidate; no checkout until an approved generation charter |

## Additional source intake

Add user-provided sources here before research begins. An entry does not imply
endorsement or that the source will be used.

| Intake ID | Source/repository/document | Supplied revision or URL | Proposed relevance | License/provenance status | Research status |
| --- | --- | --- | --- | --- | --- |
| USRC-001 | Owner handoff, `/home/emmy/Downloads/gpt-oss-rs-next-phase-pre-research-handoff.md` | Received 2026-08-11 | Service envelope, lifecycle/evidence/operator portfolio, seam stressors, sources, and staged authorization | User-supplied planning input; incorporated with status and scope refinements | Intake complete; not research evidence |
| USRC-002 | TGI, `/home/emmy/src/cpu-runtime-research/text-generation-inference` | `https://github.com/huggingface/text-generation-inference.git` at `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` | C1 admission, overload, delivery, readiness, telemetry, and test reference | Apache-2.0; clean detached sparse checkout; upstream declares maintenance mode | Intake/high-level relevance check only |
| USRC-003 | Official GPT-OSS conformance checkout, `/home/emmy/src/cpu-runtime-research/openai-gpt-oss-oracle-7802bf263` | `7802bf263f902efd4c7d18fcceff3ba72f941e80` | Blocking fixture authority | Apache-2.0; clean detached checkout | Setup only; no new oracle run |
| USRC-004 | llama.cpp conformance checkout, `/home/emmy/src/cpu-runtime-research/llama.cpp-oracle-030ebb558` | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` | Advisory fixture reference | MIT; clean detached checkout | Setup only; no build or capture |
| USRC-005 | Additional user sources | Pending | Pending | Not reviewed | Awaiting input |

For each accepted source, later research must record the canonical origin,
exact revision or document version, local path if cloned, license, relevant
symbols/sections, access date, conflicts, and limitations before citing it as
evidence.

## Additional consideration intake

Add constraints, product goals, non-goals, host availability, compatibility
requirements, deployment assumptions, and counterproposals here. Preserve the
original wording where it affects intent.

| Intake ID | Consideration | Category | Affected candidate tracks | Status |
| --- | --- | --- | --- | --- |
| UCON-001 | Preserve a narrow, inspectable, Rust-first, GPT-OSS-specific educational foothold; do not promise broad deployment, setup, model, or mature-serving compatibility. | Project posture | All | Captured from owner handoff |
| UCON-002 | Keep the service envelope safe and explicit without turning it into a compatibility or production-service promise. | Service boundary | C1, C2, E1 | Captured as starting position |
| UCON-003 | Keep CPU planning independent of the T14 Xe lane and exchange only shared-seam integration notes when needed. | Lane ownership | All shared seams | Active boundary |
| UCON-004 | Mature projects provide evidence and candidate concepts, not a mandate to wrap, clone, or generalize around them. | Source posture | All | Captured from owner handoff and project intent |
| UCON-005 | Additional user considerations | Pending | Pending | Awaiting input |

## Risks to carry into later scoping

- Fixing one endpoint's streaming text without auditing all protocol variants
  could leave inconsistent semantics.
- Moving output delivery away from the canonical owner could accidentally
  create a second progress authority unless commit state and delivery state
  remain distinct.
- Admission limits expressed only as request counts may fail to bound prompt,
  KV, staged, or response-store memory.
- A paged KV design could couple CPU and CUDA ownership unnecessarily or erase
  the simplicity gained by the dedicated CPU engine.
- Optimizing batched attention or BF16 projections before workload evidence
  may solve the wrong prefill bottleneck.
- Treating the BF16 difference as tolerance without localizing it could hide a
  semantic defect; demanding bit identity everywhere could also encode the
  wrong reference behavior.
- AMX compiler success and scalar emulation do not establish hardware
  lifecycle correctness.
- Topology or NUMA policy based on one four-core, single-node host would not be
  representative.
- Adding best-of, beam, or speculative execution before memory/admission
  ownership is explicit would multiply state and cancellation complexity.
- Treating committed and delivered tokens as one counter would obscure slow
  consumers, abandoned output, and API-visible latency.
- Full tensor tracing, per-request metric labels, or unrestricted prompt and
  environment capture could make observability expensive, high-cardinality,
  or privacy-sensitive.
- Comparing a warm repeated-buffer microkernel with end-to-end model throughput
  could produce false tuning conclusions on a DDR4 system.
- Mapped file size, allocator capacity, logical ownership, resident pages, and
  peak RSS describe different memory facts; adding them naively double-counts
  resources.
- A broad C4 campaign could conflate MoE bucket occupancy, dense BF16 work, and
  attention context cost and optimize the wrong operator family.
- A generalized KV or execution-plan interface designed for every seam stressor
  could impose complexity and slow paths before any second implementation
  exists.
- Source revision drift between research checkouts and oracle fixtures could
  make a long run irreproducible even when both revisions are individually
  documented.

## Preconditions for authorizing research

Before a research pass starts:

- the user's additional sources and considerations are entered above;
- the intended candidate tracks and explicit non-goals are confirmed;
- the repository baseline and any overnight evidence are recorded;
- each research lane has bounded questions rather than a request to survey
  everything;
- E1 states which production, diagnostic, and offline facts each charter needs
  and sets overhead/cardinality/privacy limits;
- C4 is split into separately stoppable MoE, dense BF16, and attention charters
  unless the owner explicitly approves a narrower combination;
- source priorities, allowed new checkouts, and relevant hardware access are
  known;
- the expected durable outputs and stopping criteria are agreed;
- research remains distinct from implementation planning and code changes.

## Later research and planning-readiness placeholders

A future authorized research phase may create detailed topic syntheses using
the established workstream template:

```text
Status and planning-readiness:
Objective and non-goals:
Current repository baseline:
Concrete design questions:
Explicit non-questions:
Primary sources, exact pins, licenses, paths/symbols, and source roles:
Maximum source/code exploration budget and stopping criteria:
Algorithm and dataflow findings:
Persistent and transient memory layouts:
API, ownership, and lifetime findings:
Threading, scheduling, topology, and backpressure findings:
OS/compiler portability and fallback behavior:
Correctness invariants and focused test strategy:
Required local artifacts (source map, state diagram, memory model, test matrix,
or experiment proposal):
Upstream/source comparison:
Concepts possibly retained and breadth explicitly rejected:
Alternatives considered:
Provisional design synthesis:
Risks and unknowns:
Evidence needed for adopt, narrow experiment, later, reject, or inconclusive:
Dependencies on other tracks, unavailable hardware, or owner decisions:
Open questions and required evidence:
```

Every candidate track is currently below planning readiness. There is no
decision ledger for this phase because no decision has been made.

## Intake journal

### 2026-08-11 — Post-M1-M5 trajectory captured

- Preserved the completed-program outcomes and intentional limitations.
- Recorded cumulative text-completion streaming and strict configured-model
  matching observed during the final smoke.
- Registered slow-consumer delivery coupling, implicit admission bounds,
  placeholder operations endpoints, contiguous KV, partial batched-compute
  coverage, the known BF16 difference, AMX hardware closure, optional serving
  features, and minor CI/model-runner maintenance.
- Retained seven candidate work areas and a non-binding dependency shape.
- Added empty source and consideration intake tables for the user's next
  additions.
- Performed no new research, experiment, implementation, source review, or
  detailed planning.

### 2026-08-11 — Owner handoff consolidated into CPU intake

- Added the narrow service envelope, staged authorization workflow,
  architectural principles, and commit-versus-delivery starting contract.
- Refined C1, C2, C3, C4, C5, and C7; made E1 a cross-cutting evidence spine;
  and converted optional breadth into a non-committal seam-stressor register.
- Split future C4 research into MoE, dense BF16, and attention charters and
  added instrumentation-overhead, privacy, memory-accounting, cold-cache,
  source-role, and failure-taxonomy gaps.
- Removed Tiger Lake/Iris Xe details and C8 from the active CPU portfolio while
  preserving a short external-lane coordination boundary for shared seams.
- Added a pinned maintenance-mode TGI source-intake checkout and separate clean
  official GPT-OSS and llama.cpp conformance checkouts.
- Performed no benchmark, oracle run, tuning, implementation planning, or
  runtime code change.
