# CPU Runtime Research and Pre-Planning

- Status: first research pass complete; all five steps planning-ready
- Started: 2026-08-11
- Scope: the next CPU-only architecture sprint after the promoted AVX2 x8
  MXFP4 decode path

This document is the durable entry point for research that precedes detailed
implementation plans. It records the bounded sprint scope, evidence standards,
source revisions, provisional decisions, open questions, and planning-readiness
criteria. It should grow as research proceeds, while keeping observations
separate from interpretations and decisions.

The goal of this phase is a concrete, source-grounded design for the broader
CPU feature set. It is not a benchmark campaign, an exhaustive oracle exercise,
or permission to copy or wrap an upstream implementation.

## Sprint scope: steps 1-5

These five steps are the feature-complete boundary for the next sprint. The
numbers are stable so future research notes and plans can refer to them.

### 1. Eight-output AVX-512/VNNI MXFP4 GEMV

- Research and prototype a genuine AVX-512/VNNI x8 decode kernel.
- Begin with the existing `InterleavedSplitX8V2` packed layout and require
  evidence before introducing another cache layout.
- Preserve exact E2M1 value reconstruction, per-32-value E8M0 scaling,
  residual-Q8 behavior, row tails, and scalar equivalence.
- Keep the kernel forced/experimental. Automatic dispatch remains on the
  promoted AVX2 x8 path until the later benchmark-and-tuning phase.
- Detailed synthesis:
  [`cpu-runtime-research/01-avx512-vnni-x8.md`](cpu-runtime-research/01-avx512-vnni-x8.md).

### 2. GEMM/prefill contract and packed SIMD implementation

- Introduce separate model-step and matrix-problem descriptions. Sequence,
  position, and attention metadata stay above the kernel boundary; the matrix
  problem carries dimensions, numerical views, backend preference, layout,
  output stride, and caller-owned scratch.
- Specify and implement a scalar/reference MXFP4 multi-row/multi-token API
  before optimizing it.
- Research cache layouts and loop organizations for multi-token prefill,
  including packing cost, scale reuse, accumulator precision, tails, and
  bounded scratch ownership.
- Add an explicit SIMD packed-GEMM path without an unmeasured automatic
  crossover threshold.
- Detailed synthesis:
  [`cpu-runtime-research/02-gemm-prefill.md`](cpu-runtime-research/02-gemm-prefill.md).

### 3. Immutable model and per-sequence state separation

- Separate immutable mapped weights, repack caches, model metadata, and shared
  execution resources from mutable token, attention, KV-cache, sampler, and
  generation state.
- Define ownership and lifetime boundaries that allow several sequences to
  share one model safely without cloning model-scale data.
- Make cancellation, reset, error cleanup, and sequence destruction explicit.
- Preserve the current batch-one behavior while creating the API seam needed
  by later scheduling work.
- Detailed synthesis:
  [`cpu-runtime-research/03-model-sequence-state.md`](cpu-runtime-research/03-model-sequence-state.md).

### 4. CPU batching and scheduling integration

- Generalize the CPU worker beyond one active sequence behind an experimental
  configuration.
- Connect scheduler batches to the workload descriptor and GEMM/prefill
  contract from step 2.
- Research continuous batching, prefill/decode separation, MoE token/expert
  grouping, fair cancellation, output ordering, token budgets, and KV-cache
  isolation.
- Represent topology/NUMA and memory-placement information in interfaces now,
  while deferring machine-specific policy tuning.
- Detailed synthesis:
  [`cpu-runtime-research/04-cpu-batching-scheduling.md`](cpu-runtime-research/04-cpu-batching-scheduling.md).

### 5. AMX prototype and integration seam

- Use AMX-INT8 as the first mapping and retain AMX-BF16 as an explicit later
  alternative.
- Determine how E2M1 values and per-32 E8M0 scale boundaries can be represented
  without changing model semantics across INT32/tile accumulation.
- Isolate tile configuration, Linux permission handling, thread migration,
  scratch requirements, and fallback behavior.
- Build the prototype behind an explicit experimental interface. In the
  absence of an AMX host, require compilation, packing/unit tests, and scalar
  cross-checks for all portable portions; defer hardware execution and tuning.
- Detailed synthesis:
  [`cpu-runtime-research/05-amx-prototype.md`](cpu-runtime-research/05-amx-prototype.md).

## Scope boundary and gate policy

“Feature complete” in this document means the five steps above, not every
possible future CPU optimization. The later certification/tuning phase begins
only after these interfaces and experimental implementations exist end to end.

During feature development, tests and small measurements are allowed when they
answer a design question or protect mostly-correct execution. We continue to
require focused scalar/reference equivalence, shape/tail/extrema/invalid-input
coverage, memory and cache integrity, targeted full-model parity for changed
paths, and relevant API smoke tests.

The following are deliberately deferred:

- long-length and exhaustive official-oracle certification;
- the complete 28-run conformance matrix and full API permutation matrix;
- repeated Criterion campaigns, percentage performance gates, crossover
  tuning, and cross-host dispatch policy;
- new advisory llama.cpp captures solely for certification;
- AMX hardware tuning until a suitable host is available;
- promotion of new kernels into automatic dispatch or trusted mode.

Every new optimized path remains forced/experimental until the deferred phase.
Semantic mismatches, memory-safety problems, corrupt caches, and nonfunctional
APIs remain blocking defects. The repository-wide policy is recorded in
[`NEXT_MILESTONES.md`](NEXT_MILESTONES.md#development-gate-policy).

## Dependency map

The research tracks overlap, but the implementation dependencies are not a
single serial chain:

```text
Step 1: AVX-512 x8 GEMV --------------------------> later dispatch tuning

Step 2: workload contract -> scalar GEMM -> SIMD GEMM --+
                                                        +-> Step 4 scheduling
Step 3: model/sequence state split ---------------------+

Step 2: GEMM dataflow/layout ---------------------------> Step 5 AMX integration
```

Step 1 can be designed independently. Steps 2 and 3 should be researched
together because their contracts determine step 4. Low-level AMX feasibility
can be researched early, but its integrated design should follow the GEMM
contract rather than create a second execution model.

## Research principles

1. Inspect this repository first. State the current contract and the exact gap
   before looking elsewhere.
2. Study upstream source to understand algorithms, layouts, ownership, and
   failure modes. Do not import, wrap, transcribe, or mechanically translate
   implementations.
3. Pin every local source observation to an exact commit and path/symbol. A
   moving checkout name is not evidence.
4. Prefer primary sources for web research: upstream source and design notes,
   Intel architecture and optimization manuals, Linux kernel documentation,
   Rust/toolchain documentation, and original papers.
5. Track licenses and provenance even when only ideas are being studied.
   Distinguish general algorithms and hardware facts from implementation
   choices specific to a project.
6. Compare at least two viable designs when a choice affects public/internal
   APIs, persistent layouts, ownership, or fallback behavior.
7. Treat performance claims without measurements on relevant hardware as
   hypotheses. This phase may form performance hypotheses but does not certify
   them.
8. Keep formal implementation plans separate. A workstream becomes ready for
   planning only after its readiness checklist is satisfied.

## Evidence vocabulary

Each substantial note must use one of these labels:

- **CURRENT-REPO FACT**: directly observed in this repository at a named commit
  or dirty-tree state.
- **LOCAL-SOURCE OBSERVATION**: directly observed in a pinned local upstream
  checkout.
- **PRIMARY-SOURCE FACT**: supported by a cited authoritative web document,
  paper, specification, or upstream statement.
- **EXPERIMENT**: an observation with the command/configuration, hardware,
  inputs, and result recorded.
- **INFERENCE**: our interpretation of one or more facts; it must cite those
  facts and state uncertainty.
- **PROVISIONAL DECISION**: the current design direction, subject to research
  or implementation feedback.
- **OPEN QUESTION**: unresolved and paired with the evidence needed to resolve
  it.

Words such as “faster,” “better,” “required,” or “unsupported” need evidence or
must be written as hypotheses.

## Evidence-card template

Add evidence near the relevant workstream synthesis and give it a stable ID,
for example `GEMM-E007`.

```text
ID and label:
Question:
Source kind:
Repository or URL:
Exact revision:
Path and symbol, or document section:
Accessed:
License/provenance note:
Observation (paraphrased):
Design implication:
Limitations or conflicting evidence:
Confidence:
```

For an experiment, additionally record the command, build flags, CPU/OS,
dataset or tensor shape, and retained artifact location. Do not commit model
weights, caches, binaries, traces, or bulk benchmark output.

## Source registry

Research source clones live outside this repository. New source checkouts should
use `/home/emmy/src/cpu-runtime-research/<project>` unless an existing checkout
is already suitable. Record the canonical origin, exact revision studied,
license, and purpose here before citing it. Do not add upstream source trees as
Git submodules or vendored code for this research phase.

Pinned research inventory on 2026-08-11:

Research baseline for this repository: `main` at
`3600fa45d5adeca6e183488c50d5359bf7e3177a`, plus the documentation-only dirty
tree that establishes this framework. Future source research should record the
then-current commit or dirty-tree state rather than assuming this baseline.

| Source | Local checkout | Canonical origin | Pinned local HEAD | License | Intended topics |
| --- | --- | --- | --- | --- | --- |
| llama.cpp | `/home/emmy/src/llama.cpp` | `https://github.com/ggml-org/llama.cpp.git` | `2468576f241235452013308597e6de1b78866996` | MIT | CPU kernels, graph execution, batching, server slots |
| mistral.rs | `/home/emmy/src/mistral.rs` | `https://github.com/EricLBuehler/mistral.rs.git` | `8010b6a0578e416120b590ed72fd46ed5f24ee85` | MIT | Rust ownership, scheduling, MoE and ragged batches |
| ik_llama.cpp | `/home/emmy/src/ik_llama.cpp` | `https://github.com/ikawrakow/ik_llama.cpp.git` | `26ceed9d4091a1696cf50e2ed87e5767d5811d81` | MIT | MXFP4 r8 layouts and packed GEMM/GEMV |
| OpenAI gpt-oss | `/home/emmy/src/cpu-runtime-research/openai-gpt-oss` | `https://github.com/openai/gpt-oss.git` | `7b583341fe16729127f6d5b94a7b09ccae97e1a1` | Apache-2.0 | Official MXFP4 and MoE numerical/dataflow semantics |
| vLLM | `/home/emmy/src/cpu-runtime-research/vllm` | `https://github.com/vllm-project/vllm.git` | `52be12cfac0c5a18ba906814b2d2bcadb40a9c4b` | Apache-2.0 | Token-catch-up scheduling and output commit |
| oneDNN | `/home/emmy/src/cpu-runtime-research/onednn` | `https://github.com/uxlfoundation/oneDNN.git` | `7a6406900252f010553dda6eca442610fbedc825` | Apache-2.0 | Explicit packing/scratch and AMX lifecycle |
| MegaBlocks | `/home/emmy/src/cpu-runtime-research/megablocks` | `https://github.com/databricks/megablocks.git` | `952db33d6eac334d22c61e47a0d5d41446298784` | Apache-2.0 | MoE route/group/unroute concepts |
| Sarathi-Serve | `/home/emmy/src/cpu-runtime-research/sarathi-serve` | `https://github.com/microsoft/sarathi-serve.git` | `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` | Apache-2.0 | Chunked prefill and decode-maximal scheduling |

The llama.cpp checkout above is newer than the revision used by an earlier
provenance audit in this repository. Each future observation must therefore
name its own exact revision; no conclusion silently inherits from the old
audit.

The full registry, primary-source list, environment evidence, and common
conclusions are maintained in
[`cpu-runtime-research/README.md`](cpu-runtime-research/README.md). Web sources
belong in workstream evidence cards, using stable canonical URLs,
document version/section, access date, and a local clone revision when
line-level source inspection matters.

## Research session protocol

For each bounded session:

1. Frame one or more explicit design questions and name the affected step.
2. Inventory the current repository path, types, invariants, tests, and known
   constraints relevant to those questions.
3. Inspect the pinned local upstreams and record evidence cards, including
   meaningful differences rather than only similarities.
4. Search primary web sources to verify ISA, OS, compiler, algorithm, and API
   assumptions.
5. Clone an additional project only when it answers a specific question; pin
   it in the source registry before using it.
6. Synthesize algorithms, data movement, persistent layout, scratch ownership,
   fallback behavior, portability, and failure modes.
7. Record competing alternatives and what evidence would choose between them.
8. Update provisional decisions, risks, and open questions without erasing
   superseded conclusions.
9. Mark a workstream planning-ready only when its exit criteria are met.

The dated research journal is append-only. The workstream synthesis and ledgers
are maintained summaries, so readers do not have to reconstruct the current
design from chronological notes.

## Workstream template

Each step will gain a detailed section or a linked topic document using this
shape:

```text
Status and planning-readiness:
Objective and non-goals:
Current repository baseline:
Concrete design questions:
Algorithm and dataflow findings:
Persistent and transient memory layouts:
ISA mapping and numerical representation:
API, ownership, and lifetime findings:
Threading, scheduling, and topology findings:
OS/compiler portability and fallback behavior:
Correctness invariants and focused test strategy:
Upstream/source comparison:
Alternatives considered:
Provisional design synthesis:
Risks and unknowns:
Open questions and required evidence:
```

If a topic becomes too large, move its evidence and synthesis into
`docs/cpu-runtime-research/NN-topic.md` and retain a status, decision summary,
and link here. Stable evidence and decision IDs should survive that split.

## Workstream register

| Step | Research state | Planning state | Durable synthesis |
| --- | --- | --- | --- |
| 1. AVX-512/VNNI x8 GEMV | Source/specification pass complete | Ready | [`01-avx512-vnni-x8.md`](cpu-runtime-research/01-avx512-vnni-x8.md) |
| 2. GEMM/prefill | Source/API/dataflow pass complete | Ready | [`02-gemm-prefill.md`](cpu-runtime-research/02-gemm-prefill.md) |
| 3. State separation | Ownership/failure pass complete | Ready | [`03-model-sequence-state.md`](cpu-runtime-research/03-model-sequence-state.md) |
| 4. Scheduling | Scheduler/lifecycle pass complete | Ready | [`04-cpu-batching-scheduling.md`](cpu-runtime-research/04-cpu-batching-scheduling.md) |
| 5. AMX | ISA/OS/toolchain pass complete | Ready | [`05-amx-prototype.md`](cpu-runtime-research/05-amx-prototype.md) |

## Planning-readiness criteria

A step is ready for a detailed implementation plan only when:

- the current-repository baseline and affected call graph are documented;
- relevant upstream implementations and primary specifications have been
  inspected at pinned revisions;
- numerical semantics and correctness invariants are explicit;
- dataflow, persistent layout, bounded scratch, and ownership are sketched;
- portable/scalar fallback and ISA/OS/compiler gates are specified;
- at least two material alternatives are compared where appropriate;
- cross-step dependencies and API effects are understood;
- focused pre-tuning tests can demonstrate mostly-correct behavior;
- unresolved questions are either answered or explicitly safe to defer;
- the design does not depend on an unperformed benchmark to be correct.

Planning-ready does not mean performance-certified or eligible for automatic
dispatch.

## Decision ledger

These decisions are inputs to the next per-item implementation plans. A later
measurement or implementation constraint may supersede one, but the old row is
retained rather than silently rewritten.

| ID | State | Decision | Revisit trigger |
| --- | --- | --- | --- |
| PD-001 | Accepted | Reuse `InterleavedSplitX8V2` for the first AVX-512 x8, SIMD GEMM, and transient AMX-feed designs. | Correctness or bounded-memory evidence requires a new persistent layout. |
| PD-002 | Accepted | Keep all new kernels forced/experimental and leave automatic dispatch on AVX2 x8. | Post-feature benchmark and cross-host tuning evidence. |
| PD-003 | Accepted | Define and validate the scalar matrix contract before SIMD implementations. | A correctness-preserving API constraint requires joint implementation. |
| PD-004 | Accepted | Split immutable model state from sequence state before multi-sequence execution. | Ownership implementation exposes a smaller equally safe seam. |
| PD-005 | Superseded by PD-008 | Make scheduling consume one common workload descriptor rather than kernel-specific entry points. | Research showed engine rows and matrix problems require different fields. |
| PD-006 | Accepted | Build AMX on the common GEMM contract and keep it outside automatic serving. | Hardware/specification evidence requires an incompatible semantic contract. |
| PD-007 | Accepted | Treat NUMA/topology as described inputs now, but postpone tuned placement policy. | Correctness or bounded-memory behavior requires earlier policy. |
| PD-008 | Accepted | Use a `CpuStepBatch`-equivalent model descriptor and a separate `Mxfp4MatmulProblem`-equivalent kernel descriptor. | A required cross-layer invariant cannot be represented without coupling them. |
| PD-009 | Accepted | Stage per-layer/per-sequence KV updates and commit only after the scheduled step and sampling succeed. | A bounded alternative proves equivalent rollback with less memory. |
| PD-010 | Accepted | Group MoE work by expert, then unroute and reduce in stable source-row/top-k order. | Official model semantics change or an equally deterministic representation is needed. |
| PD-011 | Accepted | Use AMX signed INT8 with transient 16x32 weight panels; retain BF16 as a later alternative. | AMX-host evidence or semantic constraints invalidate the INT8 mapping. |
| PD-012 | Accepted | Unify canonical sequence ownership and use schedule-reserve/execute/commit rather than adding a third CPU scheduler model. | Engine architecture changes before implementation begins. |

When a decision changes, retain the row, mark it superseded, cite the evidence
or decision that replaced it, and add a new row.

## Question resolution register

- **OQ-001 / resolved:** the x8 bytes naturally provide 64-byte eight-row
  chunks. Reuse the layout and vectorize scale expansion; do not create a new
  cache initially.
- **OQ-002 / resolved:** use two descriptors: stable model-step rows above the
  kernel boundary and a matrix-only problem below it.
- **OQ-003 / resolved:** persist compact x8 weights; pack activation rows and
  backend scratch transiently in a caller-owned execution context.
- **OQ-004 / resolved:** share model mappings/configuration/dispatch/pool;
  retain KV/position per sequence, generation/RNG per candidate, and scratch
  per execution owner.
- **OQ-005 / resolved:** route after the batched router projection, group stable
  route records by expert, and restore/reduce by source row and original top-k
  rank.
- **OQ-006 / resolved:** scheduling reserves without progress; model and
  generation changes are staged; cancellation tombstones discard the affected
  prepared commit; batch failures commit nothing.
- **OQ-007 / resolved:** doubled E2M1 and Q8 map directly to signed AMX INT8.
  Store INT32 and apply row/column scales after every K=32 block.
- **OQ-008 / resolved:** distinguish CPUID, kernel XSTATE support, process
  permission, and thread tile state. Request permission before worker creation
  and configure/release tiles around every calling-thread scope.

Safe deferrals for the next implementation plans or later tuning are exact
SIMD/AMX register allocation, scheduler budget defaults, ring/paged KV storage,
persistent hot AMX panels, NUMA placement, and automatic crossover thresholds.
None changes the semantic interfaces recorded here.

## Deferred kernel revisit register

No kernel is considered permanently tuned during this sprint. The later
certification/tuning phase must explicitly revisit:

- scalar/reference MXFP4 GEMV and future GEMM;
- AVX2 single-row and promoted AVX2 x8 GEMV;
- existing AVX-512 reference/single-row paths;
- future AVX-512/VNNI x8 GEMV;
- future SIMD prefill/batched GEMM variants;
- future AMX-INT8 and/or AMX-BF16 prototypes;
- packing/cache construction, scratch allocation, threading, scheduling, and
  automatic crossover/dispatch policy.

That phase will define representative hosts and workloads, rerun long oracle
and API certification, measure cold/warm behavior, tune kernels and scheduling,
and decide what can become automatic or trusted. Results from small research
experiments before then are design evidence only.

## Research journal

### 2026-08-11 — Framework established

- Recorded the fixed five-step CPU sprint scope.
- Recorded intentionally targeted development gates and the deferred
  certification/benchmark/tuning work.
- Established evidence labels, source pinning/provenance rules, a repeatable
  research session protocol, planning-readiness criteria, and stable ledgers.
- Inventoried the three existing local upstream checkouts without yet claiming
  design findings from them.
- Research for the individual steps has not started. The next session should
  begin with a current-repository baseline and common vocabulary before source
  comparisons.

### 2026-08-11 — First source-grounded research pass completed

- Inspected the current CPU kernels, runner, worker, scheduler, sequence, and
  mmap/repack ownership paths at the recorded repository baseline.
- Inspected pinned local llama.cpp, mistral.rs, and ik_llama.cpp checkouts and
  added detached filtered checkouts of official gpt-oss, vLLM, oneDNN,
  MegaBlocks, and Sarathi-Serve outside the repository.
- Consulted the OCP MX specification, Linux XSTATE documentation, Intel AMX
  programming material, Rust toolchain status, and the original ORCA,
  PagedAttention, Sarathi-Serve, and MegaBlocks papers.
- Recorded detailed evidence, alternatives, proposed interfaces, failure
  semantics, and focused pre-tuning tests in the five linked workstream
  documents.
- Resolved the initial eight open questions and marked every workstream ready
  for its own decision-complete implementation plan.
- Retained exact implementation tiling, scheduling defaults, performance
  crossovers, NUMA placement, AMX hardware results, and broad certification as
  explicit later decisions rather than presenting unmeasured claims.
