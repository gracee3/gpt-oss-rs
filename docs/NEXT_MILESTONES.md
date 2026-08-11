# Next Milestones

The native batch-one GPT-OSS CPU serving path is the CPU-first experimental
default. Capability-oriented dispatch and the size-neutral AVX2 x8 MXFP4 GEMV
layout are now promoted. CPU trusted mode remains blocked, and CUDA remains an
explicit experimental opt-in.

## Development gate policy

The project is intentionally in a feature-development phase. Until the current
CPU architecture milestone is feature-complete—eight-output AVX-512 MXFP4,
GEMM-like prefill, model/sequence state separation, the engine seams needed for
CPU batching/scheduling, and an AMX prototype—promotion gates remain targeted
rather than exhaustive.

Each feature slice must still pass the checks that directly protect its
correctness and safety:

- scalar/reference equivalence for every optimized kernel and packed layout;
- focused shape, tail, extrema, and invalid-input tests;
- targeted full-checkpoint parity covering the changed execution path;
- atomic cache creation/reopen checks when storage changes;
- relevant streaming/non-streaming API smoke tests when engine behavior changes;
- formatting, locked workspace tests, and warnings-denied kernel linting.

For this development phase, the following are explicitly deferred and do not
block architectural progress:

- the exhaustive 28-run official-oracle matrix;
- repeated Criterion suites, percentage performance thresholds, and cross-host
  dispatch tuning;
- fresh llama.cpp advisory captures;
- the complete API permutation matrix;
- trusted-mode certification review.

Semantic failures, cache corruption, memory-safety defects, and broken APIs are
not relaxed. New kernels may be implemented and exposed through forced or
experimental paths, but automatic dispatch should remain on the validated
baseline until the deferred benchmark-and-tuning phase supplies selection
evidence.

## Completed foundation

- once-detected CPU capabilities and precise per-operation requirements;
- immutable dispatch diagnostics that name the MXFP4 kernel and layout;
- canonical scalar/AVX-512 references plus `InterleavedSplitX8V2` AVX2 GEMV;
- layout-specific, source-hash-keyed, atomic memory-mapped repack caches;
- projection APIs, scalar equivalence tests, tail-row tests, packing/GEMV
  benchmarks, and exact-BF16 diagnostics;
- targeted full-model promotion on `harmony_122` and `harmony_262`, with peak
  RSS, startup, generation, and API evidence recorded outside Git.

The source-grounded research and pre-planning pass for M1-M5 is complete. The
implementation contracts, evidence, alternatives, and focused gates are in
[`cpu-runtime-research/`](cpu-runtime-research/README.md). Each milestone will
receive its own decision-complete implementation plan before code work starts.

## M1. Build a real eight-output AVX-512 MXFP4 GEMV

- prototype AVX-512 decode around relevant byte-manipulation subsets rather
  than mechanically widening AVX2;
- preserve exact E2M1/E8M0 behavior and residual-Q8 weight reuse;
- reuse `InterleavedSplitX8V2`; its 64-byte eight-row chunks directly support
  the researched ZMM/VNNI dataflow;
- cover real gate/up and down shapes, tails, extrema, and scalar equivalence;
- integrate the kernel behind a forced/experimental selection while leaving
  automatic dispatch on the validated AVX2 x8 baseline until later tuning.

## M2. Add a matrix contract and true GEMM-like prefill

- keep separate model-step and matrix-problem descriptors so scheduling and
  attention metadata do not leak into microkernels;
- retain canonical checkpoints and use the x8 persistent layout initially;
- pack multi-row Q8/residual-Q8 activations into bounded caller-owned scratch;
- implement a scalar/reference MXFP4 GEMM API before optimized backends;
- implement and integrate a SIMD packed-GEMM path for multi-token prefill;
- keep layout choice explicit during development rather than embedding an
  unmeasured automatic threshold.

## M3. Separate immutable model and per-sequence state

- separate immutable mapped weights, repack caches, model metadata, and shared
  execution resources from mutable token, attention/KV, sampler, and generation
  state;
- define ownership and lifetime boundaries that let several sequences share one
  model without cloning model-scale data;
- make cancellation, reset, cleanup, and sequence destruction explicit;
- preserve batch-one behavior while creating the seam required by scheduling.

## M4. Develop engine-level CPU scheduling seams

- generalize the CPU worker and scheduler beyond a single active sequence
  behind an experimental configuration;
- connect scheduler batches to the GEMM/prefill workload descriptors;
- preserve cancellation, token budgets, output ordering, and KV-cache isolation;
- implement stable prefill/decode rows and MoE route/group/unroute ordering;
- add NUMA/topology descriptors and ownership seams now, while deferring policy
  tuning until representative multi-socket hardware is available.

## M5. Prototype AMX outside automatic serving

- use signed doubled-E2M1 and signed Q8 with AMX-INT8 `TDPBSSD`;
- preserve 32-element E8M0 scale boundaries through INT32 accumulation;
- isolate tile configuration, OS permission handling, and fallback behavior;
- implement a scalar-cross-checked AMX-INT8 experiment behind an explicit
  experimental interface; retain BF16 as a later alternative;
- compile- and unit-test portable code without requiring local AMX hardware;
- do not select AMX automatically before the later AMX-host benchmark phase.

## After feature completion: certify and tune

Once M1-M5 are implemented and integrated, rerun the deferred exhaustive
oracle/API matrix, repeated microbenchmarks, cross-host measurements, and
advisory comparisons. Use that evidence to tune packing and dispatch
thresholds, decide which new kernels become automatic, and separately review
trusted-mode eligibility.

The detailed handoff and guiding constraints are in
[`MXFP4_CPU_BACKEND_HANDOFF.md`](MXFP4_CPU_BACKEND_HANDOFF.md). The living
source-research framework, stable five-step scope, and pre-planning ledgers are
in
[`CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md`](CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md).
