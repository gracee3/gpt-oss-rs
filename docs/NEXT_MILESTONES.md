# Next Milestones

The native batch-one GPT-OSS CPU serving path is the CPU-first experimental
default. Capability-oriented dispatch and the size-neutral AVX2 x8 MXFP4 GEMV
layout are now promoted. CPU trusted mode remains blocked, and CUDA remains an
explicit experimental opt-in.

## Completed foundation

- once-detected CPU capabilities and precise per-operation requirements;
- immutable dispatch diagnostics that name the MXFP4 kernel and layout;
- canonical scalar/AVX-512 references plus `InterleavedSplitX8V2` AVX2 GEMV;
- layout-specific, source-hash-keyed, atomic memory-mapped repack caches;
- projection APIs, scalar equivalence tests, tail-row tests, packing/GEMV
  benchmarks, and exact-BF16 diagnostics;
- targeted full-model promotion on `harmony_122` and `harmony_262`, with peak
  RSS, startup, generation, and API evidence recorded outside Git.

## M1. Complete deferred certification

- repeat the exhaustive 28-run official-oracle matrix on the promoted x8 path;
- rerun the repeated Criterion suite and fresh llama.cpp advisory captures;
- exercise the complete streaming/non-streaming API permutation matrix;
- retain performance as evidence until a stable cross-host threshold exists;
- keep trusted CPU mode blocked until this evidence receives separate review.

## M2. Build a real AVX-512 MXFP4 GEMV

- prototype AVX-512 decode around relevant byte-manipulation subsets rather
  than mechanically widening AVX2;
- compare AVX2, AVX-512, and AVX-512/VNNI on decode-shaped matrices, including
  clock and bandwidth effects;
- promote only kernels that preserve scalar correctness and win on their
  declared workload.

## M3. Add backend-specific packed GEMM layouts

- study interleaved/repacked layouts in ik_llama.cpp and related prior art;
- keep checkpoints canonical while versioning optimized CPU cache layouts;
- measure packed size, conversion time, cache locality, and GEMM improvement;
- use packing only where reuse amortizes its memory and load-time cost;
- establish prefill and batched-decode dispatch thresholds empirically.

## M4. Prototype AMX outside the serving path

- determine an exact small-integer representation for E2M1;
- preserve 32-element E8M0 scale boundaries through INT32 accumulation;
- compare AMX-INT8, AMX-BF16, and hybrid AVX-512/AMX feed strategies;
- compile- and unit-test without requiring local AMX hardware;
- benchmark on an AMX host before integrating or selecting it automatically.

## M5. Revisit engine-level CPU scheduling

Only after GEMM and AMX data exist:

- add prefill/decode workload descriptors to the model-runner/kernel boundary;
- evaluate CPU batching and concurrent-sequence scheduling;
- evaluate NUMA-aware packing and thread plans;
- reconsider trusted-mode eligibility against the new backend, full API tests,
  memory constraints, and official-oracle correctness.

The detailed handoff and guiding constraints are in
[`MXFP4_CPU_BACKEND_HANDOFF.md`](MXFP4_CPU_BACKEND_HANDOFF.md).
