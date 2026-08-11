# Next Milestones

The native batch-one GPT-OSS CPU serving path is the current experimental
baseline. The next work is not to tune that baseline to one laptop; it is to
turn the CPU kernel crate into a portable, feature- and workload-dispatched
MXFP4 backend.

## M1. Freeze and measure the current baseline

- inventory the current scalar, AVX2, AVX-512/VNNI, repack, and dispatch code;
- preserve the seven-scenario official-oracle token baseline;
- add explicit MXFP4 decode, dot, GEMV, GEMM, and packing microbenchmarks;
- record cycles/weight, effective weight GB/s, TTFT, prefill tokens/s, decode
  tokens/s, load/repack time, and peak RSS separately;
- retain the current scalar and exact-BF16 diagnostic paths as correctness
  references.

## M2. Establish capability-oriented dispatch contracts

- replace coarse path assumptions with an explicit, once-detected CPU feature
  description;
- represent kernel requirements precisely (`AVX2+FMA`, AVX-512 subsets, VNNI,
  and later AMX subsets);
- separate GEMV and GEMM planning so they may use different packed layouts;
- make operation shape, batch size, thread count, and packing availability
  inputs to plan selection;
- keep all feature checks out of hot loops.

## M3. Strengthen AVX2 and build a real AVX-512 MXFP4 GEMV

- audit current nibble decode and integer-dot code against current llama.cpp
  and mistral.rs;
- benchmark an improved AVX2 baseline before replacing it;
- prototype AVX-512 decode around relevant byte-manipulation subsets rather
  than mechanically widening AVX2;
- compare AVX2, AVX-512, and AVX-512/VNNI on decode-shaped matrices, including
  clock and bandwidth effects;
- promote only kernels that preserve scalar correctness and win on their
  declared workload.

## M4. Add backend-specific packed GEMM layouts

- study interleaved/repacked layouts in ik_llama.cpp and related prior art;
- keep checkpoints canonical while versioning optimized CPU cache layouts;
- measure packed size, conversion time, cache locality, and GEMM improvement;
- use packing only where reuse amortizes its memory and load-time cost;
- establish prefill and batched-decode dispatch thresholds empirically.

## M5. Prototype AMX outside the serving path

- determine an exact small-integer representation for E2M1;
- preserve 32-element E8M0 scale boundaries through INT32 accumulation;
- compare AMX-INT8, AMX-BF16, and hybrid AVX-512/AMX feed strategies;
- compile- and unit-test without requiring local AMX hardware;
- benchmark on an AMX host before integrating or selecting it automatically.

## M6. Revisit engine-level CPU scheduling

Only after GEMM and AMX data exist:

- add prefill/decode workload descriptors to the model-runner/kernel boundary;
- evaluate CPU batching and concurrent-sequence scheduling;
- evaluate NUMA-aware packing and thread plans;
- reconsider trusted-mode eligibility against the new backend, full API tests,
  memory constraints, and official-oracle correctness.

The detailed handoff and guiding constraints are in
[`MXFP4_CPU_BACKEND_HANDOFF.md`](MXFP4_CPU_BACKEND_HANDOFF.md).
