# Milestones and maintenance state

## Current state

The planned CPU-first research program is complete at v0.1.0. This file is no
longer an automatic implementation queue. The default state is maintenance:

- repair correctness, reproducibility, attribution, documentation, security,
  or evidence defects;
- keep dependencies and CI narrowly supportable;
- preserve the scalar oracle, exact fixtures, archive reachability, and
  release checksums; and
- avoid production-readiness, 120B execution, heterogeneous, or multi-GPU
  claims not supported by new controlled evidence.

The [final report](research/FINAL_REPORT.md) and [benchmark
protocol](research/BENCHMARK_PROTOCOL.md) define the closed publication
boundary.

## Completed CPU program

The following historical milestones are implemented and retained:

| Milestone | Closed result | Detailed record |
| --- | --- | --- |
| M1: AVX-512/VNNI x8 MXFP4 | Forced exact-bit implementation; promotion remains evidence-gated | [`cpu-runtime-plans/M1_AVX512_VNNI_X8.md`](cpu-runtime-plans/M1_AVX512_VNNI_X8.md) |
| M2: matrix contract and prefill | Typed scalar/SIMD matrix paths and transactional layer-major prefill | [`cpu-runtime-plans/M2_MATRIX_PREFILL.md`](cpu-runtime-plans/M2_MATRIX_PREFILL.md) |
| M3: model/sequence ownership | Immutable shared model and transactional per-sequence state | [`cpu-runtime-plans/M3_MODEL_SEQUENCE_STATE.md`](cpu-runtime-plans/M3_MODEL_SEQUENCE_STATE.md) |
| M4: CPU scheduling | Bounded reserve/execute/commit scheduling and cancellation seams | [`cpu-runtime-plans/M4_CPU_SCHEDULING.md`](cpu-runtime-plans/M4_CPU_SCHEDULING.md) |
| M5: AMX prototype | Forced-only portable packing/emulation; no local AMX performance claim | [`cpu-runtime-plans/M5_AMX_INT8.md`](cpu-runtime-plans/M5_AMX_INT8.md) |
| E1/C1/C2 foundation | Versioned evidence, lifecycle/delivery, and logical memory reservations | [`cpu-runtime-plans/CPU_SERVICE_FOUNDATION.md`](cpu-runtime-plans/CPU_SERVICE_FOUNDATION.md) |
| Iris Xe program | Explicit experimental path; full-model automatic promotion failed | [`xe-research/README.md`](xe-research/README.md) |

The implementation-era research questions, alternatives, gates, and deferred
certification remain available in [`cpu-runtime-research/`](cpu-runtime-research/README.md),
[`cpu-runtime-next-phase-research/`](cpu-runtime-next-phase-research/README.md),
and [`MXFP4_CPU_BACKEND_HANDOFF.md`](MXFP4_CPU_BACKEND_HANDOFF.md). Their
unchecked items are historical context, not standing authorization.

## Archived non-completions

Heterogeneous expert placement and multi-GPU layer sharding are preserved on
the immutable archive commits linked from the root README. They do not become
v0.1.0 milestones by inference.

- No GPT-OSS 120B end-to-end execution, parity, or performance result exists.
- The retained-20B capacity-one HET comparison did not complete its fixed
  matrix.
- The 58-commit layer-sharding branch did not execute activation handoff or
  reach parity.

The [multi-GPU retrospective](research/MULTI_GPU_RETROSPECTIVE.md) records the
reusable planner and ownership ideas without restoring old code.

## Gate for any post-v0.1.0 program

Before implementation, a proposed program must state:

1. one bounded research question and explicit exclusions;
2. responsible maintainer and available hardware;
3. semantic authority, provenance, and licensing;
4. the smallest source/API surface to change;
5. correctness, lifecycle, memory, thermal, and failure gates;
6. raw and normalized evidence schemas, including negative-result handling;
7. cleanup and maintenance obligations; and
8. why the work belongs here rather than in an upstream project or separate
   repository.

A failed core gate blocks promotion. Slower or negative measurements remain
publishable and must not be hidden or converted into a looser threshold after
the fact.
