# M2 Plan — Matrix Contract and Layer-Major Prefill

- Research: [`../cpu-runtime-research/02-gemm-prefill.md`](../cpu-runtime-research/02-gemm-prefill.md)
- Dependency: M1 x8 layout/kernel and M3 transactional `CpuStepBatch`
- Automatic M=1 path: established GEMV
- Automatic M>1 path: scalar matrix reference

## Entry reconciliation

Before code changes, inspect M3 batch/execution ownership, every dense and MoE
projection call, quantization buffers, x8 cache access, attention/RoPE order,
CLI/config serialization, and route stability. Refine this plan before coding
if the landed M3 API changes borrow or transaction boundaries.

## Interfaces

- `Mxfp4MatmulProblem` describes M, N, K, output stride, bias, activation kind,
  typed activation matrix views, typed x8/canonical weight views, and scratch.
- Typed Q8 and residual-Q8 matrix views validate row stride, block count,
  scale/value extents, and alignment-independent safe bounds.
- `Mxfp4MatmulBackend` is explicit and queryable: `Auto`, `Scalar`, `Avx2`,
  and feature-gated `AmxInt8` (implemented by M5).
- A backend reports scratch size/alignment before execution. Kernels allocate
  nothing and use caller-owned aligned scratch. Output is row-major with an
  explicit stride.
- Scalar is the semantic reference. AVX2 packs up to four activation rows and
  computes eight outputs from `InterleavedSplitX8V2`; output and row tails fall
  back without overread. The persistent weight cache is unchanged.

Expose `--cpu-matmul-backend auto|scalar|avx2|amx-int8`, serialized default
`auto`. During M2, `auto` routes M=1 to established GEMV and M>1 to scalar.
Explicit `avx2` selects the matrix path where valid and documents tail fallback.

## Layer-major execution

For a multi-row `CpuStepBatch`, execute each layer across all rows before
advancing to the next layer:

1. batch embeddings and normalizations/projections;
2. apply RoPE per row using the row's absolute position;
3. compute attention row-wise against committed plus earlier staged rows for
   that sequence, preserving ragged causal visibility;
4. route MoE rows stably, group expert work, and unroute to original order;
5. keep all KV/history effects staged through complete execution;
6. produce final logits only for rows with `logits_required`.

Attention stays row-wise initially. Do not add an automatic GEMV/GEMM
crossover, persistent matrix-specific weights, or scheduler policy here.

## Commit slices

1. Add problem/view/backend/scratch contracts and scalar reference tests.
2. Implement AVX2 4x8 activation-panel path and tail/scratch tests.
3. Add CLI/config serialization and explicit dispatch while preserving `auto`.
4. Convert dense projections and transactional model steps to layer-major
   multi-row execution, then stable MoE route/group/unroute.
5. Close out API/runtime docs, research status, and evidence.

## Focused gate

- scalar equality across M `1, 2, 4, 5`, N x8 boundaries/tails, multiple K
  blocks, extrema, bias, Q8, and residual-Q8;
- scratch exact-size, one-byte-short, alignment, stride, overflow, and canary
  coverage;
- explicit AVX2 selection and `auto` routing tests;
- independent ragged sequences and absolute-position RoPE;
- transactional attention with sliding/full KV boundaries;
- deterministic stable MoE route and unroute ordering;
- logits absent/present exactly as marked;
- batch-one compatibility and multi-row equivalence to row-at-a-time reference;
- formatting, kernel warnings-denied Clippy, locked kernel/model-runner tests.

## Documentation updates

- A matrix API document covering views, layout, scratch, output stride, backend
  selection, fallback, and automatic policy.
- CLI/config reference for `--cpu-matmul-backend`.
- CPU runtime prefill and attention/MoE behavior.
- research M2 status and this evidence ledger.

## Deviations and decisions

- None recorded yet.

## Completion evidence

- Implementation commits: pending
- Commands/results: pending
- Layer-major fixture evidence: pending
- Closeout commit/workflow: pending

