# MXFP4 Matrix API

The CPU-kernel crate exposes a validated, allocation-free matrix contract for
MXFP4 expert projections. It deliberately separates numeric matrix execution
from sequence identity, attention, cancellation, and scheduler policy.

## Problem shape and views

`Mxfp4MatmulProblem` represents the row-major operation

```text
C[M, N] = A[M, K] * B[N, K]^T + bias[N]
```

where `K` is a multiple of the 32-value MXFP4 block size. `B` is an
`Mxfp4MatrixView` over either canonical rows or the persistent
`InterleavedSplitX8V2` representation. `A` is one of:

- `Q8MatrixView`, containing `M` rows of `Q8Block` values; or
- `ResidualQ8MatrixView`, containing `M` rows of `ResidualQ8Block` values.

Activation views carry a block stride, so padded row storage is legal. Problem
construction validates nonzero dimensions, activation and weight block-count
agreement, bias length, output bounds, and all checked stride arithmetic. The
output is a caller-owned `f32` slice with an explicit row stride. Columns after
`N` in a padded output row are not written.

The scalar reference visits activation rows, output rows, and K blocks in
ascending order. Residual Q8 adds the primary contribution and then the
residual contribution for each K block. E8M0 handling is shared with GEMV:
`0x00` is `2^-127`, ordinary encodings are exact powers of two, and `0xff`
propagates NaN.

## Scratch ownership

Call `Mxfp4MatmulBackend::scratch_requirement(&problem)` before execution. The
returned `Mxfp4ScratchRequirement` gives the exact byte count and starting
address alignment. Pass a suitably aligned mutable slice to
`Kernels::mxfp4_matmul`; undersized or misaligned storage is rejected before
the kernel runs.

The scalar and automatic backends currently require no scratch. AVX2 uses
32-byte-aligned caller storage for a transient panel of up to four activation
rows. Q8 needs one pass per K block and residual Q8 needs two. The model runner
keeps this reusable buffer in its worker-local `CpuExecutionContext`; no
persistent activation panels or expanded weights are created.

## Backend policy

`Mxfp4MatmulBackend` has four serialized spellings:

| Value | M=1 | M>1 |
| --- | --- | --- |
| `auto` | Established dispatched GEMV | Scalar matrix reference |
| `scalar` | Scalar matrix reference | Scalar matrix reference |
| `avx2` | AVX2 matrix path | AVX2 4x8 path |
| `amx-int8` | Unavailable until the optional AMX feature is built | Unavailable until the optional AMX feature is built |

Explicit AVX2 requires x86-64 AVX2 support and
`InterleavedSplitX8V2` weights. Complete eight-output groups use the packed
4x8 microkernel; one-to-three input-row tails use the same bounded panel, and
one-to-seven output-row tails use the scalar reference over canonical tail
records. The microkernel is single-threaded. Callers own thread-pool and
disjoint-work partitioning policy.

There is intentionally no automatic GEMV/GEMM crossover or tuned shape
threshold. `auto` remains conservative until representative benchmarking and
the deferred certification campaign justify a promotion.

## Model integration

For multi-row `CpuStepBatch` execution, the model gathers embeddings once and
advances all rows through one transformer layer before starting the next.
Dense projections operate over the row collection; RoPE remains per row;
attention remains row-wise against committed KV plus earlier staged rows from
the same sequence. MoE routes are stably grouped by expert, executed as matrix
problems, and restored to source-row/top-k-rank order before weighted
reduction. Only rows marked `logits_required` receive final logits.

All KV rows and token-history effects remain in `PreparedCpuStep` until the
whole batch succeeds and its state revisions are validated at commit. Matrix
backend choice therefore does not weaken the transactional sequence-state
contract.

The server exposes the backend as
`--cpu-matmul-backend auto|scalar|avx2|amx-int8`; configuration files use the
serialized `device.cpu_matmul_backend` field and default to `auto`.
