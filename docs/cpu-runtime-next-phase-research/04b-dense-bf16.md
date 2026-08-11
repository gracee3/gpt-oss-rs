# C4-B: Dense BF16 Matrix Operations

- Outcome: **deferred**
- Scope: candidate dense-BF16 problem, preparation, and scratch contracts only
- Corpus gate: `E1-NEG-001: unavailable`
- Source budget used: current repository, official GPT-OSS (NX-SRC-009), and
  oneDNN (NX-SRC-006)

## Objective, questions, and non-questions

C4-B asks whether the current matrix-vector calls can be described as one
sibling dense-BF16 matrix problem without changing model arithmetic. It fixes
shape, stride, layout, accumulation, rounding, preparation, scratch, ownership,
and dispatch vocabulary for later evaluation.

It does not replace MXFP4 kernels, add oneDNN as a dependency, choose tiling or
packing, select an automatic M threshold, change BF16 semantics, or claim that
matrix batching is faster. No supplied evidence measures the relevant shapes.

## Current repository baseline

- **C4B-E-001 / CURRENT-REPO FACT:**
  `crates/gpt-oss-model-runner/src/cpu_runner.rs::project_bf16` validates a
  rank-2 `[N,K]` BF16 weight, runs one `Kernels::bf16_matvec` per output row,
  accumulates an FP32 output, and adds optional FP32 bias.
- **C4B-E-002 / CURRENT-REPO FACT:** `project_bf16_batch` represents M inputs
  as `Vec<Vec<bf16>>`, parallelizes across input rows, then loops all weight
  rows and calls the same matvec. M=0 returns empty and M=1 delegates to the
  scalar wrapper. It is a batched call surface, not a matrix kernel.
- **C4B-E-003 / CURRENT-REPO FACT:** callers apply model-specific BF16
  boundaries after projections. Router logits are rounded before top-k;
  attention Q/K/V and output paths have their own boundary placement. A generic
  kernel cannot guess whether its FP32 result should be rounded.
- **C4B-E-004 / CURRENT-REPO FACT:** CPU kernel dispatch includes scalar,
  AVX2, and AVX-512/VNNI eligibility. Preparation and scratch conventions are
  operation-specific; a requested name alone does not identify the path that
  executed.
- **C4B-E-005 / EXPERIMENT STATUS:** the owner results contain no E1-complete
  distribution of `(operation,M,N,K,strides,mode)` or warm repetitions.
  Decode M=1 and prefill/batched M>1 are known code shapes, but their frequency
  and cost are not recoverable evidence.

## External evidence cards

### C4B-E-006 / LOCAL-SOURCE OBSERVATION

- Question: which dense operations share official model semantics?
- Source: NX-SRC-009, official GPT-OSS research checkout
- Pin/path: `7b583341...`; `gpt_oss/torch/model.py::{AttentionBlock.forward,
  MLPBlock.forward}`
- Observation: BF16 linear operations produce attention Q/K/V and output,
  router logits, and expert intermediates, with model operations such as
  softmax and SwiGLU defining subsequent BF16 boundaries.
- Implication: one problem descriptor may cover dense shapes, but boundary
  policy stays explicit per call. MXFP4 expert storage remains a different
  sibling problem.
- Limitation/conflict: PyTorch operator choice, packing, and reduction order are
  not the required native CPU implementation contract.
- Confidence: high for operation roles, low for kernel transfer.

### C4B-E-007 / LOCAL-SOURCE OBSERVATION

- Question: what must a prepared matrix operation make explicit?
- Source: NX-SRC-006, oneDNN
- Pin/path: `7a640690...`; `examples/ukernels/cpu_brgemm.cpp` and the BRGEMM
  ukernel interfaces it exercises
- Observation: M/N/K, data and accumulator types, leading dimensions,
  K-blocking, output handling, platform support, and optional B packing are
  explicit; unsupported packing is reported before execution.
- Implication: preparation format and scratch/output contracts should be
  queried and validated rather than hidden inside the first timed call.
- Limitation/conflict: the inspected example uses integer inputs and C++; it is
  evidence for problem-shape vocabulary only, not a BF16 dependency or
  performance result.
- Confidence: high for interface vocabulary, low for direct implementation.

The source lane stops with two viable repository-level alternatives: preserve
row matvec execution or add a typed matrix seam. Evidence cannot choose an
implementation behind the seam.

## Candidate dense-BF16 problem

The semantic operation is `D[M,N] = A[M,K] * B[N,K]^T + bias[N]`.

```text
Bf16MatrixProblem<'a> {
  operation_id: DenseOperationClass,
  m: usize,
  n: usize,
  k: usize,
  lhs: MatrixView<'a, bf16> {
    data, layout: RowMajor, row_stride, logical_shape: [m, k]
  },
  weights: MatrixView<'a, bf16> {
    data, layout: OutputRowsByK, row_stride, logical_shape: [n, k]
  },
  bias: Option<VectorView<'a, f32>>,
  output: MatrixViewMut<'a, f32> {
    data, layout: RowMajor, row_stride, logical_shape: [m, n]
  },
  accumulation: F32,
  reduction_order: ReductionContract,
  output_boundary: KeepF32 | RoundEachElementToBf16,
  preparation: Option<DensePreparationHandle<'a>>,
  scratch: ScratchViewMut<'a>,
  execution: DenseExecutionMode,       // Reference | Forced(id) | Automatic
  thread_budget: ThreadBudget
}
```

All products consume BF16 values and accumulate FP32 according to the named C3
reduction contract. Bias is added exactly once in FP32 after the reduction.
`RoundEachElementToBf16` changes only the final stored value; intermediate
rounding requires a different reviewed reduction contract and evidence tuple.
Callers cannot infer a boundary from `operation_id`.

M=0 is a specified no-op that may use empty data views; N and K must be nonzero
for a model operation. Strides are in elements and must cover the logical
shape under checked arithmetic. Views must meet backend alignment requirements
but forced/automatic execution cannot read padding outside the declared view.
Overlapping mutable output and input/weight/preparation storage is rejected.
Output is entirely caller owned and cannot be partially published on error.

## Preparation and scratch contract

```text
DensePreparationKey {
  model_file_hash, tensor_name, tensor_content_hash,
  source_dtype, source_shape, source_strides,
  format_id, format_version, backend_id, isa_class
}

DenseRequirements {
  supported: bool,
  reason_code,
  preparation_bytes,
  preparation_alignment,
  scratch_bytes,
  scratch_alignment,
  output_alignment,
  reachable_fallbacks
}
```

Preparation is immutable after construction, validated by the complete key,
and charged as resident/repack memory under C2. A stale or mismatched handle is
an error, not a hint. Scratch is caller-owned for the duration of one call,
charged to its reservation, and may contain no persistent packed weights.
Requirement queries use checked arithmetic and occur before output mutation.

Reference mode names the portable, reviewable reduction order used by oracle
tests. Forced mode either executes the requested eligible backend or returns
`unsupported` before touching output. Automatic mode records the observed
backend and fallback reason. No backend may allocate unreported scratch or
prepare weights on an untimed first execute.

## Problem families and evidence boundaries

The descriptor must be exercised independently for:

- decode M=1 and prefill/batched M>1;
- combined attention Q/K/V, attention output, router, and language-model head;
- every current N/K vector tail and stride class;
- cold unprepared, explicit preparation, and warm-prepared states;
- scalar, AVX2, AVX-512/VNNI, any future AMX backend, and every reachable
  reference/tail fallback.

Coverage of one operation or M region gives no trust or performance claim to
another. In particular, a large prefill matrix cannot select a decode path and
a measured router shape cannot select the vocabulary head.

## Instrumentation contract

Production metrics use bounded operation class, backend class, mode, and
reason code. They may count calls/rows/elements, preparation hits/misses,
fallbacks, and scratch high-water and time prepare/execute separately with a
monotonic clock. Exact M/N/K, tensor name, and thread count belong in an
opt-in trace or offline manifest, not metric labels. Offline timings state
whether validation, preparation, allocation, bias, and output rounding are
included and report requested versus effective mode.

## Alternatives and decision

| Alternative | Finding |
| --- | --- |
| Preserve per-row BF16 matvec calls | Viable reference/baseline; simplicity is known, comparative cost is not. |
| Introduce the typed sibling matrix problem above | Retained seam candidate; it makes matrix work, ownership, rounding, preparation, and fallbacks reviewable without selecting a kernel. |
| Adopt a general external primitive/dependency now | Rejected for this charter. The source supplies vocabulary but the missing corpus cannot justify dependency, semantic, binary-size, or dispatch costs. |

**C4B-D-001 / PROVISIONAL DECISION:** retain the candidate problem for later
planning, but preserve the current row-matvec implementation as the only known
baseline. Do not set automatic thresholds or a preferred backend.

## Failure modes and focused future tests

- Shape/stride multiplication overflows or a short view passes validation:
  exercise maximum values and reject before requirements or output writes.
- M=1 and M>1 use different arithmetic: compare exact reference results and
  boundary bits across both paths for identical logical rows.
- N/K tails dispatch to an uncovered scalar path: force every tail class and
  require a C3 cell for its effective backend.
- Bias or final BF16 rounding occurs twice or in the wrong order: use halfway
  BF16 cases, cancellation-sensitive sums, and exact bit assertions.
- NaN, infinity, signed zero, and subnormal handling drifts: retain inputs and
  output bits, and define whether platform modes are part of the evidence key.
- Packed weights are stale: mutate tensor identity metadata and reject the
  preparation handle before execution.
- Scratch is short, misaligned, or aliases output: reject without partial
  output and refund the C2 grant.
- Automatic or forced execution silently changes modes: assert effective mode,
  fallback reason, and forced-unavailable failure.
- Nested row/kernel pools oversubscribe: exercise thread budgets and record
  live threads plus effective affinity.

The minimum future matrix includes M=0/1/multiple, N and K on/either side of
each vector/block boundary, padded row strides, bias absent/present, output
boundary variants, adversarial BF16 rounding, nonfinite values, cold/warm
preparation, insufficient scratch, aliasing, every forced mode, automatic
fallback, and repeatability across supported thread tuples.

## Risks, conclusion, and gate

The principal risks are accidentally changing model arithmetic while changing
call granularity, hiding persistent preparation in scratch, and tuning an
automatic path to unrepresentative shapes. The explicit boundary and identity
fields make those reviewable but do not resolve them.

C4-B is **deferred**. An E1-complete owner corpus must first report operation
roles, M/N/K and tail distributions, effective modes, thread policy,
preparation state, repetitions, timer inclusions, and memory high-water. That
evidence may rank row matvec and matrix candidates for a later plan; this study
does not claim performance or choose dispatch policy.
