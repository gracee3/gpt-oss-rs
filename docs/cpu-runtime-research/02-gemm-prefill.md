# Step 2 — MXFP4 GEMM Contract and True CPU Prefill

- Status: implemented and focused gates passed on the integration branch
- Initial optimized exposure: explicit/experimental
- Persistent layout default: `InterleavedSplitX8V2`

## Objective

Replace the current token-at-a-time prefill with a layer-major multi-row CPU
path and give MXFP4 projections a real matrix contract. The kernel problem and
the model/engine batch are separate interfaces: a GEMM kernel should know
matrix dimensions, numerical formats, layout, and scratch, but not sequence
IDs, cancellation, causal masks, or scheduler policy.

The semantic matrix operation is:

```text
A: M activation rows × K input features
B: N output rows × K input features (persistent MXFP4)
C: M activation rows × N output features
C[m, n] = bias[n] + sum_k A[m, k] * B[n, k]
```

GPT-OSS model widths are divisible by the K=32 microscaling block. The generic
API must reject invalid strides/dimensions rather than silently pad semantic
inputs. M and N tails are valid.

## Current repository baseline

`CpuModelRunner::prefill` in
`crates/gpt-oss-model-runner/src/cpu_runner.rs` clears one runner-owned KV
state and repeatedly calls `forward_token`. Every transformer layer therefore
sees M=1. Dense BF16 projections parallelize output rows. MXFP4 expert gate/up
and down projections prepare one Q8 or residual-Q8 activation and execute
eight-output tiles in parallel. Experts for the one token are evaluated
serially.

The kernel crate has a projection-tile API but no multi-row activation view,
output matrix stride, scratch-size contract, or matrix reference. The x8
persistent record is already independent of activation row count.

## Two-level proposed contract

### Model/engine batch

The model-facing descriptor should be equivalent to:

```text
CpuStepBatch
  rows: [CpuStepRow]

CpuStepRow
  sequence_id
  input_token_id
  absolute_position
  phase: prefill | decode
  logits_required
  sequence-local attention span / staged-row range
```

Rows have a stable batch index for the entire step. The descriptor may point to
ragged attention metadata, but it does not contain mutable sequence objects.
Workload labels such as `prefill`, `batched_decode`, or `mixed` are diagnostic
and selection hints; they cannot change numerical semantics.

### Matrix problem

The kernel-facing descriptor should be equivalent to:

```text
Mxfp4MatmulProblem
  m, n, k
  persistent weight view + Mxfp4WeightLayout
  Q8MatrixView | ResidualQ8MatrixView
  optional bias
  row-major output view + leading dimension
  explicit kernel/backend preference
  mutable caller-owned scratch view
```

Construction validates all lengths, K=32 block counts, layout compatibility,
bias length, output stride, and alignment/size requirements. It performs no
allocation. Thread pools and scheduling resources belong to the caller's
`CpuExecutionContext`; a microkernel is single-threaded.

The semantic activation views are row-major collections of current `Q8Block`
or `ResidualQ8Block` values. A backend may transform them into an internal
`Q8Panel4`, but transient packing is not exposed as the semantic API and never
outlives the call or owning execution context.

## Source findings

### GEMM-E001 — the x8 weights support both GEMV and GEMM

- **CURRENT-REPO FACT:** `InterleavedSplitX8V2` groups eight output rows while
  leaving activation preparation independent.
- **LOCAL-SOURCE OBSERVATION:** llama.cpp
  `ggml/src/ggml-cpu/repack.cpp` implements both
  `ggml_gemv_mxfp4_8x8_q8_0_generic` and
  `ggml_gemm_mxfp4_8x8_q8_0_generic` over `block_mxfp4x8`. Its matrix path
  packs four Q8 activation rows together.
- **LOCAL-SOURCE OBSERVATION:** pinned x86 llama.cpp and ik_llama.cpp paths pair
  x8 weight groups for wider AVX-512 matrix tiles instead of defining a second
  persistent MXFP4 encoding.
- **PROVISIONAL DECISION:** use x8 weights for the scalar, first SIMD, and AMX
  feed experiments. Version a new cache only if implementation evidence shows
  a correctness or bounded-memory need; performance speculation is not enough.

### GEMM-E002 — packing and scratch should be explicit caller resources

- **LOCAL-SOURCE OBSERVATION:** oneDNN `doc/ukernel/operations/brgemm.md` and
  `examples/ukernels/cpu_brgemm.cpp` require the caller to query B packing and
  scratch, allocate it, pass it to execution, and manage the hardware context.
- **INFERENCE:** this repository should not hide per-call allocation or thread
  global scratch in an optimized kernel. Explicit scratch enables reuse,
  bounded-memory tests, and later concurrent execution without changing the
  matrix API.

### GEMM-E003 — multi-token MoE is a routed matrix problem

- **LOCAL-SOURCE OBSERVATION:** OpenAI gpt-oss
  `gpt_oss/torch/model.py` keeps a `[batch, expert_slot, hidden]` result and
  performs the weighted sum over top-k slots. `gpt_oss/triton/moe.py` produces
  gather and scatter indices around expert matmuls.
- **LOCAL-SOURCE OBSERVATION:** mistral.rs GPT-OSS code exposes flattened token
  rows and gathered expert forwards. MegaBlocks expresses the broader
  route/group/unroute pattern, although its training/GPU block-sparse kernels
  do not transfer directly.
- **PROVISIONAL DECISION:** produce stable route records
  `(expert_id, source_row, topk_rank, routing_weight)`, group them by expert,
  and retain the original top-k rank as the unroute destination. Weighted
  reduction follows source-row then original top-k order, not expert execution
  order.

### GEMM-E004 — layer-major prefill is more than a matrix entry point

- **CURRENT-REPO FACT:** attention appends each layer's KV before moving to the
  next layer for a single token.
- **LOCAL-SOURCE OBSERVATION:** llama.cpp `llama_batch`/`llama_ubatch` carries
  token, position, sequence ID, and output selection per row; its context
  prepares model graph and memory for a batch. ORCA calls this selective
  batching: dense operations batch naturally while sequence-dependent
  attention remains logically ragged.
- **INFERENCE:** calling GEMM with `M > 1` inside the current token loop would
  not create true prefill. The model must advance a row matrix through one
  layer at a time and expose same-sequence causal relationships within the
  scheduled chunk.

## Scalar matrix reference

The first implementation slice should add the matrix descriptor and a scalar
reference before SIMD:

1. Iterate activation rows in ascending M order and output rows in ascending N
   order.
2. Initialize each result from `bias[n]` or zero.
3. Iterate K blocks in ascending order.
4. Decode each 32-value weight block to doubled-E2M1 integers and compute an
   exact INT32 dot with the row's Q8 block.
5. Add `dot * 0.5 * weight_scale * activation_scale` in FP32.
6. For residual-Q8, add the primary contribution and then residual
   contribution before advancing K, matching current projection semantics.

The reference is the contract oracle for matrix shape, layout tails, special
scales, bias, and accumulation boundaries. It is not a performance fallback
that expands the entire weight matrix.

## First packed SIMD path

The initial matrix microtile is four activation rows by eight output rows:

- pack four row-major activation blocks into a bounded Q8 panel organized in
  K=8 pieces so one decoded x8 weight chunk is reused across four rows;
- load/decode each 64-byte weight chunk once per K block;
- maintain 32 logical accumulators, applying each row's activation scale and
  each output row's E8M0 scale at the K-block boundary;
- repeat the dot for primary/residual activation panels while weights are live;
- write row-major output with the caller's leading dimension.

M tails use a smaller row count or the existing one-row GEMV tile. N tails use
the canonical row representation already stored by the cache. Later AVX-512
work may pair two x8 records and increase M, but this does not change the
semantic descriptor or persistent cache.

The caller parallelizes disjoint M/N or expert tiles through one Rayon pool.
The microkernel may not enter that pool. Tile ownership must prevent two
threads from sharing output or scratch cache lines.

## True layer-major prefill dataflow

For a `CpuStepBatch` containing prompt chunks and/or decode rows:

1. Validate sequence IDs, positions, context limits, and per-sequence row
   order; begin a model-state transaction.
2. Gather embeddings into a row-major `[M, hidden]` matrix.
3. For each transformer layer:
   - normalize all rows;
   - run dense Q/K/V projections as matrix operations;
   - apply RoPE independently using each row's absolute position;
   - stage K/V rows without committing sequence state;
   - compute attention row-wise or in a ragged kernel against committed prefix
     plus earlier staged rows from the same sequence, respecting full/sliding
     policy and causal order;
   - run the output projection as a matrix operation and add residuals;
   - route all rows, build stable expert buckets, run expert gate/up and down
     matrix operations, unroute, weight, and add the MoE residual.
4. Normalize final rows and compute logits only for rows with
   `logits_required=true`.
5. Commit staged KV and positions only if the entire step succeeds.

Intermediate prompt-chunk rows normally set `logits_required=false`; the last
prompt row that produces the first sample sets it true. Every ordinary decode
row sets it true. The model API must not infer this from batch position.

The first implementation may keep ragged attention row-wise. Dense projection
and MoE matrix work are enough to establish a real prefill dataflow; attention
optimization is a separate later kernel decision.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| One descriptor containing scheduler, attention, and matrix fields | Rejected. It couples microkernels to request policy and makes AMX/GEMV reuse harder. |
| Keep token-major prefill and only batch expert projections | Useful as an incremental localization tool, but not the target prefill architecture. |
| Persistently expand MXFP4 to INT8/BF16 | Rejected initially because of model-scale memory growth. Transient bounded panels preserve the compact cache. |
| Scatter-add expert results as each expert finishes | Rejected for deterministic semantics; execution order would affect FP32 reduction order. |
| Choose GEMV/GEMM automatically by M threshold | Deferred until representative measurements. Explicit backend/workload selection is sufficient during development. |

## Focused correctness plan

- matrix reference cases for M `1, 2, 3, 4, 5, 16`, N x8 groups plus tails
  `1..7`, and K one block plus real model widths;
- Q8 and residual-Q8 equality, bias/no-bias, zero/extreme blocks, invalid
  strides, output bounds, layout mismatch, and undersized scratch;
- SIMD/reference equality for every M/N tail and stable behavior when the
  caller changes tile partitioning;
- route/group/unroute fixtures with duplicate experts, empty expert buckets,
  uneven expert loads, stable top-k ranks, and deterministic weighted sums;
- one- and multi-sequence causal/sliding attention fixtures across prompt chunk
  boundaries;
- `logits_required` fixtures proving intermediate prompt rows do not allocate
  or expose logits;
- batch-one prefill/decode compatibility and a targeted checkpoint comparison
  after model integration;
- allocation accounting proving packed activation and scratch stay within the
  queryable bound.

No tuned tile size, automatic threshold, prefill throughput target, or long
oracle is required for the initial implementation gate.

## Research handoff

The semantic matrix shape, two-level descriptor boundary, persistent layout,
activation ownership, scalar accumulation, grouped-MoE order, and layer-major
prefill flow were handed to implementation without reopening the architecture.
The concrete Rust types, tile body, and commit sequence are recorded below and
in the M2 plan.

## Implementation status

M2 landed as checkpoints `a8a1e12`, `6c26e0f`, `85f5ab2`, and `fa1a733`.
The implemented contract follows the research boundary: typed Q8 and
residual-Q8 matrix views feed a validated `Mxfp4MatmulProblem`; scratch is
queried from `Mxfp4MatmulBackend` and owned by the caller; the persistent x8
cache is reused unchanged.

The scalar reference and explicit AVX2 4x8 path cover Q8/residual-Q8 shapes,
strides, scratch bounds/alignment, output canaries, extrema, and M/N tails.
Automatic M=1 uses established GEMV and automatic M>1 remains scalar. No
automatic crossover, tuning threshold, or new persistent matrix layout was
introduced.

Serving prefill now constructs a multi-row transactional batch. Execution is
layer-major, uses absolute-position RoPE, keeps ragged attention row-wise,
stably groups MoE routes by expert, restores original top-k rank before
reduction, and emits logits only for marked rows. Synthetic independent and
interleaved-sequence fixtures match isolated batch-one execution and preserve
rollback behavior. One-token 20B `harmony_122` captures for `auto` and explicit
AVX2 matrix execution both produced `200005`; captures remain outside Git
under `/data/models/openai/gpt-oss-rs-cpu-work/results/`.

Representative performance tuning, caller-side expert/tile parallelization,
optimized ragged attention, automatic crossover policy, long generation, and
the exhaustive oracle matrix remain deferred certification work.
