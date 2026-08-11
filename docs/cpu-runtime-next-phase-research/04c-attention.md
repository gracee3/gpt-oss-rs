# C4-C: Attention

- Outcome: **deferred**
- Scope: candidate storage-neutral attention and KV-read contract only
- Corpus gate: `E1-NEG-001: unavailable`
- Source budget used: current repository and official GPT-OSS (NX-SRC-009)

## Objective, questions, and non-questions

C4-C asks how a row identifies its sequence and absolute position, which
committed and same-step staged KV rows are visible, how causal/sliding/GQA/sink
semantics are expressed, and how scratch and numerical boundaries can be
reviewed without exposing contiguous KV storage as the operator interface.

It does not page KV, reuse prefixes, adopt an online-softmax kernel, choose a
tile size, fuse QKV projections, alter cache capacity, or assert performance.
The descriptor is storage-neutral so C2 may keep contiguous KV now.

## Current repository baseline

- **C4C-E-001 / CURRENT-REPO FACT:** `CpuKvCache` in
  `crates/gpt-oss-model-runner/src/cpu_runner.rs` owns contiguous BF16 key and
  value vectors, logical `len`, token width, capacity, and `start_position`.
  At capacity, append shifts both vectors and advances the absolute start.
- **C4C-E-002 / CURRENT-REPO FACT:** `StagedKvRow` carries absolute position,
  key, and value. `PreparedSequenceDelta` holds staged rows per layer until
  commit. Batched layer execution appends the current row before calling
  `attention_one_staged`, so a row can read committed KV plus the ordered
  staged prefix for its own sequence without publishing it to canonical state.
- **C4C-E-003 / CURRENT-REPO FACT:** `attention_one_staged` combines committed
  and staged lengths, clips the oldest rows to cache capacity, maps a query
  head to `head / (num_heads / num_kv_heads)`, and computes each query head
  row-wise. It validates widths but relies on its caller for sequence identity,
  consecutive positions, divisibility, and causal staged ordering.
- **C4C-E-004 / CURRENT-REPO FACT:** the QK FP32 sum is rounded through BF16,
  scaling is rounded through BF16, softmax probabilities are rounded through
  BF16, the value contraction accumulates FP32, and its result is rounded
  through BF16. Each learned sink participates in the denominator with an
  implicit zero value.
- **C4C-E-005 / CURRENT-REPO FACT:** sliding and full layers share the
  attention function. Current cache capacity determines retained history;
  absolute cache start is not passed explicitly to the arithmetic loop.
- **C4C-E-006 / EXPERIMENT STATUS:** no accepted owner corpus reports context
  length, sliding/full mix, M, staged-row count, effective path, repetitions,
  or scratch. It cannot rank row-wise and tiled/online implementations.

## External evidence card

### C4C-E-007 / LOCAL-SOURCE OBSERVATION

- Question: what model semantics must an attention problem preserve?
- Source: NX-SRC-009, official GPT-OSS research checkout
- Pin/path: `7b583341...`; `gpt_oss/torch/model.py::{sdpa,
  AttentionBlock.forward}`
- Observation: keys and values expand over grouped query heads; causal masking
  excludes future positions; a positive sliding window also excludes old
  positions; learned sinks are appended to QK logits before softmax and their
  probability has no value contribution; the result contracts visible V rows.
- Implication: GQA mapping, absolute causal/sliding bounds, sink logits, and
  numerical boundaries are semantic fields, not cache-backend choices.
- Limitation/conflict: the source handles a dense same-call token matrix, not
  committed/staged incremental KV ownership. Current repository transactions
  remain authoritative for visibility.
- Confidence: high.

The lane stops here. Current row-wise execution and a storage-neutral problem
with a future tiled implementation are viable, but workload evidence cannot
discriminate them.

## Candidate attention problem

```text
AttentionProblem<'a> {
  layer_id: StableLayerId,
  query_heads: usize,
  kv_heads: usize,
  head_dim: usize,
  scale: f32,
  sinks: VectorView<'a, f32>,          // one logit per query head
  window: Full | Sliding { tokens: NonZeroUsize },
  rows: &'a [AttentionRow<'a>],
  kv: &'a dyn KvRead,
  output: MatrixViewMut<'a, f32>,      // [rows, query_heads * head_dim]
  numerical: AttentionNumericalContract,
  scratch: AttentionScratch<'a>,
  execution: AttentionExecutionMode,   // Reference | Forced(id) | Automatic
  thread_budget: ThreadBudget,
  diagnostics: DiagnosticSink<'a>
}

AttentionRow<'a> {
  row_id: StableRowId,
  sequence_id: StableSequenceId,
  absolute_position: u64,
  query: VectorView<'a, f32>,
  committed_revision: StateRevision,
  staged_visible_through: Option<StagedOrdinal>,
  declared_visible: AbsoluteRange
}
```

`query_heads % kv_heads == 0`, dimensions and output shape must match, scale
and sinks must be finite under the numerical policy, and every range operation
uses checked arithmetic. The GQA mapping is
`kv_head = query_head / (query_heads / kv_heads)`. M=0 is a no-op; otherwise
each row has exactly one output and a unique row ID.

For query position `p`, causal end is `p + 1`. The visible start is the KV
source start for `Full`, or `max(source_start, p + 1 - window)` for `Sliding`.
The effective range is the intersection of that interval, the declared range,
and the source's committed/staged availability. A gap, duplicate absolute
position, position after `p`, wrong sequence, stale committed revision, or
unexpected truncation is an error. Visibility is derived from positions, not
from concatenated vector length.

## Storage-neutral KV read seam

```text
trait KvRead {
  fn sequence_bounds(
    &self, sequence: StableSequenceId, layer: StableLayerId,
    revision: StateRevision
  ) -> Result<AbsoluteRange>;

  fn spans(
    &self,
    sequence: StableSequenceId,
    layer: StableLayerId,
    kv_head: usize,
    range: AbsoluteRange,
    staged_visible_through: Option<StagedOrdinal>
  ) -> Result<KvSpans<'_>>;
}

KvSpan<'a> {
  absolute_start: u64,
  rows: usize,
  key_row_stride: usize,
  value_row_stride: usize,
  keys: &'a [bf16],
  values: &'a [bf16]
}
```

Spans are borrowed, ordered, gap-free views for the requested absolute range.
They reveal no page, block, prefix, or allocation identity. Multiple spans are
allowed so contiguous storage, a future paging layer, or staged suffix can
serve the same semantic request. The source must filter staged rows by the
row's sequence/layer and `staged_visible_through`; the attention kernel cannot
discover transaction membership by pointer or position alone.

The current implementation can adapt its contiguous committed vectors plus a
staged suffix to this interface. This is a proposed contract only; no source
adapter is authorized here.

## Numerical and ownership contract

```text
AttentionNumericalContract {
  qk_product: Bf16Inputs,
  qk_accumulation: F32NamedOrder,
  qk_output_boundary: Bf16,
  scale_boundary: Bf16,
  softmax_reduction: F32,
  probability_boundary: Bf16,
  value_accumulation: F32NamedOrder,
  output_boundary: Bf16
}
```

The sink is a logit included in the stable maximum and denominator and has an
implicit all-zero value. Implementations must use a stable softmax formulation
and record how nonfinite inputs are handled. A future online formulation must
prove equivalence under the accepted C3 assertions; algebraic equivalence
alone does not inherit trust.

The caller owns queries, output, and scratch. `KvRead` retains committed/staged
storage for the call. Scratch requirements are queried before output mutation,
charged to C2, and include scores, online state, indices, or tiles—no kernel may
allocate storage proportional to context behind the contract. Partial row
results cannot be committed after a source/revision or kernel failure.

## Instrumentation contract

Production metrics expose bounded layer class (full/sliding), backend class,
mode, fallback reason, and bucketed visible-row/scratch sizes; layer number,
sequence ID, absolute position, and exact range are not labels. Opt-in traces
may record row/sequence-local IDs, absolute query/range bounds, committed
revision, staged count, span count, GQA dimensions, sink/output summaries, and
requested/effective path. Offline manifests separate prefill/decode, full and
sliding layers, visible context, row count, staged suffix, thread policy,
timer inclusions, repetitions, and artifact hashes.

## Alternatives and decision

| Alternative | Finding |
| --- | --- |
| Preserve direct row-wise access to contiguous cache plus staged vectors | Viable correctness baseline with current ownership; storage and visibility assumptions are coupled to the arithmetic loop. |
| Introduce `AttentionProblem` plus `KvRead`, retain row-wise reference, and later evaluate tiled/online execution | Retained seam candidate. It separates semantic visibility from physical storage without selecting paging or a kernel. |
| Design a paging- or prefix-aware attention API now | Rejected for C4-C. Those are C6 stressors and would prematurely choose storage/scheduling policy. |

**C4C-D-001 / PROVISIONAL DECISION:** retain absolute-position rows and the
borrowed span interface as the later planning seam. Keep the reference
numerical contract explicit. Defer tiling, fusion, scratch size, and automatic
selection.

## Failure modes and focused future tests

- A row reads a later staged row or another sequence: interleave staged rows
  from two sequences and vary `staged_visible_through` for every query.
- Eviction turns relative indices into wrong positions: use a nonzero committed
  start and query at just-before/at/after capacity.
- Sliding bounds are off by one: cover empty/one-token contexts and positions
  `window-1`, `window`, and `window+1`, including a truncated source start.
- GQA maps heads incorrectly or divides by zero: cover multiple group ratios
  and reject zero/non-divisible dimensions before span reads.
- The current query cannot see its own staged K/V, or sees a future one: assert
  exact ranges for multirow prefill and single-row decode.
- Sink probability contributes a value or is omitted from the denominator:
  construct sink-dominant and equal-logit cases with exact reference outputs.
- A tiled/online path moves BF16 boundaries: use cancellation-sensitive,
  halfway, NaN, infinity, and long-context rows under C3 assertions.
- Short/misaligned scratch or a span gap causes partial output: inject each
  error and require unchanged output, no commit, and C2 refund/release.
- Automatic context/tail fallback is uncovered: force each path at transition
  boundaries and require an exact trusted-evidence cell.

The minimum future matrix includes M=0/1/multiple, absolute position zero and
large nonzero starts, visible lengths 0/1/capacity, full and sliding boundary
edges, committed-only/staged-only/mixed sources, stale revisions, sequence
isolation, fragmented multi-span reads, every GQA head group, sinks, numerical
boundaries, scratch failures, forced-unavailable modes, and pre/post-commit
cancellation.

## Risks, conclusion, and gate

The highest risk is confusing physical cache order with semantic position and
transaction visibility. Other risks are multiplying score scratch by context,
moving numerical boundaries in an online algorithm, and granting trust to a
fallback reached only at a context edge.

C4-C is **deferred**. An owner-supplied E1-complete corpus must first establish
the distribution of row counts, visible full/sliding contexts, staged suffixes,
effective paths, repetitions, and scratch/memory pressure. Only then can a
later study rank row-wise versus tiled/online execution. The candidate seam
does not authorize paging, prefix reuse, or an automatic threshold.
