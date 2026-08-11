# C4-A: MoE Orchestration

- Outcome: **deferred**
- Scope: candidate route/group/compute/unroute ownership and descriptor only
- Corpus gate: `E1-NEG-001: unavailable`
- Source budget used: current repository, official GPT-OSS (NX-SRC-009), and
  MegaBlocks (NX-SRC-007)

## Objective, questions, and non-questions

C4-A asks which stable identities survive routing, how expert buckets describe
empty and small work, who owns route/group/unroute buffers, and where bounded
parallelism and instrumentation attach. It produces an operator problem that a
later implementation could serve without embedding scheduler policy.

It does not select a grouped kernel, choose an automatic bucket threshold,
change routing semantics, add capacity dropping, adopt GPU block padding,
parallelize experts, or claim a performance opportunity. Those choices require
the missing owner workload corpus.

## Current repository baseline

- **C4A-E-001 / CURRENT-REPO FACT:**
  `crates/gpt-oss-model-runner/src/cpu_runner.rs::moe_batch` projects router
  logits in a row batch, rounds them through BF16, rejects nonfinite logits,
  selects stable top-k experts, and rounds softmax weights through BF16.
- **C4A-E-002 / CURRENT-REPO FACT:** `stable_top_k` orders descending logits
  with lower expert index as the tie break. `stable_group_routes` then uses a
  stable expert-key sort, preserving source-row and rank order inside a bucket.
- **C4A-E-003 / CURRENT-REPO FACT:** each bucket clones its source rows,
  computes MXFP4 gate/up and down projections, and stores results in a dense
  `[source_row][rank]` option matrix. Unrouting consumes rank order, accumulates
  FP32 weighted outputs, then rounds the row through BF16.
- **C4A-E-004 / CURRENT-REPO FACT:** bucket scratch and cloned inputs are
  allocated during the operation. The surrounding `CpuExecutionContext`
  provides reusable kernel scratch, but there is no explicit route-workspace
  contract or whole-request thread budget in this function.
- **C4A-E-005 / EXPERIMENT STATUS:** the corpus gate supplied no expert-bucket
  distribution by prompt/decode phase, effective kernel, thread count, or
  repetition. It cannot rank row-wise, grouped, or expert-parallel execution.

## External evidence cards

### C4A-E-006 / LOCAL-SOURCE OBSERVATION

- Question: what ordering belongs to GPT-OSS model semantics?
- Source: NX-SRC-009, official GPT-OSS research checkout
- Pin/path: `7b583341...`; `gpt_oss/torch/model.py::MLPBlock.forward`
- Observation: routing uses sorted `topk`, selected logits are softmaxed, each
  selected expert computes gate/up, SwiGLU, and down, and expert results are
  reduced by the corresponding weights.
- Implication: expert selection, per-row rank, weight association, and the
  numerical reduction boundary are semantic inputs, not scheduling metadata.
- Limitation/conflict: PyTorch's tie behavior and internal reduction order are
  not a portable orchestration contract; the repository's explicit lower-index
  tie rule remains the current authority.
- Confidence: high for stage structure, moderate for exact-order transfer.

### C4A-E-007 / LOCAL-SOURCE OBSERVATION

- Question: which reusable abstraction exists for grouped expert work?
- Source: NX-SRC-007, MegaBlocks
- Pin/path: `952db33d...`; `megablocks/layers/dmoe.py::{
  indices_and_padded_bins,sparse_forward_once}`
- Observation: expert IDs are sorted, histograms and bin bounds describe work,
  rows are gathered, expert computation runs over grouped rows, and a scatter
  associates results and weights back with their sources.
- Implication: routes plus expert offsets are a sufficient storage-neutral
  seam between routing, expert compute, and unrouting.
- Limitation/conflict: the implementation is GPU/training oriented and pads to
  block sizes. Padding, capacity, topology tensors, and distributed behavior
  are not transferred to this CPU inference charter.
- Confidence: high for the decomposition, low for implementation transfer.

The lane stops after these sources: they establish two viable implementations
of the same decomposition but, without measured buckets, cannot choose one.

## Candidate problem descriptor

```text
MoeProblem<'a> {
  rows: RowSet<'a>,                 // stable row_id + BF16 hidden view
  hidden: usize,
  intermediate: usize,
  num_experts: usize,
  top_k: usize,
  routing: RoutingSemantics {
    tie_break: LowerExpertIndex,
    weight_boundary: Bf16AfterSoftmax,
    unroute_order: AscendingRank,
    capacity: NoDrop
  },
  experts: ExpertWeights<'a>,       // immutable identity-bearing views
  preparation: Option<PreparationHandle<'a>>,
  scratch: MoeScratch<'a>,
  execution: MoeExecutionMode,      // Reference | Forced(id) | Automatic
  thread_budget: ThreadBudget,
  diagnostics: DiagnosticSink<'a>
}

Route {
  source_ordinal: usize,
  row_id: StableRowId,
  rank: usize,
  expert: usize,
  weight_bf16_bits: u16
}

GroupedRoutes<'a> {
  routes: &'a [Route],              // key: (expert, source_ordinal, rank)
  expert_offsets: &'a [usize],      // length num_experts + 1
  gathered_rows: OptionalView<'a>
}

ExpertBucket {
  expert: usize,
  route_range: Range<usize>,
  m: usize
}
```

`row_id` is semantic; `source_ordinal` locates the input view. The grouping key
is fully specified even if a stable expert-only sort produces the same current
order. `expert_offsets[e] == expert_offsets[e + 1]` represents an empty bucket,
so all experts have a descriptor without dummy rows. Route count must equal
`rows * top_k` under checked arithmetic, `top_k <= num_experts`, and every
route field is range checked before expert work.

The caller owns input/output, route storage, offsets, optional gathered rows,
and scratch for the call. Preparation is immutable and separately accounted by
C2. The implementation writes no model or KV state. A successful result is one
output row per source; partial expert output cannot be committed. Diagnostics
may observe route/bucket summaries but do not own them.

## Stage contracts

1. **Route:** compute and round router logits and weights at the declared model
   boundaries. Emit ranks in stable top-k order. Nonfinite logits fail before
   grouping.
2. **Group:** produce routes ordered by `(expert, source_ordinal, rank)` and an
   `E + 1` offset array. It may gather rows or expose indices, but cannot alter
   row/rank identity.
3. **Compute:** process one bucket using immutable expert weights. `m=0` is a
   no-op and `m=1` is valid; neither may require a dummy padded row. Any internal
   padding is scratch and cannot become a semantic route.
4. **Unroute:** associate results by stable `row_id` and rank, multiply by the
   recorded BF16 routing weight, accumulate strictly in ascending rank under
   the C3 numerical contract, then apply the final BF16 boundary.

The descriptor has a single `ThreadBudget`. A backend may divide it across
rows, experts, or a matrix kernel, but nested pools cannot independently spend
the full budget. Expert-parallel work must also bound live bucket scratch under
the C2 grant; admission cannot rely on an average number of nonempty experts.

## Instrumentation contract

Production metrics expose only bounded bucket-size histograms, empty/singleton
bucket counts, routed-row counts, route/group/compute/unroute monotonic
durations, selected backend class, fallback reason, and scratch high-water.
Expert ID and row ID are forbidden labels. An opt-in trace may record exact
expert ID, ordered `(row_id, rank)` pairs, offsets, requested/effective mode,
and hashes of route and output buffers. Offline manifests preserve phase
(prefill/decode), total rows, repetitions, thread policy, effective dispatch,
and the full bucket-size vector.

## Alternatives and decision

| Alternative | Finding |
| --- | --- |
| Keep row-wise selected-expert execution | Viable correctness baseline with simple ownership; repeated input traversal and small calls are possible costs, not measured facts. |
| Stable group with offsets and grouped expert calls | Retained descriptor candidate; it exposes reusable work and empty buckets without requiring a kernel choice. |
| Run nonempty experts concurrently | Deferred. It can oversubscribe matrix kernels and multiply scratch; no corpus establishes bucket sizes or a safe split. |

**C4A-D-001 / PROVISIONAL DECISION:** retain the explicit route plus `E + 1`
offset descriptor as the planning seam. Do not select grouping storage,
parallelism, padding, or automatic thresholds until E1-complete evidence
describes real bucket work and memory pressure.

## Failure modes and focused future tests

- Equal or nonfinite logits destabilize routing: assert lower-index ties and
  reject nonfinite values before any expert computation.
- An empty expert inherits the preceding range: test all-empty-except-one,
  leading/trailing empty experts, and `offsets.len() == E + 1`.
- Small/skewed buckets take an uncovered fallback: force each mode for `m=1`,
  small tail sizes, one dominant expert, and evenly populated experts.
- Grouping loses rank identity: permute expert execution order and require
  bit-identical rank-order unrouting.
- Checked sizes overflow or scratch is short/misaligned: reject without output
  mutation and release/refund the C2 reservation.
- Nested parallelism exceeds the budget: instrument live worker count and
  scratch high-water under row-, expert-, and kernel-parallel candidates.
- A partial bucket failure leaks output: inject failure before and after each
  bucket and assert no model commit plus deterministic C1 cleanup.
- Automatic mode reaches an untrusted tail: require C3 coverage for the exact
  effective bucket shape or a separately trusted reference fallback.

The future matrix includes zero input rows, one row, `top_k` one and maximum,
tie logits, every expert empty in turn, singleton/all-to-one/uniform buckets,
nonfinite weights, input/output alias rejection, stable ordering across thread
counts, BF16 route and unroute boundaries, forced-unavailable behavior, and
route/scratch accounting overflow.

## Risks, conclusion, and gate

The main risks are turning scheduling order into observable numerical order,
spending memory per expert outside a request grant, and allowing grouped
execution to obscure an uncovered fallback. The descriptor prevents those
errors only if row/rank identities and the one thread/memory budget remain
explicit.

The seam and tests are sufficiently bounded, but the implementation candidate
is **deferred**. Advancement requires an owner-supplied E1 manifest with
prefill/decode bucket distributions, effective modes, repetitions, and memory
high-water. That evidence may rank later candidates; it must not be used to
invent universal automatic thresholds.
