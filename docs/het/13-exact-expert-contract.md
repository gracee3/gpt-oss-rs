# Exact routed-expert contract

The current CPU implementation is the semantic source for this research. The
official OpenAI reference independently confirms BF16 routing, sorted top-k,
selected softmax, GPT SwiGLU, selected-expert projection, weighted reduction,
and residual ordering. CUDA compilation or close numerical output does not
override the repository's existing exactness gates.

## One-layer contract

| Boundary | Required invariant | Current evidence |
|---|---|---|
| Input | A contiguous or explicitly strided `[M, 2880]` BF16 post-attention-normalized activation, owned by the uncommitted layer step. Values must be finite where the current path rejects nonfinite routing. | `CpuModel::moe_batch` and `CpuModel::moe_one` in `crates/gpt-oss-model-runner/src/cpu_runner.rs`. |
| Router | BF16 weight and bias projection accumulates in f32; each logit is rounded through BF16 before selection and trace publication. | CPU trace and projection source; official `gpt_oss/torch/model.py`. |
| Selection | Select top 4 in descending logit order. Equal logits choose the lower expert ID. Rank `0..3` is semantic and must survive grouping/transport. | `stable_top_k` and `gpt-oss-moe-semantics::stable_top_k_indices`, including tie test. |
| Routing weights | Softmax only the four selected BF16-rounded logits in f32, then round each normalized weight through BF16 and retain its f32 value. | `moe_batch`; local layer-zero trace. |
| Route record | `(source_row, rank, expert_id, bf16_weight)`; grouping by expert is stable in original source-row/rank order. Empty experts do no work. | CPU `routes` construction and stable sort/group. |
| Expert weights | For expert `e`, gate/up rows are interleaved: even output is gate and odd output is linear/up. MXFP4 is 32 values per 16 packed block bytes plus one U8 exponent scale, with 90 blocks across H=2880. Bias semantics are BF16. | Local headers, loader/repack code, official weight reader. |
| Gate/up projection | Accumulate decoded quantized weights × BF16 input in f32, add bias, then round each projection result through BF16. | CPU reference projection. |
| GPT SwiGLU | Gate upper clamp is 7; up clamps to `[-7, 7]`. Preserve the scalar order and BF16 round points for alpha `1.702`, sigmoid, gate product, `(up + 1)`, and final product. | `gpt_oss_swiglu` in `cpu_runner.rs`; official reference. |
| Down projection | Accumulate in f32 from BF16 activated values, add bias, then round the unweighted expert result through BF16. | CPU reference and trace. |
| Contribution | Multiply each BF16 expert result by its BF16-rounded route weight in f32. Reduce contributions for each source row in routing-rank order `0,1,2,3`, then round final MoE output through BF16. | CPU `moe_batch`; `gpt-oss-moe-semantics` reduction. |
| Residual/output | Add the original layer residual only after MoE reduction, with the current BF16 output boundary. Output remains provisional until the layer/request step commits. | CPU block flow and `PreparedCpuStep`. |

**Verified:** prefill and decode use the same arithmetic contract. Prefill
`M > 1` groups many routes by expert; decode `M = 1` still has four rank-bearing
routes. Bucket execution order may change for concurrency, but contribution
identity and the final rank order may not.

**GPT-OSS-specific:** tensor names, H/I=2880, top-4 routing, interleaved gate/up,
MXFP4 decode, GPT SwiGLU constants/rounding, and residual position.

**Potentially reusable:** stable route records, single-owner placement lookup,
bounded destination batches, asynchronous job handles, contribution identity,
rank-ordered reduction, and prepare/discard/commit behavior. This classification
does not authorize genericizing attention, tokenizer, Harmony, or model loading.

## Narrowest candidate seam

The narrowest research candidate sits after router normalization and before
expert matrix execution:

```text
RoutedBatch {
  activation_rows: BF16 [M, H],
  routes: [{ source_row, routing_rank, expert_id, BF16 routing_weight }],
  layer/checkpoint identity
}
        -> backend-owned execution jobs
        -> [{ source_row, routing_rank, expert_id, BF16 [H] unweighted_output }]
        -> model-owned rank-ordered weighting/reduction/residual
```

**Hypothesis:** keeping weighting and reduction at the layer owner yields the
smallest exact cross-device contract and makes transport/retry evidence explicit.
The executor need not know attention, RoPE, KV layout, Harmony, or tokenization.
The expert backend still knows GPT-OSS MXFP4 and SwiGLU; hiding those assumptions
behind a falsely generic tensor operation would increase, not reduce, debt.

**Unknown:** whether returning unweighted BF16 outputs is the best actual wire
contract. Returning weighted BF16 contributions could reduce owner work but
moves a rounding point to each backend. The one-layer oracle must decide this
before an API is fixed.

## One-layer real-weight oracle

The fixture is pinned to local GPT-OSS-20B revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, layer 0, runtime config SHA-256
`3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce`,
runtime index SHA-256
`0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1`,
and tokenizer SHA-256
`0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3`.
It must retain:

1. input activation and residual identity;
2. BF16-rounded router logits;
3. selected expert IDs in routing-rank order;
4. normalized BF16 routing weights;
5. per-route gate/up projection, GPT SwiGLU output, and unweighted BF16 down
   output;
6. each rank's weighted contribution;
7. rank-ordered final MoE reduction and residual layer output;
8. placement, bytes transferred, synchronization events, and commit outcome.

The first cases are decode `M=1`, then actual 63-token prefill occupancies. The
local profile observed 597 nonempty expert buckets across 24 layers, median
`M=7`, p90 `M=24`, p95 `M=33`, maximum `M=61`; representative fixture cases are
therefore `M=7`, `24`, `33`, and `61`, not arbitrary square GEMMs.

## Acceptance policy

| Trace point | Gate |
|---|---|
| Selected IDs, source row, rank, expert identity, shape/order | Exact equality |
| BF16 activation, normalized routing weight, projection, SwiGLU, down output, MoE output, layer output | Bit-exact BF16 where the CPU contract fixes the rounding point |
| f32 router accumulation and weighted reduction | Bit-exact when the implementation preserves operation order; otherwise only an existing authoritative campaign ULP/tolerance gate may be invoked, with first divergence retained |
| Generated continuation | Exact token IDs against the pinned retained sequence before any 120B execution |
| Failure/cancellation | No visible output or KV/request revision change before a successful commit; exact cleanup/evidence outcome |

**Deferred:** changing a tolerance because a heterogeneous result is merely
close. The current Fresh CPU Oracle Campaign remains authoritative; retired
captures and the closed fused-linear policy lane cannot be used to lower this
fixture's gate.
