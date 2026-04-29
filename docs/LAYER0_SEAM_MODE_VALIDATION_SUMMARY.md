# Layer0 Seam-Mode Validation Summary

This document summarizes the current layer0 final-token validation-runtime
milestone. It is a seam-mode validation result, not production runtime routing
and not all-layer parity.

## Milestone

The validation-runtime path can reproduce the layer0 final-token downstream
seams through the layer0 output when supplied with official/PyTorch MLP1 seams
and with the isolated expert3 lane `1990` selected-output anomaly corrected.

The result is exact modulo two explicit caveats:

1. Rust-native MLP1 BF16 `einsum` backend remains open.
2. Expert3 selected-output oracle lane `1990` appears to contain the pre-bias
   value for one lane.

## Exact Seam Evidence

| Seam | Status |
| --- | --- |
| K RoPE BF16-boundary | exact |
| Raw QK | exact |
| Masked logits | exact |
| Attention probabilities | exact |
| Weighted V BF16 boundary | exact |
| Attention o_proj, chunked pairwise BF16 linear policy | exact |
| Attention residual | exact |
| MLP norm | exact |
| Router logits | exact |
| Top-k/routing | exact |
| Selected experts from official/PyTorch MLP1 seams | exact except expert3 lane `1990` oracle anomaly |
| Weighted expert sum | exact after one-lane correction |
| MLP residual / layer0 final-token output | exact after one-lane correction |

## Caveat 1: Rust-Native MLP1 BF16 Einsum Backend

The native Rust replay of expert MLP1 still does not reproduce PyTorch BF16
`einsum` semantics.

Evidence:

- Expert30 MLP1 lane `522` is the tight repro.
- PyTorch BF16 `einsum` reproduces the official lane:
  - pre-bias `0.609375`
  - bias `-0.279296875`
  - output `0.330078125`
- The best bounded Rust explicit product/sum variant produced `0.33203125`.
- Scalar accumulation, partial-sum, running BF16 sum, chunked pairwise, and
  pre-bias rounding variants did not clear the lane.

This points to a dedicated backend design task, likely comparing PyTorch BF16
`einsum` against cuBLAS BF16, CUTLASS/custom CUDA, or another validation-only
BF16 matmul backend. Production routing should not change until selected
experts clear through that backend.

## Caveat 2: Expert3 Lane 1990 Oracle Anomaly

In the selected-output oracle, rank `0` / expert `3` / hidden lane `1990`
appears isolated.

Lane window finding:

- Lanes `1988`, `1989`, `1991`, and `1992` match post-bias.
- Lane `1990` alone equals pre-bias, not post-bias.

Lane `1990` values:

- official selected: `0.48046875`
- pre-bias: `0.48046875`
- bias: `-0.0016860962`
- post-bias: `0.478515625`

Rust and PyTorch agree on the post-bias value. Replacing only this lane with
the official selected-output value clears both downstream checks:

- Weighted expert sum: exact after one-lane correction.
- MLP residual / layer0 final-token output: exact after one-lane correction.

Source identity is recorded but not assumed equivalent: the selected-output
oracle metadata points to `/data/models/openai/gpt-oss-20b`, while the MLP1
seam pack was generated from
`/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`.

## What This Proves

The validation-runtime path can reproduce layer0 final-token downstream seams
through the full layer0 output when supplied with official/PyTorch MLP1 seams
and with the isolated expert3 lane `1990` oracle anomaly corrected.

## What This Does Not Prove

This milestone does not prove:

- production runtime routing
- Rust-native MLP1 BF16 `einsum`
- all layers
- all prompts
- final logits
- 4097-token behavior
- server/default runtime parity
- permission to import raw `.live` artifacts or proof harness code wholesale

## Recommended Next Tracks

Track 1: Rust-native BF16 MLP1 backend design.

Scope:

- keep expert30 lane `522` as the microbench/repro
- compare PyTorch BF16 `einsum` with cuBLAS BF16, CUTLASS/custom CUDA, or
  another validation-only BF16 matmul backend
- do not route production MLP until selected experts clear through the backend

Track 2: Layer0 seam-mode milestone preservation.

Scope:

- preserve a small status artifact if useful
- include no raw tensor values
- keep the exact seams and two caveats explicit
- keep layer ladder, final logits, and 4097-token work deferred
