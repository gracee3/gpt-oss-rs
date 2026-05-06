# Fused Linear/AddMM Producer API Artifact Reuse Plan

## Classification

```text
fused_linear_addmm_producer_api_artifact_reuse_plan_recorded
```

## Scope

This is a docs-only Workstream A reuse plan for the
`developer-message-user-smoke` final-token ordered-surface evidence. It records
how official producer/API artifacts can remain the oracle seam for attention
o-proj validation after the Rust/CUDA policy feasibility lane stopped.

This plan does not authorize implementation, backend selection, consumer
revalidation, runtime/default/CUDA behavior changes, output emission, ladder
continuation, correction metadata, tolerance passes, final-logit claims,
all-layer claims, server claims, or 4097/context-length claims.

## Decision Summary

Workstream A remains an official CPU Torch API seam:

- API family: CPU Torch module/F.linear/_C._nn.linear/addmm.
- Concrete form: `torch.addmm(bias, input_2d, weight_t_2d)`.
- Operator: attention o-proj BF16 linear with bias.
- Input: BF16 weighted-V.
- Weight: BF16 o-proj weight.
- Bias: BF16 o-proj bias.
- Bias behavior: fused before the final observable BF16 output.
- Output: BF16 full vector.
- Required comparison: exact full-vector parity.
- Rejected references: explicit matmul, explicit einsum, and unfused-bias
  forms remain negative controls.

Focus-lane-only clears remain diagnostic only. The official producer/API seam
is an oracle boundary, not a Rust runtime backend.

## Why Rust/CUDA Policy Work Stops For Now

The current Rust/CUDA policy feasibility lane is stopped because:

- Existing local helper/backend candidates were exhausted.
- cuBLAS and cuBLASLt candidates, including fused-bias epilogue experiments,
  did not clear the sampled producer/API reference set.
- CPU Torch addmm dispatch is stable across the tested CPU thread/backend
  settings.
- Bounded Rust CPU policy synthesis evaluated 350 named policies and found only
  partial/per-layer full-vector clears.
- The closure audit replayed all 238 missing selectable focus-clearing
  candidates full-vector.
- No single Rust CPU policy cleared layers 6, 10, 13, 16, 18, and 21 exactly.
- Residuals were at most one BF16 ULP in the simple residual analysis, but no
  shared rounding/tie rule was localized.
- Gate C CUDA mirror work is therefore not authorized.

The resulting state is:

```text
stop_policy_lane_preserve_official_api_seam
```

## Artifact Reuse Objective

Future validation-only Workstream A work may consume producer/API artifacts as
the oracle seam for attention o-proj validation. This is artifact consumption,
not runtime behavior.

The purpose is to preserve official o-proj boundary evidence while work moves
to other blockers or to separately approved validation consumers. Artifact
reuse should make the official seam explicit and reproducible without claiming
that a Rust/CUDA backend has been identified.

## Required Artifact Contract

Each reused o-proj producer/API artifact must record:

- layer index
- prompt/case
- source API path
- weighted-V input metadata
- o-proj weight metadata
- o-proj bias metadata
- official output metadata
- dtype, shape, and value count
- source status path
- full-vector comparison rules
- negative controls preserved

Tensor metadata should include dtype, shape, stride or layout when available,
contiguity when available, and source/provenance. Full-vector exactness is
required for equivalence claims; mismatch counts, max abs diff, mean abs diff,
first mismatch, and worst mismatch should be recorded when comparing consumers
to these artifacts.

## Allowed Future Usage

Producer/API artifacts may be used for:

- validation-only blocked-family attribution
- documenting expected o-proj boundary values
- comparing downstream consumers against official seam artifacts
- preserving Workstream A evidence while moving to other blockers

Allowed usage must keep the reference as an oracle artifact. It must not imply
production runtime parity, default model-runner parity, CUDA correctness, or
backend identity.

## Disallowed Usage

This plan does not allow:

- treating the producer/API seam as a Rust runtime backend
- selecting a backend
- consumer revalidation from this plan alone
- output emission
- ladder continuation
- final-logit claims
- all-layer claims
- server claims
- 4097/context-length claims
- tolerance-based parity
- correction metadata promotion
- CUDA mirror work without a new design review

## Future Revisit Conditions

Reopen Rust/CUDA policy work only if one of these occurs:

- new backend attribution identifies a concrete replayable CPU arithmetic or
  kernel rule
- a new narrow rounding/tie rule is justified by stronger residual analysis
- a future PyTorch/ATen source-level investigation yields a deterministic
  policy model
- GPU/sharded Torch oracle work creates a new official seam requiring separate
  analysis
- a new design review explicitly authorizes a different policy family

Absent one of these conditions, the correct state is to preserve Workstream A
as an official producer/API artifact seam.

## Relationship To Broader Milestone

This closes the current Workstream A Rust/CUDA policy feasibility lane and
preserves the ordered-surface taxonomy. The milestone remains validation-only:
the sampled o-proj evidence coheres around CPU Torch fused-linear/addmm
producer/API semantics, while no Rust/CUDA policy or backend has been selected.

The ordered-surface taxonomy remains unchanged:

- Workstream A is an official CPU Torch API seam for sampled o-proj blockers
  and controls.
- Workstream B support-gap cases remain retired for the current milestone.
- Workstream C raw-QK source-boundary evidence remains complete for the
  minimal producer/API set.
- No global policy switch is justified.

## Guardrails

- Docs-only.
- No implementation authorization.
- No backend selected.
- No consumer revalidation authorization.
- No production runtime routing change.
- No default model-runner behavior change.
- No CUDA kernel change.
- No Torch runtime dependency in Rust.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
