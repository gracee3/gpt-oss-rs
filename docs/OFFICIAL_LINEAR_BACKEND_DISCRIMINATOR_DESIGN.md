# Official Linear Backend Discriminator Design

## Classification

`official_linear_backend_discriminator_design_recorded`

## Scope

This is a validation-only design focused on BF16 linear boundaries such as
attention o-proj, `attn.out`, and `torch.nn.functional.linear`. It is
motivated by the layer6 o-proj evidence where official live `attn.out` and
producer-side `F.linear` match the ordered o-proj artifact, while bounded
consumer accumulation variants and simpler matmul/einsum replay do not clear
the full vector.

This is not a production runtime design, not a default routing proposal, not a
CUDA kernel proposal, and not a Torch runtime dependency proposal. It does not
authorize layer output emission, ladder continuation, correction metadata,
tolerance-based pass criteria, or any final-logit, all-layer, server, or
4097-token claim.

## Motivation

Layer6 introduced an attention o-proj boundary that differs from the earlier
layer4/layer5 reverse-accumulation evidence.

Consumer statuses:

```text
/tmp/layer6_ordered_bundle_validate_status.json
/tmp/layer6_attention_oproj_policy_sweep_status.json
```

Producer status:

```text
/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json
```

Findings:

- Weighted V is exact.
- Raw-QK, masked logits, and attention probabilities are exact.
- MLP validation is exact.
- Strict/default o-proj has 2 mismatches.
- Focus lane `22`: local `9.125`, official `9.0625`, diff `0.0625`.
- Bounded consumer sweeps did not clear the full o-proj vector.
- Producer official `attn.out(weighted_flat)` matches the prior o-proj
  artifact.
- Producer `torch.nn.functional.linear` matches the prior o-proj artifact.
- Producer matmul/einsum full-vector replay has 826 mismatches.
- Weighted-V live versus prior artifact has 0 mismatches.

The evidence suggests the relevant boundary is the official BF16 linear
backend used by `attn.out` / `F.linear`, not a simple global accumulation-order
switch.

## Prior Related Evidence

- Layer0 o-proj previously cleared under a chunked-pairwise validation
  discriminator.
- Layer4 o-proj cleared under reverse f32 accumulation.
- Layer5 o-proj cleared under reverse f32 accumulation after the explicit
  weighted-V policy.
- Layer6 did not clear under current, reverse, pairwise, chunked-pairwise,
  f64 diagnostic, or bias variants without collateral mismatches.

Together, these surfaces rule out a single global o-proj accumulation switch.

## Stage 2 Backend Probe Result

Status:

```text
/tmp/layer6_official_linear_backend_discriminator_probe_status.json
```

Classification:

```text
layer6_oproj_backend_discriminator_collateral_mismatches
```

Implementation branch and commit:

```text
validation/official-linear-backend-discriminator
6e9643071ea07dcef1768ffc8ae1a05ba1697aad
```

Result: no backend selected.

| Backend | Full-vector result | Focus lane 22 | Candidate status |
| --- | --- | --- | --- |
| current sequential | 3 mismatches | not cleared | rejected |
| reverse | 2 mismatches | cleared | rejected due collateral |
| pairwise | 2 mismatches | cleared | rejected due collateral |
| chunked pairwise 16/32/64/128 | best chunk 128 has 1 mismatch | cleared | rejected due collateral |
| f64 diagnostic | 2 mismatches | cleared | diagnostic only |
| cuBLAS BF16 tensor-op | 1368 mismatches | cleared | rejected for layer6 o-proj |
| cuBLAS BF16 pedantic | 826 mismatches | cleared | rejected for layer6 o-proj |
| BF16-product guard | 1445 mismatches | cleared | evidence-only/rejected |

## Stage 2 Interpretation

Focus-lane clearing is not meaningful unless the full vector clears. Stage 2
confirms that local changes which repair lane `22` introduce or preserve
collateral o-proj mismatches.

cuBLAS tensor-op is not a layer6 o-proj parity backend. cuBLAS pedantic also
does not clear the official artifact; its 826 mismatches match the producer
matmul/einsum mismatch scale, which is evidence that pedantic-style replay is
closer to matmul/einsum than to official `F.linear` for this case.

Existing Rust accumulation families are exhausted for this layer6 surface:
current, reverse, pairwise, chunked pairwise, f64 diagnostic, and the
BF16-product negative guard all fail full-vector parity. Layer6 remains
blocked, and no runtime-policy or o-proj implementation discussion is
authorized from Stage 2.

## Stage 3 Producer-Side API Probe Result

Status:

```text
/tmp/layer6_attention_oproj_api_probe_status.json
```

Artifact directory:

```text
/tmp/layer6_attention_oproj_api_probe/
```

Oracle branch and commit:

```text
oracle/ordered-bundle-generation
0ae58d3617d62941c6a4e68624c83a2bf9ff21dd
```

Classification:

```text
layer6_oproj_producer_api_probe_layout_sensitive
```

Result: producer-side PyTorch API behavior is layout/fused-bias sensitive.
`F.linear` is not unique: module `attn.out`, `F.linear`,
`torch._C._nn.linear`, and fused `torch.addmm` all clear the full vector.
Explicit matmul/einsum forms do not explain the official artifact.

| Producer/API variant | Full-vector result | Focus lane 22 | Interpretation |
| --- | --- | --- | --- |
| module attn.out(weighted_flat) | 0 mismatches | clears | official module path |
| torch.nn.functional.linear | 0 mismatches | clears | matches artifact |
| torch._C._nn.linear | 0 mismatches | clears | matches F.linear |
| fused torch.addmm | 0 mismatches | clears | matches F.linear |
| weight @ input + bias | 826 mismatches | clears | not official-equivalent |
| input @ weight.T + bias | 826 mismatches | clears | not official-equivalent |
| torch.matmul | 826 mismatches | clears | not official-equivalent |
| torch.einsum | 826 mismatches | clears | not official-equivalent |
| stride-2 input view into F.linear | 826 mismatches | clears | layout-sensitive failure |
| F.linear(..., bias=None) + bias | 826 mismatches | clears | fused-bias-sensitive failure |

## Stage 3 Interpretation

The official artifact is explained by fused linear/addmm semantics with the
original contiguous BF16 input, weight, and bias layout. `F.linear` is not
unique: `_C._nn.linear` and fused `addmm` also match.

Explicit matmul/einsum are not sufficient proxies for official `F.linear`
parity. Unfused bias addition reproduces the 826-mismatch pattern. A stride-2
input view into `F.linear` also reproduces the 826-mismatch pattern, so input
layout matters.

MKLDNN enabled/disabled variants both clear, and default-thread vs
single-thread variants both clear; neither is the differentiator in the Stage
3 evidence. This remains producer evidence only and does not authorize
Rust/runtime implementation.

## Narrower Official Linear Backend Discriminator

The next discriminator is no longer "which PyTorch operator matches?" Stage 3
answered that: fused linear/addmm semantics match, while explicit
matmul/einsum and unfused-bias forms do not.

The next design should model fused linear/addmm semantics in a validation
discriminator, preserve layout/fused-bias constraints, and compare Rust/CUDA
candidates against fused addmm-like semantics only after that design update.
Avoid further blind accumulation sweeps.

Questions now:

1. Can a validation-only Rust/CUDA path reproduce fused addmm semantics
   without changing production defaults?
2. Which existing helper is closest to fused addmm semantics?
3. Can input layout and fused bias be represented explicitly in status/schema?
4. Should layer4/layer5 reverse o-proj be re-probed with producer API variants
   to see whether reverse was only coincidentally matching fused-linear
   semantics?
5. Should layer6 remain blocked until a fused-linear/addmm-specific
   discriminator is designed?

## Discriminator Goal

The discriminator should answer:

1. Is official fused BF16 linear/addmm parity reproducible by an existing
   validation-only Rust/CUDA helper?
2. Can the validation schema represent input layout, weight layout, and fused
   bias handling explicitly?
3. Can a candidate backend clear the full layer6 o-proj vector without
   collateral mismatches?
4. Does the same fused-linear/addmm interpretation explain layer4/layer5
   reverse o-proj evidence?
5. Can any future candidate stay disabled-by-default and avoid production
   routing, CUDA kernel, or Torch runtime dependency changes?

## Proposed Evidence Inputs

Layer6 primary inputs:

- Weighted-V artifact from the layer6 ordered attention bundle, or weighted-V
  recomputed from audit all-token V.
- Layer6 o-proj weight.
- Layer6 o-proj bias.
- Official layer6 o-proj artifact.
- Producer dtype probe status:
  `/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json`.
- Producer API probe status:
  `/tmp/layer6_attention_oproj_api_probe_status.json`.

Optional comparison surfaces:

- Layer0 o-proj, where chunked-pairwise cleared.
- Layer4 o-proj, where reverse cleared.
- Layer5 o-proj, where reverse cleared after weighted-V policy.

## Proposed Candidate Backends

These are conceptual only:

- Fused-addmm-like validation helper, if one is explicitly designed.
- Existing validation helpers only if they can model fused bias and layout
  constraints.
- Current/reverse/pairwise/chunked Rust replays as negative guards, not as new
  blind sweeps.
- Producer-side PyTorch BF16 `F.linear`, source proof only, not a Rust
  dependency.

BF16-product remains evidence-only/rejected unless used only as a negative
guard.

## Required Metrics

For every backend, report:

- o-proj full-vector max_abs_diff.
- mean_abs_diff.
- mismatch count.
- first mismatch.
- worst mismatch.
- focus lane `22` value.
- residual recompute result.
- attention-to-MLP bridge result.
- collateral mismatch count.
- whether the full vector clears.
- whether the result is diagnostic-only.

## Required Status JSON Contract

Minimum fields:

```text
classification
validation_only
runtime_behavior_changed
production_routing_changed
cuda_kernels_changed
layer_index
operator
selected_backend
source_statuses
official_backend_reference
backend_results
full_vector_metrics
focus_lane_metrics
collateral_mismatches
output_emitted
ladder_continued
final_logit_claim
all_layer_claim
server_claim
context_length_claim
next_bounded_step
```

## Expected Future Classifications

- `official_linear_backend_discriminator_design_recorded`
- `layer6_oproj_official_linear_backend_discriminator_ready`
- `layer6_oproj_cublas_tensor_op_matches_official`
- `layer6_oproj_cublas_pedantic_matches_official`
- `layer6_oproj_no_validation_backend_matches_official`
- `layer6_oproj_backend_discriminator_collateral_mismatches`
- `layer6_oproj_backend_discriminator_blocked_by_schema`
- `layer6_oproj_backend_discriminator_execution_failed`

## Guardrails

- Default behavior must remain current.
- No production runtime routing.
- No default model-runner routing.
- No CUDA kernel replacement.
- No correction metadata.
- No tolerance pass.
- No layer output emission.
- No ladder continuation.
- No producer result may be turned into a Rust Torch runtime dependency.
- No raw `/tmp` or `.live` commits.
- No Torch runtime dependency in Rust.
- No final-logit, all-layer, server, or 4097-token claim.

## Proof Gates Before Code

1. Accepted design doc.
2. Exact command examples.
3. Schema for status JSON.
4. Source proof status for the layer6 producer dtype and API probes.
5. Clear list of candidate backends.
6. Existing helper availability check.
7. Full-vector collateral metrics.
8. No-default-routing guarantee.
9. Rollback story.
10. Performance/release gate if a backend becomes a candidate.

## Suggested Future Implementation Order

1. Status-schema-only mode.
2. Fused-linear/addmm schema design for layout and fused-bias fields.
3. Artifact/source loader validation against that schema.
4. Status contract update for producer dtype/API source proof.
5. Existing helper comparison only if mapped to fused-addmm semantics.
6. Optional layer4/layer5 producer API re-probe.
7. Matrix summary update.

This design does not authorize implementation by itself.

## Recommendation

Recommended next action: create a docs-only ordered-surface batch
orchestration design, and separately create a docs-only fused-linear/addmm
validation discriminator design before any new Rust backend experiment.

Do not implement fused addmm semantics yet, select a runtime backend, generate
layer7 unless the goal is pure evidence collection, or emit/promote layer6
output.
