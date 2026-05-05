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

## Narrower Official Linear Backend Discriminator

The next discriminator should be producer/API/source-focused, not another
Rust backend sweep. It should answer:

1. What exact PyTorch operator path does `attn.out(weighted_flat)` use?
2. Is `torch.nn.functional.linear` dispatching to a different backend than
   explicit matmul/einsum?
3. Does `torch._C._nn.linear` match `F.linear`?
4. Does `torch.addmm` match `F.linear`?
5. Do tensor strides, contiguity, transposition, or memory layout affect the
   result?
6. Does forcing contiguous input or weight change the result?
7. Does disabling or enabling MKLDNN or oneDNN change the CPU BF16 result?
8. Does thread count affect determinism?
9. Does the official module call use fused bias behavior that explicit
   matmul/einsum misses?
10. Can a producer-side operator-API matrix explain the 826 matmul/einsum
    mismatches?

## Proposed Stage 3 Producer-Side API Probe

Stage 3 should be an oracle/producer-side probe, not a Rust implementation.

Inputs:

- Layer6 weighted-V live tensor.
- Layer6 o-proj weight and bias.
- Official layer6 o-proj artifact.
- Producer dtype probe status:
  `/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json`.

Operator variants:

- Module call: `attn.out(weighted_flat)`.
- `torch.nn.functional.linear`.
- `torch._C._nn.linear`, if accessible.
- `torch.addmm`.
- `weight @ input + bias`.
- `input @ weight.T + bias`.
- `torch.matmul`.
- `torch.einsum`.
- Contiguous and non-contiguous variants.
- MKLDNN enabled/disabled if applicable.
- Single-thread versus default-thread CPU, if applicable.

Required metrics:

- Full-vector mismatches.
- max_abs_diff.
- mean_abs_diff.
- Focus lane `22`.
- First/worst mismatch.
- Tensor dtype, device, stride, and contiguity.
- Determinism over repeated runs.

Expected classifications:

- `layer6_oproj_producer_api_probe_flinear_unique`
- `layer6_oproj_producer_api_probe_addmm_matches`
- `layer6_oproj_producer_api_probe_layout_sensitive`
- `layer6_oproj_producer_api_probe_thread_sensitive`
- `layer6_oproj_producer_api_probe_mkldnn_sensitive`
- `layer6_oproj_producer_api_probe_no_operator_explains`
- `layer6_oproj_producer_api_probe_execution_failed`

## Discriminator Goal

The discriminator should answer:

1. Is official BF16 `F.linear` parity reproducible by an existing Rust/CUDA
   validation backend?
2. Does cuBLAS BF16 tensor-op reproduce the official layer6 o-proj artifact?
3. Does pedantic cuBLAS BF16 reproduce it?
4. Is the mismatch CPU-vs-GPU BF16 backend behavior?
5. Is the mismatch due to operator API choice: `F.linear` / module call vs
   matmul/einsum?
6. Is the issue full-vector or lane-local?
7. Which policy, if any, clears without collateral mismatches?

## Proposed Evidence Inputs

Layer6 primary inputs:

- Weighted-V artifact from the layer6 ordered attention bundle, or weighted-V
  recomputed from audit all-token V.
- Layer6 o-proj weight.
- Layer6 o-proj bias.
- Official layer6 o-proj artifact.
- Producer dtype probe status:
  `/tmp/layer6_attention_oproj_lane22_dtype_probe_status.json`.

Optional comparison surfaces:

- Layer0 o-proj, where chunked-pairwise cleared.
- Layer4 o-proj, where reverse cleared.
- Layer5 o-proj, where reverse cleared after weighted-V policy.

## Proposed Candidate Backends

These are conceptual only:

- Current sequential Rust f32 accumulation.
- Reverse f32 accumulation.
- Pairwise f32 accumulation.
- Chunked-pairwise f32 accumulation.
- Existing cuBLAS BF16 tensor-op validation helper.
- Existing cuBLAS pedantic BF16 validation helper, if available.
- CPU BF16 reference replay, diagnostic only.
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
4. Source proof status for the layer6 producer probe.
5. Clear list of candidate backends.
6. Existing helper availability check.
7. Full-vector collateral metrics.
8. No-default-routing guarantee.
9. Rollback story.
10. Performance/release gate if a backend becomes a candidate.

## Suggested Future Implementation Order

1. Status-schema-only mode.
2. Artifact/source loader validation.
3. Current/reverse/pairwise/chunked replay reuse.
4. cuBLAS BF16 tensor-op validation backend probe.
5. cuBLAS pedantic validation backend probe.
6. Optional layer0/layer4/layer5 comparison mode.
7. Matrix summary update.

This design does not authorize implementation by itself.

## Recommendation

Do not generate layer7 yet unless the goal is pure evidence collection. Do not
open a runtime implementation branch. The recommended next action is a
producer-side API probe in the oracle lane before any new Rust backend
experiment.
