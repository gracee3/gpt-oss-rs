# Fused Linear/AddMM-Like Helper Implementation Design

Classification: `fused_linear_addmm_like_helper_implementation_design_recorded`

## Scope

- Docs-only implementation design.
- Future validation-only helper work for attention o-proj.
- Source status: `/tmp/fused_linear_addmm_helper_candidate_status.json`.
- Source classification:
  `fused_linear_addmm_helper_candidate_no_candidate_selected`.
- Sampled layers: 6, 10, 13, 16, 18, and 21.
- No Rust implementation is authorized by this document.
- No backend is selected.
- No consumer revalidation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission or ladder continuation is authorized.

## Source Result

The helper-candidate run evaluated the sampled fused-linear/addmm reference set
for layers 6/10/13/16/18/21.

Result:

- No helper cleared the full sampled set.
- `pairwise_f32_bf16_output` clears only layer10 and layer21.
- `cublas_bf16_pedantic_or_deterministic` clears only layer16 and is the best
  partial by total mismatch count.
- `cublas_bf16_tensor_op` executed but has broad collateral mismatches.
- `bf16_product_evidence_guard` remains unavailable at the required validation
  abstraction.
- Backend selected: false.
- Consumer revalidation authorized: false.
- Runtime/default/CUDA behavior changed: false.

## Why Existing Candidate Space Is Exhausted

The existing local helper space has now covered:

- current sequential f32 accumulation with BF16 output
- reverse f32 accumulation with BF16 output
- pairwise f32 accumulation with BF16 output
- chunked pairwise f32 accumulation with BF16 output
- f64 diagnostic accumulation
- BF16 prebias evidence guard
- cuBLAS BF16 tensor-op helper
- cuBLAS BF16 pedantic/deterministic helper

None is a selectable backend candidate because none gives full-vector exactness
with `max_abs_diff = 0` on every sampled layer. Pairwise remains useful local
evidence, but clearing layer10 and layer21 while failing layer6/13/16/18 proves
it is not the producer backend identity. The pedantic cuBLAS path is closer by
total mismatch count, but a one-layer clear is still collateral evidence, not a
full sampled-set proof.

Another accumulation-order sweep would mostly reshuffle adjacent BF16 boundary
misses. The next useful candidate must model the producer/API reference more
directly.

## Official Reference Contract

Any future fused-addmm-like validation helper must target the producer/API
reference already observed for sampled o-proj layers:

- Operator: attention o-proj BF16 linear with bias.
- Reference API class: module/F.linear/_C/addmm.
- Input dtype: BF16 weighted-V.
- Weight dtype: BF16 o-proj weight.
- Bias dtype: BF16 o-proj bias.
- Bias behavior: fused bias.
- Layout: original shape, stride, and contiguity are semantically relevant.
- Output boundary: BF16.
- Required comparison: full-vector exactness.
- Focus-lane metrics: diagnostics only.

Explicit matmul/einsum/unfused-bias paths remain negative controls, not
official references.

## Plausible Implementation Strategies

These are future implementation options only; this document does not approve
any code.

### Strategy A - cuBLASLt Matmul With Bias Epilogue

If the repo/toolchain exposes cuBLASLt at the right abstraction level, a future
validation-only helper could compare a BF16 matmul with fused bias epilogue
against the producer/API reference.

Required metadata:

- cuBLASLt algorithm and math mode
- input/weight/bias/output types
- epilogue type and bias pointer dtype
- transpose/layout descriptors
- leading dimensions and strides
- determinism/atomics settings where applicable

This should be a validation helper only. It must not route into production or
default model-runner paths.

### Strategy B - Custom Validation-Only CUDA Fused Linear+Bias Kernel

A narrow CUDA validation kernel could explicitly model the candidate boundary:
BF16 input, BF16 weight, fused BF16 bias, and BF16 output. This is only useful
if its accumulation/output behavior is designed to test the fused-addmm-like
hypothesis rather than repeat current local reductions under another name.

Required metadata:

- accumulation type and order
- product type
- bias fusion point
- output cast point
- block/tile shape
- deterministic behavior
- full-vector mismatch metrics

This must remain validation-only and must not change existing CUDA kernels used
by production/default runtime.

### Strategy C - CPU/Rust Exact Reference Helper

A CPU/Rust helper is acceptable only if it can plausibly reproduce the
producer/API boundary. A high-precision CPU reference by itself is not enough,
because f64 diagnostics already demonstrate that exact mathematical reduction
is not the official backend identity.

It would need to model BF16 input/weight/bias, fused bias, output cast, and any
layout-sensitive rounding behavior with enough fidelity to explain the sampled
set.

### Strategy D - Producer/API Seam Reuse

Producer/API seams can remain the oracle reference for validation. They are not
a Rust backend. Reusing these seams is useful for proof and status generation,
but it does not implement runtime parity and must not be represented as a
production helper.

## Explicit Rejections

Do not use this design to approve:

- another current/reverse/pairwise/chunked accumulation sweep
- focus-lane-only clears
- f64 diagnostic promotion
- BF16-product evidence promotion
- BF16-prebias evidence promotion
- Torch runtime dependency in Rust
- production/default runtime routing changes
- consumer full-bundle revalidation
- output emission or ladder continuation

## Future Proof Gates

Before any helper can be treated as a follow-up candidate, require:

1. Full-vector exactness on all sampled layers: 6, 10, 13, 16, 18, and 21.
2. Zero mismatches and `max_abs_diff = 0` on every sampled layer.
3. No collateral mismatches.
4. Focus-lane metrics recorded as secondary diagnostics only.
5. Input metadata: shape, dtype, stride, and contiguity.
6. Weight metadata: shape, dtype, stride, and contiguity.
7. Bias metadata: shape, dtype, stride, contiguity, and fused-bias point.
8. Math metadata: helper family, algorithm, math mode, accumulation/output
   type, and determinism/atomics settings where relevant.
9. Negative/control preservation:
   - layer6 historical blocker/context
   - layer10 pairwise-clear control
   - layer13/16/18 blocked-family
   - layer21 raw-QK-solved / o-proj-blocked
10. Diagnostic/evidence-only candidates remain non-selectable.
11. No backend selection unless all proof gates pass.
12. No runtime/default/CUDA behavior change.

## Future Status Contract

A future prototype should emit a status like:

```json
{
  "classification": "fused_linear_addmm_like_helper_candidate_recorded",
  "validation_only": true,
  "candidate_execution": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "operator": "attention_o_proj",
  "reference": {
    "api": "module/F.linear/_C/addmm",
    "dtype": "torch.bfloat16",
    "fused_bias": true,
    "layout_sensitive": true,
    "full_vector_required": true,
    "focus_lane_only_accepted": false
  },
  "layers_requested": [6, 10, 13, 16, 18, 21],
  "candidate_helpers": [],
  "candidate_matrix": [],
  "selected_backend": null,
  "backend_selected": false,
  "implementation_authorized": false,
  "consumer_revalidation_authorized": false,
  "output_emitted": false,
  "ladder_continued": false,
  "correction_metadata_applied": false,
  "tolerance_pass": false,
  "final_logit_claim": false,
  "all_layer_claim": false,
  "server_claim": false,
  "context_length_claim": false
}
```

Allowed future classifications:

- `fused_linear_addmm_like_helper_candidate_recorded`
- `fused_linear_addmm_like_helper_candidate_full_sampled_set_clear`
- `fused_linear_addmm_like_helper_candidate_no_candidate_selected`
- `fused_linear_addmm_like_helper_candidate_blocked_by_missing_reference`
- `fused_linear_addmm_like_helper_candidate_execution_failed`

## Proposed Next Branch

Only if explicitly approved:

```text
validation/fused-linear-addmm-like-helper-prototype
```

Prototype scope:

- validation-only helper prototype
- no consumer revalidation
- no backend selection unless the full sampled set clears
- no production routing
- no default model-runner behavior changes
- no output emission
- no ladder continuation
- no correction metadata or tolerance pass
- no final-logit/all-layer/server/4097 claim

## Recommended Next Step

Prefer a small design review before implementation, specifically choosing
between cuBLASLt fused-bias epilogue and a custom validation-only CUDA helper.
The prototype should start with the minimum sampled set and fail closed if
layout, bias-fusion, or output-type metadata cannot be recorded.

## Guardrails

- Validation-only.
- No implementation authorization from this document.
- No backend selected.
- No production runtime routing.
- No default model-runner behavior changes.
- No CUDA kernel behavior change.
- No Torch runtime dependency in Rust.
- No consumer revalidation.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
