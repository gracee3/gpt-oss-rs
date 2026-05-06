# Fused Linear/AddMM Helper Design

Classification: `fused_linear_addmm_helper_design_recorded`

## Scope

- Docs-only helper design.
- Validation-only future helper work.
- Target operator: attention o-proj BF16 linear with bias.
- Prompt/case: `developer-message-user-smoke`.
- Source status:
  `/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json`.
- Source classification:
  `fused_linear_addmm_backend_discriminator_no_candidate_selected`.
- No implementation is authorized.
- No backend is selected.
- No consumer revalidation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission or ladder continuation is authorized.

## Why Existing Helpers Are Exhausted

The backend candidate comparator executed the available local validation helper
families against the producer/API fused-linear/addmm references for the sampled
set:

- Layer6 historical blocker.
- Layer10 pairwise-clear control.
- Layers13/16/18 blocked-family.
- Layer21 raw-QK-solved / o-proj-blocked.

Executed helpers:

- `current_sequential_f32_bf16_output`
- `reverse_f32_bf16_output`
- `pairwise_f32_bf16_output`
- `chunked_pairwise_f32_bf16_output`
- `f64_diagnostic`
- `bf16_prebias_evidence_guard`

Unavailable helper families were recorded as unavailable, not failed:

- `bf16_product_evidence_guard`
- `cublas_bf16_tensor_op_if_available`
- `cublas_bf16_pedantic_if_available`

No selectable candidate cleared the full sampled set. Every existing
selectable local accumulation policy either failed a blocked-family layer,
failed the historical layer6 context, failed a control, or remained only a
partial local approximation. `f64_diagnostic` remains diagnostic-only.
`bf16_prebias_evidence_guard` remains evidence-only and has broad collateral
mismatches.

## Pairwise Remains Partial Evidence

`pairwise_f32_bf16_output` is the best partial candidate from the comparator:

| Layer | Role | Pairwise result |
| --- | --- | --- |
| 6 | historical blocker | 2 mismatches |
| 10 | pairwise-clear control | full-vector clear |
| 13 | blocked-family | 2 mismatches |
| 16 | blocked-family | 1 mismatch |
| 18 | blocked-family | 3 mismatches |
| 21 | raw-QK-solved / o-proj-blocked | full-vector clear |

This preserves the earlier interpretation: local pairwise clearing is useful
validation evidence, but it is not official backend identity. It cannot be
selected as a backend because it fails layer6/13/16/18 and therefore does not
meet full sampled-set exactness.

## Official Reference Contract

Future helper candidates must compare against the official producer/API
reference contract:

- Operator: attention o-proj BF16 linear with bias.
- Reference API class: module/F.linear/_C/addmm.
- Input: BF16 weighted-V.
- Weight: BF16 o-proj weight.
- Bias: BF16 o-proj bias.
- Bias behavior: fused bias, not unfused post-add as an official substitute.
- Layout: original layout/stride/contiguity is semantically relevant.
- Output boundary: BF16.
- Required comparison: full-vector exactness.
- Focus-lane metrics: secondary diagnostics only.

Explicit matmul/einsum/unfused-bias forms remain negative controls, not
official references.

## Minimum Sampled Set

Any future helper candidate run must cover at least:

| Layer | Role | Focus lane | Requirement |
| --- | --- | ---: | --- |
| 6 | historical blocker | 22 | preserve layer6 fused-linear/addmm context |
| 10 | pairwise-clear control | 915 | preserve pairwise-clear control |
| 13 | blocked-family | 151 | clear blocked-family layer |
| 16 | blocked-family | 2666 | clear blocked-family layer |
| 18 | blocked-family | 63 | clear blocked-family layer |
| 21 | raw-QK-solved / o-proj-blocked | 2807 | clear post-raw-QK o-proj blocker |

Optional negative controls can be added later, but they must not replace this
minimum set.

## Proposed Future Helper Families

Future implementation slices may compare these families, but this document
does not authorize implementation:

1. Existing local helpers as regression baselines:
   - current sequential f32 accumulation + BF16 output
   - reverse f32 accumulation + BF16 output
   - pairwise f32 accumulation + BF16 output
   - chunked pairwise f32 accumulation + BF16 output
   - f64 diagnostic
   - BF16 prebias/product evidence guards

2. Newly available cuBLAS BF16 probes, if they can be exposed
   validation-only:
   - BF16 tensor-op mode
   - BF16 pedantic or deterministic mode, if available
   - explicit record of math mode, accumulation mode, output type, and bias
     handling

3. A future fused-addmm-like validation helper, only if separately approved:
   - must model fused bias and original layout sensitivity
   - must remain validation-only
   - must not route into production/default runtime
   - must not add Torch runtime dependency in Rust

## Future Status JSON Contract

A future helper candidate run should emit:

```json
{
  "classification": "fused_linear_addmm_helper_candidate_recorded",
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
  "source_statuses": {
    "candidate_matrix": "/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json",
    "producer_api_13_16_10": "/tmp/o_proj_producer_api_probes_13_16_10_status.json",
    "producer_api_18_21": "/tmp/o_proj_producer_api_probes_18_21_status.json",
    "layer6_api_probe": "/tmp/layer6_attention_oproj_api_probe_status.json"
  },
  "layers_requested": [6, 10, 13, 16, 18, 21],
  "candidate_helpers": [
    {
      "candidate": "future_fused_addmm_like_validation_helper",
      "candidate_available": true,
      "candidate_executed": true,
      "diagnostic_only": false,
      "evidence_only": false,
      "layers": {
        "13": {
          "full_vector_mismatches": 0,
          "max_abs_diff": 0.0,
          "focus_lane_cleared": true,
          "full_vector_cleared": true,
          "collateral_mismatches": false
        }
      },
      "candidate_full_vector_cleared_sampled_set": false,
      "candidate_for_followup_design": false,
      "candidate_selected": false
    }
  ],
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

- `fused_linear_addmm_helper_candidate_recorded`
- `fused_linear_addmm_helper_candidate_full_sampled_set_clear`
- `fused_linear_addmm_helper_candidate_no_candidate_selected`
- `fused_linear_addmm_helper_candidate_blocked_by_missing_reference`
- `fused_linear_addmm_helper_candidate_execution_failed`

## Decision Rules

- Full sampled-set exactness is required.
- Full-vector mismatches must be zero on every sampled layer.
- `max_abs_diff` must be zero on every sampled layer.
- Focus-lane-only clears cannot be promoted.
- Collateral mismatch candidates cannot be promoted.
- Diagnostic-only candidates cannot be selected.
- Evidence-only candidates cannot be selected.
- Unavailable helper families must be recorded as unavailable, not failed.
- Pairwise/reverse/current local policies cannot be treated as official backend
  identity.
- No backend may be selected unless all proof gates pass.
- Even if a candidate clears the sampled set, it remains validation evidence
  until a separate promotion proof plan is accepted.

## Next Branch

Only if separately approved:

```text
validation/fused-linear-addmm-helper-candidate
```

Scope for that future branch:

- Add or expose validation-only helper candidates.
- Consume existing producer/API references.
- Run the minimum sampled set.
- Emit normalized helper candidate status JSON.
- Do not run consumer full-bundle revalidation.
- Do not select a production backend.
- Do not change runtime/default/CUDA behavior.

## Guardrails

- No runtime/default/CUDA behavior change.
- No consumer revalidation.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit claim.
- No all-layer claim.
- No server claim.
- No 4097-token claim.
- No Torch runtime dependency in Rust.
