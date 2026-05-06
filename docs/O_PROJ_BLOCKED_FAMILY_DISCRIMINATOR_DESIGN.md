# O-Proj Blocked-Family Discriminator Design

Classification:

```text
o_proj_blocked_family_discriminator_design_recorded
```

## Scope

- Docs-only design.
- Final-token ordered-surface evidence only.
- Prompt/case: `developer-message-user-smoke`.
- Target operator: attention o-proj.
- Target blocked layers: 13, 16, 18, and 21.
- Historical context: layer6 fused-linear/addmm producer/API evidence.
- Contrast cleared layers: 8, 9, 10, 20, plus older 4/5.
- No implementation authorized.
- No runtime/default/CUDA behavior change.
- No output emission or ladder continuation.

## Source Evidence

Matrix/backlog docs:

```text
docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md
docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md
docs/FUSED_LINEAR_ADDMM_DISCRIMINATOR_DESIGN.md
docs/OFFICIAL_LINEAR_BACKEND_DISCRIMINATOR_DESIGN.md
```

Batch status paths:

```text
/tmp/ordered_surface_batch_probe_status.json
/tmp/ordered_surface_batch_probe_10_15_status.json
/tmp/ordered_surface_batch_probe_16_23_oproj_status.json
/tmp/ordered_surface_batch_probe_17_21_23_raw_qk_status.json
```

Layer-specific status paths:

```text
layer8:
  /tmp/layer8_attention_oproj_policy_sweep_status.json
  /tmp/layer8_ordered_bundle_validate_oproj_policy_status.json

layer9:
  /tmp/layer9_attention_oproj_policy_sweep_status.json
  /tmp/layer9_ordered_bundle_validate_oproj_policy_status.json

layer10:
  /tmp/layer10_attention_oproj_policy_sweep_status.json
  /tmp/layer10_ordered_bundle_validate_oproj_policy_status.json

layer13:
  /tmp/layer13_attention_oproj_policy_sweep_status.json

layer16:
  /tmp/layer16_attention_oproj_policy_sweep_status.json

layer18:
  /tmp/layer18_attention_oproj_policy_sweep_status.json

layer20:
  /tmp/layer20_attention_oproj_policy_sweep_status.json
  /tmp/layer20_ordered_bundle_validate_oproj_policy_status.json

layer21:
  /tmp/layer21_ordered_bundle_validate_raw_qk_policy_status.json
  note: raw-QK reverse cleared first, then o-proj blocked

layer6 historical:
  /tmp/layer6_attention_oproj_policy_sweep_status.json
  /tmp/layer6_official_linear_backend_discriminator_probe_status.json
  /tmp/layer6_attention_oproj_api_probe_status.json
```

## Evidence Matrix

| Layer | Upstream status | Strict/default o-proj | Sweep outcome | Full bundle revalidation | Classification |
| --- | --- | --- | --- | --- | --- |
| 4 | raw-QK/weighted-V exact | failed | reverse clears | full bundle cleared | reverse-clear contrast |
| 5 | weighted-V pairwise exact | failed | reverse clears | full bundle cleared | reverse-clear contrast |
| 6 | upstream attention exact | failed | bounded family collateral | not cleared | fused-linear/addmm historical blocker |
| 8 | upstream attention exact | failed | pairwise clears | full bundle cleared | pairwise-clear contrast |
| 9 | upstream attention exact | failed | pairwise clears | full bundle cleared | pairwise-clear contrast |
| 10 | upstream attention exact | failed | pairwise clears | full bundle cleared | pairwise-clear contrast |
| 13 | upstream attention exact | failed | no full-vector clear | not cleared | blocked-family |
| 16 | upstream attention exact | failed | no full-vector clear | not cleared | blocked-family |
| 18 | upstream attention exact | failed | no full-vector clear | not cleared | blocked-family |
| 20 | upstream attention exact | failed | pairwise clears o-proj | revalidation stops later at selected MLP down | pairwise seam-cleared, full-bundle partial |
| 21 | raw-QK reverse exact before revalidation | o-proj exposed after raw-QK clear | not yet swept in o-proj lane | not cleared | blocked-family candidate |

## Key Questions

1. Are blocked layers 13/16/18/21 closer to layer6 fused-linear/addmm
   semantics than to the local bounded sweep family?
2. Are pairwise-clearing layers 8/9/10/20 explained by the same
   fused-linear/addmm API path, or are they coincidental local-policy matches?
3. Are reverse-clearing layers 4/5 a separate class or another coincidental
   match to official fused-linear behavior?
4. Does input layout, fused bias, contiguous weighted-V, or row/column
   orientation explain blocked vs clearing layers?
5. Does the official producer/API behavior differ by layer or only by tensor
   values crossing BF16 rounding boundaries?
6. What minimal producer/API probes would distinguish these classes without
   running another broad Rust backend sweep?

## Hypotheses

### Hypothesis A - Fused-linear/addmm official boundary

Layer6 producer evidence showed module/F.linear/_C._nn.linear/fused addmm
clear while explicit matmul/einsum and unfused bias reproduce broad
mismatches. Blocked layers 13/16/18/21 may require the same official
API/fused-bias semantics.

### Hypothesis B - Local accumulation policy coincidence

Pairwise or reverse policies may clear some layers only because their numeric
rounding happens to match official BF16 output for those tensor values. They
should not be treated as actual backend semantics.

### Hypothesis C - Source/layout sensitivity

Blocked layers may differ in weighted-V layout, contiguity, stride, or
fused-bias treatment. Layer6 producer API probe showed layout/fused-bias
sensitivity, so layer13/16/18/21 need source/layout metadata before backend
conclusions.

### Hypothesis D - Multiple o-proj subclasses

O-proj may split into:

- Strict/default exact.
- Reverse-clear.
- Pairwise-clear.
- Fused-linear/addmm required.
- Unresolved source/artifact precision boundary.

## Proposed Discriminator Design

This is conceptual only. The discriminator should compare, per target layer:

1. Source metadata:
   - weighted-V dtype, shape, stride, contiguity
   - o-proj weight dtype, shape, stride, contiguity
   - o-proj bias dtype, shape, stride, contiguity
   - official o-proj output dtype/shape
   - source status paths
   - full-vector digests
2. Producer/API variants:
   - module attention out projection if accessible
   - torch.nn.functional.linear
   - torch._C._nn.linear
   - fused torch.addmm
   - explicit weight @ input + bias
   - input @ weight.T + bias
   - torch.matmul
   - torch.einsum
   - F.linear(..., bias=None) + bias
   - layout/stride perturbation guard if safe
3. Consumer/Rust variants:
   - current sequential
   - pairwise
   - reverse
   - chunked pairwise
   - f64 diagnostic
   - BF16 prebias/bias variants
   - BF16-product evidence guard
4. Full-vector classification:
   - full-vector cleared
   - focus-lane only
   - collateral mismatches
   - no candidate clears
   - producer/API clears but local variants do not
   - source/layout mismatch

## Target Layers for Future Producer/API Probe

Recommend a minimal target set, not broad all-layer probing:

- Layer13: blocked after earlier 10..15 probe.
- Layer16: blocked after 16..23 o-proj probe.
- Layer18: blocked after 16..23 o-proj probe.
- Layer21: only after raw-QK reverse revalidation exposes o-proj.
- Layer8 or layer10 as pairwise-clearing control.
- Layer6 as historical comparison only; do not rerun unless needed.

Recommended first producer/API probe set:

1. Layer13 blocked case.
2. Layer16 blocked case.
3. Layer10 pairwise-clear control.

Reason: this distinguishes blocked vs pairwise-clear classes with minimal
scope before touching layer21 or 18.

## Status JSON Contract

Future status shape:

```json
{
  "classification": "o_proj_blocked_family_discriminator_ready",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "layers_requested": [13, 16, 10],
  "source_matrix": "docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md",
  "layer_results": [
    {
      "layer_index": 13,
      "class": "blocked_family",
      "focus_lane": 151,
      "producer_api_probe_status": null,
      "consumer_sweep_status": "/tmp/layer13_attention_oproj_policy_sweep_status.json",
      "source_metadata": {},
      "api_variants": {},
      "consumer_variants": {},
      "selected_backend": null,
      "full_vector_cleared": false,
      "collateral_mismatches": true,
      "recommended_next_step": "producer_api_probe"
    }
  ],
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

## Classification Vocabulary

Design/status classifications:

```text
o_proj_blocked_family_discriminator_design_recorded
o_proj_blocked_family_discriminator_ready
o_proj_blocked_family_discriminator_blocked_by_missing_status
o_proj_blocked_family_discriminator_producer_api_recommended
o_proj_blocked_family_discriminator_execution_failed
```

Future per-layer probe classifications:

```text
layerN_oproj_producer_api_probe_fused_linear_clears
layerN_oproj_producer_api_probe_layout_sensitive
layerN_oproj_producer_api_probe_unfused_bias_mismatch
layerN_oproj_producer_api_probe_matmul_einsum_mismatch
layerN_oproj_producer_api_probe_no_api_variant_clears
layerN_oproj_producer_api_probe_blocked_by_source_access
```

## Proof Gates Before Implementation

1. At least one blocked layer producer/API probe.
2. At least one pairwise-clearing control producer/API probe.
3. Full-vector comparison, not focus-lane.
4. Layout/fused-bias/source metadata.
5. No collateral mismatch policy promotion.
6. No default runtime change.
7. No CUDA kernel change.
8. Explicit validation-only status contract.
9. Negative controls preserved: strict/default cleared layers 12/14/15/22.

## Recommended Immediate Next Step

Run a producer/API probe design follow-up for layer13 and layer16, with layer10
as a pairwise-clear control.

Do not implement it in this branch.

Suggested next branch:

```text
oracle/o-proj-producer-api-probes-13-16-10
```

Only after the user separately authorizes probe execution.

## Non-Goals

- No runtime implementation.
- No default routing change.
- No CUDA kernel change.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
