# Ordered Surface Batch Matrix 7..15

Classification:

```text
ordered_surface_batch_matrix_7_15_recorded
```

This docs-only matrix consolidates the ordered-surface batch pivot for layers
7..15 so the future 16..23 batch can be compared against a stable taxonomy. It
does not authorize runtime/default routing/CUDA changes, output emission, layer
ladder continuation, correction metadata, tolerance passes, or final-logit,
all-layer, server, or 4097-token claims.

## Source Statuses

```text
/tmp/ordered_surface_batch_consumer_status.json
/tmp/ordered_surface_batch_probe_status.json
/tmp/ordered_surface_batch_generation_10_15_status.json
/tmp/ordered_surface_batch_consumer_10_15_status.json
/tmp/ordered_surface_batch_probe_10_15_status.json
```

## Matrix

| Layer | Strict/default result | First failing seam | Probe result | Selected validation-only policy | Status |
| --- | --- | --- | --- | --- | --- |
| 7 | failed | raw-QK q_head 50 / key col 57 | dtype probe artifact precision; sweep collateral mismatches | none | blocked |
| 8 | failed | o-proj lane 2578 | full-vector clear | `pairwise_f32_accum_f32_bias_bf16_output` | explicit-policy cleared |
| 9 | failed | o-proj lane 446 | full-vector clear | `pairwise_f32_accum_f32_bias_bf16_output` | explicit-policy cleared |
| 10 | failed | o-proj lane 915 | full-vector clear | `pairwise_f32_accum_f32_bias_bf16_output` | explicit-policy cleared |
| 11 | failed | selected MLP down lane 1480 | replay full MLP clear; no bundle revalidation flag | `naive_f64_sum_then_bf16_output` replay evidence | revalidation support missing |
| 12 | cleared | none | not needed | none | strict/default cleared |
| 13 | failed | o-proj lane 151 | no non-diagnostic full-vector clear | none | blocked |
| 14 | cleared | none | not needed | none | strict/default cleared |
| 15 | cleared | none | not needed | none | strict/default cleared |

## Taxonomy

1. Strict/default clear: layers 12, 14, and 15.
2. o-proj pairwise clear: layers 8, 9, and 10.
3. o-proj bounded-family blocked: layers 6 and 13.
   - Layer6 has producer API / fused-addmm evidence.
   - Layer13 has consumer-side no-candidate result only so far.
4. raw-QK artifact/source boundary: layer7.
5. selected-MLP-down replay clears but bundle-policy revalidation support is
   missing: layer11.

## Next Actions

- Keep 16..23 oracle generation running.
- After 16..23 artifacts land, run classify-only consumer validation before
  probing.
- Do not probe layer13 or layer11 further until the 16..23 classify map is
  available unless blocked-layer design is specifically prioritized.
- Candidate future support lane: selected-MLP-down bundle revalidation flag,
  but do not implement it from this matrix.
- Candidate future producer/API lane: layer13 o-proj API probe only if 16..23
  shows a similar blocked o-proj class.

## Guardrails

- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No runtime/default/CUDA behavior changes.
- No final-logit/all-layer/server/4097 claim.
