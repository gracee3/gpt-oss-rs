# O-Proj Producer/API Final Matrix

Classification: `o_proj_producer_api_final_matrix_recorded`

## Scope

- Docs-only final matrix for attention o-proj producer/API evidence.
- Prompt/case: `developer-message-user-smoke`.
- Final-token ordered-surface evidence only.
- No implementation is authorized.
- No backend is selected.
- No consumer revalidation is authorized.
- No runtime/default/CUDA behavior change is authorized.
- No output emission, ladder continuation, correction metadata, tolerance pass,
  final-logit, all-layer, server, or 4097-token claim is authorized.

## Source Statuses

- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`
- `/tmp/fused_linear_addmm_status_scaffold.json`
- `/tmp/layer6_attention_oproj_api_probe_status.json`

Oracle evidence branch:

```text
oracle/o-proj-producer-api-probes-18-21
```

Oracle evidence commit:

```text
06fb602ffe6303e538d16dae8fb227ea69e3696a
```

## Matrix

| Layer | Role | Focus lane | module/F.linear/_C/addmm | matmul/einsum | unfused bias | Layout/fused-bias sensitive | Interpretation |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| 6 | historical blocker | 22 | full-vector clear | 826 mismatches class | 826 mismatches class | yes | original fused-linear/addmm pattern |
| 10 | pairwise-clear control | 915 | full-vector clear | 822 mismatches | 822 mismatches | yes | pairwise local clear is not backend identity |
| 13 | blocked-family | 151 | full-vector clear | 819 mismatches | 819 mismatches | yes | fused-linear/addmm pattern |
| 16 | blocked-family | 2666 | full-vector clear | 763 mismatches | 763 mismatches | yes | fused-linear/addmm pattern |
| 18 | blocked-family | 63 | full-vector clear | 764 mismatches | 765 mismatches | yes | fused-linear/addmm pattern |
| 21 | raw-QK-solved / o-proj-blocked | 2807 | full-vector clear | 757 mismatches | 757 mismatches | yes | fused-linear/addmm pattern |

## Interpretation

- Workstream A's sampled o-proj producer/API coverage is now coherent.
- Every sampled blocked/control o-proj layer matches the
  module/F.linear/_C/addmm fused-bias original-layout pattern.
- Explicit matmul/einsum/unfused-bias are negative controls, not official
  references.
- Pairwise/reverse/current local policies may clear or fail depending on tensor
  values, but they do not prove backend identity.
- Layer21 has correctly moved from Workstream C to Workstream A: raw-QK is
  solved for the prompt/case; the remaining blocker is o-proj.
- This does not select a Rust backend.
- This does not authorize consumer revalidation or runtime implementation.

## Recommended Next Decision

Options:

- A. Pause and preserve milestone taxonomy.
- B. Create a docs-only fused-linear/addmm backend-discriminator design update.
- C. Create a validation-only backend-discriminator/status plan, only if
  explicitly approved.

Recommended: Option A if preparing an operator milestone. Option B if preparing
for implementation design. Do not jump directly to runtime implementation.

Milestone summary:

```text
docs/ORDERED_SURFACE_BATCH_MILESTONE_SUMMARY.md
```

## Guardrails

- Validation-only.
- No implementation authorization.
- No production runtime routing.
- No default model-runner behavior changes.
- No CUDA kernel changes.
- No Torch runtime dependency.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
