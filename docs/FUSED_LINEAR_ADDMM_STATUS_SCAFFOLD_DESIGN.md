# Fused Linear/AddMM Status Scaffold Design

Classification:

```text
fused_linear_addmm_status_scaffold_design_recorded
```

## Scope

- Docs-only design.
- Future validation-only scaffold.
- Target operator: attention o-proj BF16 fused linear/addmm.
- Source evidence: layers 13, 16, 10, and historical layer6.
- No implementation authorized.
- No backend selected.
- No runtime/default/CUDA changes.
- No output emission.
- No ladder continuation.

## Source Evidence

Reference docs:

```text
docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md
docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md
docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md
docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md
```

Reference statuses:

```text
/tmp/o_proj_producer_api_probes_13_16_10_status.json
/tmp/layer13_attention_oproj_api_probe_status.json
/tmp/layer16_attention_oproj_api_probe_status.json
/tmp/layer10_attention_oproj_api_probe_status.json
/tmp/layer6_attention_oproj_api_probe_status.json
/tmp/layer6_official_linear_backend_discriminator_probe_status.json
```

## Scaffold Purpose

The scaffold should normalize existing producer/API evidence into one status
shape. It should answer:

- Which layers have official producer/API fused-linear/addmm reference
  evidence?
- Which variants clear full-vector?
- Which variants fail due matmul/einsum/unfused-bias mismatch?
- Which layers are layout-sensitive?
- Which layers are fused-bias-sensitive?
- Which local policies cleared coincidentally but are not backend identity?
- Which layers remain blocked for consumer-side validation?

The scaffold should not:

- Run new probes.
- Run local backend sweeps.
- Select a runtime backend.
- Promote pairwise/reverse/current policies.
- Emit layer outputs.

## Input Contract

Future mode:

```text
--mode fused-linear-addmm-status-scaffold
```

Future inputs:

```text
--producer-api-probe-status /tmp/o_proj_producer_api_probes_13_16_10_status.json
--layer6-api-probe-status /tmp/layer6_attention_oproj_api_probe_status.json
--layer6-backend-discriminator-status /tmp/layer6_official_linear_backend_discriminator_probe_status.json
--output /tmp/fused_linear_addmm_status_scaffold.json
```

Required input behavior:

- Fail closed if a required source status is missing.
- Record missing statuses explicitly.
- Parse only JSON metadata and metrics.
- Do not load raw tensors.
- Do not execute model code.
- Do not run CUDA.
- Do not import Torch.

## Output Status Contract

Future status JSON:

```json
{
  "classification": "fused_linear_addmm_status_scaffold_recorded",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "operator": "attention_o_proj",
  "source_statuses": {
    "producer_api_batch": "/tmp/o_proj_producer_api_probes_13_16_10_status.json",
    "layer6_api_probe": "/tmp/layer6_attention_oproj_api_probe_status.json",
    "layer6_backend_discriminator": "/tmp/layer6_official_linear_backend_discriminator_probe_status.json"
  },
  "layers": [
    {
      "layer_index": 13,
      "class": "blocked_family",
      "official_reference": "module/F.linear/_C/addmm",
      "fused_bias_sensitive": true,
      "layout_sensitive": true,
      "producer_api_full_vector_cleared": true,
      "matmul_einsum_unfused_bias_mismatches": 819,
      "local_policy_identity_proven": false,
      "runtime_backend_selected": null
    }
  ],
  "summary": {
    "official_reference_pattern": "fused_linear_addmm_bf16_fused_bias_original_layout",
    "producer_api_reference_available_layers": [13, 16, 10],
    "historical_context_layers": [6],
    "pairwise_local_clearing_is_backend_identity": false,
    "backend_selected": false,
    "implementation_authorized": false
  },
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

Future status classifications:

```text
fused_linear_addmm_status_scaffold_recorded
fused_linear_addmm_status_scaffold_partial
fused_linear_addmm_status_scaffold_blocked_by_missing_status
fused_linear_addmm_status_scaffold_execution_failed
```

Per-layer classifications:

```text
layerN_fused_linear_addmm_reference_available
layerN_fused_linear_addmm_reference_missing
layerN_fused_linear_addmm_reference_layout_sensitive
layerN_fused_linear_addmm_reference_unfused_bias_sensitive
layerN_fused_linear_addmm_reference_backend_identity_unresolved
```

## Decision Rules

- Full-vector producer/API clear is required.
- Focus-lane-only clears are not enough.
- Pairwise/reverse/current local policies are not backend identity.
- Matmul/einsum/unfused-bias are negative controls, not official references.
- Fused-bias and original layout metadata are required.
- No backend may be selected from this scaffold.
- No runtime/default/CUDA changes may be authorized by this scaffold.

## Negative Controls

Preserve:

- Strict/default cleared layers: 12, 14, 15, 22.
- Pairwise-clear local controls: 8, 9, 10.
- Blocked-family examples: 13, 16, 18, 21.
- Historical layer6 fused-linear/addmm evidence.

## Proposed Future Implementation Slice

Only describe; do not authorize.

Future branch if explicitly approved:

```text
validation/fused-linear-addmm-status-scaffold
```

Scope:

- Add status-only mode.
- Read existing JSON statuses.
- Emit normalized scaffold JSON.
- No model execution.
- No CUDA execution.
- No backend selection.
- No consumer revalidation.
- No output emission.

Validation for future implementation:

```text
cargo fmt --package gpt-oss-bench
cargo check -p gpt-oss-bench --features cuda
status mode run
jq guard
git diff --check
```

## Recommended Next Step

Option A: authorize `validation/fused-linear-addmm-status-scaffold` as a
status-only code slice.

Option B: switch to Workstream B docs-only selected-MLP-down bundle
revalidation design.

Preference:

- If the goal is to keep Workstream A tight, do Option A next.
- If the goal is to avoid any code scaffolding until implementation is fully
  selected, do Option B.

## Non-Goals

- No runtime implementation.
- No backend selection.
- No default routing change.
- No CUDA kernel change.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency.
