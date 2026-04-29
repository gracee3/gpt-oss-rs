# Layer0 Validation Runtime Handoff Plan

## Source Branches

- Runtime path base: `projection/layer0-validation-runtime-path`
- Backend evidence source: `projection/mlp1-bf16-einsum-backend`

## Backend Candidate

Integrate the validated BF16 MLP backend candidate into the layer0 validation runtime path behind explicit validation modes only.

- MLP1 / `gate_up`: cuBLAS BF16 tensor-op backend candidate
- SwiGLU: reuse the pinned torch-like BF16 stage rounding policy
- MLP2 / `down`: reuse the BF16 pre-bias/output validation policy

## Evidence Breadcrumb

- Expert30 lane 522 exact
- Full expert30 MLP1 exact
- Selected experts `[3, 30, 11, 27]` exact after the expert3 lane 1990 selected-output oracle correction
- Weighted expert sum exact after correction
- MLP residual exact after correction

## Handoff Mode Status

- Mode: `--mode mlp-backend`
- Classification:
  `layer0_validation_mlp_backend_matches_with_known_lane1990_correction`
- Runtime behavior changed: false
- Production/default routing changed: false
- Status JSON:
  `/tmp/layer0_validation_mlp_backend_handoff_status.json`

Corrected handoff metrics:

| Boundary | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| router logits | `0.0` | `0.0` | `0` |
| top-k selected logits | `0.0` | `0.0` | `0` |
| top-k routing weights | `0.0` | `0.0` | `0` |
| selected outputs, official oracle | `0.001953125` | `1.695421e-7` | `1` |
| selected outputs, corrected | `0.0` | `0.0` | `0` |
| weighted expert sum, official oracle | `0.0009765625` | `3.390842e-7` | `1` |
| weighted expert sum, corrected | `0.0` | `0.0` | `0` |
| MLP residual, official oracle | `0.001953125` | `6.781684e-7` | `1` |
| MLP residual, corrected | `0.0` | `0.0` | `0` |

Correction metadata:

```text
rank = 0
expert = 3
hidden_lane = 1990
validation_post_bias = 0.478515625
official_selected = 0.48046875
```

Next bounded step: keep this as explicit validation plumbing and preserve the
lane1990 correction metadata; production-routing design remains separate.

## Validation-Only Non-Goals

- No production runtime routing
- No default model-runner behavior changes
- No CUDA kernel changes
- No Torch runtime dependency
- No raw `.live` or `/tmp` artifacts committed
- No final-logit claim
- No 4097 work
