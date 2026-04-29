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

## Full Layer0 Validation Status

- Mode: `--mode full-layer0`
- Attention source:
  `official_attention_residual_oracle_because_prior_mode_exact`
- Classification:
  `layer0_validation_full_layer0_matches_with_known_lane1990_correction`
- Runtime behavior changed: false
- Production/default routing changed: false
- Status JSON:
  `/tmp/layer0_validation_full_layer0_status.json`

Backend path:

```text
MLP1 = cublas_bf16_tensor_op
SwiGLU = pinned_torch_like_bf16_stage_rounding
MLP2 = bf16_prebias_output_policy
```

Corrected final layer0 metric:

| Boundary | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| MLP norm | `0.0` | `0.0` | `0` |
| router/top-k | `0.0` | `0.0` | `0` |
| selected outputs, corrected | `0.0` | `0.0` | `0` |
| weighted expert sum, corrected | `0.0` | `0.0` | `0` |
| final layer0 output, corrected | `0.0` | `0.0` | `0` |

Caveats:

- The first join mode accepts the official attention residual artifact because
  the prior attention-residual mode already proved that boundary exact.
- This remains layer0 final-token validation only.
- No all-layer, final-logit, 4097-token, or production/server parity claim is
  made.
- No raw `.live` or `/tmp` artifacts are committed.

Next bounded step: preserve this full-layer0 validation status and keep any
production-routing design separate.

## Layer Ladder Preparation Status

- Corrected layer0 output emission:
  `--emit-corrected-layer0-output /tmp/layer0_validation_corrected_layer0_output.json`
- Emission default: disabled unless the explicit argument is supplied
- Emitted boundary:
  `layer0_final_token_hidden_after_mlp_residual_corrected`
- Shape/dtype: `[2880]`, BF16 boundary values serialized as f32 JSON values
- Layer1 input guard mode: `--mode layer1-input-guard`

Layer1 input oracle path used:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-hidden-state-after-mlp-residual-add-status.json
```

This official layer0 final-token hidden-after-MLP-residual boundary is the
layer1 residual-stream input boundary for the final token.

Layer1 input guard result:

| Boundary | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| emitted corrected layer0 output vs layer1 input oracle | `0.0` | `0.0` | `0` |

Classification:

```text
layer1_input_guard_matches_oracle
```

No production behavior changed, no default routing changed, and no raw `.live`
or `/tmp` artifacts are committed.

Next bounded step: begin layer1 validation from this exact input boundary.

## Validation-Only Non-Goals

- No production runtime routing
- No default model-runner behavior changes
- No CUDA kernel changes
- No Torch runtime dependency
- No raw `.live` or `/tmp` artifacts committed
- No final-logit claim
- No 4097 work
