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

## Layer1 Attention Norm Validation Status

- Input source:
  `/tmp/layer0_validation_corrected_layer0_output.json`
- Input boundary:
  `corrected_layer0_output_layer1_input`
- Layer1 input guard: exact
- Mode: `--mode layer1-attn-norm`
- Official oracle:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer1-final-token-attention-norm-output-before-qkv-status.json`
- Norm tensor:
  `model.layers.1.input_layernorm.weight`
- Norm policy:
  `bf16_input_f32_reduction_bf16_output`
- Epsilon: `1e-5`

Metric:

| Boundary | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| layer1 attention norm before Q/K/V | `0.0` | `0.0` | `0` |

Classification:

```text
layer1_attention_norm_matches_oracle
```

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: validate the layer1 Q/K/V projection boundary or layer1 K
RoPE guard.

## Layer1 Full-Layer Validation Status

- Mode: `--mode full-layer --layer-index 1`
- Input source:
  `/tmp/layer0_validation_corrected_layer0_output.json`
- Layer1 input guard: exact
- Layer1 attention norm: exact
- Layer1 output oracle:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer1-final-token-mlp-ordered-boundary-bundle-status.json`
- Oracle boundary:
  `layer1_final_token_hidden_state_after_mlp_residual_add`
- Correction policy: none

Backend path:

```text
attention = blocked_before_full_prompt_qkv_attention_state
MLP1 = cublas_bf16_tensor_op
SwiGLU = pinned_torch_like_bf16_stage_rounding
MLP2 = bf16_prebias_output_policy
```

Classification:

```text
layer1_full_layer_blocked_by_port_scope
```

The blocker is deliberate: the exact corrected layer0 output is only the final
token layer1 input. Full layer1 attention also needs all-token layer1 K/V
history, or a validation helper that constructs layer-indexed Q/K/V for the
full prompt without importing runtime-forward proof binaries or debug capture
plumbing.

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: localize the layer1 attention path next, starting with Q/K/V
projection and K/V history construction.

## Layer1 QKV History Validation Status

Full-layer1 is blocked before full-prompt Q/K/V attention state construction:
the exact final-token layer1 input does not contain all-token K/V history.

Artifact search found:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer1-final-token-attention-ordered-boundary-bundle-status.json
```

Useful boundaries in the bundle:

- `layer1_final_token_q_projection_output_before_rope`
- `layer1_final_token_k_projection_output_before_rope`
- `layer1_final_token_v_projection_output_before_attention`
- `layer1_final_token_q_post_rope_before_attention`
- `layer1_grouped_k_post_rope_before_attention`
- raw QK, masked logits, attention probabilities, weighted V, o-proj, and
  attention residual seams

Mode added:

```text
--mode layer1-k-rope
```

Classification:

```text
layer1_k_rope_blocked_by_artifacts
```

The missing artifact is all-token layer1 K pre-RoPE history
`[74,8,64]`/`[74,512]`. The available bundle exposes final-token K pre-RoPE and
all-token grouped K post-RoPE, which is not sufficient to validate K RoPE
history construction.

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: generate or locate all-token layer1 residual/norm/QKV
artifacts. If K RoPE clears afterward, build raw-QK and attention-probability
guards.

## Validation-Only Non-Goals

- No production runtime routing
- No default model-runner behavior changes
- No CUDA kernel changes
- No Torch runtime dependency
- No raw `.live` or `/tmp` artifacts committed
- No final-logit claim
- No 4097 work
