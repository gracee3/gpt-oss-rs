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

## Layer1 QKV History / K RoPE Validation Status

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
layer1_qkv_history_blocked_by_all_token_input_generation
```

The missing artifact is all-token layer1 K pre-RoPE history
`[74,8,64]`/`[74,512]`. The available bundle exposes final-token K pre-RoPE and
all-token grouped K post-RoPE, which is not sufficient to validate K RoPE
history construction.

Scratch generation was attempted under `/tmp` with an external
PyTorch/Transformers oracle generator. It did not produce a pack: loading the
restricted model dequantized MXFP4 weights to BF16 and exceeded the available
24 GB CUDA memory before layer1 all-token QKV capture. No scratch artifact is
committed.

K pre-RoPE source: missing all-token layer1 K pre-RoPE history.

K post-RoPE source: official pinned attention ordered bundle
`layer1_grouped_k_post_rope_before_attention`.

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: generate or locate an all-token layer1 residual/norm/QKV
pack with K pre-RoPE `[74,512]` or `[74,8,64]`, then rerun `layer1-k-rope`. If
K RoPE clears afterward, validate final-token raw-QK using final-token Q
post-RoPE and grouped K post-RoPE.

## Layer1 Attention Bundle Validation Status

This mode exists to move downstream layer1 attention validation forward while
the source-complete K/V history task remains open. It starts from explicit seam
inputs already present in the official layer1 attention ordered bundle instead
of claiming K RoPE construction has been validated.

Mode:

```text
--mode layer1-attention-bundle
```

Source policy:

- grouped K post-RoPE source: official attention bundle
- Q post-RoPE source: official attention bundle
- K pre-RoPE history: still missing
- K RoPE construction validated: false
- weighted V source:
  `official_weighted_v_oracle_because_all_token_v_history_missing`

Result:

```text
classification = layer1_attention_bundle_attention_residual_matches_oracle
```

Metrics:

- raw QK: exact, `max_abs_diff = 0`, `mismatches = 0`
- masked logits: exact, `max_abs_diff = 0`, `mismatches = 0`
- attention probabilities: exact, `max_abs_diff = 0`, `mismatches = 0`
- weighted V: not recomputed; official weighted-V seam used because all-token V
  history is absent
- o-proj: exact, `max_abs_diff = 0`, `mismatches = 0`
- attention residual: exact, `max_abs_diff = 0`, `mismatches = 0`

The emitted local residual artifact path used for the validation run is
`/tmp/layer1_attention_residual_from_bundle.json`; it is not committed.

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: validate layer1 MLP from the emitted attention residual.
Separately keep all-token K pre-RoPE generation as the unresolved
source-complete ladder task.

## Layer1 MLP Backend Validation Status

Layer1 MLP validation now starts from the attention residual emitted by the
bundle-driven layer1 attention seam:

```text
/tmp/layer1_attention_residual_from_bundle.json
```

Mode:

```text
--mode layer1-mlp-backend
```

MLP bundle source:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer1-final-token-mlp-ordered-boundary-bundle-status.json
```

Backend path:

- MLP norm: BF16 input, f32 RMS reduction, BF16 output
- router/top-k: layer-indexed BF16 router tensors
- MLP1: cuBLAS BF16 tensor-op
- SwiGLU: pinned torch-like BF16 stage rounding
- MLP2: BF16 pre-bias/output policy
- residual: BF16 plus BF16 to BF16

Selected experts:

```text
[28,6,1,18]
```

Current result:

```text
classification = layer1_mlp_backend_selected_outputs_mismatch
```

Metrics:

- MLP norm: exact, `max_abs_diff = 0`, `mismatches = 0`
- router logits/top-k/routing weights: exact, `max_abs_diff = 0`,
  `mismatches = 0`, ordered experts match
- selected outputs: one mismatch at rank 0 / expert 28 / hidden lane 2269,
  `max_abs_diff = 0.0001220703125`
- weighted expert sum: one mismatch at hidden lane 2269,
  `max_abs_diff = 0.0000457763671875`
- MLP residual / layer1 output: one mismatch at hidden lane 2269,
  `max_abs_diff = 0.0009765625`

The validation run emitted
`/tmp/layer1_validation_output_from_mlp_backend.json` as a local artifact from
the current backend replay. It is not committed and is not yet an exact
layer2-input guard artifact because the selected-output mismatch remains open.

Correction policy: none. The layer0 expert3 lane1990 correction is not applied
to layer1.

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Caveat: layer1 attention was advanced from an official bundle seam; K/V
source-complete construction remains open separately.

Next bounded step: localize the rank 0 / expert 28 / hidden lane 2269 selected
output mismatch under the layer1 MLP backend path before using the emitted
layer1 output as a layer2 input guard.

## Layer1 Expert28 Lane2269 Localization Status

This slice localizes the only remaining layer1 MLP backend mismatch from the
previous run:

```text
rank = 0
expert = 28
hidden lane = 2269
```

Mode:

```text
--mode layer1-expert28-lane2269-debug
```

The layer1 MLP bundle does not expose expert28 internal MLP1, SwiGLU, or MLP2
pre-bias boundaries. It does expose the selected-output matrix, weighted expert
sum, and MLP residual/output boundaries, so the localization is a selected
output and downstream replacement-impact guard rather than an internal MLP1
or SwiGLU proof.

Lane 2269 values:

- local MLP2 pre-bias: `-0.00115203857421875`
- down bias: `0.024658203125`
- local selected post-bias: `0.0235595703125`
- official selected output: `0.0234375`
- absolute diff: `0.0001220703125`

Lane window summary for lanes 2267..2271:

- lanes 2267, 2268, 2270, and 2271 match official selected post-bias exactly
- lane 2269 is the only mismatch in the window
- no tested bias/output variant reproduced the official lane 2269 value except
  explicit official selected-lane replacement

Replacement impact:

- selected outputs corrected: `max_abs_diff = 0`, `mismatches = 0`
- weighted expert sum corrected: `max_abs_diff = 0`, `mismatches = 0`
- MLP residual corrected: `max_abs_diff = 0`, `mismatches = 0`

Classification:

```text
layer1_expert28_lane2269_downstream_corrected_matches
```

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Caveat: layer1 attention was bundle-seam driven and K/V source-complete history
remains open separately.

Next bounded step: emit a corrected layer1 output artifact behind explicit
validation metadata, then guard it against the layer2 input boundary. Keep the
lane2269 replacement as validation-only metadata, not a production rule.

## Validation-Only Non-Goals

- No production runtime routing
- No default model-runner behavior changes
- No CUDA kernel changes
- No Torch runtime dependency
- No raw `.live` or `/tmp` artifacts committed
- No final-logit claim
- No 4097 work
