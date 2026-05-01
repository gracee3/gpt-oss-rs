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

## Layer2 Input Guard Preparation Status

Corrected layer1 output emission is now available as a validation-only mode:

```text
--mode layer1-corrected-output
```

Corrected local artifact emitted during validation:

```text
/tmp/layer1_validation_corrected_output.json
```

Correction metadata:

- rank: `0`
- expert: `28`
- hidden lane: `2269`
- validation post-bias: `0.0235595703125`
- official selected: `0.0234375`

Corrected layer1 metrics:

- selected outputs: exact, `max_abs_diff = 0`, `mismatches = 0`
- weighted expert sum: exact, `max_abs_diff = 0`, `mismatches = 0`
- MLP residual / corrected layer1 output: exact, `max_abs_diff = 0`,
  `mismatches = 0`

Layer2 input guard:

```text
--mode layer-input-guard --layer-index 2
```

Layer2 input oracle path:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer1-final-token-mlp-ordered-boundary-bundle-status.json
```

Oracle boundary:

```text
layer1_final_token_hidden_state_after_mlp_residual_add
```

Guard result:

```text
classification = layer2_input_guard_matches_oracle
max_abs_diff = 0
mismatches = 0
```

Caveats:

- layer1 attention was bundle-seam driven
- all-token K/V source-complete construction remains open
- the expert28 lane2269 correction is validation-only metadata, not a
  production rule

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: begin layer2 attention/MLP validation using the same
bundle-driven pattern, while keeping source-complete K/V history construction
as a separate unresolved ladder task.

## Bundle-Driven Layer Ladder Status

Generic bundle-driven ladder scaffolding has been added so remaining layers can
use one validation shape instead of hand-coded layer-specific breakpoints.

Modes added:

- `--mode layer-bundle-discover`
- `--mode layer-bundle-validate`
- `--mode layer-ladder`

Current exact input seed:

```text
/tmp/layer1_validation_corrected_output.json
```

Attempted ladder range:

```text
start_layer = 2
end_layer = 23
```

Layer2 discovery result:

```text
classification = layer_bundle_discovery_missing_attention_bundle
attention_bundle_path = null
mlp_bundle_path = null
```

The pinned artifact root currently contains this layer2 coarse bundle:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer2-to-final-final-token-coarse-layer-ladder-bundle-status.json
```

It does not satisfy the ordered-boundary contract expected by the scaffold:

- `*layer<N>*attention*ordered*boundary*bundle*status.json`
- `*layer<N>*mlp*ordered*boundary*bundle*status.json`

Ladder run result:

```text
classification = layer_ladder_stopped_on_missing_bundle
completed_layers = []
stopped_at_layer = 2
stop_reason = missing_attention_bundle
corrections = []
emitted_outputs = []
```

Source-complete caveats remain unchanged:

- attention validation is bundle-seam driven
- all-token K/V source-complete construction remains open
- selected-output lane corrections are validation-only metadata

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: either produce ordered layer2 attention/MLP boundary bundles
or add a scoped adapter for the existing coarse layer2-to-final ladder bundle,
then rerun `layer-bundle-validate` for layer2 and the ladder.

## Coarse Ladder Bundle Adapter Status

This slice inspected the existing coarse layer2-to-final ladder bundle to decide
whether it can unblock layer2 validation without generating new ordered
artifacts.

Coarse bundle:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer2-to-final-final-token-coarse-layer-ladder-bundle-status.json
```

Modes added:

- `--mode coarse-ladder-inspect`
- `--mode coarse-layer-output-guard`
- `--mode layer-bundle-validate-from-coarse`

Inspection result:

```text
classification = coarse_ladder_inspect_output_guards_only
available_layers = [2, 3, 4, ..., 23]
can_support_layer_output_guards = true
can_support_ordered_bundle_adapter = false
attention_seams_available = false
mlp_seams_available = false
```

Available per-layer coarse boundaries for layers 2..23:

- `layerN_final_token_layer_input_before_attention_norm`
- `layerN_final_token_attention_norm_output_before_qkv`
- `layerN_final_token_hidden_state_after_attention_residual_add_before_mlp`
- `layerN_final_token_mlp_norm_output_before_mlp_projections`
- `layerN_final_token_hidden_state_after_mlp_residual_add`

Layer1 coarse output guard:

```text
classification = coarse_layer_output_guard_missing_layer
```

The coarse bundle starts at layer2 and does not include
`layer1_final_token_hidden_state_after_mlp_residual_add`, so it cannot reproduce
the already-passed corrected-layer1-to-layer2 input guard.

Layer2 coarse adapter:

```text
classification = layer2_coarse_adapter_blocked_by_missing_attention_seams
```

The coarse bundle does not contain Q/K/V, raw-QK, attention probabilities,
weighted-V, o-proj, router/top-k, selected-output, or weighted-sum seams. It
therefore cannot be adapted into the ordered attention/MLP bundle validation
interface.

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: generate ordered layer2 attention and MLP bundles, or add a
deliberately narrower norm/residual-only coarse validation mode. Backend seam
validation still requires ordered layer2 attention/MLP artifacts.

## Coarse Norm/Residual Ladder Validation Status

This slice added a deliberately narrower coarse ladder mode for the existing
layer2-to-final bundle. It is not ordered seam validation. It uses the exact
prior layer output as input, checks the layer attention norm, accepts the
official coarse attention residual as an explicit seam, checks the MLP norm,
then attempts the local BF16 MLP backend replay against the coarse final layer
output.

Modes added:

- `--mode coarse-layer-validate`
- `--mode coarse-ladder-validate`

What the coarse bundle can validate:

- layer input guard
- attention norm output before QKV
- MLP norm output before MLP projections
- final hidden state after MLP residual add, if the local MLP backend reaches it

What it cannot validate:

- source-complete attention internals
- Q/K/V history construction
- raw-QK, masked logits, attention probabilities, weighted-V, or o-proj seams
- router/top-k oracle parity
- selected expert outputs
- weighted expert sum
- selected expert internals
- selected-output one-lane corrections

Layer2 coarse validation result:

```text
classification = coarse_layer_validate_attention_norm_mismatch
input_guard = exact
attention_norm max_abs_diff = 0.001953125
attention_norm mismatches = 1
attention_norm first/worst lane = 2108
local = -0.451171875
official = -0.453125
mlp_norm_from_official_attention_residual_seam = exact
mlp_output = not run
emitted_output = null
```

Coarse ladder result:

```text
classification = coarse_ladder_validate_stopped_on_attention_norm_mismatch
start_layer = 2
end_layer = 23
completed_layers = []
stopped_at_layer = 2
stop_reason = attention_norm_mismatch
emitted_outputs = []
```

Caveats:

- attention residual is accepted as the official coarse seam
- attention internals are not recomputed
- no selected-output corrections are applied in coarse mode
- ordered bundles are still needed for true attention/MLP seam validation

No production behavior changed, no default routing changed, no CUDA kernels
changed, and no raw `.live` or `/tmp` artifacts are committed.

Next bounded step: localize the layer2 attention norm lane 2108 mismatch, or
generate ordered layer2 attention/MLP bundles for source-complete layer2 seam
validation before trying to promote any layer2 output.

## Layer2 Attention Norm Lane2108 Debug Status

The coarse ladder stopped at layer2 attention RMSNorm with one BF16-lane
mismatch. The layer2 input guard was exact, so this slice localized RMSNorm
policy variants before allowing any coarse continuation.

Debug mode added:

- `--mode layer2-attn-norm-debug`

Focus lane:

```text
layer = 2
lane = 2108
local_current_output = -0.451171875
official_output = -0.453125
diff = 0.001953125
input = -1.0078125
weight = 2.109375
classification = layer2_attn_norm_debug_policy_matches_with_variant
```

Variant summary:

```text
A_current = 1 mismatch at lane 2108
D_f64_reduction = exact
F_pairwise_reduction = exact
C_bf16_square_terms = 188 mismatches
I_pre_scale_rounding = 747 mismatches
```

The lane window `2106..2110` shows neighbors match under the current policy.
The exact candidates are global reduction variants, so this is recorded as a
policy localization result rather than a selected-lane correction.

Continuation flag added:

- `--continue-after-attn-norm-diagnostic`

The flag keeps the default coarse behavior strict. When explicitly enabled, the
coarse validator records the attention norm mismatch as diagnostic, then
continues from the official coarse attention residual seam. It does not claim
attention norm parity.

Layer2 continuation:

```text
classification = coarse_layer_validate_mlp_output_matches_with_attn_norm_diagnostic
input_guard = exact
attention_norm = diagnostic mismatch at lane 2108
mlp_norm = exact
mlp_output = exact
selected_experts = [21, 26, 29, 4]
emitted_output = /tmp/coarse_ladder_outputs/layer2_output.json
```

Coarse ladder continuation:

```text
classification = coarse_ladder_validate_stopped_on_mlp_norm_mismatch
completed_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
stopped_at_layer = 11
stop_reason = mlp_norm_mismatch
emitted_outputs = /tmp/coarse_ladder_outputs/layer2_output.json through layer10_output.json
```

Layer11 stop:

```text
mlp_norm lane = 248
local = 0.0308837890625
official = 0.03076171875
max_abs_diff = 0.0001220703125
mismatches = 1
```

Caveats:

- attention residual is still an official coarse seam
- attention internals are not recomputed
- ordered bundles remain required for true attention/MLP seam validation
- diagnostic continuation is validation-only metadata
- no raw `.live` or `/tmp` artifacts are committed

No production behavior changed, no default routing changed, and no CUDA kernels
changed.

Next bounded step: localize the layer11 MLP norm lane 248 policy from the
official coarse attention residual seam, or generate ordered layer11 bundles if
the next step needs source-complete seam evidence.

## Layer11 MLP Norm Lane248 Debug Status

The coarse ladder stopped at layer11 MLP RMSNorm after completing layers
2 through 10 under explicit diagnostic continuation. The source for this
check is the official coarse layer11 attention residual seam; attention
internals remain unrecomputed in coarse mode.

Generic debug mode added:

- `--mode coarse-norm-debug`
- `--norm-kind attention|mlp`

Focus lane:

```text
layer = 11
norm_kind = mlp
lane = 248
local_current_output = 0.0308837890625
official_output = 0.03076171875
diff = 0.0001220703125
input = 0.9296875
weight = 0.88671875
classification = coarse_norm_debug_policy_matches_with_variant
```

Variant summary:

```text
A_current = 1 mismatch at lane 248
D_f64_reduction = exact
F_pairwise_reduction = exact
C_bf16_square_terms = 729 mismatches
I_pre_scale_rounding = 731 mismatches
```

The lane window `246..250` keeps the neighboring lanes exact under the current
policy, while the exact candidates are global reduction variants. This matches
the layer2 attention-norm localization and is recorded as an RMSNorm reduction
policy issue, not as lane correction metadata.

Norm policy option added:

- `--norm-reduction-policy current|pairwise|f64`

The default remains `current`. The `pairwise` and `f64` choices are explicit
validation-only options for coarse checks and do not change production runtime
behavior.

Layer11 pairwise continuation:

```text
classification = coarse_layer_validate_mlp_output_mismatch
norm_reduction_policy = pairwise
input_guard = exact
attention_norm = exact
mlp_norm = exact
selected_experts = [30, 13, 4, 20]
mlp_output max_abs_diff = 0.125
mlp_output mismatches = 1
mlp_output lane = 1480
local = -18.25
official = -18.125
emitted_output = none
```

The pairwise coarse ladder was not rerun past layer11 because layer11 did not
produce an exact coarse layer output. The next blocker is now the layer11 MLP
output mismatch at lane 1480.

Caveats:

- attention residual remains an official coarse seam
- attention internals are not recomputed
- ordered bundles remain required for true attention/MLP seam validation
- norm policy selection is validation-only unless separately promoted
- no raw `.live` or `/tmp` artifacts are committed

No production behavior changed, no default routing changed, and no CUDA kernels
changed.

Next bounded step: localize layer11 MLP output lane 1480, preferably with an
ordered layer11 MLP bundle or a focused selected-expert replay/debug mode.

## Layer11 MLP Output Lane1480 Debug Status

After the pairwise RMSNorm policy cleared layer11 input, attention norm, and
MLP norm, coarse validation exposed one final layer-output mismatch:

```text
layer = 11
lane = 1480
local_final = -18.25
official_final = -18.125
diff = 0.125
classification = coarse_mlp_output_debug_requires_ordered_mlp_bundle
```

Focused debug mode added:

- `--mode coarse-mlp-output-debug`

The mode starts from the official coarse attention residual seam, recomputes
layer11 MLP norm with `--norm-reduction-policy pairwise`, computes router/top-k
locally, replays the selected experts, and attributes the focus lane without
applying corrections.

Layer11 selected experts:

```text
selected_experts = [30, 13, 4, 20]
routing_weights = [0.322265625, 0.287109375, 0.208984375, 0.1806640625]
```

Lane 1480 per-rank contribution summary:

```text
rank0 expert30 selected = -12.75, contribution = -4.10888671875
rank1 expert13 selected = -9.875, contribution = -2.835205078125
rank2 expert4  selected =  1.046875, contribution =  0.218780517578125
rank3 expert20 selected = -2.234375, contribution = -0.4036712646484375

local_weighted_sum = -7.125
attention_residual = -11.0625
local_final = -18.25
official_final = -18.125
```

The required weighted-sum diagnostic shows that a BF16 weighted sum of
`-7.09375`, `-7.0625`, or `-7.03125` would produce the official final output
after BF16 residual add, compared with the local weighted sum `-7.125`.

Policy variant summary:

```text
A_current = 1 mismatch at lane 1480
B_f32_weighted_sum_then_bf16 = 1 mismatch at lane 1480
C_bf16_product_then_f32_sum = clears lane 1480 but causes 392 full-output mismatches
D_pairwise_rank_sum = 1 mismatch at lane 1480
E_sequential_bf16_rank_accum = 574 full-output mismatches
F_residual_f32_add = 1 mismatch at lane 1480
G_residual_bf16_add = 1 mismatch at lane 1480
```

No valid global weighted-sum or residual policy variant clears the full layer11
output from coarse-only evidence. The ordered-consumer slice then consumed the
generated ordered MLP evidence:

```text
/tmp/layer11_ordered_mlp_lane1480_bundle_status.json
/tmp/layer11_ordered_mlp_lane1480_bundle/
```

Focused ordered consumer mode added:

- `--mode coarse-mlp-output-ordered-debug`

Ordered consumer result:

```text
classification = layer11_ordered_mlp_consumer_selected_output_localized
ordered selected experts = [30, 13, 4, 20]
ordered routing weights = [0.322265625, 0.287109375, 0.208984375, 0.1806640625]
```

Local and ordered selected experts match. Local and ordered routing weights
match. MLP input and MLP norm match the ordered bundle exactly.

Ordered lane1480 attribution:

```text
rank0 expert30 local selected = -12.75
rank0 expert30 ordered selected = -12.6875
rank0 diff = 0.0625

rank1 expert13 local/ordered selected = -9.875
rank2 expert4  local/ordered selected =  1.046875
rank3 expert20 local/ordered selected = -2.234375

local_weighted_sum = -7.125
ordered_weighted_sum = -7.09375
local_final = -18.25
ordered_final = -18.125
```

Replacement diagnostics:

```text
rank0/expert30/lane1480 ordered selected-output replacement:
  selected_outputs = exact
  weighted_sum = exact
  final_output = exact

ordered selected-output replacement:
  weighted_sum = exact
  final_output = exact
```

The ordered consumer localizes the layer11 lane1480 mismatch to rank0 /
expert30 selected output. No correction was applied in this slice, no corrected
layer11 output was emitted, and the coarse ladder was not continued.

Caveats:

- the coarse bundle lacks selected-output and weighted-sum seams
- attention residual remains an official coarse seam
- the ordered lane bundle provides selected outputs, weighted sum, and final
  output, but not expert30 internal MLP1/SwiGLU/MLP2 lane1480 boundaries
- no correction was applied from coarse final-output evidence alone
- no raw `.live` or `/tmp` artifacts are committed

No production behavior changed, no default routing changed, and no CUDA kernels
changed.

Next bounded step: either generate expert30 internal MLP1/SwiGLU/MLP2 lane1480
boundaries to localize below selected output, or in a separate slice record the
validation-only rank0/expert30/lane1480 correction metadata before emitting a
corrected layer11 output.

## Layer11 Ordered MLP Down-Projection Reduction Candidate

The layer11 lane1480 proof chain now reaches a validation-only down-projection
candidate convention. This is provenance only: no runtime behavior changed, no
production routing changed, no CUDA kernels changed, no correction metadata was
applied, and no final-logit, all-layer, server, or 4097-token claim is made.

Status chain:

```text
/tmp/layer11_ordered_mlp_consumer_compare_status.json
/tmp/layer11_expert30_internal_consumer_compare_status.json
/tmp/layer11_expert30_down_terms_consumer_compare_status.json
/tmp/layer11_expert30_down_lane1480_einsum_dtype_probe_status.json
/tmp/layer11_expert30_down_cast_policy_sweep_status.json
/tmp/layer11_selected_mlp_down_policy_replay_status.json
```

Proof chain:

- The ordered MLP bundle localized the coarse layer11 lane1480 mismatch to
  selected expert output for rank0 / expert30.
- Focused expert30 internals localized that selected-output mismatch to
  `mlp2_down_pre_bias`.
- Down-term comparison proved the SwiGLU source vector, down weight vector,
  weight orientation, and dot terms match the oracle evidence.
- The dtype probe showed official CPU BF16 Torch `einsum` returns BF16 `-12.0`
  for lane1480, while the local sequential FP32 accumulator crosses the BF16
  midpoint and rounds to `-12.0625`.
- The full-vector expert30 down-cast sweep cleared expert30 down-pre-bias under
  several validation-only policies.
- The selected MLP replay then cleared all selected expert outputs, weighted
  expert sum, and final MLP residual output for selected experts
  `[30, 13, 4, 20]` under those policies.

Valid validation-only full-clearing policies:

```text
naive_f64_sum_then_bf16_output
pairwise_f64_sum_then_bf16_output
pairwise_f32_sum_then_bf16_output
deterministic_f32_abs_ascending_sum_then_bf16_output
```

Baseline and replay summary:

| Policy | selected-output mismatches | weighted-sum mismatches | final-output mismatches | Status |
| --- | ---: | ---: | ---: | --- |
| `current_sequential_f32_accum_bf16_output` | `1` | `1` | `1` | reproduces lane1480 mismatch |
| `naive_f64_sum_then_bf16_output` | `0` | `0` | `0` | clears ordered MLP replay |
| `pairwise_f64_sum_then_bf16_output` | `0` | `0` | `0` | clears ordered MLP replay |
| `pairwise_f32_sum_then_bf16_output` | `0` | `0` | `0` | clears ordered MLP replay |
| `deterministic_f32_abs_ascending_sum_then_bf16_output` | `0` | `0` | `0` | clears ordered MLP replay |

Rejected evidence-only policy:

```text
bf16_product_then_f32_sum_then_bf16_output
```

This policy is not a correction candidate: in the selected-MLP replay it
produces `3473` selected-output mismatches, `1089` weighted-sum mismatches, and
`444` final-output mismatches.

Recommended next proof gate: before opening a runtime-design branch, validate
the candidate convention on at least one additional ordered MLP surface if
available. Prefer an existing layer1 ordered MLP bundle if present; otherwise
request or generate a layer2 ordered MLP pilot before runtime-policy design.

## Layer1 Selected MLP Down Policy Replay Status

The additional ordered MLP proof gate consumed the existing layer1 ordered MLP
bundle:

```text
/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer1-final-token-mlp-ordered-boundary-bundle-status.json
```

Mode:

- `--mode selected-mlp-down-policy-replay-status`

Result:

```text
classification = layer1_selected_mlp_down_policy_replay_full_mlp_cleared
selected_experts = [28, 6, 1, 18]
routing_weights = [0.451171875, 0.2021484375, 0.17578125, 0.1708984375]
focus_lane = 1480
```

Layer1 baseline replay reproduces the already-known rank0 / expert28 /
lane2269 mismatch:

```text
selected outputs: 1 mismatch, max_abs_diff = 0.0001220703125
weighted sum:     1 mismatch, max_abs_diff = 0.0000457763671875
final output:     1 mismatch, max_abs_diff = 0.0009765625
```

Layer1 candidate replay:

| Policy | selected-output mismatches | weighted-sum mismatches | final-output mismatches | Status |
| --- | ---: | ---: | ---: | --- |
| `current_sequential_f32_accum_bf16_output` | `1` | `1` | `1` | baseline mismatch |
| `naive_f64_sum_then_bf16_output` | `1` | `1` | `1` | non-regressive, does not clear |
| `pairwise_f64_sum_then_bf16_output` | `1` | `1` | `1` | non-regressive, does not clear |
| `pairwise_f32_sum_then_bf16_output` | `1` | `1` | `1` | non-regressive, does not clear |
| `deterministic_f32_abs_ascending_sum_then_bf16_output` | `0` | `0` | `0` | clears full ordered MLP replay |
| `bf16_product_then_f32_sum_then_bf16_output` | `3133` | `952` | `475` | evidence-only, rejected |

Best policy by ordered MLP full vector and focus lane:

```text
deterministic_f32_abs_ascending_sum_then_bf16_output
```

This is a validation-only consumer proof surface. No runtime policy discussion
is allowed from this slice, no correction metadata was applied, no ladder was
continued, no production behavior changed, and no raw `/tmp` or `.live`
artifacts are committed.

Next bounded step: if runtime design is pursued, scope it separately using the
recorded layer11 and layer1 ordered MLP proof surfaces; otherwise add another
ordered MLP pilot before design.

## Layer2 Selected MLP Down Policy Replay Status

The third ordered MLP proof surface consumed the oracle-produced layer2 ordered
MLP bundle:

```text
/tmp/layer2_ordered_mlp_bundle_status.json
/tmp/layer2_ordered_mlp_bundle/
```

The layer2 bundle is MLP-only evidence. Its seed is the official/coarse
attention residual seam, so this does not claim source-complete layer2 attention
validation.

Mode:

- `--mode selected-mlp-down-policy-replay-status`

Result:

```text
classification = layer2_selected_mlp_down_policy_replay_full_mlp_cleared
ordered_mlp_seed_policy = official_coarse_attention_residual_seam
selected_experts = [21, 26, 29, 4]
routing_weights = [0.55078125, 0.15625, 0.150390625, 0.142578125]
focus_lane = 1480
```

Layer2 baseline replay:

```text
selected outputs: 1 mismatch, max_abs_diff = 0.00048828125
weighted sum:     1 mismatch, max_abs_diff = 0.00048828125
final output:     exact
```

Layer2 candidate replay:

| Policy | selected-output mismatches | weighted-sum mismatches | final-output mismatches | Status |
| --- | ---: | ---: | ---: | --- |
| `current_sequential_f32_accum_bf16_output` | `1` | `1` | `0` | baseline selected/weighted mismatch |
| `naive_f64_sum_then_bf16_output` | `1` | `1` | `0` | non-regressive, does not clear |
| `pairwise_f64_sum_then_bf16_output` | `1` | `1` | `0` | non-regressive, does not clear |
| `pairwise_f32_sum_then_bf16_output` | `1` | `1` | `0` | non-regressive, does not clear |
| `deterministic_f32_abs_ascending_sum_then_bf16_output` | `0` | `0` | `0` | clears full ordered MLP replay |
| `bf16_product_then_f32_sum_then_bf16_output` | `3371` | `919` | `416` | evidence-only, rejected |

Best policy by ordered MLP full vector and focus lane:

```text
deterministic_f32_abs_ascending_sum_then_bf16_output
```

This supports moving to a separate scoped runtime-design discussion for the
selected MLP down convention. The replay itself is MLP-only; the later ordered
attention and audit sections record the layer2 attention evidence. No runtime
behavior changed, no correction metadata was applied, no ladder was continued,
and no raw `/tmp` or `.live` artifacts are committed.

## Layer2 Ordered Bundle Validation Status

The consumer lane now consumes the oracle-produced layer2 ordered attention
bundle alongside the existing layer2 ordered MLP bundle:

```text
/tmp/layer2_ordered_attention_bundle_status.json
/tmp/layer2_ordered_attention_bundle/
/tmp/layer2_ordered_mlp_bundle_status.json
/tmp/layer2_ordered_mlp_bundle/
```

Mode:

- `--mode layer-bundle-validate`
- split status inputs: `--attention-bundle-status` and `--mlp-bundle-status`

Result:

```text
classification = layer2_ordered_bundle_validate_attention_cleared_mlp_cleared
source_complete_attention_capture = true
all_token_v_emitted = false
selected_experts = [21, 26, 29, 4]
routing_weights = [0.55078125, 0.15625, 0.150390625, 0.142578125]
best_mlp_down_policy = deterministic_f32_abs_ascending_sum_then_bf16_output
```

Attention seam results:

```text
Q final-token RoPE: exact
K final-token RoPE: exact
raw QK: exact
masked logits: exact
attention probabilities: exact
weighted V: official boundary used because all-token V is not emitted
o_proj: exact
attention residual -> ordered MLP input bridge: exact
```

MLP replay results:

```text
MLP norm: exact
router logits: exact
top-k/routing weights: exact
baseline current sequential policy:
  selected outputs: 1 mismatch
  weighted sum:     1 mismatch
  final output:     exact
deterministic abs-ascending policy:
  selected outputs: exact
  weighted sum:     exact
  final output:     exact
BF16-product evidence policy:
  selected outputs: 3371 mismatches
  weighted sum:     919 mismatches
  final output:     416 mismatches
```

This is validation-only ordered layer2 evidence. It does not continue the
layer ladder, does not emit a layer2 output, does not change runtime behavior,
and does not claim final logits, all-layer parity, server parity, or 4097-token
coverage. All-token V was used by the oracle producer internally for weighted-V
construction but is not emitted as a separate consumer boundary, so the consumer
uses the ordered weighted-V seam for o-proj validation.

## Layer2 Attention Audit Validation Status

The oracle lane added a supplemental layer2 ordered attention audit bundle:

```text
/tmp/layer2_ordered_attention_audit_bundle_status.json
/tmp/layer2_ordered_attention_audit_bundle/
```

Mode:

- `--mode attention-audit-validate`
- source attention status: `/tmp/layer2_ordered_attention_bundle_status.json`
- audit status: `/tmp/layer2_ordered_attention_audit_bundle_status.json`
- prior ordered validation status: `/tmp/layer2_ordered_bundle_validate_status.json`

Result:

```text
classification = layer2_ordered_attention_audit_weighted_v_and_residual_cleared
source_complete_attention_capture = true
all_token_v_emitted = true
all_token_v_shape = [74, 8, 64]
all_token_v_layout = all-real-token V projection tensor [token, kv_head, head_dim]
```

Audit recomputation:

```text
weighted V:
  attention probabilities + all-token V -> weighted V
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

attention residual:
  layer input before attention norm + o_proj -> attention residual
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

attention residual -> ordered MLP input bridge:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0
```

This removes the prior consumer caveats that weighted V was only an official
seam and that the attention residual was only bridge-checked. Layer2 MLP remains
validated by the ordered MLP bundle with the deterministic abs-ascending
selected MLP down policy. The ladder was not continued, no layer2 output was
emitted, no runtime/default routing/CUDA behavior changed, and this makes no
final-logit, all-layer, server, or 4097-token claim.

## Layer3 Ordered Surface Validation Status

The oracle lane generated a focused layer3 ordered surface pilot:

```text
/tmp/layer3_ordered_surface_pilot_status.json
/tmp/layer3_ordered_attention_bundle_status.json
/tmp/layer3_ordered_attention_bundle/
/tmp/layer3_ordered_attention_audit_bundle_status.json
/tmp/layer3_ordered_attention_audit_bundle/
/tmp/layer3_ordered_mlp_bundle_status.json
/tmp/layer3_ordered_mlp_bundle/
```

Consumer summary status:

```text
/tmp/layer3_ordered_consumer_surface_status.json
classification = layer3_ordered_consumer_bundle_validation_failed
```

The layer3 attention audit itself cleared:

```text
classification = layer3_ordered_attention_audit_weighted_v_and_residual_cleared
source_complete_attention_capture = true
all_token_v_emitted = true
all_token_v_shape = [74, 8, 64]
all_token_v_layout = all-real-token V projection tensor [token, kv_head, head_dim]

weighted V max_abs_diff = 0
weighted V mismatches = 0
attention residual max_abs_diff = 0
attention residual mismatches = 0
attention-to-MLP bridge mismatches = 0
```

The split ordered bundle validation consumed the audit all-token V boundary for
weighted-V recomputation, but the full attention seam check stopped on a narrow
raw-QK/masked-logit mismatch:

```text
classification = layer3_ordered_bundle_validate_attention_seam_mismatch
raw QK mismatches = 1
raw QK max_abs_diff = 0.0000019073486328125
first/worst raw QK mismatch = q_head 2, column 1
local = 0.000293731689453125
official = 0.0002918243408203125

masked logits mismatches = 1
masked logits max_abs_diff = 0.0000019073486328125
attention probabilities mismatches = 0
weighted V mismatches = 0
o_proj mismatches = 0
attention-to-MLP bridge mismatches = 0
```

Layer3 ordered MLP replay was exact under the current sequential policy, so the
deterministic abs-ascending down policy was not required for this layer:

```text
classification = layer3_selected_mlp_down_policy_replay_baseline_already_clear
selected_experts = [0, 9, 23, 1]
routing_weights = [0.3984375, 0.30078125, 0.1630859375, 0.1376953125]

baseline selected outputs mismatches = 0
baseline weighted sum mismatches = 0
baseline final output mismatches = 0
deterministic abs-ascending selected outputs mismatches = 0
deterministic abs-ascending weighted sum mismatches = 0
deterministic abs-ascending final output mismatches = 0
BF16-product evidence policy selected-output mismatches = 3336
BF16-product evidence policy weighted-sum mismatches = 974
BF16-product evidence policy final-output mismatches = 589
```

This is validation-only layer3 evidence. It does not continue the layer ladder,
does not emit a layer3 output, does not change runtime/default routing/CUDA
behavior, and does not claim final logits, all-layer parity, server parity, or
4097-token coverage. Next bounded step: localize the layer3 raw-QK/masked-logit
single-entry mismatch before claiming full ordered layer3 attention seam parity.

## Validation-Only Non-Goals

- No production runtime routing
- No default model-runner behavior changes
- No CUDA kernel changes
- No Torch runtime dependency
- No raw `.live` or `/tmp` artifacts committed
- No final-logit claim
- No 4097 work
