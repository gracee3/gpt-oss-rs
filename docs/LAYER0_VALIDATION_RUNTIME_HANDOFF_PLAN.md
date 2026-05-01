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

## Layer3 Raw-QK Single Mismatch Debug Status

The consumer-side focused debug mode localized the one remaining layer3
raw-QK/masked-logit mismatch:

```text
/tmp/layer3_raw_qk_single_mismatch_debug_status.json
classification = layer3_raw_qk_single_mismatch_accumulation_policy_mismatch

layer = 3
q_head = 2
key_column = 1
kv_head = 0
scale = 0.125
local = 0.000293731689453125
official = 0.0002918243408203125
diff = 0.0000019073486328125
```

The selected Q post-RoPE and grouped K post-RoPE source vectors loaded from the
ordered attention bundle were finite and matched the prior source seam checks.
The GQA mapping check kept the expected `kv_head = q_head / 8 = 0`; alternate
KV heads were far from the official value, so the mismatch is not a head mapping
or sink-column issue. Column 1 is a real-token column, not the sink column.

Dot-product variant results:

```text
current sequential f32, scale after sum, BF16 output:
  0.000293731689453125

reverse f32, scale after sum, BF16 output:
  0.0002918243408203125

pairwise f32, scale after sum, BF16 output:
  0.0002918243408203125

f64 diagnostic, scale after sum, BF16 output:
  0.0002918243408203125

deterministic abs-ascending f32:
  0.000293731689453125

BF16-product evidence policy:
  -0.00933837890625
```

The local and official raw-QK values are adjacent BF16 lattice values at this
magnitude. The earliest localized mismatch source is the raw-QK dot-product
accumulation policy. The same one-entry difference appears in masked logits, but
it does not propagate:

```text
masked logits mismatches = 1
attention probabilities mismatches = 0
weighted V mismatches = 0
o_proj mismatches = 0
```

No tolerance, correction metadata, or runtime policy change was applied. Layer3
still should not be recorded as a strict full ordered attention seam pass until
the raw-QK accumulation convention is either matched or explicitly scoped as a
validation-only diagnostic. No layer3 output was emitted, the ladder was not
continued, no runtime/default routing/CUDA behavior changed, and there is no
final-logit, all-layer, server, or 4097-token claim.

## Layer3 Raw-QK Policy Sweep Status

The oracle dtype probe confirmed that the official full raw-QK expression is
deterministic BF16 output and that the single-entry mismatch is consistent with
an accumulation/output-cast boundary:

```text
/tmp/layer3_raw_qk_qhead2_col1_dtype_probe_status.json
classification = layer3_raw_qk_dtype_probe_confirms_accumulation_boundary
official full expression = 0.0002918243408203125
isolated dot / matmul-equivalent = 0.000293731689453125
```

The consumer then swept full final-token raw-QK policies over the entire
`[64, 74]` real-key matrix and rebuilt masked logits over `[64, 75]` using the
official sink column:

```text
/tmp/layer3_raw_qk_policy_sweep_status.json
classification = layer3_raw_qk_policy_sweep_pairwise_clears_full_matrix

focus:
  layer = 3
  q_head = 2
  key_column = 1
  kv_head = 0
  scale = 0.125
```

Policy sweep results:

```text
current sequential f32:
  raw-QK mismatches = 1
  masked-logit mismatches = 1
  attention-probability mismatches = 0

reverse f32:
  raw-QK mismatches = 0
  masked-logit mismatches = 0
  attention-probability mismatches = 0

pairwise f32:
  raw-QK mismatches = 0
  masked-logit mismatches = 0
  attention-probability mismatches = 0

f64 diagnostic:
  raw-QK mismatches = 0
  masked-logit mismatches = 0
  attention-probability mismatches = 0

scale-per-term sequential f32:
  raw-QK mismatches = 1
  masked-logit mismatches = 1
  attention-probability mismatches = 0

deterministic abs-ascending f32:
  raw-QK mismatches = 1
  masked-logit mismatches = 1
  attention-probability mismatches = 0

BF16-product evidence policy:
  raw-QK mismatches = 2433
  masked-logit mismatches = 2433
  attention-probability mismatches = 2581
```

Reverse f32 and pairwise f32 both clear the full raw-QK and masked-logit
matrices without collateral mismatches; f64 remains diagnostic evidence only.
The BF16-product policy remains rejected/evidence-only because it introduces
broad collateral mismatches. Existing downstream provenance remains
non-propagating: attention probabilities, weighted V, and o-proj are exact.

This is validation-only evidence that layer3 raw-QK can be made exact under an
explicit accumulation policy, but no tolerance, correction metadata, runtime
policy, or CUDA/default routing change was applied. No layer3 output was emitted,
the ladder was not continued, and there is no final-logit, all-layer, server, or
4097-token claim.

## Layer3 Ordered Bundle Validation With Raw-QK Policy

The full split ordered bundle validator now accepts an explicit validation-only
raw-QK accumulation policy:

```text
--raw-qk-accum-policy current|pairwise|reverse|f64-diagnostic
```

The default remains `current`. Re-running layer3 with the preferred pairwise f32
policy clears the full ordered layer surface:

```text
/tmp/layer3_ordered_bundle_validate_pairwise_raw_qk_status.json
classification = layer3_ordered_bundle_validate_attention_cleared_mlp_cleared_with_raw_qk_policy
raw_qk_accum_policy = pairwise_f32_scale_after_sum_bf16_output
raw_qk_policy_source_status = /tmp/layer3_raw_qk_policy_sweep_status.json
oracle_dtype_probe_status = /tmp/layer3_raw_qk_qhead2_col1_dtype_probe_status.json

raw QK mismatches = 0
masked logits mismatches = 0
attention probabilities mismatches = 0
weighted V mismatches = 0
o_proj mismatches = 0
attention-to-MLP bridge mismatches = 0

MLP norm mismatches = 0
router logits mismatches = 0
top-k ordered match = true
routing weights mismatches = 0
selected outputs mismatches = 0
weighted sum mismatches = 0
final MLP output mismatches = 0
```

The optional reverse f32 corroboration also clears:

```text
/tmp/layer3_ordered_bundle_validate_reverse_raw_qk_status.json
classification = layer3_ordered_bundle_validate_attention_cleared_mlp_cleared_with_raw_qk_policy
raw_qk_accum_policy = reverse_f32_scale_after_sum_bf16_output
```

Layer3 MLP did not require the deterministic abs-ascending selected-MLP down
policy; the current sequential selected MLP replay is already exact. The
deterministic abs-ascending policy is also exact, while BF16-product remains
rejected/evidence-only due broad collateral mismatches.

This is not default layer3 parity and is not a runtime fix. It is a scoped
validation result: layer3 ordered validation clears under an explicit
validation-only raw-QK pairwise accumulation policy. No tolerance, correction
metadata, layer3 output emission, ladder continuation, runtime/default routing
change, CUDA change, final-logit claim, all-layer claim, server claim, or
4097-token claim was made.

## Layer4 Ordered Surface Validation Status

The oracle lane generated a focused layer4 ordered surface pilot:

```text
/tmp/layer4_ordered_surface_pilot_status.json
/tmp/layer4_ordered_attention_bundle_status.json
/tmp/layer4_ordered_attention_bundle/
/tmp/layer4_ordered_attention_audit_bundle_status.json
/tmp/layer4_ordered_attention_audit_bundle/
/tmp/layer4_ordered_mlp_bundle_status.json
/tmp/layer4_ordered_mlp_bundle/
```

Consumer summary status:

```text
/tmp/layer4_ordered_consumer_surface_status.json
classification = layer4_ordered_consumer_bundle_validation_failed
```

The layer4 attention audit clears the all-token V and residual-add checks:

```text
classification = layer4_ordered_attention_audit_weighted_v_and_residual_cleared
source_complete_attention_capture = true
all_token_v_emitted = true
all_token_v_shape = [74, 8, 64]

weighted V mismatches = 0
attention residual mismatches = 0
attention-to-MLP bridge mismatches = 0
```

Strict/default split bundle validation did not need a raw-QK policy sweep:
raw-QK, masked logits, probabilities, weighted V, bridge, and MLP are exact.
The remaining blocker is a narrow o-proj seam mismatch:

```text
classification = layer4_ordered_bundle_validate_attention_seam_mismatch
raw QK mismatches = 0
masked logits mismatches = 0
attention probabilities mismatches = 0
weighted V mismatches = 0
o-proj mismatches = 2
o-proj max_abs_diff = 0.0000152587890625
first/worst o-proj mismatch = hidden lane 884
local = -0.003265380859375
official = -0.0032806396484375

attention-to-MLP bridge mismatches = 0
MLP norm/router/top-k/selected outputs/weighted sum/final output mismatches = 0
```

Layer4 selected MLP down replay confirms the current sequential baseline is
exact. Unlike layers 1, 2, and 3, deterministic abs-ascending is not a clearing
candidate on this surface because it introduces a one-lane selected-output,
weighted-sum, and final-output collateral mismatch:

```text
classification = layer4_selected_mlp_down_policy_replay_collateral_mismatches
selected_experts = [6, 15, 0, 24]
routing_weights = [0.380859375, 0.255859375, 0.220703125, 0.14453125]

baseline selected outputs mismatches = 0
baseline weighted sum mismatches = 0
baseline final output mismatches = 0
deterministic abs-ascending selected outputs mismatches = 1
deterministic abs-ascending weighted sum mismatches = 1
deterministic abs-ascending final output mismatches = 1
BF16-product evidence policy selected-output mismatches = 3292
BF16-product evidence policy weighted-sum mismatches = 941
BF16-product evidence policy final-output mismatches = 484
```

Raw-QK policy sweep and pairwise revalidation were skipped because strict
raw-QK and masked logits already match exactly.

## Layer4 o-proj Policy Sweep Status

Layer4 o-proj was localized with a focused validation-only policy sweep:

```text
/tmp/layer4_attention_oproj_policy_sweep_status.json
classification = layer4_attention_oproj_policy_sweep_reverse_clears

strict/default blocker:
  o-proj mismatches = 2
  o-proj max_abs_diff = 0.0000152587890625
  first/worst lane = 884
  local = -0.003265380859375
  official = -0.0032806396484375
```

The sweep uses the ordered weighted-V artifact that was already validated by
the attention audit. Upstream attention seams remain exact: raw-QK, masked
logits, probabilities, and weighted V all have zero mismatches. Downstream,
attention residual and the attention-to-MLP bridge remain exact when recomputed
from the clearing o-proj candidate.

Policy summary:

```text
current sequential f32:
  o-proj mismatches = 1
  focus lane 884 = exact
  collateral lane = 2632

sequential f32 with f32 bias:
  o-proj mismatches = 1
  focus lane 884 = exact
  collateral lane = 2632

reverse f32:
  o-proj mismatches = 0
  attention residual mismatches = 0
  attention-to-MLP bridge mismatches = 0

pairwise f32:
  o-proj mismatches = 2

chunked pairwise f32, sizes 16/32/64:
  o-proj mismatches = 2

chunked pairwise f32, size 128:
  o-proj mismatches = 1

f64 diagnostic:
  o-proj mismatches = 1

BF16 pre-bias/bias variant:
  broad collateral mismatches
```

The full split bundle validation was then rerun with an explicit validation-only
o-proj policy:

```text
/tmp/layer4_ordered_bundle_validate_oproj_policy_status.json
classification =
  layer4_ordered_bundle_validate_attention_cleared_mlp_cleared_with_oproj_policy
attention_oproj_policy = reverse_f32_accum_f32_bias_bf16_output

raw QK mismatches = 0
masked logits mismatches = 0
attention probabilities mismatches = 0
weighted V mismatches = 0
o-proj mismatches = 0
attention-to-MLP bridge mismatches = 0
MLP baseline total ordered mismatches = 0
```

Layer4 MLP remains exact under the current sequential down policy. The
deterministic abs-ascending MLP down policy is not used on layer4 because it
introduces three total ordered MLP mismatches; BF16-product remains
evidence-only/rejected due broad collateral mismatches. This is explicit
validation-only evidence, not a default runtime policy. No layer4 output was
emitted, the ladder was not continued, no tolerance or correction metadata was
applied, no runtime/default routing/CUDA behavior changed, and there is no
final-logit, all-layer, server, or 4097-token claim.

Next bounded step: either collect another ordered surface or decide whether
reverse o-proj accumulation belongs in a scoped validation-only policy design
discussion. It is not promoted here.

## Layer5 Ordered Surface Validation Status

The oracle lane generated a focused layer5 ordered surface pilot:

```text
/tmp/layer5_ordered_surface_pilot_status.json
/tmp/layer5_ordered_attention_bundle_status.json
/tmp/layer5_ordered_attention_bundle/
/tmp/layer5_ordered_attention_audit_bundle_status.json
/tmp/layer5_ordered_attention_audit_bundle/
/tmp/layer5_ordered_mlp_bundle_status.json
/tmp/layer5_ordered_mlp_bundle/
```

Consumer summary status:

```text
/tmp/layer5_ordered_consumer_surface_status.json
classification = layer5_ordered_consumer_attention_audit_failed
```

The layer5 attention audit consumes the supplemental all-token V boundary
emitted by the oracle:

```text
classification = layer5_ordered_attention_audit_weighted_v_mismatch
source_complete_attention_capture = true
all_token_v_emitted = true
all_token_v_shape = [74, 8, 64]

weighted V mismatches = 1
weighted V max_abs_diff = 0.00000095367431640625
first/worst weighted V mismatch = hidden lane 3028
local = 0.000194549560546875
official = 0.00019550323486328125

attention residual mismatches = 0
attention-to-MLP bridge mismatches = 0
```

Because Phase A stopped on this weighted-V audit mismatch, the consumer did not
run strict/default layer5 ordered bundle validation, selected MLP down replay,
raw-QK policy sweep, o-proj policy sweep, or policy revalidation. The ordered
MLP bundle metadata remains recorded but unconsumed:

```text
selected_experts = [24, 6, 12, 1]
routing_weights = [0.392578125, 0.25390625, 0.19140625, 0.1611328125]
```

No BF16-product correction or other policy was applied. No layer5 output was
emitted, the ladder was not continued, no tolerance or correction metadata was
applied, no runtime/default routing/CUDA behavior changed, and there is no
final-logit, all-layer, server, or 4097-token claim. Next bounded step:
localize the layer5 weighted-V single-lane audit mismatch, including sink-column
handling, GQA mapping, shape/layout, and rounding policy, before using the
layer5 ordered attention surface for full bundle validation.

## Layer5 Weighted-V Single Mismatch Debug Status

The layer5 weighted-V audit blocker was localized with a focused
validation-only diagnostic:

```text
/tmp/layer5_weighted_v_single_mismatch_debug_status.json
classification =
  layer5_weighted_v_single_mismatch_full_vector_policy_candidate

weighted_v_lane = 3028
q_head = 47
kv_head = 5
head_dim_lane = 20

local = 0.000194549560546875
official = 0.00019550323486328125
diff = 0.00000095367431640625
```

The source layout and mapping checks matched the expected layer5 audit schema:
attention probabilities are `[64, 75]` with sink column index `74`,
all-token V is `[74, 8, 64]`, weighted V is `[4096]` interpreted as
`[64, 64]`, and `q_head = 47` maps to `kv_head = 5`. The probability row and
all-token V values for the focused lane are finite. Including the sink as a
zero-valued term does not change the mismatch, while intentionally mapping the
sink to a real V row is rejected by a large guard mismatch, so the issue is not
localized to sink handling or GQA layout.

The focused weighted-sum variants show an accumulation/output-boundary
difference. Current sequential f32 accumulation produces the local value,
while reverse f32, pairwise f32, and f64 diagnostic accumulation round to the
official BF16 value. The local and official values are adjacent BF16 lattice
values. A lightweight full-vector guard found that reverse f32 and pairwise
f32 clear the entire weighted-V vector, while BF16-product remains
evidence-only/rejected with broad collateral mismatches.

This does not by itself claim full layer5 ordered attention parity. No
tolerance or correction was applied, no full layer5 ordered bundle validation
or MLP replay was run in this slice, no layer5 output was emitted, the ladder
was not continued, no runtime/default routing/CUDA behavior changed, and there
is no final-logit, all-layer, server, or 4097-token claim. Next bounded step:
add or use an explicit validation-only weighted-V accumulation policy, with
pairwise f32 as the conventional deterministic candidate, before rerunning the
layer5 attention audit/full bundle path.

## Layer5 Ordered Bundle Validation With Weighted-V Policy

The layer5 weighted-V policy candidate was then consumed by the attention audit
and full split-status bundle validator:

```text
weighted-V policy source:
  /tmp/layer5_weighted_v_single_mismatch_debug_status.json

pairwise audit:
  /tmp/layer5_ordered_attention_audit_validate_pairwise_weighted_v_status.json
  classification =
    layer5_ordered_attention_audit_weighted_v_and_residual_cleared_with_weighted_v_policy

reverse audit, corroborating:
  /tmp/layer5_ordered_attention_audit_validate_reverse_weighted_v_status.json

full bundle validation:
  /tmp/layer5_ordered_bundle_validate_pairwise_weighted_v_status.json
  classification =
    layer5_ordered_bundle_validate_weighted_v_policy_attention_mismatch

selected MLP down replay:
  /tmp/layer5_selected_mlp_down_policy_replay_status.json
  classification =
    layer5_selected_mlp_down_policy_replay_full_mlp_cleared

summary:
  /tmp/layer5_ordered_consumer_surface_status.json
  classification =
    layer5_ordered_consumer_bundle_validation_failed_under_weighted_v_policy
```

Under explicit validation-only `pairwise_f32_bf16_output`, weighted V is exact
against the ordered reference, and the attention residual recompute plus
attention-to-MLP bridge remain exact. The full ordered bundle validation then
clears raw QK, masked logits, attention probabilities, weighted V, the bridge,
MLP norm, router/top-k, and routing weights, but it stops on an o-proj
mismatch:

```text
o-proj mismatches = 1
o-proj max_abs_diff = 0.0009765625
first/worst lane = 2602
```

The layer5 ordered MLP bundle was also replayed because the bridge/MLP input is
exact. Baseline current sequential MLP down has one selected-output mismatch
and one weighted-sum mismatch while final output remains exact. Deterministic
abs-ascending clears selected outputs, weighted sum, and final output.
BF16-product remains evidence-only/rejected with broad collateral mismatches:
3432 selected-output, 1025 weighted-sum, and 441 final-output mismatches.

No tolerance or correction was applied. The weighted-V policy is
validation-only and is not default runtime behavior. No runtime/default
routing/CUDA behavior changed, no layer5 output was emitted, the ladder was
not continued, and there is no final-logit, all-layer, server, or 4097-token
claim. Next bounded step: localize the layer5 o-proj mismatch under the
explicit pairwise weighted-V policy before any layer5 output emission or
ladder continuation.

## Validation-Only Non-Goals

- No production runtime routing
- No default model-runner behavior changes
- No CUDA kernel changes
- No Torch runtime dependency
- No raw `.live` or `/tmp` artifacts committed
- No final-logit claim
- No 4097 work
