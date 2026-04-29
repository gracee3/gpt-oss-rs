# Layer0 Validation Runtime Path Plan

This plan turns the scratch layer0 downstream-equivalence result into a clean,
validation-only Rust/CUDA path. It does not promote a production route and does
not change default model-runner behavior.

## Current Evidence

The source scratch branch proved the exact `developer-message-user-smoke` layer0
final-token path using local artifact generation and oracle/model calls:

| Seam | Result | max abs diff | mismatches |
| --- | --- | ---: | ---: |
| Q/K raw QK | exact | `0.0` | `0` |
| V weighted sum | exact | `0.0` | `0` |
| Attention o-proj before residual | exact | `0.0` | `0` |
| Attention residual add before MLP | exact | `0.0` | `0` |
| MLP norm | exact | `0.0` | `0` |
| Router logits | exact | `0.0` | `0` |
| Top-k/routing | exact | `0.0` | `0` |
| Selected expert outputs | exact after fresh expert30 reconstruction | `0.0` | `0` |
| Weighted expert sum | exact | `0.0` | `0` |
| MLP residual / layer0 final-token output | exact | `0.0` | `0` |

The important correction was that the earlier Q/K raw-QK drift came from the
scratch RoPE generator. Once the official/model RoPE call was used, custom
decomposed Q/K reached the official raw-QK BF16 boundary exactly.

## Target Validation Runtime Path

The target is a validation-only bench path that runs layer0 for one exact case
under Rust-controlled CUDA execution and compares the final-token layer0 output
to the official oracle.

Input:

- Exact layer0 input/residual or token hidden states for
  `developer-message-user-smoke`.
- Layer0 model tensors needed for attention, MLP norm, router, selected experts,
  and output projections.
- Official layer0 final-token hidden state after MLP residual add, shape
  `[2880]`.

Operations:

1. Layer0 attention norm.
2. Q/K/V projections under the validation decomposed BF16 policy.
3. Runtime/kernel RoPE for Q/K, not scratch Python RoPE.
4. Raw QK, mask, and softmax for the final query token.
5. Weighted V.
6. Attention o-proj.
7. Attention residual add.
8. MLP norm.
9. Router logits, top-k, and routing weights.
10. Selected expert replay for the selected experts.
11. Weighted expert sum.
12. MLP residual add.

Output:

- Layer0 final-token hidden state after MLP residual add, shape `[2880]`.

Comparison:

- Compare against the official layer0 final-token hidden state after MLP
  residual add with max/mean absolute difference, mismatch count, and first/worst
  hidden lane.

## Reusable Pieces

- `crates/gpt-oss-gpu/src/cublas.rs`
  - Public BF16 GEMM APIs exist for current tensor-op and pedantic validation
    policies.
- `kernels/bf16_linear_bias_validation.cu`
  - Validation-only BF16 biased linear kernel for decomposed projection checks.
- `crates/gpt-oss-gpu/src/kernel_loader.rs`
  - Registers CUDA kernels, including RoPE kernels and the BF16 biased linear
    validation kernel.
- `crates/gpt-oss-bench/src/bin/qkv_projection_policy_compare.rs`
  - Contains flexible JSON artifact loading, BF16 conversion, comparison
    reporting, and policy-classification patterns that can be factored later.
- `crates/gpt-oss-model-runner/src/gpu_layer.rs`
  - Contains runtime layer pieces for RMSNorm, RoPE application, attention, and
    MoE execution.
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
  - Contains model-level tensor loading and RoPE table setup that should be
    reused carefully rather than duplicated.
- `docs/CUDA_BF16_LAYER0_DOWNSTREAM_EQUIVALENCE_SUMMARY.md`
  - Records the scratch equivalence result and caveats.

## Gaps

- RoPE table/source parity:
  scratch Python RoPE was wrong. The validation runtime path must use the
  runtime/kernel RoPE setup and explicitly guard it against official K pre/post
  RoPE before treating Q/K downstream evidence as valid.
- Model loading APIs:
  the bench path needs a narrow way to load only layer0 tensors without routing
  production model-runner execution through a validation policy.
- Custom projection policy API:
  the decomposed BF16 policy needs an explicit validation-only execution surface,
  separate from production projection routes.
- Artifact schema:
  official input/output artifacts and intermediate optional guards should use a
  small, stable JSON schema with comparable digests and no raw `.live` commits.
- Selected expert fresh replay:
  stale selected-output artifacts caused the expert30 false mismatch. The runtime
  path should recompute selected expert outputs directly.
- Scope control:
  layer ladder, final logits, performance work, and 4097 remain later milestones.

## Staged Implementation

1. Skeleton/status binary.
   - Parse explicit layer0 input and official output paths.
   - Emit the planned JSON schema and `runtime_behavior_changed = false`.
   - No CUDA execution and no model-runner routing.
2. Layer0 attention-only validation.
   - Load exact layer0 input and attention tensors.
   - Run attention norm, validation Q/K/V projections, runtime/kernel RoPE,
     raw-QK guard, weighted V, o-proj, and attention residual.
3. Layer0 MLP validation.
   - Run MLP norm, router/top-k, selected expert replay, and weighted expert sum
     from the attention residual boundary.
4. Full layer0 final-token validation.
   - Join attention and MLP stages and compare final layer0 output `[2880]` to
     official.
5. Layer ladder and final logits later.
   - Extend only after layer0 is clean, benchmarked, and integration-safe.

## Initial Skeleton Binary

`crates/gpt-oss-bench/src/bin/layer0_validation_runtime_path.rs` is a
non-executing status binary for this first slice. It validates that the explicit
input and official-output paths exist, then emits a JSON plan artifact.

It intentionally does not:

- launch CUDA kernels,
- load a model,
- call model-runner layer execution,
- alter production routing,
- import Torch/oneDNN/runtime-forward proof code, or
- write raw tensor artifacts.

## K RoPE Validation Status

The first attention-only validation slice targets the K RoPE guard:

```text
official K pre-RoPE
  -> runtime/kernel RoPE path
  -> official K post-RoPE
```

This guard comes first because the prior scratch Python RoPE generator was
wrong, while the official/model RoPE call made Q/K raw-QK exact. The validation
runtime path must therefore prove it can reproduce official K post-RoPE before
it uses K/Q downstream evidence.

Status:

- Submode added: `--mode k-rope`.
- Runtime RoPE table helper added:
  `gpt_oss_model_runner::rope_validation::build_runtime_rope_tables`.
- Validation K RoPE helper added:
  `gpt_oss_model_runner::rope_validation::apply_k_rope_f16_validation`.
- Runtime `GpuModelRunner` table setup now calls the shared table helper, so the
  validation path does not own independent table math.
- Required inputs:
  - `--k-pre-rope`
  - `--k-post-rope-oracle`
  - `--output`
- Artifact metadata/loading supports:
  - pre-RoPE `[74,512]` flat K or `[74,8,64]` grouped K,
  - post-RoPE `[74,8,64]`,
  - the existing runtime-forward K projection status key
    `official_projection_outputs.official_module_k_output_f32`.
- Runtime behavior changed: no.
- CUDA execution in this submode: yes, validation-only.

Current classification:

```text
layer0_validation_k_rope_mismatch
```

Metric from `/tmp/layer0_validation_k_rope_status.json`:

| Comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| runtime f16 K RoPE vs official K post-RoPE | `31.828125` | `0.8329416` | `37882` |

First mismatch:

- token `0`, kv head `0`, lane `0`
- runtime output `-1.7734375`
- official output `-2.390625`
- abs diff `0.6171875`

Worst mismatch:

- token `69`, kv head `5`, lane `11`
- runtime output `26.671875`
- official output `58.5`
- abs diff `31.828125`

Interpretation:

- The API blocker is cleared: the bench now uses a narrow model-runner helper
  and the existing `rotary_embedding_f16_kernel`.
- The parity guard does not clear yet.
- The mismatch begins at token `0`, where RoPE should be identity, so the next
  slice should first verify that the supplied K pre-RoPE artifact is the exact
  tensor that generated the official K post-RoPE artifact before changing RoPE
  math.
- If artifact identity is confirmed, the remaining issue is runtime RoPE
  table/source parity with the official/model RoPE call.

Reproduction command:

```bash
cargo run -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda -- \
  --mode k-rope \
  --k-pre-rope /home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-k-projection-weight-arithmetic.cpu.json \
  --k-post-rope-oracle /home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-k-post-rope-grouped.cpu.json \
  --output /tmp/layer0_validation_k_rope_status.json
```

Required next bounded step:

Pin the exact K pre-RoPE artifact identity against the official K post-RoPE
artifact. If token-0 identity still fails with the correct source tensor, expose
or align the runtime RoPE table source with the official/model RoPE source.

## RoPE Table/Artifact Identity Investigation

Status JSON:

`/tmp/layer0_validation_rope_table_investigation.json`

The K pre-RoPE artifact used by `--mode k-rope` is:

`/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-k-projection-weight-arithmetic.cpu.json`

Metadata:

- boundary: `layer0_attn_qkv_k_projection_before_grouped_view`
- case: `developer-message-user-smoke`
- schema: `runtime_forward_layer0_k_projection_official_weight_arithmetic/v1`
- token count: `74`
- K dimension: `512`
- value count: `37888`
- selected value key: `official_projection_outputs.official_module_k_output_f32`
- source: official Torch CPU BF16 `torch.nn.functional.linear/module attn.qkv`
- model root: `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- K weight tensor: `block.0.attn.qkv.weight[K slice]`
- K bias: present, BF16, all zero

The official K post-RoPE oracle is:

`/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-k-post-rope-grouped.cpu.json`

Metadata:

- boundary: `layer0_k_post_rope_grouped`
- case: `developer-message-user-smoke`
- schema: `pinned-prompt-official-intermediate-capture-output/v2`
- token count: `74`
- shape: `[74, 8, 64]`
- value count: `37888`
- backend: official Torch
- official model: `/data/models/openai/gpt-oss-20b`
- prompt renderer: `harmony_gpt_oss_rs`

Artifact identity finding:

`k_pre_rope_artifact_identity_confirmed`

A prior official/model RoPE guard summary at
`/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/qk_official_rope_attribution_summary.json`
records that applying
`gpt_oss.torch.model.py::RotaryEmbedding.forward` through
`AttentionBlock.rope` to this same K pre-RoPE artifact reproduces the official
K post-RoPE oracle exactly:

| guard | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| official K pre-RoPE through official/model RoPE vs official K post-RoPE | `0` | `0` | `0` |

This rules out the K pre/post artifact pair as the source of the current
validation mismatch.

Runtime table source finding:

`build_runtime_rope_tables` currently constructs a plain `rope_theta`
inverse-frequency table:

```text
freq = 1 / rope_theta^(2i / head_dim)
theta = position * freq
cos = cos(theta)
sin = sin(theta)
```

This matches the previous plain-table failure pattern:

| source | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| validation runtime helper / plain `rope_theta` table | `31.828125` | `0.8329416` | `37882` |
| prior scratch plain config `rope_theta` table | about `31.875` | about `0.8329` | not recorded here |

The integration/runtime branch does carry RoPE scaling fields in
`ModelRunnerConfig`, but the current runtime table helper does not consume
`rope_scaling_type`, `rope_scaling_factor`, `rope_ntk_alpha`,
`rope_ntk_beta`, `initial_context_length`, or `rope_scaling_truncate`.

The proven/runtime-forward and archived replay-probe branches contain a
YaRN-aware table helper. That helper:

- enables YaRN when `rope_scaling_type == "yarn"` and factor `> 1`
- computes `concentration = 0.1 * ln(factor) + 1`
- derives a low/high ramp from `initial_context_length`, `beta_fast`,
  `beta_slow`, and `rope_theta`
- blends interpolation/extrapolation inverse frequencies
- scales cos/sin rows by the concentration factor

Conclusion:

The current K RoPE mismatch is no longer an API-access blocker or an artifact
identity ambiguity. It is a runtime table-source mismatch: validation is using
the plain `rope_theta` table while the official/proven path requires the
YaRN-scaled table for this restricted GPT-OSS checkpoint.

Recommended next bounded step:

Add a validation-safe YaRN-aware table helper that reuses the proven
`ModelRunnerConfig` fields instead of changing kernel math. Preserve existing
runtime behavior until the helper is validated, then rerun `--mode k-rope`
against the same confirmed K pre/post artifact pair.

## YaRN-Aware RoPE Validation Helper Status

Validation helper added:

- `build_validation_rope_tables_from_config`
- `apply_k_rope_f16_validation_with_config`

Scope:

- validation-only
- uses `ModelRunnerConfig` RoPE fields
- does not change production runtime table construction
- does not modify CUDA kernel math
- launches the existing `rotary_embedding_f16_kernel`

YaRN configuration used for the exact smoke case:

| field | value |
| --- | --- |
| `rope_theta` | `150000` |
| `rope_scaling_type` | `yarn` |
| `rope_scaling_factor` | `32` |
| `rope_ntk_alpha` / beta_slow | `1` |
| `rope_ntk_beta` / beta_fast | `32` |
| `initial_context_length` | `4096` |
| `rope_scaling_truncate` | `false` |

Result:

`layer0_validation_k_rope_mismatch`

| comparison | table source | max abs diff | mean abs diff | mismatches |
| --- | --- | ---: | ---: | ---: |
| validation K RoPE vs official K post-RoPE | `yarn_scaled` | `0.5` | `0.006694896` | `15473` |

First mismatch:

- token `0`, kv head `0`, lane `1`
- validation output `-5.5625`
- official output `-5.53125`
- abs diff `0.03125`

Worst mismatch:

- token `1`, kv head `5`, lane `11`
- validation output `73.0`
- official output `72.5`
- abs diff `0.5`

Interpretation:

- The YaRN-aware helper eliminates the plain-table failure scale
  (`31.828125` max abs diff down to `0.5`).
- The remaining mismatch matches the earlier "YaRN family but not exact
  official/model call" pattern.
- Because the official/model implementation casts cos/sin factors to the
  tensor dtype inside `_apply_rotary_emb`, and the current validation launcher
  still uses the f16 runtime kernel, the remaining delta is likely in the
  BF16 official/model factor/input/output casting boundary rather than the
  YaRN frequency formula.

Recommended next bounded step:

Keep production runtime unchanged. Add a validation-only BF16-boundary RoPE
application path using the existing float kernel or a scoped helper that
rounds factors, inputs, and outputs like `gpt_oss.torch.model.py`, then compare
against the same confirmed K pre/post artifact pair before implementing the
attention-only path.

## RoPE Application Dtype Policy

Official/model behavior inspected:

- Source: `/home/emmy/openai/gpt-oss/gpt_oss/torch/model.py`
- `_apply_rotary_emb` casts `cos` and `sin` to `x.dtype` before applying RoPE.
- The official K tensor for this boundary is BF16.
- The official formula uses half-split lanes:
  `x1*cos - x2*sin`, `x2*cos + x1*sin`.
- Output remains at the BF16 tensor boundary.

The validation binary now emits a CPU-side discriminator before introducing a
new CUDA path. This keeps the YaRN table source shared with
`build_validation_rope_tables_from_config` while isolating the application
dtype boundary.

Current f16-kernel guard:

| comparison | table source | application policy | max abs diff | mean abs diff | mismatches |
| --- | --- | --- | ---: | ---: | ---: |
| `rotary_embedding_f16_kernel` then BF16 output | `yarn_scaled` | f16 input/output kernel | `0.5` | `0.006694896` | `15473` |

CPU discriminator results:

| variant | max abs diff | mean abs diff | mismatches | result |
| --- | ---: | ---: | ---: | --- |
| f32 input, f32 cos/sin, f32 math, BF16 output | `0.5` | `0.006613972` | `15416` | mismatch |
| BF16 input, BF16 cos/sin, BF16-rounded multiply/add, BF16 output | `0` | `0` | `0` | exact |
| BF16 input, BF16 cos/sin widened to f32 math, BF16 output | `0.5` | `0.0037691281` | `7783` | mismatch |
| f16 input, f16 cos/sin widened to f32 math, f16 output | `0.5625` | `0.0076969406` | `34468` | mismatch |

Classification:

`layer0_validation_k_rope_bf16_boundary_matches_oracle`

Conclusion:

- The YaRN table is correct for the validation artifact pair.
- The production f16 RoPE kernel is not the official/model BF16 boundary for
  this check.
- The official/model K post-RoPE boundary is reproduced exactly by BF16 input,
  BF16 RoPE factors, BF16-rounded multiply/add, and BF16 output.
- This remains validation-only. No production runtime routing or CUDA kernel
  math changed.

Recommended next bounded step:

Add a narrow validation-only RoPE application helper for the identified BF16
boundary, then use it in the attention-only validation path through raw QK.

## BF16-Boundary RoPE Validation Helper Status

Helper added:

- `gpt_oss_model_runner::rope_validation::apply_k_rope_bf16_boundary_validation`

Scope:

- validation-only CPU/Rust helper
- uses `build_validation_rope_tables_from_config`
- consumes the pinned YaRN table source
- does not route production runtime paths
- does not replace or modify the production f16 RoPE CUDA kernel

Policy:

- K pre-RoPE input rounded to BF16
- YaRN `cos`/`sin` factors rounded to BF16, matching the official/model
  `cos.to(x.dtype)` / `sin.to(x.dtype)` boundary
- half-split formula:
  `x1*cos - x2*sin`, `x2*cos + x1*sin`
- BF16-rounded multiply/add boundaries
- BF16 output returned as f32 values for artifact comparison

Result:

`layer0_validation_k_rope_bf16_boundary_matches_oracle`

| comparison | table source | application policy | max abs diff | mean abs diff | mismatches |
| --- | --- | --- | ---: | ---: | ---: |
| BF16-boundary helper vs official K post-RoPE | `yarn_scaled` | BF16 input/factors/math/output | `0` | `0` | `0` |
| existing f16 kernel diagnostic vs official K post-RoPE | `yarn_scaled` | f16 input/output kernel then BF16 output | `0.5` | `0.006694896` | `15473` |

Conclusion:

- The official K pre/post RoPE artifact pair is now reproduced exactly by the
  validation helper.
- The remaining f16 kernel mismatch is retained as a diagnostic contrast; it
  is not used as the official/model BF16 validation boundary.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Use the BF16-boundary helper in the layer0 attention-only validation path
through raw QK, then compare against the official raw-QK oracle.

## Raw-QK Validation Status

Submode added:

- `--mode raw-qk`

Inputs used for the current validation run:

| input | source |
| --- | --- |
| Q pre-RoPE | `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/q_pre_rope_custom.json` |
| K pre-RoPE | `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-k-projection-weight-arithmetic.cpu.json` |
| raw-QK oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-final-token-raw-scaled-qk-logits-pre-mask.cpu.json` |

Q source note:

- No full-value official Q pre-RoPE artifact was found in the runtime-forward
  `.live` search.
- This run uses the previously validated custom/decomposed Q pre-RoPE scratch
  artifact from the downstream Q/K artifact pack.
- The scratch artifact is not committed.

Policy:

- Q and K pre-RoPE values are passed through
  `apply_k_rope_bf16_boundary_validation`.
- RoPE table source is `yarn_scaled`.
- RoPE application policy is
  `bf16_input_bf16_factors_bf16_rounded_math_bf16_output`.
- Final-token raw QK uses `kv_head = q_head / 8`, scale `0.125`, and BF16
  output before mask.

Result:

`layer0_validation_raw_qk_matches_oracle`

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| BF16-boundary Q/K RoPE + raw-QK vs official raw-QK oracle | `0` | `0` | `0` |

Conclusion:

- The validation-runtime path can now reproduce the first layer0 attention seam
  through final-token raw scaled QK pre-mask.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Extend attention-only validation to the mask/softmax probability boundary, then
weighted V.

## Attention Mask and Softmax Validation Status

Submode added:

- `--mode attention-probs`

Inputs for the validation run:

| input | source |
| --- | --- |
| Q pre-RoPE | `/tmp/qkv_projection_qk_downstream_artifacts-20260428-111933/q_pre_rope_custom.json` |
| K pre-RoPE | `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-k-projection-weight-arithmetic.cpu.json` |
| raw-QK oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.official-layer0-final-token-raw-scaled-qk-logits-pre-mask.cpu.json` |
| masked-logits oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-masked-scaled-qk-logits-pre-softmax-status.json` |
| attention-probs oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-probs-post-softmax-status.json` |

Policy:

- Regenerate final-token raw-QK internally from Q/K pre-RoPE using the
  BF16-boundary RoPE helper.
- Real-key masked-logit columns are copied from the regenerated raw-QK scores.
- No real key is masked for the final token in this 74-token prompt.
- Sink column source:
  `official_masked_logits_oracle_sink_column`.
- Softmax policy:
  `f32_subtract_max_exp_sum_bf16_output`.

Current classification:

```text
layer0_validation_attention_probs_match_oracle
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| raw-QK guard | `0` | `0` | `0` |
| masked logits, real-key columns | `0` | `0` | `0` |
| masked logits, sink column | `0` | `0` | `0` |
| masked logits, all columns | `0` | `0` | `0` |
| attention probabilities, real-key columns | `0` | `0` | `0` |
| attention probabilities, sink column | `0` | `0` | `0` |
| attention probabilities, all columns | `0` | `0` | `0` |

Conclusion:

- The validation-runtime path now reproduces the layer0 final-token attention
  mask and post-softmax probability boundary exactly.
- Sink generation is not integration-native yet; this slice uses the official
  masked-logits sink column as an explicit validation input.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Use the exact attention probabilities to validate weighted V, then continue to
attention o-proj only after weighted V clears.

## Weighted V Validation Status

Submode added:

- `--mode weighted-v`

Inputs for the validation run:

| input | source |
| --- | --- |
| attention probabilities | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-probs-post-softmax-status.json` |
| V values | `/tmp/qkv_projection_v_artifacts-pinned-20260428-092126/v_oracle_matmul_plus_bias.json` |
| weighted-V oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-weighted-value-sum-before-output-projection-status.json` |

V source note:

- The pinned V pack also contains
  `/tmp/qkv_projection_v_artifacts-pinned-20260428-092126/v_cpu_bf16_replay_post_bias.json`,
  but that artifact does not clear this weighted-V boundary against the current
  official oracle.
- This run uses the pinned explicit matmul-plus-bias BF16 output artifact. That
  artifact differs from the official/module V projection boundary by only the
  known tiny projection-boundary ambiguity, and it clears weighted V exactly
  after BF16 output rounding.
- The selected V artifact is not committed.

Policy:

- Attention probabilities are `[64,75]` and include the sink column.
- Sink position `74` is dropped for weighted V.
- V is accepted as `[74,512]` or logical `[74,8,64]`.
- GQA mapping uses `kv_head = q_head / 8`.
- The validation binary emits both:
  - f32 exploratory weighted sum vs oracle,
  - BF16-rounded weighted sum vs oracle.

Current classification:

```text
layer0_validation_weighted_v_matches_oracle
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| f32 exploratory weighted V vs official | `0.013309479` | `0.0006673922` | `4096` |
| BF16-rounded weighted V vs official | `0` | `0` | `0` |

Conclusion:

- The validation-runtime path now reproduces the layer0 final-token weighted-V
  boundary exactly after BF16 output rounding.
- Sink handling is explicit: the sink participates in softmax normalization but
  is not included in weighted V.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Extend attention-only validation to the attention output projection before the
attention residual add.

## Attention o_proj Validation Status

Submode added:

- `--mode attention-oproj`

Inputs for the validation run:

| input | source |
| --- | --- |
| weighted V | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-weighted-value-sum-before-output-projection-status.json` |
| o_proj weight | `/tmp/layer0_validation_attention_oproj_artifacts-20260428-180208/oproj_weight.json` |
| o_proj bias | `/tmp/layer0_validation_attention_oproj_artifacts-20260428-180208/oproj_bias.json` |
| o_proj oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-attention-output-after-o-proj-before-residual-status.json` |

Input-source caveat:

- This first o_proj validation run uses the official weighted-V oracle as the
  o_proj input because the weighted-V submode does not yet emit its BF16 output
  values as a reusable artifact.
- The prior weighted-V seam established that the BF16-rounded weighted-V output
  matches this official weighted-V oracle exactly.
- The o_proj weight and bias JSON files are scratch artifacts generated from
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration/model-00000-of-00002.safetensors`
  using tensor names `model.layers.0.self_attn.o_proj.weight` and
  `model.layers.0.self_attn.o_proj.bias`.
- No scratch artifacts are committed.

Policy tested in the Rust validation binary:

- Weighted-V input is rounded to BF16.
- o_proj weight and bias are rounded to BF16.
- Products are accumulated in f32.
- The post-bias output is rounded to BF16.

Current classification:

```text
layer0_validation_attention_oproj_mismatch
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| Rust BF16-input/f32-accum/BF16-output o_proj vs official | `0.000061035156` | `2.1192763e-8` | `1` |

Local policy discriminator:

- Scratch PyTorch BF16 `torch.nn.functional.linear` with the same weighted-V
  input, o_proj weight, and o_proj bias reproduces the official o_proj oracle
  exactly.
- Therefore the remaining one-lane mismatch is a BF16 linear backend policy
  issue in the Rust replay path, not evidence of a weighted-V or tensor-source
  mismatch.

Conclusion:

- The validation-runtime path reaches the attention o_proj seam, but the current
  Rust BF16 linear replay does not yet reproduce the official/module BF16
  `F.linear` boundary exactly.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Either emit the validation weighted-V BF16 artifact and rerun this seam from
that source, or add a validation-only BF16 linear policy discriminator for
o_proj before proceeding to the attention residual add.

## Attention o_proj BF16 Linear Policy Discriminator

Submode added:

- `--mode attention-oproj-policy`

Why this slice was needed:

- The initial Rust o_proj replay had one tiny mismatch against the official
  o_proj-before-residual oracle at hidden lane `1587`.
- Scratch PyTorch BF16 `torch.nn.functional.linear` with the same weighted-V
  input, o_proj weight, and o_proj bias matched the official oracle exactly.
- That isolated the issue to Rust-side BF16 linear accumulation policy rather
  than the weighted-V input or o_proj tensor source.

Inputs for the discriminator:

| input | source |
| --- | --- |
| weighted V | official weighted-V oracle, already matched by weighted-V validation at BF16 boundary |
| o_proj weight/bias | scratch safetensors extraction from the restricted integration checkpoint |
| o_proj oracle | official attention o_proj before residual artifact |
| residual input | `layer0_attn_norm_input_f32` final token from the official attention-norm full-input artifact |
| residual oracle | official hidden after attention residual add before MLP artifact |

Bounded Rust variants tested:

| variant | policy | o_proj max diff | o_proj mismatches | residual max diff | residual mismatches |
| --- | --- | ---: | ---: | ---: | ---: |
| A current | BF16 input, BF16 weight, sequential f32 accumulation, BF16 bias add, BF16 output | `0.000061035156` | `1` | `0` | `0` |
| B f32 bias | BF16 input, BF16 weight, sequential f32 accumulation, f32 bias add, BF16 output | `0.000061035156` | `1` | `0` | `0` |
| C pre-bias round | BF16 input, BF16 weight, f32 accumulation, BF16-round pre-bias, BF16 bias add, BF16 output | `0.03125` | `858` | `0.0625` | `426` |
| D reverse accumulation | BF16 input, BF16 weight, reverse f32 accumulation, f32 bias add, BF16 output | `0.000061035156` | `1` | `0` | `0` |
| E chunked pairwise | BF16 input, BF16 weight, chunked pairwise f32 accumulation, f32 bias add, BF16 output | `0` | `0` | `0` | `0` |
| F f32 input/BF16 weight | f32 input values, BF16 weight, f32 accumulation, f32 bias add, BF16 output | `0.000061035156` | `1` | `0` | `0` |

Lane `1587` trace:

| variant | o_proj actual | o_proj expected | o_proj diff | residual actual | residual expected | residual diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A current | `-0.01171875` | `-0.011657715` | `0.000061035156` | `0.24023438` | `0.24023438` | `0` |
| E chunked pairwise | `-0.011657715` | `-0.011657715` | `0` | `0.24023438` | `0.24023438` | `0` |

Current classifications:

```text
attention_oproj_rust_policy_matches_oracle
attention_residual_after_rust_oproj_matches_oracle
```

Status JSON:

```text
/tmp/layer0_validation_attention_oproj_policy_status.json
```

Conclusion:

- A bounded Rust-side policy reproduces the official o_proj boundary exactly:
  chunked pairwise f32 accumulation over BF16 input/weight, f32 bias add, BF16
  output.
- The original one-lane sequential replay mismatch also washes out at the BF16
  attention residual add boundary.
- The discriminator is validation-only and does not change production runtime
  behavior.

Recommended next bounded step:

Use the identified validation-only o_proj policy to continue the clean
validation-runtime path through the attention residual add before MLP.

## Attention Residual Validation Status

Submode added:

- `--mode attention-residual`

Target seam:

```text
layer0_final_token_hidden_state_after_attention_residual_add_before_mlp
```

Inputs for the validation run:

| input | source |
| --- | --- |
| weighted V | official weighted-V oracle, already matched by weighted-V validation at BF16 boundary |
| o_proj weight/bias | scratch safetensors extraction from the restricted integration checkpoint |
| residual input | `layer0_attn_norm_input_f32` final token from `/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-layer0-k-consumption-20260423/developer-message.official-layer0-attn-norm-full-input.cpu.json` |
| attention residual oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-hidden-state-after-attention-residual-add-before-mlp-status.json` |

Policy:

- o_proj uses the validation-only `E_chunked_pairwise` policy identified by the
  BF16 linear discriminator:
  - BF16 input,
  - BF16 weight,
  - chunked pairwise f32 accumulation,
  - f32 bias add,
  - BF16 output.
- Residual add uses BF16 residual input plus BF16 o_proj output with BF16 output.

Current classification:

```text
layer0_validation_attention_residual_matches_oracle
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| o_proj guard vs official | `0` | `0` | `0` |
| attention residual before MLP vs official | `0` | `0` | `0` |

Conclusion:

- The validation-runtime path now clears the layer0 final-token attention
  residual-add boundary exactly using the chunked pairwise o_proj policy.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Extend the validation-runtime path to MLP norm.

## MLP Norm Validation Status

Submode added:

- `--mode mlp-norm`

Target seam:

```text
layer0_final_token_mlp_norm_output_before_mlp_projections
```

Inputs for the validation run:

| input | source |
| --- | --- |
| attention residual input | official attention-residual oracle, accepted because `--mode attention-residual` matched this boundary exactly |
| MLP norm weight | `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration/model-00000-of-00002.safetensors` |
| MLP norm tensor name | `model.layers.0.post_attention_layernorm.weight` |
| MLP norm oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json` |

Policy:

- Input is rounded to the BF16 attention-residual boundary.
- RMS reduction computes `mean(x*x)` in f32 over hidden size `2880`.
- Epsilon is `1e-5`.
- Operation order is `x * inverse_rms` then scale.
- Output is BF16.

Current classification:

```text
layer0_validation_mlp_norm_matches_oracle
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| MLP norm output before projections vs official | `0` | `0` | `0` |

Conclusion:

- The validation-runtime path now clears the layer0 final-token MLP norm
  boundary exactly after the exact attention residual boundary.
- The mode uses a narrow local safetensors reader for this validation tensor
  only; it does not change model-runner loading or production runtime behavior.

Recommended next bounded step:

Extend the validation-runtime path to router logits and top-k routing.

## Router And Top-K Validation Status

Submode added:

- `--mode router`

Target seams:

```text
layer0_final_token_mlp_router_logits_before_routing
layer0_final_token_mlp_topk_expert_indices_and_routing_weights
```

Inputs for the validation run:

| input | source |
| --- | --- |
| MLP norm input | official MLP norm oracle, accepted because `--mode mlp-norm` matched this boundary exactly |
| router weight | `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration/model-00000-of-00002.safetensors` |
| router weight tensor | `model.layers.0.mlp.router.weight`, BF16, shape `[32,2880]` |
| router bias | `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration/model-00000-of-00002.safetensors` |
| router bias tensor | `model.layers.0.mlp.router.bias`, BF16, shape `[32]` |
| router logits oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-router-logits-before-routing-status.json` |
| top-k/routing oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-topk-expert-indices-and-routing-weights-status.json` |

Policy:

- Router logits use BF16 input, BF16 weight, BF16 bias, f32 accumulation, and
  BF16 output.
- Top-k uses sorted descending selected logits with `top_k = 4`.
- Routing weights are softmax over selected logits with BF16 output.

Current classification:

```text
layer0_validation_router_and_topk_match_oracle
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| router logits vs official | `0` | `0` | `0` |
| selected expert logits vs official | `0` | `0` | `0` |
| routing weights vs official | `0` | `0` | `0` |

Top-k result:

| field | value |
| --- | --- |
| selected experts | `[3, 30, 11, 27]` |
| routing weights | `[0.4453125, 0.2275390625, 0.189453125, 0.13671875]` |
| routing weight sum | `0.99902344` |

Conclusion:

- The validation-runtime path now clears the layer0 final-token router logits
  and top-k/routing boundaries exactly after the exact MLP norm boundary.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Extend the validation-runtime path to selected expert outputs.

## Selected Expert Output Validation Status

Submode added:

- `--mode selected-experts`

Target seam:

```text
layer0_final_token_selected_expert_outputs_before_routing_weighted_sum
```

Selected expert rank order:

| rank | expert |
| ---: | ---: |
| 0 | `3` |
| 1 | `30` |
| 2 | `11` |
| 3 | `27` |

Inputs for the validation run:

| input | source |
| --- | --- |
| MLP norm input | official MLP norm oracle, accepted because `--mode mlp-norm` matched this boundary exactly |
| selected expert outputs oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-selected-expert-outputs-before-routing-weighted-sum-status.json` |
| expert tensor source | `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration/model-00000-of-00002.safetensors` |

Detected expert tensors:

| tensor | dtype | shape |
| --- | --- | --- |
| `model.layers.0.mlp.experts.gate_up_proj_blocks` | `U8` | `[32,5760,90,16]` |
| `model.layers.0.mlp.experts.gate_up_proj_scales` | `U8` | `[32,5760,90]` |
| `model.layers.0.mlp.experts.gate_up_proj_bias` | `BF16` | `[32,5760]` |
| `model.layers.0.mlp.experts.down_proj_blocks` | `U8` | `[32,2880,90,16]` |
| `model.layers.0.mlp.experts.down_proj_scales` | `U8` | `[32,2880,90]` |
| `model.layers.0.mlp.experts.down_proj_bias` | `BF16` | `[32,2880]` |

Expert formula recorded for the eventual replay:

- `mlp1` fused output shape is `[5760]`.
- Gate slice is `values[0::2]`; up/value slice is `values[1::2]`.
- Clamp policy is gate max `7.0`, up min `-7.0`, up max `7.0`.
- SwiGLU is `gate * sigmoid(1.702 * gate) * (up + 1)`.
- Selected expert output boundary is BF16.

Current classification:

```text
layer0_validation_selected_experts_blocked_by_mxfp4_loader_api
```

Metrics:

| comparison | result |
| --- | --- |
| selected expert outputs vs official | not run, blocked before replay |
| per-rank metrics | not run |
| expert30 fresh replay | not run |

Conclusion:

- The validation mode now validates the selected expert input/oracle paths and
  records the actual checkpoint expert tensor layout.
- The checkpoint stores expert weights as MXFP4 `U8` blocks/scales plus BF16
  biases. The bench path currently has no narrow validation dequantization and
  selected-expert replay API, so the mode stops before arithmetic rather than
  importing proof/debug plumbing or reusing stale selected-output artifacts.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Extract a narrow validation-only MXFP4 selected-expert loader/replay helper that
uses existing model-runner semantics, then rerun `--mode selected-experts` for
fresh experts `[3, 30, 11, 27]`.

## MXFP4 Selected Expert Validation Loader Status

Helper added:

- `gpt_oss_model_runner::mxfp4_validation::load_selected_experts_mxfp4_validation`

Scope:

- Validation-only model-runner helper.
- Loads only the requested layer0 expert tensors from the checkpoint.
- Uses the existing runtime `gpt_oss_dequant_expert_f16_kernel` to decode
  MXFP4 blocks/scales for selected experts.
- Returns dequantized f16 weights and BF16 biases widened to f32 for the bench
  replay.
- Does not modify production routing, default model-runner behavior, or CUDA
  kernel math.

Decode source:

| item | source |
| --- | --- |
| MXFP4 value/dequant semantics | `gpt_oss_dequant_expert_f16_kernel` |
| selected experts | `[3, 30, 11, 27]` |
| gate/up blocks/scales | `model.layers.0.mlp.experts.gate_up_proj_blocks/scales` |
| down blocks/scales | `model.layers.0.mlp.experts.down_proj_blocks/scales` |
| biases | `model.layers.0.mlp.experts.gate_up_proj_bias`, `model.layers.0.mlp.experts.down_proj_bias` |

Current classification:

```text
layer0_validation_selected_experts_mismatch_large
```

Metrics:

| comparison | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| selected expert outputs overall | `0.25` | `0.0030456155` | `8462` |
| rank 0 / expert 3 | `0.25` | `0.0034140944` | `2083` |
| rank 1 / expert 30 | `0.125` | `0.0027241078` | `2170` |
| rank 2 / expert 11 | `0.125` | `0.0029283462` | `2107` |
| rank 3 / expert 27 | `0.25` | `0.0031159138` | `2102` |

Conclusion:

- The previous MXFP4 loader/API blocker is resolved: the validation path now
  uses existing model-runner GPU dequant semantics and reaches selected expert
  arithmetic.
- The selected expert outputs do not yet match the official oracle. The broad
  mismatch across all four selected experts suggests the remaining issue is a
  replay boundary/policy difference after MXFP4 dequantization, not the stale
  expert30 selected-output artifact seen in the earlier scratch path.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Localize the selected-expert replay policy after MXFP4 dequantization: compare
MLP1 boundary, SwiGLU boundary, and down-projection/bias boundary against any
available official internal artifacts, and test bounded BF16/F16 accumulation
and boundary variants without changing production routing.

## Selected Expert MXFP4 Replay Localization

Debug mode added:

```text
--mode selected-experts-debug
```

Scope:

- Focuses on selected experts `[3, 30, 11, 27]`, with expert30 as the internal
  boundary probe because official MLP1/SwiGLU/MLP2-pre-bias artifacts are
  available for it.
- Reuses
  `gpt_oss_model_runner::mxfp4_validation::load_selected_experts_mxfp4_validation`.
- Emits finite summaries for decoded expert30 gate/up and down weights/biases,
  but does not write full decoded weights to the repo.
- Does not change production runtime behavior, routing, or CUDA kernels.

Root classification:

```text
selected_expert_mismatch_starts_at_mxfp4_dequant_or_mlp1
```

Expert30 internal-boundary metrics:

| boundary | max abs diff | mean abs diff | mismatches | first/worst lane |
| --- | ---: | ---: | ---: | --- |
| MLP1 before SwiGLU | `0.00390625` | `0.0000006781684` | `1` | lane `522` |
| SwiGLU before MLP2 | `0.03125` | `0.00045324018` | `1383` | worst lane `863` |
| MLP2 before bias | `0.125` | `0.0014393684` | `1748` | worst lane `1167` |
| selected output after bias | `0.125` | `0.0014289034` | `1585` | worst lane `1167` |

Bounded MLP1 variants:

| variant | result vs official expert30 MLP1 |
| --- | --- |
| current: BF16 input, f16-dequant weight widened to f32, f32 accumulation, BF16 bias/output | `1` mismatch, max `0.00390625` |
| BF16-rounded dequant weight | `1` mismatch, max `0.00390625` |
| f32 input | `1` mismatch, max `0.00390625` |
| f16 output | `4869` mismatches, max `0.03125` |

Bounded SwiGLU variants using official MLP1 as input:

| variant | result vs official expert30 SwiGLU |
| --- | --- |
| current BF16 inputs/output, interleaved gate/up | `1382` mismatches, max `0.03125` |
| BF16-round `gate * sigmoid` before multiplying by `up + 1` | `1133` mismatches, max `0.015625` |
| BF16-round sigmoid value | `1296` mismatches, max `0.03125` |
| swapped gate/up lane order | `2877` mismatches, max `13.749096` |

Interpretation:

- The first nonzero expert30 divergence appears at MLP1, but it is isolated to
  one BF16 lane.
- The broad mismatch starts at the SwiGLU boundary and remains broad through
  MLP2 and selected-output readout.
- Since the SwiGLU variants still mismatch even when fed the official MLP1
  boundary, the next slice should inspect the official SwiGLU dtype/operator
  policy before changing MXFP4 decode semantics.
- The swapped gate/up guard strongly argues against a simple gate/up lane-order
  reversal.

Recommended next bounded step:

Localize the official SwiGLU policy using the expert30 MLP1 oracle as input:
operator order, clamp/rounding boundaries, sigmoid implementation, and output
dtype. Keep MXFP4 dequant/layout changes deferred until SwiGLU policy is pinned.

## SwiGLU Dtype Policy Localization

Focused mode added:

```text
--mode swiglu-debug
```

Why this slice was needed:

- The previous selected-experts debug run showed only one BF16 lane mismatch at
  expert30 MLP1 before SwiGLU, but the mismatch expanded to 1383 lanes at the
  SwiGLU boundary.
- This mode removes MXFP4 and MLP1 replay from the equation by using the
  official expert30 MLP1-before-SwiGLU oracle as the sole input.

Official implementation summary from `gpt_oss/torch/model.py`:

```python
x_glu, x_linear = x[..., ::2], x[..., 1::2]
x_glu = x_glu.clamp(min=None, max=limit)
x_linear = x_linear.clamp(min=-limit, max=limit)
out_glu = x_glu * torch.sigmoid(alpha * x_glu)
return out_glu * (x_linear + 1)
```

Key official semantics:

- Split is interleaved gate/up, not half split.
- Gate clamp is max-only at `7.0`.
- Up/value clamp is `[-7.0, 7.0]`.
- Sigmoid scale is `1.702`.
- The official function has no explicit cast; the tensor expression runs on
  BF16 tensors and the official boundary is BF16.

Rust bounded-variant result:

```text
swiglu_dtype_rounding_policy_mismatch
```

Best Rust variant:

| variant | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| interleaved official clamp, BF16-round `1.702 * gate` before sigmoid, BF16 output | `0.03125` | `0.00037176925` | `1035` |

Important guards:

| guard | max abs diff | mean abs diff | mismatches |
| --- | ---: | ---: | ---: |
| interleaved official clamp, f32 sigmoid/multiply, BF16 output | `0.03125` | `0.00045188385` | `1382` |
| interleaved official clamp, BF16-round `gate * sigmoid`, BF16 output | `0.015625` | `0.00037546654` | `1133` |
| half-split gate/up | `5.486328` | `0.30293885` | `2877` |

External PyTorch BF16 discriminator:

- Using `/data/models/.venv-awq/bin/python`, the exact official tensor
  expression on a `torch.bfloat16` MLP1 tensor matched the official SwiGLU
  oracle exactly:

```text
max_abs_diff = 0
mean_abs_diff = 0
mismatches = 0
```

Interpretation:

- The split and clamp policy are pinned.
- The exact behavior is PyTorch BF16 elementwise sigmoid/multiply semantics, not
  the current Rust `f32::exp` approximation with BF16 boundary rounding.
- No selected-experts replay policy was changed in this slice, because encoding
  the PyTorch BF16 elementwise sigmoid behavior in Rust/CUDA needs a separate
  narrow validation helper rather than a Torch runtime dependency.
- Production runtime behavior remains unchanged.

Recommended next bounded step:

Add a validation-only SwiGLU helper that reproduces PyTorch BF16 elementwise
semantics, or temporarily use the official SwiGLU boundary as the MLP2 seam input
to localize down-projection independently. Do not change MXFP4 dequant/layout
semantics until this SwiGLU policy is resolved.

## Expert30 MLP2 From Official SwiGLU Status

Focused mode added:

```text
--mode expert30-mlp2-debug
```

Why official SwiGLU was used as seam input:

- `swiglu-debug` showed the official PyTorch BF16 tensor expression matches the
  official expert30 SwiGLU oracle exactly, while bounded Rust variants do not.
- Using the official SwiGLU boundary isolates the downstream expert30 MLP2/down
  projection and bias path from the unresolved SwiGLU elementwise policy.

Artifacts:

| boundary | artifact |
| --- | --- |
| expert30 SwiGLU input | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-expert30-swiglu-output-before-mlp2-status.json` |
| expert30 MLP2 pre-bias oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-expert30-mlp2-output-before-bias-status.json` |
| selected expert outputs oracle | `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-selected-expert-outputs-before-routing-weighted-sum-status.json` |

Down-projection source:

- Loader: `gpt_oss_model_runner::mxfp4_validation::load_selected_experts_mxfp4_validation`
- Decode source: `gpt_oss_dequant_expert_f16_kernel`
- Expert: `30`
- Down weight layout: `[out_hidden=2880, in_intermediate=2880]`, row-major
- Down bias: BF16 `[2880]`

Classification:

```text
expert30_mlp2_from_official_swiglu_matches_oracle
```

Best variant:

| variant | MLP2 pre-bias max | MLP2 pre-bias mismatches | selected-output max | selected-output mismatches |
| --- | ---: | ---: | ---: | ---: |
| `A_current` | `0` | `0` | `0` | `0` |

Other exact variants:

- `B_weight_bf16_round`
- `C_weight_f16`
- `D_f32_accum_bf16_output`
- `E_chunked_pairwise`
- `F1_bf16_prebias_bf16_bias`

Non-matching guard:

| variant | MLP2 pre-bias max | MLP2 pre-bias mismatches | selected-output max | selected-output mismatches |
| --- | ---: | ---: | ---: | ---: |
| `F2_f32_prebias_f32_bias` | `0.10264969` | `2880` | `0.03125` | `899` |

Weighted-sum replacement:

- Not run in this slice. Expert30 selected output now clears when fed the
  official SwiGLU seam, so weighted-sum replacement is a safe next validation
  step after deciding whether to use official SwiGLU as a temporary seam input.

Conclusion:

- Expert30 MLP2/down projection and bias replay are pinned when the input SwiGLU
  boundary is exact.
- The selected-expert replay blocker is upstream of MLP2: the official PyTorch
  BF16 SwiGLU elementwise policy still needs a validation-safe Rust/CUDA
  equivalent, or the validation path should temporarily use official SwiGLU as a
  seam input to continue weighted-sum localization.
- Production runtime behavior remains unchanged, and no raw artifacts are
  committed.

Recommended next bounded step:

Run weighted expert sum with official selected outputs for ranks `0,2,3` and
expert30 selected output produced from official SwiGLU through the validation
MLP2 path, or add a validation-only SwiGLU helper that reproduces PyTorch BF16
elementwise semantics exactly.

## SwiGLU BF16 Policy Pinning Status

Focused mode added:

```text
--mode swiglu-policy-pin
```

Why this slice was needed:

- `selected-experts-debug` showed expert30 MLP1 before SwiGLU was nearly pinned
  (`max_abs_diff = 0.00390625`, `mismatches = 1`), while the broad mismatch
  appeared at the SwiGLU boundary.
- `expert30-mlp2-debug` then showed MLP2/down projection and bias replay match
  exactly when the input is the official expert30 SwiGLU boundary.
- This left the BF16 elementwise SwiGLU policy as the active operator semantic
  to pin before changing MXFP4 dequant/layout assumptions.

PyTorch BF16 intermediate artifact:

```text
/tmp/layer0_validation_swiglu_pytorch_intermediates.json
```

The scratch artifact was generated outside the repo with
`/data/models/.venv-awq/bin/python` and is not committed. It uses the official
expert30 MLP1-before-SwiGLU oracle as input and runs the PyTorch BF16 tensor
expression:

```text
x_glu, x_linear = x[..., ::2], x[..., 1::2]
x_glu = x_glu.clamp(min=None, max=7.0)
x_linear = x_linear.clamp(min=-7.0, max=7.0)
out_glu = x_glu * torch.sigmoid(1.702 * x_glu)
out = out_glu * (x_linear + 1)
```

PyTorch BF16 discriminator:

| comparison | max_abs_diff | mean_abs_diff | mismatches |
| --- | ---: | ---: | ---: |
| PyTorch BF16 `swiglu_output` vs official SwiGLU | `0` | `0` | `0` |

Pinned Rust validation policy:

```text
I_torch_like_stage_rounding
```

Policy:

- Interleaved split: `x[..., ::2]` / `x[..., 1::2]`
- BF16-round input slices.
- Clamp gate to max `7.0`; clamp up/value to `[-7.0, 7.0]`.
- BF16-round `1.702 * gate`.
- Compute sigmoid with Rust `exp`, then BF16-round sigmoid output.
- BF16-round `gate * sigmoid`.
- BF16-round `up + 1`.
- BF16-round final multiply.

Stage-by-stage localization for the pinned Rust policy:

| stage | max_abs_diff | mean_abs_diff | mismatches |
| --- | ---: | ---: | ---: |
| gate raw | `0` | `0` | `0` |
| up raw | `0` | `0` | `0` |
| gate clamped | `0` | `0` | `0` |
| up clamped | `0` | `0` | `0` |
| alpha gate | `0` | `0` | `0` |
| sigmoid alpha gate | `0` | `0` | `0` |
| out_glu | `0` | `0` | `0` |
| up_plus_one | `0` | `0` | `0` |
| SwiGLU output | `0` | `0` | `0` |

Classification:

```text
swiglu_policy_matches_oracle
```

Selected-experts rerun after encoding the pinned policy:

```text
layer0_validation_selected_experts_mismatch_large
```

| metric | max_abs_diff | mean_abs_diff | mismatches |
| --- | ---: | ---: | ---: |
| selected expert outputs overall | `0.28125` | `0.002888643` | `8339` |

Per-rank selected-output metrics:

| rank / expert | max_abs_diff | mean_abs_diff | mismatches |
| --- | ---: | ---: | ---: |
| rank 0 / expert 3 | `0.28125` | `0.0033855785` | `2091` |
| rank 1 / expert 30 | `0.125` | `0.0026705414` | `2154` |
| rank 2 / expert 11 | `0.125` | `0.0029463163` | `2096` |
| rank 3 / expert 27 | `0.1875` | `0.0025521358` | `1998` |

Weighted-sum rerun:

- Not run. Selected experts did not clear end-to-end after encoding the pinned
  SwiGLU policy, so weighted sum remains deferred.

Conclusion:

- The PyTorch BF16 SwiGLU elementwise boundary is now encoded in the validation
  path and matches the official expert30 SwiGLU oracle exactly.
- Because selected-experts still mismatches broadly after the exact SwiGLU
  policy is encoded, the remaining end-to-end selected-expert blocker is no
  longer the standalone SwiGLU operator. The next localization should revisit
  MLP1/MXFP4 replay across all selected experts under the pinned SwiGLU policy,
  with expert30's one-lane MLP1 delta as the first concrete guardrail.
- Production runtime behavior remains unchanged, and no raw artifacts are
  committed.

Recommended next bounded step:

Rerun `selected-experts-debug` with the pinned SwiGLU policy and official
expert30 internal boundaries to localize the residual selected-expert mismatch
under the corrected SwiGLU semantics, then decide whether the blocker is MLP1
replay, MXFP4 dequant precision/layout, or selected-output artifact/readout.

## Selected Experts With Pinned SwiGLU Debug Status

Focused mode added:

```text
--mode selected-experts-pinned-swiglu-debug
```

Why this rerun was needed:

- `swiglu-policy-pin` proved the standalone PyTorch BF16 SwiGLU semantics can
  be reproduced exactly in Rust validation replay.
- Normal `selected-experts` still mismatched broadly after that pinned policy was
  encoded.
- This mode verifies the pinned policy is actually used by selected-experts
  replay and then localizes the first expert30 boundary that still mismatches.

Pinned policy usage:

| path | uses pinned `I_torch_like_stage_rounding` policy |
| --- | --- |
| normal `selected-experts` replay | `true` |
| `selected-experts-debug` replay | `true` |

Per-rank selected-output metrics:

| rank / expert | max_abs_diff | mean_abs_diff | mismatches |
| --- | ---: | ---: | ---: |
| rank 0 / expert 3 | `0.28125` | `0.0033855785` | `2091` |
| rank 1 / expert 30 | `0.125` | `0.0026705414` | `2154` |
| rank 2 / expert 11 | `0.125` | `0.0029463163` | `2096` |
| rank 3 / expert 27 | `0.1875` | `0.0025521358` | `1998` |

Expert30 focused variants:

| variant | MLP1 max/mismatches | SwiGLU max/mismatches | MLP2 pre-bias max/mismatches | selected output max/mismatches |
| --- | ---: | ---: | ---: | ---: |
| local MLP1 -> pinned SwiGLU -> local MLP2 | `0.00390625 / 1` | `0.0048828125 / 1` | `0.015625 / 262` | `0.015625 / 151` |
| official MLP1 -> pinned SwiGLU -> local MLP2 | `0 / 0` | `0 / 0` | `0 / 0` | `0 / 0` |
| local MLP1 -> official SwiGLU -> local MLP2 | `0.00390625 / 1` | `0 / 0` | `0 / 0` | `0 / 0` |
| official SwiGLU -> local MLP2 | n/a | `0 / 0` | `0 / 0` | `0 / 0` |

MLP1 replay variants for expert30:

| variant | max_abs_diff | mean_abs_diff | mismatches |
| --- | ---: | ---: | ---: |
| current | `0.00390625` | `6.781684e-7` | `1` |
| weight BF16-rounded | `0.00390625` | `6.781684e-7` | `1` |
| f32 input | `0.00390625` | `6.781684e-7` | `1` |
| output f16 | `0.03125` | `0.0020467665` | `4869` |

First mismatching boundary:

```text
expert30_mlp1_before_swiglu
```

Classification:

```text
selected_expert_mismatch_starts_at_mlp1_mxfp4_replay
```

Conclusion:

- The selected-experts replay is using the pinned SwiGLU policy.
- The standalone pinned policy is not the remaining blocker.
- Expert30 local replay first differs at MLP1 by one BF16 lane:
  lane `522`, local `0.33398438`, official `0.33007812`.
- Feeding official MLP1 through the pinned Rust SwiGLU and local MLP2 clears
  expert30 exactly, and feeding official SwiGLU through local MLP2 also clears.
- The next boundary to localize is expert30 MLP1/MXFP4 replay precision,
  accumulation, or dequant semantics. Non-expert30 ranks still have only
  selected-output metrics in this path, so their internal first mismatch is not
  yet localized.
- Production runtime behavior remains unchanged, and no raw artifacts are
  committed.

Recommended next bounded step:

Localize expert30 MLP1 by comparing the validation MXFP4 dequant + GEMM replay
against the official MLP1 oracle at lane `522`, including scale/block source,
decoded weights for that output lane, accumulation order, and BF16/f16 output
boundary.

## Expert30 MLP1 Lane 522 MXFP4 Replay Localization

This slice narrows the remaining selected-expert replay blocker to the single
expert30 MLP1 output lane that first differs before SwiGLU. Lane `522` is an
even fused gate/up lane, so under the official interleaved split it is gate lane
`261`. The row maps to expert `30`, output row `522`, and input dimension
`2880`; the MXFP4 checkpoint layout is `90` groups by `16` packed bytes, with
two 4-bit values per byte.

The focused validation mode is:

```bash
--mode expert30-mlp1-debug
```

It uses:

- MLP norm input:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json`
- Official expert30 MLP1 oracle:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-expert30-mlp1-output-before-swiglu-status.json`
- Checkpoint:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`

The mode reuses the validation MXFP4 row helper:

```text
gpt_oss_model_runner::mxfp4_validation::load_gate_up_row_mxfp4_validation
```

and keeps the runtime dequant source as the primary row:

```text
gpt_oss_dequant_expert_f16_kernel
```

It also emits row-local CPU guard variants derived from the visible kernel
formula:

- current kernel nibble/scale formula
- high/low nibble swap guard
- `exp2(scale_byte - 127)` scale interpretation guard
- BF16-rounded decoded-weight guard
- f16-rounded decoded-weight guard

It tests accumulation and output variants for lane `522` only:

- sequential f32 accumulation
- reverse f32 accumulation
- chunked/pairwise f32 accumulation
- MXFP4-group-aligned 32-wide blockwise accumulation
- f64 diagnostic accumulation
- BF16 pre-bias plus BF16 bias
- f16 output

The status JSON is written outside the repo:

```text
/tmp/layer0_validation_expert30_mlp1_lane522_status.json
```

This is still validation-only. Production runtime behavior, routing, and CUDA
kernels remain unchanged, and no raw `/tmp` or `.live` artifacts are committed.

Result:

```text
classification = expert30_mlp1_lane522_accumulation_unresolved

current full replay lane 522:
  local = 0.33398438
  official = 0.33007812
  diff = 0.00390625

best bounded variant:
  A_current_gpu_dequant_row
  local = 0.33203125
  official = 0.33007812
  diff = 0.001953125
```

Decode finding:

- The CPU mirror of the current kernel nibble/scale formula reproduces the
  runtime dequant row result.
- Swapping high/low nibbles is much worse (`diff = 0.45996094` at lane `522`).
- The `exp2(scale_byte - 127)` guard is equivalent for this row.
- BF16- and f16-rounded decoded-weight guards do not clear the lane.

Accumulation/output finding:

- Sequential, reverse, chunked/pairwise, MXFP4-group-aligned blockwise, and f64
  diagnostic accumulation all land at `0.33203125` after f32 bias add and BF16
  output.
- The current full replay policy, which rounds the pre-bias value to BF16 before
  adding BF16 bias, lands at `0.33398438`.
- No bounded decode, accumulation, bias, or output variant clears the official
  `0.33007812` value.

The next bounded target is source-contribution inspection for expert30 lane
`522`: compare per-group or per-term contributions against a PyTorch/official
MLP1 lane computation, without changing MXFP4 layout or production execution.

## Expert30 MLP1 Lane 522 Contribution Analysis

This slice compares the lane-local Rust validation replay with a scratch
PyTorch reference generated outside the repo:

```text
/tmp/layer0_validation_expert30_mlp1_lane522_pytorch_terms.json
```

The scratch reference used `/data/models/.venv-awq/bin/python` with
`PYTHONPATH=/home/emmy/openai/gpt-oss` and the official GPT-OSS MXFP4 decode
formula from `gpt_oss.torch.weights`. It does not add a Torch dependency to the
repo or runtime.

Source identity:

- Rust validation model path:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- Scratch PyTorch model path:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- The restricted model path symlinks model shards to
  `/data/models/openai/gpt-oss-20b`.
- Tensor names:
  `model.layers.0.mlp.experts.gate_up_proj_blocks`,
  `model.layers.0.mlp.experts.gate_up_proj_scales`,
  `model.layers.0.mlp.experts.gate_up_proj_bias`.

Lane `522` values:

```text
official output:
  0.33007812

Rust current full replay:
  pre_bias = 0.61132824
  bias = -0.27929688
  output = 0.33398438

Rust best explicit row-local variant:
  output = 0.33203125

PyTorch BF16 tensor reference:
  einsum pre_bias = 0.609375
  bias = -0.27929688
  einsum + bias output = 0.33007812
```

Per-block/per-term status:

- The Rust status emits per-block sums, absolute sums, and top local
  contributions in:
  `/tmp/layer0_validation_expert30_mlp1_lane522_terms_status.json`.
- The largest local absolute contribution is from hidden lane `1204`:
  input `1.6171875`, weight `-0.09375`, contribution `-0.15161133`.
- PyTorch terms confirm the source tensors and decode formula; the clearing
  difference is the BF16 `einsum` accumulation/source policy, not MXFP4 nibble
  ordering, scale interpretation, bias identity, or selected-output layout.

Classification:

```text
expert30_mlp1_lane522_accumulation_policy_mismatch
```

Conclusion:

The official lane is reproduced by PyTorch BF16 `einsum` for the same decoded
row and input, while Rust explicit product/sum variants do not reproduce it.
The next bounded step is to encode a validation-only BF16 `einsum`-style MLP1
policy or use a PyTorch/module-generated MLP1 seam while the exact Rust
accumulation semantics are implemented. Production runtime behavior remains
unchanged, and no raw `/tmp` or `.live` artifacts are committed.

## MLP1 BF16 Einsum-Style Validation Policy Status

This slice attempted to encode the observed PyTorch BF16 `einsum` boundary in a
narrow Rust-side validation replay for expert30 MLP1 lane `522`. The PyTorch
scratch artifact remains outside the repo:

```text
/tmp/layer0_validation_expert30_mlp1_lane522_pytorch_terms.json
```

Lane `522` reference:

```text
official output:
  0.330078125

PyTorch BF16 einsum:
  pre_bias = 0.609375
  bias = -0.279296875
  output = 0.330078125
```

Bounded Rust policies tested:

- `A_current_explicit_f32_sum`: output `0.33203125`, diff `0.001953125`.
- `B_bf16_product_f32_sum`: output `0.33398438`, diff `0.00390625`.
- `C_bf16_block32_partial_sum`: output `0.33789062`, diff `0.0078125`.
- `D_bf16_running_sum_each_term`: output `0.29882812`, diff `0.03125`.
- `F_f32_accum_bf16_prebias_f32_bias`: output `0.33398438`, diff `0.00390625`.
- `E_chunked_pairwise_{16,32,64,128}`: output `0.33203125`, diff
  `0.001953125`.

Classification:

```text
mlp1_bf16_einsum_policy_not_encoded
```

No bounded Rust product/sum, partial-sum, running-sum, or chunked variant
reproduced the PyTorch BF16 `einsum` lane. Because lane `522` did not clear,
the selected-experts rerun and weighted-expert-sum rerun were intentionally not
run in this slice:

```text
selected_experts_rerun:
  classification = not_run_policy_did_not_clear_lane522

weighted_sum_rerun:
  classification = layer0_validation_weighted_expert_sum_not_run
```

The next bounded step is to implement or expose a validation-only BF16
matmul/einsum backend that reproduces PyTorch BF16 dot/einsum semantics, or to
continue layer0 MLP validation through an official/PyTorch MLP1 seam while that
backend is developed. Production runtime behavior remains unchanged, no Torch
runtime dependency is added, and no raw `/tmp` or `.live` artifacts are
committed.

## Selected Experts From Official MLP1 Seams Status

This slice adds a validation-only seam mode that consumes official/PyTorch MLP1
outputs before SwiGLU for the selected experts instead of using the unresolved
Rust-native MLP1 BF16 `einsum` replay. The Rust-native MLP1 backend remains
open; this mode exists only to continue localizing downstream layer0 MLP seams.

The selected MLP1 seam pack was generated outside the repo:

```text
/tmp/layer0_validation_selected_experts_mlp1_seams-20260428-215204/
```

Source:

- model:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- MLP norm input:
  official MLP norm oracle, because `--mode mlp-norm` matched exactly
- operation:
  scratch PyTorch BF16 expert MLP1 `einsum` plus BF16 bias
- selected experts:
  `[3, 30, 11, 27]`

Downstream validation result:

```text
classification:
  selected_experts_from_mlp1_seams_mismatch

selected expert outputs:
  max_abs_diff = 0.001953125
  mean_abs_diff = 0.0000001695421
  mismatches = 1

per-rank:
  rank 0 / expert 3:
    max_abs_diff = 0.001953125
    mismatches = 1
  rank 1 / expert 30:
    exact
  rank 2 / expert 11:
    exact
  rank 3 / expert 27:
    exact

weighted expert sum:
  max_abs_diff = 0.0009765625
  mean_abs_diff = 0.0000003390842
  mismatches = 1

MLP residual:
  max_abs_diff = 0.001953125
  mean_abs_diff = 0.0000006781684
  mismatches = 1
```

The remaining downstream mismatch is isolated to rank `0` / expert `3`,
hidden lane `1990`. It propagates into weighted expert sum and MLP residual at
the same hidden lane. This is separate from the already-open Rust-native MLP1
BF16 `einsum` backend gap and points to a bounded expert3 MLP2/down-projection
or selected-output boundary check from exact MLP1 seam input.

No production behavior changed, no Torch runtime dependency was added, and no
raw `/tmp` or `.live` artifacts are committed. The next bounded step is either
to design the Rust-native BF16 `einsum` backend or to localize expert3 MLP2 from
official/PyTorch MLP1/SwiGLU seam input.

## Expert3 Lane 1990 Post-MLP1 Localization

This slice focuses on the single remaining selected-output mismatch from the
MLP1 seam continuation path:

```text
rank 0 / expert 3 / hidden lane 1990
```

Inputs:

- MLP1 seam:
  `/tmp/layer0_validation_selected_experts_mlp1_seams-20260428-215204/expert3_mlp1_before_swiglu.json`
- PyTorch discriminator:
  `/tmp/layer0_validation_expert3_lane1990_pytorch_debug.json`
- official selected-output, weighted-sum, and MLP residual oracles from
  `pinned-prompt-parity-official-reference-20260424`.

Rust and PyTorch agree at the focused lane:

```text
SwiGLU lane:
  rust = -0.1953125
  pytorch = -0.1953125

MLP2 pre-bias lane:
  rust = 0.48046875
  pytorch = 0.48046875

down bias lane:
  rust = -0.0016860962
  pytorch = -0.0016860962

selected output lane:
  rust = 0.478515625
  pytorch = 0.478515625
  official = 0.48046875
```

Selected-output metric for current Rust validation replay:

```text
max_abs_diff = 0.001953125
mean_abs_diff = 0.0000006781684
mismatches = 1
```

Bounded bias/output variants:

- Current Rust/PyTorch-style down bias add preserves the one-lane mismatch.
- Omitting down bias clears lane `1990` only, but mismatches `2842` lanes and
  is not a valid selected-output policy.
- F32-bias and BF16-prebias/BF16-bias variants are equivalent to current for
  this lane.

Classification:

```text
expert3_lane1990_bias_or_output_rounding_mismatch
```

Weighted-sum and MLP-residual impact were not rerun with a replacement because
no full expert3 selected-output variant cleared. The focused evidence points to
expert3 selected-output oracle semantics around down bias/output at lane `1990`,
not SwiGLU or MLP2 pre-bias. The Rust-native MLP1 BF16 `einsum` backend remains
open separately. No production behavior changed, no Torch runtime dependency
was added, and no raw `/tmp` or `.live` artifacts are committed.

## Expert3 Lane 1990 Selected-Output Oracle Semantics

This slice checks whether the isolated rank `0` / expert `3` / hidden lane
`1990` mismatch is a real down-bias policy issue or an oracle/readout anomaly.
No official expert3 internal MLP2 artifacts were found, so the check uses the
generated expert3 MLP1 seam plus the scratch PyTorch post-MLP1 discriminator as
source context.

Source identity:

- selected-output oracle:
  `/home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-selected-expert-outputs-before-routing-weighted-sum-status.json`
- selected-output oracle model metadata:
  `/data/models/openai/gpt-oss-20b`
- MLP1 seam:
  `/tmp/layer0_validation_selected_experts_mlp1_seams-20260428-215204/expert3_mlp1_before_swiglu.json`
- MLP1 seam model:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`

Lane window `1988..1992`:

```text
lane 1988:
  official selected = 0.10400390625
  pre_bias = 0.11767578125
  bias = -0.013671875
  post_bias = 0.10400390625
  official equals post-bias

lane 1989:
  official selected = 0.43359375
  pre_bias = 0.37109375
  bias = 0.0615234375
  post_bias = 0.43359375
  official equals post-bias

lane 1990:
  official selected = 0.48046875
  pre_bias = 0.48046875
  bias = -0.0016860962
  post_bias = 0.478515625
  official equals pre-bias, not post-bias

lane 1991:
  official selected = -0.4609375
  pre_bias = -0.453125
  bias = -0.007873535
  post_bias = -0.4609375
  official equals post-bias

lane 1992:
  official selected = -0.8203125
  pre_bias = -0.90625
  bias = 0.087890625
  post_bias = -0.8203125
  official equals post-bias
```

Selected-output metric for rank `0` / expert `3` remains isolated:

```text
max_abs_diff = 0.001953125
mean_abs_diff = 0.0000006781684
mismatches = 1
```

Downstream impact:

```text
weighted expert sum, original:
  max_abs_diff = 0.0009765625
  mean_abs_diff = 0.0000003390842
  mismatches = 1

weighted expert sum, one-lane corrected:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

MLP residual, original:
  max_abs_diff = 0.001953125
  mean_abs_diff = 0.0000006781684
  mismatches = 1

MLP residual, one-lane corrected:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0
```

Classification:

```text
expert3_lane1990_official_selected_output_anomaly
```

The best explanation is an isolated selected-output oracle/readout anomaly:
neighboring lanes match post-bias, lane `1990` alone equals pre-bias, and
replacing only that lane with the official selected-output value clears both
weighted expert sum and MLP residual. This is separate from the Rust-native
MLP1 BF16 `einsum` backend, which remains open. Production behavior is
unchanged, no Torch runtime dependency was added, and no raw `/tmp` or `.live`
artifacts are committed.

The next bounded step is to summarize the layer0 seam-mode path as exact modulo
the Rust-native MLP1 BF16 `einsum` backend gap and this isolated selected-output
oracle anomaly, then decide whether to pursue a validation BF16 `einsum`
backend or use official MLP1 seams for final layer0 seam reporting.

## Validation Commands

For the skeleton slice:

```bash
cargo fmt --package gpt-oss-bench
cargo check -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda
cargo run -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda -- --help
git diff --check
```

For the K RoPE API-boundary slice:

```bash
cargo run -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda -- \
  --mode k-rope \
  --k-pre-rope <official-k-pre-rope-json> \
  --k-post-rope-oracle <official-k-post-rope-json> \
  --output /tmp/layer0_validation_k_rope_status.json
```

For a future status run:

```bash
cargo run -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda -- \
  --mode skeleton \
  --layer0-input <layer0-input-json> \
  --official-layer0-output <official-layer0-output-json> \
  --output /tmp/layer0_validation_runtime_path_status.json
```

## Do-Not-Promote Boundaries

- No production runtime routing.
- No default model-runner behavior changes.
- No production CUDA kernel replacement.
- No oneDNN/Torch runtime dependency.
- No runtime-forward proof binary import.
- No debug capture plumbing import.
- No raw `.live` or `/tmp` artifacts committed.
- No all-layer, final-logit, or 4097 claims from this layer0-only path.
