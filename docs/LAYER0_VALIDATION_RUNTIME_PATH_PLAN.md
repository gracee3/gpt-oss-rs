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
