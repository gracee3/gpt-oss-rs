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
