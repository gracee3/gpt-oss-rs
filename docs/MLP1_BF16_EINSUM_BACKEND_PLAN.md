# MLP1 BF16 Einsum Backend Plan

This branch is for designing a Rust-native BF16 MLP1 matmul/einsum backend for
the layer0 validation-runtime path. It starts from the layer0 seam-mode
checkpoint where downstream seams can be validated exactly when supplied with
official/PyTorch MLP1 seams.

## Milestone Context

The layer0 seam-mode validation path has reached a clean checkpoint:

- layer0 final-token downstream seams are exact when supplied with
  official/PyTorch MLP1 seams
- expert3 lane `1990` selected-output oracle anomaly is isolated
- Rust-native MLP1 BF16 `einsum` backend remains open
- production runtime routing remains unchanged

The open task is the BF16 MLP1 matmul/einsum backend. It is not RoPE, SwiGLU,
MLP2/down projection, selected-output layout, routing, weighted sum, or MLP
residual.

## Tight Repro

Use expert30 MLP1 lane `522` as the microbench.

Known values:

```text
official output:
  0.330078125

PyTorch BF16 einsum:
  pre_bias = 0.609375
  bias = -0.279296875
  output = 0.330078125

best bounded Rust explicit product/sum:
  output = 0.33203125
```

Prior conclusion:

- PyTorch BF16 `einsum` reproduces official.
- Rust explicit product/sum variants do not.
- The gap is a Rust/CUDA BF16 MLP1 matmul/einsum backend task.

## Non-Goals

- No production runtime routing changes.
- No default model-runner behavior changes.
- No Torch runtime dependency.
- No runtime-forward proof binary import.
- No debug capture plumbing import.
- No raw `.live` or `/tmp` artifacts committed.
- No 4097-token work.
- No all-layer, final-logit, or server/default runtime parity claims.

## Backend Candidates

1. cuBLAS BF16 GEMM validation microbench.

   Compare expert30 lane `522`, full expert30 MLP1, then selected experts
   `[3, 30, 11, 27]` against the PyTorch BF16 `einsum`/official references.

2. cuBLAS pedantic BF16 validation microbench.

   Use as a controlled variant if normal cuBLAS BF16 does not match PyTorch
   BF16 `einsum` semantics.

3. CUTLASS/custom CUDA kernel exploration.

   Keep this validation-only until selected experts clear. Prefer a small
   microbench kernel over broad runtime integration.

4. CPU reference only as diagnostic.

   CPU code may help isolate grouping/rounding behavior, but it is not a
   runtime backend target.

## Staged Plan

### Stage 1: Lane 522 and Full Expert30 MLP1

- Reproduce expert30 MLP1 lane `522`.
- Compare backend variants to:
  - official lane output
  - PyTorch BF16 `einsum` pre-bias/output
- If lane `522` clears, run full expert30 MLP1 `[5760]`.

### Stage 2: Selected Experts

- Apply the candidate backend to selected experts `[3, 30, 11, 27]`.
- Preserve the pinned SwiGLU policy.
- Preserve the existing MXFP4 loader/dequant semantics.
- Compare selected expert outputs before routing-weighted sum.

### Stage 3: Weighted Expert Sum and MLP Residual

- If selected experts clear, rerun weighted expert sum.
- Rerun MLP residual / layer0 final-token output.
- Keep the expert3 lane `1990` selected-output oracle anomaly explicitly
  documented.

### Stage 4: Runtime Routing Consideration

- Only after selected experts, weighted sum, and MLP residual clear through the
  Rust/CUDA backend should production routing be considered.
- Runtime routing changes are out of scope for this branch until explicitly
  requested.

## Stage 1 Lane 522 Microbench Status

Status JSON:

```text
/tmp/mlp1_bf16_einsum_backend_lane522_status.json
```

Classification:

```text
mlp1_bf16_backend_candidate_matches_pytorch_lane522
```

Source identity:

- Model: `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- Tensor names:
  - `model.layers.0.mlp.experts.gate_up_proj_blocks`
  - `model.layers.0.mlp.experts.gate_up_proj_scales`
  - `model.layers.0.mlp.experts.gate_up_proj_bias`
- Row loader:
  `gpt_oss_model_runner::mxfp4_validation::load_gate_up_row_mxfp4_validation`
- Decode source:
  `gpt_oss_dequant_expert_f16_kernel` plus row-local CPU guard variants

Official/PyTorch reference:

```text
official output = 0.330078125
PyTorch BF16 einsum pre_bias = 0.609375
bias = -0.279296875
PyTorch BF16 einsum output = 0.330078125
```

Scalar baseline summary:

```text
A_current_explicit_f32_sum:
  output = 0.33203125
  diff = 0.001953125

B_bf16_product_f32_sum:
  output = 0.333984375
  diff = 0.00390625

C_bf16_block32_partial_sum:
  output = 0.337890625
  diff = 0.0078125

D_bf16_running_sum_each_term:
  output = 0.298828125
  diff = 0.03125

E_chunked_pairwise_16/32/64/128:
  output = 0.33203125
  diff = 0.001953125

F_f32_accum_bf16_prebias_f32_bias:
  output = 0.333984375
  diff = 0.00390625
```

Backend candidate summary:

```text
cuBLAS BF16 tensor-op:
  pre_bias = 0.609375
  output = 0.330078125
  diff = 0

cuBLAS BF16 pedantic/no-tensor-op:
  pre_bias = 0.609375
  output = 0.330078125
  diff = 0

existing bf16_linear_bias_validation mode 0:
  output = 0.33203125
  diff = 0.001953125

existing bf16_linear_bias_validation mode 1:
  output = 0.333984375
  diff = 0.00390625

CUTLASS/custom CUDA:
  not imported in this slice
```

Best candidate:

```text
cuBLAS BF16 tensor-op
```

The cuBLAS BF16 tensor-op path and the scoped cuBLAS BF16
pedantic/no-tensor-op path both reproduce the PyTorch BF16 `einsum` and
official expert30 lane `522` value exactly. The scalar explicit product/sum
baselines and the existing validation BF16 linear kernel retain the known lane
gap.

Production runtime behavior did not change. This microbench is a new isolated
validation binary only.

Next bounded step:

```text
Run full expert30 MLP1 with the matching cuBLAS BF16 backend candidate.
```

## Validation Commands

Start each slice with:

```bash
pwd
git branch --show-current
git status --short
git log --oneline -6
```

Expected branch:

```text
projection/mlp1-bf16-einsum-backend
```

Baseline checks:

```bash
cargo fmt --package gpt-oss-model-runner --package gpt-oss-bench
cargo check -p gpt-oss-model-runner --features cuda
cargo check -p gpt-oss-bench --bin layer0_validation_runtime_path --features cuda
git diff --check
```

Use `/tmp` for generated diagnostic artifacts. Do not commit raw tensor dumps,
`.live` artifacts, or scratch JSON with raw values.
