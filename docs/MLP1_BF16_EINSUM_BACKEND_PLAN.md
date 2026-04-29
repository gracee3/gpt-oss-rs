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

## Stage 1 Full Expert30 MLP1 Status

Status JSON:

```text
/tmp/mlp1_bf16_einsum_backend_expert30_status.json
```

Classification:

```text
mlp1_bf16_backend_expert30_matches_oracle
```

Lane `522` result recap:

```text
official lane 522 = 0.330078125
scalar baseline lane 522 = 0.33203125
cuBLAS BF16 tensor-op lane 522 = 0.330078125
cuBLAS BF16 pedantic lane 522 = 0.330078125
```

Full expert30 MLP1 metrics:

```text
scalar baseline:
  max_abs_diff = 0.0625
  mean_abs_diff = 0.0014097106
  mismatches = 1791

cuBLAS BF16 tensor-op:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

cuBLAS BF16 pedantic/no-tensor-op:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0
```

Gate/up parity:

```text
scalar baseline:
  even gate lanes:
    max_abs_diff = 0.0625
    mean_abs_diff = 0.0012783934
    mismatches = 888
  odd up lanes:
    max_abs_diff = 0.03125
    mean_abs_diff = 0.0015410278
    mismatches = 903

cuBLAS BF16 tensor-op:
  even gate lanes exact
  odd up lanes exact

cuBLAS BF16 pedantic/no-tensor-op:
  even gate lanes exact
  odd up lanes exact
```

Best backend:

```text
cuBLAS BF16 tensor-op
```

The lane `522` cuBLAS result generalizes to the full expert30 MLP1 output
before SwiGLU. Both normal cuBLAS BF16 tensor-op and the scoped
pedantic/no-tensor-op variant reproduce the official `[5760]` oracle exactly.
The scalar Rust explicit replay remains mismatched and is now a diagnostic
contrast, not the candidate backend.

Production runtime behavior did not change. The full expert30 path is still an
isolated validation/backend microbench.

Next bounded step:

```text
Run selected experts [3,30,11,27] through MLP1 with the matching cuBLAS BF16
backend, then reuse the pinned Rust SwiGLU and existing MLP2/down projection
validation path.
```

## Stage 2 Selected Experts Status

Status JSON:

```text
/tmp/mlp1_bf16_einsum_backend_selected_experts_status.json
```

Classification:

```text
mlp1_bf16_backend_selected_experts_mismatch
```

Selected experts:

```text
[3, 30, 11, 27]
```

Backend:

```text
cuBLAS BF16 tensor-op for MLP1/gate_up
pinned Rust SwiGLU policy
existing validation MLP2/down replay
```

Selected-output metric vs official oracle:

```text
max_abs_diff = 0.0625
mean_abs_diff = 0.0007102039
mismatches = 3295
```

Per-rank selected-output metrics:

```text
rank 0 / expert 3:
  max_abs_diff = 0.0625
  mean_abs_diff = 0.00075478986
  mismatches = 790

rank 1 / expert 30:
  max_abs_diff = 0.03125
  mean_abs_diff = 0.00060357933
  mismatches = 899

rank 2 / expert 11:
  max_abs_diff = 0.0625
  mean_abs_diff = 0.0007105877
  mismatches = 779

rank 3 / expert 27:
  max_abs_diff = 0.0625
  mean_abs_diff = 0.00077185896
  mismatches = 827
```

Expert3 lane `1990` anomaly handling:

```text
not applied
official selected = 0.48046875
local post-bias selected = 0.48046875
```

The known expert3 lane `1990` selected-output oracle anomaly is not the active
issue in this run. The selected-output mismatch is broad across all four
selected experts.

Weighted expert sum:

```text
max_abs_diff = 0.0625
mean_abs_diff = 0.0005672084
mismatches = 1041
```

MLP residual:

```text
max_abs_diff = 0.0625
mean_abs_diff = 0.00054486364
mismatches = 516
```

Conclusion:

The full expert30 MLP1 result generalized for the MLP1 backend itself, but the
end-to-end selected expert path still mismatches after applying pinned SwiGLU
and the existing MLP2/down replay. This points the next slice at downstream
selected-expert localization under exact cuBLAS MLP1, not at the MLP1 BF16
backend candidate.

Production runtime behavior did not change. The selected-experts mode remains
an isolated validation/backend microbench.

Next bounded step:

```text
Localize selected-expert mismatch under cuBLAS MLP1: compare per-expert
SwiGLU and MLP2/down boundaries, starting with expert30 because its MLP1 is
known exact.
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
