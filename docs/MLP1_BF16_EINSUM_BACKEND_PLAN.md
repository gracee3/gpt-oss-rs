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

## Stage 2 Selected Experts Debug Status

This debug slice localized the selected-expert mismatch under the exact cuBLAS
BF16 MLP1 backend. It focused on rank 1 / expert30 because full expert30 MLP1
is already exact against the official MLP1-before-SwiGLU oracle.

Selected expert order/layout:

```text
expected order: [3, 30, 11, 27]
actual order:   [3, 30, 11, 27]
expert30 rank:  1
layout:         [rank, hidden]
rank1 maps to expert30: true
```

Expert30 boundary metrics:

```text
cuBLAS MLP1 vs official expert30 MLP1:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

pinned SwiGLU vs official expert30 SwiGLU:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

local MLP2 pre-bias vs official expert30 MLP2 pre-bias:
  max_abs_diff = 0.10264969
  mean_abs_diff = 0.0006558841
  mismatches = 2880

selected output vs official selected-output rank1/expert30:
  max_abs_diff = 0.03125
  mean_abs_diff = 0.00060357933
  mismatches = 899
```

Variant table summary:

```text
A. cuBLAS MLP1 -> pinned SwiGLU -> local MLP2
   MLP1 exact
   SwiGLU exact
   MLP2 pre-bias mismatches all 2880 lanes
   selected output mismatches 899 lanes

B. official MLP1 -> pinned SwiGLU -> local MLP2
   SwiGLU exact
   MLP2 pre-bias mismatches all 2880 lanes
   selected output mismatches 899 lanes

C. cuBLAS MLP1 -> official SwiGLU -> local MLP2
   MLP1 exact
   MLP2 pre-bias mismatches all 2880 lanes
   selected output mismatches 899 lanes

D. official SwiGLU -> local MLP2
   MLP2 pre-bias mismatches all 2880 lanes
   selected output mismatches 899 lanes
```

Classification:

```text
mlp1_bf16_backend_selected_experts_debug_mlp2_mismatch
```

First mismatching boundary:

```text
expert30_mlp2_pre_bias
```

Conclusion:

The Stage 2 broad selected-expert mismatch is not caused by the cuBLAS MLP1
backend, the pinned SwiGLU policy, or selected-output rank/order/layout for
expert30. Under both local and official SwiGLU inputs, this branch's local
MLP2/down replay diverges at expert30 MLP2 pre-bias.

Production runtime behavior did not change. This remains an isolated
validation/backend diagnostic. No raw `/tmp` or `.live` artifacts were
committed.

Next bounded step:

```text
Port or match the prior exact expert30 MLP2/down replay policy in this backend
branch, then rerun selected experts [3,30,11,27].
```

## Stage 2 MLP2 Policy Port Status

This slice ported the prior exact expert30 MLP2/down replay policy from
`projection/layer0-validation-runtime-path` into this backend branch. The
current branch mismatch was the same as the prior branch's negative guard:
MLP2 pre-bias was kept as f32 before bias/output, which mismatched all 2880
expert30 MLP2 lanes.

Prior exact branch reference:

```text
classification:
  expert30_mlp2_from_official_swiglu_matches_oracle

best variant:
  A_current

matching variants:
  A_current
  B_weight_bf16_round
  C_weight_f16
  D_f32_accum_bf16_output
  E_chunked_pairwise
  F1_bf16_prebias_bf16_bias

negative guard:
  F2_f32_prebias_f32_bias mismatched all lanes
```

Policy/layout difference found:

```text
old branch-local behavior:
  f32 MLP2 pre-bias plus f32 bias/output

ported validation behavior:
  BF16 input
  decoded down weight row-major [out_hidden, in_intermediate]
  f32 accumulation
  BF16-rounded pre-bias boundary
  BF16 bias add / BF16 selected-output boundary
```

Expert30 MLP2 metrics after the port, using official expert30 SwiGLU as input:

```text
MLP2 pre-bias:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0

selected output:
  max_abs_diff = 0
  mean_abs_diff = 0
  mismatches = 0
```

Variant table summary:

```text
A_current:
  MLP2 pre-bias exact
  selected output exact

B_weight_bf16_round:
  MLP2 pre-bias exact
  selected output exact

C_weight_f16:
  MLP2 pre-bias exact
  selected output exact

D_f32_accum_bf16_output:
  MLP2 pre-bias exact
  selected output exact

E_chunked_pairwise:
  MLP2 pre-bias exact
  selected output exact

F1_bf16_prebias_bf16_bias:
  MLP2 pre-bias exact
  selected output exact

F2_f32_prebias_f32_bias:
  MLP2 pre-bias max_abs_diff = 0.10264969
  MLP2 pre-bias mismatches = 2880
  selected output max_abs_diff = 0.03125
  selected output mismatches = 899
```

Selected-experts rerun with:

```text
MLP1:
  cuBLAS BF16 tensor-op

SwiGLU:
  pinned torch-like BF16 stage rounding

MLP2:
  ported BF16 pre-bias/output validation policy
```

Selected-output metric against the official selected-output oracle:

```text
max_abs_diff = 0.001953125
mean_abs_diff = 1.695421e-7
mismatches = 1
```

The remaining selected-output mismatch is the known expert3 lane 1990 oracle
anomaly:

```text
rank = 0
expert = 3
hidden lane = 1990
official selected = 0.48046875
validation post-bias selected = 0.478515625
```

With the known expert3 lane 1990 selected-output correction, selected expert
outputs are exact:

```text
max_abs_diff = 0
mean_abs_diff = 0
mismatches = 0
```

Per-rank selected-output status:

```text
rank 0 / expert 3:
  official oracle comparison has only the known lane 1990 mismatch

rank 1 / expert 30:
  exact

rank 2 / expert 11:
  exact

rank 3 / expert 27:
  exact
```

Weighted expert sum rerun:

```text
max_abs_diff = 0.0009765625
mean_abs_diff = 3.390842e-7
mismatches = 1
```

MLP residual rerun:

```text
max_abs_diff = 0.001953125
mean_abs_diff = 6.781684e-7
mismatches = 1
```

The weighted-sum and residual reruns still report the same one-lane downstream
impact because this branch's MLP2 policy status mode records the corrected
selected-output comparison but does not yet recompute weighted sum and residual
from an actually replaced rank0/expert3 lane 1990 value.

Classification:

```text
mlp1_bf16_backend_mlp2_policy_fixed_selected_experts_match
```

Conclusion:

The selected-expert backend path now has exact cuBLAS BF16 MLP1, exact pinned
SwiGLU, exact expert30 MLP2/down replay, and exact selected expert outputs
modulo the previously isolated expert3 lane 1990 selected-output oracle
anomaly. Production runtime behavior did not change. No production routing or
CUDA kernel changed.

Next bounded step:

```text
Recompute weighted expert sum and MLP residual with an actual one-lane
expert3 lane 1990 selected-output replacement in this backend branch, then
record whether the downstream weighted/residual path clears exactly.
```

## Stage 3 Corrected Downstream Status

This slice recomputed weighted expert sum and MLP residual from an actually
corrected selected-output matrix. The correction is the previously isolated
expert3 selected-output oracle anomaly at rank 0 / hidden lane 1990.

Backend path:

```text
MLP1:
  cuBLAS BF16 tensor-op

SwiGLU:
  pinned torch-like BF16 stage rounding

MLP2:
  BF16 pre-bias/output validation policy
```

Known correction:

```text
rank = 0
expert = 3
hidden lane = 1990
from validation post-bias selected = 0.478515625
to official selected-output oracle = 0.48046875
```

Uncorrected selected-output metric:

```text
max_abs_diff = 0.001953125
mean_abs_diff = 1.695421e-7
mismatches = 1
```

Corrected selected-output metric:

```text
max_abs_diff = 0
mean_abs_diff = 0
mismatches = 0
```

Uncorrected weighted expert sum metric:

```text
max_abs_diff = 0.0009765625
mean_abs_diff = 3.390842e-7
mismatches = 1
```

Corrected weighted expert sum metric:

```text
max_abs_diff = 0
mean_abs_diff = 0
mismatches = 0
```

Uncorrected MLP residual metric:

```text
max_abs_diff = 0.001953125
mean_abs_diff = 6.781684e-7
mismatches = 1
```

Corrected MLP residual metric:

```text
max_abs_diff = 0
mean_abs_diff = 0
mismatches = 0
```

Classification:

```text
mlp1_bf16_backend_corrected_downstream_mlp_residual_matches
```

Conclusion:

The layer0 final-token MLP backend validation path now clears through MLP
residual when using cuBLAS BF16 tensor-op for MLP1, the pinned BF16 SwiGLU
policy, the ported BF16 MLP2/down policy, and the isolated expert3 lane 1990
selected-output oracle correction. Production runtime behavior did not change.
No production routing or CUDA kernel changed.

Next bounded step:

```text
Summarize the BF16 backend branch result and prepare a narrow
validation-runtime handoff proposal. Do not route production MLP until
explicitly requested.
```

## BF16 MLP Backend Validation Summary

Milestone statement:

The validation/backend path clears layer0 selected experts through MLP residual
using:

```text
MLP1 / gate_up:
  cuBLAS BF16 tensor-op

SwiGLU:
  pinned torch-like BF16 stage rounding

MLP2 / down:
  BF16 pre-bias/output validation policy

Known oracle correction:
  rank 0 / expert 3 / hidden lane 1990
```

Evidence table:

| Seam | Result |
| --- | --- |
| expert30 lane 522 | exact |
| full expert30 MLP1 | exact |
| selected experts [3,30,11,27] | exact after expert3 lane 1990 correction |
| weighted expert sum | exact after expert3 lane 1990 correction |
| MLP residual | exact after expert3 lane 1990 correction |

Correction details:

```text
rank = 0
expert = 3
hidden lane = 1990
validation post-bias selected = 0.478515625
official selected-output oracle = 0.48046875
```

Caveats:

```text
validation-only
no production routing
no default model-runner behavior change
expert3 lane 1990 selected-output oracle anomaly remains documented
cuBLAS tensor-op is the selected validation candidate
cuBLAS pedantic also matched expert30 MLP1, but selected-experts used tensor-op
not all layers
not final logits
not 4097
not server/default runtime parity
```

Handoff proposal:

```text
proposed next branch:
  projection/layer0-validation-runtime-handoff

goal:
  integrate this backend candidate into layer0_validation_runtime_path behind
  explicit validation modes only
```

Proposed staged handoff:

```text
1. Port/enable cuBLAS BF16 MLP1 validation backend in layer0_validation_runtime_path.
2. Reuse pinned SwiGLU and BF16 MLP2 policy.
3. Reproduce selected experts / weighted sum / MLP residual in the validation-runtime branch.
4. Only after exact validation, consider a separate production-routing design doc.
```

Do-not-promote boundaries:

```text
no production routing from this branch
no raw artifacts
no proof harness wholesale import
no 4097
no final logits claim from this branch alone
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
