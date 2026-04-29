# MLP1 BF16 Backend Validation Summary

This branch validates a Rust/CUDA-side BF16 backend candidate for the layer0
final-token MLP path. It does not change production runtime behavior, default
model-runner routing, or CUDA kernels.

## Milestone

The validation/backend path clears layer0 selected experts through MLP residual
using:

```text
MLP1 / gate_up:
  cuBLAS BF16 tensor-op

SwiGLU:
  pinned torch-like BF16 stage rounding

MLP2 / down:
  BF16 pre-bias/output validation policy

Known correction:
  expert3 selected-output oracle lane 1990
```

## Evidence

| Boundary | Result |
| --- | --- |
| expert30 lane 522 | exact |
| full expert30 MLP1 | exact |
| selected experts [3,30,11,27] | exact after expert3 lane 1990 correction |
| weighted expert sum | exact after expert3 lane 1990 correction |
| MLP residual | exact after expert3 lane 1990 correction |

Correction:

```text
rank = 0
expert = 3
hidden lane = 1990
validation post-bias selected = 0.478515625
official selected-output oracle = 0.48046875
```

## Caveats

This is validation-only evidence.

It does not prove:

```text
production runtime routing
default model-runner behavior
all layers
all prompts
final logits
4097 behavior
server/default runtime parity
```

The expert3 lane 1990 selected-output oracle anomaly remains documented. cuBLAS
BF16 tensor-op is the selected validation candidate. cuBLAS BF16 pedantic also
matched expert30 MLP1, but the selected-experts path used tensor-op.

## Handoff Proposal

Proposed branch:

```text
projection/layer0-validation-runtime-handoff
```

Goal:

```text
Integrate the BF16 backend candidate into layer0_validation_runtime_path behind
explicit validation modes only.
```

Stages:

```text
1. Port/enable cuBLAS BF16 MLP1 validation backend in layer0_validation_runtime_path.
2. Reuse pinned SwiGLU and BF16 MLP2 policy.
3. Reproduce selected experts, weighted expert sum, and MLP residual.
4. After exact validation, consider a separate production-routing design doc.
```

Do not promote from this branch directly into production routing. Do not import
raw artifacts, proof harnesses, Torch runtime dependencies, final-logit claims,
or 4097 work.
