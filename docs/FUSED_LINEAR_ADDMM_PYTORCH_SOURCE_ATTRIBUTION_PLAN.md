# Fused Linear/AddMM PyTorch Source Attribution Plan

## Classification

```text
fused_linear_addmm_pytorch_source_attribution_plan_recorded
```

## Scope

This is a docs-only planning branch for a future PyTorch source/backend
attribution workspace. It does not clone PyTorch, create virtual environments,
build PyTorch, patch PyTorch, run probes, run consumer revalidation, modify
runtime code, or authorize implementation.

The future research workspace should live outside the `gpt-oss-rs` worktrees:

```text
/home/emmy/openai/pytorch
/home/emmy/openai/pytorch-research
/home/emmy/openai/.venvs/torch-wheel-attribution
/home/emmy/openai/.venvs/pytorch-src-cpu
```

Do not use one shared `/home/emmy/openai/.venv` for this lane:

- installed-wheel introspection and source editable builds should remain
  isolated
- source builds can contaminate import paths and oracle behavior
- current `gpt-oss-rs` worktrees should remain untouched

## Research Question

Can PyTorch/ATen source and dispatch attribution identify the exact CPU BF16
addmm implementation path behind the official Workstream A seam?

Official seam:

- CPU Torch module/F.linear/_C._nn.linear/addmm
- `torch.addmm(bias, input_2d, weight_t_2d)`
- operator: attention o-proj
- BF16 input
- BF16 weight
- BF16 bias
- fused bias before final observable BF16 output
- BF16 output
- sampled layers: 6, 10, 13, 16, 18, 21
- explicit matmul/einsum/unfused-bias remain negative controls

## Stage 0 — Workspace/Bootstrap Plan Only

This docs branch performs no clone/build/setup work. Future branches should
define and execute commands only after the workspace plan is accepted.

The `gpt-oss-rs` repository and all existing worktrees must remain in place.
PyTorch checkout/build state belongs under `/home/emmy/openai/pytorch`,
`/home/emmy/openai/pytorch-research`, and the dedicated virtual environments
listed above.

## Stage 1 — Installed Wheel Attribution

Future branch:

```text
oracle/fused-linear-addmm-torch-wheel-dispatch-attribution
```

Purpose: inspect the currently installed Torch wheel before any source build.

Required metadata:

- Python executable
- `sys.prefix`
- `torch.__version__`
- `torch.version.git_version`, if available
- `torch.__config__.show()`
- `torch.get_num_threads()`
- `torch.backends.mkldnn.enabled`
- `torch.cuda.is_available()`
- `cuda_used = false`
- import path for `torch`

Required dispatch probes if available:

- `torch._C._dispatch_dump_table("aten::addmm")`
- `torch._C._dispatch_dump_table("aten::linear")`
- `torch._C._dispatch_dump_table("aten::mm")`
- `torch._C._dispatch_dump_table("aten::matmul")`
- related dispatch/key/kernel registration introspection that is safe

Output:

```text
/tmp/fused_linear_addmm_torch_wheel_dispatch_attribution_status.json
```

Allowed classifications:

- `fused_linear_addmm_torch_wheel_dispatch_attribution_recorded`
- `fused_linear_addmm_torch_wheel_dispatch_backend_identified`
- `fused_linear_addmm_torch_wheel_dispatch_inconclusive`
- `fused_linear_addmm_torch_wheel_dispatch_blocked_by_missing_torch`
- `fused_linear_addmm_torch_wheel_dispatch_failed`

### Stage 1 Result

Implementation branch:

```text
oracle/fused-linear-addmm-torch-wheel-dispatch-attribution
```

Status:

```text
/tmp/fused_linear_addmm_torch_wheel_dispatch_attribution_status.json
```

Classification:

```text
fused_linear_addmm_torch_wheel_dispatch_attribution_recorded
```

Result:

- Selected Python executable:
  `/home/emmy/openai/gpt-oss/.venv/bin/python`.
- Torch version: `2.11.0+cu130`.
- Torch git version: `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
- Torch import path:
  `/home/emmy/openai/gpt-oss/.venv/lib/python3.13/site-packages/torch/__init__.py`.
- Dispatch dump tables were available for `aten::addmm`, `aten::linear`,
  `aten::mm`, and `aten::matmul`.
- Each target op showed CPU registration and MKLDNN/oneDNN registration
  signals, so the per-op inferred signal is `multiple_possible`.
- A tiny CPU BF16 `torch.addmm` sanity probe produced CPU BF16 output and the
  CPU profiler reported ATen-level `aten::addmm` activity.
- No PyTorch clone, PyTorch build, or source patch was performed.
- No new virtual environment was created.

Interpretation:

The installed wheel attribution captures useful dispatch/source-registration
hints for Stage 2 source mapping, but it does not identify a concrete active
CPU BF16 addmm backend or microkernel strongly enough to reopen Rust/CUDA
policy synthesis. Backend identity remains unresolved.

## Stage 2 — Source Checkout And Source Map

Future branch:

```text
oracle/fused-linear-addmm-pytorch-source-map
```

Purpose: clone PyTorch source into `/home/emmy/openai/pytorch` and checkout
the best matching tag or commit.

Rules:

- first discover installed Torch version and `torch.version.git_version`
- prefer the exact commit when `torch.version.git_version` is available
- otherwise checkout the matching release tag if available
- do not initialize submodules unless source reading requires it
- do not build yet

Source areas to inspect:

- `aten/src/ATen/native/Linear.cpp`
- `aten/src/ATen/native/native_functions.yaml`
- addmm/mm/matmul registration and dispatch files
- CPU/native addmm implementation files
- oneDNN/MKLDNN integration points
- BLAS/MKL fallback points
- generated ATen dispatch metadata if available

Output:

```text
/tmp/fused_linear_addmm_pytorch_source_map_status.json
```

## Stage 3 — Minimal Seam Reproducer

Future branch:

```text
oracle/fused-linear-addmm-pytorch-minimal-reproducer
```

Purpose: use captured Workstream A tensors, not full model loading, to
reproduce:

```text
torch.addmm(bias, input_2d, weight_t_2d)
```

Required layers:

```text
6, 10, 13, 16, 18, 21
```

Compare:

- official artifact
- `torch.addmm` fused bias
- zero-bias addmm plus bias
- explicit matmul/einsum/unfused-bias negative controls

Record:

- tensor metadata
- operation outputs
- dispatch/profiler metadata if useful
- whether the minimal reproducer still matches official artifacts

## Stage 4 — Optional CPU-Only Source Build

Future branch, only if Stages 1-3 justify it:

```text
oracle/fused-linear-addmm-pytorch-source-cpu-build
```

Rules:

- use `/home/emmy/openai/.venvs/pytorch-src-cpu`
- CPU-only first
- `USE_CUDA=0`
- no full model loading
- no GPU attribution
- no `gpt-oss-rs` runtime changes
- build only if source-level instrumentation is needed

## Stage 5 — Optional Instrumented Source Patch

Only if the CPU source build succeeds and source inspection identifies a likely
addmm path.

Purpose: add minimal print/log instrumentation to identify the active CPU BF16
addmm path.

Disallowed:

- broad PyTorch modifications
- GPU builds unless separately authorized
- model-scale runs
- changing `gpt-oss-rs` runtime behavior

## Success Criteria

Strong success:

- exact CPU BF16 addmm path and replayable arithmetic/kernel rule identified

Medium success:

- backend path identified but not replayable

Useful negative:

- behavior is backend/microkernel-defined and not safely portable to Rust/CUDA

Inconclusive:

- source/dispatch evidence insufficient

## Rust/CUDA Reopen Conditions

Reopen Rust/CUDA policy synthesis only if:

- a concrete replayable rule is identified
- the rule is global across sampled layers
- a new design explains why the prior bounded policy space missed it

Absent those conditions, preserve Workstream A as the official CPU Torch API
seam and do not proceed to CUDA mirror work.

## Guardrails

- Docs-only.
- No PyTorch clone in this branch.
- No virtual environment creation in this branch.
- No PyTorch build in this branch.
- No backend selected.
- No implementation authorized.
- No consumer revalidation authorized.
- No runtime/default/CUDA behavior change.
- No CUDA mirror.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency in Rust.

## Forward Python Environment Baseline Plan

Plan:

```text
docs/ORACLE_FORWARD_PYTHON_ENV_BASELINE_PLAN.md
```

Classification:

```text
oracle_forward_python_env_baseline_plan_recorded
```

The source-attribution lane should preserve `/home/emmy/openai/gpt-oss/.venv`
and `/data/models/.venv-awq` as historical/provenance environments. Future
oracle/source-attribution work should use a separately validated forward
environment, planned as `/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130`,
with PyTorch source work isolated under `/home/emmy/openai/pytorch*`.

This planning branch creates no virtual environment, installs no packages,
clones or builds no PyTorch source, reruns no oracle probes, and authorizes no
runtime/default/CUDA behavior changes.
