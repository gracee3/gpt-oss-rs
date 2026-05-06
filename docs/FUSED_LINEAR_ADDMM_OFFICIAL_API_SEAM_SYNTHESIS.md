# Fused Linear/AddMM Official API Seam Synthesis

## Classification

```text
fused_linear_addmm_official_api_seam_synthesis_recorded
```

## Scope

This is a docs-only synthesis and decision record for the official
fused-linear/addmm attention o-proj API seam. It covers the
`developer-message-user-smoke` final-token ordered-surface evidence for sampled
layers 6, 10, 13, 16, 18, and 21.

This record does not authorize implementation, backend selection, consumer
revalidation, runtime/default routing changes, CUDA kernel changes, output
emission, or ladder continuation.

## Evidence Chain

The evidence chain now includes:

- Producer/API final matrix:
  `docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md`
- Backend/helper candidate exhaustion:
  `docs/FUSED_LINEAR_ADDMM_BACKEND_DISCRIMINATOR_DESIGN.md`
- cuBLASLt fused-bias prototype failure:
  `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- CPU producer attribution:
  `/tmp/fused_linear_addmm_cpu_producer_attribution_status.json`
- AddMM boundary localization:
  `/tmp/fused_linear_addmm_addmm_boundary_localization_status.json`
- Fused-bias arithmetic contract:
  `/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json`

The latest arithmetic-contract classification is:

```text
fused_linear_addmm_fused_bias_arithmetic_contract_inconclusive
```

## Strongest Current Conclusion

The strongest current conclusion is that the official reference for the sampled
attention o-proj seam is the CPU Torch API seam:

- module call
- `torch.nn.functional.linear`
- `torch._C._nn.linear`
- `torch.addmm(bias, input, weight.T)`

The observed reference uses BF16 input, BF16 weight, and BF16 bias. The strongest
arithmetic signal is fused bias before the final observable BF16 output.

Full-vector exactness is required. Focus-lane clears are diagnostic only.

The following remain negative controls:

- explicit matmul plus bias
- explicit einsum plus bias
- zero-bias addmm plus separate bias
- explicit unfused BF16 bias

## Unresolved

The following remain unresolved:

- concrete CPU backend identity
- one global accumulation/product policy
- a Rust/CUDA helper candidate that clears layers 6, 10, 13, 16, 18, and 21
- runtime promotion path

The arithmetic-contract probe supports bias-before-output-rounding, but no
explicit arithmetic variant localizes the exact accumulation/product policy
across the full sampled set.

## Why Blind Sweeps Should Stop

Further blind sweeps are lower value because:

- existing local helper families are exhausted for this evidence set
- cuBLAS/cuBLASLt candidates did not clear the sampled producer/API reference
- arithmetic variants are partial and layer-sensitive
- focus-lane clears remain diagnostic only and cannot select a policy
- explicit matmul/einsum/unfused-bias forms are negative controls, not
  alternative official references

## Decision

Preserve Workstream A as an official Torch API seam for now.

Do not implement runtime o-proj changes from this evidence. Do not run consumer
revalidation from this evidence. If future validation needs this boundary, use
producer/API artifacts as oracle seams rather than promoting local arithmetic
policies.

## Future Options

### Option A — Preserve And Move On

Preserve this synthesis and move to the next blocker or workstream.

### Option B — Producer/API Artifact Reuse Path

Design a validation-only path that reuses producer/API artifacts as oracle seam
inputs. This would be an artifact-consumption design, not runtime behavior.

### Option C — Custom Validation Helper Later

Only later design a custom validation helper if a global arithmetic policy is
discovered and proves full-vector safe across blocked layers and controls.

### Option D — GPU/Sharded Torch Oracle Later

Revisit GPU or sharded Torch oracle generation later if needed. For single-GPU
work, use GPU1 because displays are on GPU0, and consult the multi-GPU sharding
lane before any sharded run.

## Rust/CUDA Policy Feasibility Plan

Follow-up plan:

```text
docs/FUSED_LINEAR_ADDMM_RUST_CUDA_POLICY_FEASIBILITY_PLAN.md
```

Classification:

```text
fused_linear_addmm_rust_cuda_policy_feasibility_plan_recorded
```

The plan defines three gates before any policy implementation discussion:

- Gate A: CPU Torch dispatch-stability.
- Gate B: bounded Rust CPU policy synthesis, only if Gate A is stable.
- Gate C: CUDA mirror, only if one Rust CPU policy clears the sampled set.

The plan keeps the official API seam as the reference and explicitly rejects
per-layer policy selection, per-lane policy selection, focus-lane promotion,
tolerance passes, f64 diagnostic promotion, and runtime backend promotion from
this evidence.

## Gate A Dispatch-Stability Result

Status:

```text
/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json
```

Classification:

```text
fused_linear_addmm_cpu_dispatch_stability_stable
```

The official CPU Torch addmm seam is stable across the tested fresh-process CPU
thread/backend settings for layers 6, 10, 13, 16, 18, and 21. No configuration
changed the full-vector addmm output relative to baseline, and every
configuration matched the official o-proj artifacts exactly. Baseline
matmul/einsum/unfused-bias controls remained negative.

This strengthens the official API seam as a stable oracle reference, but it
does not select a backend or authorize Rust CPU policy synthesis, consumer
revalidation, runtime/default/CUDA changes, output emission, or ladder
continuation.

## Gate B Rust CPU Policy Synthesis Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_synthesis_partial_only
```

The bounded Rust CPU search did not find one selectable arithmetic policy that
clears all sampled o-proj full vectors for layers 6, 10, 13, 16, 18, and 21.
Some per-layer candidates clear layers 10, 13, or 21, but the sampled-set gate
fails because layers 6, 16, and 18 retain full-vector mismatches. Focus-lane
clears, diagnostic-only candidates, and evidence-only candidates remain
non-promotable.

This preserves the official CPU Torch module/F.linear/_C/addmm seam as the
current Workstream A oracle boundary. It does not authorize Gate C CUDA mirror
work, consumer revalidation, runtime/default/CUDA behavior changes, output
emission, or ladder continuation.

## Gate B Closure Audit Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_closure_no_global_policy
```

The validation-only closure audit replayed all 238 previously bounded
focus-clearing selectable candidates full-vector. No single candidate cleared
the sampled layers 6, 10, 13, 16, 18, and 21 as a global policy after closure.

The top near-global candidates still leave residual mismatches on the sampled
set. Residual samples in layers 6, 16, and 18 are within one BF16 ULP in the
simple check, but the audit did not localize a shared rounding/tie rule that
justifies another narrow policy branch.

The recommended state is `stop_policy_lane_preserve_official_api_seam`. This
reinforces the synthesis decision: preserve Workstream A as an official CPU
Torch API seam, not a Rust/CUDA backend identity. No backend is selected, no
implementation is authorized, no consumer revalidation is authorized, and no
runtime/default/CUDA behavior change follows.

## PyTorch Source Attribution Plan

Plan:

```text
docs/FUSED_LINEAR_ADDMM_PYTORCH_SOURCE_ATTRIBUTION_PLAN.md
```

Classification:

```text
fused_linear_addmm_pytorch_source_attribution_plan_recorded
```

The next source-attribution option is to inspect the installed Torch wheel,
then map the matching PyTorch/ATen source path for `aten::addmm` and related
linear/mm/matmul registrations. The plan keeps this work outside the current
`gpt-oss-rs` worktrees and requires isolated virtual environments. It does not
clone or build PyTorch in this branch, and it does not authorize implementation
or consumer revalidation.

## Torch Wheel Dispatch Attribution Result

Status:

```text
/tmp/fused_linear_addmm_torch_wheel_dispatch_attribution_status.json
```

Classification:

```text
fused_linear_addmm_torch_wheel_dispatch_attribution_recorded
```

The installed Torch wheel dispatch probe selected the existing oracle Python
environment and captured dispatch tables for `aten::addmm`, `aten::linear`,
`aten::mm`, and `aten::matmul`. The tables provide CPU and MKLDNN/oneDNN
registration signals for all four ops, while a tiny CPU BF16 addmm sanity case
confirmed CPU BF16 output and ATen-level `aten::addmm` profiler activity.

This supports Stage 2 source mapping, but it does not identify the concrete
active CPU BF16 addmm backend or authorize Rust/CUDA policy work. No PyTorch
clone/build/source patch was performed.

## Forward Python Environment Baseline Plan

Plan:

```text
docs/ORACLE_FORWARD_PYTHON_ENV_BASELINE_PLAN.md
```

Classification:

```text
oracle_forward_python_env_baseline_plan_recorded
```

The official API seam synthesis now distinguishes historical/provenance
environments from future oracle/source-attribution environments. The recorded
Torch `2.11.0+cu130` wheel in `/home/emmy/openai/gpt-oss/.venv` remains tied
to prior artifacts; future work should validate a separate Python 3.12 forward
baseline before producing new oracle evidence.

This is docs-only and does not create a venv, install packages, clone/build
PyTorch, rerun probes, or authorize implementation.

## Forward Python Environment Baseline Result

Status:

```text
/tmp/oracle_forward_python_env_baseline_status.json
```

Classification:

```text
oracle_forward_python_env_baseline_validated
```

The forward baseline now exists as a uv-managed Python 3.12.12 environment at
`/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130`, with Torch
`2.11.0+cu130` and CPU-only tiny BF16 addmm validation. Historical artifacts
remain tied to their recorded environments; the new forward environment does
not silently replace prior exactness evidence.

No PyTorch clone/build/patch, model loading, Workstream A artifact rerun,
consumer revalidation, or cross-env artifact comparison was performed.

## Forward Environment Smoke

Status:

```text
/tmp/fused_linear_addmm_forward_env_smoke_status.json
```

Classification:

```text
fused_linear_addmm_forward_env_smoke_matched
```

The uv-managed Python 3.12.12 / Torch `2.11.0+cu130` forward environment
matched the existing official attention o-proj artifacts for required layers 6
and 18 using CPU `torch.addmm(bias, input_2d, weight_t_2d)`. Optional layer10
also matched. The same forward-env run preserved the known negative controls:
zero-bias addmm plus separate bias, explicit matmul plus bias, and explicit
einsum plus bias did not clear full-vector exactness.

This supports the official API seam as portable across the historical and
forward Torch `2.11.0+cu130` environments for the smoke subset only. It is not
a full rebaseline and does not authorize implementation, consumer
revalidation, backend selection, output emission, ladder continuation, or
runtime/default/CUDA behavior changes.

## PyTorch Source Map

Status:

```text
/tmp/fused_linear_addmm_pytorch_source_map_status.json
```

Classification:

```text
fused_linear_addmm_pytorch_source_map_exact_commit_mapped
```

PyTorch source was checked out at the exact Torch wheel git commit
`70d99e998b4955e0049d13a98d77ae1b14db1f45`. The source map confirms that
2D biased `aten::linear` routes to `at::addmm(*bias, input, weight.t())`, and
that `aten::addmm` maps to CPU `addmm_out_cpu` / `addmm_impl_cpu_` with
additional MKLDNN/oneDNN BF16 matmul candidates visible in source.

This strengthens the official CPU Torch API seam attribution, but it still
does not expose a single replayable BF16 arithmetic or microkernel rule. Rust
policy synthesis remains closed, `reopen_rust_policy_synthesis = false`, and
no backend is selected or authorized for implementation.

## PyTorch Minimal Reproducer

Status:

```text
/tmp/fused_linear_addmm_pytorch_minimal_reproducer_status.json
```

Classification:

```text
fused_linear_addmm_pytorch_minimal_reproducer_backend_attribution_recorded
```

The minimal reproducer replayed captured Workstream A tensors for layers 6,
10, 13, 16, 18, and 21. `torch.addmm`, `torch.nn.functional.linear`, and
`torch._C._nn.linear` all matched the existing official o-proj artifacts
full-vector exactly under baseline, MKLDNN enabled, and MKLDNN disabled
runtime configs. The zero-bias, explicit matmul, and explicit einsum controls
remained negative.

Profiler evidence recorded `aten::addmm`; verbose ONEDNN/DNNL/MKL attempts did
not identify one concrete active CPU backend. The active backend inference is
therefore `multiple_possible`, with no concrete replayable rule found and no
Rust/CUDA policy synthesis reopened.

## CPUBlas GEMM Attribution

Status:

```text
/tmp/fused_linear_addmm_cpublas_gemm_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpublas_gemm_attribution_recorded
```

The lower-GEMM attribution stage confirmed the source chain below the official
API seam:

```text
linear 2D+bias -> addmm -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm
```

The source map and `libtorch_cpu.so` symbol scan expose multiple lower-GEMM
candidates: native CPU stubs, MKL/BLAS BF16 GEMM symbols, and MKLDNN/oneDNN
BF16 matmul/GEMM paths. Runtime telemetry preserved the sampled official seam
for baseline, MKLDNN enabled/disabled, verbose DNNL/ONEDNN/MKL, and ONEDNN ISA
configs, while optional `ATEN_CPU_CAPABILITY=default` changed layer18 and is
recorded only as attribution telemetry.

Current decision:

- `active_backend_inference = multiple_possible`.
- `active_backend_confidence = medium`.
- `concrete_replayable_rule_found = false`.
- `reopen_rust_policy_synthesis = false`.
- The official seam remains CPU Torch API artifact/provenance, not a selected
  Rust/CUDA backend.

## CPU Capability Differential

Status:

```text
/tmp/fused_linear_addmm_cpu_capability_differential_status.json
```

Classification:

```text
fused_linear_addmm_cpu_capability_differential_official_depends_on_cpu_capability
```

The CPU capability differential confirmed that `ATEN_CPU_CAPABILITY=default`
changes the official seam on layer18 only, while the no-override baseline and
optional `avx2`, `avx512`, `avx512_bf16`, and `avx512_vnni` settings preserve
the sampled official artifacts. The layer18 change is one BF16 ULP or less at
hidden lane 1641 and overlaps a prior Rust CPU closure-audit residual lane.

This narrows attribution toward an optimized CPU capability path:

- `official_baseline_requires_optimized_cpu_capability = true`.
- `active_backend_inference = optimized_cpu_kernel_likely`.
- `concrete_replayable_rule_found = false`.
- `reopen_rust_policy_synthesis = false`.

The finding is still not a replayable arithmetic or microkernel rule. It does
not replace the official artifacts, select a backend, authorize consumer
revalidation, or permit runtime/default/CUDA behavior changes.

## PyTorch CPU Instrumentation Plan

Plan:

```text
docs/FUSED_LINEAR_ADDMM_PYTORCH_CPU_INSTRUMENTATION_PLAN.md
```

Classification:

```text
fused_linear_addmm_pytorch_cpu_instrumentation_plan_recorded
```

The planned future branch is
`oracle/fused-linear-addmm-pytorch-source-cpu-build-instrumentation`. It would
use a separate `/home/emmy/openai/.venvs/pytorch-src-cpu` build environment
and gate minimal addmm/GEMM logs behind `GPT_OSS_TRACE_ADDMM=1`. The branch is
only justified as source attribution; backend/path identification alone would
not reopen Rust/CUDA policy synthesis or authorize implementation.

## PyTorch CPU Instrumentation Result

Status:

```text
/tmp/fused_linear_addmm_pytorch_cpu_instrumentation_status.json
```

Classification:

```text
fused_linear_addmm_pytorch_cpu_instrumentation_path_identified_not_replayable
```

The CPU-only instrumented PyTorch source build reached the Workstream A
layer18 seam without perturbing baseline numeric behavior. Baseline
`torch.addmm`, `torch.nn.functional.linear`, and `torch._C._nn.linear` matched
the official artifact exactly. `ATEN_CPU_CAPABILITY=default` reproduced the
known lane1641 one-BF16-ULP differential:

- official/baseline: `0.0289306640625`;
- `default`: `0.02880859375`.

Instrumentation traced both baseline and `default` through:

```text
addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub
```

The traced path did not change under `default`, so this identifies the lower
source path but does not explain the numeric split as a concrete replayable
arithmetic or microkernel rule. `concrete_replayable_rule_found = false` and
`reopen_rust_policy_synthesis = false`.

The official seam remains a CPU Torch API seam. No backend is selected, no
consumer revalidation is authorized, and no runtime/default/CUDA behavior
changes are authorized.

## GEMM Stub Internals

Status:

```text
/tmp/fused_linear_addmm_gemm_stub_dispatch_internals_status.json
```

Classification:

```text
fused_linear_addmm_gemm_stub_replayable_rule_identified
```

The GEMM-stub dispatch-internals branch archived the pre-existing PyTorch
instrumentation patch and then traced the lower dispatch target. `gemm_stub`
is declared/defined in `aten/src/ATen/native/CPUBlas.h` and `CPUBlas.cpp`, and
registered through `aten/src/ATen/native/cpu/BlasKernel.cpp`.

Baseline/no override runs with runtime CPU capability `AVX512`, but the
`AVX512` dispatch table entry is null, so it selects the AVX2-compiled
`cpublas_gemm_impl`. `ATEN_CPU_CAPABILITY=default` selects the
DEFAULT-compiled `cpublas_gemm_impl`. That target difference explains the
layer18 lane1641 split:

- official/baseline value: `0.0289306640625`;
- `default` value: `0.02880859375`;
- AVX2 dot/pre-BF16 combined: `0.1587543488` / `0.02887153625`;
- DEFAULT dot/pre-BF16 combined: `0.1587524414` / `0.02886962891`.

The traced rule is bias-as-prior, BF16 dot accumulation into f32, and one final
BF16 cast, with CPU-capability-specific GEMM-stub target behavior determining
which side of the BF16 rounding boundary layer18 lane1641 lands on.

This is the first concrete source-level replayable rule for the observed
baseline/default differential, but it is not a sampled-set Rust/CUDA policy.
`reopen_rust_policy_synthesis = false` remains in force until a separately
approved design verifies global sampled-set replay. No backend is selected,
and no consumer revalidation, CUDA mirror, rebaseline, or runtime/default/CUDA
behavior change is authorized.

## Guardrails

- No backend selected.
- No implementation authorized.
- No consumer revalidation authorized.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency in Rust.
