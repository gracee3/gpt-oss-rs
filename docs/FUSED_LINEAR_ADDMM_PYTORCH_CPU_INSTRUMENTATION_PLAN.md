# Fused Linear/AddMM PyTorch CPU Instrumentation Plan

Classification:

```text
fused_linear_addmm_pytorch_cpu_instrumentation_plan_recorded
```

## Scope

This is a docs-only plan for a future minimal CPU-only instrumented PyTorch
build focused on the Workstream A fused-linear/addmm seam.

It does not build PyTorch, patch PyTorch, create a venv, modify
`/home/emmy/openai/pytorch`, run probes, modify runtime code, or reopen
Rust/CUDA policy synthesis.

Primary question:

```text
Can a minimal CPU-only instrumented PyTorch build identify the active optimized
CPU BF16 addmm/GEMM path behind the official seam, and explain the layer18
lane1641 differential under ATEN_CPU_CAPABILITY=default?
```

Future branch:

```text
oracle/fused-linear-addmm-pytorch-source-cpu-build-instrumentation
```

Source checkout:

```text
/home/emmy/openai/pytorch
```

Expected commit:

```text
70d99e998b4955e0049d13a98d77ae1b14db1f45
```

Build venv:

```text
/home/emmy/openai/.venvs/pytorch-src-cpu
```

The source build venv must remain separate from:

- `/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130`
- `/home/emmy/openai/gpt-oss/.venv`
- `/data/models/.venv-awq`

## Source Evidence

Input status:

```text
/tmp/fused_linear_addmm_cpu_capability_differential_status.json
```

Input classification:

```text
fused_linear_addmm_cpu_capability_differential_official_depends_on_cpu_capability
```

Known result:

- sampled layers: 6, 10, 13, 16, 18, 21;
- executed CPU capability configs: baseline/no override, `default`, `avx2`,
  `avx512`, `avx512_bf16`, `avx512_vnni`;
- only `ATEN_CPU_CAPABILITY=default` changed the official seam;
- only layer18 changed;
- changed official variants: `torch.addmm`, `torch.nn.functional.linear`,
  `torch._C._nn.linear`;
- layer18 changed at lane 1641;
- baseline/official expected value: `0.0289306640625`;
- `ATEN_CPU_CAPABILITY=default` actual value: `0.02880859375`;
- absolute difference: `0.0001220703125`;
- difference is one BF16 ULP or less;
- lane 1641 overlaps prior Rust CPU closure-audit residual evidence;
- `active_backend_inference = optimized_cpu_kernel_likely`;
- `concrete_replayable_rule_found = false`;
- `reopen_rust_policy_synthesis = false`;
- no backend selected, implementation authorized, consumer revalidation
  authorized, or runtime/default/CUDA behavior changed.

## Instrumentation Targets

### API/source chain

`aten/src/ATen/native/Linear.cpp`

- 2D+bias `linear` route to `addmm`.

### AddMM CPU path

`aten/src/ATen/native/LinearAlgebra.cpp`

- `ADDMM_META`
- `addmm_out_cpu`
- `addmm_impl_cpu_`
- `mm_out_cpu`
- `_AT_DISPATCH_ADDMM_TYPES`
- `cpublas::gemm` callsite
- beta/self/result handling
- copy/bias expansion behavior
- `transpose_a`, `transpose_b`, `transpose_c` decisions
- `lda`, `ldb`, `ldc`, `M`, `N`, `K` values

### Lower GEMM

Inspect and instrument only where needed:

- `aten/src/ATen/native/CPUBlas.cpp`, if present
- `aten/src/ATen/native/Blas.cpp`
- `aten/src/ATen/native/cpu/BlasKernel.cpp`
- `aten/src/ATen/native/mkldnn/Matmul.cpp`
- `aten/src/ATen/native/mkldnn/Matmul.h`
- `aten/src/ATen/native/mkldnn/MKLDNNCommon.h`

## Minimal Trace Contract

Instrumentation should log:

- whether `cpublas::gemm` enters a native, MKL/BLAS, MKLDNN/oneDNN, or stub
  path;
- scalar type;
- `alpha` and `beta`;
- `M`, `N`, `K`;
- `transa`, `transb`, `transc`;
- `lda`, `ldb`, `ldc`;
- selected CPU capability;
- whether a BF16-specific branch fires;
- whether `mkldnn_bf16_gemm`, `mkldnn_gemm`, or
  `use_mkldnn_bf16_matmul` is called;
- whether `gemm_no_downcast_stub` or an equivalent native stub is called;
- whether `BLAS_HAS_SBGEMM` or `MKL_HAS_SBGEMM` is compiled and used;
- cheap oneDNN primitive or post-op/fuse-sum metadata, if visible.

Logging should be gated by an environment variable:

```text
GPT_OSS_TRACE_ADDMM=1
```

`printf`/`eprintln`-style logging is acceptable. Do not add broad PyTorch
tracing framework changes.

## Future Build Constraints

- CPU-only first.
- Use `USE_CUDA=0`.
- Do not initialize unnecessary submodules unless the build requires them.
- Do not run full model loading.
- Do not perform GPU attribution.
- Do not modify `gpt-oss-rs` runtime behavior.
- Do not instrument CUDA.
- Do not change production/default routing.

## Future Runtime Comparison

Use captured Workstream A tensors only.

Required layer order:

1. layer18 first;
2. optionally layers 6, 10, 13, 16, and 21 after layer18 is validated.

Required configs:

- baseline/no `ATEN_CPU_CAPABILITY` override;
- `ATEN_CPU_CAPABILITY=default`.

Optional configs:

- `avx512_bf16`
- `avx512_vnni`
- `avx512`
- `avx2`

Focus output:

- layer18 lane 1641;
- compare baseline versus `default`;
- record whether the active lower-GEMM path changes;
- record whether the numeric change is explained by the path difference.

Do not treat the `default` output as a new oracle artifact. Do not rebaseline.

## Future Status Contract

Allowed future classifications:

- `fused_linear_addmm_pytorch_cpu_instrumentation_backend_identified`
- `fused_linear_addmm_pytorch_cpu_instrumentation_replayable_rule_identified`
- `fused_linear_addmm_pytorch_cpu_instrumentation_path_identified_not_replayable`
- `fused_linear_addmm_pytorch_cpu_instrumentation_inconclusive`
- `fused_linear_addmm_pytorch_cpu_instrumentation_build_failed`

Future status JSON should include:

```json
{
  "validation_only": true,
  "source_attribution_probe": true,
  "pytorch_cpu_instrumentation": true,
  "oracle_device": "cpu",
  "cuda_used": false,
  "source_checkout_path": "/home/emmy/openai/pytorch",
  "checked_out_commit": "70d99e998b4955e0049d13a98d77ae1b14db1f45",
  "build_env_path": "/home/emmy/openai/.venvs/pytorch-src-cpu",
  "pytorch_build_performed": true,
  "pytorch_source_patched": true,
  "instrumentation_env_var": "GPT_OSS_TRACE_ADDMM",
  "layers_evaluated": [],
  "configs_evaluated": [],
  "layer18_lane1641_result": {},
  "active_path_baseline": null,
  "active_path_default": null,
  "path_changed_under_default": null,
  "concrete_replayable_rule_found": false,
  "reopen_rust_policy_synthesis": false,
  "backend_selected": false,
  "implementation_authorized": false,
  "consumer_revalidation_authorized": false,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "output_emitted": false,
  "ladder_continued": false,
  "final_logit_claim": false,
  "all_layer_claim": false,
  "server_claim": false,
  "context_length_claim": false
}
```

## Decision Rules

- Backend or path identification alone does not reopen Rust/CUDA policy
  synthesis.
- Reopen Rust/CUDA policy synthesis only if instrumentation identifies a
  concrete, global, replayable arithmetic or microkernel rule.
- If the active path is an optimized CPU microkernel or oneDNN/MKL primitive
  without replayable arithmetic detail, record
  `path_identified_not_replayable`.
- Do not treat `ATEN_CPU_CAPABILITY=default` output as a new oracle artifact.
- Do not rebaseline.
- Do not select a backend.
- Do not authorize consumer revalidation.
- Do not proceed to CUDA mirror from path identification alone.

## Stop Conditions

Stop and preserve the official CPU Torch API seam if:

- the source build fails or becomes too costly;
- instrumentation cannot reach the lower-GEMM path;
- the active path remains opaque;
- the active path is a library primitive or microkernel with no replayable
  rule;
- instrumentation perturbs numeric behavior;
- only a layer-specific explanation is found.

If any stop condition occurs:

- preserve Workstream A as the official CPU Torch API seam;
- keep Rust/CUDA policy synthesis closed;
- do not proceed to CUDA mirror;
- do not run consumer revalidation;
- do not change runtime/default/CUDA behavior.

## CPU Instrumentation Result

Status:

```text
/tmp/fused_linear_addmm_pytorch_cpu_instrumentation_status.json
```

Classification:

```text
fused_linear_addmm_pytorch_cpu_instrumentation_path_identified_not_replayable
```

The CPU-only source-build instrumentation branch created the separate
`/home/emmy/openai/.venvs/pytorch-src-cpu` build environment, patched the
checked-out PyTorch source at
`70d99e998b4955e0049d13a98d77ae1b14db1f45`, and built PyTorch with
`USE_CUDA=0`. Instrumentation was gated by `GPT_OSS_TRACE_ADDMM=1` and touched
only:

- `aten/src/ATen/native/Linear.cpp`;
- `aten/src/ATen/native/LinearAlgebra.cpp`;
- `aten/src/ATen/native/CPUBlas.cpp`;
- `aten/src/ATen/native/mkldnn/Matmul.cpp`.

Layer18 was evaluated under the baseline/no-override configuration and
`ATEN_CPU_CAPABILITY=default`. The baseline instrumented build reproduced the
official artifact exactly, so the instrumentation did not perturb numeric
behavior. The `default` run reproduced the known one-lane differential:

- official/baseline lane 1641: `0.0289306640625`;
- `ATEN_CPU_CAPABILITY=default` lane 1641: `0.02880859375`;
- absolute difference: `0.0001220703125`;
- official variants changed: `torch.addmm`, `torch.nn.functional.linear`, and
  `torch._C._nn.linear`.

Trace output identified the active source path for both baseline and
`default` as:

```text
linear 2D+bias -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub
```

The path label did not change under `ATEN_CPU_CAPABILITY=default`. This
identifies the lower source path but still does not expose a concrete global
replayable arithmetic or microkernel rule. Therefore:

- `active_path_baseline = native_cpublas_stub`;
- `active_path_default = native_cpublas_stub`;
- `path_changed_under_default = false`;
- `concrete_replayable_rule_found = false`;
- `reopen_rust_policy_synthesis = false`.

Build and trace logs are under:

```text
/home/emmy/openai/pytorch-research/fused-linear-addmm-pytorch-cpu-instrumentation/
```

This result preserves the official CPU Torch API seam. It does not authorize a
rebaseline, backend selection, consumer revalidation, CUDA mirror, output
emission, ladder continuation, or runtime/default/CUDA behavior change.

## GEMM Stub Dispatch Internals Result

Status:

```text
/tmp/fused_linear_addmm_gemm_stub_dispatch_internals_status.json
```

Classification:

```text
fused_linear_addmm_gemm_stub_replayable_rule_identified
```

The follow-up GEMM-stub internals branch archived the pre-existing dirty
PyTorch instrumentation patch before adding more source-attribution logging:

```text
/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-dispatch-internals/pre_gemm_stub_internals.patch
```

Source mapping identified `gemm_stub` declaration/definition in
`aten/src/ATen/native/CPUBlas.h` and `CPUBlas.cpp`, with registration in
`aten/src/ATen/native/cpu/BlasKernel.cpp`. Runtime traces showed that the
baseline/no-override path sees runtime CPU capability `AVX512`, but the
`AVX512` dispatch table entry is null and selects the `AVX2`-compiled
`cpublas_gemm_impl`. `ATEN_CPU_CAPABILITY=default` selects the
`DEFAULT`-compiled `cpublas_gemm_impl`.

Both paths execute the BF16 GEMM-stub implementation, but their lane1641 dot
accumulators differ:

- bias prior: `-0.1298828125`;
- baseline/official lane 1641: `0.0289306640625`;
- default lane 1641: `0.02880859375`;
- baseline dot: `0.1587543488`;
- default dot: `0.1587524414`;
- baseline pre-BF16 combined value: `0.02887153625`;
- default pre-BF16 combined value: `0.02886962891`.

This explains the layer18 lane1641 split: the selected GEMM-stub target changes
from the AVX2-compiled implementation to the DEFAULT-compiled implementation,
and the small dot-product difference crosses the BF16 rounding boundary. The
status records `concrete_replayable_rule_found = true` for this traced
source-level mechanism, while keeping `reopen_rust_policy_synthesis = false`
because sampled-set/global replay validation still needs a separate approved
design.

No production backend was selected. No runtime/default/CUDA behavior change,
consumer revalidation, rebaseline, output emission, or ladder continuation was
authorized.

## Plan-Branch Guardrails

- The planning branch was docs-only.
- The planning branch performed no PyTorch build.
- The planning branch performed no PyTorch source patch.
- The planning branch created no venv.
- The planning branch modified no `/home/emmy/openai/pytorch` source.
- The planning branch ran no probes.
- No runtime implementation.
- No backend selected.
- No consumer revalidation authorized.
- No Rust/CUDA policy synthesis reopened.
- No output emission.
- No ladder continuation.
- No final-logit/all-layer/server/4097 claim.
