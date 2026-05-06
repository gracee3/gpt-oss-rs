# Fused Linear/AddMM Source Attribution Closure

Classification:

```text
fused_linear_addmm_source_attribution_closure_recorded
```

## Final Conclusion

Workstream A remains an official CPU Torch API seam:

```text
CPU Torch module/F.linear/_C/addmm
torch.addmm(bias, input_2d, weight_t_2d)
```

The source-attribution lane identified the traced source path down to:

```text
linear 2D+bias -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub
```

No concrete global replayable arithmetic or microkernel rule was found.
Rust/CUDA policy synthesis remains closed, and a CUDA mirror remains
unauthorized.

This is a closure record, not an implementation authorization.

## Evidence Chain

The closure follows this Workstream A evidence chain:

- producer/API final matrix: sampled o-proj layers match the CPU Torch
  module/F.linear/_C/addmm fused-bias original-layout seam;
- helper/backend/cublas/cublasLt candidates: local helpers, cuBLAS BF16
  helpers, and the cuBLASLt fused-bias prototype did not produce a selectable
  sampled-set backend;
- CPU producer attribution: `torch.addmm`, `torch.nn.functional.linear`,
  `torch._C._nn.linear`, and module call cleared full-vector official
  artifacts while explicit matmul/einsum/unfused-bias stayed negative;
- addmm boundary localization: fused-bias `torch.addmm(bias, input, weight.T)`
  cleared while zero-bias addmm plus separate bias did not;
- fused-bias arithmetic contract: bias-before-final-observable-BF16-rounding
  remained the strongest signal, but the full accumulation/product policy was
  not localized;
- dispatch stability: sampled CPU Torch addmm outputs were stable across
  tested thread, MKLDNN, OMP, and MKL settings;
- Rust CPU policy synthesis and closure audit: 350 policies plus 238 missing
  full-vector replays did not find one global selectable Rust CPU policy;
- Torch wheel dispatch attribution: dispatch tables showed CPU and
  MKLDNN/oneDNN registration signals, with backend identity still unresolved;
- forward Python environment validation and smoke: the forward Python 3.12 /
  Torch 2.11.0+cu130 environment reproduced the existing official artifacts
  for the smoke subset;
- PyTorch source map: exact PyTorch commit
  `70d99e998b4955e0049d13a98d77ae1b14db1f45` mapped linear 2D+bias through
  addmm and into lower GEMM candidates;
- minimal reproducer: captured Workstream A tensors reproduced official
  variants and preserved negative controls while runtime attribution remained
  `multiple_possible`;
- lower cpublas/GEMM attribution: source and binary evidence confirmed
  `addmm_impl_cpu_ -> cpublas::gemm` with native/MKL/MKLDNN candidates visible;
- CPU capability differential: only `ATEN_CPU_CAPABILITY=default` changed
  layer18, at lane 1641, by one BF16 ULP or less;
- CPU-only instrumented PyTorch source build: traced both baseline and
  `default` through `cpublas::gemm -> gemm_stub` without identifying a
  replayable rule.

## Instrumentation Result

Status:

```text
/tmp/fused_linear_addmm_pytorch_cpu_instrumentation_status.json
```

Classification:

```text
fused_linear_addmm_pytorch_cpu_instrumentation_path_identified_not_replayable
```

Source checkout:

```text
/home/emmy/openai/pytorch
```

Checked-out commit:

```text
70d99e998b4955e0049d13a98d77ae1b14db1f45
```

Source-build env:

```text
/home/emmy/openai/.venvs/pytorch-src-cpu
```

Instrumentation env var:

```text
GPT_OSS_TRACE_ADDMM=1
```

Instrumented files:

- `aten/src/ATen/native/Linear.cpp`;
- `aten/src/ATen/native/LinearAlgebra.cpp`;
- `aten/src/ATen/native/CPUBlas.cpp`;
- `aten/src/ATen/native/mkldnn/Matmul.cpp`.

Both baseline/no-override and `ATEN_CPU_CAPABILITY=default` traced through:

```text
linear 2D+bias -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub
```

Layer18 lane1641 values:

- official/baseline: `0.0289306640625`;
- `ATEN_CPU_CAPABILITY=default`: `0.02880859375`;
- absolute difference: `0.0001220703125`.

Recorded interpretation fields:

- `active_path_baseline = native_cpublas_stub`;
- `active_path_default = native_cpublas_stub`;
- `path_changed_under_default = false`;
- `instrumentation_perturbed_numeric_behavior = false`;
- `concrete_replayable_rule_found = false`;
- `reopen_rust_policy_synthesis = false`.

## Interpretation

The remaining behavior is below the traced PyTorch source branch, inside or
under `gemm_stub` and CPU dispatch-stub-selected microkernel behavior.

Path identification is not a replayable arithmetic rule. A selected CPU source
path also does not equal a Rust/CUDA backend candidate. The result is useful
negative evidence: it closes broad source-attribution work while preserving the
official seam as a producer/API oracle.

## Stop Decision

Stop the current source-attribution lane here unless a new design explicitly
targets `gemm_stub` dispatch internals.

Preserve Workstream A as the official CPU Torch API seam. Do not run another
blind arithmetic sweep. Do not open a CUDA mirror. Do not run consumer
revalidation from this evidence.

## Future Options

Allowed future options:

- use the producer/API artifact reuse plan to consume official seam artifacts
  in validation-only contexts;
- continue downstream blocker discovery using official seam artifacts;
- write a separate, narrow PyTorch `gemm_stub` dispatch-internals plan only if
  explicitly approved;
- perform a full rebaseline only if explicitly approved and separately
  designed.

Disallowed from this closure:

- runtime implementation;
- backend selection;
- consumer revalidation;
- output emission;
- ladder continuation;
- tolerance or correction metadata promotion;
- final-logit, all-layer, server, or 4097/context-length claims;
- Torch runtime dependency in Rust.

## PyTorch Workspace Hygiene

The PyTorch checkout at `/home/emmy/openai/pytorch` remains dirty with
instrumentation patches. Those patches were intentionally not committed into
`gpt-oss-rs`.

A future explicitly authorized hygiene step should archive the patch before
resetting the checkout:

```bash
git -C /home/emmy/openai/pytorch diff > /home/emmy/openai/pytorch-research/fused-linear-addmm-pytorch-cpu-instrumentation/instrumentation.patch
git -C /home/emmy/openai/pytorch rev-parse HEAD
```

Only after the patch archive is confirmed, an explicitly authorized cleanup may
reset the source checkout:

```bash
git -C /home/emmy/openai/pytorch reset --hard 70d99e998b4955e0049d13a98d77ae1b14db1f45
```

Do not perform those commands in this docs-only branch.

## Guardrails

- Docs-only.
- No PyTorch source modifications.
- No PyTorch reset.
- No PyTorch build.
- No probes.
- No backend selected.
- No implementation authorized.
- No consumer revalidation authorized.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.
- No Rust/CUDA policy synthesis reopened.
