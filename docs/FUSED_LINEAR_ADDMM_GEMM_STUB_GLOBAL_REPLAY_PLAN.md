# Fused Linear/AddMM GEMM Stub Global Replay Plan

Classification:

```text
fused_linear_addmm_gemm_stub_global_replay_plan_recorded
```

## Scope

This is a docs-only plan for sampled-set/global replay verification of the
GEMM-stub-derived Workstream A source rule.

This branch does not modify PyTorch, reset PyTorch, build PyTorch, run probes,
implement Rust/CUDA behavior, reopen Rust/CUDA policy synthesis, select a
backend, run consumer revalidation, emit outputs, continue the ladder, or
change runtime/default/CUDA behavior.

Primary question:

```text
Can the GEMM-stub-derived source rule be verified across the full sampled
Workstream A o-proj set as one global validation policy?
```

Required sampled layers:

- 6
- 10
- 13
- 16
- 18
- 21

Official reference:

- CPU Torch module/F.linear/_C/addmm
- `torch.addmm(bias, input_2d, weight_t_2d)`
- BF16 weighted-V input
- BF16 o-proj weight
- BF16 o-proj bias
- fused bias before final observable BF16 output
- BF16 output
- full-vector exactness required
- focus-lane-only clears rejected

## Decision Summary

The GEMM-stub internals branch achieved lane-level strong success for the
layer18 lane1641 baseline/default split.

Status:

```text
/tmp/fused_linear_addmm_gemm_stub_dispatch_internals_status.json
```

Classification:

```text
fused_linear_addmm_gemm_stub_replayable_rule_identified
```

The source-attribution result identified:

- `gemm_stub` declaration/definition:
  `aten/src/ATen/native/CPUBlas.h` and
  `aten/src/ATen/native/CPUBlas.cpp`
- `gemm_stub` registration:
  `aten/src/ATen/native/cpu/BlasKernel.cpp`
- baseline/no override runtime CPU capability: `AVX512`
- baseline/no override AVX512 dispatch table entry: null
- baseline/no override selected target: AVX2-compiled `cpublas_gemm_impl`
- `ATEN_CPU_CAPABILITY=default` selected target:
  DEFAULT-compiled `cpublas_gemm_impl`
- `ATEN_CPU_CAPABILITY=avx2` selected the same AVX2 target as baseline
- explicit `ATEN_CPU_CAPABILITY=avx512` fell back to the AVX2 target

Layer18 lane1641 was explained by the selected GEMM-stub target and a BF16
rounding-boundary crossing:

| Quantity | Value |
| --- | --- |
| official/baseline output | `0.0289306640625` |
| default output | `0.02880859375` |
| bias prior | `-0.1298828125` |
| AVX2 dot | `0.1587543488` |
| AVX2 pre-BF16 combined | `0.02887153625` |
| DEFAULT dot | `0.1587524414` |
| DEFAULT pre-BF16 combined | `0.02886962891` |

This explains one traced differential, but it does not yet authorize a global
sampled-set validation policy. No runtime/default/CUDA behavior change follows.

## Rule Hypothesis To Verify

The future sampled-set replay work should verify one global rule:

- source API path:
  `linear 2D+bias -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub`
- baseline/no-override target:
  AVX2-compiled `cpublas_gemm_impl` is selected through `DispatchStub`, even
  when runtime CPU capability is AVX512 and the AVX512 table entry is null
- bias behavior:
  bias is used as prior accumulator / `beta = 1` self input
- product/reduction:
  BF16 dot is accumulated into f32 by the selected GEMM-stub target
- output:
  one final BF16 cast after fused dot plus bias
- target-specific behavior:
  AVX2-compiled and DEFAULT-compiled targets can produce adjacent BF16 results
  near rounding boundaries

The global question is whether the baseline AVX2-selected GEMM-stub rule can be
replayed for every sampled layer full-vector exactly.

## Required Verification Ladder

### A. Instrumentation Expansion Branch

Future branch:

```text
oracle/fused-linear-addmm-gemm-stub-sampled-trace
```

Purpose:

Use the existing instrumented PyTorch build to trace all sampled Workstream A
layers and selected diagnostic lanes.

Required:

- preserve or verify the current PyTorch patch archive:
  `/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-dispatch-internals/pre_gemm_stub_internals.patch`
- trace layers 6, 10, 13, 16, 18, and 21
- verify the selected `DispatchStub` target for each layer
- log lane-level dot and pre-BF16 combined values for:
  - all prior Rust residual lanes
  - focus lanes
  - first/worst mismatch lanes from negative controls where useful
- verify baseline/no override still matches the official full vector exactly
- verify `default` and other CPU capability variants remain diagnostic only
- keep `cuda_used = false`
- avoid full model loading and model forward execution

Outputs should include per-layer target summaries, residual-lane traces,
full-vector official comparisons, and a status JSON. This branch is still
source attribution, not runtime implementation.

### B. Source-Derived Replay Design Branch

Future branch:

```text
docs/fused-linear-addmm-gemm-stub-source-replay-design
```

Purpose:

Design how to replay the selected AVX2 `cpublas_gemm_impl` behavior outside
PyTorch.

Required:

- identify the exact source function body for AVX2 `cpublas_gemm_impl`
- identify whether scalar-equivalent replay is possible
- identify vector width, tile shape, reduction order, and tail handling
- identify BF16-to-f32 product behavior
- identify final BF16 conversion behavior
- identify fused `beta`/bias handling
- define why prior Rust policies missed the mechanism
- define the status schema and proof gates for a validation-only prototype

The design must keep PyTorch source attribution separate from a Rust/CUDA
runtime policy. A likely source rule is not enough to select a backend.

### C. Validation Prototype Branch

Future branch:

```text
validation/fused-linear-addmm-gemm-stub-source-replay-prototype
```

Purpose:

Implement a validation-only replay candidate for the source-derived rule. This
is not production runtime behavior.

Required:

- full sampled set: layers 6, 10, 13, 16, 18, and 21
- full-vector exactness for every sampled layer
- `full_vector_mismatches = 0`
- `max_abs_diff = 0`
- no tolerance
- no correction metadata
- no per-layer policy
- no focus-lane promotion
- no CUDA
- no consumer revalidation
- no output emission
- no ladder continuation
- no runtime/default/CUDA behavior change

## Acceptance Criteria For Reopening Rust Policy Synthesis

Set `reopen_rust_policy_synthesis = true` only if all of the following hold:

- one single source-derived replay policy clears every sampled layer full-vector
  exactly
- the same target-selection rule applies across the sampled set
- the rule explains the prior residual lanes
- negative controls remain negative
- no per-layer or per-lane policy selection is used
- no tolerance is used
- no correction metadata is used
- the implementation does not call PyTorch at runtime
- the design documents why prior bounded Rust policies missed the mechanism

Even then, reopening Rust policy synthesis does not select a production backend
or authorize consumer revalidation. It only authorizes a separately reviewed
validation-policy lane.

## Non-Acceptance Criteria

Do not reopen Rust/CUDA policy synthesis if:

- only layer18 lane1641 is explained
- only selected residual lanes clear
- only per-layer target choice works
- exactness requires tolerance
- exactness requires correction metadata
- source replay cannot be implemented without calling PyTorch
- source replay is too microarchitecture-specific to represent safely
- the mechanism is not global across sampled layers
- the candidate changes negative controls into accepted references
- the candidate requires runtime/default/CUDA behavior changes to validate

## PyTorch Workspace Hygiene

`/home/emmy/openai/pytorch` remains dirty with source-attribution
instrumentation from the prior CPU-only build work. The patch archive exists:

```text
/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-dispatch-internals/pre_gemm_stub_internals.patch
```

Future trace/replay branches must either:

- continue from the instrumented dirty checkout intentionally, or
- archive and reset before applying a new patch.

This docs-only branch does not reset PyTorch source. It does not modify,
build, or probe PyTorch.

## Guardrails

- Docs-only.
- No PyTorch modification.
- No PyTorch reset.
- No PyTorch build.
- No probe.
- No Rust/CUDA behavior implementation.
- No Rust/CUDA policy synthesis reopening in this branch.
- No backend selected.
- No implementation authorized.
- No consumer revalidation authorized.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.
- No final-logit claim.
- No all-layer claim.
- No server claim.
- No 4097/context-length claim.
- No Torch runtime dependency in Rust.

## Sampled Trace Result

Status:

```text
/tmp/fused_linear_addmm_gemm_stub_sampled_trace_status.json
```

Classification:

```text
fused_linear_addmm_gemm_stub_sampled_trace_supports_replay_design
```

The instrumentation expansion branch reused the existing CPU-only instrumented
PyTorch build and did not patch, reset, or rebuild PyTorch in this branch. It
archived the current external PyTorch diff at:

```text
/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-sampled-trace/pre_sampled_trace.patch
```

Sampled layers evaluated:

```text
6, 10, 13, 16, 18, 21
```

Configs evaluated:

```text
baseline, default, avx2, avx512, avx512_bf16, avx512_vnni
```

Result:

- baseline/no override selected the AVX2-compiled `cpublas_gemm_impl` for all
  sampled layers;
- `ATEN_CPU_CAPABILITY=default` selected the DEFAULT-compiled target for all
  sampled layers;
- `avx2`, explicit `avx512`, `avx512_bf16`, and `avx512_vnni` selected or
  fell back to the AVX2 target for all sampled layers;
- baseline official variants matched the official artifact full-vector exactly
  for every sampled layer;
- negative controls remained negative;
- 25 residual lanes were traced;
- layer18 lane1641 remains the only fully explained residual lane;
- 24 residual lanes still need a source-derived replay design to model the
  selected GEMM-stub reduction behavior.

The sampled trace supports moving to
`docs/fused-linear-addmm-gemm-stub-source-replay-design`, but it does not prove
a global replay policy. Therefore:

- `sampled_trace_supports_source_replay_design = true`;
- `concrete_global_replay_policy_found = false`;
- `replayable_rule_scope = lane_level`;
- `reopen_rust_policy_synthesis = false`;
- no backend is selected;
- no implementation, consumer revalidation, CUDA mirror, rebaseline, output
  emission, ladder continuation, or runtime/default/CUDA behavior change is
  authorized.

## Source Replay Design

Design:

```text
docs/FUSED_LINEAR_ADDMM_GEMM_STUB_SOURCE_REPLAY_DESIGN.md
```

Classification:

```text
fused_linear_addmm_gemm_stub_source_replay_design_recorded
```

The replay design records how a future validation-only prototype should model
the selected AVX2 `cpublas_gemm_impl` source rule outside PyTorch. It identifies
`CPUBlas.h`, `CPUBlas.cpp`, `BlasKernel.cpp`, `DispatchStub.*`, and
`ReducedPrecisionFloatGemvFastPathKernel.cpp` as the relevant source targets,
then lists the remaining unknowns: vector width, tile shape, K-loop grouping,
horizontal reduction order, tail handling, lane-dependent behavior, and exact
BF16 conversion/rounding parity.

The design keeps the next step bounded to
`validation/fused-linear-addmm-gemm-stub-source-replay-prototype`. It does not
implement the prototype, call PyTorch at runtime, reopen Rust/CUDA policy
synthesis, select a backend, authorize consumer revalidation, or change
runtime/default/CUDA behavior.

## AVX2 Contract Extraction

Status:

```text
/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json
```

Classification:

```text
fused_linear_addmm_gemm_stub_avx2_contract_replay_ready
```

The AVX2 contract extraction branch converted the source-replay design into a
replay-ready validation contract without patching, resetting, rebuilding, or
rerunning PyTorch. It inspected:

- `aten/src/ATen/native/cpu/BlasKernel.cpp`;
- `aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp`;
- `aten/src/ATen/native/CPUBlas.*`;
- `aten/src/ATen/native/DispatchStub.*`;
- AVX2 vector and reduction helpers;
- `torch/headeronly/util/BFloat16.h`.

Extracted rule:

- baseline target selection remains AVX2 `cpublas_gemm_impl`;
- `K=4096` is reduced as 64 chunks of 64 BF16 products;
- the dot uses eight f32 vector accumulators and AVX2 fused multiply-add;
- reduction order is the PyTorch `VectorizedN` pairwise reduction followed by
  AVX2 f32 horizontal shuffle reduction;
- BF16 bias is fused as the prior `c` value with `beta=1`;
- the final observable value is a single BF16 round-to-nearest-even cast.

The branch records `replay_contract_complete = true` and
`supports_validation_prototype = true`, but still records
`concrete_global_replay_policy_found = false` and
`reopen_rust_policy_synthesis = false`. The next permitted step remains only a
validation-only source-replay prototype; no backend selection, implementation,
consumer revalidation, output emission, ladder continuation, or
runtime/default/CUDA behavior change is authorized.
