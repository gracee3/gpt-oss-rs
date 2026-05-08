# Fused Linear/AddMM GEMM Stub Source Replay Design

Classification:

```text
fused_linear_addmm_gemm_stub_source_replay_design_recorded
```

## Scope

This is a docs-only source-derived replay design for the selected
AVX2-compiled `cpublas_gemm_impl` behavior observed in the Workstream A
fused-linear/addmm seam.

This branch does not modify PyTorch, reset PyTorch, build PyTorch, run probes,
implement Rust/CUDA behavior, reopen Rust/CUDA policy synthesis, select a
backend, run consumer revalidation, emit outputs, continue the ladder, or
change runtime/default/CUDA behavior.

Primary question:

```text
Can we define a validation-only replay design for the AVX2-selected
cpublas_gemm_impl rule that could be implemented outside PyTorch and tested
against the full sampled Workstream A set?
```

Required sampled layers for any future prototype:

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

The sampled trace supports source replay design:

- sampled layers 6, 10, 13, 16, 18, and 21 were traced;
- baseline/no override selected AVX2-compiled `cpublas_gemm_impl` for all
  sampled layers;
- `ATEN_CPU_CAPABILITY=default` selected DEFAULT-compiled
  `cpublas_gemm_impl` for all sampled layers;
- `avx2`, explicit `avx512`, `avx512_bf16`, and `avx512_vnni` selected or fell
  back to the AVX2 target;
- baseline `torch.addmm`, `torch.nn.functional.linear`, and
  `torch._C._nn.linear` matched official artifacts full-vector exactly for all
  sampled layers;
- negative controls remained negative.

The global replay policy is not yet proven. The sampled trace explains only
layer18 lane1641 completely:

- residual lanes traced: 25;
- residual lanes explained: 1;
- residual lanes not yet modeled outside PyTorch: 24;
- `concrete_global_replay_policy_found = false`;
- `replayable_rule_scope = lane_level`;
- `reopen_rust_policy_synthesis = false`.

Rust/CUDA policy synthesis remains closed until future prototype evidence
clears the full sampled set.

## Source Target

The source path for the official seam is:

```text
linear 2D+bias -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub
```

Relevant source files at PyTorch commit
`70d99e998b4955e0049d13a98d77ae1b14db1f45`:

| Role | Source |
| --- | --- |
| `gemm_stub` declaration | `aten/src/ATen/native/CPUBlas.h` |
| `gemm_stub` definition | `aten/src/ATen/native/CPUBlas.cpp` |
| BF16 GEMM implementation registration | `aten/src/ATen/native/cpu/BlasKernel.cpp` |
| DispatchStub selection | `aten/src/ATen/native/DispatchStub.h` and `DispatchStub.cpp` |
| BF16 dot helper | `aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp` |

The registered function is:

```text
REGISTER_DISPATCH(cpublas::gemm_stub, &cpublas::cpublas_gemm_impl)
```

The target function body is `cpublas_gemm_impl` in
`aten/src/ATen/native/cpu/BlasKernel.cpp`. It is not a single host-neutral
body at runtime: the file is compiled under CPU capability variants, so the
sampled host dispatches to the AVX2-compiled instantiation for baseline and
to the DEFAULT-compiled instantiation when `ATEN_CPU_CAPABILITY=default`.

For the sampled seam:

- `type = at::kBFloat16`;
- `transa != NoTranspose`;
- `transb = NoTranspose`;
- `m = 2880`;
- `n = 1`;
- `k = 4096`;
- `alpha = 1`;
- `beta = 1`;
- `lda = 4096`;
- `ldb = 4096`;
- `ldc = 2880`;
- output is downcast to BF16.

The BF16 path enters `gemm_core_`, selects the BF16 specialization of
`gemm_transa_`, and calls:

```text
at::native::CPU_CAPABILITY::bf16_dot_with_fp32_arith(a_, b_, k)
```

`DispatchStubImpl::choose_cpu_impl` chooses the runtime capability in
decreasing order. On the sampled host, runtime capability is AVX512, but
`REGISTER_DISPATCH` registers a null AVX512 pointer for this stub. The
DispatchStub therefore falls back to AVX2. Explicit `avx512`,
`avx512_bf16`, and `avx512_vnni` do not change the sampled target-selection
rule; the observed target remains AVX2 or falls back to AVX2.

The DEFAULT implementation is useful diagnostic contrast only. It should not
be promoted as a new oracle artifact.

## Replay Rule Hypothesis

Candidate replay rule:

- target selection:
  baseline/no override uses AVX2-compiled `cpublas_gemm_impl` for the sampled
  host;
- matrix shape:
  `M = 2880`, `N = 1`, `K = 4096`;
- source path:
  `linear 2D+bias -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm -> gemm_stub`;
- beta/bias:
  `beta = 1`; BF16 bias/self is the prior accumulator;
- alpha:
  `alpha = 1`;
- input types:
  BF16 input, BF16 weight, BF16 bias;
- computation:
  BF16 dot accumulated into f32 by the selected AVX2 GEMM-stub target;
- output:
  one final BF16 cast after fused dot plus bias;
- references:
  explicit matmul, einsum, and unfused-bias forms remain negative controls,
  not official references.

For layer18 lane1641, the traced values are:

| Quantity | Value |
| --- | --- |
| official/baseline output | `0.0289306640625` |
| default output | `0.02880859375` |
| bias prior | `-0.1298828125` |
| AVX2 dot | `0.1587543488` |
| AVX2 pre-BF16 combined | `0.02887153625` |
| DEFAULT dot | `0.1587524414` |
| DEFAULT pre-BF16 combined | `0.02886962891` |

Interpretation: target-specific f32 dot reduction changes the fused
pre-BF16 value enough to cross a BF16 rounding boundary.

## Unknowns Before Prototype

The design intentionally leaves these items as explicit prototype blockers or
bounded design tasks:

| Question | Current state |
| --- | --- |
| AVX2 vector width | not yet documented in replay-ready terms |
| tile shape | unknown for replay |
| inner K loop order | implied by `bf16_dot_with_fp32_arith`, not yet replay-specified |
| lane grouping/order | unknown |
| pairwise/sequential accumulation within vector lanes | unknown |
| horizontal reduction order | not yet specified outside PyTorch |
| tail handling for `K = 4096` | likely aligned, but must be verified |
| output-lane-dependent accumulation behavior | possible; must be checked |
| BF16 load/convert behavior | source-visible, replay details still needed |
| f32 accumulator behavior | source-visible, exact reduction order still needed |
| final f32 plus bias fusion point | traced as `beta * prior + alpha * dot` before BF16 cast |
| final BF16 rounding mode | must match PyTorch/C++ conversion exactly |
| same rule across all output lanes | not yet proven |
| scalar-equivalent replay possible | unknown and likely insufficient if AVX2 reduction order matters |
| exact replay requires SIMD/AVX2 emulation | open question |

Any prototype must close or explicitly bound these unknowns before claiming a
global replay policy.

## Residual-Lane Design

The sampled trace recorded these residual lanes:

| Layer | Lanes |
| --- | --- |
| 6 | 22, 962, 1772, 2122 |
| 10 | 143, 298, 345, 915, 1166, 2106 |
| 13 | 151, 169, 1615, 1768, 1927, 2257 |
| 16 | 1839, 1927, 2666 |
| 18 | 63, 1641, 2441, 2457 |
| 21 | 2277, 2807 |

Obligations for future replay work:

- explain why prior Rust CPU policy residuals occurred;
- classify each residual as target selection, reduction order, tile order,
  horizontal reduction, BF16 conversion, or another mechanism;
- show whether one source rule explains all residual lanes;
- preserve the layer18 lane1641 explanation;
- avoid promoting residual-lane-only success to full-vector policy success.

Current state:

- layer18 lane1641 is explained by target selection plus BF16 rounding
  boundary;
- 24 lanes remain traced but not modeled outside PyTorch;
- sampled trace supports source replay design, not global policy acceptance.

## Future Prototype Design

Future branch:

```text
validation/fused-linear-addmm-gemm-stub-source-replay-prototype
```

Purpose:

Implement a validation-only replay candidate for the source-derived AVX2 rule.

Required constraints:

- no PyTorch runtime dependency;
- no CUDA;
- no production runtime path;
- no default routing change;
- no consumer revalidation;
- no output emission;
- no ladder continuation;
- no tolerance;
- no correction metadata;
- no per-layer policy;
- no per-lane policy;
- no focus-lane promotion.

Required status output:

```text
/tmp/fused_linear_addmm_gemm_stub_source_replay_prototype_status.json
```

Allowed prototype classifications:

- `fused_linear_addmm_gemm_stub_source_replay_global_policy_cleared`
- `fused_linear_addmm_gemm_stub_source_replay_partial_only`
- `fused_linear_addmm_gemm_stub_source_replay_no_global_policy`
- `fused_linear_addmm_gemm_stub_source_replay_blocked_by_unmodeled_kernel`
- `fused_linear_addmm_gemm_stub_source_replay_failed`

Prototype acceptance requires:

- one single source-derived replay policy clears layers 6, 10, 13, 16, 18,
  and 21 full-vector exactly;
- `full_vector_mismatches = 0` for every sampled layer;
- `max_abs_diff = 0` for every sampled layer;
- negative controls remain negative;
- no tolerance;
- no correction metadata;
- no per-layer or per-lane choice;
- no PyTorch call;
- status explicitly explains why prior bounded Rust policy search missed the
  mechanism.

## Possible Implementation Strategies

### A. Source-Faithful Scalar Replay

This is simplest to reason about and review. It should model:

- BF16 inputs and weights converted to f32 products;
- `beta * prior + alpha * dot`;
- one final BF16 cast.

Risk: scalar replay likely fails if AVX2 vector grouping and horizontal
reduction order are responsible for exact sampled values.

### B. AVX2-Structured Replay In Rust Validation Code

This models vector lanes, tile/chunk boundaries, partial sums, and horizontal
reduction order explicitly. It is more likely to match the selected
AVX2-compiled target if the relevant behavior is SIMD-specific.

Risk: it may be host/compiler-sensitive and must remain validation-only until
reviewed separately.

### C. C/C++ Reference Helper Compiled Only For Validation

This could reproduce PyTorch source behavior more directly while staying out
of production runtime. It may be useful if exact AVX2 intrinsics or compiler
behavior matter.

Risk: dependency and build complexity increase. It must not become a runtime
backend by accident.

### D. Direct Extraction Or Translation Of `cpublas_gemm_impl` Logic

This has the highest fidelity and should preserve the source mechanism most
closely.

Requirements:

- no PyTorch runtime import;
- clear provenance notes;
- licensing review if source-derived logic is copied or translated;
- explicit separation from production routing.

## Decision Gates

### Gate R1 — Source Rule Design Complete

Required:

- target function identified;
- vector/tile/reduction order defined or bounded;
- BF16 conversion and rounding behavior defined;
- status schema defined;
- prior Rust miss explanation documented.

### Gate R2 — Validation Prototype Clears Sampled Set

Required:

- full-vector exactness for layers 6, 10, 13, 16, 18, and 21;
- no tolerance;
- no correction metadata;
- negatives preserved;
- one global policy.

### Gate R3 — Promotion Review

Only after Gate R2:

- decide whether to reopen Rust policy synthesis or keep this as a validation
  helper only;
- still no runtime promotion by default;
- still no consumer revalidation without a separate plan.

## Non-Acceptance Criteria

Do not proceed if:

- only layer18 lane1641 remains explained;
- exactness requires per-layer or per-lane choices;
- exactness requires tolerance or correction metadata;
- exactness requires calling PyTorch;
- exactness depends on host-specific microarchitecture too narrowly to encode
  safely;
- explicit matmul, einsum, or unfused-bias become accepted references;
- global sampled-set result is partial only.

## PyTorch Workspace Hygiene

`/home/emmy/openai/pytorch` remains dirty with source-attribution
instrumentation.

Patch archives exist:

```text
/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-dispatch-internals/pre_gemm_stub_internals.patch
/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-sampled-trace/pre_sampled_trace.patch
```

Future prototype work should not rely on untracked PyTorch state unless that
state is explicitly declared in the status. If PyTorch needs to be reset,
archive the current diff first.

This docs-only branch performs no cleanup.

## Guardrails

- Docs-only.
- No PyTorch modification.
- No PyTorch reset.
- No build.
- No probe.
- No Rust/CUDA implementation.
- No Rust/CUDA policy synthesis reopened.
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

## AVX2 Contract Extraction Result

Status:

```text
/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json
```

Classification:

```text
fused_linear_addmm_gemm_stub_avx2_contract_replay_ready
```

The contract-extraction branch inspected the PyTorch source and reused the
existing sampled-trace artifacts. It archived the current external PyTorch diff
at:

```text
/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-avx2-contract-extraction/pre_avx2_contract_extraction.patch
```

No PyTorch source was patched, reset, or rebuilt in this branch.

Replay-ready AVX2 contract:

- baseline/no override selects the AVX2-compiled `cpublas_gemm_impl` on the
  sampled host;
- the source path remains `cpublas_gemm_impl -> gemm_core_ ->
  BF16-specialized gemm_transa_ -> compute_dot ->
  CPU_CAPABILITY::bf16_dot_with_fp32_arith`;
- sampled GEMM shape is `M=2880`, `N=1`, `K=4096`;
- `alpha = 1`, `beta = 1`, and BF16 bias is the prior `c` accumulator;
- BF16 input and weight values are converted exactly to f32 before multiply;
- the AVX2 dot processes `K` in 64-BF16 chunks;
- each chunk uses four BF16 vector pairs, producing eight f32 vector
  accumulators;
- accumulator updates use AVX2 f32 fused multiply-add;
- `K=4096` has no vector tail and no scalar tail;
- the eight f32 vector accumulators are reduced by PyTorch's `VectorizedN`
  pairwise order, then by the AVX2 f32 horizontal shuffle reduction;
- bias is fused after dot reduction in f32;
- final output is one BF16 round-to-nearest-even cast.

The extraction marks:

```text
replay_contract_complete = true
supports_validation_prototype = true
concrete_global_replay_policy_found = false
reopen_rust_policy_synthesis = false
```

This result authorizes only the future validation prototype design already
described here. It does not select a backend, reopen Rust/CUDA policy
synthesis, authorize implementation, authorize consumer revalidation, emit
outputs, continue the ladder, or change runtime/default/CUDA behavior.
