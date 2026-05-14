# Fused Linear/AddMM Source Step-Through Plan

Classification: `fused_linear_addmm_source_stepthrough_plan_recorded`

## Scope

- Docs-only step-through plan.
- Target operator: attention o-proj BF16 linear/addmm with fused bias.
- Prompt/case: `developer-message-user-smoke`.
- Sampled layers: 6, 10, 13, 16, 18, and 21.
- Source status:
  `/tmp/fused_linear_addmm_cpu_producer_attribution_status.json`.
- AVX2 contract extraction status:
  `/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json`.
- No implementation authorized.
- No backend selected.
- No PyTorch patch/rebuild in this branch.
- No Rust helper implementation.
- No runtime/default/CUDA behavior change.
- No consumer revalidation.
- No output emission.
- No ladder continuation.

## Source Evidence

Docs:

- `docs/FUSED_LINEAR_ADDMM_CPU_PRODUCER_ATTRIBUTION_PLAN.md`
- `docs/ORACLE_PRODUCER_SEAM_PIPELINE_AND_SCALING_PLAN.md`
- `docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md`
- `docs/FUSED_LINEAR_ADDMM_BACKEND_DISCRIMINATOR_DESIGN.md`
- `docs/ORDERED_SURFACE_BATCH_MILESTONE_SUMMARY.md`
- `docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md`

Statuses:

- `/tmp/fused_linear_addmm_cpu_producer_attribution_status.json`
- `/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json`
- `/tmp/o_proj_producer_api_probes_13_16_10_status.json`
- `/tmp/o_proj_producer_api_probes_18_21_status.json`
- `/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json`
- `/tmp/fused_linear_addmm_like_helper_prototype_status.json`

## Current Finding

- Module/F.linear/_C/addmm/addmm clear all sampled layers.
- Explicit matmul/einsum/unfused-bias remain negative controls.
- CPU profiler observed ATen linear/addmm but did not prove lower-level
  dispatch.
- AVX2 contract consistency is true for layers 6/10/13/16/18/21.
- Source-level dispatch is not proven.
- Backend identity is not proven.
- Backend selected: false.
- Implementation authorized: false.

## Problem Statement

The project now needs to prove where CPU BF16 `linear/addmm` dispatches inside
PyTorch/ATen/oneDNN for this exact o-proj seam. The AVX2 contract is plausible
and replay-ready, but the current evidence is consistency, not source-level
proof. A Rust helper should not be implemented until the source/dispatch path
is either proven or explicitly classified as unresolved.

## Step-Through Questions

1. Does `torch.nn.functional.linear` lower to `torch.addmm` for the sampled
   BF16 CPU input/weight/bias shapes?
2. Does `_C._nn.linear` dispatch through the same path?
3. Does `torch.addmm(bias, input, weight.T)` enter ATen native CPU `addmm` or
   an MKLDNN/oneDNN path?
4. Which dispatch key wins for BF16 CPU tensors under the installed Torch
   wheel?
5. Which source file/function owns the observed fused-bias semantics?
6. Does the path use a gemm stub with AVX2-style vectorized reduction?
7. Is the extracted AVX2 contract source-level correct, or merely
   behaviorally consistent?
8. Under what toggles does dispatch change:
   - MKLDNN enabled/disabled
   - thread count
   - CPU capability flags
   - tensor contiguity/layout
   - fused vs unfused bias

## Planned Source Workspace

Use a separate source workspace, not this repo branch:

```text
/home/emmy/openai/pytorch
```

Recommended setup:

- Clone PyTorch source matching the installed wheel git version if available
  from `torch.version.git_version`.
- Keep a separate venv, for example
  `/home/emmy/openai/.venv-pytorch-src`.
- Do not modify the `gpt-oss-rs` repo for PyTorch source work.
- Do not vendor PyTorch source into this repo.
- Record exact PyTorch commit, wheel version, Python path, and build flags.

If a source clone already exists, use it read-only first.

## Phase 0 - Read-Only Dispatch Table Attribution

Docs-only plan; do not implement here.

Future branch:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

Goal: read dispatch registrations and runtime operator tables without patching
or rebuilding PyTorch.

Steps:

- Record `torch.__version__`.
- Record `torch.version.git_version`.
- Record `torch.__config__.show()`.
- Inspect dispatch tables for:
  - `aten::linear`
  - `aten::addmm`
  - `aten::mm`
  - `aten::matmul`
- Record CPU, MkldnnCPU, AutogradCPU, Composite, and related dispatch
  registrations.
- Record whether BF16 CPU has specialized path visibility.
- Run minimal profiler for sampled shape `[1,4096] x [4096,2880]` with BF16
  bias.
- Record whether profiler exposes `aten::addmm` only or deeper backend names.
- Do not patch/rebuild.

Expected status:

```text
/tmp/fused_linear_addmm_source_dispatch_table_status.json
```

Expected classifications:

- `fused_linear_addmm_source_dispatch_table_recorded`
- `fused_linear_addmm_source_dispatch_table_inconclusive`
- `fused_linear_addmm_source_dispatch_table_blocked`
- `fused_linear_addmm_source_dispatch_table_execution_failed`

## Phase 1 - Source Walk Plan

Docs-only plan; do not implement here.

Goal: map the code path from Python API to lower-level CPU kernel.

Candidate source areas to inspect:

- `aten/src/ATen/native/Linear.cpp`
- `aten/src/ATen/native/Blas.cpp`
- `aten/src/ATen/native/cpu/BlasKernel.cpp`
- `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp`
- `aten/src/ATen/native/mkldnn/*`
- `aten/src/ATen/cpu/vec/*`
- `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp` only if reduction helpers
  appear relevant
- `c10/core/DispatchKey*`
- `torch/csrc/autograd/generated/*` only for wrapper visibility, not semantics

The exact file list should be corrected by source search in the future branch.
Do not assert these paths are final.

Record target symbols if found:

- `linear`
- `addmm`
- `addmm_impl_cpu_`
- `gemm_stub`
- `addmm_stub`
- `mkldnn_linear`
- oneDNN matmul / inner product wrappers
- Vectorized / VectorizedN BF16 reduction helpers
- CPU capability dispatch macros

## Phase 2 - Lightweight Instrumented Source Run

Docs-only plan; do not implement here.

Future branch:

```text
oracle/fused-linear-addmm-source-instrumentation
```

Only if Phase 0/1 are insufficient.

Goal: add minimal instrumentation to a PyTorch source build or debug build to
identify the executed code path for the exact sampled o-proj shape.

Instrumentation should record:

- reached function name
- dispatch key / CPU capability
- dtype
- matrix dimensions M/N/K
- beta/alpha
- whether bias is fused as c with beta=1
- whether oneDNN/MKLDNN path is used
- whether AVX2/AVX512/BF16-specific kernel is used
- whether the code path matches the AVX2 extracted contract

Guardrails:

- Do not alter arithmetic.
- Do not change tensor layout.
- Do not change thread count unless testing toggles.
- Do not commit PyTorch source diffs into `gpt-oss-rs`.
- Record patch separately if needed.

Expected status:

```text
/tmp/fused_linear_addmm_source_stepthrough_status.json
```

Expected classifications:

- `fused_linear_addmm_source_stepthrough_dispatch_proven`
- `fused_linear_addmm_source_stepthrough_avx2_contract_confirmed`
- `fused_linear_addmm_source_stepthrough_dispatch_inconclusive`
- `fused_linear_addmm_source_stepthrough_contract_revision_needed`
- `fused_linear_addmm_source_stepthrough_execution_failed`

## Phase 3 - Contract Decision

Docs-only plan; do not implement here.

Decision outcomes:

### Outcome A - AVX2 Contract Source-Confirmed

If source-level path proves the extracted AVX2 contract, next branch:

```text
docs/fused-linear-addmm-avx2-contract-rust-replay-design
```

Goal: design a validation-only Rust replay helper for the source-confirmed
AVX2 contract.

Still do not:

- change production runtime
- select backend
- run consumer revalidation
- emit outputs

### Outcome B - AVX2 Contract Behaviorally Consistent But Source-Unproven

If attribution remains inconclusive, next branch:

```text
docs/fused-linear-addmm-source-attribution-inconclusive-summary
```

Goal: record why Rust implementation should remain deferred or proceed only as
experimental replay.

### Outcome C - Source Contradicts AVX2 Contract

If source path differs, next branch:

```text
docs/fused-linear-addmm-contract-revision-plan
```

Goal: revise the replay contract before any Rust helper implementation.

## Required Status Contract For Future Step-Through

Future status fields:

```json
{
  "classification": "fused_linear_addmm_source_stepthrough_plan_recorded",
  "validation_only": true,
  "docs_only": true,
  "source_stepthrough_authorized": false,
  "pytorch_patch_authorized": false,
  "pytorch_rebuild_authorized": false,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "backend_selected": false,
  "implementation_authorized": false,
  "consumer_revalidation_authorized": false,
  "operator": "attention_o_proj",
  "sampled_layers": [6, 10, 13, 16, 18, 21],
  "source_statuses": {
    "cpu_producer_attribution": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
    "avx2_contract": "/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json"
  },
  "next_branch": "oracle/fused-linear-addmm-source-dispatch-table-attribution",
  "output_emitted": false,
  "ladder_continued": false,
  "correction_metadata_applied": false,
  "tolerance_pass": false,
  "final_logit_claim": false,
  "all_layer_claim": false,
  "server_claim": false,
  "context_length_claim": false
}
```

## Recommended Immediate Next Branch

Recommend after this docs branch:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

Scope:

- read-only dispatch table and profiler attribution
- no PyTorch patch/rebuild
- no Rust implementation
- no backend selection
- no consumer revalidation

Do not jump directly to:

- Rust helper implementation
- CUDA helper work
- production runtime integration

## Source Dispatch Table Attribution Result

Branch:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

Status:

```text
/tmp/fused_linear_addmm_source_dispatch_table_status.json
```

Classification:

```text
fused_linear_addmm_source_dispatch_table_recorded
```

Result:

- Dispatch tables inspected: `aten::linear`, `aten::addmm`, `aten::mm`, and
  `aten::matmul`.
- Visible dispatch registrations include CPU registrations and MkldnnCPU table
  labels, but the tables do not prove the lower-level BF16 CPU kernel path.
- CPU profiler toggles ran for default environment, MKLDNN disabled, MKLDNN
  enabled, single thread, and default thread count.
- Profiler operators observed include `aten::linear`, `aten::addmm`,
  `aten::matmul`, `aten::mm`, `aten::einsum`, and `aten::bmm`.
- No MKLDNN/oneDNN/DNNL/MKL backend event name was visible in the profiler
  output.
- AVX2 contract consistency remains true from the CPU producer attribution
  source status.
- Source-level dispatch proven: false.
- Backend identity proven: false.

Interpretation:

Read-only dispatch table and profiler attribution narrows the evidence to
ATen-level dispatch visibility, but it still does not prove the concrete
lower-level fused-linear/addmm CPU implementation. Source instrumentation is
recommended only after reviewing this status; no PyTorch patch/rebuild, Rust
helper implementation, backend selection, consumer revalidation,
runtime/default/CUDA change, output emission, ladder continuation,
correction/tolerance, or final-logit/all-layer/server/4097 claim is
authorized by this result.

## Source Walk Attribution Result

Branch:

```text
oracle/fused-linear-addmm-source-walk-attribution
```

Status:

```text
/tmp/fused_linear_addmm_source_walk_attribution_status.json
```

Classification:

```text
fused_linear_addmm_source_walk_attribution_recorded
```

Source tree:

```text
/home/emmy/openai/pytorch
```

The source tree is available and its HEAD matches the installed Torch git
version `70d99e998b4955e0049d13a98d77ae1b14db1f45`. The tree was already
dirty before this read-only lane, with local edits in `CPUBlas.cpp`,
`Linear.cpp`, `LinearAlgebra.cpp`, `cpu/BlasKernel.cpp`, and `mkldnn/Matmul.cpp`.
No PyTorch source file was modified by this branch.

Candidate source path:

```text
torch.nn.functional.linear / torch._C._nn.linear
  -> aten::linear
  -> Linear.cpp 2D + bias route
  -> at::addmm(*bias, input, weight.t())
  -> native_functions.yaml CPU: addmm_out_cpu
  -> LinearAlgebra.cpp addmm_impl_cpu_
  -> cpublas::gemm
  -> CPUBlas.cpp BF16 cpublas path
  -> BlasKernel.cpp gemm_stub / cpublas_gemm_impl candidates
```

AVX2 contract source candidates were found for BF16 inputs, f32 accumulation,
fused bias as addmm self/beta=1, vectorized reduction helpers, and final BF16
conversion. These are source candidates only:

- Source-level dispatch proven: false.
- Backend identity proven: false.
- AVX2 contract source-confirmed: false.
- AVX2 contract behaviorally consistent: true.
- Source instrumentation recommended next: true.

The source walk maps plausible files and symbols, but it still does not prove
which lower-level path the installed wheel executes for the sampled BF16
o-proj seam. Review is required before authorizing lightweight PyTorch source
instrumentation. No backend selection, implementation, consumer revalidation,
runtime/default/CUDA change, output emission, ladder continuation,
correction/tolerance, or final-logit/all-layer/server/4097 claim is
authorized by this result.

## Non-Goals

- No runtime implementation.
- No backend selection.
- No default routing change.
- No CUDA kernel change.
- No Torch runtime dependency in Rust.
- No consumer revalidation.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
