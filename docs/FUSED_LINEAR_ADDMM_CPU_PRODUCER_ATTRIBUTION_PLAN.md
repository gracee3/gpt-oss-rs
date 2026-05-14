# Fused Linear/AddMM CPU Producer Attribution Plan

Classification: `fused_linear_addmm_cpu_producer_attribution_plan_recorded`

## Scope

- Docs-only plan.
- CPU-first producer attribution.
- Target operator: `attention_o_proj` BF16 linear with bias.
- Prompt/case: `developer-message-user-smoke`.
- Sampled layers: 6, 10, 13, 16, 18, and 21.
- Source prototype status:
  `/tmp/fused_linear_addmm_like_helper_prototype_status.json`.
- Source prototype classification:
  `fused_linear_addmm_like_helper_candidate_no_candidate_selected`.
- No runtime implementation.
- No Rust backend.
- No CUDA kernel changes.
- No consumer revalidation.
- No output emission.
- No ladder continuation.

The validation-only cuBLASLt fused-bias epilogue prototype remains evidence
only. It executed, but did not reproduce the producer/API reference. The next
step is CPU-first producer attribution, not another CUDA/helper sweep.

## Future Branch

The next implementation branch, only after this docs branch is reviewed:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-impl
```

That future branch should remain oracle evidence only. It should not select a
backend, run consumer revalidation, alter runtime/default routing, or emit
layer outputs.

## Primary Question

What exact CPU Torch producer/API path makes `module` /
`torch.nn.functional.linear` / `torch._C._nn.linear` / addmm-style semantics
match the official o-proj reference, while explicit matmul/einsum/unfused-bias
forms and Rust/CUDA helpers do not?

Treat the Torch oracle as CPU-first unless a future probe proves otherwise.
The future implementation should explicitly set and record:

```text
oracle_device = cpu
```

CUDA availability may be recorded, but CUDA must not be used unless a separate
GPU-specific attribution run is authorized.

## CPU-First Attribution Requirements

Future implementation must record:

- `torch.__version__`.
- `torch.__config__.show()`.
- Python executable and venv path.
- CPU thread count and `torch.get_num_threads()`.
- `torch.backends.mkldnn.enabled`.
- oneDNN/MKLDNN availability if visible.
- MKL/OpenMP related environment variables if visible.
- Whether CUDA is available.
- Whether CUDA is used.
- `oracle_device = cpu` unless explicitly overridden by a future approved run.

The probe should avoid accidental CUDA model or tensor allocation:

- Load tensors onto CPU.
- Assert sampled input, weight, bias, and output tensors are CPU tensors before
  comparing variants.
- Record any CUDA visibility separately from actual device use.
- Fail closed if a required tensor unexpectedly lands on CUDA.

## Per-Layer Metadata

For each sampled layer 6/10/13/16/18/21, future implementation should record
the following for input, weight, bias, and output tensors.

Input tensor metadata:

- shape
- dtype
- device
- stride
- contiguity
- storage offset if available
- finite summary

Weight tensor metadata:

- shape
- dtype
- device
- stride
- contiguity
- storage offset if available

Bias tensor metadata:

- shape
- dtype
- device
- stride
- contiguity
- storage offset if available

Output tensor metadata:

- shape
- dtype
- device
- stride
- contiguity

The metadata should be recorded for original tensors and for any safe layout
perturbation variants used as guards.

## API Variants To Compare

Compare the official producer/API reference against:

- Module call if accessible.
- `torch.nn.functional.linear`.
- `torch._C._nn.linear` or equivalent internal linear API if accessible.
- `torch.addmm` form if semantically equivalent and constructible.
- Explicit matmul.
- Explicit einsum.
- Explicit unfused-bias forms.
- Existing negative controls from the producer/API matrix.

All comparisons must be full-vector comparisons against the official
producer/API reference. Focus-lane-only clears are diagnostic only.

## Layout Perturbation Guards

Future implementation should compare:

- Original layout.
- Input contiguous clone.
- Weight contiguous clone.
- Bias clone.
- Safe transpose/layout variants only if semantically equivalent and clearly
  documented.

Every perturbation must record whether the full vector matches the official
producer/API reference. A perturbation that only fixes the focus lane must be
classified as focus-only and rejected for any attribution claim.

## Backend And Profiler Attribution

Prefer CPU-first attribution tools:

- `torch.profiler` with CPU activities.
- oneDNN/DNNL verbose output if available, for example `ONEDNN_VERBOSE=1` or
  `DNNL_VERBOSE=1`.
- `MKL_VERBOSE=1` if relevant.
- PyTorch operator stack and ATen op names if available.

The future status should record whether backend attribution is:

- conclusive
- partially informative
- inconclusive
- blocked by profiler support

The attribution result should distinguish API equivalence from backend
identity. A full-vector clear from `F.linear` is not enough by itself to claim
which CPU backend kernel executed.

## CUDA And GPU Profiler Work

CUDA/GPU profiler work is future-only unless explicitly authorized.

If a future single-GPU Torch or runtime run is needed:

- Default to GPU1 because displays are on GPU0.
- Record the selected GPU explicitly.
- Expect full Torch model loading on a single 24 GB card to be fragile or OOM
  unless sharding is used.

Do not run GPU Torch attribution in this docs branch.

## Future Status Path

Future implementation should write:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Allowed classifications:

- `fused_linear_addmm_cpu_producer_attribution_recorded`
- `fused_linear_addmm_cpu_backend_identified`
- `fused_linear_addmm_cpu_backend_attribution_inconclusive`
- `fused_linear_addmm_cpu_producer_attribution_blocked_by_profiler`
- `fused_linear_addmm_cpu_producer_attribution_failed`

## Future Status Flags

The future status must record:

```json
{
  "validation_only": true,
  "producer_probe": true,
  "oracle_device": "cpu",
  "cuda_available": false,
  "cuda_used": false,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "backend_selected": false,
  "implementation_authorized": false,
  "consumer_revalidation_authorized": false,
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

`cuda_available` should reflect the environment. `cuda_used` must remain false
unless an explicitly authorized future GPU attribution run is requested.

## Decision Rules

- CPU producer attribution comes before more CUDA/helper guessing.
- Full-vector exactness is required.
- Focus-lane-only clears are rejected.
- Explicit matmul/einsum/unfused-bias variants are negative controls, not
  official references.
- Layout perturbations are attribution guards, not policy promotions.
- Profiler/backend labels must be reported as inconclusive if the evidence is
  incomplete.
- No backend may be selected by this plan.
- No consumer revalidation may be authorized by this plan.
- No runtime/default/CUDA behavior change may be authorized by this plan.

## CPU Producer Attribution Probe Results

Branch:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-probes
```

Batch status:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_recorded
```

Layers evaluated:

```text
6, 10, 13, 16, 18, 21
```

API paths tested from the existing CPU producer/API traces:

- module `attn.out(weighted_v)`
- `torch.nn.functional.linear(weighted_v, weight, bias)`
- `torch._C._nn.linear(weighted_v, weight, bias)`
- `torch.addmm(bias, input[1xK], weight.T)`
- `torch.ops.aten.addmm.default` profiler attribution
- explicit matmul plus bias
- explicit einsum plus bias
- `F.linear(..., bias=None) + bias`

Result:

- Module/F.linear/_C/addmm/addmm clear full-vector for all sampled layers.
- Explicit matmul/einsum/unfused-bias variants do not clear any sampled layer.
- Source statuses cover default environment, MKLDNN enabled/disabled, one
  thread, default thread count, layout perturbation guards, and fused-bias
  guards.
- CPU profiler attribution observed `aten::linear` and `aten::addmm`, but did
  not prove the lower-level source dispatch.
- AVX2 contract consistency is recorded for all sampled layers, using the
  extracted contract status
  `/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json`.
- Source-level dispatch proven: false.
- Backend identity proven: false.
- Backend selected: false.
- Implementation authorized: false.
- Consumer revalidation authorized: false.

The CPU probe supports the AVX2-style contract as a plausible attribution
target, but it does not provide source-level dispatch proof. Review this status
before any Rust fused-addmm helper design or implementation.

## CPU Producer Attribution Result Update

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_result_update_recorded
```

Source branch:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-probes
```

Source commit:

```text
2e5e5791a9c353a07ba40929a216056364af164c
```

Source status:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Attribution classification:

```text
fused_linear_addmm_cpu_producer_attribution_recorded
```

Result summary:

- Layers evaluated: 6, 10, 13, 16, 18, and 21.
- API paths tested: module `attn.out`, F.linear, _C linear, `torch.addmm`,
  ATen addmm profiler attribution, explicit matmul/einsum, and unfused
  `F.linear(..., bias=None) + bias`.
- Module/F.linear/_C/addmm/addmm clear full-vector for all sampled layers.
- Explicit matmul/einsum/unfused-bias remain negative controls.
- Environment toggles covered default, MKLDNN on/off, single/default thread,
  layout guards, and fused-bias guard.
- AVX2 contract consistency: true for all sampled layers.
- Source-level dispatch proven: false.
- Backend identity proven: false.
- Backend selected: false.
- Implementation authorized: false.
- Runtime/default/CUDA changes: false.
- Consumer revalidation: false.

Interpretation:

The CPU attribution probe confirms that the official producer/API family
remains module/F.linear/_C/addmm/addmm and that explicit
matmul/einsum/unfused-bias variants remain negative controls. The AVX2
extracted contract is consistent with the observed API matrix across all
sampled layers, but source-level dispatch and backend identity are still not
proven. Therefore this result does not authorize a Rust helper implementation,
backend selection, or consumer revalidation.

Recommended next branch, only after separate approval:

```text
docs/fused-linear-addmm-source-stepthrough-plan
```

Reason: AVX2 contract consistency remains plausible, but source-level dispatch
is unresolved. The next useful step is a source-level step-through/attribution
plan, not another CUDA/helper sweep and not runtime implementation.

## Source Step-Through Plan

The source step-through plan is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_SOURCE_STEPTHROUGH_PLAN.md
```

Classification:

```text
fused_linear_addmm_source_stepthrough_plan_recorded
```

The CPU attribution result led to source step-through planning because AVX2
contract consistency is plausible but source-level dispatch remains unproven.
This plan does not authorize implementation, backend selection, PyTorch
patch/rebuild, consumer revalidation, runtime/default/CUDA behavior changes,
output emission, or ladder continuation.

Recommended next executable branch:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

## Source Dispatch Table Attribution Result

Read-only dispatch table/profiler attribution is recorded in:

```text
/tmp/fused_linear_addmm_source_dispatch_table_status.json
```

Classification:

```text
fused_linear_addmm_source_dispatch_table_recorded
```

Branch:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

The probe inspected dispatch tables for `aten::linear`, `aten::addmm`,
`aten::mm`, and `aten::matmul`, and profiled the BF16 CPU sampled shape under
default, MKLDNN-disabled, MKLDNN-enabled, single-thread, and default-thread
settings. The profiler observed ATen-level `linear`, `addmm`, `matmul`, `mm`,
`einsum`, and `bmm` operators, but no deeper MKLDNN/oneDNN/DNNL/MKL backend
event name.

Outcome:

- Source-level dispatch proven: false.
- Backend identity proven: false.
- AVX2 contract consistency remains true from the prior CPU attribution.
- Source instrumentation is the next candidate step only after review.
- No PyTorch patch/rebuild, backend selection, implementation, consumer
  revalidation, runtime/default/CUDA change, output emission, ladder
  continuation, correction/tolerance, or final-logit/all-layer/server/4097
  claim is authorized.

## Source Walk Attribution Result

Read-only PyTorch source-walk attribution is recorded in:

```text
/tmp/fused_linear_addmm_source_walk_attribution_status.json
```

Classification:

```text
fused_linear_addmm_source_walk_attribution_recorded
```

Branch:

```text
oracle/fused-linear-addmm-source-walk-attribution
```

The source walk used `/home/emmy/openai/pytorch` read-only. The checkout HEAD
matches the installed Torch git version, but the tree is dirty from existing
local edits in relevant ATen files; this branch did not modify it.

Candidate path summary:

- `aten::linear` in `Linear.cpp` routes 2D input with defined bias to
  `at::addmm(*bias, input, weight.t())`.
- `native_functions.yaml` maps CPU `addmm` to `addmm_out_cpu`.
- `LinearAlgebra.cpp` delegates `addmm_out_cpu` to `addmm_impl_cpu_`, which
  contains a `cpublas::gemm` call site.
- `CPUBlas.cpp` and `cpu/BlasKernel.cpp` contain BF16 cpublas/gemm_stub
  candidates that align with the AVX2 contract vocabulary.
- `vec_n.h` provides VectorizedN reduction helpers; `mkldnn/Linear.cpp`
  remains an alternate low-confidence path.

Outcome:

- AVX2 source candidates found: yes.
- Source-level dispatch proven: false.
- Backend identity proven: false.
- Source instrumentation recommended next: true.
- No PyTorch patch/rebuild, backend selection, implementation, consumer
  revalidation, runtime/default/CUDA change, output emission, ladder
  continuation, correction/tolerance, or final-logit/all-layer/server/4097
  claim is authorized.

## Guardrails

- Docs-only in this branch.
- Oracle evidence only in the future implementation branch.
- No runtime implementation.
- No Rust backend.
- No CUDA kernel changes.
- No Torch runtime dependency in Rust.
- No consumer revalidation.
- No backend selection.
- No production/default routing changes.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
