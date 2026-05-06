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

## CPU Producer Attribution Result

Implementation branch:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-impl
```

Status:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_backend_attribution_inconclusive
```

The CPU-first probe reconstructs sampled o-proj seams from ordered attention
bundle JSON vectors plus checkpoint o-proj weights/biases on CPU. It records
`oracle_device = "cpu"` and `cuda_used = false`.

Result:

- Layers 6, 10, 13, 16, 18, and 21 reproduce the official o-proj full vector
  exactly through module call, `torch.nn.functional.linear`,
  `torch._C._nn.linear`, and fused `torch.addmm`.
- Explicit matmul, explicit einsum, and explicit unfused-bias forms remain
  negative controls for every sampled layer.
- CPU profiler records ATen operator names including `aten::linear`,
  `aten::addmm`, `aten::matmul`, and `aten::einsum`.
- Backend verbose capture was attempted, but backend identity remains
  inconclusive.
- Layout perturbations are attribution guards only; focus-lane-only results
  remain diagnostic and are not promoted.

The result explains the producer/API versus explicit matmul/einsum/unfused-bias
split on CPU, but it does not identify or select a backend. It does not
authorize consumer revalidation, runtime/default/CUDA behavior changes, output
emission, or ladder continuation.

## AddMM Boundary Localization Result

Implementation branch:

```text
oracle/fused-linear-addmm-addmm-boundary-localization
```

Status:

```text
/tmp/fused_linear_addmm_addmm_boundary_localization_status.json
```

Classification:

```text
fused_linear_addmm_addmm_boundary_inconclusive
```

Result:

- `torch.addmm(bias, input_2d, weight_t_2d)` clears the full official vector
  for layers 6, 10, 13, 16, 18, and 21.
- `torch.addmm(zero_bias, input_2d, weight_t_2d) + bias` does not clear any
  sampled layer and matches the explicit unfused-bias negative-control class.
- The zero-bias addmm core matches `input @ weight.T` for every sampled layer,
  but small addmm/einsum-core differences appear on layers 10, 13, 16, 18, and
  21.
- Noncontiguous same-shape weight layout remains a guard signal on layers 10,
  13, 16, and 18.

Interpretation: fused-bias handling is the strongest localization signal, but
core/einsum and layout guard signals are also present. The probe therefore
records inconclusive localization rather than claiming a single mechanism. No
backend is selected and no consumer revalidation or runtime/default/CUDA change
is authorized.

## Fused-Bias Arithmetic Contract Result

Implementation branch:

```text
oracle/fused-linear-addmm-fused-bias-arithmetic-contract
```

Status:

```text
/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json
```

Classification:

```text
fused_linear_addmm_fused_bias_arithmetic_contract_inconclusive
```

Result:

- The CPU-only arithmetic-contract probe confirms the previous universal
  signal: fused `torch.addmm(bias, input_2d, weight_t_2d)` clears all sampled
  layers, while zero-bias addmm plus a separate bias add, explicit matmul plus
  bias, explicit einsum plus bias, and explicit unfused BF16 bias remain
  full-vector negative controls.
- Bias-before-output-rounding is supported as the strongest arithmetic signal.
  Lane-level support appears on layers 6, 10, 13, 16, and 21, and full-vector
  pre-round-bias support appears on layers 10, 13, and 16.
- No explicit arithmetic model clears all selected lanes and full vectors
  across layers 6, 10, 13, 16, 18, and 21. Layer18 remains the key non-clearing
  sampled case for this contract.

Interpretation: Torch addmm behaves consistently with bias participating before
the final observable BF16 rounding, but the exact accumulation/product policy
is not localized. No backend is selected, no implementation is authorized, and
no consumer revalidation or runtime/default/CUDA change follows from this
oracle evidence.

## Official API Seam Synthesis

Decision record:

```text
docs/FUSED_LINEAR_ADDMM_OFFICIAL_API_SEAM_SYNTHESIS.md
```

Classification:

```text
fused_linear_addmm_official_api_seam_synthesis_recorded
```

The synthesis preserves Workstream A as an official CPU Torch API seam:
module/F.linear/_C/addmm with BF16 input, BF16 weight, BF16 bias, fused bias
before the final observable BF16 output, and full-vector exactness required.
Explicit matmul/einsum/unfused-bias remain negative controls. The exact CPU
backend identity and a single global accumulation/product policy remain
unresolved.

No backend is selected. No implementation, consumer revalidation,
runtime/default/CUDA behavior change, output emission, or ladder continuation
is authorized.

## Rust/CUDA Policy Feasibility Plan

Follow-up plan:

```text
docs/FUSED_LINEAR_ADDMM_RUST_CUDA_POLICY_FEASIBILITY_PLAN.md
```

Classification:

```text
fused_linear_addmm_rust_cuda_policy_feasibility_plan_recorded
```

The plan keeps this CPU attribution result as oracle seam evidence only. The
next permissible implementation branch is CPU Torch dispatch-stability, not
Rust/CUDA policy code. Rust CPU policy synthesis is gated behind stable Torch
CPU addmm outputs, and CUDA mirroring is gated behind one global Rust CPU
policy clearing layers 6, 10, 13, 16, 18, and 21 full-vector exactly.
