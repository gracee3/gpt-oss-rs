# Fused Linear/AddMM Validation Plan Update

Classification:

```text
fused_linear_addmm_validation_plan_update_recorded
```

## Scope

- Docs-only plan update.
- Validation modeling only.
- Target operator: attention o-proj BF16 linear with bias.
- Target evidence: layers 13/16 blocked-family, layer10 pairwise-clear
  control, and layer6 historical context.
- No implementation authorized.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.

## Source Evidence

Docs:

```text
docs/O_PROJ_BLOCKED_FAMILY_DISCRIMINATOR_DESIGN.md
docs/ORDERED_SURFACE_BATCH_MATRIX_7_23.md
docs/ORDERED_SURFACE_BATCH_BACKLOG_DESIGN.md
docs/FUSED_LINEAR_ADDMM_DISCRIMINATOR_DESIGN.md
docs/OFFICIAL_LINEAR_BACKEND_DISCRIMINATOR_DESIGN.md
```

Note: `docs/FUSED_LINEAR_ADDMM_DISCRIMINATOR_DESIGN.md` is not present on
this branch. This active plan is recorded as:

```text
docs/FUSED_LINEAR_ADDMM_VALIDATION_PLAN.md
```

Statuses:

```text
/tmp/o_proj_producer_api_probes_13_16_10_status.json
/tmp/layer13_attention_oproj_api_probe_status.json
/tmp/layer16_attention_oproj_api_probe_status.json
/tmp/layer10_attention_oproj_api_probe_status.json
/tmp/layer6_attention_oproj_api_probe_status.json
/tmp/layer6_official_linear_backend_discriminator_probe_status.json
```

## Confirmed Producer/API Pattern

| Layer | Class | Focus lane | module/F.linear/_C/addmm | matmul/einsum/unfused bias | Layout/fused-bias sensitive | Interpretation |
| --- | --- | ---: | --- | --- | --- | --- |
| 13 | blocked-family | 151 | 0 mismatches | 819 mismatches | yes | fused-linear/addmm pattern |
| 16 | blocked-family | 2666 | 0 mismatches | 763 mismatches | yes | fused-linear/addmm pattern |
| 10 | pairwise-clear control | 915 | 0 mismatches | 822 mismatches | yes | fused-linear/addmm pattern |
| 6 | historical blocker | 22 | 0 mismatches in producer API probe | 826 mismatches in matmul/einsum/unfused class | yes | original fused-linear/addmm pattern |

The final sampled matrix expands this evidence to layer18 and layer21:

```text
docs/O_PROJ_PRODUCER_API_FINAL_MATRIX.md
/tmp/o_proj_producer_api_probes_18_21_status.json
```

Layer18 and layer21 also clear through module/F.linear/_C/addmm and fail
through explicit matmul/einsum or unfused-bias forms. The added evidence does
not authorize implementation, backend selection, consumer revalidation, or
runtime/default/CUDA behavior changes.

The producer API result is the same for blocked and pairwise-clear classes.
Local pairwise, reverse, and current sweep results are local approximations,
not producer-backend identities. Explicit matmul/einsum and unfused-bias forms
are not valid official references for sampled o-proj layers. The official
reference should be treated as fused linear/addmm semantics with original BF16
layout and fused bias.

## Revised Interpretation

The question is no longer whether blocked layers match layer6. They do. The
next question is how to model official fused-linear/addmm semantics in
validation without changing production runtime.

- Pairwise-clear layers may clear because their local approximation happens to
  land on the fused-linear/addmm BF16 result.
- Blocked layers may fail because the local approximation lands on adjacent
  BF16 values or creates collateral mismatches.
- Both cases may share the same true official backend.
- Therefore a local policy taxonomy is not an official backend taxonomy.

## Validation Modeling Problem

Need a validation-only representation of official BF16 fused linear/addmm
semantics for:

```text
y = linear(weighted_v, o_proj_weight, o_proj_bias)
```

where:

- Input is BF16 weighted-V.
- Weight is BF16 o-proj weight.
- Bias is BF16 o-proj bias.
- Output boundary is BF16.
- Fused bias matters.
- Input layout/contiguity matters.
- Matmul/einsum/unfused-bias cannot be used as reference.

## Candidate Validation Modeling Paths

### Option 1 - Producer/API Reference Oracle

Use producer-side F.linear/addmm outputs as explicit oracle seams for o-proj
blocked-family validation.

Pros:

- Exact official semantics.
- Avoids guessing Rust backend.
- Useful for design/proof.

Cons:

- Not a Rust backend.
- Not production implementation.
- Requires producer artifacts per layer/operator.

### Option 2 - Validation-Only Fused-AddMM Backend Discriminator

Add a Rust status mode that compares existing Rust/CUDA candidate helpers
against the producer/API fused-addmm reference.

Pros:

- Keeps Torch out of runtime.
- Can classify candidate backends without changing production.

Cons:

- May still not reproduce exact fused-addmm semantics.
- Must not choose a backend from focus-lane-only clears.

### Option 3 - Explicit API/Source Metadata Contract First

Before new backend work, require every o-proj status to record:

- Input/weight/bias contiguity.
- Fused vs unfused bias.
- Source API class.
- Layout perturbation guards.
- Full-vector mismatch metrics.
- Focus-lane metrics only as secondary.

Pros:

- Low-risk.
- Prevents overclaiming.
- Improves later implementation discipline.

Cons:

- Does not clear blocked layers by itself.

Recommended plan: Option 3 first, then Option 2, while keeping Option 1 as
oracle reference.

## Future Status Contract

Future validation status shape:

```json
{
  "classification": "fused_linear_addmm_validation_status_recorded",
  "validation_only": true,
  "runtime_behavior_changed": false,
  "production_routing_changed": false,
  "cuda_kernels_changed": false,
  "operator": "attention_o_proj",
  "layers": [13, 16, 10],
  "official_reference": {
    "api": "module/F.linear/_C/addmm",
    "fused_bias": true,
    "layout_sensitive": true,
    "dtype": "torch.bfloat16"
  },
  "source_metadata_required": [
    "weighted_v dtype/shape/stride/contiguity",
    "weight dtype/shape/stride/contiguity",
    "bias dtype/shape/stride/contiguity",
    "official output dtype/shape"
  ],
  "candidate_results": [],
  "full_vector_required": true,
  "focus_lane_only_accepted": false,
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

## Proof Gates Before Any Code Implementation

1. Producer/API reference statuses for at least one blocked layer and one
   pairwise-clear control.
2. Full-vector comparison only.
3. Explicit fused-bias metadata.
4. Explicit layout/contiguity metadata.
5. Negative controls preserved: strict/default cleared layers 12/14/15/22.
6. No focus-lane-only backend selection.
7. No collateral mismatch promotion.
8. No production/default routing change.
9. No CUDA kernel change.
10. No output emission.

## Recommended Next Branch

Recommend docs/status-only next, not backend implementation:

```text
validation/fused-linear-addmm-status-scaffold
```

Scope:

- Status/scaffold only.
- Consume existing producer/API probe statuses.
- Emit normalized fused-linear/addmm validation status.
- No new probes.
- No consumer revalidation.
- No backend selection.
- No runtime/default/CUDA changes.

Alternative:

```text
docs/fused-linear-addmm-status-scaffold-design
```

Use the alternative if implementation is not explicitly approved.

## Status Scaffold Design

The docs-only status scaffold design is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_STATUS_SCAFFOLD_DESIGN.md
```

It defines the future status-only contract for consuming existing
producer/API probe statuses and emitting a normalized fused-linear/addmm
validation status.

## Status Scaffold Mode

The validation-only scaffold mode is available as:

```text
--mode fused-linear-addmm-status-scaffold
```

It emits:

```text
/tmp/fused_linear_addmm_status_scaffold.json
```

Classification:

```text
fused_linear_addmm_status_scaffold_recorded
```

The mode reads existing JSON probe statuses only. It remains
non-implementation and non-backend-selecting: no model execution, no CUDA
execution, no consumer revalidation, no runtime/default/CUDA behavior change,
and no output emission or ladder continuation.

## Backend Discriminator Design Update

The docs-only backend-discriminator design update is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_BACKEND_DISCRIMINATOR_DESIGN.md
```

Classification:

```text
fused_linear_addmm_backend_discriminator_design_update_recorded
```

It turns the final Workstream A producer/API matrix into requirements for a
future validation-only backend discriminator. It does not authorize
implementation, backend selection, consumer revalidation, runtime/default/CUDA
behavior changes, output emission, or ladder continuation.

## Backend Discriminator Status Mode

The validation-only status-readiness mode is recorded as:

```text
--mode fused-linear-addmm-backend-discriminator-status
```

It emits:

```text
/tmp/fused_linear_addmm_backend_discriminator_status.json
```

Classification:

```text
fused_linear_addmm_backend_discriminator_status_recorded
```

The mode consumes the scaffold plus the layer6, 13/16/10, and 18/21
producer/API statuses. It records readiness, candidate families, and decision
rules only: no candidate execution, backend selection, consumer revalidation,
runtime/default/CUDA change, output emission, or ladder continuation.

## Backend Candidate Comparator

The validation-only candidate comparator is recorded as:

```text
--mode fused-linear-addmm-backend-discriminator
```

It emits:

```text
/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json
```

Classification:

```text
fused_linear_addmm_backend_discriminator_no_candidate_selected
```

The comparator executed existing local o-proj candidate helpers against the
producer/API fused-linear/addmm references for layers 6, 10, 13, 16, 18, and
21. No selectable candidate cleared the full sampled set. Pairwise is the best
partial local candidate, clearing layer10 and layer21 while retaining
collateral mismatches on layer6/13/16/18.

No production backend is selected. No runtime/default/CUDA behavior change,
consumer revalidation, output emission, or ladder continuation is authorized.

## Helper Design Follow-Up

The missing validation-only helper work after the no-candidate-selected result
is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_HELPER_DESIGN.md
```

Classification:

```text
fused_linear_addmm_helper_design_recorded
```

It defines the official reference contract, minimum sampled set, future helper
families, status JSON contract, and decision rules for any future helper
candidate run. The next branch is
`validation/fused-linear-addmm-helper-candidate` only if separately approved.

No implementation, backend selection, consumer revalidation,
runtime/default/CUDA behavior change, output emission, or ladder continuation
is authorized by the helper design.

## Helper Candidate Result

The validation-only helper candidate mode is recorded in:

```text
/tmp/fused_linear_addmm_helper_candidate_status.json
```

Classification:

```text
fused_linear_addmm_helper_candidate_no_candidate_selected
```

The run covers layers 6, 10, 13, 16, 18, and 21. It reruns existing helper
baselines and executes available cuBLAS BF16 validation-only helpers. No helper
matches the module/F.linear/_C/addmm fused-bias original-layout reference across
the full sampled set.

Candidate summary:

- `pairwise_f32_bf16_output`: clears layer10 and layer21 only.
- `cublas_bf16_pedantic_or_deterministic`: clears layer16 only and is the best
  partial by total mismatch count.
- `cublas_bf16_tensor_op`: broad collateral mismatches.
- Diagnostic/evidence-only helpers remain non-selectable.

No backend is selected. No consumer revalidation, runtime/default/CUDA behavior
change, output emission, ladder continuation, correction metadata, tolerance
pass, final-logit, all-layer, server, or 4097-token claim is authorized.

## Fused-AddMM-Like Helper Design

The next docs-only implementation design is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_LIKE_HELPER_IMPLEMENTATION_DESIGN.md
```

Classification:

```text
fused_linear_addmm_like_helper_implementation_design_recorded
```

It narrows future work to a helper that models the producer/API
module/F.linear/_C/addmm BF16 fused-bias original-layout reference directly.
Another local accumulation sweep, focus-lane promotion, diagnostic promotion,
and evidence-only promotion are explicitly rejected. The next executable
branch, only if separately approved, is
`validation/fused-linear-addmm-like-helper-prototype`.

No runtime/default/CUDA behavior change, backend selection, consumer
revalidation, output emission, ladder continuation, correction metadata,
tolerance pass, final-logit, all-layer, server, or 4097-token claim is
authorized.

## Fused-AddMM-Like Helper Prototype

The validation-only cuBLASLt fused-bias prototype is recorded in:

```text
/tmp/fused_linear_addmm_like_helper_prototype_status.json
```

Mode:

```text
--mode fused-linear-addmm-like-helper-prototype
```

Classification:

```text
fused_linear_addmm_like_helper_candidate_no_candidate_selected
```

The prototype candidate `cublaslt_bf16_matmul_bias_epilogue` executed on the
minimum sampled set 6/10/13/16/18/21. It did not clear the full sampled set and
did not clear any sampled layer full-vector exactly. The matrix recorded
collateral mismatches on all six layers, with total sampled mismatches 8432.

Conclusion:

- cuBLASLt API availability is confirmed for this validation binary.
- A plain BF16 cuBLASLt matmul plus bias epilogue is not sufficient to match
  the producer/API module/F.linear/_C/addmm reference.
- No backend is selected.
- No consumer revalidation is authorized.
- No runtime/default/CUDA behavior change, output emission, ladder
  continuation, correction metadata, tolerance pass, final-logit, all-layer,
  server, or 4097-token claim follows from this result.

## CPU Producer Attribution Plan

CPU-first producer attribution and reusable oracle seam planning are recorded
in:

```text
docs/FUSED_LINEAR_ADDMM_CPU_PRODUCER_ATTRIBUTION_PLAN.md
docs/ORACLE_PRODUCER_SEAM_PIPELINE_AND_SCALING_PLAN.md
```

Classifications:

```text
fused_linear_addmm_cpu_producer_attribution_plan_recorded
oracle_producer_seam_pipeline_and_scaling_plan_recorded
```

The cuBLASLt prototype result remains validation evidence only. Plain cuBLASLt
fused-bias epilogue did not reproduce the producer/API reference. Treat the
Torch oracle as CPU-first unless a future probe proves otherwise.

Next implementation branch, only after review:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-impl
```

The next step is CPU-first producer attribution, not another CUDA/helper sweep.
No backend is selected. No implementation is authorized by this docs branch. No
consumer revalidation or runtime/default/CUDA behavior change is authorized.

## CPU Producer Attribution Result

Status:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_backend_attribution_inconclusive
```

The CPU-first probe confirms the producer/API reference class on CPU: module
call, `torch.nn.functional.linear`, `torch._C._nn.linear`, and fused
`torch.addmm` clear the full official o-proj vector for layers 6, 10, 13, 16,
18, and 21. Explicit matmul/einsum/unfused-bias forms remain full-vector
negative controls. CPU profiler output records ATen-level operators but does
not identify a concrete CPU backend strongly enough to select one.

This records oracle evidence only. No backend is selected, no consumer
revalidation is authorized, and no runtime/default/CUDA behavior change follows
from the result.

## AddMM Boundary Localization Result

Status:

```text
/tmp/fused_linear_addmm_addmm_boundary_localization_status.json
```

Classification:

```text
fused_linear_addmm_addmm_boundary_inconclusive
```

The boundary-localization probe decomposes addmm on CPU across layers 6, 10,
13, 16, 18, and 21. Fused addmm with bias clears the full vector everywhere;
zero-bias addmm plus a separate bias add does not. Explicit matmul/einsum plus
bias and explicit unfused-bias remain negative controls. Because additional
small core/einsum and layout guard signals are present, the result remains
inconclusive rather than selecting a single boundary or backend.

No backend is selected and no consumer revalidation, runtime/default/CUDA
behavior change, output emission, or ladder continuation is authorized.

## Fused-Bias Arithmetic Contract Result

Status:

```text
/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json
```

Classification:

```text
fused_linear_addmm_fused_bias_arithmetic_contract_inconclusive
```

The CPU-only arithmetic-contract probe tests whether the clearing fused addmm
result can be reconstructed by explicit bias-placement and BF16-rounding
models. It supports bias entering before the final observable BF16 rounding,
but does not localize a complete accumulation/product policy across all
sampled layers.

Summary:

- Addmm with fused bias still clears every sampled full vector.
- Zero-bias addmm plus bias, explicit matmul plus bias, explicit einsum plus
  bias, and explicit unfused BF16 bias remain negative controls.
- Pre-round-bias variants provide lane-level support on layers 6, 10, 13, 16,
  and 21 and full-vector support on layers 10, 13, and 16.
- No explicit arithmetic contract clears layers 6, 10, 13, 16, 18, and 21 as a
  set, so no backend or runtime policy is selected.

This result should feed future validation-only discriminator requirements. It
does not authorize consumer revalidation, output emission, ladder continuation,
or runtime/default/CUDA behavior changes.

## Official API Seam Synthesis

Decision record:

```text
docs/FUSED_LINEAR_ADDMM_OFFICIAL_API_SEAM_SYNTHESIS.md
```

Classification:

```text
fused_linear_addmm_official_api_seam_synthesis_recorded
```

The current validation stance is to treat Workstream A as an official CPU Torch
API seam: module/F.linear/_C/addmm clears the sampled full vectors, fused bias
before final observable BF16 output is the strongest arithmetic signal, and
explicit matmul/einsum/unfused-bias remain negative controls. Existing
helpers, cuBLAS/cuBLASLt candidates, and explicit arithmetic variants have not
produced a global runtime candidate.

No further blind sweeps are recommended from this evidence. No backend is
selected and no consumer revalidation or runtime/default/CUDA behavior change
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

The plan defines a staged feasibility path rather than a validation
implementation. Gate A checks CPU Torch addmm dispatch-stability. Gate B, only
after Gate A review, searches a bounded Rust-replayable CPU arithmetic policy
space. Gate C, only after one global CPU policy clears, mirrors that exact
policy in CUDA for validation-only comparison.

The plan explicitly prohibits per-layer policy selection, focus-lane promotion,
tolerance passes, f64 diagnostic promotion, producer/API seam promotion as a
runtime backend, consumer revalidation, and runtime/default/CUDA changes.

## CPU Dispatch-Stability Result

Status:

```text
/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json
```

Classification:

```text
fused_linear_addmm_cpu_dispatch_stability_stable
```

Gate A passed: default, Torch thread-count, interop-thread, MKLDNN, OMP, MKL,
and combined OMP/MKL fresh-process configurations all reproduced the same
addmm full vectors for sampled layers 6, 10, 13, 16, 18, and 21. Every tested
configuration also matched the official o-proj artifact exactly.

This means CPU dispatch instability is not currently blocking feasibility. It
does not select a backend and does not authorize Gate B, CUDA mirror work,
consumer revalidation, or runtime/default/CUDA behavior changes.

## Rust CPU Policy Synthesis Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_synthesis_partial_only
```

Gate B executed as a validation-only Rust CPU policy search. It evaluated 350
named policy records and found only partial/per-layer evidence: layers 10, 13,
and 21 have selectable full-vector clears, while layers 6, 16, and 18 retain
full-vector mismatches under the best selectable replays. Therefore no single
Rust-replayable policy clears the sampled set, and the lane should not proceed
to CUDA mirror work without a new design review.

Focus-lane-only clears, diagnostic-only f64, BF16-product evidence, and
evidence-only output policies remain rejected. No backend is selected, no
consumer revalidation is authorized, and no runtime/default/CUDA behavior
change follows.

## Rust CPU Policy Closure Audit Result

Status:

```text
/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json
```

Classification:

```text
fused_linear_addmm_rust_cpu_policy_closure_no_global_policy
```

The closure audit completed the bounded Gate B replay coverage by executing all
238 previously unrun full-vector replays for selectable focus-clearing
candidates. None were blocked, and no single Rust CPU policy cleared the full
sampled set 6/10/13/16/18/21.

The best near-global candidate still leaves five full-vector mismatches across
the sampled set. Residuals on layers 6, 16, and 18 are one BF16 ULP or less in
the simple residual analysis, but no shared residual rule was localized. The
recommended state is `stop_policy_lane_preserve_official_api_seam`.

This closes the current Rust CPU policy synthesis lane. Continue treating the
CPU Torch module/F.linear/_C/addmm seam as the oracle reference; do not proceed
to CUDA mirror work or consumer revalidation from this result.

## PyTorch Source Attribution Plan

Plan:

```text
docs/FUSED_LINEAR_ADDMM_PYTORCH_SOURCE_ATTRIBUTION_PLAN.md
```

Classification:

```text
fused_linear_addmm_pytorch_source_attribution_plan_recorded
```

Future validation research may inspect the installed Torch wheel and matching
PyTorch/ATen source to identify the CPU BF16 addmm implementation path behind
the official seam. This is a planning record only: no PyTorch clone, venv,
source build, runtime implementation, consumer revalidation, CUDA mirror work,
or default behavior change is authorized.

## Non-Goals

- No runtime implementation.
- No default routing change.
- No CUDA kernel change.
- No output emission.
- No ladder continuation.
- No correction metadata.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency in Rust.
