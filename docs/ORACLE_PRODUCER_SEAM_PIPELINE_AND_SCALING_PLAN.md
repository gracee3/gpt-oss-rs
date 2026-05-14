# Oracle Producer Seam Pipeline And Scaling Plan

Classification: `oracle_producer_seam_pipeline_and_scaling_plan_recorded`

## Scope

- Docs-only higher-level plan.
- Reusable oracle seam attribution pipeline.
- Immediate use case: attention o-proj fused-linear/addmm producer seam.
- Prompt/case: `developer-message-user-smoke`.
- No implementation in this branch.
- No probes in this branch.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.

This document defines how to turn future blocker seams into producer/API
attribution work before choosing implementation or prototype work. It should
not sidetrack into multi-GPU work.

## Pipeline Shape

For each future blocker/operator seam:

1. Classify sampled surfaces across layers.
2. Choose representative positive, negative, and control layers.
3. Record exact source tensor metadata.
4. Compare the official API path against negative controls.
5. Attribute the backend/producer path where feasible.
6. Classify the blocker.
7. Only then choose implementation/prototype work.

Possible blocker classifications:

- API semantics
- layout/stride/contiguity
- dtype/output boundary
- backend dispatch
- source artifact precision
- consumer replay policy
- true runtime implementation gap

The attribution status should preserve source statuses, tensor metadata, API
variant metrics, profiler evidence, and guardrail flags before recommending
any runtime or validation helper work.

## Ladder Stepping Versus Pipeline Attribution

Ladder stepping is layer-by-layer progression. It is useful when the next seam
is unknown and each cleared surface exposes the next layer-local blocker.

Pipeline attribution is horizontal operator/seam closure across sampled layers.
It is better when multiple layers point at the same operator class and the
question becomes producer/API semantics rather than local one-off mismatch
repair.

The ordered-surface pivot moved the project from unknown mismatch collection
to operator-specific workstreams. Future work should prefer pipeline
attribution when the sampled matrix already identifies a coherent seam class.

## First Use Case: O-Proj Fused Linear/AddMM

The o-proj fused-linear/addmm lane is the first use case because:

- The sampled set is already defined: layers 6, 10, 13, 16, 18, and 21.
- Producer/API evidence is coherent across blocked and control layers.
- `module` / `F.linear` / `_C._nn.linear` / addmm-style forms match the
  official reference.
- Explicit matmul/einsum/unfused-bias forms are negative controls.
- Existing local helper policies did not clear the sampled set.
- cuBLAS and cuBLASLt CUDA helpers did not reproduce the reference.

The next step is CPU Torch producer attribution, not another CUDA/helper sweep.

Immediate plan:

```text
docs/FUSED_LINEAR_ADDMM_CPU_PRODUCER_ATTRIBUTION_PLAN.md
```

Future implementation branch, only after review:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-impl
```

## Reusable Status Expectations

Each future producer seam attribution status should record:

- classification
- validation-only/probe-only flags
- prompt/case
- operator
- sampled layers and roles
- official API path
- negative controls
- source tensor metadata
- full-vector metrics
- focus-lane metrics as diagnostics only
- profiler/backend attribution evidence
- whether attribution is conclusive or inconclusive
- next bounded step
- guardrails

Required guardrails:

- `runtime_behavior_changed = false`
- `production_routing_changed = false`
- `cuda_kernels_changed = false`
- `backend_selected = false`
- `consumer_revalidation_authorized = false`
- `output_emitted = false`
- `ladder_continued = false`
- `correction_metadata_applied = false`
- `tolerance_pass = false`
- `final_logit_claim = false`
- `all_layer_claim = false`
- `server_claim = false`
- `context_length_claim = false`

## Scaling And GPU Lane Note

CPU oracle attribution remains the immediate path.

GPU Torch oracle generation is future work only. If a future single-GPU Torch
or runtime run is needed, default to GPU1 because displays are on GPU0. Loading
the full Torch model on a single 24 GB GPU is expected to OOM or be fragile
unless sharding is used.

Before GPU Torch attribution or larger GPU oracle generation, inspect the
multi-GPU layer-sharding worktree docs:

```text
/home/emmy/openai/worktrees/runtime-multi-gpu-layer-sharding/docs/
```

The multi-GPU sharding lane is a future enabler for larger oracle/runtime
validation. It is not a dependency for the immediate CPU producer attribution
plan.

Do not modify that worktree from this branch. Do not import multi-GPU logic
here.

## Current Next Step

Record this docs-only pipeline and then, only after review, implement:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-impl
```

The implementation should be CPU-first and should attribute the producer seam
before any further CUDA/helper or backend-discriminator work.

## CPU Producer Attribution Probe Result

The first pipeline application is recorded in:

```text
/tmp/fused_linear_addmm_cpu_producer_attribution_status.json
```

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_recorded
```

Branch:

```text
oracle/fused-linear-addmm-cpu-producer-attribution-probes
```

Pipeline outcome:

- Sampled layers 6/10/13/16/18/21 all emitted attribution rows.
- The official module/F.linear/_C/addmm/addmm family clears all sampled
  full-vector references.
- Explicit matmul/einsum/unfused-bias variants remain negative controls.
- Environment toggles from source traces cover MKLDNN enabled/disabled and
  thread-count guards.
- CPU profiler evidence observes ATen `linear` and `addmm`, but source-level
  dispatch is not proven.
- The AVX2 contract is consistent with the observed API matrix, but backend
  identity remains unproven.

This keeps the lane in producer-attribution territory. No backend is selected,
no implementation is authorized, no consumer revalidation is authorized, and no
runtime/default/CUDA behavior change follows from the status.

## Producer Attribution Result Update

Classification:

```text
fused_linear_addmm_cpu_producer_attribution_result_update_recorded
```

The first pipeline application has completed on source branch
`oracle/fused-linear-addmm-cpu-producer-attribution-probes` at commit
`2e5e5791a9c353a07ba40929a216056364af164c`.

Result note:

- API semantics are confirmed for the sampled fused-linear/addmm o-proj seam:
  module/F.linear/_C/addmm/addmm clear all sampled full-vector references.
- Explicit matmul/einsum/unfused-bias variants remain the negative controls.
- AVX2 contract consistency is true across layers 6/10/13/16/18/21.
- Lower-level source dispatch remains unresolved.
- Backend identity remains unproven.

Recommended next design branch:

```text
docs/fused-linear-addmm-source-stepthrough-plan
```

The pipeline should now move to source-level step-through planning before any
helper implementation, backend selection, or consumer revalidation.

## Source Step-Through Pipeline Stage

The next pipeline stage is recorded in:

```text
docs/FUSED_LINEAR_ADDMM_SOURCE_STEPTHROUGH_PLAN.md
```

Classification:

```text
fused_linear_addmm_source_stepthrough_plan_recorded
```

This stage follows inconclusive backend attribution: API semantics are known,
the AVX2 contract is behaviorally consistent, but source-level dispatch is not
yet proven. The next branch should be read-only dispatch table attribution:

```text
oracle/fused-linear-addmm-source-dispatch-table-attribution
```

No PyTorch patch/rebuild, Rust helper implementation, backend selection, or
consumer revalidation is authorized by this plan.

## Source Dispatch Table Attribution Result

The read-only dispatch table attribution stage is recorded in:

```text
/tmp/fused_linear_addmm_source_dispatch_table_status.json
```

Classification:

```text
fused_linear_addmm_source_dispatch_table_recorded
```

Pipeline result:

- Dispatch tables for `aten::linear`, `aten::addmm`, `aten::mm`, and
  `aten::matmul` were collected from the installed Torch wheel.
- CPU profiler attribution ran under default, MKLDNN disabled, MKLDNN enabled,
  single-thread, and default-thread-count settings.
- ATen-level operators were visible, including `linear`, `addmm`, `matmul`,
  `mm`, `einsum`, and `bmm`.
- No deeper MKLDNN/oneDNN/DNNL/MKL profiler event name was visible.
- Source-level dispatch and backend identity remain unproven.

This confirms the pipeline stage can collect read-only dispatch evidence
without patching PyTorch, but the next decision still requires review before
any source instrumentation. No backend selection, implementation, consumer
revalidation, runtime/default/CUDA behavior change, output emission, or ladder
continuation follows from this status.

## Source Walk Attribution Result

The read-only source-walk attribution stage is recorded in:

```text
/tmp/fused_linear_addmm_source_walk_attribution_status.json
```

Classification:

```text
fused_linear_addmm_source_walk_attribution_recorded
```

Pipeline result:

- The local source tree `/home/emmy/openai/pytorch` is available and matches
  the installed Torch git version, but it is dirty from existing local ATen
  edits; the source walk did not modify it.
- The candidate source graph maps `aten::linear` to the 2D+bias
  `at::addmm` route, CPU `addmm` to `addmm_out_cpu`, and then to
  `addmm_impl_cpu_` / `cpublas::gemm` / BF16 cpublas and gemm_stub
  candidates.
- AVX2 contract source candidates were found in `CPUBlas.cpp`,
  `cpu/BlasKernel.cpp`, and vectorized helper headers.
- The graph is candidate evidence, not source-level dispatch proof.

Source instrumentation remains the next possible pipeline stage only after
review. No PyTorch patch/rebuild, backend selection, implementation, consumer
revalidation, runtime/default/CUDA behavior change, output emission, or ladder
continuation follows from this status.

## Source Instrumentation Result

The lightweight source-instrumentation stage is recorded in:

```text
/tmp/fused_linear_addmm_source_instrumentation_status.json
```

Classification:

```text
fused_linear_addmm_source_instrumentation_blocked_by_no_source_build
```

Pipeline result:

- A separate PyTorch instrumentation worktree exists at
  `/home/emmy/openai/pytorch-worktrees/fused-linear-addmm-source-instrumentation`.
- The dirty main PyTorch source tree was not modified.
- The instrumentation worktree matches the installed Torch git version and
  remained clean.
- No usable PyTorch source build or editable source install was detected.
- The lane generated and preserved a proposed instrumentation patch but did
  not apply it.
- No trace markers were collected.

The pipeline is blocked on an explicit source-build setup before it can prove
or falsify lower-level dispatch. No backend selection, Rust helper
implementation, consumer revalidation, runtime/default/CUDA behavior change,
output emission, or ladder continuation follows from this status.

## Non-Goals

- No implementation in this branch.
- No producer probe execution in this branch.
- No multi-GPU work in this branch.
- No GPU Torch attribution in this branch.
- No Rust backend.
- No CUDA kernel changes.
- No runtime/default routing changes.
- No consumer revalidation.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
