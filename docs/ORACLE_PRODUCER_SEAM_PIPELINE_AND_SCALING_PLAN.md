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
