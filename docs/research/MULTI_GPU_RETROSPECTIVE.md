# Multi-GPU layer-sharding retrospective

## Scope and outcome

The historical
[`runtime/multi-gpu-layer-sharding`](https://github.com/gracee3/gpt-oss-rs/tree/166c0573c970334333f3fed567e1c88bf00bfe4f)
branch contains 58 commits between parent `2680a3f` and tip `166c057`. It
targeted a midpoint split of GPT-OSS 20B: embeddings and layers 0-11 on GPU0,
layers 12-23 plus final norm and LM head on GPU1.

The branch produced planning, placement, selective-loading, KV ownership, and
allocation scaffolding. It never reached executable inter-device activation
handoff, end-to-end generation, or output parity. It is therefore historical
research evidence, not a working multi-GPU backend. No source from it is
restored onto the v0.1.0 publication line.

## Reusable designs

### Absolute and local layer identity

A shard may store its first layer at local slot zero, but tensor lookup,
diagnostics, errors, traces, cache identity, and evidence must retain the
absolute model layer number. For the historical midpoint, GPU1 local layer 0
is absolute layer 12. Confusing these identities can silently load a valid
shape under the wrong semantic owner.

### Deterministic whole-layer placement

Every layer should have exactly one stable device owner. The planner must
reject gaps, overlap, unknown device identities, arithmetic overflow, and
noncanonical order before CUDA context creation or payload mapping. Embeddings,
final normalization, and the output head also need explicit ownership; they
are not implicit consequences of a layer range.

### Selective loading

Shape discovery does not grant load authority. A bounded metadata catalog and
exact native/runtime mapping should produce deterministic, owner-specific
whole-tensor actions. Only the owning process or device path consumes a tensor
payload. The archived HET work's bounded catalog, consumer plan, and scoped
mapping transaction supersede the historical ad hoc loaders, but their runtime
implementation is itself archived and incomplete.

### Per-owner KV planning

KV cache must follow the owner of each absolute layer. Capacity arithmetic,
context limits, allocation order, and evidence should be derived per owner
before allocation. A single global byte total is insufficient because it
cannot prove that either device fits its weights, KV state, scratch, and
reserve simultaneously.

### Bounded activation handoff

The available two-GPU host has no CUDA peer access in either direction. A
future sequential layer-boundary transfer would therefore need a bounded,
pinned-host relay with generation identity, explicit enqueue/completion state,
drain-before-reuse, cancellation suppression, and fail-closed quarantine when
synchronization cannot be proven. The historical branch identified the
boundary but did not implement or validate this handoff.

## What should not be ported

- the positional `usize` device-map parser, because stable hardware identity is
  required;
- duplicate SafeTensors header and loading stacks, because later bounded
  metadata work defined stricter ownership contracts;
- the large `sharded_resources` type graph, which predates generation-tagged
  relay and terminal-drain rules;
- old smoke binaries as runtime evidence; individual invariants should become
  current-source fixtures instead; or
- assumptions about P2P, NCCL dispatch, tensor parallelism, graph capture, or
  performance before sequential host-relayed parity exists.

## Model-fit limit

Pure two-GPU layer placement is not a plausible GPT-OSS 120B fit on two 24 GiB
GPUs. Archived metadata measured 65,248,815,744 payload bytes before KV cache,
contexts, scratch, and reserves. No end-to-end 120B execution occurred, so the
archive supports a metadata and capacity conclusion only.

## If research resumes

The smallest honest next package is a CUDA-free placement planner and fixture
suite. It should consume a model layer count plus stable device identities,
assign every absolute layer exactly once, derive embedding/head/tensor/KV
ownership, bind the result to catalog and native-map identities, and reject all
invalid plans before payload access. Runtime wiring, allocation, activation
relay, execution, and public CLI syntax should remain separately gated.

The source-grounded, documentation-only
[`bc8cf36` assessment](https://github.com/gracee3/gpt-oss-rs/blob/bc8cf36f7ba79d318c9264e0f9f4198ac4135c60/docs/het/36-multi-gpu-layer-sharding-salvage-assessment.md)
is the detailed authority for this retrospective.
