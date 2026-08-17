# Multi-GPU layer-sharding salvage assessment

**Status:** documentation-only assessment on `main` at
`249abfbf5f21dddb434a7975c02df396e0608dc7`. No old implementation was
ported, no public surface changed, and no model or GPU workload was run.

## Purpose and boundary

This assessment asks whether the historical
`runtime/multi-gpu-layer-sharding` work contains concepts worth adapting to the
current heterogeneous architecture. It does not revive that branch, merge its
history, or authorize a layer-sharded runtime.

The future mode considered here assigns complete transformer layers to CUDA
devices and transfers the hidden state at a layer boundary. It is distinct
from the current heterogeneous mode, in which GPU0 remains the layer owner and
selected experts have static CPU/GPU0/GPU1 owners. The existing heterogeneous
path remains canonical.

## Source lineage and achieved boundary

The exact historical segment reviewed is:

```text
branch: runtime/multi-gpu-layer-sharding
base parent: 2680a3fdf44e401dfd8368e9388907ae81bba226
first multi-GPU commit: 044d7f31b8cb301724c8d486646bc0941ba67336
tip: 166c0573c970334333f3fed567e1c88bf00bfe4f
range: 2680a3f..166c057
commits: 58
net diff: 23 files, 24,257 insertions, 138 deletions
```

The target was GPT-OSS 20B split at the midpoint: embeddings and layers 0–11
on GPU0, then layers 12–23, final norm, and LM head on GPU1. It explicitly
excluded tensor parallelism, collectives, per-tensor splitting, kernel changes,
and production-default changes.

The branch established an inert device-map parser, whole-tensor placement and
KV-cache plans, filtered loading helpers, per-device resource ownership, and
increasingly complete real-model allocation/status smoke paths. Its final
checkpoint still classified executable layer construction, activation
handoff, graph decode, and output parity as deferred. It therefore supplies
reconnaissance and allocation evidence, not a working layer-sharded runtime.

## Salvage decisions

| Historical area | Current-main disposition | Decision |
| --- | --- | --- |
| `device_map.rs` | Current placement uses stable CUDA identities and an expert-placement manifest rather than positional device ids. | Retain coverage, non-overlap, deterministic order, embedding-first/final-head-last, and pre-allocation validation requirements. Do not port the parser or its `usize` device identity. |
| `shard_plan.rs` | Current shard catalog and consumer plan already provide bounded metadata and placement-bound action planning. | Retain whole-layer tensor ownership, absolute-to-local layer mapping, per-owner KV planning, and checked byte reports. Re-express them later as a layer-placement policy over current metadata. |
| `model_loader/safetensor_headers.rs` | [`SafeTensorShardCatalog`](31-h3-bounded-shard-catalog.md) validates bounded indexes, headers, identities, tensor ranges, and order. | Superseded; do not port. |
| `model_loader/shard_weights.rs` and filtered loader edits | Current native mapping, `GptOssShardConsumerPlan`, `ScopedShardConsumerTransaction`, and owner-selective construction define stricter mapping and lifetime contracts. | Preserve selective whole-tensor consumption as a requirement, but do not transplant the loaders or stores. |
| `sharded_resources.rs` | The 5,838-line status/allocation scaffold predates current stable identities, bounded relay, terminal-drain proof, and quarantine semantics. | Historical evidence only. Mine allocation categories and failure cases; do not port the type graph or status schema. |
| `fused_f16.rs` and runner extraction | Current CUDA runner, loading, dtype, and heterogeneous ownership paths have materially diverged. | Do not port. Re-audit each required conversion or scratch allocation only after a new planner exists. |
| Three multi-GPU smoke binaries | They record useful 20B allocation sequencing and absolute-layer invariants but exercise the retired scaffolding. | Historical evidence only. Translate individual assertions into future tests instead of restoring the binaries. |
| Host-staged activation handoff | The host has verified no CUDA P2P in either direction, and the current path has bounded pinned relay plus generation/drain ownership. | Retain the boundary concept. Redesign transfer around the current pinned-host relay and fail-closed lifetime rules, not ad hoc stream synchronization. |
| Original 8,312-line plan | It records why each allocation category exists, but most source paths and assumptions are stale. | Keep the branch as the authoritative archive. Carry forward only conclusions and cited invariants, not the document wholesale. |

The strongest reusable idea is the separation between absolute model identity
and shard-local execution identity. A future GPU1 shard may store its first
layer at local index zero, but every tensor lookup, trace, failure, and evidence
record must continue to identify it as absolute layer 12. Shape visibility must
not imply ownership or load authority.

## Recommended future architecture

Treat layer sharding as a separate, opt-in mode rather than generalizing the
active heterogeneous runtime immediately:

```text
native shard catalog + exact native/runtime mapping
    -> layer-placement policy bound to stable CUDA identities
    -> deterministic whole-tensor consumer actions per layer owner
    -> owner-local weights, KV cache, metadata, kernels, streams, and scratch
    -> sequential layer-group execution
    -> bounded pinned-host activation relay at each ownership boundary
```

The first proof target should remain GPT-OSS 20B on the two local RTX 3090s.
That matches the only real-model allocation evidence on the historical branch
and isolates the ownership/handoff problem from CPU expert execution. It must
not run at the same time as heterogeneous expert placement in its first form.

Pure two-GPU layer placement is not a credible 120B fit on this host. The
recorded native 120B payload is 65,248,815,744 bytes (60.7677 GiB), already
larger than the two GPUs' aggregate nominal 48 GiB before KV cache, contexts,
scratch, and reserves. The current unpassed 120B existence envelope is
feasible only by assigning experts across GPU0, GPU1, and CPU. Composition
between layer and expert placement can be reconsidered after each mode has an
independently proven ownership model; it is not part of the first
layer-sharding design.

The future mode should continue to exclude tensor parallelism, NCCL dispatch,
P2P assumptions, CUDA graph capture, and performance claims until a sequential
host-relayed boundary is correct and repeatable.

## Verdict and next bounded work

**Verdict:** salvage the architectural invariants and historical allocation
evidence, not the implementation. A merge, full-range cherry-pick, or direct
restoration would reintroduce duplicate loaders and weaker ownership contracts
while obscuring the current heterogeneous work.

If layer sharding becomes active work, the next bounded package should be a
CUDA-free layer-placement planner design and fixture suite. It should:

- consume model layer count plus stable CUDA identities;
- assign every layer exactly once while preserving absolute layer ids;
- derive embedding, final-head, whole-tensor, and KV-cache ownership;
- bind its result to the current catalog/native-map identity;
- reject gaps, overlap, unknown devices, arithmetic overflow, and noncanonical
  ordering before any CUDA context or payload mapping; and
- leave all runtime/CLI wiring, payload consumption, allocation, and execution
  deferred behind a separate review gate.

No public `DeviceMap`, CLI syntax, or compatibility promise should be created
until that pure plan has a reviewed contract and current-source fixtures.

## Assessment validation

The assessment used Git/source inspection only. It verified the pinned range,
the 58-commit count and diff summary, the historical plan's explicit
non-execution boundary, and the current stable-device, bounded-catalog,
consumer-plan, scoped-transaction, owner-selective, and no-P2P surfaces.

No Cargo build, GPU probe, checkpoint access, model execution, download,
package installation, or system change was needed or performed.
