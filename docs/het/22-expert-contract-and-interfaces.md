# Expert contract and crate interfaces

These signatures are planning pseudocode, not compilable production code. The
semantic authority is [document 13](13-exact-expert-contract.md) and
`CpuModel::moe_batch` in
`crates/gpt-oss-model-runner/src/cpu_runner.rs`.

## Exact backend contract

Every backend receives BF16 activation rows and one or more selected experts.
It must implement the same per-route operation:

1. native MXFP4 gate/up projection accumulates in f32, adds the BF16-defined
   bias, and rounds each result to BF16;
2. GPT SwiGLU applies alpha `1.702`, gate clamp 7, up clamp `[-7,7]`, `up+1`,
   and the current scalar operation/BF16 round points;
3. native MXFP4 down projection accumulates in f32, adds bias, and returns an
   **unweighted BF16** `[2880]` output.

The backend does not apply routing weight, reduce ranks, add residual, advance
K/V, sample, or publish. Prefill and decode share arithmetic. They differ only
in bucket/packing strategy and promoted kernel shapes.

## Proposed types and ownership

```rust
// gpt-oss-gpu::device -- durable value; ordinal is resolved separately.
StableCudaDeviceId { pci_bdf, expected_name, compute_capability, minimum_vram }
ResolvedCudaDevice { stable_id, transient_ordinal, context }

// gpt-oss-model-runner::heterogeneous::placement
GptOssExpertKey { layer: u16, expert: u16 }
ExpertOwner = Cpu { pool: CpuPoolId }
            | LayerOwnerGpu { device: StableCudaDeviceId }
            | RemoteGpu { device: StableCudaDeviceId }
GptOssExpertPlacementManifestV1 { model, devices, assignments, budgets, policy }
ResolvedExpertPlacement { manifest_hash, resolved_devices, assignments }

// GPT-OSS-specific on purpose. Weight bits are not widened in the descriptor.
GptOssRouteDescriptor {
    source_row: u32,
    route_rank: u8,       // exactly 0..3
    expert_id: u16,       // 0..31 or 0..127
    weight_bf16_bits: u16,
    activation_slot: u32,
}
GptOssRoutedBatch<'step> {
    layer, phase, hidden_size, activation_bf16, routes, placement_epoch
}

ExpertRepresentationTag = CpuMxfp4InterleavedX8V2
                        | CudaNativeMxfp4BlocksScalesV1
CpuExpertWeightHandle<'model> { key, x8_gate_up, x8_down, biases, identity }
CudaExpertWeightHandle<'model> { key, blocks, scales, biases, device, identity }
GptOssExpertWeightHandle<'model> = Cpu(...) | Cuda(...)

PackedOwnerWork<'step, 'model> {
    owner, phase, route_descriptors, activation_offsets, weight_handles,
    result_slots, buffer_leases
}
ExpertResultDescriptor { source_row, route_rank, expert_id, owner, result_slot }
PreparedOwnerResult<'step> { descriptors, bf16_output_arena, completion }

CompletionHandle = CpuJoin(...) | CudaEvent { device, stream_role, event }
PreparedHeterogeneousStep<'model> {
    identity, expected_revision, expected_visibility_epoch,
    placement, provisional_kv, queues, buffers, owner_jobs, results,
    output_image, evidence_draft, state
}
```

### Type invariants

| Type | Sole owner and lifetime | Mutation policy |
|---|---|---|
| `StableCudaDeviceId` | Placement manifest; process-independent | Immutable; PCI BDF is serialized, ordinal is not |
| `ResolvedCudaDevice` | Runtime model instance | Immutable after startup validation; dropped after workers and events drain |
| Placement manifest | `Arc` shared by model/engine/jobs for model lifetime | Immutable and hashed; replacement requires full unload/reconstruction |
| `GptOssRouteDescriptor` | Prepared layer route arena | Constructed once row-major/rank-major; no rank or weight reconstruction after packing |
| `GptOssRoutedBatch` | Borrowed view into prepared layer activation/route arenas | Read-only for workers; owner reduction writes a separate arena |
| Expert handle | Exactly one owner pool for model lifetime | Weight bytes immutable; handle is never transferred or cloned into another representation |
| `PackedOwnerWork` | Prepared step until worker accepts it, then worker ticket plus step retain shared leases | Descriptors immutable; only completion/error state changes |
| Result arena | Prepared step | A worker writes only its assigned slots once; slots become read-only after its completion handle signals |
| Completion handle | Prepared step/drain coordinator | Monotonic pending→complete/error; dropping it cannot imply completion |
| `PreparedHeterogeneousStep` | Engine transaction coordinator; never `Clone` | State-machine transitions only; active drop transfers to mandatory drain, never pool reuse |

## Route creation and packing

Router projection on GPU0 accumulates in f32, rounds every logit through BF16,
rejects non-finite values, chooses stable descending top-4 with lower expert ID
on ties, computes selected softmax in f32, and stores each weight's BF16 bits.
The canonical route arena is row-major and route-rank-major. Its cardinality is
exactly `M×4`.

Packing performs a stable grouping by `(owner, expert_id)` while preserving the
original descriptor order within a group. `activation_slot` permits one source
activation per destination rather than changing route identity. Each packed
entry retains its canonical `result_slot = source_row×4 + route_rank`.
Unpacking is therefore a checked scatter, not a heuristic sort. Before
reduction, all result slots must be present exactly once and agree on row,
rank, expert, owner, placement epoch, and weight identity.

The existing `stable_group_routes`/`stable_top_k` behavior in `cpu_runner.rs`
and `stable_top_k_indices` in `gpt-oss-moe-semantics` are semantic inputs. The
existing public semantics crate does not yet encode BF16 weight bits, source
row/rank descriptors, destination, or lifetime; those GPT-OSS execution types
belong in model runner, not in a generic engine crate.

## Placement and stable identity interfaces

Planning-level APIs:

```rust
StableCudaDeviceId::enumerate_and_resolve(manifest_device) -> ResolvedCudaDevice
GptOssExpertPlacementManifestV1::validate(model, devices, envelope)
    -> ResolvedExpertPlacement
ResolvedExpertPlacement::owner(GptOssExpertKey) -> ExpertOwner
ResolvedExpertPlacement::validate_materialized(owner_pools) -> OwnershipReport
```

Validation requires all configured experts exactly once, stable device matches,
representation/owner agreement, exact payload counts, quotas within envelope,
and no handle in two pools. An ordinal-only worker config is invalid for a
durable placement. Evidence includes sanitized PCI BDFs; GPU UUIDs/serials are
neither required nor published.

## Backend submission interface

The narrow executor operation is conceptually:

```rust
trait GptOssSelectedExpertBackend {
    fn prepare<'step, 'model>(
        &'model self,
        work: PackedOwnerWork<'step, 'model>,
        scratch: ScratchLease<'step>,
    ) -> Result<PreparedExpertJob<'step, 'model>, HeterogeneousError>;

    fn submit(job) -> Result<CompletionHandle, HeterogeneousError>;
}
```

There are only two implementations in initial scope: the approved CPU x8 path
and CUDA native-MXFP4. The CUDA implementation is instantiated once per stable
GPU identity. `prepare` validates every handle and bound before any enqueue;
`submit` never chooses a different backend or representation. Unsupported
phase/shape is an explicit error unless the manifest/config selects the named
serial-`M=1` correctness path.

## Prepared step and transaction interface

```rust
PreparedHeterogeneousStep::reserve(...) -> Reserved
Reserved::prepare_routes(...) -> Prepared
Prepared::dispatch_all_or_none(...) -> Dispatched
Dispatched::join_and_validate(...) -> ReducedOrFailed
Reduced::ready_to_commit(output_image, evidence) -> ReadyToCommit
Engine::commit_prepared(ReadyToCommit) -> CommittedStep
Engine::drain_and_discard(FailedOrCancelled) -> DiscardedStep
```

Typestate is preferred where it does not make error drainage impossible; the
runtime also records an explicit state enum for evidence. The engine, not a
worker, owns transitions. `PreparedHeterogeneousStep` owns:

- the generation-tagged private K/V lease;
- all pinned/device/scratch leases;
- CPU join and CUDA event handles;
- the immutable expected sequence revision, visibility epoch, and placement
  epoch;
- separate rank-slot results and a fully built, fallible-work-complete commit
  image; and
- an unpublished evidence draft.

No buffer or cache lease can escape this object except into a worker job that
keeps the same step lifetime/generation. Because CUDA is not cancellable after
launch, active-step drop moves ownership to an engine drain queue. It never
returns buffers from `Drop` merely because the request was cancelled.

## Reduction contract

The GPU0 reducer takes canonical rank slots and BF16 weight bits. For each
`(source_row, hidden_index)` it converts BF16 values/weights to f32, performs
multiply and addition in the CPU contract's rank order `0,1,2,3` with no
atomic, tree, expert-order, or reassociated reduction, and rounds the final MoE
output to BF16 before residual handling. Result descriptor validation precedes
all arithmetic. The reducer cannot consume a packed-owner order directly.

IDs, row, rank, expert, placement, weight bits, and BF16 boundary values are
bit-exact. Internal f32 fields use bit-exact comparison when operation order is
matched; otherwise only the pre-existing authoritative ULP rule may apply, and
the final BF16 boundary and retained tokens remain exact.

## Telemetry and evidence records

`gpt-oss-evidence` gains a versioned `HeterogeneousStepTraceV1` artifact that a
normal `RunManifestV1` references. It includes:

- model/mapping/placement/build/device identities;
- request revision, visibility and placement epochs;
- layer/phase/chunk, all canonical route descriptors, owners, and result slots;
- persistent/scratch/staging reservations and high-water bytes;
- enqueue, queue wait, copy, kernel, completion, reduction, drain, and commit
  intervals with CPU monotonic and CUDA event clocks explicitly labeled;
- errors sorted by deterministic precedence plus secondary cleanup errors; and
- `committed` or `discarded` terminal outcome and first-divergence artifact.

Active partial traces remain private. A committed trace is published after the
visibility epoch advances; a failed/cancelled trace is published only after
drain and discard. This extends the existing manifest/artifact system rather
than inventing a second campaign identity.

## Error taxonomy and precedence

```text
HeterogeneousError = Manifest | StableDevice | Ownership | Bounds | Route
                   | Reservation | Queue | Cpu | CudaLaunch | CudaAsync
                   | H2D | D2H | ResultIdentity | Reduction | StaleRevision
                   | Cancelled | Publication | Drain | Cleanup
```

All observed errors are retained. The primary error is chosen after mandatory
drain by a fixed key, never by racing completion time:

1. invariant errors (`Manifest`, `Ownership`, `Bounds`, `Route`, result
   identity) in logical-stage order;
2. operational errors in stage order: reserve, pack, queue, H2D, kernel, D2H,
   reduction, ready/publication;
3. within a stage: GPU0, CPU, GPU1, then ascending route slot;
4. cancellation if no invariant/operational error exists; and
5. drain/cleanup errors are secondary unless they are the only error.

Cancellation before dispatch is terminal without jobs. During dispatch it
suppresses new enqueue/publication but cannot mask a device failure found while
draining. Panic at a host worker boundary is converted to a CPU error; it never
unwinds across coordinator ownership.

## Deliberate non-interfaces

These types do not mention Harmony, HTTP, tokenizer assets, attention sinks,
RoPE, sampling, model-family registries, adaptive policy, migration, or a graph
runtime. K/V and token publication interact with the prepared-step transaction
but are not imported into the selected-expert backend trait.
