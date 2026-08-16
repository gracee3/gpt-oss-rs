# H3 device-resident exact-router handoff

**Status:** source and synthetic-CUDA prerequisite implemented for review; not
integrated into owner-selective construction or the heterogeneous runtime. H8
remains unpassed, and H9/H10 remain prohibited.

## Purpose and boundary

The production owner-selective constructor already uploads layer-owner dense
tensors once, but the current heterogeneous control shell constructs
`CudaExactRouter` from host slices. This milestone adds the narrow ownership
boundary needed to eliminate that future payload reread:

```text
owned resident GPU0 router weight bytes + owned resident GPU0 bias bytes
    -> stable-device and exact-context validation
    -> bounded same-context D2D copies
    -> terminal stream drain
    -> source release
    -> router-owned resident BF16 byte storage
```

`OwnerSelectiveModel`, `OwnerSelectiveConstructor`, and
`HeterogeneousControlRuntime` are unchanged. The existing host-backed
`CudaExactRouter::new` API and upload behavior remain available and serve as
the synthetic equivalence authority. No checkpoint path is consulted by the
new primitive or its tests.

## Verified contract

`ResidentExactRouterWeights` consumes two `CudaSlice<u8>` allocations matching
the current dense-tensor storage representation. Before enqueue it requires:

- a durable `StableCudaDeviceId` that resolves to the allocations' current
  CUDA ordinal;
- one identical raw CUDA context for the weight and bias allocations;
- `E=32` or `E=128`;
- exactly `E * 2,880 * 2` weight bytes and `E * 2` bias bytes; and
- `max_rows` in `1..=64`, with every destination dimension calculated using
  checked arithmetic.

`CudaExactRouter::from_resident_weights` creates its streams and all fixed
destination allocations on that same context. The whole-allocation `u8`
sources are reborrowed through cudarc's checked-length unsafe `CudaView<u16>`
transmute and copied into ordinary router-owned `CudaSlice<u16>` destinations.
This is sound because CUDA allocation alignment exceeds `u16` alignment, both
byte lengths are exact and even, every `u16` bit pattern is valid, and the
views cannot outlive the owned source state. Weight and bias movement uses
same-context D2D only. It neither enables peer access nor calls NCCL.

The largest surface pair is bounded to 737,536 bytes at `E=128`; the returned
router owns an independent typed destination pair and reports the same total
owned device bytes as the host-backed path. The existing `CudaExactRouter`
weight/bias fields and `launch_projection` implementation remain literally
unchanged, so the new handoff does not add a production kernel-launch branch.

## Drain and failure ownership

The source pair is owned, not borrowed. It is released only after the handoff
stream synchronizes after both D2D copies. A recoverable injected failure after
the first enqueue performs the mandatory fallback drain, releases the source,
and permits a clean retry.

If a post-enqueue drain cannot be proven, `ResidentRouterHandoffState` retains
for process life:

- both source allocations and their streams/contexts;
- both destination router-weight allocations;
- every other destination router allocation;
- the compute and relay streams; and
- the kernel loader and its context handle.

The unproven path returns no router and exposes no reusable allocation. This is
intentional bounded leakage under an injected destructive fault, not a normal
fallback. No later synchronization is allowed to rehabilitate that state.

## Synthetic validation

The focused CUDA test uses generated BF16 values only. It proves:

- `E=32` on one local RTX 3090 and `E=128` on the other produce bit-identical
  logits, canonical routes, weights, and batch descriptors versus
  `CudaExactRouter::new`;
- the source drop probe fires before successful constructor return, after the
  terminal copy drain;
- wrong stable device, mixed CUDA contexts, and short surfaces reject;
- a recoverable post-enqueue failure drains and allows a fresh clean retry;
  and
- an injected fallback-drain failure does not drop its source and increments
  the bounded quarantine witness.

All validation commands explicitly remove `GPT_OSS_MODEL_PATH`,
`GPT_OSS_MODEL_20B_PATH`, `GPT_OSS_MODEL_120B_PATH`, and the repository's
opt-in model-run gates from their environment. No real checkpoint was opened,
statted, mapped, hashed, constructed, or executed.

## Remaining production integration boundary

The current `LayerOwnerDenseTensor` keeps its allocation private inside
`OwnerSelectiveModel`. A later reviewed integration must transfer the two
router allocations into `ResidentExactRouterWeights` without duplication or
provide an equally strong owned lease. That change requires the real 20B
construction/control gate because it changes model ownership and teardown.

Until then, this primitive is deliberately detached. It does not prove a
model-backed resident-router handoff, H8 construction, H9 inference, or H10
performance. It does not change swap, watchdog, reserve, placement,
protected-storage, network, or authorization gates.

Bounded validation evidence is indexed in
[`evidence/implementation-2026-08/h3-resident-router-handoff/README.md`](evidence/implementation-2026-08/h3-resident-router-handoff/README.md).
