# H3 production runtime checkpoint retirement

**Status:** production ownership integration implemented and source/synthetic
validated for supervisor review. No real model was opened or executed. H8
remains unpassed; H9/H10 remain stopped.

## Purpose and boundary

This slice retires checkpoint payload ownership at the publication boundary of
the existing owner-selective constructor. It integrates the previously detached
resident-router handoff into the production ownership path:

```text
full checkpoint view during existing construction
    -> classify and upload every dense surface once
    -> pair each layer's resident router weight + bias
    -> publish payload-free model metadata + resident owner allocations
    -> drop the complete checkpoint view
    -> control runtime single-consumes ordered router pairs by same-context D2D
```

The constructor still maps the complete checkpoint until publication. This
change therefore does **not** claim lower construction peak RSS or pass the H8
memory gate. Per-shard GPU expert assembly and atomic incremental CPU x8
publication remain required before the next real 20B construction gate.

## Published model ownership

`OwnerSelectiveModel` no longer contains `GptOssCheckpointView` and exposes no
`checkpoint()` accessor. It retains only `OwnerSelectiveNativeMetadata`:

- the validated native config;
- the checkpoint revision string;
- config, metadata, and mapping SHA-256 identities.

These values contain no shard store, tensor mapping, source root, mmap, or
payload byte view. Layer-shell and control-runtime config checks now use this
metadata. The constructor explicitly drops its owned checkpoint after the
model value is complete and before returning it.

A source audit finds no `model.checkpoint()` or `checkpoint_bf16_bits` call in
the Rust workspace. The retained H6 oracle now computes its exact selected-
expert authority traces before construction consumes the checkpoint, then
uses those owned trace values after publication. It does not reopen a model.

## Exact router extraction

Within the existing dense upload loop, only these canonical names classify as
router surfaces:

```text
model.layers.<canonical-decimal-layer>.mlp.router.weight
model.layers.<canonical-decimal-layer>.mlp.router.bias
```

Classification rejects malformed, leading-zero, out-of-range, or unsupported
router names. Each layer must publish exactly one weight and one bias. The
weight and bias must match the exact E=32 or E=128 BF16 byte shape already
enforced by `ResidentExactRouterWeights`; duplicate, missing, mispaired, wrong-
layer, wrong-shape, and non-numeric-order publication fail closed.

Router allocations remain included in `ConstructionLedger.layer_owner_dense_bytes`.
They are removed only from the generic `LayerOwnerDenseTensor` vector and
transferred without another checkpoint read into a deterministic
`ResidentExactRouterSources` set. Construction evidence continues to report
the same total dense tensor count and device bytes by adding the router-source
ledger to the remaining generic dense ledger.

## Single-consumer runtime handoff

`ResidentExactRouterSources` validates the expected layer count, E=32/E=128,
stable CUDA device identity, exact ascending layer order, checked tensor count,
and checked device-byte total. `take_ordered` consumes the set once; a second
attempt is rejected.

`HeterogeneousControlRuntime::new` creates the layer-owner shell, consumes the
entire ordered source set, verifies the runtime layer count, and creates each
`CudaExactRouter` with `from_resident_weights(1, source)`. The old host slice
lookup and H2D constructor path is absent from this runtime.

Partial handoff behavior remains fail closed:

- sources not yet handed to CUDA have never been enqueued and are ordinary
  rollback-owned values;
- each resident handoff owns its source and destination allocations until its
  terminal stream proof;
- recoverable post-enqueue failure drains before release;
- unproven drain retains source, destination, streams, loader, and context for
  process life and returns no router; and
- if the published model's executor teardown is unproven before runtime
  consumption, all still-resident router sources join the model quarantine.

## Validation and limits

Tiny source fixtures prove classification, exact pair coverage, rejection
cases, payload-free metadata validation, deterministic order, checked byte
accounting, and single consumption. Synthetic CUDA fixtures on both local RTX
3090s prove resident/host constructor equivalence, terminal source release,
wrong device/context/shape rejection, recoverable retry, and unproven-drain
quarantine.

All validation removed the three model-path variables and every known model
run gate from the environment. No real 20B/120B path was opened, statted,
mapped, hashed, constructed, or executed. No H8/H9/H10 action ran.

The remaining integration boundary is deliberately unchanged: replace the
whole-checkpoint construction lifetime with plan-bound, capacity-one per-shard
GPU expert assembly and atomic incremental CPU x8 publication, then pass a
separately authorized real 20B construction/control gate. This document does
not authorize that work.

Bounded validation evidence is indexed in
[`evidence/implementation-2026-08/h3-runtime-checkpoint-retirement/README.md`](evidence/implementation-2026-08/h3-runtime-checkpoint-retirement/README.md).
