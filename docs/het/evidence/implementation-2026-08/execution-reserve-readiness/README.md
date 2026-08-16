# Execution-reserve readiness follow-up

**Status:** passed on 2026-08-16. This pre-H8 package makes
`ConstructionStage::ExecutionReserve` an exact, checked admission contract.
It does not run H8/H9, load 120B weights, or pretend that a 4 GiB dummy
allocation proves fit.

The final non-document source fingerprint is
`b7c8fc6720c76dd347edd6b9e69896d758f9de007bfd91317f703978e415a944`.
The construction and control executables are respectively
`1efc843d174196df9f264cba15020c077e60b0fb7ab683ac137d7545b22cbf3b`
and `8257bd4130640df6aea7a823c599aa9f90be49e2fdc7bc6fdc466ba546160d37`.

## Records

| Record | Result |
|---|---|
| [`execution-reserve-manifest.json`](execution-reserve-manifest.json) | Final source/binary/input identities, exact byte ledgers, watchdog results, and gate inventory |
| [`20b-warm-construction.json`](20b-warm-construction.json) | v4 warm construction; L6 plan reviewed, executor buffers materialized before admission, remaining runtime resources explicitly deferred |
| [`20b-construction-faults.json`](20b-construction-faults.json) | v4 real eight-stage constructor fault campaign; exact CUDA cleanup, no pinned/temp residue, then a clean reconstruction |
| [`20b-retained-control.json`](20b-retained-control.json) | v3 two-repeat 20B continuation; exact tokens, real three-owner work, strict overlap, bounded cleanup |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 identities for the bounded retained records |

## Exact admission boundary

H8 measures free VRAM after both selected-expert executors and their CUDA
context/modules exist. Each executor owns 46,080 bytes of input, scratch, and
trace storage at that point. Those bytes are therefore recorded as
`materialized_before_admission_bytes` and are not charged again to the 4 GiB
deferred reserve.

For each GPU the checked equations are:

```text
reviewed_deferred_after_admission_bytes
  + runtime_and_safety_remainder_bytes
  = reserve_cap_bytes

materialized_before_admission_bytes
  + reviewed_deferred_after_admission_bytes
  = planned_owned_bytes
```

`execution_runtime_resources_materialized_at_construction=false` means only
that the owner-selective L6 constructor has not yet created K/V, router,
relay, reduction, canonical result-slot, or pinned-relay resources. The later
20B control runtime does materialize its bounded execution resources; the
field no longer ambiguously describes the whole process lifetime.

## Byte-exact plans

Both plans bind the fixed runtime shape, decode `M=1`, and a 4,096-token
context. Arithmetic is checked and unsupported GPT-OSS shapes fail closed.

| Model / device | Already materialized | Deferred reviewed | Eventual owned | 4 GiB remainder |
|---|---:|---:|---:|---:|
| 20B proof / GPU0 | 46,080 | 210,206,880 | 210,252,960 | 4,084,760,416 |
| 20B proof / GPU1 | 46,080 | 23,040 | 69,120 | 4,294,944,256 |
| 120B proof metadata / GPU0 | 46,080 | 334,801,968 | 334,848,048 | 3,960,165,328 |
| 120B proof metadata / GPU1 | 46,080 | 299,520 | 345,600 | 4,294,667,776 |

GPU0 includes K/V for every layer, fixed shell storage, per-layer native BF16
router storage, canonical contribution arenas, serial reducers, and four
result slots per layer. GPU1 includes one executor and four result slots for
each layer that owns at least one remote expert. The actual proof manifests
have one such 20B layer and thirteen such 120B layers. Property tests also
cover every remote-layer count through the model layer count.

The exact warm decode pinned relay needs 74,944 raw bytes: 5,760 source bytes,
64 descriptor bytes, and three 23,040-byte result/input arenas. Its hard cap is
131,072 bytes, and production reporters assert raw `<=` cap. The 8,388,608-byte
prefill value remains an explicit non-materialized policy cap; it is not
presented as measured raw demand.

## Validation and resource evidence

The final gate passed:

- seven owner-selective ledger/property tests, including metadata-only local
  20B/120B configs, every fixed-shape mismatch, tampering, and real checked
  overflow;
- the actual bounded pinned-pool exhaustion/reuse test;
- all eight real constructor-stage injected faults plus clean reconstruction;
- final-source real H4 relay, H5 reduction, and H6a/H6b shell/three-owner
  integration regressions;
- the two-repeat 20B retained continuation
  `[200005,35644,200008,976,1825,5003,25,392]`, with exact routes/results,
  strict three-way overlap, and cleanup on both runs;
- locked workspace check/tests, three configured strict Clippy lanes, and
  strict plain-CUDA/fault-feature construction/control lanes.

Five idle samples over two minutes were byte-identical at
`SwapFree=104836164 KiB`, `SwapCached=1676 KiB`, memory PSI `0.00`, and
dockerd `VmSwap=4208 KiB`. Every accepted real run used an external watchdog:
the minimum SwapFree never fell below its run baseline and target-process
`VmSwap` remained zero. The retained run released 56 KiB of pre-existing swap;
that is recorded as no new allocation, not described as byte-identical end
state. Earlier functional-only regressions during unrelated dockerd swap
growth are not used as acceptance evidence.

## Scope

The 120B plan reads config/mapping/placement metadata only. No 120B weight was
loaded or executed, no complete representation was created, no model was
copied or transformed, and prior H8 failure evidence was not edited. H8
remains stopped on its separately recorded host-run gate, and H9 has not
started. `/dev/nvme1n1` remained read-only and unmounted.
