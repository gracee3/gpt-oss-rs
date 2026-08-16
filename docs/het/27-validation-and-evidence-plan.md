# Validation and evidence plan

## Global rules

Every tier has its own terminal artifact and stop gate. A later token match
cannot erase an earlier route, BF16, ownership, memory, or lifecycle failure.
No performance gate applies until all relevant correctness tiers pass.

All real-model commands use an internal benchmark/control binary, never HTTP.
Command lines below are shapes; exact names introduced by the owning package
must be recorded in the evidence manifest.

The first-divergence order is:

```text
model/view identity -> input BF16 -> router f32 lanes/logit BF16
-> IDs/ranks -> selected-weight BF16 -> placement/packed descriptors
-> per-route gate/up BF16 -> SwiGLU BF16 boundaries -> down BF16
-> returned descriptor/output -> weighted contribution f32
-> rank-ordered reduction/BF16 -> residual BF16 -> provisional K/V/output
-> visibility epoch -> logits -> sampled/retained token IDs
```

## Tier 1 — route semantics and representation fixtures

**Fixtures:** stable top-4 ties/near-ties for E=32 and E=128; non-finite
rejection; BF16 weight bits; row/rank/expert descriptor round trips; stable
grouping with repeated experts; native expert slice boundaries; x8 round trip;
placement coverage/duplicates; stable PCI identity resolution; generation-tag
ABA; pool exhaustion.

**Command shape:**

```bash
cargo test --locked -p gpt-oss-moe-semantics
cargo test --locked -p gpt-oss-model-runner heterogeneous::contract
cargo test --locked -p gpt-oss-model-runner owner_selective
cargo test --locked -p gpt-oss-gpu stable_device_and_bounded_pinned
```

**Evidence:** unit result summary, source/lock/build identity, fixture seeds,
exact expected/actual bits, map/placement validation report.

**Stop:** any tie mismatch, route-rank loss, descriptor reinterpretation,
zero/multiple owner, unbounded pool allocation, or stale generation accepted.

## Tier 2 — one selected expert on CPU authority

**Fixtures:** synthetic native blocks/scales and the pinned 20B layer-0 expert
31 activation/weight. Run `CpuExpertProjection::ExactBf16`; retain 16-lane f32
states, gate/up, every GPT SwiGLU BF16 boundary, down output, and output hash.
Then confirm the approved x8 path produces the existing accepted result without
promoting any closed multirow policy.

**Command shape:**

```bash
cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_expert_oracle -- \
  --backend cpu-exact --fixture <20b-layer0-expert31-fixture> --trace all
```

**Evidence:** fixture/model identity, CPU capability/dispatch, representation,
scratch/repack timing separated from arithmetic, exact boundary artifacts.

**Stop:** failure to reproduce the retained CPU authority or an implicit use of
a non-authoritative historical campaign.

## Tier 3 — one selected expert through CUDA

Run the synthetic H2 set and the same real expert on both stable GPU devices,
once cold and repeatedly warm. Force the new native-packed selected-expert
primitive directly; current CUDA MoE symbols are forbidden.

**Command shape:**

```bash
cargo test --locked -p gpt-oss-model-runner --features cuda selected_expert_cuda
cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_expert_oracle \
  --features cuda -- --backend cuda-selected --stable-device <role> \
  --fixture <20b-layer0-expert31-fixture> --trace first-divergence
```

**Evidence:** stable/sanitized device identity, PTX/binary hash, exact resident
weight bytes, actual scratch/alignment, event timing, allocation high water,
GPU0/GPU1 boundary traces and repeat hashes.

**Stop:** any BF16 boundary mismatch, all-expert scan, FP16 matrix expansion,
unbounded scratch, unsupported-shape fallback, asynchronous error loss, device
identity mismatch, or repeat allocation growth.

## Tier 4 — one real layer across CPU, GPU0, and GPU1

Use the pinned 20B layer-0 real activation and its actual router output
`[31,21,22,6]`. The proof manifest owns 31/6 on GPU0, 21 on CPU, and 22 on
GPU1. Routing is not synthetically replaced. Compare:

- CPU authority all-local;
- host-owned oracle coordination; and
- target GPU0-owner coordination.

H6a first supplies CPU-authority expert outputs to the GPU0 owner shell and
compares attention, private K/V, post-norm, native-BF16 router projection,
logit rounding, selection, reduction, and residual. Only after that shell is
first-divergence-clean does H6b exercise all three owners, return four exact
rank descriptors, reduce on GPU0, and produce the exact layer output while the
visibility epoch remains provisional.

**Command shape:**

```bash
cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_layer_oracle \
  --features cuda -- --model <20b> --layer 0 --phase decode --m 1 \
  --placement <20b-proof-manifest> --compare cpu-exact --trace first-divergence
```

Follow with the actual prefill occupancies `M=7,24,33,61` through the explicit
serial-`M=1` CUDA correctness adapter and exact CPU buckets.

**Evidence:** canonical routes/weights, placement hash, per-owner counts,
packed descriptor tables, byte directions, per-expert boundaries, rank slots,
reduction/residual, queue/buffer high water, events, and a globally correlated
CPU/GPU0/GPU1 timeline.

**Stop:** H6a owner-shell mismatch, all three owners not actually executing in
H6b, synthetic route substitution, rank reconstructed from completion/expert
order, BF16 mismatch, no correlated concurrency proof, or any current rejected
CUDA/NCCL/TP path reached.

## Tier 5 — transaction, cancellation, failure, and cleanup

Run every injection in [document 26](26-transaction-failure-cancellation.md)
at pre-dispatch, each owner enqueue/completion, pre-reduction, post-reduction,
pre-commit, and shutdown. Permute CPU/GPU completion/failure timing. Each case
is followed by a clean successful step and repeated load/unload where relevant.

**Command shape:**

```bash
cargo test --locked -p gpt-oss-engine --features cuda heterogeneous_transaction
cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_failure_probe \
  --features cuda -- --case <matrix-case> --repeat 2
```

**Evidence:** state transitions, expected/observed revision and visibility
epoch, private/committed block tables, block generations, cancellation time,
all errors in deterministic order, drain acknowledgements, lease counts,
allocator/pool baselines, terminal committed/discarded marker.

**Stop:** any uncommitted K/V/output/token/evidence read, pool reuse before
event drain, stale append on the next request, timing-dependent primary error,
leaked/quarantined capacity without a terminal error, or shutdown with live
work.

## Tier 6 — 20B cold load, prefill, and retained continuation

First rerun the pinned CPU control if its binary/source/model identity has
changed. Then run the target internal path under the 20B proof manifest:

1. cold owner-selective construction and measured unload/reload;
2. 63-token prefill under the explicit bounded correctness path;
3. one layer trace on the first retained decode step;
4. the eight-token retained continuation; and
5. a second warm run after cleanup.

The mandatory retained tokens are exactly:

```text
[200005, 35644, 200008, 976, 1825, 5003, 25, 392]
```

**Command shape:**

```bash
cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_control \
  --features cuda -- --model <20b> --fixture cpu_harmony_parity.json \
  --scenario harmony_63 --placement <20b-proof-manifest> \
  --max-new-tokens 8 --expected-token-ids 200005,35644,200008,976,1825,5003,25,392
```

**Safety/evidence:** predeclared RSS/MemAvailable/swap/GPU-reserve stops;
model/map/placement/build/device identity; stage RSS/PSS/VRAM; zero swap;
context/chunk; exact route/rank/owner evidence; buffer/queue highs; event
timeline; visibility epochs; output IDs; cleanup and repeat baseline.

**Stop:** control identity ambiguity, allocation stop, swap, reserve erosion,
non-exact tokens, no real three-owner layer, evidence gap, or failed repeat
cleanup.

## Tier 7 — owner-selective 120B construction

H8 is construction only. Validate the local 543-to-687 mapping and a
quota-balanced proof manifest, then execute stages L0–L8 from
[document 23](23-owner-selective-loading-memory.md). No forward layer runs.

**Command shape:**

```bash
cargo run --locked --release -p gpt-oss-bench --bin owner_selective_load \
  --features cuda -- --model <120b-native> --placement <120b-proof-manifest> \
  --context-cap 4096 --construct-only --repeat 2
```

**Evidence:** native/map manifests, placement hash, every expert key and one
owner, CPU x8 file identities, actual/predicted bytes, mapped/RSS/PSS/page-cache/
anonymous/pinned categories, per-stage per-GPU allocation categories, reserve
remaining, no-swap assertion, partial-failure rollback, unload and repeat.

**Stop before forward:** any mapping/artifact mismatch, full alternate expert
form, whole U8/FP16 host copy, swap, host/GPU guard breach, missing reserve,
owner count mismatch, or retained allocation after unload.

## Tier 8 — 120B one-layer then retained proof

Only after Tier 7 passes:

1. run an earliest real one-layer/router fixture and compare every boundary to
   the CPU semantic authority/control capture;
2. prove one real layer routes selected experts to CPU, GPU0, and GPU1 under the
   immutable proof manifest;
3. run the bounded 120B prefill/one-token control;
4. only then run the reviewed retained-continuation length; and
5. repeat after clean unload/reload.

No synthetic reassignment satisfies the real-route gate. If the deterministic
proof manifest does not yield a real three-owner layer, H9 stops. A new
manifest requires documented router evidence, complete unload, H8 construction
revalidation, and review; live movement is forbidden.

**Command shape:**

```bash
cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_layer_oracle \
  --features cuda -- --model <120b-native> --placement <120b-proof-manifest> \
  --phase decode --m 1 --trace first-divergence

cargo run --locked --release -p gpt-oss-bench --bin heterogeneous_control \
  --features cuda -- --model <120b-native> --placement <120b-proof-manifest> \
  --fixture <reviewed-retained-fixture> --max-new-tokens <reviewed-bound>
```

**Evidence:** everything from Tiers 4–7 plus exact retained token IDs against
the approved CPU/oracle capture, real three-owner routes, complete commit/drain
records, and repeat cleanup.

**Stop:** any earlier tier regression, missing CPU/oracle expected continuation,
non-exact output, no real three-owner layer, memory/reserve/swap violation,
transaction failure, or incomplete durable evidence.

## Tier 9 — performance characterization

Only after applicable Tier 8 gates pass, measure exact selected-expert GPU
costs, packing/events, CPU interference, route frequencies, and prefill buckets.
Derive static proof-independent performance placement with uncertainty. Re-run
all exact arithmetic/transaction gates after any scheduling/kernel change.

No speedup is required for the correctness milestone. A slow exact result is
reported as slow; it is not relabeled a performance success. Adaptive policy,
migration, prediction, replication, and approximate deferral remain deferred.

## Evidence bundle

Every tier's `RunManifestV1` references bounded artifacts:

- model/native-to-runtime map and asset manifests;
- immutable placement manifest and SHA-256;
- stable sanitized PCI identities and resolved ordinals recorded separately;
- repository commit, dirty fingerprint, lock/build/features, binary/PTX hashes;
- canonical route assignments, ranks, BF16 weights, owners, and expert counts;
- stage RSS/PSS, file/anonymous/pinned bytes, swap, per-GPU allocations/reserve;
- queue/pool/scratch high-water marks;
- per-device queue/copy/kernel/event/drain intervals and correlated timeline;
- first-divergence trace;
- expected/actual revision, placement and visibility epochs;
- cancellation/failure/error precedence and terminal outcome; and
- unload/repeat baseline.

Absolute paths and machine identifiers are redacted according to the existing
evidence policy. No GPU UUID/serial, hostname, IP/MAC, filesystem UUID, or token
is published.
