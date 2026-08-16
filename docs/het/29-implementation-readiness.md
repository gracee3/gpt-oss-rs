# Implementation readiness

## Verdict

**`conditionally_ready`**

The architecture, routed-expert seam, hybrid representation, selected-expert
CUDA gate, transfer schedule, private-slot commit model, validation ladder, and
work-package ownership are settled. Production implementation may not begin in
the current dirty tree until H0 establishes a reviewed branch/commit boundary.
H0 is the only package permitted to begin after Phase 2 review.

The conditions are:

1. the user reviews Phase 2 and approves the fixed architecture and private-slot
   visibility model;
2. H0 verifies the nine-file patch still has full SHA-256
   `792b545405494ca2a5be543b24e29ee0f68420db0f3aa5ec59adf4ea114a374e`;
3. the user chooses/authorizes how the nine readiness fixes and Phase 0–2 docs
   become separately attributable before H1 edits; and
4. H0 produces a clean or otherwise isolated implementation base without
   reset, stash, cleaning, or loss of user-owned work.

Exact selected-expert CUDA time, owner-selective 120B construction peak, and
performance placement remain later evidence gates, not unresolved Phase 2
design decisions.

## Final fixed decisions

- GPU0 is a layer-owner role resolved from stable PCI identity, never durable
  CUDA ordinal 0.
- GPU0 owns dense/attention, provisional K/V, exact router, local experts,
  rank-ordered reduction/residual, and the authoritative prepared result.
- GPU1 and CPU are expert-only workers; every expert has one immutable owner.
- GPU1 traffic is explicit pinned-host relay; no P2P, NCCL dispatch, TP model
  route, or decode weight movement.
- The seam is BF16 activations plus rank-bearing selected routes to unweighted
  BF16 selected-expert results. Routing and reduction remain GPT-OSS-owned.
- Native shards and the proven 543-to-687 map are authoritative. Q/K/V are
  views; GPU experts use native packed MXFP4; only CPU experts have persisted
  x8 records.
- The first CUDA expert primitive is native-packed, selected-only, exact
  decode `M=1`, with 17,280 B logical scratch and no full FP16 expansion.
- One GPU0 activation D2H feeds CPU and GPU1 packing. GPU0 reduces canonical
  route slots in rank order.
- Queues and pinned/device pools are hard bounded; the proof has one active
  layer dispatch and one in-flight step per sequence.
- **Commit model:** generation-tagged private physical K/V append slots plus an
  exclusive allocation-free host commit and one visibility epoch advanced
  last. Cancellation after enqueue mandates drain.
- The first milestone is correctness/ownership/memory/transaction evidence,
  not speedup.

## Unresolved evidence, with package owner

| Evidence not yet established | Owning gate | Does it block H0/H1? |
|---|---|---|
| Native selected-expert CUDA exactness and time | H2 | No; it blocks H4/H6 |
| 20B owner-selective cold construction/repeat peak | H3 | No; it blocks H4 integration/H8 |
| Exact GPU0 E=128 router and relay/event lifetime | H4 | No; it blocks H5/H6 |
| Private-slot invisibility and failure/cancel drain | H5 | No; it blocks H6 |
| Correlated real CPU/GPU0/GPU1 overlap | H6 | No; it blocks H7 |
| 20B target continuation through heterogeneous path | H7 | No; it blocks all 120B work |
| Measured 120B construction/reserve/no-swap/repeat | H8 | No; it blocks H9 |
| Approved 120B expected retained continuation and real three-owner route | H9 preflight | No; it blocks 120B retained execution |
| Static performance placement thresholds | H10 | No; deferred until correctness proof |

If H5 disproves private-slot isolation, implementation stops and returns to
planning; fully staged K/V is not an automatic fallback.

## Package order and critical path

```text
H0 baseline boundary
 -> H1 contracts
 -> H2 exact selected CUDA expert
 -> H3 owner-selective construction
 -> H4 exact router + bounded relay
 -> H5 reduction + private-slot transaction
 -> H6 one real three-owner layer
 -> H7 20B retained continuation
 -> H8 120B construct-only
 -> H9 120B retained heterogeneous proof
 -> H10 measured static placement
```

H0–H9 are the correctness critical path. H10 is post-proof characterization.
No later package may compensate for an earlier failed gate.

## First authorized package: H0 only

The following is the complete verbatim handoff. It does not authorize H1 or a
commit by itself.

> Execute H0 only from `~/gpt-oss-rs`; stop before H1 or any production-code
> change. Read `AGENTS.md` plus `docs/het/20-planning-charter.md`,
> `28-work-packages-risk-register.md`, and `29-implementation-readiness.md`.
> Capture branch, HEAD, upstream, exact `git status --short --branch`, full
> diff-stat, and the SHA-256 of exactly these nine paths: `Cargo.lock`,
> `crates/gpt-oss-gpu/src/cublas.rs`, `cublas_ops.rs`, `device.rs`,
> `memory/cpu_pool.rs`, `memory/gpu_pool.rs`, `memory/swap.rs`, `nccl.rs`, and
> `pinned_memory.rs`. Require HEAD
> `0113e8214e765d168216bbee2120654555a4cfe4`, 0/0 versus `origin/main`, patch
> stat `35 insertions, 13 deletions`, and full patch SHA-256
> `792b545405494ca2a5be543b24e29ee0f68420db0f3aa5ec59adf4ea114a374e`.
> Enumerate Phase 0–2 documentation separately and report any unrelated delta.
> Propose exact branch and commit/staging boundaries that keep the nine
> user-owned readiness fixes and Phase 0–2 docs separately attributable. Do not
> reset, clean, stash, rebase, format, stage, commit, push, or edit production
> files unless the invoking user separately and explicitly authorizes those
> exact operations. If authorization is absent, return the boundary proposal
> and stop with H0 still open. If the baseline differs, stop immediately and
> report it. Do not begin H1.

## Implementation-wide exit criterion

Implementation is complete only when a reviewed GPT-OSS-120B retained
continuation matches its exact authority while at least one real layer's
selected experts actually execute on CPU, stable-identity GPU0, and
stable-identity GPU1; every expert has one resident representation; no expert
weight moves during decode; construction/execution remain within approved host
and device reserves with no swap; K/V/output/token/evidence publish through the
single visibility epoch; cancellation/failure drains safely; and durable
first-divergence, owner, memory, transfer, timing, commit, and cleanup evidence
passes H9. Allocation on three devices, liveness, or a synthetic route alone is
not completion.

## Explicit deferred backlog

Adaptive migration, expert caching/eviction, prediction/prefetch, approximate
deferral, replication, decode weight streaming, P2P/NCCL dispatch, tensor
parallelism, alternating layer owners, CUDA graphs, grouped-prefill optimization
before proof, Xe/AMX, Qwen/model-general graph work, distributed/multi-node,
vision/MTP, upstream integration, and unrelated HTTP/Harmony semantics remain
outside H0–H9. H10 may address only measured static placement and separately
gated grouped prefill.

## Source and evidence anchors

- [Target architecture](21-target-architecture.md)
- [Interfaces](22-expert-contract-and-interfaces.md)
- [Loading/memory](23-owner-selective-loading-memory.md)
- [CUDA gate](24-selected-expert-cuda-plan.md)
- [Transfer/reduction](25-transfer-scheduling-reduction.md)
- [Transaction model](26-transaction-failure-cancellation.md)
- [Validation](27-validation-and-evidence-plan.md)
- [Work packages/risks](28-work-packages-risk-register.md)
- [Phase 1 conclusions](19-research-conclusions.md)
- [Bounded evidence](evidence/research-2026-08/README.md)

Key current-main source anchors are
`crates/gpt-oss-model-runner/src/cpu_runner.rs::{moe_batch,PreparedCpuStep}`,
`kv_cache/cache.rs::KVCache`, `gpu_layer.rs` K/V write and current MoE,
`crates/gpt-oss-engine/src/worker/input.rs`,
`gpu_engine.rs::{build_metadata,process_worker_outputs}`,
`crates/gpt-oss-gpu/src/{device,pinned_memory}.rs`,
`kernels/gpt_oss_moe.cu`, and `crates/gpt-oss-evidence/src/lib.rs`.

## Commands run and omitted in Phase 2

**Run:** required Git baseline/fingerprint; complete Phase 0/1 docs/evidence
index reads; targeted source inspection of CPU MoE/prepared state, GPU K/V,
scheduler/block tables, current CUDA MoE/loader/pinned/events/evidence;
`cargo metadata --no-deps --locked --format-version 1`; tiny byte/quota/buffer
arithmetic; final Markdown link, whitespace, sanitization, scope, and Git
fingerprint checks. The protected-device check read sysfs `ro=1` and found no
mount-table entry for `/dev/nvme1n1`; the block device was not opened.

**Omitted:** all builds/tests/Clippy/formatters, model load/generation,
checkpoint hash/transform/download, CUDA/CPU/transfer research reruns, Docker,
HTTP, external checkout changes, installs, system tuning, and 120B execution.

## Exact Phase 2 Git delta

Phase 2 changed only `docs/het/README.md` and added exactly:

```text
docs/het/20-planning-charter.md
docs/het/21-target-architecture.md
docs/het/22-expert-contract-and-interfaces.md
docs/het/23-owner-selective-loading-memory.md
docs/het/24-selected-expert-cuda-plan.md
docs/het/25-transfer-scheduling-reduction.md
docs/het/26-transaction-failure-cancellation.md
docs/het/27-validation-and-evidence-plan.md
docs/het/28-work-packages-risk-register.md
docs/het/29-implementation-readiness.md
```

The pre-Phase-2 `docs/het/README.md` SHA-256 was
`9ea9b09a7951476795b8c15b4ca7f9c9ff00a6252ddd37ad1a972dfb936737d2`.
Final validated identities (line count / SHA-256) are:

| File | Lines | SHA-256 |
|---|---:|---|
| `README.md` | 121 | `ec0c029567080e228b34a30b8bb764a43e5a3f0e485dd6c718e7195c6c864b85` |
| `20-planning-charter.md` | 120 | `dcc224c338bc9e46c78c7aa3e06628a20d816d457519a0454ba6ed8f1b3adc62` |
| `21-target-architecture.md` | 163 | `1b41148ccc7cf57285547e9d854ba2483837bf1ce46bae5bbcb89ef8bec60eda` |
| `22-expert-contract-and-interfaces.md` | 243 | `6636cab34545c7a94bbf37640a17d1165f27a223f44cca1d37627bbcef4b2b9a` |
| `23-owner-selective-loading-memory.md` | 212 | `9547275cb9752b455d70a6f51f65a103ff67ef897221d02aad37c6f6901b055e` |
| `24-selected-expert-cuda-plan.md` | 177 | `fc5ca4b48a5c6845d3b63bb1e5cb111e4ca0f7422f975b6d4454d52a765b26a0` |
| `25-transfer-scheduling-reduction.md` | 235 | `63f02198d5af423fba091336d0e0e6a571784720224419d2aca2a0abf7d128ab` |
| `26-transaction-failure-cancellation.md` | 229 | `eeafd6e930717b78bd150f22b25983f1b64f69ae7cc9dbc87d357075a23d5c05` |
| `27-validation-and-evidence-plan.md` | 279 | `4bd8e790456d44bc96518fc815e81d4b442c7cd06a2e8a7de4c3b1d10a20b256` |
| `28-work-packages-risk-register.md` | 296 | `79c882fff3529bc10eb1bc78815c89f53bb42b67e40840bde0ecc04773d15548` |

This `29` record is the self-referential member of the set; its final hash is
reported in the Phase 2 handback rather than embedded inside itself.

The final nine-file code/lock patch identity remains the startup value. Nothing
was staged, committed, pushed, downloaded, transformed, built, or executed.
Existing checkouts under `~/src` were neither updated nor modified, and the
protected NVMe remained read-only and unmounted.
