# Phase 2 planning charter

**Stage:** planning; **capture date:** 2026-08-15; **status:** complete and
awaiting review.

## Objective and stop boundary

Convert the frozen Phase 0 baseline and Phase 1 research into one
source-grounded implementation plan for this proof:

> A real retained continuation executes selected experts in at least one layer
> concurrently on CPU, GPU0, and GPU1, while every expert has one resident
> owner, staging and queues stay bounded, publication is transactional, and
> retained evidence can identify the first divergence.

This phase selects interfaces, ownership, a commit model, work-package order,
and evidence gates. It stops before implementation. No Rust, CUDA, Python,
manifest, dependency, lockfile, model, build, configuration, CI, or Docker file
was changed. No implementation prototype, model transformation, 120B load, or
performance campaign was run.

## Frozen inputs and decisions

The authoritative inputs are [the phase index](README.md), Phase 0 documents
`00` through `05`, Phase 1 documents `10` through `19`, and the
[Phase 1 evidence index](evidence/research-2026-08/README.md). Phase 2 does not
rewrite those records.

The following decisions are fixed:

- GPU0 is the first-proof layer owner; GPU1 and CPU are expert-only workers.
- Expert ownership is static, versioned, and exactly one owner per
  `(layer, expert)`.
- GPU0 owns attention, routing, dense state, local experts, exact rank-ordered
  reduction, residual, and the authoritative step result.
- GPU1 traffic is relayed through bounded pinned host memory. CUDA P2P is
  unsupported and NVLink is inactive.
- The existing tensor-parallel/NCCL model path and current CUDA prefill/decode
  MoE are not implementation foundations.
- Native shards are authoritative. Q/K/V use validated slices; GPU experts keep
  native MXFP4; only CPU-owned experts have persistent x8 records.
- The CPU route/expert/reduction path remains the semantic authority. The
  exact boundaries in [document 13](13-exact-expert-contract.md) are not
  relaxed.
- Proof placement and later performance placement are separate. No popularity,
  crossover, migration, or prefetch threshold is frozen here.
- This plan selects **private KV append slots with one visibility epoch**. The
  source basis and the ten-question selection test are in
  [document 26](26-transaction-failure-cancellation.md).

## Planning assumptions

| Status | Assumption | Consequence or gate |
|---|---|---|
| `measured` | Pinned 23,040-byte H2D/D2H takes about 11–12/6–7 us; one GPU-host-GPU relay leg takes about 18.4–18.5 us; a 13,236,480-byte expert H2D takes 1.66–1.69 ms. | Move activations/results, never weights during decode; pool pinned buffers. |
| `measured` | The approved CPU `M=1` expert matrix core is about 4.569 ms; exact resident selected-expert GPU cost is not measured. | No speedup gate or performance placement threshold exists before H2/H6/H9. |
| `proven` | The 20B retained control emitted `[200005, 35644, 200008, 976, 1825, 5003, 25, 392]`. | This exact sequence remains the pre-120B regression gate. |
| `proven` | The 120B native-to-runtime map covers 543 native and 687 runtime tensors; only Q/K/V are slices and every other payload is an alias/rename. | H3 validates the map against the local artifact before allocating owners. |
| `proven` | Native plus a full x8 120B copy is 117.647 GiB before execution state. | Any whole-model alternate expert representation is a stop condition. |
| `source-derived` | `KVCache` is raw physical storage; block tables, context lengths, and slot mappings are supplied per step (`kv_cache/cache.rs`, `worker/input.rs`). | Private physical slots can remain unreachable by withholding committed metadata. |
| `source-derived` | The current GPU layer writes K/V and then reads it for attention on the same runner stream (`gpu_layer.rs`), while generated tokens are appended only after worker output (`gpu_engine.rs`). | A prepared step needs a private read-your-writes view and a later metadata publication boundary. |
| `source-derived` | Current `GpuLLMEngine::build_metadata` mutates `seq_block_tables` before launch and has no in-flight guard. | H5 must replace early publication with a lease and enforce one in-flight step per sequence; current behavior is not transaction-safe. |
| `inferred` | A bounded, generation-tagged block lease plus exclusive engine commit is a smaller change than duplicating every layer's K/V and copying it at commit. | Private slots are selected; H5 must prove the inference with failure injection. |
| `inferred` | One D2H of each remote-needed activation can feed CPU work and GPU1 packing safely after one GPU0 event. | H4 must prove buffer lifetimes and timeline overlap. |
| `unresolved` | Cold owner-selective 120B construction peak, page-cache behavior, CUDA allocator retention, and exact selected-expert GPU time are not measured. | H8 and H10, respectively, own these gates; they do not block H0/H1. |

## Authority and prohibited scope

Authorized in this phase: read-only repository/source/evidence inspection,
`cargo metadata --no-deps --locked`, small arithmetic, documents `20` through
`29`, and this index update.

Deferred: implementation, adaptive placement, caching/eviction, expert
replication, weight streaming, P2P/NCCL dispatch, tensor parallelism,
alternating layer ownership, Xe/AMX/Qwen, graph-runtime generalization,
multi-node work, unrelated HTTP semantics, model downloads, checkpoint
transformation, and 120B execution.

## Success and failure criteria

Planning succeeds only if:

1. one architecture and one visibility model are specified at crate/source
   precision;
2. every expert, buffer, event, result, and provisional state has one owner and
   a terminal reclamation rule;
3. checkpoint construction cannot create an unbounded or whole-model alternate
   representation;
4. route rank and BF16 boundaries survive packing and backend execution;
5. packages H0–H10 are individually reviewable and later packages cannot mask
   an earlier failed gate; and
6. the first permitted handoff authorizes only H0.

Planning fails if an unresolved design choice is hidden inside a work package,
if the selected commit model permits an uncommitted read, if a current rejected
CUDA/TP path is used as a fallback, or if memory/queue bounds depend on an
unmeasured marketing specification.

## Baseline and command inventory

The startup baseline was `main` at
`0113e8214e765d168216bbee2120654555a4cfe4`, zero ahead/behind `origin/main`.
The nine pre-existing code/lock files remained `35 insertions(+), 13
deletions(-)` with full diff SHA-256
`792b545405494ca2a5be543b24e29ee0f68420db0f3aa5ec59adf4ea114a374e`.
Pre-existing Phase 0/1 documentation remained uncommitted.

**Run:** the required Git status/HEAD/upstream/diff-stat/fingerprint commands;
bounded `rg`/`sed` source and document inspection; `cargo metadata --no-deps
--locked --format-version 1`; small byte/count arithmetic; Markdown link,
whitespace, sanitization, file-scope, and final Git fingerprint checks.

**Deliberately omitted:** builds, tests, Clippy, formatters, model loads,
checkpoint conversion/hashing, CUDA/transfer/CPU benchmarks, Docker, HTTP,
external source updates, installs, privileged/system changes, and all 120B
execution.

The Phase 2 delta is exactly the ten new documents `20` through `29` plus the
Phase 2 index entries in `docs/het/README.md`; the final exact Git status and
diff-stat are recorded in [document 29](29-implementation-readiness.md).
