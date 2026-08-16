# H5 gate record — deterministic reduction and atomic transaction

**Status:** passed on 2026-08-16. H5 adds exact GPU0 rank-ordered
reduction over H4's canonical contribution arena and an opt-in private-K/V
transaction coordinator with generation-tagged metadata. It does not yet wire
the complete real layer forward path; H6 owns that integration gate. The
existing default GPU engine remains unchanged rather than publishing block
tables before sampling, output, evidence, and visibility are ready. The commit
introducing this record is the H5 completion commit.

All final machine-readable records bind non-document source fingerprint
`0d5b19f1b5b92fe0a6689d029607febbabadab33236c882a60b0593c8dde6439`
to package-start commit `b610b968715a810d6c1191140c6cfb951dacc3eb`.
The fingerprint is SHA-256 over sorted tracked and untracked, non-ignored file
paths outside `docs/`, with each path and file hash separated by NUL bytes.

## Final records

| Record | Result |
|---|---|
| [`reduction.json`](reduction.json) | Passed real CPU/GPU0/GPU1 canonical-arena reduction, route/result identity negatives, generation lifecycle, construction faults, post-enqueue drain faults, exact traces, and bounded memory accounting |
| [`transaction.json`](transaction.json) | Passed 64 named public-interface cases at their true state boundaries, including six owner completion permutations and a clean second commit after every failure/cancellation case |
| [`stale-identity-cleanup.json`](stale-identity-cleanup.json) | Passed separate stale revision/visibility/placement/block-generation and cleanup-failure quarantine cases, each followed by repair and a clean commit |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 identities for the bounded JSON evidence |

The real reduction executable is
`f28f30c9692f7d00b013fb91a13b3780e14a84c45005f25722c0c2b52d9704f2`;
its compiled `sm_86` reduction PTX is
`ce428af9a65a71b7148fbe63dac26a0c2f69b9212994d1ff9ff732c553535896`.

## Canonical arena, identity, and generation lifecycle

GPU0 owns one `[route_rank=4][hidden=2880]` BF16 contribution arena on the
same CUDA context and stream as H4 relay. CPU and GPU1 results are uploaded
first after the arena clear; GPU0-local results then enter their route-bound
slots by D2D. A local result slot carries its transaction generation and full
route contract, so a computed slot cannot be relabeled with an identical-
looking descriptor. The reducer requires all four exact contracts and reads
the resident arena directly; it uploads only the four BF16 routing weights.

The relay now owns an explicit active generation. Binding succeeds only when
none is active and the new generation is greater than both the last binding
and arena generation. CPU/GPU1 upload does not release that ownership. The
real gate rejects a rebind both before upload and after remote upload, rejects
abandon without a proven all-owner drain, then proves drained abandon followed
by a clean higher-generation bind. Successful reduction closes the active
generation only after the terminal event and all output/trace D2H operations
are drained. Faulted reduction leaves the generation active for an explicit
drained discard or a same-generation retry.

Missing, duplicate, wrong expert, wrong BF16 weight, wrong owner, wrong slot,
wrong plan, relabeled local slot, and stale-generation results all fail before
reduction. The selected real route remains `[31,21,22,6]`, with GPU0 owning
31/6, CPU owning 21, and GPU1 owning 22. Output, every weighted f32 value, and
every rank accumulator bit match the CPU authority. The retained hashes are:

| Boundary | SHA-256 |
|---|---|
| Final BF16 output | `e3275456e27b341c38b5c9146d1fb9843d2a03f0e016a3d90cfc9f433200edbd` |
| Per-rank weighted f32 trace | `03d9c3ae8827abb546193e7ef70fde8047dbfe14286f31a1e0fb7af5a14e0a6e` |
| Per-rank accumulator f32 trace | `e7f8224ab7585b43d883b0e16046b6e14175a6eab5ae79b2d10807128264f2b6` |

## Bounded reduction memory and fault recovery

Every host result/trace vector and canonical descriptor is prepared before
dispatch. The production reduction path performs no opportunistic host
allocation. Device accounting is:

| Allocation | Bytes |
|---|---:|
| H4 canonical contribution arena | 23,040 |
| Four BF16 weights | 8 |
| BF16 output | 5,760 |
| Weighted f32 trace | 46,080 |
| Accumulator f32 trace | 46,080 |
| **Reducer-owned device bytes** | **97,928** |
| **H4 + H5 pipeline bytes** | **120,968** |
| Workspace class | 131,072 |

Four deterministic construction faults occur after each reducer allocation;
each drains initialization work and restores driver-visible free memory within
4 KiB. Three execution faults occur after weight H2D, kernel launch, and trace
D2H enqueue; each performs a mandatory stream drain before host buffers can be
released. If both terminal synchronization and the fallback drain fail, the
relay/reducer is poisoned and DMA-referenced host storage is quarantined for
process lifetime.

The final run observed zero change in `cuMemGetInfo` before/after reducer
construction because cudarc reused already-reserved context allocator capacity.
That is not recorded as zero allocation: the exact logical ownership remains
97,928 bytes, the combined bound remains 120,968 bytes, and allocator reuse is
reported separately. Dropping the healthy reducer returned driver-visible free
memory to the pre-construction value within 0 bytes. All five pinned pools
reached high-water one and returned to one available lease with none
quarantined.

Two preliminary measurement assertions are preserved in the attempt history:
one incorrectly required driver free-memory delta to cover logical ownership;
the independent rerun measured the actual zero delta and established allocator
reuse. The final schema separates these quantities instead of weakening the
120,968-byte bound. One earlier exploratory build used invalid
`CUDA_ARCH=86`; it was interrupted and is not gate evidence.

## Private K/V publication and failure semantics

`HeterogeneousTransactionCoordinator` owns generation-tagged private append
tables, one active step per sequence, pre-reserved reduction/output/evidence
storage, and the single visibility epoch. K/V writes address final physical
slots through a request-private table, but committed readers retain the prior
table, length, tokens, output, evidence, revision, and epoch. The commit
revalidates request revision, placement epoch, visibility epoch, active-step
identity, block generations, output image, and drain state before mutation.
It then performs allocation-free swaps and advances visibility last. The
counting allocator observed zero allocations during commit.

The opt-in `HeterogeneousGpuMetadataAdapter` binds metadata tickets to an
engine-instance identity and checked generation. A dropped active ticket moves
its physical `ModelInput` vectors into adapter quarantine until a proven drain;
cross-engine collection, `collect(None)` with an active ticket, undrained
cancellation, and generation reuse fail closed. Coordinated shutdown succeeds
only when both adapter tickets and coordinator steps reach zero. H6 must wire
this adapter and transaction together; H5 deliberately does not alter the
default engine's prelaunch/publication behavior in isolation.

The public matrix has 9 pre-dispatch and 55 post-dispatch cases. These are
coordinator lifecycle injections, not substitutes for concrete CUDA fault
tests. Lower-layer authority is split precisely: H2 `failure.json` and
`resolution.json` cover selected-expert submit/kernel/D2H drain; H4
`real-x8-relay.json` covers pinned GPU0/GPU1 relay legs and post-enqueue drain;
and H5 `reduction.json` covers canonical result identity, weight H2D,
reduction-kernel, trace-D2H, terminal-drain, and reducer-allocation faults. The
H2 and H4 executable gates were rerun on this final H5 source.

The coordinator matrix covers
manifest, stable-device, ownership, bounds, route, K/V reservation, real
capacity-one queue reservation, CPU/GPU launch/H2D/D2H/result/reduction errors,
Reserved/Prepared/Reduced/Ready cancellation, 42 owner-order/completion-boundary
cancellations, publication-image failure, post-commit delivery failure, and
opposite CPU/GPU1 failure orders. Every case snapshots revision, visibility and
placement epochs, table/generations, tokens, output, evidence, delivery state,
free blocks, and active steps; then it runs a clean second commit. Every
pre-commit discard case has an identical authoritative before/after snapshot.
The one post-commit delivery case explicitly permits only `delivery_failure`
to change and records that exception; committed K/V, revision, epoch, tokens,
output, and evidence remain unchanged.

The private probe separately proves stale request revision, visibility epoch,
placement epoch, and block generation. A simulated cleanup/free failure retains
the original CPU primary, authoritative step ownership, and quarantined block
capacity until repair. Registration rejects token/length mismatch; commit image
length must equal private K/V length; checked slot geometry prevents `u32`
aliasing. Pinned allocation authority remains the real H4 bounded-pool test,
while device/scratch construction authority is the real reduction record; the
host taxonomy matrix does not inflate those into synthetic CUDA claims.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the lockfile, Rust 1.97.1,
CUDA 13.3 targeting `sm_86`, and NVIDIA driver 610.43.02.

| Command or check | Result |
|---|---|
| Release `heterogeneous_reduction_cuda` with `cuda,heterogeneous-test-faults` and H5 evidence enabled | Passed the real three-owner result arena, exact rank traces, route/generation negatives, four construction faults, three post-enqueue faults, and clean higher-generation reuse |
| `heterogeneous_transaction` integration evidence test | Passed 64 named cases, six owner permutations, zero-allocation commit, coordinated shutdown, and clean second commits |
| Private stale-identity/cleanup probe | Passed five stale/quarantine cases with repair and clean retry |
| `cargo test --locked -p gpt-oss-gpu --features cuda bounded_pool` | 1 passed; the actual bounded-pool filter proved exhaustion and drained reuse |
| Release H4 real x8 three-owner relay regression | Passed exact work, fixed pools, correlated overlap, and post-enqueue drain with the H5 relay lifecycle |
| H2 synthetic selected-expert CUDA regression on both GPUs | Passed exact special values, lifecycle faults, and repeat reuse |
| H2 real four-expert/two-GPU oracle regression | Passed all eight expert/device routes exactly |
| `cargo check --workspace --locked` and `cargo test --workspace --locked` | Passed on final H5 source |
| Three configured strict Clippy lanes | Passed on final H5 source |
| `CUDA_ARCH=sm_86 cargo build --release --locked --features cuda` | Passed; pre-existing experimental model-runner/engine warnings remain inventoried |
| Python unit discovery | 35 benchmark-tool and 10 oracle tests passed |
| `cargo fmt --all -- --check`, `git diff --check`, Markdown links, fingerprint, checksum, fallback/scope audit | Passed at package close |

## Scope and safety

H5 adds no dependency or lockfile change. The reduction kernel is serial in
route-rank order and cannot use atomics, completion ordering, NCCL, P2P,
NVLink, tensor parallelism, the rejected all-expert CUDA MoE path, or expert
weight movement. No 20B generation or 120B load/execution was needed for this
package. No model was copied or transformed. The authorized 26,508,424-byte H3
x8 cache remains reusable and unchanged. No remote state changed. At package
close, swap was zero, both GPUs were idle, and `/dev/nvme1n1` remained
read-only and unmounted.
