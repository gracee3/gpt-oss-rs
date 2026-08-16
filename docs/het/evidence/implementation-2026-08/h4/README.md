# H4 gate record — exact router, bounded packing, and pinned relay

**Status:** passed on 2026-08-16. H4 remains a detached correctness harness:
it computes exact routes on the stable-identity layer-owner GPU, packs bounded
CPU/GPU0/GPU1 work, executes one selected route on each owner, and relays
unweighted BF16 results. It does not reduce results, mutate K/V state, publish
a model step, or enter the model forward path. Those boundaries remain H5 and
H6 work. The commit introducing this record is the H4 completion commit.

All final machine-readable records bind source fingerprint
`a3bc9b8221da7c1c6dc57b9ac51222a2cc1041f8fd27e59ec29e902832c8e563`
to package-start commit `fa7750fc28e2db18ea48ec7f9f1cfa62de4afa21`.

## Final records

| Record | Result |
|---|---|
| [`native-router.json`](native-router.json) | Passed native BF16 E=32 and E=128 checkpoint router projection, bias, selection, selected softmax, and canonical-descriptor checks on the layer-owner GPU |
| [`real-x8-relay.json`](real-x8-relay.json) | Passed the real-weight GPU0/CPU/GPU1 selected-work relay, exact outputs, fixed pools, injected post-enqueue drain, and correlated-overlap assertions |
| [`queue-contract.json`](queue-contract.json) | Passed capacity-one CPU/GPU1 queue exhaustion, all-or-none rollback, high-water, and release checks |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 identities for the bounded JSON evidence |

The native router executable is
`d3d4915aa11f440f54abba2d4fff6da03b72ba8021730f1e09c4af3777510fc7`;
its `sm_86` router PTX is
`4dbd720e19616db6968bc9fec84b353bca0c8cdbe2a0ed56bb6f5458de3fd26b`.
The relay and queue test executables are independently identified in their
records.

## Exact GPU0 router and canonical descriptors

`gpt_oss_router_stable_top4` reads native BF16 router weight and bias on the
layer-owner GPU and evaluates the 16-lane projection in CPU-authority order.
Bias is added at the documented BF16 boundary. Selection is stable, lower
expert IDs win ties, selected weights are BF16, and any non-finite input,
weight, bias, logit, or normalization result is rejected.

The final native cases use actual local checkpoint tensors, not synthetic
weights:

| Checkpoint case | Selected IDs | Selected BF16 weight bits | Result |
|---|---|---|---|
| 20B, E=32 | `29,31,15,0` | `16020,16000,15986,15974` | Bit-exact logits, IDs, and weights |
| 120B, E=128 | `74,73,72,87` | `16052,15977,15968,15951` | Bit-exact logits, IDs, and weights |

Synthetic tests separately cover exact tie ordering, bias placement, E=32,
E=128, non-finite rejection, and post-enqueue drain/reuse.

The kernel authors the complete 16-byte `GptOssRouteWireV1` record for every
route: `source_row`, `activation_slot`, `expert_id`, BF16 weight bits,
`route_rank`, and three zero reserved bytes. Decode therefore D2Hs four
canonical records, exactly 64 bytes. Rust validates the reserved bytes,
row/rank/activation identity, expert range, and canonical slot before packing;
it does not reconstruct rank or selected weight from host loop order.

## Packing and fixed-capacity accounting

Packing starts with GPU-authored canonical row/rank records, stable-groups by
owner and expert, and preserves the canonical source row even when earlier
rows are local-only. The single downloaded row-major activation arena is not
mistaken for a compact nonlocal arena.

`owner_route_slot` is category-global across every owner bucket sharing an
arena. Multiple `CpuPoolId` buckets therefore cannot both claim slot zero.
The runtime validator requires each layer-owner, CPU, and remote-GPU category
to cover one contiguous, collision-free slot range and verifies canonical
source rows. Tests include duplicate experts across rows/ranks and two CPU
owner groups, plus 192 deterministic route-batch property cases (96 each for
E=32 and E=128) that round-trip every route identity and selected weight.

Reservations charge five fixed maximum-capacity buffers regardless of the
observed owner mix:

| Buffer | Decode M=1 | Prefill C=64 |
|---|---:|---:|
| Source activation | 5,760 B | 368,640 B |
| Canonical descriptors | 64 B | 4,096 B |
| GPU1 input | 23,040 B | 1,474,560 B |
| GPU1 result | 23,040 B | 1,474,560 B |
| CPU result | 23,040 B | 1,474,560 B |
| **Raw fixed reservation** | **74,944 B** | **4,796,416 B** |

The decode and prefill hard caps remain 128 KiB and 8 MiB. Each final decode
pool owns exactly one prewarmed allocation, reached high-water one, and
returned to availability one with zero quarantined leases. Actual exhaustion
of both the first and a later pool proves all-or-none rollback without a new
allocation. The public `HostSlice` implementation and raw bounded-lease
buffer accessors were removed: safe code cannot enqueue DMA against a lease
without the relay's explicit raw-copy/event/drain discipline.

Capacity-one CPU and GPU1 queues likewise reject a second reservation,
rollback partial dual-owner reservation, and return to unoccupied state. Queue
growth and opportunistic pinned allocation are not fallback behavior.

## Real selected work, transfers, and correlated overlap

The real relay fixture uses the H3 20B placement identity and immutable x8
owner record. GPU0 executes native expert 31, the CPU executes owner-filtered
x8 expert 21 with 17,280 bytes of bounded scratch, and GPU1 executes native
expert 22. Selected IDs are `[31,21,22,6]`; the harness executes one selected
route per owner and compares each unweighted BF16 output bit-exactly with the
CPU semantic authority. It does not claim a full layer or reduction.

One GPU0 source download feeds both nonlocal owners. The final direction and
byte evidence is:

| Leg | Bytes |
|---|---:|
| GPU0 source activation plus four canonical descriptors, D2H | 5,824 |
| GPU1 packed selected activation, H2D | 5,760 |
| GPU1 unweighted BF16 result, D2H | 5,760 |
| CPU plus GPU1 results, H2D to disjoint GPU0 result slots | 11,520 |

Every leg has explicit begin/end markers on one host-monotonic timeline. GPU1
input H2D ends before GPU1 compute begins. The CPU interval
`476,599,685..565,435,478 ns` overlaps both GPU0 compute
`476,588,768..481,214,432 ns` and GPU1 compute
`476,737,093..481,284,799 ns`. The test asserts those interval relations,
marker uniqueness, ordering, and direction/byte counts; overlap is not inferred
from total wall time.

An injected failure after the first GPU0 result H2D enqueue synchronizes the
relay stream before any result slot, event, weight, scratch, or pinned lease is
released. A subsequent clean run reuses every resource exactly. The final clean
timeline excludes the fault run and contains one begin/end pair per transfer.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the lockfile, Rust 1.97.1,
CUDA 13.3 targeting `sm_86`, and NVIDIA driver 610.43.02.

| Command or check | Result |
|---|---|
| Release `heterogeneous_router_cuda` with `cuda,heterogeneous-test-faults` and native-router evidence enabled | 4 passed; actual E=32/E=128 tensors, ties/non-finites, canonical 16-byte records, and fault drain |
| Release `heterogeneous_relay_cuda` with real H3 cache/model inputs | Passed exact three-owner selected work, fixed pools, transfer legs, correlated overlap, and post-enqueue drain |
| Release `heterogeneous_queue` evidence test | Passed capacity-one/all-or-none queue contract |
| `cargo test --locked -p gpt-oss-model-runner --lib heterogeneous::packing::tests` | 5 passed, including 192-case property sweep and multiple-CPU-owner collision fixture |
| `cargo test --locked -p gpt-oss-gpu --features cuda bounded_pool` | 1 passed; exhaustion did not allocate and a drained return reused the fixed allocation |
| H2 synthetic selected-expert CUDA regression on both GPUs | Passed exact special values and lifecycle faults |
| H2 real four-expert/two-GPU oracle regression | Passed all eight expert/device routes exactly |
| `cargo check --workspace --locked` and `cargo test --workspace --locked` | Passed on final H4 source |
| Three configured strict Clippy lanes | Passed on final H4 source |
| `CUDA_ARCH=sm_86 cargo build --release --locked --features cuda` | Passed; pre-existing experimental model-runner/engine warnings remain inventoried |
| Python unit discovery | 35 benchmark-tool and 10 oracle tests passed |
| `cargo fmt --all -- --check`, `git diff --check`, Markdown links, fallback/scope audit | Passed at package close |

An exploratory broader engine strict-Clippy command and a broad model-runner
all-target feature combination expose the already recorded unrelated warning
and mock/CUDA test-shape inventory; neither is a configured gate, and no H4
warning was hidden or broadly allowed.

## Scope and safety

H4 adds no dependency or lockfile change and does not use NCCL, P2P, NVLink,
tensor parallelism, weight movement during dispatch, the rejected all-expert
CUDA MoE path, or host route reconstruction. It does not load or execute 120B;
the native E=128 router test reads only the bounded router tensors and metadata.
No model was copied or transformed. The authorized 26,508,424-byte H3 x8 cache
remains reusable and unchanged. No remote state changed. At package close,
swap was zero, both GPUs were idle, and `/dev/nvme1n1` remained read-only and
unmounted.
