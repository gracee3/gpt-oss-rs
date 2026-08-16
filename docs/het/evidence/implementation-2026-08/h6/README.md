# H6 gate record — real one-layer three-owner oracle

**Status:** passed on 2026-08-16. H6 executes the real 20B layer-0 decode
fixture at position 63 through the GPU0 layer-owner shell and executes all four
selected experts on their resident static owners: GPU0 experts 31 and 6, CPU
expert 21, and GPU1 expert 22. This is a one-layer correctness and transaction
gate, not the H7 end-to-end retained-continuation gate.

All final records bind non-document source fingerprint
`3635bcf9bcc46a50c480b99bb1c4f5601c1aca68420c8e054490811f52bd1183`
to package-start commit `dff8b0916017cc736067f806f0a1e3464eb6cd70`.
The fingerprint is SHA-256 over sorted tracked and untracked, non-ignored file
paths outside `docs/`, with each path and file hash separated by NUL bytes.

## Final records

| Record | Result |
|---|---|
| [`h6a-owner-shell.json`](h6a-owner-shell.json) | v4: twelve bit-exact dense/KV/attention/router/reduction/residual boundaries, fixed owned host staging, five post-enqueue drain/retry cases, and device-resident shell handoffs |
| [`h6b-three-owner.json`](h6b-three-owner.json) | v4: real four-route CPU/GPU0/GPU1 execution, full packed and completion identities, per-expert boundaries, strict three-way overlap, commit/discard/repeat, and process/global resource snapshots |
| [`h6b-owned-d2d-unproven.json`](h6b-owned-d2d-unproven.json) | Destructive final-binary probe: unproven D2D drain returns no slot, retains all referenced storage, quarantines executor/model/shell, and forbids retry |
| [`h3-teardown-warm.json`](h3-teardown-warm.json) | v2 warm owner-selective construction regression after fail-closed public-drain changes |
| [`h3-teardown-faults.json`](h3-teardown-faults.json) | v2 eight-stage real-constructor rollback campaign plus clean construction |
| [`h6a-run-manifest.json`](h6a-run-manifest.json) | v3 final source, binary, PTX, artifact, test, resource, and cleanup identities |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 identities for every bounded retained record |

The final fault-enabled layer-oracle executable is
`f8f46e2b3fc1dabff88e49452a2a39fb15436b3da8f8bfab226b8c112b72fb23`.
The construction probe is
`674c849bebe3a620d9916c191f69b2d44ce4ebb1c5057c1912bcdf0dade2a4ca`.

## Exact one-layer authority

The retained `ResidualQ8` runner is authority only through the dense, private
K/V, attention, post-attention residual, router-input, and real route/weight
boundaries. Expert authority is independently recomputed from the native
MXFP4 views using the exact H2 semantic path. GPU0 authors the BF16 router
result `[31,21,22,6]` with selected BF16 weight bits
`[16128,15926,15915,15903]`; every GPU descriptor must exactly match the
pre-reserved 2/1/1 oracle admission before dispatch.

GPU0-local activations remain device resident: the two local expert routes
use 11,520 bytes of D2D activation movement and zero host execution bytes.
GPU1 receives and returns 5,760 bytes. CPU and GPU1 each upload one 5,760-byte
unweighted BF16 result, while both GPU0 results enter their canonical slots
through 11,520 bytes of local D2D. GPU0 reduces strictly in original routing
rank order and applies the resident final residual. All four per-expert
gate/up, SwiGLU, and down boundaries, the reduction, and the layer output are
bit-exact.

All three cases—commit, drained cancellation/discard, and the clean repeated
commit—prove a strict simultaneous compute intersection:
`max(cpu_begin,gpu0_begin,gpu1_begin) < min(cpu_end,gpu0_end,gpu1_end)`.
The evidence retains every interval rather than inferring overlap from wall
time. Each case carries four full packed descriptors and four matching
completion identities, including source/activation slots, route rank, expert,
BF16 weight, CPU pool or stable GPU identity, placement epoch, and canonical
result slot.

## Lifecycle and publication boundary

The coordinator step ID is the relay, result-slot, and reducer generation.
Every bounded lease, result slot, trace, and reducer allocation is reserved
before router dispatch. The rank-reduction obligation becomes terminal only
after the resident residual is terminal and exact. The committed case advances
one visibility epoch; cancellation suppresses publication, drains all six
roles, and discards; the same sequence then commits cleanly. Coordinator active
steps, generation-tagged blocks, result pools, and all five pinned pools return
to zero/available state.

The proof claims atomic generation-tied coordinator metadata publication. It
does **not** claim that the owner shell's private flat K/V oracle storage is the
coordinator's physical block store. H7 owns the end-to-end model integration
and retained-token gate.

Every public final drain is fail-closed. A failed shell synchronization poisons
the shell and retains its CUDA state plus referenced host staging. A failed
owner-selective model drain or Drop quarantines both executors and all GPU
expert/dense weights. Router, selected-expert, relay, and reducer unproven
drains likewise retain complete CUDA/host ownership and cannot be rehabilitated
by a later successful sync.

## Resource interpretation

The target process reports `VmSwap=0` before and after H3 and H6. The host had a
pre-existing 28,672-byte global swap allocation, all reported as `SwapCached`;
it remained exactly 28,672 bytes before and after every final retained run.
No live process reported nonzero `VmSwap`, and global swap growth was zero. This
is recorded honestly rather than treating unrelated unchanged swap cache as
target-process swapping.

H6b observed zero free-memory loss on both GPUs, zero checked-out pinned leases,
zero active coordinator steps, and zero unreturned generation blocks. The H3
warm and eight-stage fault regressions preserved the 26,508,424-byte immutable
owner cache, left no partial artifact, returned every pinned lease, and restored
CUDA memory within tolerance.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the lockfile, Rust 1.97.1,
CUDA 13.3 targeting `sm_86`, and NVIDIA driver 610.43.02.

| Command or check | Result |
|---|---|
| Final fault-enabled `heterogeneous_layer_oracle` H6a+H6b run | Passed H6a twelve-boundary/five-fault gate and H6b commit/discard/repeat three-owner gate |
| Final destructive owned-D2D unproven-drain probe | Passed; no slot returned, full ownership retained, retry forbidden |
| Final fault-enabled H3 warm plus eight-stage construction campaign | Passed process/global swap, cache, pinned, partial-artifact, CUDA cleanup, and clean-retry gates |
| H2 synthetic selected-expert CUDA regression | Passed exactness and lifecycle on both GPUs |
| H2 real four-expert/two-GPU regression | Passed all eight expert/device routes exactly |
| H4 synthetic/native/destructive router suite | 5 passed, including real E=32/E=128 weights and full pinned quarantine |
| H4 real x8 relay suite | 3 passed, including real work and recoverable/unproven lifecycle faults |
| H5 real reduction suite | 2 passed, including exact arena reduction and unproven quarantine |
| Actual bounded pinned-pool filter | 1 passed; exhaustion did not allocate and drained return reused the pool |
| `cargo check --workspace --locked` and `cargo test --workspace --locked` | Passed |
| Three configured strict Clippy lanes plus both fault-enabled bench lanes | Passed |
| Python benchmark-tool and oracle discovery | 35 and 10 passed |
| Format, diff, Markdown links, checksum, fallback/scope, and cleanup checks | Passed at package close |

No model was copied or transformed, no 120B execution occurred, nothing was
pushed or merged, both GPUs returned to idle, and `/dev/nvme1n1` remained
read-only and unmounted.
