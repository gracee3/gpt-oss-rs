# H7 gate record — 20B end-to-end retained continuation

**Status:** passed on 2026-08-16. The final 20B correctness path executes the
63-token retained prompt as bounded serial `M=1` steps, then generates the
required eight-token continuation twice. Every layer uses the native BF16 GPU0
router, resident single-owner selected experts, bounded pinned host relay, and
GPU0 route-rank reduction. The rejected CUDA prefill/all-expert MoE path,
tensor parallelism, NCCL, peer access, and decode weight movement are absent.

All final records bind non-document source fingerprint
`a1f42c03cd1c2f46ea805d0623d4aa66247d05cf0eac0ea436adf40ad6422f10`
to package-start commit `e63967abb60bc248cd006fa9d40a98f8de591a79`. The
fingerprint is SHA-256 over sorted tracked and untracked, non-ignored file
paths outside `docs/`, with each path and file hash separated by NUL bytes.

## Final records

| Record | Result |
|---|---|
| [`retained-control.json`](retained-control.json) | v2: exact cold/warm eight-token continuation, 70 committed steps per run, real three-owner layer evidence, resource high-water marks, discard/retry, and cleanup |
| [`recoverable-post-dispatch.json`](recoverable-post-dispatch.json) | v1: remote-GPU post-enqueue fault after a GPU0 sibling submission; both jobs drained, all five pinned leases returned, and the clean retry succeeded |
| [`unproven-drain-quarantine.json`](unproven-drain-quarantine.json) | v1: post-enqueue plus fallback-drain failure; `drain_proven=false`, all components poisoned/quarantined, five leases retained, and retry rejected |
| [`h7-run-manifest.json`](h7-run-manifest.json) | Final source, executable, PTX, model, gate, regression, resource, and safety identities |
| [`SHA256SUMS`](SHA256SUMS) | SHA-256 identities for every bounded retained record |

The final control executable is
`c5852d6d2f3621f859aba72ad3e9fd5f7be5a7eb704f964c7188f6296d87e8e6`;
the fault executable is
`9e72177b6de47f1b1f394935e2cadc2e377a41b2b7c61608db7b2f55d849c672`.

## Exact retained result

Both the cold-process load with the existing identity-valid CPU x8 cache and
the same-process warm reload produced exactly:

```text
[200005, 35644, 200008, 976, 1825, 5003, 25, 392]
```

Each run committed all 63 prompt tokens plus seven generated inputs: committed
length, request revision, visibility epoch, and runtime-visible token count all
ended at 70. The first generated token is the output of the final prompt step,
so eight outputs require seven additional committed inputs. Every run first
executed a fully dispatched cancellation, drained and discarded it without
changing committed state, then reproduced the same prediction on a clean
retry.

The retained layer-0 capture contains the real route `[31, 21, 22, 6]` with
BF16 selected-weight bits `[16128, 15926, 15915, 15903]`. Its static owners are
GPU0, CPU, GPU1, and GPU0 respectively. Four full packed admission descriptors
match four returned completion descriptors, all four expert gate/up, SwiGLU,
and down boundaries are retained as hashes, and both runs prove a strict
three-way compute intersection from one correlated timeline. GPU0-local inputs
use D2D; decode expert-weight transfer bytes are zero.

## Failure and ownership closeout

After the first selected-expert submission, `execute_layer` has no bare error
propagation. Every preparation, enqueue, sibling completion, CPU job, relay,
reduction, and residual error preserves typed ownership and the original
`drain_proven` result. Fixed result slots and the five pinned leases return only
after an all-component drain. An uncertain drain instead poisons the runtime,
owner model, shell, routers, reducers, and relays and retains every possibly
referenced CUDA or host allocation so ordinary `Drop` cannot free live state.

The recoverable real fault occurs at generation 21 after 20 committed prompt
tokens. Its fallback drain succeeds, the already-submitted sibling is drained,
the pool returns to zero checked-out leases, and generation 22 executes and
discards cleanly. The destructive case faults at the same point but also fails
the fallback drain. It returns `drain_proven=false`, leaves all five fixed pool
leases checked out for process-lifetime quarantine, poisons model/runtime/shell,
and rejects a later generation. Process exit is the destructive test boundary;
both GPUs return to idle afterward.

The coordinator's exclusive publication callback commits the shell's private
token/K/V state before the single coordinator visibility epoch advances last.
Any step/finalization failure handled inside the runtime performs the same
all-component drain/abandon discipline; unproven work suppresses discard and
publication.

## Memory and cleanup

| Metric | Cold run | Warm reload |
|---|---:|---:|
| Peak process RSS | 13.1511 GiB | 13.1694 GiB |
| GPU0 used at execution snapshot | 13.0859 GiB | 13.0859 GiB |
| GPU1 used at execution snapshot | 0.2910 GiB | 0.2910 GiB |
| Fixed pinned raw capacity | 74,944 bytes | 74,944 bytes |
| CPU worker high-water | 1 job | 1 job |
| Process `VmSwap` at every snapshot | 0 | 0 |
| Global swap growth | 0 | 0 |
| Pinned checked out/quarantined after run | 0 / 0 | 0 / 0 |

The host's pre-existing 28,672-byte global swap allocation remained entirely
unchanged and `SwapCached`; the target process reported `VmSwap=0` throughout.
After both unloads the GPUs returned to the same same-process idle baseline
(12 MiB and 4 MiB, including retained CUDA context state), with no second-run
free-memory loss. The authorized owner cache remains two immutable records and
26,508,424 bytes.

## Gate commands

All commands ran from `/home/emmy/gpt-oss-rs` with the lockfile, Rust 1.97.1,
CUDA 13.3 targeting `sm_86`, and NVIDIA driver 610.43.02.

| Command or check | Result |
|---|---|
| Final fault-enabled `heterogeneous_control` release build | Passed; final control/fault binaries share the frozen source fingerprint |
| Final recoverable remote post-enqueue fault | Passed; sibling drain, five-lease recovery, immediate clean retry |
| Final unproven remote post-enqueue/fallback-drain fault | Passed; no reuse, five-lease retention, full component quarantine |
| Final two-repeat full-prompt retained control | Passed exact tokens, real CPU/GPU0/GPU1 routes, strict overlap, bounded memory, discard/retry, and cleanup |
| Final H6a/H6b compatibility rerun | Passed exact boundaries, five owner-shell faults, real three-owner commit/discard/repeat, and resource gates |
| H2 synthetic and real four-expert/two-GPU regressions | Passed |
| H4 native router and real x8 relay suites | 5 and 3 passed |
| H5 real rank-reduction suite | 2 passed |
| `cargo check --workspace --locked` and `cargo test --workspace --locked` | Passed |
| Three configured strict Clippy lanes and final fault-enabled H7 bench lane | Passed |
| Python benchmark-tool and oracle discovery | 35 and 10 passed |
| Format, diff, Markdown links, checksum, fallback/scope, and cleanup checks | Passed at package close |

No model was copied or transformed, no 120B load or execution occurred,
nothing was pushed or merged, the pinned source checkouts were not changed,
both GPUs returned to idle, and `/dev/nvme1n1` remained read-only and
unmounted.
