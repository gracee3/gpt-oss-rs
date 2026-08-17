# R3 capacity-one construction evidence

Date: 2026-08-16 America/New_York

R2 decision: `approved_for_r3`

R2 policy identity: `f269a4c984bbfa0d2a18c037b42ded2c81330094b18c6fc8dc668b7ad81bb90f`

Terminal verdict: `ready_for_20b`

This is bounded source, synthetic-fixture, and supported non-model CUDA
evidence. No real 20B or 120B payload was read, mapped, copied, hashed,
transformed, or executed. The retained-20B comparison is a separate handoff
and was not executed; H8, H9, and H10 remain unauthorized.

## Identity and local history

The accepted starting content was
`249abfbf5f21dddb434a7975c02df396e0608dc7`. The initial checkout label was
`agent/h3-incremental-gpu-expert-assembly`; local `main`, `origin/main`, and
`HEAD` all named the accepted content and the worktree was clean. The task
checked out local `main` without changing content and created exactly
`agent/r3-capacity-one-constructor`.

Complete implementation commit list:

1. `0e3e650` — Implement capacity-one shard planning and release proofs.
2. `28fc7f0` — Add durable incremental CPU x8 record transactions.
3. `4c8edc9` — Integrate production capacity-one owner construction.
4. This record's introducing commit — documentation and validation closure.

The fourth hash cannot be embedded in its own committed bytes. It is the sole
commit after `4c8edc9` on the final branch and is resolved by `git log
4c8edc9..agent/r3-capacity-one-constructor`. No commit was amended, merged, or
pushed. The code-complete diff is 13 files, 5,103 insertions, and 84 deletions;
`changed-files.tsv` records final task-file ownership. The final reviewed diff
from the accepted baseline is 20 files, 5,406 insertions, and 105 deletions.

## Production call graph

```text
heterogeneous_construct --constructor capacity-one
  -> SafeTensorShardCatalog (bounded headers; payload fds closed)
  -> GptOssNativeCatalogMap
  -> GptOssShardConsumerPlan
  -> bounded warm-record validation and exact action elision
  -> OwnerSelectiveConstructor::construct_capacity_one
       -> final dense/GPU allocations and two fixed pinned leases
       -> plan-derived ExpertPartialStore
       -> one ScopedShardConsumerTransaction at a time
       -> synchronous dense/GPU ownership
       -> one cold CpuOwnerLayerTransaction at a time
       -> MADV_DONTNEED, unmap, POSIX_FADV_DONTNEED, close
  -> OwnerSelectivePublicationProof
  -> fresh CPU-record validation and mapping after source count zero
  -> OwnerSelectiveModel publication
```

The selector defaults to `monolithic-control`; `capacity-one` is explicit and
requires the frozen policy identity and an explicit placement manifest. The
new path branches before any `GptOssCheckpointView::open`. Its constructor body
contains no `GptOssCheckpointView`, `CpuTensorStore`, whole-model
`SafeTensors` view, or `Vec<Mmap>`. The model architecture and decode graph are
shared with the control path.

## Large-object lifetimes and measured fixture bounds

| object | production bound | synthetic observation | publication state |
|---|---:|---:|---|
| native source mapping | exactly one; at most 10,544,040,680 B | high-water 1; current 0 | absent |
| native source fd/view/slice | one scoped fd and callback borrow | all current counts 0 | absent |
| split-bias store | plan-derived; at most 1,474,560 B | 2 entries / 23,040 B high-water; empty at end | absent |
| pinned construction leases | two fixed 16-MiB owner leases, 33,554,432 B | fake sink proves terminal-gated reuse; static production bound 33,554,432 B | absent |
| x8 conversion scratch | one, at most 1,114,112 B | one whole-expert scratch | absent |
| cold CPU output | one layer, at most 675,941,760 B | one expert / 13,253,760 B | durable record only |
| runtime CPU record mapping | deferred | first map at source count 0 | immutable runtime owner |
| dense/GPU destination | final allocation | exact deterministic bytes | terminal stable-PCI owner |

The retained-plan test independently derives 640 split entries and the exact
1,474,560-byte simultaneous bound, with owner maxima 403,200 B (GPU0),
495,360 B (GPU1), and 587,520 B (CPU). A plan above the bound is rejected.

## CPU publication and warm-record results

The incremental transaction exercises the exact state sequence:

```text
absent -> temporary -> partially_filled -> complete_unpublished -> synced
       -> renamed_visible -> directory_synced -> runtime_mapped
```

Each expert admits six exact native actions, converts a whole expert, handles
short positioned writes, and marks completion only after exact output
coverage. File sync precedes optional advice and close. Same-filesystem hard
link publication is atomic and no-overwrite, directory sync follows, equal
collisions reuse an exactly matching record, and mismatches remain untouched.
A post-rename/pre-directory-sync failure preserves and freshly revalidates the
visible record. Cleanup removes only the captured device/inode task temporary.
Warm validation is bounded and retains no mmap; action elision binds source,
mapping, placement, format, layer, ordered expert IDs, and record identity.
Runtime mmap requires a fresh match after every native mapping is gone.

## Failure-injection matrix

| failure or missing proof | asserted terminal outcome |
|---|---|
| header/range/path/source replacement | rejected before payload admission |
| capacity-two mapping, callback error, panic unwind | high-water remains one; fd/mapping counters return to zero |
| duplicate/missing/out-of-order action | exact coverage or CPU transaction rejects; no publication |
| split bias wrong length/owner/order/hash/later identity | entry rejected or retained until explicit cancellation; store then empty |
| before CUDA enqueue | recoverable cleanup; lease not reused early |
| after enqueue before proof or sync error | fatal process quarantine; device, pin, and transaction ownership retained; publication forbidden |
| positioned short writes | loop completes exact bytes before completion |
| injected ENOSPC/write error and gap | owned temporary removed; no target |
| before sync or after sync/before publish | owned temporary removed; no target |
| equal no-replace collision | exact immutable target reused |
| mismatched no-replace collision | target preserved; task temporary removed; error returned |
| after publish/before directory sync | target preserved and freshly validated; error returned |
| cancellation in pre-visible states | only identity-verified task temporary removed |
| advice unsupported/error/no-effect | exact errno/result reported; never treated as eviction or ownership proof |
| invalid warm record | rejected; native actions remain required |
| every publication proof field missing | publication rejected independently |

The deterministic three-shard fixture contains dense, local GPU0, local GPU1,
split GPU, split CPU, cold CPU, valid warm, and mismatched warm cases. It proves
exact action/range/byte coverage, warm elision, old/new x8 payload parity,
deterministic repeat output, mapping reuse, empty terminal partial state, and
zero source objects in the published representation. A deterministic fake
terminal sink covers exhaustive upload ownership; the supported real-CUDA
router tests add stable-identity/context evidence without model payloads.

## Validation summary

`validation.tsv` is authoritative for individual results. Highlights:

- locked metadata, targeted CPU/CUDA compile, and the capacity-one benchmark
  binary compile passed;
- all catalog, native-map, plan, transaction, split-store, record durability,
  publication, policy, and synthetic parity tests passed;
- the CUDA owner-selective contract suite passed 10/10 and the supported
  heterogeneous CUDA router suite passed 6/6;
- the complete locked workspace test suite passed with every real-model opt-in
  unset;
- format, whitespace, Markdown links, final source searches, and diff checks
  passed;
- CPU-kernel warnings-denied Clippy passed; a changed-file/changed-line
  warnings-denied classifier reported zero R3 findings. Whole-crate CUDA
  `-D warnings` remains unavailable because unchanged dependency/crate code has
  pre-existing Rust/Clippy findings; no baseline lint was modified in this
  focused task.

## Frozen admission policy

The canonical bytes are the full UTF-8 contents of
`/home/emmy/workspace/gpt-oss-het-120b-r0-2026-08-16/r2/16-memory-release-and-admission-policy.md`.
Its SHA-256 is recorded in `r2-policy.sha256` and compiled into the selector.

| gate | frozen value |
|---|---:|
| preflight | at least 120,000 ms and five samples |
| post-release settle | 30,000 ms, then five samples 1,000 ms apart |
| `MemAvailable` floor | 12,884,901,888 B |
| memory PSI `some/full avg10` | exactly zero millionths |
| process/cgroup swap | exactly zero |
| global swap used | no increase from admitted baseline |
| cgroup clean-file allowance | 11,488,417,896 B |
| dirty plus writeback allowance | 944,377,216 B |
| post-exit cgroup drift | 67,108,864 B |
| cold-record filesystem reserve | 68,719,476,736 B |

The checked equations reproduce exactly: expected process construction
10,930,236,520 B; expected host/cgroup CPU stage 11,606,178,280 B; and
no-reclaim upper 89,756,854,784 B. Conservatively adding the separately named
33,554,432-byte pinned budget yields 10,963,790,952 B,
11,639,732,712 B, and 89,790,409,216 B. Metadata, allocator, filesystem, CUDA,
OS, and unrelated-host reserves remain separate.

The predeclared fallback predicate is unchanged: only exact old/new identity
and output parity combined with a declared cache/residency gate failure may
motivate a separately authorized synchronous-`pread` proposal. Correctness
failure blocks. No fallback is implemented.

## Deliberate omissions and remaining risks

- No real-model construction or inference, retained-20B comparison, 120B,
  H8/H9/H10, fallback, Docker, download, external checkout update, dependency
  update, or system/cgroup/swap/filesystem/NVIDIA mutation was performed.
- The first retained-20B cold/warm/repeat comparison must measure advice/cache
  efficacy, process PSS, cgroup clean/dirty residue, pinned/VRAM ledgers, PSI,
  swap, and post-exit drift under the frozen gates.
- Atomic publication relies on same-filesystem hard-link no-replace support;
  unsupported filesystems fail before visibility rather than weaken the gate.
- Real CUDA success paths retain their existing synchronous stream proof.
  Exhaustive failure injection uses the deterministic sink; the driver-level
  sync-error state remains intentionally fatal and process-quarantining.

The standalone comparison handoff is
`/home/emmy/workspace/gpt-oss-het-120b-r0-2026-08-16/r2/18-retained-20b-comparison-handoff.md`.
It states prominently that this task does not authorize its execution.

No real model or protected device was touched. `/dev/nvme1n1` remained
read-only and unmounted. No remote ref, existing branch, host setting, or
external checkout changed.
