# H3 loading research follow-up

**Status:** instrumentation implemented and source/fixture validated; H8 remains
unpassed and H9/H10 remain prohibited. No 120B checkpoint was opened, mapped,
loaded, or executed during this follow-up.

## Scope and evidence boundary

This follow-up returns the unresolved 120B construction peak to H3/loading
research. It does not change the static placement, exact expert arithmetic,
global no-new-swap rule, GPU reserves, or any H8/H9 acceptance criterion.
Prior H8 records remain immutable at
[`evidence/implementation-2026-08/h8/`](evidence/implementation-2026-08/h8/)
and
[`evidence/implementation-2026-08/h8-final-authorized-attempt/`](evidence/implementation-2026-08/h8-final-authorized-attempt/).

**Verified:** the first H8 construction attempt was interrupted after global
swap allocation grew by 7,409,664 bytes, and its one authorized retry stopped
at `GpuExperts` after a further 704,512-byte increase. The target process had
`VmSwap=0` in both cases. A later final admission never launched construction
because `SwapFree` and `SwapCached` were not byte-stable during its 120-second
preflight. These facts identify a host-wide gate failure but do not localize
the constructor's file-backed, anonymous, cgroup, or GPU residency by stage.

## Current loading behavior

| Fact | Status | Source consequence |
|---|---|---|
| `CpuTensorStore` keeps every read-only shard `Mmap` in one `Vec` | **Verified** | All shard mappings live until the store drops; tensor views borrow a shard range (`crates/gpt-oss-model-runner/src/cpu_tensor_store.rs:38`, `:71`, `:120`). |
| `GptOssCheckpointView` owns that store | **Verified** | The complete mapping lifetime is the checkpoint-view lifetime (`crates/gpt-oss-model-runner/src/model_loader/gpt_oss_native.rs:83`, `:98`, `:154`). |
| The constructor receives an already-open view | **Verified** | The old first callback could not measure the checkpoint-open transition; `Mappings` merely copied `mapped_payload_bytes` into the ledger (`crates/gpt-oss-model-runner/src/model_loader/owner_selective.rs:875`). |
| Dense and GPU-expert uploads use one bounded pinned lease per GPU | **Verified** | Source shard pages are consumed while the full checkpoint remains mapped (`owner_selective.rs:880`, `:931`, `:951`). |
| CPU x8 records are created by layer after GPU uploads | **Verified** | Native expert source remains required through the CPU stage (`owner_selective.rs:973`, `:984`). |
| The current end-to-end control builds routers from checkpoint-host BF16 views | **Verified** | Dropping the checkpoint immediately after construction would break runtime router creation (`crates/gpt-oss-model-runner/src/heterogeneous/control.rs:1061`, `:2421`). |

**Conclusion:** mapping is not equivalent to resident memory, but the current
ownership shape prevents deterministic per-shard unmap. The retained
checkpoint also permits pages already consumed for GPU upload or CPU repack to
remain file-resident. The previous process RSS/HWM plus global `MemAvailable`
record could not distinguish this from anonymous conversion, cgroup pressure,
or page-cache growth.

## Implemented construction-memory contract

`crates/gpt-oss-bench/src/construction_memory.rs` now samples these domains at
the measured checkpoint-open boundary, every real constructor callback, and
post-drop:

| Domain | Retained fields |
|---|---|
| `/proc/self/status` | `VmSize`, `VmRSS`, `VmHWM`, `VmSwap`, `RssAnon`, `RssFile`, `RssShmem` |
| `/proc/self/smaps_rollup` | RSS/PSS split into anon/file/shmem, shared/private clean/dirty, anonymous, swap/swap-PSS, locked |
| `/proc/meminfo` | available/cached/reclaimable/shmem, anon/file LRUs, exact swap total/free/used/cache |
| `/proc/vmstat` | cumulative `pswpin`, `pswpout`, `nr_swapcached`, anon/file/shmem pages, major faults |
| current cgroup v2 | hashed relative identity, `memory.current`, `memory.swap.current`, optional peak, selected `memory.stat` and `memory.events` fields |
| derived residency | self RSS/PSS components, global page-cache estimate `Cached + SReclaimable - Shmem`, and global/cgroup file-versus-anon LRUs |
| CUDA | PCI identity plus used/free MiB for each visible GPU |
| construction | checkpoint mapped-address bytes and the complete existing `ConstructionLedger` |

The sampler is fail closed: a required field, arithmetic operation, cgroup
lookup, sample, or durable publication failure aborts construction through the
existing observer error path. It does not replace the established process-zero
swap and global-no-growth guards.

Events use `gpt-oss-rs.construction-memory-event/v1`. Construction modes now
require a new journal directory. Each event is serialized before the next
allocation, limited to 64 KiB, and published with create-new hard-link
semantics after file synchronization. A run is capped at 64 events and 4 MiB.
The v5 construction record indexes every file by sequence, byte count, and
SHA-256. The watchdog independently requires this exact policy, validates
checkpoint/build/placement identity, re-hashes every regular non-symlink event,
rejects extra files, and checks the summary totals before it can accept a
successful future H8 child.

**Verified by fixture/source tests:** missing fields and cgroup traversal fail;
disabled persistence still enforces byte bounds; before/after-open mapping
claims are consistent; create-new journals cannot be reused; event tampering is
rejected by the watchdog. The construction harness drops its preliminary
metadata views before reopening the measured checkpoint, so its retained
`before_checkpoint_open` and `after_checkpoint_open` events delimit the view
used by actual construction.

**Limitation:** the current harness must first open views to validate identities
and build manifests. Those preliminary views are dropped before the measured
open, but may warm shard header or payload pages in the global page cache.
Therefore a future event sequence is exact for process address-space and
resident state at each retained boundary, not proof of pristine cold-disk I/O.

## Bounded per-shard release candidate

This is a **design candidate**, not an implementation decision or authorization
to run 120B.

1. Replace the all-shard `Vec<Mmap>` catalog with immutable shard descriptors:
   canonical file identity, file length, SafeTensors data start, tensor dtype,
   shape, and checked absolute range. Parse headers with bounded reads; do not
   map payload merely to build the catalog.
2. Validate the complete 543-to-687 map and placement before materialization.
   Derive a deterministic per-shard consumption plan covering dense tensors,
   GPU0 experts, GPU1 experts, and CPU x8 records. Every required runtime view
   must occur exactly once in the plan before any payload access.
3. Admit one read-only shard mapping at a time. Its hard address-space bound is
   the largest validated shard length, not total checkpoint bytes. Within that
   mapping, consume all planned ranges in file order using the existing
   at-most-16-MiB pinned GPU staging lease and at-most-2-MiB CPU x8 conversion
   workspace. No borrowed shard slice may escape the shard transaction.
4. Before releasing a shard, prove all CUDA copies sourced from its pinned
   staging have terminally drained, all GPU weight handles own device storage,
   and every CPU record is either atomically published or still an owned
   rollback temporary. A drain uncertainty quarantines the shard mapping and
   destination state; it must never unmap referenced storage.
5. Remove the runtime's remaining checkpoint borrow: construct exact router
   handles from already-resident `LayerOwnerDenseTensor` allocations (prefer a
   validated device subrange/handle over a second payload), and retain copied
   config/mapping identities rather than the payload-bearing checkpoint view.
   This is a prerequisite for deterministic shard release.
6. After the last consumer and proven drain, a process-local
   `madvise(MADV_DONTNEED)` may be recorded as a best-effort diagnostic before
   dropping the mapping. A later `posix_fadvise(POSIX_FADV_DONTNEED)` experiment
   is optional and must be separately justified because it influences shared
   file cache. Neither hint is a fit proof, a guaranteed eviction, or grounds
   to relax the unchanged swap/memory gates.
7. Emit per-shard open/terminal/release events with checked mapped bytes,
   file/anon deltas, page-cache counters, destination VRAM deltas, and advice
   result. Keep one active shard and the existing bounded staging/workspace; a
   partial failure drains and drops only its owned transaction while preserving
   already-published identity-valid CPU records.

### Candidate stop/go tests

- Metadata fixtures prove range coverage, non-overlap where required, exact
  tensor/owner consumption, deterministic order, and checked arithmetic.
- Synthetic shard fixtures prove at most one active mapping, no escaped borrow,
  terminal-before-unmap, advice-error evidence, and rollback after every stage.
- A proven 20B cold/warm construction compares old/new exact identities,
  ownership, tokens, peak RSS/PSS/file/anon/cgroup/VRAM, no swap, and cleanup.
- Only after that review may a separately authorized H8 construction use the
  new path. The global no-new-swap gate and watchdog remain unchanged.

## Readiness conclusion

**Verified:** future H3/H8 failures can now retain the last complete,
identity-bound process/global/cgroup/GPU sample instead of only a final summary.

**Unknown:** no new model run was made, so this change supplies diagnostic
capability rather than a new 120B fit result. The actual per-shard resident
peak, page-cache response to unmap/advice, and router-handle redesign cost are
not yet measured.

**Blocked:** H8 remains unpassed. H9/H10 cannot begin. Any later 120B attempt
still requires explicit authorization and the existing exact watchdog
admission.
