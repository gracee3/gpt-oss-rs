# Failure, ownership, and commit contracts

## Current boundary

**Verified:** `PreparedCpuStep` owns provisional logits, token/KV deltas, and
captured traces. `CpuBatchEngine::commit` rechecks sequence revision, lifecycle,
and cancellation, retains valid rows, then publishes model/request changes.
Dropping/discarding prepared work exposes no partial output.

**Verified:** the current GPU forward path writes KV cache before reaching the
MoE result. A later expert/copy/kernel error therefore has no equivalent
multi-device transaction; worker failure can leave device storage mutated even
if no token is published. This path is not a safe fallback model.

## Candidate multi-device ownership timeline

```text
reserve placement + provisional KV/output/staging
  -> pack immutable rank-bearing route jobs
  -> enqueue CPU, GPU0, GPU1 work and copies
  -> collect completion events/results/errors
  -> validate all four contributions per source row
  -> deterministic rank-order reduction and residual
  -> atomically publish KV visibility, output, revision, and evidence
  -> recycle buffers only after every outstanding event is drained
```

Before publish, weights are immutable and single-owner; route jobs and outputs
belong to the prepared step; KV/output are inaccessible to later requests. After
publish, no execution fallback may replace a contribution in that step.

## Failure-state table

| Failure/cancellation point | Current behavior | Required invariant for a candidate executor | Retry/fallback status |
|---|---|---|---|
| Placement lookup or owner missing | No heterogeneous behavior | Fail before reserving or enqueueing; emit layer/expert/placement identity | Safe to retry after configuration repair; no automatic owner substitution |
| Route packing/shape/rank validation | CPU returns error before commit | No buffer/event published; exact route count remains `M×4` | Safe before dispatch |
| Pinned-pool reservation/backpressure | No production expert consumer | Bounded reservation is all-or-nothing for the layer; wait/cancel must not hold unrelated model locks | Safe before dispatch; timeout is clean failure |
| CPU task submission | CPU path is synchronous inside prepared work | A rejected job leaves no partial visible contribution; cancel other unissued jobs | Safe only before any irreversible publish |
| CPU arithmetic failure/panic | Existing Rust errors propagate; panic behavior is not a heterogeneous contract | Catch worker failure at job boundary, mark contribution unavailable, drain other device work, discard prepared state | No fallback unless a duplicate immutable weight owner was explicitly reserved |
| GPU allocation/module/kernel/enqueue failure | Error propagates; current KV may already be written | Device error belongs to prepared step; record enqueue status and do not publish partial reduction/KV visibility | Retry only after stream/device health and provisional resources are drained |
| Async H2D/D2H error | Narrow current path synchronizes stream | Staging cannot be reused until event/stream reports completion or teardown proves quiescence | No in-place retry on the same buffer while outstanding |
| One relay leg succeeds, second fails | No current cross-GPU relay | Destination buffer and host region stay provisional; drain source/destination streams; discard both | Safe retry only from original immutable input after drain |
| Partial expert completion | Current CUDA accumulates directly into output | Results carry source/rank/expert identity and remain separate until all required ranks arrive | Subset reduction/output is forbidden |
| Cancellation while jobs are outstanding | CPU engine drains cancellation before commit | Stop issuing new jobs; signal cancellable host work; device kernels may run to completion; withhold publish; drain events before reuse | Cancellation is discard, not immediate memory reclamation |
| Reduction finds duplicate/missing rank | CPU grouping assumes valid routes | Exact four unique ranks per row is validated before arithmetic; first divergence recorded | Forbidden to fill missing work with zero or reorder |
| Commit revision/lifecycle conflict | CPU commit rejects stale rows | Recheck request revision, block table/length, lifecycle, cancellation, and placement epoch under publish authority | Discard prepared result; future request may recompute |
| Cleanup/free failure | Pools have local error paths | Cleanup is idempotent; poisoned buffers/devices are quarantined; original execution error remains primary with cleanup error attached | Never return poisoned capacity to pool |
| Request/service shutdown | Async CPU engine has bounded shutdown; GPU semantics incomplete | Stop admission, cancel queued jobs, drain active streams/workers, discard provisional state, then release immutable weights | No output after shutdown boundary |

## Candidate commit barrier A: fully staged step

All new K/V, routed contributions, layer output, and request/token deltas reside
in separate provisional allocations. The coordinator awaits every CPU/GPU/copy
completion, validates exact contribution identity, performs reduction, then
copies or swaps all state into the committed cache/request under one revision
check.

**Strengths:** clearest proof, simple discard, no partially visible cache, and
evidence aligns with one prepared object.

**Costs/unknowns:** duplicates step K/V and output until commit; prefill copy
cost and peak memory grow with `M`; an atomic multi-buffer pointer swap may not
exist in current cache types.

## Candidate commit barrier B: private append slots plus visibility epoch

Reserve request-private KV append slots and output buffers in existing pools.
Kernels may write them, but block-table pointers, committed length, request
revision, and consumer-visible epoch do not advance. After every expert job and
reduction succeeds, publish the reserved slots and output by advancing one
visibility/revision boundary. Failure drains events, then returns unreachable
slots to the allocator.

**Strengths:** avoids copying complete staged K/V at commit and can use normal
append storage while preserving invisibility.

**Costs/unknowns:** requires allocator isolation, no speculative reader of
unpublished slots, generation/version checks on reused blocks, and a publish
operation spanning host request metadata plus device-visible block tables.

Neither barrier is selected here. **Inferred:** barrier B is likely the slimmer
steady-state mechanism, while barrier A is the easier first oracle fixture. The
interaction with the final executor seam and cache allocator must be settled in
planning after review.

## Fallback and evidence rules

Static single ownership deliberately avoids duplicate experts. Therefore:

- **Safe fallback:** before any job is dispatched; or after all outstanding
  streams/tasks are drained, provisional state is discarded, the request
  revision is unchanged, and an independently valid placement owns the needed
  immutable weights.
- **Forbidden fallback:** substituting CPU/GPU output after some ranks have been
  reduced or any KV/output/revision is visible; reusing staging while DMA is
  outstanding; silently changing a route owner; zero-filling a failed
  contribution; or retrying a possibly poisoned CUDA stream without reset
  evidence.

Every prepared layer record must include checkpoint/layer/request revision,
placement epoch, route/rank identities, bytes packed and copied, enqueue and
completion timestamps, first error, cancellation time, cleanup/drain outcome,
reduction validation, and a final `committed` or `discarded` marker. This extends
the existing evidence system; it does not create a parallel oracle.
