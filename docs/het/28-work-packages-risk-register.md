# Work packages and risk register

Packages are dependency-ordered and stop at their own gate. “Rollback” means an
explicit opt-out or revertable package boundary between runs; it never means
mid-step owner substitution. Size is expected agent-run class, not elapsed-time
forecast.

## H0 — establish the implementation baseline

- **Objective:** make the nine workstation-readiness changes, Phase 0–2 docs,
  and later implementation separately attributable on a reviewed branch.
- **Expected files:** Git metadata only if explicitly authorized; otherwise a
  read-only boundary report. No production behavior file.
- **Interfaces:** none.
- **Prerequisites:** Phase 2 review and explicit user decision on staging/commit
  boundaries. The current nine-file hash must remain
  `792b545405494ca2a5be543b24e29ee0f68420db0f3aa5ec59adf4ea114a374e`.
- **Tests/evidence:** exact status, full diffs/hashes, branch/HEAD/upstream,
  separately enumerated Phase 0/1/2 doc delta, no unrelated file.
- **Memory/failure:** none. Never clean/reset/stash/rewrite user changes.
- **Non-goals:** code, formatting, dependency changes, squashing ownership, or
  choosing a commit policy without the user.
- **Stop/go gate:** implementation branch has an unambiguous reviewed base; the
  nine fixes and docs remain separately recoverable/attributable. Stop on any
  changed code patch or missing authorization.
- **Rollback/bypass:** remain on the untouched current tree and report; no H1
  starts until the boundary is resolved.
- **Size:** `small`.

## H1 — freeze contracts, placement, identity, and prepared-step types

- **Objective:** add reviewed data contracts with no execution behavior.
- **Expected files:** `gpt-oss-gpu/src/{lib,device}.rs`;
  `gpt-oss-model-runner/src/heterogeneous/{mod,contract,placement}.rs` and
  `lib.rs`; `gpt-oss-evidence/src/lib.rs` for versioned artifact types.
- **Interfaces:** stable PCI identity/resolution, expert key/owner/representation,
  placement manifest/validation, BF16-bit route/result descriptors, prepared
  state/evidence/error enums.
- **Prerequisites:** H0; documents 21/22/26 approved.
- **Tests/evidence:** serialization stability, E=32/128 coverage, duplicate/
  missing owners, ordinal permutation, PCI mismatch, BF16 bit roundtrip,
  descriptor stable grouping, deterministic error sort, schema golden files.
- **Memory/failure:** manifest validation is allocation-bounded by model expert
  count; no contexts/weights/jobs. Invalid manifests fail before materialization.
- **Non-goals:** loader, kernels, dispatch, K/V mutation, CLI/service wiring.
- **Stop/go gate:** contract review passes and existing runtime behavior is
  unchanged.
- **Rollback/bypass:** new modules remain unreferenced by execution.
- **Size:** `medium`.

## H2 — exact CUDA selected-expert `M=1`

- **Objective:** execute one selected native-packed expert on either GPU and
  match the CPU exact boundary.
- **Expected files:** `kernels/gpt_oss_selected_expert.cu`;
  `gpt-oss-gpu/src/kernel_loader.rs` and event/error wrappers as needed;
  `gpt-oss-model-runner/src/heterogeneous/cuda_expert.rs`; focused tests and an
  internal `gpt-oss-bench` oracle binary.
- **Interfaces:** H1 CUDA expert handle, scratch lease, prepared job,
  completion event, result descriptor.
- **Prerequisites:** H1; native weight slice fixture; no H3 full loader needed.
- **Tests/evidence:** exhaustive synthetic MXFP4/E8M0/SwiGLU, malformed inputs,
  CPU exact real 20B expert fixture on GPU0/GPU1, async failure/drain, repeated
  allocation high water and first divergence.
- **Memory/failure:** native 13,236,480-byte resident fixture; 17,280-byte
  logical scratch plus 5,760-byte output and measured alignment. No FP16 weight
  matrix or per-call allocation.
- **Non-goals:** router, all-expert scan, weighting/reduction, prefill
  optimization, whole model, speedup claim.
- **Stop/go gate:** synthetic and real-weight one-expert BF16 oracle passes on
  both GPUs; unsupported shapes fail explicitly.
- **Rollback/bypass:** wrapper is not connected to model forward; remove/disable
  the new module without changing existing CUDA/CPU paths.
- **Size:** `medium`.

## H3 — owner-selective construction and manifest

- **Objective:** construct exactly one representation for every manifest owner,
  first on 20B; mechanically prove the 120B envelope before execution.
- **Expected files:** `model_loader/{gpt_oss_native,owner_selective}.rs` and
  `mod.rs`; `cpu_tensor_store.rs`, `cpu_repack.rs`, GPU weight-store/loader
  adapters; heterogeneous placement/weight modules; internal construct-only
  probe.
- **Interfaces:** mapped aliases/QKV slices, owner-filtered x8 layer record,
  native CUDA expert handle, construction ledger/rollback guard.
- **Prerequisites:** H1; H2 handle representation fixed.
- **Tests/evidence:** complete 20B map equivalence, synthetic partial failure at
  every stage, exact ownership/bytes, 20B cold load/unload/reload, zero swap,
  stage RSS/PSS/VRAM; 120B metadata arithmetic and manifest quotas validated
  without loading it.
- **Memory/failure:** 16 MiB pinned upload, 256 MiB construction-temp cap,
  category reserve checks, reverse-order rollback; no whole tensor host `Vec`
  or full x8 alternative.
- **Non-goals:** 120B construction, execution, routing, or performance placement.
- **Stop/go gate:** 20B measured; 120B byte envelope mechanically proven; no
  retained unowned allocations.
- **Rollback/bypass:** owner-selective loader is selected only by an explicit
  manifest; existing CPU control loader remains unchanged.
- **Size:** `overnight`.

## H4 — exact router, bounded packing, and pinned relay

- **Objective:** compute exact GPU0 BF16 router projection/rounding and route
  descriptors, then issue bounded CPU/GPU0/GPU1 selected work without
  NCCL/P2P.
- **Expected files:** a focused exact router/packing CUDA source (new module or
  audited addition to `gpt_oss_selected_expert.cu`), `kernel_loader.rs`,
  `heterogeneous/{packing,cuda_expert}.rs`, `gpt-oss-gpu/src/{event,pinned_memory}.rs`,
  `gpt-oss-engine/src/worker/heterogeneous_worker.rs`, tests/probe.
- **Interfaces:** exact E=128 router record, hard-capped pinned/device leases,
  per-owner queue ticket, event dependency and result descriptor.
- **Prerequisites:** H2; H3 expert handles/manifest.
- **Tests/evidence:** native-BF16 router projection/logit/softmax oracle for
  E=32 and E=128, router ties, pack/unpack property tests,
  route-rank preservation, one-D2H dual-consumer lifetime, pool exhaustion,
  GPU0/GPU1 transfer/event and CPU overlap timelines, direction/byte counts,
  no current CUDA/NCCL symbol reached.
- **Memory/failure:** 128 KiB decode and 8 MiB `C=64` prefill pinned caps;
  capacity-one queues; all-or-none reservation; drain on copy/kernel error.
- **Non-goals:** reduction, K/V commit, full layer, placement tuning, grouped
  prefill kernel.
- **Stop/go gate:** exact routes and bounded relay/event tests pass; no P2P or
  NCCL dependency; concurrency is timeline-proven where claimed.
- **Rollback/bypass:** harness-only dispatch remains detached from model forward.
- **Size:** `medium`.

## H5 — deterministic reduction and private-slot commit

- **Objective:** integrate canonical route-slot reduction with the selected
  visibility model and prove every failure/cancellation drain.
- **Expected files:** `heterogeneous/reduction.rs` plus reduction CUDA kernel;
  `gpt-oss-engine/src/heterogeneous_engine.rs`; focused adapters in
  `gpu_engine.rs`, `worker/input.rs`, and K/V allocator metadata; transaction
  tests/evidence.
- **Interfaces:** `ProvisionalKvLease`, block generation, one-in-flight ticket,
  `SequenceCommitImage`, visibility epoch, active-step drain guard.
- **Prerequisites:** H4; H1 state/error types; document 26 approved.
- **Tests/evidence:** exact rank 0→3 reduction, all failure-injection matrix
  cases, timing permutations, stale revision/generation, cancellation at every
  boundary, output delivery failure, shutdown drain, second-run cleanup.
- **Memory/failure:** no second K/V; private metadata/buffers are reserved
  before dispatch; no reuse until terminal events; prefix cache off.
- **Non-goals:** throughput, multiple in-flight steps/sequence, CUDA graphs,
  HTTP behavior, alternate commit model.
- **Stop/go gate:** no uncommitted K/V/state/output/evidence read; deterministic
  primary errors; zero live leases at shutdown.
- **Rollback/bypass:** heterogeneous mode remains explicit opt-in; existing CPU
  control path is unchanged. Failure inside a dispatched step has no fallback.
- **Size:** `medium`.

## H6 — one-layer owner-shell and three-owner oracle

- **Objective:** first prove the GPU0 attention/KV/router/reduction/residual
  owner shell against the real CPU layer oracle, then execute the same real
  20B layer-0 decode route across all three expert owners.
- **Expected files:** integration adapters in model runner/heterogeneous engine,
  focused corrections in `gpu_layer.rs`/`gpu_runner.rs` only when a retained
  first-divergence proves them necessary; `gpt-oss-bench` one-layer oracle and
  evidence output; no service route.
- **Interfaces:** consumes H2–H5 only; adds no wider model abstraction.
- **Prerequisites:** H2/H3/H4/H5 pass independently.
- **Tests/evidence:** H6a runs the GPU0 owner shell with CPU-authority expert
  outputs and compares attention/K/V/router/reduction/residual boundaries;
  H6b uses retained real route `[31,21,22,6]`, all per-expert
  boundaries, packed/returned descriptors, deterministic reduction/residual,
  transaction commit/discard, correlated CPU/GPU0/GPU1 timeline, repeat run.
- **Memory/failure:** 20B proof placement and documented small buffer bounds;
  zero swap and no allocator drift.
- **Non-goals:** full continuation, performance, synthetic route as proof.
- **Stop/go gate:** H6a owner shell is first-divergence-clean before H6b;
  H6b produces an exact layer result with CPU, GPU0, and GPU1 actually
  executing concurrently. A shell correction larger than this bounded package
  returns for review instead of being hidden in integration.
- **Rollback/bypass:** disable heterogeneous integration; CPU oracle remains.
- **Size:** `overnight`.

## H7 — 20B end-to-end retained continuation

- **Objective:** owner-selective cold load, bounded exact prefill, and exact
  eight-token target continuation with transaction evidence.
- **Expected files:** internal heterogeneous control binary and narrow engine/
  model integration; explicit serial-`M=1` prefill correctness adapter.
- **Interfaces:** existing H1–H6; no new placement policy.
- **Prerequisites:** H6 and Tier-5 failures pass; CPU control identity valid.
- **Tests/evidence:** exact tokens
  `[200005,35644,200008,976,1825,5003,25,392]`, real three-owner layer,
  stage memory/reserve, no swap, all epochs, correlated timeline, unload/repeat.
- **Memory/failure:** 20B envelope and hard guards; prefill `C≤64`; stop on
  current CUDA prefill/all-expert fallback.
- **Non-goals:** HTTP semantics, throughput, grouped-prefill optimization.
- **Stop/go gate:** exact retained tokens, bounded memory, durable evidence,
  deterministic cleanup and repeat.
- **Rollback/bypass:** explicit CPU-only control for a new request; never
  substitute within a failed heterogeneous step.
- **Size:** `overnight`.

## H8 — owner-selective 120B construction

- **Objective:** materialize the reviewed hybrid 120B representation twice
  without forward execution.
- **Expected files:** ideally evidence/harness only after H3; production loader
  changes require renewed H3 tests and review.
- **Interfaces:** H3 construction ledger and proof manifest.
- **Prerequisites:** H7; exact local artifact/map validation; reviewed context
  cap/reserves and stop monitor.
- **Tests/evidence:** every owner once, stage-by-stage RSS/PSS/page-cache/
  anonymous/pinned/VRAM, zero swap, no full duplicate, partial failure cleanup,
  unload/reload and reserve remainder.
- **Memory/failure:** document 23 hard stops; construction only.
- **Non-goals:** layer execution, generation, placement tuning, transformation.
- **Stop/go gate:** no full duplicate, no swap, all reserves maintained,
  deterministic cleanup twice.
- **Rollback/bypass:** unload all owner pools; native shards and persistent
  identity-valid CPU records remain read-only/reusable.
- **Size:** `overnight`.

## H9 — 120B heterogeneous retained proof

- **Objective:** one-layer oracle then reviewed retained continuation with a
  real CPU/GPU0/GPU1-routed layer.
- **Expected files:** evidence/fixture integration only unless a failed prior
  gate sends work back to its owning package.
- **Interfaces:** frozen H1–H8 contracts.
- **Prerequisites:** H8, approved CPU/oracle expected continuation and safety
  bounds, review immediately before execution.
- **Tests/evidence:** first-divergence-clean one layer; exact retained tokens;
  real three-owner routes; placement/owner bytes; full timeline, memory,
  transaction, cleanup and repeat artifacts.
- **Memory/failure:** H8 measured envelope/reserves; stop before/while execution
  on guard, swap, divergence, or transaction error.
- **Non-goals:** throughput claim, synthetic route as final proof, adaptive
  placement.
- **Stop/go gate:** correct output, real three-owner routes, bounded memory and
  complete durable evidence.
- **Rollback/bypass:** terminate/discard request after drain; no owner fallback.
- **Size:** `overnight`.

## H10 — measure and tune static performance placement

- **Objective:** derive a new static performance manifest from exact measured
  GPU, CPU, transfer, event, occupancy and reserve costs.
- **Expected files:** benchmark/evidence and placement-policy module only; any
  arithmetic kernel change returns to H2/H6/H9 gates.
- **Interfaces:** immutable placement manifest version; no migration API.
- **Prerequisites:** H2, H6 and H9 evidence; no unresolved correctness/memory
  gate.
- **Tests/evidence:** route-frequency captures, exact selected GPU/CPU costs,
  interference/packing/events, decode/prefill crossovers with uncertainty,
  exact regression after every change.
- **Memory/failure:** quotas recomputed with measured reserves; no weight move
  during a run.
- **Non-goals:** adaptive migration, caching, prediction, replication,
  approximation, universal policy.
- **Stop/go gate:** static manifest improves an honestly reported metric without
  arithmetic/transaction/memory regression; no minimum speedup is predeclared.
- **Rollback/bypass:** use the versioned proof placement manifest on a new load.
- **Size:** `overnight`.

## Dependency and critical path

```text
H0 -> H1 -> H2 -> H3 -> H4 -> H5 -> H6 -> H7 -> H8 -> H9 -> H10
```

H2 can begin with an extracted fixture while H3 is being reviewed, but the
promotion path remains ordered: no H4 integration consumes unvalidated owner
handles, and no H8 starts before H7. H10 is never used to compensate for H9.

## Risk register

Likelihood is `low`, `medium`, or `high` before mitigation.

| Risk | Likelihood / consequence | Detection evidence | Prevention | Explicit stop condition |
|---|---|---|---|---|
| Cross-backend rounding divergence | high / wrong layer/tokens | first-divergence BF16 and 16-lane traces | CPU exact operation order, per-boundary kernel gates | Any mandatory BF16 mismatch |
| GPU0 attention/dense owner-shell divergence | high / retained proof impossible | H6a attention/K/V/router/residual first-divergence trace | isolate owner shell before three-owner integration; patch only proven boundary | H6a mismatch or correction exceeds bounded scope |
| CPU x8 creation peak | medium / swap or unresponsive host | per-layer RSS/PSS/anonymous/temp, swap | owner-only streamed layer records, 256 MiB temp cap | Guard breach, swap growth, whole-layer/full-model alternate |
| Mmap/page-cache misconception | high / hidden host exhaustion | virtual vs file RSS/PSS vs `MemAvailable` | separate ledger; conservative guards; no “mmap is free” claim | `MemAvailable<12 GiB`, RSS cap, or swap |
| Pinned allocation exhaustion | medium / host pressure/failure | leased/free/high-water counts and bytes | warmed hard-capped pools, all-or-none reservation | allocation beyond cap or untracked pinned bytes |
| CUDA event/stream lifetime bug | medium / use-after-free/corrupt result | event generations, injected cancel/drop, sanitizer where available | step-owned handles, mandatory drain, quarantine | buffer/handle reclaimed before terminal event |
| GPU ordinal instability | medium / wrong owner/weights | PCI-to-ordinal resolution artifact and permutation tests | durable PCI identity plus capability/memory validation | missing/duplicate/mismatched identity |
| Route-rank loss during compaction | medium / nondeterministic wrong reduction | canonical vs packed/result descriptor tables | immutable canonical result slot in every record | rank inferred/reconstructed or descriptor mismatch |
| Nondeterministic reduction | high / flaky parity | repeated contribution/reduction bit traces | fixed rank loop, no atomics/tree/reassociation | repeat hash differs or BF16 mismatch |
| Partial failure/premature reuse | high / stale/corrupt request | failure matrix, lease/event/block generations | publication-forbidden monotonic state, drain coordinator | active reference when pool accepts lease |
| Cancellation with kernels in flight | high / UAF/device error | cancel at every event boundary, shutdown counts | cancellation suppresses publish; kernels drain | context/buffer/weight reclaimed before quiescence |
| GPU0 execution-reserve erosion | high / OOM mid-run | category ledger, `cudaMemGetInfo`, KV/scratch highs | quota solver includes context/KV/safety before weights | projected/actual reserve deficit |
| CPU oversubscription/SMT contention | medium / tail latency, false crossover | worker/affinity/interference records | bounded worker count; capability-based later tuning | proof misses timeout/host responsiveness or hashes differ |
| False concurrency claim | high / proof target misstated | correlated global timeline plus events | require Nsight/CUPTI equivalent; wall-time is secondary | no correlated interval evidence |
| Silent current-CUDA fallback | medium / wrong arithmetic/unbounded memory | symbol/path IDs in evidence, negative tests | explicit unsupported error and allowlist of new kernels | rejected MoE/prefill/TP/NCCL path reached |
| 120B artifact/map mismatch | medium / wrong weights/slices | source revision/index/map cardinality and byte ranges | validate 543→687 before allocation | any missing/extra/overlap/hash/revision mismatch |
| Private-slot accidental visibility | medium / corrupt K/V after failed step | committed/private tables, epoch/read tests | one in-flight/sequence, exclusive commit, generation leases | any old-epoch reader addresses private slot |
| Output delivery failure after commit | medium / client ambiguity | committed epoch plus delivery result | build internal output before commit; never rollback committed model state | delivery reports partial model state or double token |
| Stable proof placement lacks 120B three-owner route | medium / final target not exercised | real route/owner trace | quota-balanced deterministic proof map; reviewable reload with new map only | H9 has no real all-three-owner layer |

Every stopped package returns to the package that owns the failed invariant. A
later package may not weaken its gate or hide the risk with a fallback.
