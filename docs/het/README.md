# Heterogeneous GPT-OSS phase (`het`)

**Stage:** R4 retained-20B supervisor implemented; H0 through H7 passed, an
advised-release correction is validated for one final authorized R4 attempt,
and H8 remains paused;
**baseline captured:** 2026-08-15;
**scope:** exact GPT-OSS heterogeneous inference on this workstation.

Documents `00` through `05` are the unchanged pre-research baseline. Documents
`10` through `19` record the bounded Phase 1 source research and local
measurements. The research compares candidate architecture families and names
a conditional finalist. Documents `20` through `29` select one architecture,
one commit model, and bounded implementation/evidence packages. Implementation
was authorized after review and now advances only when each H0–H10 package
gate passes.

Evidence labels have the same meaning in every document:

- **Verified:** directly established by repository source, a checked-in
  artifact, supplied workstation-sanity evidence, or a recorded local command.
- **Inferred:** strongly implied by verified evidence; the inference chain is
  stated.
- **Hypothesis:** a candidate explanation or design idea requiring research.
- **Unknown:** not established by the available evidence.
- **Deferred:** intentionally outside this stage or phase.
- **Conflict:** sources disagree; both are retained with a resolution test.

## Index

1. [Phase charter](00-phase-charter.md) — mission, exclusions, stage gates,
   provisional proof target, and decision discipline.
2. [Repository baseline](01-repository-baseline.md) — exact starting Git state,
   workspace topology, evidence authority, historical/closed lanes, warnings,
   and protected artifacts.
3. [Runtime code map](02-runtime-code-map.md) — checkpoint-to-token execution,
   ownership and state, CPU and CUDA MoE behavior, transfers, multi-GPU code,
   diagnostics, and reuse/specificity classification.
4. [Host and model baseline](03-host-model-baseline.md) — sanitized topology,
   toolchains, point-in-time memory, checkpoint manifests and shapes, exact
   storage calculations, and current-host validation status.
5. [Heterogeneous readiness matrix](04-heterogeneous-readiness-matrix.md) —
   capability-by-capability implementation and proof status.
6. [Research backlog](05-research-backlog.md) — prioritized, answerable questions
   and an unvalidated source queue for the later research stage.
7. [Research charter](10-research-charter.md) — Phase 1 question, authority,
   safety gates, evidence identity, and completion boundary.
8. [Source ledger](11-source-ledger.md) — pinned source checkouts, papers,
   specifications, licenses, inspected locations, and provenance.
9. [Checkpoint, loader, and memory](12-checkpoint-loader-memory.md) — exact
   native/runtime mapping and 20B/120B byte envelopes.
10. [Exact expert contract](13-exact-expert-contract.md) — CPU semantic
    invariants, candidate seam, and one-layer oracle specification.
11. [Topology, transfer, and concurrency](14-topology-transfer-concurrency.md) —
    P2P/NCCL transport proof and model-sized transfer/CPU measurements.
12. [Prior art](15-prior-art.md) — question-led primary-source comparison.
13. [Current path validation](16-current-path-validation.md) — bounded 20B
    controls and reuse/reject decisions for CUDA and multi-GPU machinery.
14. [Failure and ownership contracts](17-failure-ownership-contracts.md) —
    failure-state analysis and two candidate commit barriers.
15. [Design candidates](18-design-candidates.md) — evidence-based comparison of
    five architecture families, without work packages.
16. [Research conclusions](19-research-conclusions.md) — answers to the required
    questions, remaining gates, and planning-readiness recommendation.
17. [Research evidence index](evidence/research-2026-08/README.md) — bounded,
    sanitized machine-readable records and complete checkpoint maps.
18. [Planning charter](20-planning-charter.md) — Phase 2 authority, fixed
    inputs, labeled assumptions, success boundary, and command inventory.
19. [Target architecture](21-target-architecture.md) — one GPU0-owner,
    CPU/GPU0/GPU1 static-expert architecture and concrete dataflow.
20. [Expert contracts and interfaces](22-expert-contract-and-interfaces.md) —
    crate-level types, lifetimes, mutation rules, evidence, and errors.
21. [Owner-selective loading and memory](23-owner-selective-loading-memory.md) —
    hybrid construction stages, exact envelopes, abort bounds, and rollback.
22. [Selected-expert CUDA plan](24-selected-expert-cuda-plan.md) — native MXFP4
    decode primitive, arithmetic boundaries, scratch, and oracle gate.
23. [Transfer, scheduling, and reduction](25-transfer-scheduling-reduction.md) —
    bounded pinned relay, streams/events, packing, prefill, and rank reduction.
24. [Transaction, failure, and cancellation](26-transaction-failure-cancellation.md) —
    selected private-slot/visibility-epoch model and failure matrix.
25. [Validation and evidence](27-validation-and-evidence-plan.md) — tiered
    fixtures, command shapes, artifacts, stop gates, and retained controls.
26. [Work packages and risks](28-work-packages-risk-register.md) — H0–H10
    dependency order, per-package gates/bypass, and risk register.
27. [Implementation readiness](29-implementation-readiness.md) — verdict,
    fixed decisions, conditions, exit criterion, and the H0-only handoff.
28. [Implementation evidence index](evidence/implementation-2026-08/README.md) —
    bounded, sanitized H0–H10 gate records.
29. [H3 loading research follow-up](30-h3-loading-research.md) — bounded
    construction-peak instrumentation, current mmap-lifetime audit, and a
    concrete per-shard release candidate; no new H8 attempt.
30. [H3 bounded shard catalog](31-h3-bounded-shard-catalog.md) — metadata-only
    index/header validation and capacity-one scoped mapping over tiny synthetic
    fixtures; not integrated into construction.
31. [H3 native metadata plan](32-h3-native-metadata-plan.md) — payload-free
    exact native/runtime mapping plus deterministic owner-specific per-shard
    consumption over source-only synthetic fixtures.
32. [H3 resident-router handoff](33-h3-resident-router-handoff.md) — bounded
    same-context D2D initialization of the exact router from owned resident
    dense surfaces, with terminal-drain release and fail-closed quarantine.
33. [H3 scoped shard transaction](34-h3-scoped-shard-transaction.md) — joins
    the bounded catalog and deterministic consumer plan through exact action
    slices, capacity-one mapping, terminal proof, and irreversible quarantine;
    not integrated into construction.
34. [H3 runtime checkpoint retirement](35-h3-runtime-checkpoint-retirement.md) —
    removes checkpoint payload ownership from the published model, extracts
    exact resident router pairs during dense upload, and single-consumes them
    in the production control runtime; source/synthetic validated only.
35. [R3 capacity-one construction evidence](evidence/implementation-2026-08/r3-capacity-one/README.md) —
    explicit non-default production constructor, one-source-mapping lifetime,
    bounded split state, durable incremental CPU records, terminal publication
    proof, and non-model validation; retained-20B comparison remains stopped.
36. [R4 retained-20B constructor comparison](36-r4-retained-20b-comparison.md) —
    commit-bound supervisor and 20B-only constructor seams; the first attempt
    failed admission and the topology-corrected retry exposed a premature
    clean-file check. An advised source-release handshake now places that
    unchanged gate at its frozen post-source-release boundary; one final R4
    attempt is authorized, while H8/120B remains stopped.

## Current headline

**Verified:** the real 20B CPU control loads, produces the pinned eight-token
retained continuation, and emits a layer-zero per-rank trace on this host. The
CPU path defines stable top-k, BF16 rounding points, GPT SwiGLU, and rank-ordered
reduction. Its prepared-step boundary remains the strongest lifecycle model.

**Verified:** every byte of all 363 native 20B tensors maps to the 459 runtime
tensors; the only payload transformation is a contiguous Q/K/V row split plus
renaming. The same complete 543-to-687 name/shape/dtype mapping is established
for 120B from local native headers and the pinned official runtime index,
without downloading payloads. Its tokenizer/protocol assets are byte-identical
to the local 20B assets, but they are not colocated with the local 120B package.

**Verified:** CUDA peer access is unsupported in both directions; the pinned
NCCL build uses shared-memory host transport. H2 now provides an exact
native-packed selected-expert CUDA M=1 primitive on both GPUs. H4 adds an exact
native BF16 GPU0 router, GPU-authored canonical route descriptors,
collision-free bounded packing, and fixed pinned CPU/GPU0/GPU1 relay with
timeline-proven overlap. Current CUDA all-expert MoE semantics and the
tensor-parallel model path remain excluded.

**Planning decision:** GPU0 is a stable-identity layer-owner role; GPU1 and CPU
are expert workers. Experts have one static owner, native shards remain
authoritative, only CPU-owned experts have x8 records, nonlocal GPU work uses
bounded pinned relay, and GPU0 reduces unweighted BF16 expert results in route
rank order. Current CUDA MoE, tensor parallelism, NCCL dispatch, P2P, and weight
streaming are excluded.

**Commit decision:** private physical K/V append slots remain unreachable
through generation-tagged per-step metadata until an exclusive, allocation-free
host commit advances one visibility epoch. Cancellation after enqueue suppresses
publication and mandates drain before any buffer, slot, weight, stream, or
context reclamation.

**Implementation status:** H0 established separate local attribution commits;
H1 froze the narrow contracts; H2 passed exact selected-expert CUDA arithmetic
and drain gates; H3 passed owner-selective 20B construction and 120B metadata
envelope gates; H4 passed exact routing, bounded packing, real three-owner
selected-work relay, queue/pool exhaustion, and correlated-concurrency gates;
and H5 passed exact GPU0 canonical-arena rank reduction, explicit active relay
generation/drained reuse, generation-tagged private K/V metadata, allocation-
free visibility-last commit, failure/cancellation/quarantine matrices, and clean
second-run gates. H6 now passes the real layer-0 GPU0 owner shell and native
BF16 router, then executes all four real selected experts across GPU0, CPU, and
GPU1 with resident single-owner weights, strict three-way compute overlap,
exact per-expert/reduction/residual boundaries, and full packed/completion
identities. Its generation-tied transaction commits, drains and discards, then
commits a clean repeat with zero active coordinator/pinned state. Public drains
and every CUDA component fail closed by retaining complete device/host
ownership after an unproven synchronization. H7 now passes the end-to-end 20B
control twice: the exact retained continuation is produced through real
single-owner CPU/GPU0/GPU1 selected work, the private shell token/K/V image is
published inside the coordinator's exclusive visibility-last commit, fixed
pools and memory remain bounded, and a real post-enqueue fault proves both
recoverable clean retry and unproven-drain quarantine. H8 owner-selective 120B
construction remains unpassed: the final separately authorized launch was
blocked before model load because its fresh 120-second watchdog preflight did
not keep `SwapFree` and `SwapCached` byte-identical. No 120B execution has
begun; H9/H10 remain stopped.

**Loading-research follow-up:** future construction runs now require bounded,
identity-bound before/after-checkpoint, per-stage, and post-drop memory events
covering process RSS/PSS file/anon, global swap/vmstat/page cache, current
cgroup memory, and per-GPU residency. This adds diagnostic evidence only. It
does not pass H8 or relax its exact swap, memory, reserve, and watchdog gates.

**Bounded-catalog follow-up:** a metadata-only SafeTensors catalog validates
index/header identity, tensor ranges, and deterministic ordering without
reading payloads. It was deliberately disconnected at that phase boundary;
R3 now uses the same scoped capacity-one mapping API in the explicit new
constructor.

**Native-plan follow-up:** caller-supplied config bytes and the bounded catalog
produce the exact 363-to-459 or 543-to-687 native/runtime mapping and a
fully covered, placement-bound per-shard consumer plan without payload mmap.
Synthetic 20B/120B action counts and schema-framed plan identities are pinned.
R3 integrates this plan into the explicit capacity-one constructor while the
monolithic path remains the default comparison control.

**Resident-router follow-up:** an isolated owned handoff can now initialize the
exact GPU router from already-resident layer-owner BF16 byte allocations using
bounded same-context D2D copies. Synthetic E=32/E=128 cases on both local GPUs
match the host-backed router bit-for-bit, and an unproven post-enqueue drain
retains every source/destination/stream/context handle. Production model
ownership and runtime wiring remain unchanged pending a separately reviewed
real-20B gate.

**Scoped-shard follow-up:** the transaction revalidates the framed
consumer-plan identity, catalog identity, shard identity, and exact action
ranges before admitting one mapping. Pre-handoff failure releases normally;
an external handoff must supply terminal proof or the mapping is retained for
process life and that catalog instance is permanently quarantined. R3 connects
it to synchronous dense/GPU upload, incremental CPU publication, bounded
release telemetry, and the final zero-source publication proof.

**R3 capacity-one follow-up:** `heterogeneous_construct --constructor
capacity-one` now reaches a production owner-selective constructor without a
whole-checkpoint payload view. It keeps the monolithic constructor as the
default explicit control, maps exactly one native shard at a time, consumes
the immutable owner plan, carries only checked split biases, publishes cold
CPU x8 layer records through a durable no-overwrite state machine, releases
all source mappings before fresh runtime record maps, and publishes only after
the complete zero-source/terminal-device proof passes. Synthetic and non-model
CUDA validation passed.

**R4 retained-20B follow-up:** the commit-bound comparison supervisor and the
monolithic/capacity-one H7 selector are implemented. The initial attempt
stopped before model load on cgroup swap and PSI. After a topology-based
protected-NVMe correction, the authorized retry passed its full preflight and
fresh admission, then stopped the first cold monolithic cell when clean-file
growth exceeded the frozen allowance by 22,124,440 bytes. No terminal
construction output, comparison cache, later matrix cell, H7 repeat, H8, or
120B access followed. The corrected constructors now advise, unmap, and close
source mappings, publish a fail-closed release proof, and wait while the
supervisor performs the unchanged clean-file settle gate. One fresh R4 attempt
is authorized from a clean pushed correction commit; H8/H9/H10 remain stopped.
