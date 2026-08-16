# Heterogeneous GPT-OSS phase (`het`)

**Stage:** implementation campaign; H0 through H5 passed, H6 is next;
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
second-run gates. H6a now passes the real layer-0 GPU0 owner shell through the
native-BF16 router, exact native-MXFP4 CPU-authority contributions, GPU0 rank
reduction, and final residual with resident device handoffs. H6b must still
replace the authority uploads with real concurrent CPU/GPU0/GPU1 expert work
and wire the opt-in transaction before H6 is complete. The H6a shell lifecycle
is hardened with fixed owned host staging, terminal-drained boundary/output
D2H, poison-and-retain behavior after an unproven fallback drain, and five
fault-and-immediate-retry cases in the retained v3 record.
