# CPU Runtime Next-Phase Research

- Status: research complete; implementation planning is not authorized
- Started: 2026-08-11
- Repository content baseline: `main` at
  `a090bb0e81457e4302deb36d6e52a0847c14bfb0`
- Recoverable intake checkpoint: `9df99d24891aa95c2bd9aa39bab9d5a1fa4b1555`
- Active branch: `agent/cpu-next-phase-research`
- Scope: documentation-only CPU runtime research after M1-M5

This is the durable entry point for the bounded research program authorized by
the next-phase intake ledger. The independently reviewable syntheses, shared
source registry, evidence rules, and cross-track closeout are in
[`cpu-runtime-next-phase-research/`](cpu-runtime-next-phase-research/README.md).

The research does not change a Rust interface, runtime policy, kernel, API,
test, workflow, or configuration. Rust-like interfaces and JSON examples in
the corpus are candidate contracts only. They are not implementation plans or
authorization to begin implementation planning.

## Evidence-gate status

The owner results directory was inspected on 2026-08-11. It contains the
earlier M1-M5 and final-smoke captures, but no newly supplied benchmark/oracle
corpus with recoverable commands, host, repetitions, and raw-output hashes.
Consequently:

- E1, C1, C2, C5, C6, and C7 can reach their evidence-appropriate outcomes;
- C3 records the existing numerical boundary and a precise future diagnostic,
  but cannot localize or certify it;
- C4 records separate candidate descriptors and correctness strategies, but
  cannot rank implementation work, select automatic thresholds, or make
  performance claims;
- old captures remain historical/advisory unless their complete provenance is
  recoverable under the E1 manifest.

No fresh 20B execution, benchmark campaign, oracle capture, tuning, or
Tiger Lake/Xe inspection was performed.

## Corpus map

- [`00-evidence-and-observability.md`](cpu-runtime-next-phase-research/00-evidence-and-observability.md)
  defines production metrics, diagnostic traces, offline manifests, source
  roles, privacy rules, and negative-result states.
- [`01-service-lifecycle-api.md`](cpu-runtime-next-phase-research/01-service-lifecycle-api.md)
  covers request ownership, commit versus delivery, cancellation, readiness,
  failure, shutdown, and route semantics.
- [`02-memory-reservations.md`](cpu-runtime-next-phase-research/02-memory-reservations.md)
  inventories CPU memory and defines grant/expand/refund/release semantics.
- [`03-numerical-trust.md`](cpu-runtime-next-phase-research/03-numerical-trust.md)
  separates BF16 localization from configuration-specific trusted evidence.
- [`04a-moe-orchestration.md`](cpu-runtime-next-phase-research/04a-moe-orchestration.md),
  [`04b-dense-bf16.md`](cpu-runtime-next-phase-research/04b-dense-bf16.md), and
  [`04c-attention.md`](cpu-runtime-next-phase-research/04c-attention.md) keep the
  three operator questions independently stoppable.
- [`05-amx-hardware.md`](cpu-runtime-next-phase-research/05-amx-hardware.md)
  defines the real-hardware bring-up and evidence matrix.
- [`06-long-horizon-seams.md`](cpu-runtime-next-phase-research/06-long-horizon-seams.md)
  stress-tests current seams without starting feature research.
- [`07-maintenance-audit.md`](cpu-runtime-next-phase-research/07-maintenance-audit.md)
  records bounded cleanup candidates without fixing them.
- [`08-cross-track-closeout.md`](cpu-runtime-next-phase-research/08-cross-track-closeout.md)
  resolves dependencies, records planning-readiness gates, and attests the
  documentation-only scope.

## Boundary

The active implementation posture remains unchanged: CPU execution is
experimental; trusted CPU serving remains rejected; new optimized paths do not
gain automatic selection; contiguous sequence-local KV remains the current
implementation; and the external T14 lane remains separate.
