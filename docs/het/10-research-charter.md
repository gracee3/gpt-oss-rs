# Phase 1 research charter

**Stage:** research; **capture date:** 2026-08-15; **status:** complete and
awaiting review.

## Question

Determine whether exact static heterogeneous GPT-OSS-120B inference is
memory-viable on this host and identify the narrowest architecture family worth
planning. Research may reject the supplied CPU/GPU0/GPU1 expert-ownership
hypothesis. It must not convert a surviving family into an implementation plan.

The provisional proof target from [the Phase 0 charter](00-phase-charter.md)
remains unchanged: one 120B layer must route selected experts across CPU, GPU0,
and GPU1 within one inference operation, preserve exact routing and reduction,
bound weight/scratch/staging memory, fail deterministically, and emit complete
timing. GPT-OSS-20B is the smaller correctness control.

## Authority and exclusions

**Verified:** research-only sources, harnesses, lockfiles, builds, and raw
results were isolated under `~/src`. Within this repository, Phase 1 changes
only `docs/het/` documentation and bounded evidence. The nine supplied
workstation-readiness changes and the Phase 0 documentation state were not
rewritten, staged, or committed.

**Deferred:** production implementation, task/work-package planning, adaptive
placement, migration, prediction, expert deferral, Qwen, Tiger Lake Xe,
generic serving, upstream integration, and protocol repair.

**Verified:** no model payload was downloaded, copied, transformed, or fully
hashed for 120B. The only network retrieval from the official 120B model was
small configuration/index/asset metadata. No Docker image was built. The
official Python/PyTorch oracle was not escalated because the existing Rust CPU
control answered the local retained-continuation question.

## Evidence identity

The research began at repository `HEAD`
`0113e8214e765d168216bbee2120654555a4cfe4` on `main`, tracking
`origin/main` at zero ahead/behind. The nine pre-existing code/lock changes had
the stable diff fingerprint
`792b545405494ca2a5be543b24e29ee0f68420db0f3aa5ec59adf4ea114a374e`
and `35 insertions(+), 13 deletions(-)`. Phase 0 also had `docs/README.md`
modified and `docs/het/` untracked.

The common measurement environment was Ubuntu 26.04, kernel
`7.0.0-29-generic`, Rust/Cargo 1.97.1, CUDA runtime/driver API 13.3,
CUDA toolkit 13.3.73, NVIDIA driver 610.43.02, GCC 15.2.0, and two RTX 3090
devices at compute capability 8.6. Hardware fingerprints are deliberately
sanitized: no hostname, UUID, serial number, or filesystem UUID is retained.

The [bounded evidence index](evidence/research-2026-08/README.md) records exact
harness/result hashes and parameters. Raw high-volume measurements remain in
`~/src/het-research/results`; the checked-in records retain complete checkpoint
maps and compact result distributions sufficient for review.

## Measurement discipline

**Verified:** timed harnesses warmed up explicitly, synchronized CUDA completion,
separated pinned allocation from steady-state copies, retained sample counts and
median/p95 values, and were rerun when surprising. The 20B controls were guarded
by 80 GiB process RSS, 8 GiB minimum `MemAvailable`, and 256 MiB maximum swap
growth. All finished with zero process/system swap and no stop condition.

**Unknown:** the workstation was not an isolated laboratory environment. No
privileged affinity, governor, clock, or security setting was changed. Results
therefore define local distributions and uncertainty, not universal hardware
constants.

## Stop boundary

This record ends with architecture-family comparison and a
`conditionally_ready` recommendation. It intentionally contains no production
API patch, dependency decision, staged migration, work package, or delivery
schedule. Review is the next action.
