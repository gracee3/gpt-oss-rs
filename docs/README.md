# Documentation

## Runtime and backend

- [`CPU_RUNTIME.md`](CPU_RUNTIME.md): native CPU storage, numerical, serving,
  and current dispatch invariants.
- [`MXFP4_CPU_BACKEND_HANDOFF.md`](MXFP4_CPU_BACKEND_HANDOFF.md): forward Intel
  CPU architecture, research questions, benchmarks, and staged kernel plan.
- [`CPU_I7_CONFORMANCE.md`](CPU_I7_CONFORMANCE.md): repeatable full-checkpoint
  CPU regression and comparison procedure.
- [`UPSTREAM_PROVENANCE.md`](UPSTREAM_PROVENANCE.md): pinned upstream audits and
  attribution map.
- [`CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md`](CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md):
  control record for the completed first research pass over the next five CPU
  runtime steps, including evidence standards, decisions, and readiness gates.
- [`cpu-runtime-research/`](cpu-runtime-research/README.md): pinned source
  registry and the detailed AVX-512, GEMM/prefill, state, scheduling, and AMX
  implementation research syntheses.
- [`cpu-runtime-plans/`](cpu-runtime-plans/README.md): durable program and
  milestone implementation plans, validation gates, deviations, and completion
  evidence for the experimental CPU feature set, including its
  [final integration record](cpu-runtime-plans/FINAL_INTEGRATION.md).

## Project direction

- [`PROJECT_INTENT.md`](PROJECT_INTENT.md): educational purpose, current
  hardware envelope, CPU-first scope, role of external references, and the
  decision rule for future targets.
- [`NEXT_MILESTONES.md`](NEXT_MILESTONES.md): current repository milestones.
- [`REPO_ALIGNMENT_AND_WORKSTREAMS.md`](REPO_ALIGNMENT_AND_WORKSTREAMS.md):
  branch, worktree, and integration policy.

## External research and contribution tracking

- [`XE_RESEARCH_AND_PREPLANNING.md`](XE_RESEARCH_AND_PREPLANNING.md): T14
  OpenCL/Level Zero source corpus, host baseline, workstream split, research
  gates, and decision deliverable for a possible Iris Xe backend.
- [`LEVEL_ZERO_AND_ONEAPI_RS.md`](LEVEL_ZERO_AND_ONEAPI_RS.md): Intel Level Zero
  and oneAPI-rs architecture, dependency boundary, Rust design lessons, and
  possible future integrated-GPU experiment.
- [`UPSTREAM_CONTRIBUTION_DISCOVERY.md`](UPSTREAM_CONTRIBUTION_DISCOVERY.md):
  Rust-first upstream opportunity map, qualification gates, and candidate
  ledger.
- [`BORROWED_CONCEPTS.md`](BORROWED_CONCEPTS.md): explicit design-influence and
  conceptual provenance ledger, separate from code/license attribution.

## CUDA validation

- [`TIER2_FP16_CUDA_WORKFLOW.md`](TIER2_FP16_CUDA_WORKFLOW.md): restricted-fp16
  compare, localization, and same-input replay contract.
- [`TIER2_RESULTS_AND_STATUS.md`](TIER2_RESULTS_AND_STATUS.md): retained CUDA
  investigation results.
- `WORKSTREAM_*_TODO.md`: historical lane-specific TODOs. These are not the
  current CPU backend plan.

## Historical records

- [`integration_plan.md`](integration_plan.md): earlier integration history.
- [`cpu-agent-coordination/`](cpu-agent-coordination/): host-specific CPU
  validation logs. These are evidence records, not public API contracts.

When a status document and a runtime/design document disagree, prefer
`CPU_RUNTIME.md`, `MXFP4_CPU_BACKEND_HANDOFF.md`, and the current source tree.
