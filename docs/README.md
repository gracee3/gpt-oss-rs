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
  evidence for the experimental CPU feature set.

## Project direction

- [`NEXT_MILESTONES.md`](NEXT_MILESTONES.md): current repository milestones.
- [`REPO_ALIGNMENT_AND_WORKSTREAMS.md`](REPO_ALIGNMENT_AND_WORKSTREAMS.md):
  branch, worktree, and integration policy.

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
