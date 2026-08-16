# Documentation

- [`het/`](het/README.md): active heterogeneous GPT-OSS pre-research baseline,
  runtime map, host/model evidence, readiness matrix, and research backlog.
- [Tiger Lake optimization baseline](TIGER_LAKE_BASELINE.md)
- [Bounded CPU execution profiling](CPU_EXECUTION_PROFILING.md)
- [Tiger Lake MXFP4 matrix candidate](TIGER_LAKE_MXFP4_MATRIX.md)
- [Tiger Lake forced Xe expert residency](TIGER_LAKE_XE_RESIDENCY.md)
- [Tiger Lake candidate-specific closure](TIGER_LAKE_CLOSURE.md)

## Runtime and backend

- [`CPU_RUNTIME.md`](CPU_RUNTIME.md): native CPU storage, numerical, serving,
  and current dispatch invariants.
- [`MXFP4_CPU_BACKEND_HANDOFF.md`](MXFP4_CPU_BACKEND_HANDOFF.md): forward Intel
  CPU architecture, research questions, benchmarks, and staged kernel plan.
- [`CPU_I7_CONFORMANCE.md`](CPU_I7_CONFORMANCE.md): repeatable full-checkpoint
  CPU regression and comparison procedure.
- [`TIGER_LAKE_CPU_CORPUS.md`](TIGER_LAKE_CPU_CORPUS.md): hashed seven-scenario
  operation, timing, and expert-bucket evidence used for matrix promotion.
- [`TIGER_LAKE_MXFP4_MATRIX.md`](TIGER_LAKE_MXFP4_MATRIX.md): genuine 8x8
  AVX-512/VNNI candidate and the evidence-backed negative Tiger Lake Auto
  decision.
- [`TIGER_LAKE_XE_RESIDENCY.md`](TIGER_LAKE_XE_RESIDENCY.md): forced-only
  OpenCL LRU design, live isolated win, and negative full-model capacity result.
- [`CPU_FRESH_ORACLE_CAMPAIGN.md`](CPU_FRESH_ORACLE_CAMPAIGN.md): immutable
  v0.0.9 CPU-oracle image publication, lock, fresh-lineage campaign order, and
  closure rules.
- [`CPU_FRESH_ORACLE_CLOSURE.md`](CPU_FRESH_ORACLE_CLOSURE.md): candidate-A
  historical closure keyed to `af6c0a2` and its own E1 artifact set.
- [`TIGER_LAKE_CLOSURE.md`](TIGER_LAKE_CLOSURE.md): current implementation
  candidate, profiler/corpus evidence, CPU and Xe negative promotion results,
  fresh 42-cell certification, service outcomes, and E1 identity.
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

- [`CPU_RUNTIME_NEXT_PHASE_PRE_RESEARCH_LEDGER.md`](CPU_RUNTIME_NEXT_PHASE_PRE_RESEARCH_LEDGER.md):
  closed CPU-only intake ledger for post-M1-M5 service/lifecycle, evidence,
  memory, numerical, operator-contract, AMX, maintenance, and long-horizon
  seam questions.
- [`CPU_RUNTIME_NEXT_PHASE_RESEARCH.md`](CPU_RUNTIME_NEXT_PHASE_RESEARCH.md):
  completed documentation-only next-phase research entry point, including
  per-track outcomes and the cross-track planning-readiness closeout.
- [`XE_RESEARCH_AND_PREPLANNING.md`](XE_RESEARCH_AND_PREPLANNING.md): T14
  OpenCL/Level Zero source corpus, passing online-build/binary-cache and SPIR-V
  kernel probes, host baseline, workstream split, and decision gates for a
  possible Iris Xe backend.
- [`XE_SPRINT_PRE_RESEARCH.md`](XE_SPRINT_PRE_RESEARCH.md): current source and
  tool inventory plus the authoritative expanded one-sweep X0-X7 research
  charter, preserved with its original performance-gated closeout.
- [`xe-research/`](xe-research/README.md): completed X0-X9 reports, immutable
  research evidence, the explicit CPU+Xe production integration, and the
  evidence-gated decision to leave automatic dispatch on CPU.
- [`xe-research/09-production-integration-and-auto-promotion.md`](xe-research/09-production-integration-and-auto-promotion.md):
  runtime-loaded explicit hybrid serving and its full-model promotion gate.
- [`XE_SPRINT_PRE_RESEARCH_EXPANSION_HANDOFF.md`](XE_SPRINT_PRE_RESEARCH_EXPANSION_HANDOFF.md):
  owner-supplied documentation handoff that expanded the charter from bounded
  kernel feasibility to real runtime attachment research.
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
