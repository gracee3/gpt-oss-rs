# CPU Runtime Feature-Set Integration Plan

- Integration branch: `agent/cpu-runtime-feature-set`
- Baseline: `3600fa4` (`Promote CPU-first AVX2 x8 MXFP4 runtime`)
- Workstream order: M1, M3, M2, M4, M5
- Publication target: verified fast-forward integration into `origin/main`

This directory is the durable execution record for the experimental CPU
feature set researched in [`../cpu-runtime-research/`](../cpu-runtime-research/README.md).
The research documents establish the architectural evidence; these plans
define repository interfaces, commit slices, gates, and closeout evidence.

## Program invariants

- Keep one integration branch for the complete program and never force-push it.
- Re-read the relevant research and milestone plan before each workstream.
- Refine a plan in its own commit before implementation when repository facts
  require a changed interface or sequence.
- Push each coherent compiling checkpoint. Each workstream ends with a
  documentation and evidence closeout before the next begins.
- Preserve canonical checkpoints. Persistent CPU repacks remain versioned,
  atomic, and outside Git; new transient panels use caller-owned scratch.
- Keep `auto` on the established path unless a milestone explicitly says
  otherwise. AVX-512 x8, optimized matrix paths, multi-sequence CPU scheduling,
  and AMX are experimental selections during this program.
- Preserve CUDA engine behavior. CPU server startup moves to the unified
  scheduler only in M4.
- Do not add model files, source clones, repack caches, targets, traces, or
  captured results to Git. Use `/data/models/openai/gpt-oss-rs-cpu-work`.

## Shared interfaces and dependency flow

```text
M1: x8 layout + AVX-512/VNNI GEMV
  |
M3: CpuModel + sequence state + CpuStepBatch + transactions
  |
M2: Mxfp4MatmulProblem + matrix backends + layer-major execution
  |
M4: SequenceTable + reserve/execute/commit scheduler + server
  |
M5: optional AMX-INT8 backend over the matrix contract
```

`CpuStepBatch` is introduced by M3 with row identity, token, absolute position,
and `logits_required`. M2 gives its multi-row execution true layer-major
semantics. M4 constructs those batches from scheduler reservations. M5 plugs
into M2's matrix backend contract and does not create a second model path.

## Commit and publication protocol

The expected minimum slices are:

1. research and relaxed development gates;
2. these program and milestone plans;
3. any pre-workstream plan reconciliation;
4. one or more compiling implementation checkpoints per milestone;
5. one closeout per milestone with docs and evidence;
6. final integration evidence and deferred-certification ledger.

After each push, inspect the ordinary CPU workflow and fix affected failures
forward. At a workstream boundary, fetch `origin/main`; if it advanced, merge it
into the integration branch, rerun the fast gates, and push the merge. Do not
rebase or force-push the published branch.

After all milestones, review the complete diff and run the final gate below.
When the integration branch and its CPU workflow are green, fast-forward local
`main` to it and push `main` directly. Verify the remote commit and main CPU
workflow. Delete the local and remote integration branch only after that
verification. Fix post-push environmental defects forward through the
integration branch if needed.

## Per-workstream fast gate

- `cargo fmt --all --check`
- locked checks and focused tests for affected packages
- warnings-denied Clippy for `gpt-oss-cpu-kernels`, including tests, whenever
  kernel code changes
- focused synthetic/tiny fixtures for model, scheduler, and server behavior
- one short targeted 20B comparison only where the changed path reaches the
  full model

Every milestone plan records exact commands and evidence as implementation
lands. Failures in semantics, bounds, cache integrity, cancellation, state
atomicity, or API behavior block closeout. Benchmark thresholds do not.

## Final integration gate

Run:

```text
cargo fmt --all --check
cargo check --workspace --locked
cargo test -p gpt-oss-cpu-kernels --locked
cargo test -p gpt-oss-model-runner --locked
cargo test -p gpt-oss-engine --locked
cargo test -p gpt-oss-server --locked
cargo clippy -p gpt-oss-cpu-kernels --all-targets --locked -- -D warnings
cargo check -p gpt-oss-server --features amx-int8 --locked
cargo test -p gpt-oss-cpu-kernels --features amx-int8 --locked
```

Then perform only this short 20B smoke campaign:

- default CPU-first startup and one non-streaming request;
- forced AVX-512 x8 with explicit AVX2 matrix backend;
- two concurrent short requests, one streaming and one non-streaming;
- require completion, finite outputs, request isolation, and first-token
  agreement between default and forced paths.

Record commands, host capabilities, artifact locations, commits, and outcomes
in this directory. Do not run Criterion, timing thresholds, long generations,
the 28-run oracle matrix, fresh llama.cpp captures, exhaustive API
permutations, or AMX hardware tests during this feature program.

## Milestone index

- [M1 — AVX-512/VNNI x8 GEMV](M1_AVX512_VNNI_X8.md)
- [M2 — Matrix contract and layer-major prefill](M2_MATRIX_PREFILL.md)
- [M3 — Immutable model and sequence state](M3_MODEL_SEQUENCE_STATE.md)
- [M4 — CPU batching and scheduling](M4_CPU_SCHEDULING.md)
- [M5 — AMX-INT8 prototype](M5_AMX_INT8.md)
- [Final integration record](FINAL_INTEGRATION.md)

## Completion record

- Research checkpoint: `7aaa26f`
- Plan checkpoint: `67e0d4c`
- M1 closeout: `849785e`, `eb92640`, and `250b1c6`; focused gates and short
  20B comparison passed
- M3 closeout: `c9ca550`, `4c63de1`, `94a01a6`, `2247df9`, and the subsequent
  documentation/evidence checkpoint `fa72852`; model/engine gates and short
  20B smoke passed
- M2 closeout: `a8a1e12`, `6c26e0f`, `85f5ab2`, `fa1a733`, and this
  documentation/evidence checkpoint; kernel/model/configuration gates and
  short automatic/explicit-AVX2 20B comparison passed
- M4 closeout: `671dd54`, `aa301f5`, `d58c335`, `818bff6`, `f1cce94`, and the
  documentation/evidence checkpoint; scheduler/model/server gates and a
  concurrent streaming/non-streaming 20B smoke passed
- M5 closeout: `59b6d99`, `6c8cba4`, `9dc1d2e`, `9bba075`, and the
  documentation/CI checkpoint; portable packing, emulation, diagnostics,
  feature compilation, and warnings-denied Clippy passed, while AMX-hardware
  execution remains deferred
- Final integration evidence: [`FINAL_INTEGRATION.md`](FINAL_INTEGRATION.md);
  complete local gate, short 20B smoke, and branch CPU workflow passed
- `origin/main` verification: pending
