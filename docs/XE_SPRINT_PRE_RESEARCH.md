# Iris Xe One-Sweep Research Sprint Intake

- Status: pre-research inventory complete; one-sweep research goal proposed
- Created: 2026-08-11
- Repository baseline: `main` at
  `29537feb59d3e3526f2ebd9f4a186a43e8bd977f`
- Research host: T14, Tiger Lake-LP GT2 Iris Xe (`8086:9a49`)
- Current system baseline: Ubuntu 26.04, kernel `7.0.0-29-generic`, `i915`
- External corpus: `/home/emmy/src/xe-research`
- Production implementation authorization: none

This is the intake and execution charter for a bounded Iris Xe research sprint.
It converts the open R0-R7 sequence in
[`XE_RESEARCH_AND_PREPLANNING.md`](XE_RESEARCH_AND_PREPLANNING.md) into one
goal with explicit inputs, gates, artifacts, stopping rules, and a terminal
decision. The intended sweep may complete source research and forced hardware
experiments without repeatedly returning for phase approval. It does not
authorize a production backend, automatic dispatch, model-scale layout
conversion, or serving integration.

## Goal

In one continuous research sweep, determine whether this Tiger Lake Iris Xe
justifies a narrow experimental GPT-OSS prefill backend and, if so, produce a
decision-complete implementation pre-plan. The sweep must select and support
one of these outcomes:

1. OpenCL host API with online compilation and a versioned binary cache;
2. Level Zero host API with reproducibly generated SPIR-V;
3. a deliberately staged OpenCL/Level Zero path justified by measurements;
4. no Xe backend because correctness, memory behavior, performance, tooling,
   or maintenance cost fails the recorded gates.

The goal is complete only when the source map, environment manifest,
capability/failure captures, compiler/cache comparison, memory experiments,
exact MXFP4 fixture, bounded matrix comparison, and final decision are all
recorded, or when a named stop condition makes later stages unjustified.

## What is already complete

- The Iris Xe is visible through OpenCL and Level Zero and remains bound to
  `i915` with render-node access.
- OpenCL online compilation, kernel execution, program-binary retrieval, and
  same-generation binary reload pass.
- Level Zero context, command-list, event, synchronization, shared-allocation
  copy, SPIR-V module, and kernel execution probes pass.
- Clang/LLVM-SPIR-V and Intel `ocloc` both produce modules accepted by the
  device; the older generated modules survived the distribution upgrade.
- A clean post-upgrade rebuild against Level Zero 1.28.2 headers and the system
  loader passes. The fresh capture reports API 1.14 and SPIR-V 1.5.
- The CPU crate already supplies the scalar numerical oracle, K=32 MXFP4/Q8
  block types, E2M1/E8M0 semantics, matrix contracts, and bounded benchmark
  shapes needed by the Xe fixture.
- A local GPT-OSS 20B snapshot occupies about 13 GiB under
  `/data/models/gpt-oss`; model-scale execution is not required until the
  bounded operator evidence justifies it.

## Local source and tool inventory

The external corpus is approximately 2.4 GiB. All sixteen primary source
checkouts were inspected at intake and are clean:

| Source family | Local directories | Available use |
| --- | --- | --- |
| Host APIs | `level-zero`, `cl3`, `opencl3`, `oneapi-rs` | ABI, ownership, errors, queues, events, allocation, and Rust API comparison |
| Specifications | `level-zero-spec`, `OpenCL-Headers`, `OpenCL-Docs`, `OpenCL-Guide` | normative lifecycle, capability, memory, compilation, and SPIR-V contracts |
| Loaders/runtime | `OpenCL-ICD-Loader`, `compute-runtime` | loader discovery and Intel OpenCL/Level Zero frontend behavior |
| Samples | `compute-samples` | small official lifecycle and capability examples |
| Compiler boundary | `intel-graphics-compiler`, `SPIRV-Headers`, `SPIRV-Tools`, `SPIRV-LLVM-Translator` | module production, validation, disassembly, and Intel code-generation path |
| Optional Rust kernel study | `rust-gpu` | bounded Rust-to-SPIR-V feasibility only if the required path leaves time |

The corpus also contains the old exact-match 23.43 runtime/1.16.1 loader
worktrees, the extracted 1.28.2 development headers, cached Ubuntu packages,
cached Cargo dependencies, Clang 18, `llvm-spirv-18`, `ocloc`, SPIR-V tools,
CMake, Ninja, and the rebuilt post-upgrade probes. Approximately 37 GiB of
root storage remained free at intake.

### Source gap to close first

The installed 26.05 Compute Runtime source and Level Zero 1.28.2 loader source
are not yet retained as exact source worktrees. The current `level-zero`
checkout is newer 1.32 source, while the exact retained runtime-matched trees
represent the former 23.43/1.16.1 stack. Before using driver internals as
evidence, the sweep must acquire or reconstruct exact 26.05 and 1.28.2 source
pins, record package-to-source correspondence, licenses, and clean revisions,
and preserve the old worktrees rather than replacing them.

## Planned trajectory

The sprint starts API-neutral but experiment-practical:

- use OpenCL first to establish kernel semantics because online OpenCL C
  compilation and full build logs already work;
- preserve a Level Zero SPIR-V version of the same kernels for lifecycle,
  memory, and submission comparison;
- keep both paths only through the point where same-input measurements can
  decide whether Level Zero adds value;
- target bounded multi-token prefill first; do not promise single-token decode;
- keep CPU execution as the default and scalar CPU results as the oracle;
- produce a no-backend conclusion if the GPU cannot beat the relevant CPU path
  after compilation, allocation, synchronization, and shared-DDR costs are
  accounted for.

This is a research ordering, not a backend selection.

## One-sweep work packages

### X0 — freeze the evidence environment

- capture package versions, package origins/candidates, kernel, firmware,
  Mesa, `i915`, PCI/render-node identity, ACL/groups, ICD files, and loader
  resolution;
- retain exact 26.05 Compute Runtime and 1.28.2 Level Zero source pins;
- write rollback/rebuild commands without changing the working driver stack;
- define an artifact manifest containing command, source revision, tool and
  driver versions, hashes, host, timestamps, and exit status;
- prove offline reruns for the already cached toolchain.

Exit: one internally consistent current-generation manifest exists and no
probe can silently load a 23.43 library.

### X1 — map the minimal host lifecycle

Trace the same operation through OpenCL and Level Zero specifications, Rust
references, samples, loader, and Intel runtime. Record required calls, object
ownership, thread/asynchronous lifetime, cleanup ordering, error taxonomy, and
the smallest unsafe Rust boundary. Inspect source by pinned path and symbol;
do not design from examples alone.

Exit: a side-by-side lifecycle and failure map supports a minimal host API and
identifies which differences are semantic rather than naming differences.

### X2 — close capability and negative-path evidence

- retain full current `clinfo` and rebuilt `ze_info` captures;
- query every optional feature used by later experiments;
- exercise missing device, invalid source, failed build, corrupt binary,
  invalid SPIR-V, unsupported capability, bad allocation, bad shape, failed
  launch, timeout/cancellation, and cleanup paths where safely reproducible;
- record structured diagnostics and ensure forced selection fails clearly.

Exit: later code relies only on queried capabilities and errors remain
distinguishable at the proposed Rust boundary.

### X3 — compare compiler, module, and cache paths

Use one canonical trivial kernel and identical inputs to compare:

- cold OpenCL online compilation;
- warm same-generation OpenCL program-binary reload;
- Clang/LLVM-SPIR-V loaded through Level Zero;
- `ocloc` SPIR-V loaded through Level Zero.

Record build logs, source/options, validation/disassembly, module hashes, cold
and warm timing distributions, submission and execution timing, and cache
invalidation across device/driver/compiler/source/options/backend format.

Exit: one reproducible kernel-production policy is preferred, or the remaining
alternatives have a precise reason to continue into X4-X6.

### X4 — measure integrated-memory behavior

For named sizes and repetitions, compare ordinary buffers, mapped buffers,
OpenCL SVM/USM mechanisms actually exposed, and Level Zero host/shared/device
allocations. Measure allocation, first touch, copy/map, visibility,
synchronization, warm reuse, and cleanup. Add CPU-only and simultaneous CPU/GPU
bandwidth cases. Do not infer zero-copy from integrated physical memory.

Exit: the report states observed transfer/visibility behavior and identifies a
safe allocation strategy for the numerical fixture.

### X5 — prove exact MXFP4 block semantics

Implement a forced research fixture for one K=32 block:

```text
packed MXFP4 nibbles + E8M0 scale + Q8 activation
  -> exact doubled-E2M1 integer dot -> FP32 scale application
```

Compare bit/exact integer intermediates and final tolerance policy with the
repository scalar oracle. Cover nibble order, every E2M1 encoding, E8M0
special values, signs/extrema, invalid inputs, and tails or explicit rejection.

Exit: every fixture passes on every API/compiler path still under
consideration. Any unexplained numerical mismatch stops performance work.

### X6 — run one bounded matrix/prefill experiment

Extend only the passing fixture to a small multi-row matrix shape using compact
weights and caller-owned buffers. Compare CPU scalar, current optimized CPU
matrix execution, and surviving Xe paths. Separate preparation, cold, warm,
submission, synchronization, and end-to-end time; report repetitions and
variance. Include shared-DDR contention and retain negative results.

Exit: evidence shows whether a narrow prefill operation is useful enough to
justify maintenance. No model-wide integration is needed to answer this gate.

### X7 — close the decision and pre-plan

Publish:

- selected API/compiler/cache path or a no-backend decision;
- exact device/driver support boundary;
- minimal Rust types, raw FFI/dependency recommendation, and unsafe boundary;
- kernel source and generated-artifact policy;
- allocation, weight layout, scratch, fallback, cleanup, and error policy;
- forced-only implementation milestones and correctness/performance gates;
- explicit rejected alternatives and upstream contribution candidates;
- a list of evidence still required before automatic selection or model-scale
  serving claims.

Exit: another implementation goal could execute without reopening architectural
questions, or the lane is closed with a defensible negative result.

## Stop conditions

The sweep stops later hardware stages and records a negative/blocked outcome if:

- exact current-generation source/runtime correspondence cannot be established;
- device access becomes unreliable or requires an unaudited driver change;
- a required capability is absent and no bounded fallback preserves the goal;
- X5 cannot match the scalar oracle;
- memory behavior requires a model-scale expanded cache or unsafe coherence
  assumptions;
- X6 shows no plausible benefit over the relevant optimized CPU operation once
  full costs are counted;
- completion would require Vulkan, DPC++ at runtime, a framework-scale
  abstraction, or broad production integration outside this charter.

A stop is a completed research result when its evidence and boundary are
recorded. It is not permission to weaken a gate.

## Explicit exclusions

- production backend implementation or automatic device selection;
- full 20B inference, API serving, decode offload, KV-cache migration, or
  heterogeneous scheduling;
- Vulkan, SYCL/DPC++ runtime adoption, Arc/discrete-Xe generalization, Windows,
  macOS, or non-Intel GPU support;
- driver/package replacement unless separately approved after an audit;
- performance claims based only on kernel timestamps or warm caches;
- upstream patches before a repository-relevant, reproducible issue exists.

## Deliverable layout

The sweep should create a tracked `docs/xe-research/` corpus with:

```text
README.md
00-environment-and-manifest.md
01-host-lifecycle-and-rust-boundary.md
02-capabilities-and-negative-paths.md
03-compiler-module-cache.md
04-integrated-memory.md
05-mxfp4-exactness.md
06-bounded-prefill.md
07-decision-and-implementation-preplan.md
```

Large logs, binaries, generated modules, traces, and benchmark raw data remain
outside Git under `/home/emmy/src/xe-research/results/<run-id>/`, referenced by
manifest and SHA-256. Small source fixtures and scripts may be committed when
their license/provenance and role are explicit.

## Pre-research readiness verdict

The conceptual scope, host, oracle, initial probes, source families, evidence
order, stop conditions, and deliverables are now defined. The local corpus is
sufficient to begin the sweep. X0 must first close the two current-generation
source pins, environment/rollback snapshot, artifact manifest, and
mixed-library guard. Those are first work inside the one-sweep goal rather than
reasons to split the sprint into another approval cycle.
