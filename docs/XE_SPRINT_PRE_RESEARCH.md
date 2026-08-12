# Iris Xe One-Sweep Research Sprint Intake

- Status: expanded pre-research charter complete; ready for one-sweep research
- Created: 2026-08-11
- Original intake input baseline: `main` at
  `29537feb59d3e3526f2ebd9f4a186a43e8bd977f`
- Expansion-handoff baseline: `main` at
  `f04949c78d69b0a4f8b9aeb50d40f5d9c87a8a49`
- Research host: T14, Tiger Lake-LP GT2 Iris Xe (`8086:9a49`)
- Current system baseline: Ubuntu 26.04, kernel `7.0.0-29-generic`, `i915`
- External corpus: `/home/emmy/src/xe-research`
- Production implementation authorization: none

This is the intake and execution charter for a bounded Iris Xe research sprint.
It converts the open R0-R7 sequence in
[`XE_RESEARCH_AND_PREPLANNING.md`](XE_RESEARCH_AND_PREPLANNING.md) into one
goal with explicit inputs, gates, artifacts, stopping rules, and a terminal
decision. It incorporates the owner-supplied expansion in
[`XE_SPRINT_PRE_RESEARCH_EXPANSION_HANDOFF.md`](XE_SPRINT_PRE_RESEARCH_EXPANSION_HANDOFF.md).
The intended sweep may complete source research and forced hardware experiments
without repeatedly returning for phase approval. It does not authorize a
production backend, automatic dispatch, model-scale layout conversion, or
serving integration.

## Goal

In one continuous research sweep, determine whether this Tiger Lake Iris Xe
justifies one forced experimental GPT-OSS MXFP4 prefill operator and, if so,
produce a decision-complete implementation pre-plan. X7 must decide each axis
independently rather than treating an API and artifact format as a bundle:

| Decision axis | Terminal choices |
| --- | --- |
| Host API | OpenCL, Level Zero, measured staged use of both, or none |
| Kernel delivery | online OpenCL C, reproducible SPIR-V, native binary cache, measured combination, or none |
| Memory/residency | OpenCL buffer/host mechanisms or Level Zero host/shared/device allocation, with an explicit checkpoint, weight, and scratch policy |
| Submission | queue/list/event construction, reuse, synchronization, completion, timeout, and shutdown policy |
| Integration boundary | one forced MXFP4 real-tensor prefill operator or no backend |
| Fallback/commit | exact pre-commit CPU recomputation boundary and externally committed-state rule |

A no-backend result is the terminal choice when correctness, memory behavior,
performance, tooling, dependency cost, or maintenance fails a recorded gate.

The goal is complete only when the source map, environment manifest,
capability/failure captures, compiler/cache comparison, memory and checkpoint
experiments, exact MXFP4 and Xe-LP code-generation evidence, one real-tensor
vertical slice, and final decisions are all recorded, or when a named stop
condition makes later stages unjustified.

## Architectural boundary

OpenCL and Level Zero are host APIs; neither loads a GPT-OSS model. They supply
driver/device discovery, contexts, allocations, programs/modules, kernels,
queues or command lists, events, submission, and synchronization. This
repository remains responsible for SafeTensors lookup and validation, tensor
representation, any derived GPU layout, residency, scratch, operator
inputs/outputs, fallback, and model/runtime ownership.

The research must follow and document this vertical path:

```text
SafeTensors mmap
  -> tensor metadata and validation
  -> canonical compact or derived GPU representation
  -> allocation/residency policy
  -> versioned kernel ABI and arguments
  -> submission and completion
  -> validated output
  -> transactional runtime/model-state commit
```

Official Level Zero and OpenCL sources control the host lifecycle. Mature LLM
backends are references for model/tensor ownership, derived layouts,
residency, scratch reuse, and partial offload. The sweep must not port a loader,
driver, or framework-scale backend into this repository.

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
  `/data/models/gpt-oss`; X4 must ingest at least one real SafeTensors tensor,
  but full-model execution remains excluded.
- Exact clean source worktrees now match Compute Runtime tag `26.05.37020.3`
  and Level Zero tag `v1.28.2` while the earlier 23.43/1.16.1 trees remain
  preserved.

## Local source and tool inventory

The external corpus is approximately 4.2 GiB. All primary source checkouts and
the current-generation matched worktrees were inspected at intake and are
clean:

| Source family | Local directories | Available use |
| --- | --- | --- |
| Host APIs | `level-zero`, `cl3`, `opencl3`, `oneapi-rs` | ABI, ownership, errors, queues, events, allocation, and Rust API comparison |
| Specifications | `level-zero-spec`, `OpenCL-Headers`, `OpenCL-Docs`, `OpenCL-Guide` | normative lifecycle, capability, memory, compilation, and SPIR-V contracts |
| Loaders/runtime | `OpenCL-ICD-Loader`, `compute-runtime` | loader discovery and Intel OpenCL/Level Zero frontend behavior |
| Samples | `compute-samples` | small official lifecycle and capability examples |
| Compiler boundary | `intel-graphics-compiler`, `SPIRV-Headers`, `SPIRV-Tools`, `SPIRV-LLVM-Translator` | module production, validation, disassembly, and Intel code-generation path |
| Optional Rust kernel study | `rust-gpu` | bounded Rust-to-SPIR-V feasibility only if the required path leaves time |
| LLM backend references | `llama.cpp` at `0b1bad14ff204627636aeb1de22ddcd5acb859d4` | OpenCL and reference-only SYCL tensor buffers, residency, caches, MoE, and failure history |
| Level Zero tests | `level-zero-tests` at `d373228d721184255597790310c3d13e8216a43d` | negative paths, list/event reuse, selection, memory, and performance-harness patterns |
| Timing methodology | `pti-gpu` at `c71e8316e19bb5316157b9046d877b5eff0e262c` | Level Zero timestamp, correlation, tracing, and metrics methodology only unless later justified |

The corpus also contains exact clean worktrees for Compute Runtime
`26.05.37020.3` at `a5e0dd79db5ff7b3ed6c5cd3d11064ab7cbb9aa5` and Level Zero
`v1.28.2` at `6369d8d642e9c7625e67f38664267f171b8e42dc`, the old exact-match
23.43/1.16.1 trees, extracted 1.28.2 development headers, cached Ubuntu
packages and Cargo dependencies, Clang 18, `llvm-spirv-18`, `ocloc`, SPIR-V
tools, CMake, Ninja, and the rebuilt post-upgrade probes. Exact Ubuntu source
archives and extracted package trees are also cached for
`intel-compute-runtime` 26.05.37020.3-1, `level-zero` 1.28.2-2, and
`intel-graphics-compiler` 1.0.17791.18+1-3, including Debian patch/build
metadata. Approximately 35 GiB of root storage remained free after intake.

### Source-intake boundary

The focused pre-research source intake is complete. During research, an
installed binary/header/ABI fact is distinct from exact driver-source
correspondence. The Ubuntu source packages were acquired through a signed APT
index and their declared file hashes passed extraction, although `dpkg-source`
could not authenticate the inline maintainer signature with the locally
installed keyrings; X0 must preserve that distinction. A missing or disputed
correspondence blocks claims about
specific driver internals, but does not erase specification-grounded or
black-box hardware evidence when installed binaries, headers, package
provenance, and resolved loaders are exact and unmixed. Any additional source
requires a named question, exact revision, license, studied paths/symbols, and
a role as design reference, adoption candidate, or negative evidence.

## Planned trajectory

The sprint starts API-neutral but experiment-practical:

- use OpenCL first to establish kernel semantics because online OpenCL C
  compilation and full build logs already work;
- preserve a Level Zero SPIR-V version of the same kernels for lifecycle,
  memory, and submission comparison;
- feed the same reproducibly generated SPIR-V bytes through OpenCL IL and
  Level Zero before attributing differences to the host API;
- keep both paths only through the point where same-input measurements can
  decide whether Level Zero adds value;
- target bounded multi-token prefill first; do not promise single-token decode;
- keep CPU execution as the default and scalar CPU results as the oracle;
- produce a no-backend conclusion if the GPU cannot beat the relevant CPU path
  after compilation, allocation, synchronization, and shared-DDR costs are
  accounted for.

This is a research ordering, not a backend selection.

## One-sweep work packages

### X0 — evidence environment, provenance, and dependency budget

- capture package versions, package origins/candidates, kernel, firmware,
  Mesa, `i915`, PCI/render-node identity, ACL/groups, ICD files, and loader
  resolution;
- retain exact 26.05 Compute Runtime and 1.28.2 Level Zero source pins;
- write rollback/rebuild commands without changing the working driver stack;
- define an artifact manifest containing command, source revision, tool and
  driver versions, hashes, host, timestamps, and exit status;
- prove offline reruns for the already cached toolchain;
- distinguish exact installed binary/header/ABI provenance from source-tag
  correspondence and guard every run against mixed-generation libraries;
- record the minimum loader symbols/API versions and compare an established
  small Rust wrapper with checked-in minimal raw declarations, including
  ordinary linking and runtime symbol loading where relevant;
- budget system-runtime, contributor-toolchain, Cargo-runtime, and checked-in
  artifact dependencies separately. "No new Cargo runtime dependency" must
  not be reported as "no external runtime dependency";
- record license/provenance obligations for adopted concepts, selectively
  adopted code, kernel sources, and generated artifacts.

Exit: one internally consistent current-generation manifest exists and no
probe can silently load a 23.43 library. Claims about driver internals require
stronger exact-source correspondence than specification or black-box claims.

### X1 — host, model-resource, and execution lifecycles

Trace the same operation through OpenCL and Level Zero specifications, Rust
references, samples, loader, and Intel runtime. Connect two ownership maps:

```text
XeApi -> selected driver/device -> context -> module/artifact cache
      -> queue/list/event pools

XeModelResources -> persistent weight allocations or reusable residency slab

XeExecution -> activation/output scratch -> queue/list lease
            -> completion token -> validated result
```

Resolve or explicitly defer multi-driver/device selection, process-versus-model
context ownership, queue serialization/thread safety, fixed pools versus
per-launch construction, in-flight resource lifetimes, synchronous/asynchronous
Rust ownership, partial initialization, explicit shutdown versus `Drop`,
timeout/cancellation, device loss/context invalidation, and pre-commit CPU
recomputation. Inspect source by pinned path and symbol and identify the
smallest auditable unsafe boundary.

Exit: object/lifetime and failure maps support a minimal host API and model
attachment without competing ownership or untracked in-flight resources.

### X2 — capabilities, selection, diagnostics, and instrumentation

- retain full current `clinfo` and rebuilt `ze_info` captures;
- query every optional feature used by later experiments;
- exercise missing device, invalid source, failed build, corrupt binary,
  invalid SPIR-V, unsupported capability, bad allocation, bad shape, failed
  launch, timeout/cancellation, and cleanup paths where safely reproducible;
- record OpenCL IL/SPIR-V ingestion, native-binary extraction/reload,
  subgroup, integer-dot/DP4A, timestamp width/resolution/valid bits, clock
  domain, and host/device-correlation capabilities;
- distinguish unsupported capability, wrong device, invalid ABI/artifact,
  allocation failure, launch failure, timeout, and device loss;
- test deterministic selection with multiple Intel drivers/devices and record
  whether the active display/compositor changes the controlled environment;
- record structured diagnostics and ensure forced selection fails clearly.

Exit: later code relies only on queried capabilities and errors remain
distinguishable at the proposed Rust boundary.

### X3 — artifact pipeline, cache, and versioned kernel ABI

Use one canonical trivial kernel and identical inputs to compare:

- OpenCL C online compilation;
- OpenCL cached program-binary reload;
- OpenCL program creation from reproducibly generated SPIR-V IL;
- Level Zero module creation from the exact same SPIR-V bytes;
- Level Zero native-binary extraction and compatible reload;
- `ocloc` only with an explicit output role and reproducibility record.

Record build logs, source/options, validation/disassembly, module hashes, cold
and warm timing distributions, submission/execution timing, compatible reload,
corrupt/stale cache behavior, and invalidation across device, runtime, driver,
compiler, source, options, ABI, and backend format. Treat same-SPIR-V behavior
through both APIs as a hypothesis.

Define a versioned kernel ABI manifest with artifact hashes and provenance,
entry points, every argument's index/size/type/address space/mutability/
alignment/buffer contract, work-group/subgroup assumptions, and cache identity.
Add negative ABI tests: `spirv-val` does not validate Rust/kernel agreement.
Prohibit unexplained optimization passes and fast-math during correctness work.

Exit: one reproducible kernel-production policy is preferred, or the remaining
alternatives have a precise reason to continue into X4-X6.

### X4 — checkpoint ingestion, residency, and integrated memory

For named sizes and repetitions, compare ordinary buffers, mapped buffers,
OpenCL SVM/USM mechanisms actually exposed, and Level Zero host/shared/device
allocations. Measure allocation, first touch, copy/map, visibility,
synchronization, warm reuse, and cleanup. Add CPU-only and simultaneous CPU/GPU
bandwidth cases. Do not infer zero-copy from integrated physical memory.

Add one real SafeTensors tensor-to-GPU slice. Determine whether the read-only
mmap is copied, registered/pinned, mapped, or directly usable; record first
touch, page migration, prefetch/advice where exposed, alignment, maximum and
practical slab sizes, peak RSS, temporary duplication, and coexistence with CPU
fallback. Compare canonical compact MXFP4, the CPU x8 derived layout, and any
narrow GPU-derived layout without creating a general allocator or LRU. Resolve
per-tensor/per-layer/fixed-slab/persistent compact residency plus reusable
activation/output scratch.

Exit: the report states observed transfer/visibility behavior and identifies a
safe narrow checkpoint, weight-residency, and scratch policy. A model-scale
expanded cache or second retained model representation is a negative gate.

### X5 — exact MXFP4 semantics and Xe-LP code generation

Implement a forced research fixture for one K=32 block:

```text
packed MXFP4 nibbles + E8M0 scale + Q8 activation
  -> exact doubled-E2M1 integer dot -> FP32 scale application
```

Compare bit/exact integer intermediates and final tolerance policy with the
repository scalar oracle. Cover nibble order, every E2M1 encoding, E8M0
special values, signs/extrema, invalid inputs, and tails or explicit rejection.
After the minimal exact fixture passes, add randomized differential cases and
study scalar versus vector nibble decode, signed DP4A/integer-dot mapping,
standard versus Intel forms, subgroup widths 8/16/32, work-group sizing,
one-versus-multi-output reuse, vector alignment, local-memory reuse, register
pressure/private memory/spilling, and occupancy. Compare the surviving online
IGC, Clang/LLVM-SPIR-V, and `ocloc` paths and retain native-code or sufficiently
strong compiler/profiling evidence that intended dot-product lowering occurs.

Exit: every fixture passes on every API/compiler path still under
consideration. Any unexplained numerical mismatch stops performance work;
correct SPIR-V alone is not an efficiency result.

### X6 — one real-tensor GPT-OSS prefill vertical slice

Select one real tensor from the local 20B checkpoint using a recorded census
of dense projections, MoE expert shapes, prefill token rows, routed rows per
expert, launch count, and weight/activation reuse opportunities. Extend only
the passing fixture, selected residency/scratch path, and versioned ABI into a
real workload descriptor with caller-owned buffers. Preserve compact weights
unless X4/X5 justify a versioned derived representation.

Compare scalar CPU, the best validated optimized CPU backend, and surviving Xe
paths. Measure preparation/layout conversion, allocation/residency,
module/kernel creation, first dispatch, warm dispatch, reused command/events,
synchronization, result visibility/readback, total operator time, shared-DDR
contention, thermal/power behavior, variance, and break-even token rows.
Validate device timestamps rather than treating them as end-to-end time. If
Level Zero remains plausible, compare a recycled regular command list with an
immediate command list.

Before timing begins, the X6 protocol must register a numeric useful-win gate.
The default floor is 1.25x end-to-end operator speedup over the best validated
CPU path with the confidence interval above parity at a plausible interactive
prefill shape, without model-scale duplicate weights. The protocol may raise,
but not lower, that floor when maintenance and integration costs are known.

Exit: the real-tensor evidence clears the predeclared useful-win gate or closes
the lane with a bounded negative result. Full-model inference is not required.

### X7 — decision-complete forced implementation pre-plan

Publish:

- independent host-API, kernel-delivery, memory/residency, submission,
  integration-boundary, and fallback/commit decisions;
- exact T14 Tiger Lake-LP/driver boundary, without a generic Intel GPU claim;
- proposed Rust modules/types, FFI/loading/dependency strategy, and unsafe
  boundary;
- kernel source, SPIR-V, native-cache, ABI manifest, regeneration, and
  validation policies;
- checkpoint ingestion, weight residency, scratch, event/list reuse,
  synchronization, shutdown, cleanup, and error policies;
- transactional CPU fallback and external-commit behavior;
- forced-only milestones plus correctness and useful-win gates;
- explicit rejected alternatives and upstream contribution candidates;
- a list of evidence still required before automatic selection or model-scale
  serving claims.

Exit: another implementation goal could execute without reopening architectural
questions, or the lane is closed with a defensible negative result.

## Stop conditions

The sweep stops later hardware stages and records a negative/blocked outcome if:

- exact installed binary/header/ABI/package provenance cannot be established
  or resolved libraries cannot be kept unmixed. Missing source correspondence
  blocks driver-internal attribution rather than all black-box research;
- device access becomes unreliable or requires an unaudited driver change;
- a required capability is absent and no bounded fallback preserves the goal;
- X5 cannot match the scalar oracle;
- memory behavior requires a model-scale expanded cache, second retained model
  representation, or unsafe coherence assumptions;
- X6 shows no plausible benefit over the relevant optimized CPU operation once
  full costs are counted;
- completion would require Vulkan, DPC++ at runtime, a framework-scale
  abstraction, or broad production integration outside this charter.

A stop is a completed research result when its evidence and boundary are
recorded. It is not permission to weaken a gate.

## Explicit exclusions

- production backend implementation or automatic device selection;
- full 20B inference, API serving, decode offload, KV-cache migration, or
  heterogeneous scheduling beyond the one real-tensor slice;
- Vulkan, SYCL/DPC++ runtime adoption, Arc/discrete-Xe generalization, Windows,
  macOS, or non-Intel GPU support;
- driver/package replacement unless separately approved after an audit;
- performance claims based only on kernel timestamps or warm caches;
- upstream patches before a repository-relevant, reproducible issue exists.

## Deliverable layout

The sweep should create a tracked `docs/xe-research/` corpus with:

```text
README.md
00-environment-provenance-and-dependencies.md
01-host-model-execution-lifecycles.md
02-capabilities-diagnostics-and-timing.md
03-artifact-pipeline-cache-and-kernel-abi.md
04-checkpoint-ingestion-and-integrated-memory.md
05-mxfp4-exactness-and-xelp-codegen.md
06-real-tensor-prefill-vertical-slice.md
07-decision-and-forced-implementation-preplan.md
```

Large logs, binaries, generated modules, traces, and benchmark raw data remain
outside Git under `/home/emmy/src/xe-research/results/<run-id>/`, referenced by
manifest and SHA-256. Small source fixtures and scripts may be committed when
their license/provenance and role are explicit.

## Pre-research readiness verdict

The expanded scope, host, oracle, initial probes, focused source corpus,
orthogonal decision axes, evidence order, stop conditions, useful-win floor,
and deliverables are defined. Exact 26.05/1.28.2 source worktrees and the three
focused reference repositories are present and clean. The local corpus is
sufficient to begin the one-sweep research goal.

X0 still must produce the current environment/rollback snapshot, dependency
budget, artifact manifest, and mixed-library guard before later experiments.
Those are first research work inside the one-sweep goal, not another intake or
approval cycle. Completion of this charter changes research readiness only; it
does not claim implementation readiness, performance, or a selected backend.
