# Handoff: Expand the Iris Xe Sprint Pre-Research Charter

## Purpose

Revise the existing Iris Xe sprint intake so the next research sweep can answer
not only whether one bounded MXFP4 kernel works, but whether that kernel can be
attached to the real GPT-OSS runtime without reopening fundamental questions
about artifacts, ABI, model ingestion, residency, ownership, submission, and
fallback.

This is a documentation-only pre-research refinement. Do not implement a GPU
backend, change runtime behavior, install or replace drivers, launch a new
benchmark campaign, or begin the X0-X7 research sweep as part of this task.

## Repository starting point

The reported published baseline is:

- repository: `~/gpt-oss-rs`
- branch: `main`
- reported commit: `f04949c`
- primary document: `docs/XE_SPRINT_PRE_RESEARCH.md`
- related documents:
  - `docs/XE_RESEARCH_AND_PREPLANNING.md`
  - `docs/NEXT_MILESTONES.md`
  - the documentation index that links the Xe material

Before editing, verify the actual branch, `HEAD`, worktree status, and remote
relationship. Preserve unrelated user changes. The primary intake currently
contains an older repository-baseline hash in its header; determine whether
that field is intentionally the pre-intake baseline or is now stale, and make
its meaning explicit rather than silently substituting a hash.

## Current assessment

The existing X0-X7 design is a strong feasibility sweep. It already has the
right high-level discipline:

- CPU execution remains the default;
- the scalar CPU implementation is the numerical oracle;
- optimized CPU execution is the performance comparator;
- OpenCL and Level Zero are measured rather than selected in advance;
- a negative/no-backend conclusion is a valid completed result;
- full-model serving, decode, automatic dispatch, Vulkan, and broad GPU support
  are excluded;
- numerical correctness precedes performance work.

Preserve that spine and the X0-X7 numbering. Expand the content inside the work
packages instead of creating another pre-sprint phase.

The current charter can answer:

> Can Iris Xe execute an exact bounded MXFP4 operation, and can it beat the CPU
> for a selected multi-row prefill shape?

The revision must also make the sweep capable of answering:

> Can that operation consume a real GPT-OSS tensor and attach to the existing
> runtime with a narrow, understandable artifact, memory, lifecycle, ownership,
> and fallback design?

## Architectural clarification to add

Do not describe Level Zero as providing model loading. Separate these layers:

1. The host API provides driver/device discovery, contexts, allocations,
   modules, kernels, queues or command lists, events, submission, and
   synchronization.
2. The repository remains responsible for SafeTensors lookup and validation,
   tensor representation, any derived GPU layout, residency, scratch,
   operator inputs/outputs, fallback, and model/runtime ownership.

Capture the intended vertical path explicitly:

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

Study official Level Zero sources for the host lifecycle. Study mature LLM
backends for model and tensor ownership, derived layouts, residency, scratch
reuse, and partial offload. Do not propose porting the Level Zero loader or
driver into the repository. The expected exercise is selective design/code
adoption around the system loader, not importing a framework-scale stack.

## Required changes to the terminal decision

Replace the bundled four-option outcome framing with orthogonal decisions. The
final X7 result must select, justify, or reject each axis independently:

- host API: OpenCL, Level Zero, a measured staged use of both, or none;
- kernel delivery: online OpenCL C, reproducible SPIR-V, native binary cache,
  or a measured combination;
- memory/residency: buffer, host, shared, or device allocation plus checkpoint,
  weight, and scratch policy;
- submission: queue/command-list/event construction, reuse, synchronization,
  and completion policy;
- integration boundary: one forced MXFP4 prefill operator or no backend;
- fallback/commit: when CPU recomputation is allowed and when state becomes
  externally committed.

This avoids treating "OpenCL source" and "Level Zero SPIR-V" as inseparable
bundles.

## Required X0-X7 expansion

### X0 — evidence environment, provenance, and dependency budget

Preserve the environment freeze, exact package/source pins, rollback material,
artifact manifest, offline rerun, and mixed-library guard. Add:

- distinguish installed binary/header/ABI provenance from exact driver-source
  correspondence;
- define the minimum loader symbols and API versions the experiment requires;
- record system-runtime, contributor-toolchain, Cargo, and checked-in-artifact
  dependencies separately;
- compare a small established Rust wrapper with checked-in minimal raw
  declarations, using both ordinary linking and runtime symbol loading where
  relevant;
- state that "no new Cargo runtime dependency" does not mean "no external
  runtime dependency";
- record license/provenance obligations for selectively adopted code and
  generated artifacts.

Revise the stop rule concerning missing exact Compute Runtime source. Missing
exact source correspondence should block claims about driver internals, but it
need not invalidate specification-grounded or black-box hardware evidence when
the installed binaries, headers/ABI, package provenance, and resolved loaders
are exact and unmixed. Preserve a stronger gate before attributing behavior to
specific implementation internals.

### X1 — host, model-resource, and execution lifecycles

Expand the current API lifecycle comparison into two connected ownership maps:

```text
XeApi
  -> selected driver/device
  -> context
  -> module/artifact cache
  -> queue/list/event pools

XeModelResources
  -> persistent weight allocations or reusable residency slab

XeExecution
  -> activation/output scratch
  -> command-list or queue lease
  -> completion token
  -> validated result
```

Research outputs must resolve or explicitly defer:

- exact multi-driver/multi-device selection;
- process-owned versus model-owned contexts and queues;
- queue serialization and thread-safety;
- fixed event/list pools versus per-launch creation;
- how in-flight work keeps allocations, modules, kernels, and buffers alive;
- synchronous and asynchronous Rust ownership boundaries;
- partial-initialization cleanup;
- explicit shutdown versus best-effort `Drop`;
- timeout/cancellation semantics;
- device-loss/context-invalidation behavior;
- whether a failed GPU operation may be recomputed on CPU before any
  transactional state commit;
- the smallest auditable unsafe boundary.

The work package should produce object/lifetime and failure maps, not merely a
side-by-side call-name table.

### X2 — capabilities, device selection, diagnostics, and instrumentation

Preserve the existing capability and negative-path work. Add:

- capabilities required for OpenCL IL/SPIR-V ingestion;
- native-binary extraction/reload support;
- subgroup sizes and integer-dot-product/DP4A-related capability paths;
- timestamp width, resolution, valid bits, clock domain, and host/device
  correlation requirements;
- diagnostic distinction among unsupported capability, invalid ABI, invalid
  artifact, failed allocation, launch failure, timeout, and device loss;
- exact-device selection behavior when multiple Intel drivers/devices exist;
- whether the active display or compositor materially changes the controlled
  measurement environment.

### X3 — source-to-SPIR-V-to-native artifact pipeline and kernel ABI

This is a major expansion. Use one canonical kernel and identical logical
inputs to compare all applicable paths:

1. OpenCL C online compilation.
2. OpenCL cached program-binary reload.
3. OpenCL program creation from reproducibly generated SPIR-V IL.
4. Level Zero module creation from the exact same SPIR-V bytes.
5. Level Zero native-binary extraction and compatible reload.
6. `ocloc` only where its precise output role and reproducibility are recorded.

The same-SPIR-V-through-both-APIs experiment is required because the host-API
decision and the kernel-toolchain decision may be independent. Treat identical
behavior as a hypothesis to test, not an assumption.

Define a versioned kernel ABI manifest covering:

- source and generated-artifact hashes;
- source language, flags, tool versions, target environment, and SPIR-V
  version;
- kernel entry-point names;
- argument index, size, scalar/vector type, address space, mutability,
  alignment, and buffer-size contract;
- work-group/subgroup assumptions;
- device, runtime, driver, compiler, source, options, ABI, and backend cache
  identity;
- cold creation, warm creation, compatible reload, corrupt cache, and stale
  cache behavior.

Add negative ABI tests. `spirv-val` validates the module, not the Rust/kernel
argument agreement or the quality of generated Xe code.

During correctness work, prohibit unexplained compiler optimization passes and
fast-math. Any later relaxation must have separate differential evidence.

### X4 — checkpoint ingestion, residency, and integrated-memory behavior

Preserve the bounded allocation/coherence/bandwidth experiments and add a real
checkpoint-ingestion study. Determine:

- how an existing read-only SafeTensors mmap participates: direct use,
  registration/pinning, mapped buffer, or explicit copy;
- observed behavior of OpenCL buffers, `USE_HOST_PTR`, mapped buffers, and
  exposed SVM/USM mechanisms;
- observed behavior of Level Zero host, shared, and device allocations;
- first touch, page migration, prefetch/advice where exposed, visibility, and
  warm reuse;
- alignment, maximum-allocation, and practical slab-size limits;
- whether weights remain in canonical compact MXFP4 form;
- whether the existing CPU x8 layout is reusable, irrelevant, or harmful;
- per-tensor, per-layer, fixed-slab, and persistent compact-allocation options;
- peak RSS and temporary duplication while the 20B checkpoint remains mapped;
- coexistence with CPU fallback without retaining another model-scale
  representation;
- activation/output scratch ownership and reuse;
- CPU/GPU DDR contention and package/thermal interactions.

Do not design a general allocator or LRU. The expected output is one narrow
policy, a small set of measured alternatives, or a no-backend result.

Require at least one real SafeTensors tensor-to-GPU vertical slice before the
research can claim implementation readiness.

### X5 — exact MXFP4 semantics plus Xe-LP code generation

Preserve the K=32 oracle-first fixture and edge coverage. Add randomized
differential cases after the minimal exact fixture passes. Then investigate:

- scalar versus vectorized nibble decode;
- signed DP4A/integer-dot-product mapping and accumulation;
- standard OpenCL versus Intel-specific integer-dot-product forms when
  exposed;
- subgroup widths 8, 16, and 32 where supported;
- work-group sizing relative to reported preferred multiples;
- one-output versus multi-output weight/activation reuse;
- vector load width and alignment;
- local-memory reuse versus direct global loads;
- register pressure, private memory, spilling, and occupancy evidence;
- code-generation differences among online IGC, Clang/LLVM-SPIR-V, and
  `ocloc` paths still under consideration;
- generated native code or sufficiently strong compiler/profiling evidence
  that the intended dot-product lowering occurred.

Correct SPIR-V and correct numerical output are necessary but do not prove an
efficient Xe-LP kernel.

### X6 — one real-tensor GPT-OSS prefill vertical slice

Replace the purely bounded/synthetic framing with a bounded but real vertical
slice:

- select one actual GPT-OSS tensor from the local 20B checkpoint;
- derive the candidate from a recorded census of real dense and MoE prefill
  shapes;
- use a real workload descriptor and caller-owned buffers;
- include the selected residency/scratch path and versioned ABI;
- compare scalar CPU, the best validated optimized CPU backend, and surviving
  Xe paths;
- preserve compact weights unless X4/X5 justify a derived representation.

Before selecting the operator, record:

- dense projection shapes;
- MoE expert shapes;
- tokens per prefill batch and routed tokens per expert;
- expected launch count;
- opportunities to reuse decoded weights or activations;
- whether small expert-local batches undermine GPU utilization.

Measure preparation/layout conversion, allocation/residency, module/kernel
creation, first dispatch, warm dispatch, reused command structures/events,
synchronization, result visibility/readback, total operator time, shared-DDR
contention, thermal/power behavior, variance, and the break-even token-row
count. Validate device timestamps rather than treating them as end-to-end time.

For Level Zero, include a focused comparison of a recycled regular command
list with an immediate command list if both remain plausible.

Predeclare what constitutes a useful win. The threshold should account for
maintenance and integration cost, not merely a statistically detectable
kernel-only improvement.

### X7 — decision-complete forced implementation pre-plan

Publish the orthogonal decisions listed above. The pre-plan must also state:

- exact supported hardware/driver boundary, initially the T14's Tiger Lake-LP
  Iris Xe rather than a generic Intel GPU claim;
- proposed Rust modules/types, FFI/loading strategy, and unsafe boundary;
- kernel source, SPIR-V, native-cache, manifest, regeneration, and validation
  policies;
- checkpoint ingestion, weight residency, scratch, and cleanup policies;
- command/event reuse and synchronization policy;
- transactional fallback behavior;
- correctness gates and a useful-win performance gate;
- forced-only integration milestones;
- rejected alternatives and what evidence could reopen them;
- evidence still required before automatic dispatch, decode offload,
  full-model serving, or broader hardware support.

## Source-corpus additions to consider

Do not broadly collect more repositories. Add only pinned, clean sources that
answer a named question and record the revision, license, exact files/symbols
studied, and whether the result is design reference, code-adoption candidate,
or negative evidence.

Focused candidates:

- current pinned `llama.cpp` OpenCL backend: MXFP4 kernels, program-binary cache
  keys, tensor-buffer ownership, MoE handling, and failure history;
- current pinned `llama.cpp` SYCL backend, reference-only: Intel integrated-GPU
  model loading, buffer residency, and weight-reorder placement; do not adopt
  SYCL/DPC++ runtime dependencies;
- `oneapi-src/level-zero-tests`: negative paths, loader behavior, event/list
  reuse, and focused performance harness patterns;
- Intel PTI Level Zero timing documentation/source, methodology-only unless a
  minimal existing local path justifies more;
- official OpenCL IL-program and Level Zero module/native-cache specification
  material already represented by the source corpus.

Prefer selective adoption of mature, narrow code/design over new runtime
dependencies. Do not copy code until its license, provenance, fit, and local
maintenance advantage are explicit.

## Deliverable-layout revision

Keep `docs/xe-research/00` through `07`, but broaden the descriptions so the
future sweep has explicit homes for the new evidence:

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

Renaming is optional if stable existing names are preferable, but the required
contents are not optional. Keep large logs, binaries, generated modules,
traces, and raw benchmark data outside Git with manifest hashes, as the current
intake specifies.

## Explicit non-goals for this documentation update

Do not:

- implement OpenCL or Level Zero Rust code;
- add Cargo or system dependencies;
- copy third-party source into the repository;
- compile kernels or generate SPIR-V/native binaries;
- run hardware experiments or benchmarks;
- change driver/runtime packages;
- broaden the target beyond the specific Iris Xe research host;
- authorize full-model inference, serving, decode, KV-cache migration,
  heterogeneous scheduling, automatic dispatch, or production integration;
- promise that SPIR-V is portable between the OpenCL and Level Zero paths;
- promise zero-copy merely because the GPU and CPU share physical memory.

## Documentation integration

Make `docs/XE_SPRINT_PRE_RESEARCH.md` the authoritative revised charter. Update
`docs/NEXT_MILESTONES.md`, the docs index, and
`docs/XE_RESEARCH_AND_PREPLANNING.md` only where needed to prevent stale scope,
status, links, or outcome language. Avoid duplicating the full charter across
multiple documents.

Preserve the existing historical distinction between completed preplanning,
the newly expanded research charter, and future implementation authorization.
No document should imply that X0-X7 experiments or a production Xe backend
have already begun.

## Completion criteria

This handoff is complete when:

1. The primary charter contains the architectural clarification and all X0-X7
   expansions above.
2. Host API, artifact delivery, memory/residency, submission, integration, and
   fallback are independent terminal decision axes.
3. OpenCL-from-SPIR-V and Level Zero-from-the-same-SPIR-V are explicit X3
   experiments.
4. Level Zero native-binary extraction/reload and cache invalidation are
   explicit X3 experiments.
5. A versioned kernel ABI and negative ABI tests are required.
6. Real checkpoint ingestion and model-wide residency feasibility are required
   in X4.
7. X5 requires both numerical proof and Xe-LP code-generation evidence.
8. X6 consumes at least one real GPT-OSS tensor and uses a real prefill-shape
   census rather than only a synthetic matrix.
9. Lifecycle ownership covers API, model resources, execution scratch,
   completion tokens, shutdown, device loss, and transactional CPU fallback.
10. The dependency budget distinguishes system, contributor, Cargo, and
    repository-artifact costs.
11. Existing boundaries and the valid no-backend outcome remain intact.
12. Cross-document links and terminology are consistent.

## Verification and publication

Run documentation-focused checks first:

- inspect the final diff for accidental scope expansion;
- verify Markdown links and referenced paths;
- run `git diff --check`;
- run the repository's documented formatting or documentation checks if they
  are available and proportionate;
- do not launch heavyweight tests solely for a Markdown-only edit unless local
  repository policy requires them.

Report:

- files changed;
- the main scope expansions;
- checks run and their results;
- any question intentionally left for X0-X7 research;
- final branch/commit/publication state if publication is authorized by the
  repository's standing workflow.

Do not claim that research evidence, implementation readiness, or performance
has changed merely because the charter became more complete.
