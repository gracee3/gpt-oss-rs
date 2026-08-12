# Tiger Lake Iris Xe Research and Pre-Planning

- Status: OpenCL online compilation/program-binary reload and version-matched
  Level Zero offline SPIR-V copy kernels pass; no backend selected or
  implemented
- Started: 2026-08-11
- Research host: T14 with Tiger Lake-LP GT2 Iris Xe (`8086:9a49`)
- External source corpus: `/home/emmy/src/xe-research`
- Active API comparison: OpenCL and Level Zero
- Explicit exclusion: Vulkan is not part of this research phase

This document separates two concurrent project lanes:

- the i7 continues CPU hardening, certification, and tuning of the completed
  CPU feature set;
- the T14 investigates whether its integrated Iris Xe can support a narrow,
  useful, maintainable `gpt-oss-rs` compute path.

The Xe lane is research and pre-planning. This repository work does not add a
dependency, change automatic dispatch, install a driver, introduce a GPU
backend, or reduce the priority of CPU work.

## Decision to produce

The research must select and justify one of these outcomes:

1. OpenCL host API with online OpenCL C kernel compilation;
2. OpenCL host API with a versioned offline program/binary cache;
3. Level Zero host API with reproducibly generated SPIR-V;
4. a staged path that uses OpenCL to establish kernel semantics and Level Zero
   only where a measured control or submission advantage exists;
5. no Iris Xe backend because the toolchain, shared-memory behavior,
   performance, or maintenance burden does not justify it.

Cloning source repositories does not favor an outcome. A final pre-plan must
state the evidence for the chosen path and the reasons the alternatives were
rejected.

## OpenCL and Level Zero: the actual distinction

Both APIs provide host-side GPU control:

| Responsibility | OpenCL | Level Zero |
| --- | --- | --- |
| Runtime discovery | platform and device | driver and device |
| Ownership boundary | context | context |
| Memory | buffers and optional SVM | explicit host/shared/device allocations |
| Executable container | program | module |
| Entry point | kernel | kernel |
| Submission | enqueued operations | recorded/submitted command lists |
| Completion | events | events and fences |

The principal initial difference is kernel production. OpenCL can accept
OpenCL C source and invoke Intel Graphics Compiler through `clBuildProgram`.
Level Zero accepts SPIR-V or a device-native binary through module creation and
does not provide a source language compiler.

Therefore, discovery and memory management do not require OpenCL. OpenCL is
interesting because it supplies a standardized source-to-device compilation
path. Level Zero becomes attractive when an auditable module already exists
and its more explicit lifecycle provides a measured benefit.

The two APIs are frontends to Intel Compute Runtime and much of the same
underlying driver/compiler machinery. An API choice alone does not make the
MXFP4 arithmetic faster.

## T14 environment baseline and transition

The first read-only capture at lane initialization found Tiger Lake-LP GT2 Iris
Xe bound to `i915`, the generic `ocl-icd` loader 2.3.2, zero OpenCL platforms,
no `/etc/OpenCL/vendors`, and no Intel OpenCL/Level Zero package set.

At approximately 18:03 local time, while the source corpus was being cloned,
the user installed the Intel OpenCL and Level Zero packages through Ubuntu
`apt`. The current host has:

| Package | Version |
| --- | --- |
| `intel-opencl-icd` | `23.43.27642.40-1ubuntu3` |
| `libigc1` | `1.0.15468.25-2ubuntu0.1` |
| `libze-intel-gpu1` | `23.43.27642.40-1ubuntu3` |
| `libze1` | `1.16.1-1build1` |
| `ocl-icd-libopencl1` | `2.3.2-1build1` |

The kernel is `7.0.0-28-generic`; the device remains bound to `i915` and the
user has an explicit read/write ACL on `/dev/dri/renderD128`. The Intel ICD now
discovers one `Intel(R) Iris(R) Xe Graphics` device. The runtime packages do not
provide `ze_info` or `zello_world`; version-matched copies have now been built
locally. A cached user-local `ocloc` works, while `sycl-ls` is absent and is not
required for the direct OpenCL/Level Zero research path.

Initial OpenCL capability evidence is encouraging but constrained:

- OpenCL 3.0 NEO driver `23.43.027642`, with OpenCL C 1.2;
- 80 compute units at up to 1300 MHz; work-group limit 512, preferred multiple
  64, and subgroup sizes 8, 16, and 32;
- FP16, Intel DP4A, `cl_khr_integer_dot_product`, `cl_khr_il_program`, Intel
  subgroups, and Intel unified shared memory extensions;
- 13.97 GiB reported global memory, shared with the host, but a 4 GiB maximum
  allocation;
- coarse-grained buffer SVM only, without fine-grained buffer/system SVM or
  SVM atomics;
- 128-byte minimum datatype alignment, 3.75 MiB global cache, and a 64-byte
  cache line.

This immediately prevents two shortcuts: reported global memory is not
dedicated VRAM or permission to make a single model-sized allocation, and
integrated physical memory does not prove zero-copy. FP16 and integer-dot make
a bounded MXFP4 experiment plausible, but there is no native MXFP4 operation.
OpenCL 3.0 makes features beyond the 1.2 baseline optional, so every used
capability still needs an explicit query. The user-installed package set still
needs a kernel compatibility and rollback audit before any refresh.

## Initial OpenCL, Level Zero, and compiler results

The first Level Zero and R3 compiler gates now pass:

- the current Level Zero 1.32 source builds, but its loader returns
  `ZE_RESULT_ERROR_UNSUPPORTED_VERSION` against the installed 23.43 driver;
- the package-matched Level Zero `v1.16.1` source at
  `ac99dbfb937f0715171eb39f83b5fadf20474b68` discovers the Iris Xe, reports
  driver API 1.3, and passes context, immediate-command-list, event, wait, and
  cleanup operations;
- Compute Runtime tag `23.43.27642.40` at
  `54d973fca784cf71ed5e2c59bea6b9445d59547e` passes shared-allocation device
  copy and byte validation;
- Clang 18 plus `llvm-spirv-18` produces validated SPIR-V 1.0, within the
  device's reported SPIR-V 1.2 maximum, that the matched Level Zero path loads
  and executes successfully;
- Intel `ocloc` 23.43 independently produces validated Tiger Lake SPIR-V and
  a native device binary; its SPIR-V also executes successfully;
- `opencl3` at `072410552fecfc1e3f5395856735cb8684501f74` builds offline and
  successfully drives online OpenCL C compilation and an upstream SAXPY
  example on the Iris Xe;
- a research `opencl3` probe retrieves a 3,648-byte Intel program binary,
  rebuilds a program from it, executes it, and validates all 4,096 results;
- a minimally built `ze_info` from compute-samples revision
  `efa767b95de64c4103d3fc17338ec03d63a9387a` records one compute/copy queue,
  64 KiB shared local memory, SPIR-V 1.2, FP16, and read/write plus atomic
  host/device/shared-single-device allocation capabilities.

The successful shared-buffer tests prove usability, not zero-copy or useful
model performance. Current upstream and installed-runtime source generations
must not be mixed casually. Full artifact hashes and offline-cache provenance
are retained in `/home/emmy/src/xe-research/PROBE_RESULTS_2026-08-11.md`.

### Post-distribution-upgrade revalidation

After an Ubuntu distribution upgrade later on 2026-08-11, the installed stack
changed from Compute Runtime 23.43 and Level Zero loader 1.16.1 to Compute
Runtime 26.05 and system loader 1.28.2; the kernel changed from 7.0.0-28 to
7.0.0-29. The core feasibility results still pass:

- OpenCL discovery, online compilation, execution, fresh program-binary
  retrieval, binary reload, and result validation pass;
- shared-allocation copy and validation pass;
- the previously generated Clang and `ocloc` SPIR-V modules both load, execute,
  synchronize, and validate through the upgraded system loader;
- the upgraded driver reports Level Zero API 1.14 and OpenCL SPIR-V support
  through 1.5, while the main device and memory limits used by this research
  remain unchanged.

The system 1.28.2 loader works with both the matched probe and the probe built
from the current Level Zero source headers. The separately built current
Level Zero 1.32 loader still returns `ZE_RESULT_ERROR_UNSUPPORTED_VERSION`, so
"current upstream" remains an invalid substitute for the distribution-matched
loader. The old compute-samples `ze_info` capture is no longer reusable: it
initially returned `ZE_RESULT_ERROR_UNSUPPORTED_VERSION` with the system loader
and aborted inside the upgraded Compute Runtime when forced through the old
1.16.1 loader. These were not driver regressions: adding the entire cached
23.43 sysroot to `LD_LIBRARY_PATH` to find Boost also selected the old Intel
GPU driver. A clean rebuild against extracted 1.28.2 headers and the system
loader, with only cached Boost isolated, passes and produces a fresh JSON
capability capture.

The newly retrieved OpenCL program binary changed from 3,648 to 4,184 bytes,
as expected across a compiler/runtime upgrade. This reinforces the requirement
that cached device programs include the runtime/compiler/device generation in
their cache key. Detailed commands and results are retained in
`/home/emmy/src/xe-research/POST_UPGRADE_VALIDATION_2026-08-11.md`.

## External research corpus

The T14 source corpus is external to Git at `/home/emmy/src/xe-research`.
`README.md` in that directory records the exact-revision manifest, license
notes, baseline, questions, and refresh protocol. The probe-results note
records build inputs, hashes, and outcomes. The initial repositories were
shallow and blob-filtered; compute-samples history was later fetched to select
a compatible revision. All primary checkouts and matched worktrees are clean.

The corpus groups are:

### Host APIs and Rust ownership

- Level Zero loader at `1ca51c950d97f34d9d271615af8d797836fe6974`;
- Level Zero specification at
  `60c28d2051071fdd484a72306c43fa0519a515b0`;
- oneAPI-rs at `1581663fdd0fd73e79df2900a2576d6cca8ff2a1`;
- `cl3` at `9e2cdd8f34f09abfe49a8c2718ac58f1f762ae61`;
- `opencl3` at `072410552fecfc1e3f5395856735cb8684501f74`.

### OpenCL specification and loading

- Khronos OpenCL Headers, ICD Loader, Docs, and Guide;
- Intel Compute Runtime at
  `8719a80795943a24b826c7db25836745ae515e86`;
- Intel compute samples at
  `b18e178ba784e8eff20c2a57314a6df4d9d2f7c1`.

### Compiler and SPIR-V boundary

- Intel Graphics Compiler at
  `6a33231cbe0252baf68b6ee3d4c002ba9129a181`;
- SPIR-V Headers, Tools, and LLVM Translator;
- `rust-gpu` at `7cb31ae4f9fb069f6c539fdcdfd68b8b6e77e85f`.

The corpus is evidence, not a dependency shortlist. Intel LLVM and broad
graphics/framework repositories are intentionally not cloned until a specific
unanswered question requires them.

## Research workstreams

### R0 — source and lifecycle map

Trace the minimal equivalent operation through:

- Khronos/Level Zero specifications;
- `cl3` and `opencl3` Rust ownership/error boundaries;
- oneAPI-rs typed memory and completion ownership;
- Intel compute samples;
- Intel Compute Runtime OpenCL and Level Zero frontends;
- Intel Graphics Compiler module/build path.

Output: an exact API/resource map, required functions, unsafe boundaries,
failure categories, and meaningful semantic differences. Do not design a
wrapper by copying the full upstream surface.

### R1 — environment provenance and compatibility plan

Before changing the user-installed packages, record:

- distribution, kernel, Mesa, `i915`, firmware, PCI ID, and `/dev/dri` state;
- the exact `apt` package selection and intended Xe research purpose;
- distro package candidates versus Intel release packages;
- a single mutually compatible Compute Runtime/IGC/loader version set;
- expected files and ICD registration;
- disk impact, package conflicts, rollback commands, and verification steps.

Decide whether the working distro set should be retained or deliberately
refreshed. Any further package change occurs only after reviewing this plan.
Do not combine distro and upstream release components opportunistically.

### R2 — device and capability inventory

Preserve the initial OpenCL and Level Zero captures and complete the remaining
capability evidence for the same device:

- names, IDs, API/driver versions, compute units, work-group/subgroup limits;
- global/local memory, maximum allocation, alignment, and SVM/shared-memory
  capabilities;
- FP16, integer-dot, subgroup, IL/SPIR-V, and Intel extension strings;
- queue groups, command modes, timestamps, and event support;
- build logs and structured diagnostics for intentionally invalid inputs.

Every optional feature is queried. No Tiger Lake capability is inferred from a
newer Xe/Arc architecture.

### R3 — reproducible trivial kernel paths

For a simple integer buffer operation:

1. build OpenCL C online and retain its full build log;
2. retrieve and reload any driver program binary where supported;
3. produce an offline SPIR-V module, validate/disassemble it, and load it
   through Level Zero;
4. record source, tools, versions, flags, hashes, cache key, and invalidation;
5. compare cold compilation, warm reload, submission, and execution timing.

The OpenCL online-build/retrieved-binary path and the offline
Clang/LLVM-SPIR-V and `ocloc` module paths pass without a runtime DPC++
dependency. The remaining R3 work is structured failure logs, explicit cache
keys/invalidation, and cold-versus-warm timing. A Rust-to-SPIR-V path is a
candidate experiment, not a requirement.

### R4 — integrated-memory behavior

Measure ordinary buffers, mapping, and advertised shared/SVM mechanisms.
Shared physical DDR does not prove zero-copy. Record whether the driver copies,
pins, maps, or requires cache/synchronization transitions, and measure:

- allocation and first-touch cost;
- host-to-device and device-to-host visibility;
- warm repeated access;
- simultaneous CPU/GPU bandwidth contention;
- cleanup, cancellation, and failed-submission behavior.

### R5 — exact MXFP4 block fixture

Implement the smallest numerical proof:

```text
packed MXFP4 nibbles + E8M0 scale + Q8 activation block (K=32)
  -> exact doubled-E2M1 integer dot
  -> FP32 scale application
```

The existing Rust scalar implementation is the oracle. Cover nibble order,
all E2M1 encodings, E8M0 special values, positive/negative extrema, tails or
explicit shape rejection, and invalid arguments. Retain no performance claim
from this correctness fixture.

### R6 — bounded matrix/prefill experiment

Only after R5 passes, implement one multi-row matrix shape behind a forced
research executable or branch. Keep weights compact unless evidence requires a
new versioned layout. Measure:

- one-time program/module and weight preparation;
- warm matrix execution and synchronization;
- end-to-end operation time;
- CPU scalar, current optimized CPU matrix path, OpenCL, and Level Zero where
  both remain viable;
- shared-DDR effects while CPU work is active.

Multi-token prefill is the first candidate because it exposes more independent
work. Single-token decode remains a question, not a promised GPU target.

### R7 — decision and implementation pre-plan

Produce a written recommendation with:

- selected API/compiler/cache path or a no-backend result;
- exact supported host/device boundary;
- minimal Rust types and raw-FFI surface;
- kernel source/artifact ownership and reproducibility policy;
- memory/persistent-layout decision;
- fallback, errors, forced-selection, and cleanup behavior;
- implementation milestones and correctness/performance gates;
- upstream contribution candidates discovered along the way.

No production implementation begins merely because a smoke kernel runs.

## Likely narrow Rust shape

Regardless of API, the useful ownership model is small:

```text
XeRuntime
XeDeviceCaps
XeContext
XeAllocation<Host | Shared | Device>
XeProgramOrModule
XeKernel
XeQueue
XeEvent
```

The raw ABI and asynchronous lifetime rules remain in a narrow unsafe layer.
Missing loader/driver/device/capability/compiler/module/launch conditions must
be distinguishable. CPU stays the default, and a research backend must fail
clearly when forced on an unsupported host.

`cl3`, `opencl3`, and oneAPI-rs are design references and possible upstream
contribution destinations. They are not automatically the dependencies used by
this project.

## Evidence and promotion rules

- Pin every source claim to an exact revision, path, and symbol.
- Preserve compiler/build logs for failures; do not commit bulk traces or
  generated binaries without a source and reproducibility policy.
- State cold versus warm caches and include compilation/allocation/sync costs
  in end-to-end claims.
- Name the host, driver, compiler, kernel, work-group, tensor shape, repetitions,
  and variance for performance results.
- Compare numerical results with the scalar oracle before comparing speed.
- Keep experiments forced-only and outside automatic dispatch.
- Do not modify canonical model checkpoints; GPU layouts and program binaries
  are disposable, versioned derivatives.
- Do not let Xe research block i7 CPU hardening.

## Upstream discovery focus

The Rust-first contribution questions are:

- can `cl3` or `opencl3` improve missing-platform/build-log/error reporting,
  lifecycle safety, capability queries, or integrated-GPU examples?
- can oneAPI-rs improve its typed memory, async ownership, device queries, or
  missing-runtime diagnostics?
- does `rust-gpu` expose a small reproducible SPIR-V issue relevant to compute
  without requiring a broad compiler project?

Intel Compute Runtime, Intel Graphics Compiler, Level Zero, and Khronos sources
remain non-Rust contribution targets only for a clear reproducible defect,
specification error, or missing test that belongs there.
