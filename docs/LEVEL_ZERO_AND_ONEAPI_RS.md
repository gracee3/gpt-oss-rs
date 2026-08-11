# Level Zero and oneAPI-rs Research Note

- Status: research and pre-planning only
- Recorded: 2026-08-11
- Repository baseline: `752b0c5138eb46efd573759bf2ea9f89afc29298`
- Implementation decision: no new runtime or Cargo dependency
- Possible future target: explicit, experimental Intel integrated-GPU backend
- Active T14 lane: [`XE_RESEARCH_AND_PREPLANNING.md`](XE_RESEARCH_AND_PREPLANNING.md)

This note is a handoff for hosts participating in `gpt-oss-rs` development.
It records what oneAPI-rs and Level Zero are, where they overlap with this
project, which design ideas are worth retaining, and what would actually be
required to run a GPT-OSS operation on an Intel GPU.

The current project priority remains native CPU execution. Nothing here makes
OpenCL, Level Zero, SYCL, Intel's compute runtime, or an integrated GPU part of
the supported runtime.

## Reviewed sources

The source trees were inspected at these revisions on 2026-08-11. They are
research inputs, not submodules, vendored sources, or dependencies.

| Source | Revision | License | Reviewed area |
| --- | --- | --- | --- |
| [oneAPI-rs](https://github.com/oneapi-src/oneapi-rs) | `1581663fdd0fd73e79df2900a2576d6cca8ff2a1` | MIT OR Apache-2.0 | workspace shape, SYCL wrapper, C++ build bridge, USM ownership, events, kernel arguments |
| [Level Zero loader](https://github.com/oneapi-src/level-zero) | `1ca51c950d97f34d9d271615af8d797836fe6974` | MIT | C ABI headers, loader/layers, resource model, logging and contribution boundary |
| [Intel Compute Runtime](https://github.com/intel/compute-runtime) | `566bac0e8757baf7e6f680d63da6b5be3f775e2f` | MIT | location of Intel's Level Zero/OpenCL GPU driver implementation |
| [Level Zero specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/) | rolling `latest` documentation accessed 2026-08-11 | specification repository license | core programming and module model |

Refresh the pins before relying on implementation details. Do not silently
treat a moving `main` or `latest` page as the source used by this note.

## The three relevant layers

| Layer | Direction | Primary audience | What it does not provide |
| --- | --- | --- | --- |
| oneAPI-rs / `sycl-rs` | Mostly safe Rust access to the broad SYCL heterogeneous-compute model | Rust developers willing to install Intel oneAPI, DPC++, and a SYCL runtime | A pure-Rust compiler, a direct Level Zero binding, or a lightweight driver-free deployment |
| Level Zero | Low-level, explicit C ABI plus loader, validation, and tracing layers | Runtime, framework, compiler, profiler, and systems-library authors | An ML framework, MXFP4 operations, a kernel compiler, or the Intel GPU driver itself |
| Intel Compute Runtime | Intel's system implementation of Level Zero and OpenCL for supported GPUs | Driver/platform integrators and applications consuming the installed runtime | A Rust-native host API or GPT-OSS kernels |

Level Zero's loader is implemented largely in C++, but its application-facing
API in `include/ze_api.h` is a C ABI with opaque handles. That boundary is
friendly to Rust FFI. Rewriting the loader or Intel GPU driver in Rust is not
part of this project.

## oneAPI-rs direction and fit

At the reviewed revision, oneAPI-rs describes itself as experimental, has not
released 0.1.0, and tests Linux with Intel oneAPI Toolkit 2026.1. Its workspace
contains:

- `sycl/sycl-rs-sys`: Rust/C++ bridge and SYCL shim;
- `sycl/sycl-rs`: the mostly safe public wrapper;
- `sycl/sycl-rs-derive`: typed kernel-argument derives.

`sycl-rs-sys/build.rs` invokes a C++17 SYCL compiler with `-fsycl`, compiles
C++ shims, and links `sycl` and `intlc`. Runtime kernel compilation accepts
SYCL/C++ source through the installed toolchain. It therefore solves a
different problem than a dependency-light, repository-owned Rust runtime.

The useful design observations are:

- native objects are owned by RAII wrappers;
- host, shared, and device USM allocations have distinct Rust types;
- only host-accessible allocations expose ordinary host dereferencing;
- enqueued allocation/initialization carries an event or future until it is
  safe to consume;
- kernel launch and argument ABI agreement remain explicitly `unsafe`;
- typed argument lists and plain-data bounds narrow accidental ABI mistakes;
- device, queue, context, event, memory, and kernel concerns remain separate.

These ideas can inform a narrow implementation. The SYCL object model, C++
bridge, runtime C++ compilation, and broad portability layer should not be
imported merely to obtain them.

## Level Zero direction and execution model

Level Zero is intentionally explicit. A minimal compute application performs
roughly this lifecycle:

```text
initialize loader
  -> enumerate driver and devices
  -> query capabilities and queue groups
  -> create context
  -> allocate host/shared/device memory
  -> create module from SPIR-V or a native binary
  -> create kernel and set arguments/group size
  -> append copies and launch to a command list
  -> close and submit the list to a command queue
  -> wait on event or fence
  -> destroy owned resources in reverse order
```

The application owns synchronization and correctness. Validation and tracing
layers assist development without turning the API into a high-level runtime.
The loader dispatches calls to an installed driver implementation; on Intel
GPU systems that implementation normally comes from Intel Compute Runtime.

The API covers much more than this project needs. A future experiment should
initially ignore images, IPC, peer access, metrics, debugger APIs, Sysman,
ray-tracing extensions, and most extension chains.

## What a small Rust port would mean

A small Rust host layer is plausible. It would contain checked declarations
for only the Level Zero types and calls exercised by one experiment, optional
dynamic loading of `libze_loader.so`, RAII newtypes, capability records, and a
narrow error model.

That port would not remove these external requirements:

- a compatible Level Zero loader on the host;
- an Intel GPU driver implementing the interface;
- firmware and operating-system device access;
- a kernel binary in SPIR-V or native device format;
- exact synchronization, address-space, alignment, and kernel-ABI handling.

Level Zero does not compile Rust source into GPU code. oneAPI-rs handles this
gap by invoking a SYCL C++ toolchain, which is intentionally outside the
lightweight path considered here.

Practical kernel-source choices for a later experiment are:

1. use OpenCL C online compilation to establish a small auditable kernel and
   retrieve a device-specific program binary where supported;
2. produce and validate a reproducible offline SPIR-V artifact for direct
   Level Zero module creation;
3. investigate a Rust-to-SPIR-V toolchain separately, without making an
   unstable compiler path a runtime requirement;
4. use Level Zero native binaries only after documenting device/version
   portability and cache invalidation.

Generated binaries must have source, build instructions, license, toolchain
version, and a reproducibility story. They must not become unexplained blobs.

## Overlap with `gpt-oss-rs`

The current CPU runtime already establishes several reusable boundaries:

- scalar numerical oracles and explicit optimized backends;
- immutable mapped weights and versioned derived repack caches;
- `Mxfp4MatmulProblem` as a kernel-facing matrix contract;
- caller-owned scratch and explicit persistent/transient layouts;
- capability discovery separated from forced and automatic dispatch;
- transactional sequence execution and commit.

A future Intel-GPU experiment can preserve those boundaries rather than create
a second model runtime. A possible host-side shape is:

```text
IntelGpuRuntime
DeviceCaps
GpuAllocation<Host | Shared | Device>
Module
Kernel
CommandQueue
Event
```

The first kernel should implement one bounded operation behind a forced
experimental backend and compare against the existing scalar result. It should
not begin with an entire transformer, scheduler, or compatibility abstraction.
The existing MXFP4 cache remains canonical unless measurements justify a
separate versioned GPU layout.

## Concepts worth carrying forward

1. Use typed opaque owners and deterministic destruction for native handles.
2. Make memory visibility/location part of the allocation type.
3. Return completion tokens from asynchronous copies and launches.
4. Keep raw FFI and kernel ABI operations in a very small `unsafe` module.
5. Query capabilities before choosing queue, memory, module, and kernel paths.
6. Distinguish loader presence, driver presence, device presence, capability,
   module-build failure, and execution failure in diagnostics.
7. Keep validation/tracing opt-in and preserve enough context to reproduce a
   failed submission.
8. Cache only derived artifacts with explicit format, device, driver, and
   source/toolchain identity.

## Ideas deliberately not adopted

- no oneAPI-rs, OpenCL, SYCL, Level Zero, or GPU-driver dependency now;
- no general SYCL compatibility layer;
- no runtime dependency on DPC++ or C++ kernel compilation;
- no claim that a host wrapper makes the complete GPU stack pure Rust;
- no direct translation of broad upstream APIs or implementation files;
- no Vulkan work in the current Xe research phase;
- no automatic integrated-GPU selection before full correctness and measured
  benefit on named hardware;
- no attempt to replace Intel Compute Runtime.

## Future Tiger Lake research gate

Before implementing even a small Level Zero backend, record independently on
each candidate host:

1. PCI device and kernel driver identity;
2. installed Level Zero loader and Intel driver versions;
3. device discovery output and queue-group capabilities;
4. supported module formats and relevant integer/FP16/subgroup properties;
5. available shared/device memory behavior;
6. a copy-and-integer-kernel smoke test with validation logging;
7. an OpenCL and Level Zero comparison for the same numerical operation when
   both compiler/module routes are reproducible;
8. a concrete GPT-OSS operation and the scalar equivalence fixture.

The experiment remains forced-only until failures are clear, artifacts are
reproducible, and measured end-to-end benefit exceeds transfer, repacking, and
submission cost.

## Possible upstream work

oneAPI-rs is the more natural Rust destination for generally useful ownership,
lifetime, typed-memory, query, test, or diagnostic improvements. A direct
Level Zero binding is not its stated direction, so that idea requires an
upstream discussion before implementation.

Potential evidence-backed contributions include:

- better missing-toolchain/runtime/device error categories and messages;
- additional device-query coverage and tests;
- USM allocation/lifetime/host-access safety tests;
- examples and documentation for older integrated Intel GPUs;
- small reproductions of loader or validation-layer diagnostic gaps.

Level Zero and Intel Compute Runtime are C/C++ projects. They remain
contribution targets only for a clear, reproducible problem that cannot be
fixed usefully in Rust-facing documentation or wrappers. The repository-wide
policy and candidate ledger live in
[`UPSTREAM_CONTRIBUTION_DISCOVERY.md`](UPSTREAM_CONTRIBUTION_DISCOVERY.md).

## Handoff questions for the next research pass

- Does OpenCL online compilation, cached OpenCL binaries, or Level Zero with
  offline SPIR-V provide the cleanest reproducible path on the T14?
- Which one existing MXFP4 matrix shape best isolates transfer, module,
  submission, and compute costs?
- Can the current x8 packed cache feed the GPU efficiently, or would any GPU
  layout merely duplicate model-scale data?
- Which SPIR-V production path is reproducible without adding a runtime C++
  compiler requirement?
- Is a useful oneAPI-rs diagnostic or safety improvement discovered while
  answering those questions?
