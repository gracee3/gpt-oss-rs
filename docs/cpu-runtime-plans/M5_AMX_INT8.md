# M5 Plan — AMX-INT8 Prototype

- Research: [`../cpu-runtime-research/05-amx-prototype.md`](../cpu-runtime-research/05-amx-prototype.md)
- Dependency: M2 matrix contract
- Exposure: Cargo feature plus explicit matrix backend
- Automatic path: never AMX during this program
- Local hardware expectation: portable validation only

## Entry reconciliation

Before implementation, inspect the landed matrix problem/view/scratch contract,
feature propagation across kernel/model/engine/server crates, build toolchain,
CPU feature detection, Rayon pool construction, and Linux target constraints.
Confirm the repository compiler accepts the intended C++ intrinsics and record
any shim/compiler adjustment before native code lands.

## Interfaces

- Add an `amx-int8` feature to CPU kernels and propagate it through model
  runner, engine, and server without making it a default feature.
- Portable Rust owns panel packing, validation, scratch calculation, scalar
  tile emulation, scale accumulation, diagnostics, and fallback selection.
- A small repository-owned C++ shim exists only for Linux x86-64 feature builds.
  Its no-allocation/no-unwind ABI accepts validated pointers/shape/strides,
  configures palette-one tiles, runs `TDPBSSD`, stores INT32, releases tile
  state on every exit, and returns status codes.
- Use transient M<=16, N=16, K=32 A/B/C panels. Store INT32 after every K
  block, then apply per-row activation and per-column E8M0 scales in FP32.
  Residual Q8 repeats the dot while decoded weights are reusable.
- Pack B from paired x8 records outside the M-tile loop. Add no persistent
  AMX-expanded cache.
- M=1 and N tails use documented vector/scalar fallback within the explicit
  backend contract.

Represent and diagnose separately: build support, CPUID AMX-TILE/AMX-INT8,
Linux XSTATE support, process tile-data permission, and per-call tile
configuration. Explicit initialization requests permission before worker-thread
construction. `amx-int8` selection fails clearly for every unavailable gate;
`auto` neither selects AMX nor requests permission.

### Reconciled repository seam

The landed M2 API already has the `amx-int8` enum value, selects
`InterleavedSplitX8V2` for its model weights, queries scratch before execution,
and preserves `auto` as GEMV/scalar. M5 replaces only the current explicit
unavailable result. No matrix, model-step, or persistent-repack interface needs
another variant.

The existing `CpuFeatures` AMX fields include an XCR0 tile-state gate and
therefore cannot diagnose CPUID, Linux XSTATE support, and granted permission
separately. A new AMX runtime-status interface will read raw CPUID independently
and use Linux `ARCH_GET_XCOMP_SUPP`, `ARCH_GET_XCOMP_PERM`, and
`ARCH_REQ_XCOMP_PERM`. The legacy general feature snapshot remains unchanged
for non-AMX dispatch. Explicit model loading initializes this status before
`rayon::ThreadPoolBuilder`; automatic/scalar/AVX2 loading does not call it.

Portable AMX scratch is one reusable, 64-byte-aligned 2,048-byte region: a
16x32 signed A panel, an 8x64 VNNI B panel, and a 16x16 INT32 C tile. Panels
are repacked per K=32 block, B outside the M-tile loop. M=1 and problems with
no complete N=16 tile use the scalar fallback and require no panel scratch,
but explicit runtime initialization still enforces every AMX availability
gate. N tails are calculated by the scalar range helper after full tiles.

The native boundary is one `noexcept` C++ function that receives validated
A/B/C pointers and `M<=16`, loads a fixed palette-one configuration, performs
one signed-signed dot tile, stores C, and releases tile state. Rust owns all
packing, scaling, accumulation, bounds, capability, and permission logic.

## Commit slices

1. Add feature propagation, capability/permission abstractions, injected
   diagnostic tests, and build-absent behavior.
2. Add portable A/B/C packing, scalar tile emulation, scratch/bounds tests, and
   matrix scale accumulation.
3. Add the Linux x86-64 C++ shim/build integration and guarded Rust FFI.
4. Wire explicit backend selection, fallbacks, worker pre-initialization, and
   user-facing diagnostics.
5. Add portable feature CI coverage and close out docs/evidence with the
   no-hardware limitation.

## Focused gate

- exact x8/canonical decode into VNNI B-panel order;
- scalar tile emulation for M `2, 4, 15, 16`, N=16, multiple K blocks;
- Q8/residual ordering, extrema, bias, row/column scales, and FP32 block
  accumulation against the scalar matrix reference;
- scratch exact/short/alignment/overflow/canary coverage;
- injected build/hardware/kernel/permission denial diagnostics;
- forced AMX never silently falls back and `auto` never initializes AMX;
- Linux x86-64 feature compilation without executing AMX;
- formatting, portable tests, warnings-denied kernel Clippy, and propagated
  server feature check.

Hardware execution, tile release under real signals/errors, performance,
tiling/cache tuning, and automatic crossover are explicitly deferred.

## Documentation updates

- build/feature instructions and supported-target matrix;
- matrix backend and runtime diagnostics, permission lifecycle, and fallbacks;
- CI portable-compile coverage;
- research M5 status and this evidence ledger, prominently noting no local
  AMX-hardware validation.

## Deviations and decisions

- The workspace `cc` crate was only transitive; the kernel crate adds it as an
  explicit build dependency and adds an optional Linux x86-64 `libc`
  dependency for `arch_prctl`.
- GCC on the development host compiled a no-execution probe using
  `_tile_loadconfig`, `_tile_loadd`, `_tile_zero`, `_tile_dpbssd`,
  `_tile_stored`, and `_tile_release` with C++17 and the two AMX target flags.
  The repository shim can therefore use intrinsics; no assembly fallback or
  toolchain change is needed.
- The development i7 has no AMX CPUID flags. Native execution remains
  intentionally unreachable locally; feature compilation, injected status
  diagnostics, portable packing, and scalar tile emulation are the local gate.

## Completion evidence

- Implementation commits: pending
- Portable commands/results: pending
- Native feature compilation: pending
- AMX hardware execution: deferred/untested
- Closeout commit/workflow: pending
