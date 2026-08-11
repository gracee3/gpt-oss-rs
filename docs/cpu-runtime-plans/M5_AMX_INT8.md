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

- None recorded yet.

## Completion evidence

- Implementation commits: pending
- Portable commands/results: pending
- Native feature compilation: pending
- AMX hardware execution: deferred/untested
- Closeout commit/workflow: pending

