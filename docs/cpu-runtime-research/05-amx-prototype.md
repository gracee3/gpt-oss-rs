# Step 5 — AMX-INT8 Prototype and Integration Seam

- Status: portable implementation complete; AMX hardware execution deferred
- Hardware status: no AMX-capable local host
- Planned exposure: explicit build feature plus forced/experimental backend
- Initial numerical mapping: AMX-INT8, not AMX-BF16

## Objective

Define an AMX prototype that consumes the common MXFP4 matrix problem without
creating a second model execution architecture. The prototype must preserve
doubled-E2M1, Q8/residual-Q8, per-K=32 E8M0 scaling, FP32 block accumulation,
and fallback behavior.

The portable panel packer, scalar tile emulator, capability/permission model,
and build integration can be completed without AMX hardware. Hardware
execution, performance claims, and automatic selection are deferred.

## Exact integer mapping

For one activation row `m`, output row `n`, and K=32 microscaling block `b`:

```text
integer_dot[m,n,b] = sum(k=0..31, q8[m,b,k] * doubled_e2m1[n,b,k])

contribution[m,n,b] = integer_dot
                        * activation_scale[m,b]
                        * e8m0_weight_scale[n,b]
                        * 0.5
```

Both integer inputs are signed bytes, so AMX `TDPBSSD` is a direct exact dot.
The maximum absolute one-block value is `32 * 127 * 12 = 48,768`, safely
inside INT32.

Integer accumulators **cannot** span multiple K blocks before scaling:
activation scales vary by `(m,b)` and weight scales by `(n,b)`. The kernel must
zero/compute/store its INT32 tile for every K=32 panel, apply the outer product
of row and column scales in FP32, and then add to the FP32 output tile.

Residual-Q8 repeats the signed dot with the residual activation bytes, applies
the residual scale, and adds it after the primary contribution for the same K
block. Decoded weights can remain loaded while the two activation panels are
processed.

## Tile and panel layout

The first main tile is `M <= 16`, `N = 16`, `K = 32`:

```text
A tile: M rows × 32 signed bytes
B tile: 8 rows × 64 bytes
        each row contains N groups of four signed K bytes (VNNI order)
C tile: M rows × 16 INT32 values (64 bytes per tile row)
```

AMX palette one provides eight tile registers, each at most 16 rows by 64
bytes. The above A, B, and C shapes fit. N tails and M=1 use AVX2/AVX-512
fallback initially; the prototype does not reconfigure tiles for every small
tail.

### Transient B pack

For each K=32 block:

1. Take two adjacent `InterleavedSplitX8V2` groups to cover 16 output rows.
2. Decode both low/high nibble halves through the doubled-E2M1 LUT.
3. Write signed bytes in AMX/VNNI B order:
   `B[k_group_of_4][output_row][k_within_group]`.
4. Retain the 16 E8M0 column scales separately for FP32 rescaling.

This panel is 512 bytes (`8 * 64`) and is reused across every M tile for the
same output/K block. The first prototype packs it into caller-owned scratch;
it does not persist an 8-bit expansion of the model. The implementation loop
order must pack B outside the M-tile loop so activation rows amortize the
transformation.

### Transient A and C storage

For a fixed K block, gather up to 16 semantic Q8 rows into one contiguous
`M * 32` signed-byte A panel and retain M FP32 activation scales. Residual mode
has a second A panel/scales. Store the INT32 C tile into an aligned
`M * 16 * 4`-byte scratch region before FP32 scale/accumulation.

A simple sequential residual prototype needs one A, one B, and one C tile at a
time. A later register allocation may keep B resident and use separate
primary/residual A/C tiles, but that is a local optimization and not part of
the semantic contract.

## Source findings

### AMX-E001 — signed INT8 preserves the existing quantized contract

- **CURRENT-REPO FACT:** MXFP4 is already decoded to exact signed doubled-E2M1
  bytes and activations are Q8 or two-pass residual-Q8 signed bytes.
- **PRIMARY-SOURCE FACT:** Intel AMX supplies signed-signed `TDPBSSD` with INT32
  results and palette-one tiles up to 16 rows by 64 bytes.
- **PROVISIONAL DECISION:** use AMX-INT8 first. It requires only bounded panel
  reordering and preserves the same per-block scale points as scalar/SIMD.

### AMX-E002 — x8 weights are compact storage, not the AMX feed layout

- **CURRENT-REPO FACT:** x8 nibbles are compact and arranged for LUT decode;
  AMX consumes full signed bytes in four-byte VNNI groups.
- **LOCAL-SOURCE OBSERVATION:** pinned llama.cpp and ik_llama.cpp reuse x8
  persistent records for wider vector kernels but still transform operands for
  their matrix instruction order.
- **PROVISIONAL DECISION:** transiently unpack a 16x32 B panel. A separate
  persistent AMX cache or bounded hot-expert cache is a later memory/performance
  tradeoff requiring measurements.

### AMX-E003 — hardware detection and OS permission are distinct

- **CURRENT-REPO FACT:** `features.rs` combines AMX CPUID bits with XCR0 tile
  state in one `CpuFeatures` snapshot.
- **PRIMARY-SOURCE FACT:** Linux exposes `ARCH_GET_XCOMP_SUPP`,
  `ARCH_GET_XCOMP_PERM`, and `ARCH_REQ_XCOMP_PERM`; AMX tile-data permission is
  per process, inherited across fork, cleared on exec, and subject to signal
  alternate-stack size checks. First AMX use allocates task XSTATE storage.
- **PROVISIONAL DECISION:** represent hardware capability, kernel XSTATE
  support, granted process permission, and calling-thread tile state
  separately. Do not treat XCR0 alone as a usable-kernel gate.

### AMX-E004 — tile context has an explicit lifetime

- **PRIMARY-SOURCE FACT:** Intel's sample requests permission, loads a 64-byte
  tile configuration, executes, stores results, and calls `TILERELEASE`.
- **LOCAL-SOURCE OBSERVATION:** oneDNN
  `examples/ukernels/cpu_brgemm.cpp` pairs `set_hw_context` with
  `release_hw_context`; `src/cpu/x64/amx_tile_configure.cpp` compares/configures
  tile state rather than assuming one permanent process configuration.
- **PROVISIONAL DECISION:** request process permission before creating the CPU
  Rayon workers. Every thread that calls the shim enters a scoped tile context
  and releases it on all normal/error exits. A thread-local "configured once
  forever" flag is insufficient because other code may change tile state.

### AMX-E005 — stable Rust needs an isolated implementation boundary

- **PRIMARY-SOURCE FACT:** Rust AMX target features and intrinsics remain under
  `x86_amx_intrinsics` issue 126622 on the repository's Rust 1.97.1 toolchain.
- **CURRENT-REPO FACT:** the CPU kernel crate is pure Rust and has no build
  script; the workspace lock already contains the `cc` crate transitively.
- **PROVISIONAL DECISION:** add a small repository-owned C/C++ intrinsic
  translation unit only under an explicit Linux x86-64 AMX feature. Keep panel
  packing, validation, dispatch, permission reporting, scale accumulation, and
  scalar emulation in Rust.

The native shim should expose a narrow no-allocation/no-unwind ABI over
validated pointers, dimensions, strides, and aligned scratch. It should only
configure/load/compute/store/release tiles. Rust owns bounds and converts shim
status codes into clear errors. Non-Linux, non-x86-64, missing-compiler, absent
hardware, or denied-permission cases must never reach an AMX instruction.

## Runtime lifecycle

1. A build without the explicit AMX feature contains no C/C++ shim and reports
   the backend unavailable.
2. A build with the feature checks CPUID AMX-TILE and AMX-INT8 independently of
   runtime permission.
3. Explicit AMX runtime initialization queries kernel XSTATE support and
   requests tile-data permission before the shared worker pool is constructed.
4. Failure returns a diagnostic distinguishing hardware absence, kernel
   support absence, permission/signal-stack rejection, and build-feature
   absence.
5. A worker invoking the kernel enters a scoped tile guard, loads the fixed
   configuration, executes panels, stores C, and releases tile state.
6. Forced AMX fails clearly if any gate is missing. `auto` never selects AMX in
   this milestone and falls through without requesting permission.

Permission is process-wide, but tile contents/configuration and first-use
XSTATE allocation are thread concerns. Requesting permission before thread
creation simplifies signal-stack and worker initialization behavior; it does
not eliminate the per-calling-thread tile guard.

## Integration with the matrix contract

AMX is another implementation of `Mxfp4MatmulProblem`:

- valid for Q8/residual-Q8, K divisible by 32, M greater than one, and at least
  one full N=16 tile;
- consumes the same x8 persistent weight view and caller-owned scratch;
- writes the same row-major FP32 output and bias semantics as scalar/SIMD;
- delegates M=1 and N tails to the explicitly selected VNNI/AVX2/scalar
  helpers;
- exposes required scratch size before execution;
- has no scheduler, sequence, attention, or allocation logic.

No AMX-specific model runner API or repack manifest is introduced for the
first prototype.

## AMX-BF16 alternative

AMX-BF16 would expand MXFP4 values to BF16, use BF16 activations, and accumulate
FP32 directly. It avoids per-K-block INT32 stores but changes the activation
contract, adds expansion/rounding decisions, and does not naturally reuse the
current residual-Q8 semantics. A persistent BF16 weight expansion would also
materially increase cache size.

Record BF16 as a later experimental adapter for comparison with exact-BF16
diagnostics. It is not the first integrated AMX path and is not needed to claim
an AMX prototype seam.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| Nightly Rust AMX intrinsics | Rejected for the initial path because it would impose an unstable toolchain on the workspace. |
| Large `global_asm!` microkernel | Technically possible, but more opaque to audit and maintain than a small intrinsic translation unit. Retain only if compiler support blocks the shim. |
| Persistent decoded INT8 model cache | Deferred due substantial memory growth; transient 512-byte panels prove semantics first. |
| AMX for M=1 decode | Rejected initially. Tile setup/panel work is unnecessary when existing VNNI kernels directly serve GEMV. |
| Carry INT32 across K blocks | Incorrect because block scales differ. |
| Automatic AMX dispatch without host data | Rejected; keep forced/experimental. |

## Portable correctness plan

Tests that do not require AMX hardware:

- decode x8 plus canonical fixtures into the exact signed VNNI B-panel order
  and compare every value with scalar MXFP4 decoding;
- scalar-emulate `TDPBSSD` over packed A/B panels and compare INT32 tiles with
  the scalar matrix reference for M `2, 4, 15, 16` and N=16;
- verify primary/residual order, per-row activation scales, per-column E8M0
  scales, multiple K blocks, bias, and FP32 block accumulation;
- assert the one-block overflow bound and exercise signed extrema;
- validate scratch-size/alignment calculations and canaries around A/B/C
  buffers;
- compile the explicit feature on Linux x86-64 with no AMX execution;
- test hardware-absent, build-feature-absent, kernel-support-absent, and denied
  permission diagnostics through injected capability providers;
- verify forced AMX never silently falls back while `auto` never requests AMX
  permission.

Deferred AMX-host tests:

- hardware/emulator equality for full/tail matrix problems;
- tile guard configuration and release on every Rayon worker;
- signal/cancellation/error paths under real XSTATE allocation;
- performance, M/N tile choice, B-panel caching, and dispatch crossover.

## Planning handoff

The INT8 numerical mapping, K-block scaling boundary, A/B/C tile shapes,
transient panel ownership, x8 reuse, FFI/toolchain boundary, permission model,
tile lifecycle, fallback, and no-hardware gate are ready for implementation
planning. Hardware tuning cannot invalidate these interfaces; it can only
change internal tiling, caching, and later selection policy.

## Implementation outcome

The `amx-int8` feature now propagates from CPU kernels through the model runner,
engine, and server. Rust owns status diagnostics, permission acquisition,
portable A/B/C panel packing, validation, scalar tile emulation, FP32 scale
accumulation, tails, and fallbacks. A repository-owned C++17 intrinsic shim is
built only for feature-enabled Linux x86-64 and performs the signed AMX tile
dot while releasing tile state on every exit.

The model requests tile-data permission before mapping the snapshot and before
constructing worker pools. Explicit selection fails at the first unavailable
build, target, CPUID, XSTATE, or permission gate; automatic selection neither
chooses AMX nor requests permission. Portable tests cover matrix shapes,
primary and residual Q8, extrema, scratch bounds, canaries, and injected status
failures, and CPU CI compiles and tests the feature. The development host has
no AMX CPUID support, so native execution, AMX-worker lifecycle stress, and
performance remain deferred certification work.
