# Step 1 — AVX-512/VNNI Eight-Output MXFP4 GEMV

- Status: research complete; ready for implementation planning
- Planned exposure: forced/experimental only
- Automatic baseline retained: AVX2 x8 with `InterleavedSplitX8V2`

## Objective and terminology

Build a genuine AVX-512/VNNI kernel that evaluates one quantized activation
row against eight MXFP4 output rows. Earlier milestone text called this
"multi-row AVX-512"; the precise term is **eight-output GEMV**. Multiple
activation rows belong to the GEMM workstream.

The implementation must preserve the current doubled-E2M1 integer domain,
per-32-value E8M0 scale, Q8 and residual-Q8 modes, FP32 accumulation order at
block boundaries, bias, and canonical row tails. Performance is not a gate and
the kernel will not become automatic in this sprint.

## Current repository baseline

`crates/gpt-oss-cpu-kernels/src/lib.rs` defines scalar, AVX2 row, AVX-512/VNNI
row, AVX2 x8, and exact-BF16 choices. Automatic dispatch selects AVX2 x8 on an
AVX2-capable host even when AVX-512 is present. Forced AVX-512 currently selects
the canonical one-row VNNI implementation.

`crates/gpt-oss-cpu-kernels/src/x86.rs` provides two reusable ideas:

- `mxfp4_x8_block_dots` decodes one 136-byte x8 block and produces eight exact
  INT32 dots. AVX2 consumes four rows per 256-bit load.
- `mxfp4_q8_dot_unpacked_avx512_vnni` shifts signed doubled-E2M1 values by 12,
  uses unsigned-weight × signed-activation `VPDPBUSD`, then subtracts
  `12 * sum(activation)`.

`InterleavedSplitX8V2` stores each complete K=32 group as:

```text
8 E8M0 scale bytes
chunk 0: 8 output rows × 8 packed bytes = 64 bytes
chunk 1: 8 output rows × 8 packed bytes = 64 bytes
```

The low nibbles represent one 16-value half of K and the high nibbles the
other. One 64-byte chunk therefore fits a ZMM register and contains the same
eight K positions for all eight output rows. Incomplete output groups remain
canonical and do not need padding or a second persistent layout.

## Source findings

### AVX-E001 — x8 layout is already a matrix-friendly persistent layout

- **CURRENT-REPO FACT:** `Mxfp4WeightLayout::InterleavedSplitX8V2`,
  `x8_block`, and `mxfp4_x8_block_dots` in
  `crates/gpt-oss-cpu-kernels/src/{lib.rs,x86.rs}` expose eight scales followed
  by two 64-byte value chunks.
- **LOCAL-SOURCE OBSERVATION:** llama.cpp `ggml/src/ggml-cpu/repack.cpp`,
  `block_mxfp4x8` and `make_block_mxfp4x8`, uses the same structural eight-row
  grouping; its source GGUF MXFP4 nibble convention is already split-half.
- **LOCAL-SOURCE OBSERVATION:** ik_llama.cpp `ggml/src/ggml-common.h` and
  `ggml/src/iqk/iqk_gemm_legacy_quants.cpp` define a 136-byte
  `block_mxfp4_r8` and combine two r8 blocks for AVX-512 work.
- **INFERENCE:** the current cache layout can feed both an eight-output ZMM
  GEMV and later sixteen-output GEMM tiles. A new model-sized cache would add
  storage/versioning without resolving a correctness constraint.

The layouts are structurally equivalent after each project's canonical MXFP4
conversion. This is not a claim that official SafeTensors bytes, GGUF
canonical bytes, and repacked bytes are interchangeable.

### AVX-E002 — VNNI shifted-weight correction is exact and already audited

- **CURRENT-REPO FACT:** doubled E2M1 weights lie in `[-12, 12]`, Q8 values in
  `[-127, 127]`, and the existing row VNNI kernel computes `(w + 12) * a -
  12 * a` in INT32.
- **PRIMARY-SOURCE FACT:** Intel `VPDPBUSD` multiplies unsigned bytes by signed
  bytes and accumulates groups of four into INT32.
- **INFERENCE:** shifting the weights is simpler for a ZMM x8 kernel than
  applying activation signs to every decoded weight byte. The largest one-block
  exact dot, `32 * 12 * 127 = 48,768`, is far below INT32 limits.

### AVX-E003 — upstream matrix paths reuse x8 rather than introducing AVX-512 weights

- **LOCAL-SOURCE OBSERVATION:** llama.cpp
  `ggml/src/ggml-cpu/arch/x86/repack.cpp`,
  `gemm_q4_b32_8x8_q8_0_lut_avx`, pairs two `block_mxfp4x8` records for its
  AVX-512 main matrix path and retains smaller AVX2 paths for tails.
- **LOCAL-SOURCE OBSERVATION:** ik_llama.cpp
  `mul_mat_mxfp4_r8_q8_2` follows the same two-r8 organization.
- **LIMITATION:** these kernels have different activation packing and
  accumulation contracts. They are design evidence, not code templates or
  performance evidence for this repository.

### AVX-E004 — E8M0 special values need a common semantic fix

- **CURRENT-REPO FACT:** `e8m0_scale` currently builds FP32 bits as
  `(scale as u32) << 23`, mapping byte `0x00` to floating zero and `0xff` to a
  non-NaN exponent pattern; tests lock that behavior.
- **PRIMARY-SOURCE FACT:** MX v1.0 defines E8M0 `0x00` as `2^-127` and `0xff`
  as invalid/NaN.
- **EXPERIMENT:** all 597,196,800 scale bytes in the pinned 20B checkpoint lie
  in `[115, 136]`.
- **PROVISIONAL DECISION:** establish specification-correct behavior in the
  scalar contract, update synthetic edge tests, and make all SIMD scale
  expansion use the same helper. This is not a current-checkpoint mismatch.

## Proposed kernel dataflow

For each K=32 block and eight-row output tile:

1. Load the eight E8M0 scale bytes and expand them through the common semantic
   helper into eight FP32 scales.
2. For each of the two 64-byte layout chunks, load the whole chunk into a ZMM
   register.
3. Decode low and high nibbles with a 16-entry doubled-E2M1 LUT broadcast into
   every 128-bit lane. `VPSHUFB` is lane-local, so every lane must contain the
   same LUT.
4. Broadcast the applicable eight signed activation bytes into all eight
   64-bit row lanes. Low and high nibbles use K positions 0-15 and 16-31 in the
   same split-half order as the cache.
5. Add 12 to the decoded signed weights, execute `VPDPBUSD`, reduce each pair
   of INT32 partials belonging to one eight-byte output row, and subtract
   `12 * sum(the same eight activation bytes)`.
6. Sum the four K=8 pieces to one exact INT32 dot per output row.
7. Convert the eight dots to FP32 and apply `0.5 * weight_e8m0_scale *
   activation_scale`; add to the eight FP32 bias accumulators.

Residual-Q8 runs the integer dot twice while the decoded low/high weight
vectors are live. Primary and residual contributions use their own activation
scales and are added in the existing primary-then-residual order before the
next K block.

The full-width kernel needs AVX-512F, AVX-512BW, and AVX-512VNNI. AVX-512VL is
only required if the implementation deliberately retains a 256-bit VNNI
helper. The implementation plan must verify generated instructions and keep
the declared capability mask no broader than the emitted code.

## API and dispatch shape

No new public user-facing API is required. The kernel layer needs a new
`Mxfp4GemvKernel` identity such as `Avx512VnniX8` and a forced path that pairs
it with `InterleavedSplitX8V2`. The projection tile entry point remains the
contract:

- exactly eight rows use the ZMM x8 body;
- a remaining full canonical row uses the existing forced AVX-512 row kernel;
- non-x8 tails use canonical storage and never read past the tile;
- `auto` continues to select `Avx2X8` until the later benchmark phase.

Diagnostics must distinguish the new kernel from `Avx512VnniRow` and continue
to report the persistent layout independently.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| AVX2-style `abs(activation)` plus signed weights | Exact and viable, but adds sign/absolute work already avoided by the audited VNNI correction. Retain as a localization fallback, not the first design. |
| New AVX-512-specific persistent cache | Rejected initially. Current x8 bytes naturally fill a ZMM chunk and upstream AVX-512 paths reuse paired x8 records. |
| Sixteen-output decode kernel | Useful as a later matrix tile, but doubles decode working set and is not required to prove a genuine x8 GEMV. |
| Automatic AVX-512 selection | Deferred. Frequency, bandwidth, and host-dependent crossover are benchmark questions. |

## Focused correctness plan

Implementation planning must include:

- scalar equality for Q8 and residual-Q8 at K block counts 1, 2, and model
  widths;
- eight-row aligned tiles at real gate/up and down shapes;
- row starts around x8 boundaries and canonical output tails of 1 through 7;
- all 16 E2M1 codes, Q8 extrema, zero blocks, mixed signs, bias, and residual
  reconstruction;
- synthetic E8M0 `0x00`, normal, and `0xff` behavior shared across scalar,
  AVX2, and AVX-512;
- forced-feature rejection with incomplete capability sets;
- dispatch and diagnostic identity tests;
- one targeted full-model forced-path comparison after the kernel is wired.

No timing threshold, automatic crossover, long oracle, or repeated Criterion
run is part of this step's initial gate.

## Planning handoff

The implementation plan can treat persistent layout, integer mapping,
capability boundary, residual reuse, and tail behavior as decided. It must
still choose exact intrinsic helper boundaries and register allocation, which
are local implementation details rather than architectural questions.
