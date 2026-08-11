# M1 Plan — AVX-512/VNNI Eight-Output GEMV

- Research: [`../cpu-runtime-research/01-avx512-vnni-x8.md`](../cpu-runtime-research/01-avx512-vnni-x8.md)
- Exposure: explicit `avx512-vnni`
- Automatic path: unchanged AVX2 x8
- Persistent layout: `InterleavedSplitX8V2`

## Entry reconciliation

Before code changes, inspect `gpt-oss-cpu-kernels` kernel identity, scalar scale
expansion, x8 tile helpers, dispatch requirements, repack layout selection,
diagnostics, and tests. Confirm compiler support for the intended intrinsics.
If stable Rust cannot express a required AVX-512 intrinsic, revise this plan in
a standalone commit while preserving the public identity and semantics.

Completed on Rust 1.97.1. Stable intrinsics expressed the intended ZMM decode
and VNNI dot directly, so no pre-implementation plan refinement was required.

## Interfaces

- Add `Mxfp4GemvKernel::Avx512VnniX8` distinct from the existing row kernel.
- Add a genuine eight-output ZMM/VNNI implementation for Q8 and residual-Q8.
- The implementation consumes complete x8 tiles from
  `InterleavedSplitX8V2`; output tails use the canonical-row path.
- All scalar and SIMD paths share E8M0 behavior: `0x00 -> 2^-127`, normal
  exponents expand exactly, and `0xff -> NaN`.
- Explicit AVX-512 selection pairs the x8 identity and x8 layout. `auto`
  retains `Avx2X8` and its capability boundary.
- Diagnostics report kernel and layout independently.

The x8 body processes each K=32 scale block without crossing scale boundaries.
Weights decoded for one block are reused for the primary and residual integer
dots. Each contribution is scaled and accumulated in existing
primary-then-residual order.

## Commit slices

1. Correct common E8M0 expansion and update extrema/special-value tests.
2. Add the x8 AVX-512/VNNI helper, Q8/residual integration, bounds tests, and
   instruction/capability audit.
3. Wire identity, explicit dispatch, x8 repack selection, diagnostics, and
   forced-path tests.
4. Close out docs, research status, focused command evidence, and the short
   forced full-model comparison.

Every slice must compile and is pushed. A compiler-driven plan deviation lands
before the affected implementation slice.

## Focused gate

- scalar equivalence for Q8 and residual-Q8 at one, two, and representative
  model K-block counts;
- aligned x8 groups and canonical tails 1 through 7;
- all E2M1 codes, Q8 extrema, mixed signs, zero blocks, bias, and residuals;
- E8M0 `0x00`, ordinary values, and `0xff` across shared helpers;
- forced-feature rejection and kernel/layout diagnostic identity;
- `cargo fmt --all --check`;
- `cargo test -p gpt-oss-cpu-kernels --locked`;
- `cargo clippy -p gpt-oss-cpu-kernels --all-targets --locked -- -D warnings`;
- affected model-runner checks/tests;
- one short forced AVX-512 full-model comparison on a capable host.

## Documentation updates

- `docs/CPU_RUNTIME.md`: semantic fix, forced selection, capability contract,
  tail path, and unchanged automatic baseline.
- `docs/cpu-runtime-research/01-avx512-vnni-x8.md`: implementation status and
  deviations.
- kernel diagnostics/help text and any repack layout documentation.
- this file: commands, commits, host details, and results.

## Deviations and decisions

- No architectural deviations. FP32 scaling is applied lane-by-lane after the
  ZMM integer dots so accumulation order remains identical to the scalar row
  contract. This is an internal implementation choice, not a changed API or
  numerical boundary.
- Release assembly confirms the x8 body itself uses AVX-512F/BW/VNNI without
  AVX-512VL. The complete forced compatibility path retains AVX2 and
  AVX-512VL requirements because canonical tails use the existing row kernel.

## Completion evidence

- Implementation commits: `849785e`, `eb92640`.
- `cargo test -p gpt-oss-cpu-kernels --locked`: 25 passed on the
  AVX-512/VNNI-capable i7-1185G7.
- `cargo test -p gpt-oss-model-runner --lib --locked`: 345 passed before the
  final explicit AVX-512 diagnostic extension; the closeout gate reruns the
  affected test.
- `cargo clippy -p gpt-oss-cpu-kernels --all-targets --locked -- -D warnings`:
  passed.
- Release instruction audit: ZMM `vpshufb` and ZMM `vpdpbusd` present in the
  x8 block-dot body.
- Full-model captures:
  `/data/models/openai/gpt-oss-rs-cpu-work/results/m1-harmony_122-auto.json`
  and `m1-harmony_122-avx512-x8.json`. Both completed with first token
  `200005`, finite model output, exit status zero, and zero swaps. Wall times
  were 28.05 s and 27.57 s respectively; timing is informational.
- Closeout commit/workflow: this documentation and diagnostic checkpoint;
  remote CPU workflow verification follows publication.
