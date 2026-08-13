# Tiger Lake MXFP4 Matrix Candidate

The forced `avx512-vnni` matrix backend is a genuine 8-input by 8-output
AVX-512/VNNI microkernel. It retains `InterleavedSplitX8V2`: no persistent
weight format, cache key, or mapped-memory contract changes. Each 64-byte
split-weight chunk is decoded into ZMM vectors once and reused across up to
eight activation rows. Residual-Q8 reuses the same decoded vectors for the
primary and residual dot before advancing K. FP32 accumulation preserves the
scalar primary-then-residual and bias order.

The caller provides a 64-byte-aligned activation panel. One Q8 pass consumes
288 bytes per K block (eight FP32 scales plus eight 32-byte rows); residual-Q8
uses two passes. The kernel allocates nothing. Complete x8 output groups use
ZMM `VPDPBUSD`; output tails retain the scalar contract. Forced execution
checks AVX-512F/BW/VL/VNNI and OS vector state before touching output.

Tests cover Q8 and residual-Q8, bias/no-bias, M through 61 including both
8-row tile boundaries, x8 and output tails, multiple K blocks, exact scalar
and AVX2 equivalence, E8M0 edges, non-finite scales, signed zero, caller-owned
scratch canaries, invalid layout, unavailable capabilities, short/misaligned
scratch, and non-overwrite on validation failure.

`mxfp4_matrix_bench` is the standalone evidence runner. It validates exact
scalar agreement before timing, rotates scalar/AVX2/AVX-512/Auto order,
requires at least seven trials and thirty samples per method/shape, times only
kernel execution, and records requested/effective backend, scratch, output
hash, source/binary/CPU identity, and every raw duration. The paired bootstrap
analyzer promotes only maximal consecutive observed-M regions whose 95%
interval establishes lower median latency than scalar and every other legal
explicit candidate. Gaps, ties, uncertainty, unknown profiles, and unobserved
shapes remain scalar.

## Tiger Lake decision evidence

The promotion input commit is
`ea72bb19cc3dd3b6ebb4c810110553d167a61a4f`. Raw benchmark and deterministic
analysis artifacts are under
`/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/ea72bb19cc3dd3b6ebb4c810110553d167a61a4f/matrix/`;
the `SHA256SUMS` file hashes to
`106f81536bba49a22f8b7f6e64c85d9fbce9e55d8331f10382cb59bac446d989`.
The source embeds the five raw/analysis hashes used for dispatch.

For the exact residual-Q8 GPT-OSS shapes, M=3 was thermally clean across all
35 samples per method. AVX2 median latency was 5.028 ms for gate/up versus
59.745 ms scalar and 5.320 ms AVX-512/VNNI. Down was 2.464 ms versus
29.186 ms scalar and 2.599 ms AVX-512/VNNI. The paired bootstrap 95% interval
established AVX2 below both comparators for both shapes.

That microbenchmark result did not survive the mandatory full-request gate.
Three cool-start, order-alternating `harmony_63` pairs compared Candidate B
(`91f5c93d03f9d1e3d6b4a775cd6ef45328f0dd59`) with the exact pre-promotion
commit. Candidate B regressed by 0.522%, 0.653%, and 0.345%; the mean was
0.507% and the paired bootstrap 95% interval was 0.345%-0.653%. The raw A/B
evidence is under
`/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/91f5c93d03f9d1e3d6b4a775cd6ef45328f0dd59/promotion-ab-v1/`;
its `SHA256SUMS` file hashes to
`532f0b9ec2151043e5af829446b008ccfc7938d4c3f2087bb050df7d31d2a7eb`.

The immutable promotion result is therefore negative: the checked-in region
table is empty and every M>1 Auto problem resolves to scalar, including the
otherwise qualifying M=3 shapes. M=1 retains the existing AVX2 x8 GEMV. The
normalized Tiger Lake profile matcher and positive isolated benchmark hashes
remain recorded for provenance; neither enables a region. No processor brand
string participates in dispatch.

The benchmark attempted representative higher buckets through M=32. The
stock laptop reaches its package thermal limit during the much slower scalar
control at M>=4. The harness records start/end temperature and package/core
throttle-time deltas, and the analyzer makes any such row ineligible. It was
not technically sound to extrapolate those rows or spend days timing all 260
observed M values under throttling. They remain explicit controls and safe
scalar Auto fallbacks. AVX-512/VNNI did not win any thermally valid region;
its value in this sprint is forced correctness and benchmarking coverage.
