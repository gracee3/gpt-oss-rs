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
