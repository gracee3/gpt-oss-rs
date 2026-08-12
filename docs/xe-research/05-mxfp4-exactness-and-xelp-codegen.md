# X5 — MXFP4 Exactness and Xe-LP Code Generation

- Numerical result: pass
- Current-driver DP4A lowering: demonstrated
- Performance interpretation: lowering exists but no useful end-to-end win

## Exact arithmetic contract

The K=32 oracle and device kernel decode canonical adjacent weights low nibble
first using the exact doubled-E2M1 integer table:

```text
0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
```

Each Q8 or residual-Q8 block dot is accumulated in `i32`; only then is the
integer result converted to FP32 and multiplied by `0.5 * E8M0 * Q8 scale`.
FP contraction is disabled. K tails and non-K=32 exact-block inputs are
rejected before FFI.

The suite exhausts all 16 E2M1 codes at all 256 E8M0 bytes, including `0xff`
NaN behavior, sign, zeros, extrema, and nibble order: 4,096 cases. A fixed
ChaCha8 seed adds 10,000 randomized blocks; the same blocks validate both Q8
and the production-default residual-Q8 result, for 14,096 device cases per
API.

Both APIs reported:

- zero exact integer-intermediate mismatches;
- zero bit mismatches for finite one-block FP32 results;
- zero BF16-boundary mismatches;
- zero NaN-behavior mismatches;
- `E8M0 0xff` is NaN and byte `0x00` decodes to FP32 bits `0x00400000`.

The later full projection comparison permits at most four ULP plus a `1e-6`
absolute floor and demands identical BF16 boundaries. The X6 paths in fact
matched the scalar oracle at zero ULP for every measured output.

## Native compiler evidence

The current 26.05 driver produced the modules; the preserved 23.43 `ocloc` was
used only as an offline container decoder. It did not compile or modify the
modules and was never mapped in the hardware process.

| Path | Native SHA-256 | Projection assembly SHA-256 | Metadata SHA-256 |
| --- | --- | --- | --- |
| OpenCL source | `c06ce561479e141fbd75aedc89a5b76234180afbf5ef669a3d06445c49844e45` | `913451924a118ed5760cd81fc6f3a68509c531a3392ac504891348f0feaa54e9` | `92b482bac11eded1a6af189dca5186c75a79aebc9a66eed616d83c8de4783a8a` |
| Level Zero SPIR-V | `a516e9350f7552f464c54dd9c8d16b2234ff36606d5b7eddb6f1ffbd0809be58` | `a2b3d7bace2aeed4eeb17d4a0eb05227c06447f390775c785b118c371e45ea43` | `2d15ee0b880b9e59a413083c57b32f0938b101b8bb1e40b97fc6df35af309b6d` |

Each scalar projection assembly contains 32 identified `dp4a` instructions.
The exact-block kernel also contains 32. Metadata reports SIMD32 and 128 GRFs
for both kernels and emits no scratch/spill allocation field. This is accepted
evidence of integer-dot lowering and register/private-memory pressure, not a
claim that the overall projection is efficient.

## Vector, subgroup, work-group, and layout exploration

The source retains scalar decode and an explicit `char4` integer-dot form for
OpenCL subgroup sizes 8, 16, and 32. The DP4A native container has distinct
SIMD8, SIMD16, and SIMD32 entries, each at 128 GRFs, with 16, 16, and 32 DP4A
instructions respectively. All produced zero-ULP/BF16-identical M=4 outputs.

End-to-end medians at M=4, work-group 64, were:

| Subgroup | Xe | AVX2 | AVX2/Xe speedup | 95% interval |
| ---: | ---: | ---: | ---: | --- |
| 8 | 8.687 ms | 4.933 ms | 0.568x | 0.566–0.571 |
| 16 | 5.650 ms | 4.907 ms | 0.869x | 0.864–0.879 |
| 32 | 5.320 ms | 4.954 ms | 0.931x | 0.922–0.938 |

For the best subgroup 32 path, work-groups 32, 64, and 128 yielded 0.917x,
0.909x, and 0.879x respectively at M=4; their upper confidence bounds all
remain below parity. Work-group 32 was nominally best, but no configuration
survives the useful-win gate. Weight reuse is the per-output 90-block loop;
activation data is reused only through the integrated GPU cache. No derived
weight layout was needed, and creating one would violate the duplicate-weight
gate without evidence of a compensating win.

Alignment follows the ABI manifest and driver allocation guarantees. The
compiler's automatic DP4A lowering of the straightforward scalar source makes
the explicit vector form unnecessary on this stack. Any future compiler change
must reproduce native-code evidence; capability strings alone are insufficient.

## Evidence records

| ID | Manifest SHA-256 | Raw validation SHA-256 |
| --- | --- | --- |
| X5-OCL | `51cc124d2c3e2a5a635dde1b54de2bfd57afcef30574f881ad908b0f294352bf` | `8b5d5104dfe8ff450d1d3555dea9cf5bd1e84a008f474956f99ee9aca2b8e589` |
| X5-L0 | `4ce3e32bd3c17cb47b2a54be571e0a20bc17e7e172bd7ea3bfa9b2f20e50af8c` | `9551e8579d6417d4ba5148fd914dfb35ded824518340233a979c15360aff9ff8` |

Both X5 records are `pass` at revision
`7034d3284711cbe83030b0704462afa290c03b24`. Subgroup and work-group evidence
is indexed in the final README because it uses the complete X6 timing protocol.
