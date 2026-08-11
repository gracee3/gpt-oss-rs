# Experimental AMX-INT8 Matrix Backend

The `amx-int8` backend is an explicit prototype over the common
`Mxfp4MatmulProblem` contract. It is never selected by `auto`, creates no
persistent expanded-weight cache, and has not been executed on AMX hardware in
this development environment.

## Build and target support

Build the propagated feature at the server boundary:

```bash
cargo build -p gpt-oss-server --features amx-int8 --locked
```

| Build/target | Result |
| --- | --- |
| Feature disabled | Portable workspace builds normally; explicit `amx-int8` fails with a build-feature diagnostic |
| Feature enabled, Linux x86-64 | Builds the repository C++ intrinsic shim with `-mamx-tile -mamx-int8` and enables Linux permission handling |
| Feature enabled, another target | Portable status and emulation code remain available; native execution fails as unsupported |

The Linux x86-64 native build requires a C++17 compiler that provides the
Intel `_tile_*` intrinsics. The repository build compiles only
`native/amx_int8.cpp`; no external AMX runtime library is linked.

## Runtime gates and lifecycle

Explicit initialization reports and checks these gates separately:

1. Cargo feature support;
2. Linux x86-64 native-target support;
3. raw CPUID AMX-TILE;
4. raw CPUID AMX-INT8;
5. Linux `ARCH_GET_XCOMP_SUPP` tile-config and tile-data support;
6. existing or newly requested `ARCH_REQ_XCOMP_PERM` tile-data permission.

`CpuModel::load_with_matmul_backend` performs this initialization before it
maps the snapshot or constructs the Rayon worker pool. The successful status
is retained in `CpuModel` and included in server diagnostics. Initialization
is cached process-wide. `auto`, `scalar`, and `avx2` do not call it and never
request AMX permission.

Every native tile call loads a fixed palette-one configuration, zeros and
computes one signed-signed `TDPBSSD` C tile, stores INT32, and calls
`TILERELEASE`. Pointer/shape errors return before tile configuration. Rust
validates all buffer extents and alignment before entering the no-allocation,
`noexcept` shim.

## Numerical and scratch contract

The main tile is `M<=16`, `N=16`, `K=32`:

- A: up to 16 rows by 32 signed Q8 bytes (512 bytes);
- B: eight 64-byte VNNI rows containing 16 outputs by four K bytes (512
  bytes);
- C: up to 16 rows by 16 INT32 values (1,024 bytes).

Problems with at least two input rows and one full 16-output tile request one
2,048-byte caller-owned scratch region aligned to 64 bytes. B is decoded from
two adjacent x8 groups once per K block outside the M-tile loop. C is stored
after every K=32 block because activation and weight scales change at that
boundary. Rust then applies the per-row Q8 scale, per-column E8M0 scale, and
doubled-E2M1 factor in FP32. Residual Q8 repeats the A/C dot while retaining
the decoded B panel and adds primary before residual for each block.

M=1 and matrices without a full N=16 tile use the scalar matrix fallback after
all explicit AMX runtime gates pass. Output columns after complete N=16 tiles
also use the scalar range helper. These fallbacks preserve correctness; they
do not allow a forced backend to bypass unavailable build, hardware, kernel,
or permission gates.

`Kernels::mxfp4_matmul_amx_int8_emulated` is a portable correctness oracle. It
uses the same packers, scratch layout, FP32 accumulation, bias, stride, and
tail logic but emulates the signed tile dot in Rust. It is not used by serving.

## Validation boundary

Portable tests cover exact A/B packing, VNNI order, M=2/4/15/16, multiple K
blocks, Q8 and residual-Q8 ordering, bias and output stride, N tails, signed
extrema, aligned/exact/short/misaligned scratch, and canaries. Injected tests
cover every runtime denial gate. CPU CI compiles the propagated server feature,
runs the portable feature tests, and applies warnings-denied Clippy.

The development i7 exposes AVX-512/VNNI but no AMX flags. Consequently, this
milestone makes no claim about native hardware equality, signal/error behavior
under allocated tile XSTATE, performance, tiling quality, or automatic
crossover. Those require a later AMX-capable host campaign.
