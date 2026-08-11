# MXFP4 CPU Backend / Intel ISA Handoff

## Context

Repository:

```text
~/gpt-oss-rs
https://github.com/gracee3/gpt-oss-rs
```

Local upstream study checkouts created for this work:

```text
/home/emmy/src/llama.cpp
/home/emmy/src/mistral.rs
/home/emmy/src/ik_llama.cpp
```

These are sibling reference repositories, not submodules or runtime
dependencies. The project already isolates CPU kernel work in:

```text
crates/gpt-oss-cpu-kernels/
├── src/
└── benches/
```

The broader workspace includes dedicated benchmark, engine, runtime-plan,
reference, conformance, and MoE crates. CPU kernel development should remain
behind a clean dispatch/interface layer rather than leaking ISA details into
the engine.

## Goal

Develop a serious CPU backend for GPT-OSS that treats MXFP4 as a native
storage and compute format rather than dequantizing the whole model to FP16 or
FP32.

Do not optimize specifically for the available Tiger Lake laptops. They are
useful development and benchmark hosts, but the architecture should target
modern x86 broadly with runtime ISA dispatch.

The intended progression is:

```text
scalar reference
    ↓
AVX2 + FMA
    ↓
AVX-512 + byte manipulation / VNNI where useful
    ↓
AMX
    ↓
future AVX10 / next-generation AMX
```

The fundamental design rule is:

> Dispatch based on required CPU features and workload shape, not CPU model
> names or generations.

Avoid backends named after Tiger Lake, Sapphire Rapids, Granite Rapids, or
other product generations. Prefer capability-oriented names such as:

```text
mxfp4_scalar
mxfp4_avx2
mxfp4_avx512
mxfp4_avx512_vnni
mxfp4_amx_int8
mxfp4_amx_bf16
```

Future extensions must be possible without redesigning the engine-facing API.

## MXFP4 refresher

MXFP4 uses 4-bit E2M1 floating-point elements in blocks of 32 values:

```text
32 × E2M1 values
+
1 × E8M0 shared scale
```

Each E2M1 element contains one sign bit, two exponent bits, and one mantissa
bit. The positive representable magnitudes are approximately:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

with corresponding negative values. Every group of 32 values shares an E8M0
power-of-two scale.

One canonical block occupies:

```text
32 × 4 bits = 128 bits = 16 bytes
shared scale             = 1 byte
total                    = 17 bytes
```

Effective storage is 136 / 32 = 4.25 bits per weight. Preserving this compact
representation is central to GPT-OSS CPU inference because it minimizes expert
weight memory traffic.

## Do not fully dequantize MXFP4

A correctness/reference implementation may perform:

```text
packed MXFP4
    ↓
decode E2M1 → FP32
    ↓
apply E8M0 scale
    ↓
FP32 multiply
```

That should not be the primary optimized path. E2M1 has only 16 bit patterns,
so codes can be mapped cheaply to a small integer representation and kept
compact until the hot loop:

```text
packed MXFP4 weights
        │
        ▼
extract 4-bit codes
        │
        ▼
E2M1 lookup / transformation
        │
        ▼
small signed integer values
        │
        │       quantized activations
        │              │
        └──────┬───────┘
               ▼
          integer dot product
               │
               ▼
          integer sums
               │
               ▼
       apply combined scales
               │
               ▼
          FP32 accumulator
```

This preserves MXFP4's bandwidth advantage and lets modern x86 integer
dot-product hardware participate.

## Existing prior art

Do not invent this entirely from scratch.

### llama.cpp

Mainline llama.cpp has MXFP4 × Q8 CPU dot-product implementations. Its x86
path demonstrates the basic technique:

```text
load packed FP4
↓
separate low/high nibbles
↓
small lookup-table decode
↓
multiply against Q8 activations
↓
integer accumulation
↓
apply block scales
```

The implementation is particularly useful as a readable AVX2 reference. It
demonstrates that MXFP4 does not require a persistent FP32 expansion.

### mistral.rs

mistral.rs is a semantic and layout reference for GPT-OSS, including MXFP4
nibble order and scaling, expert tensors, routing, SwiGLU, attention sinks, and
YaRN behavior. Its current quantized matmul and device-specific paths should be
audited for changes since the revision pinned in `UPSTREAM_PROVENANCE.md`.

### ik_llama.cpp

ik_llama.cpp is important for studying:

```text
AVX-512
AVX-512 VNNI
quantized GEMM
repacked/interleaved weight layouts
```

Its work suggests that high-performance GEMM should consider backend-specific
repacking rather than insisting on the canonical 17-byte block layout.

## The MXFP4 layout problem

Canonical blocks are awkward for wide SIMD:

```text
[scale][16 packed FP4 bytes]
[scale][16 packed FP4 bytes]
[scale][16 packed FP4 bytes]
...
```

The 17-byte stride is poorly aligned for repeated AVX2, AVX-512, and AMX
loads. A backend-specific representation might instead resemble:

```text
[FP4 data from block 0]
[FP4 data from block 1]
[FP4 data from block 2]
[FP4 data from block 3]
...

[scale 0 scale 1 scale 2 scale 3 ...]
```

or an interleaved layout shaped around a microkernel's K/N dimensions.

> Keep the model/checkpoint representation canonical, but allow CPU backends
> to create optimized, versioned packed representations at model-load time.

The cost is paid once and amortized across inference. Packed size, load time,
integrity, and reuse must be measured.

## AVX2 backend

AVX2 is the broadly available optimized x86 baseline. Useful capabilities
include 256-bit vectors, byte operations, `PSHUFB` lookup, integer
multiply/add, and FMA.

Likely flow:

```text
load packed MXFP4
↓
split nibbles
↓
PSHUFB-based E2M1 decode
↓
load/quantize activation data
↓
integer multiply/add
↓
INT32 accumulation
↓
apply MXFP4 + activation scales
↓
FP32 accumulate
```

The AVX2 path should be well optimized because it covers a large installed
base and remains a serious competitor for memory-bound decode.

## AVX-512 backend

AVX-512 is not merely AVX2 at twice the width. Relevant subsets may include:

```text
AVX512F
AVX512BW
AVX512VL
AVX512DQ
AVX512VBMI / VBMI2
AVX512VNNI
```

Each kernel must declare exactly which features it requires. For packed
MXFP4, byte permutation/manipulation may matter as much as vector width.

A VNNI-oriented path might be:

```text
packed MXFP4
↓
nibble extraction
↓
E2M1 → integer mapping
↓
Q8 activation values
↓
VNNI dot product
↓
INT32 accumulators
↓
combined block scaling
↓
FP32 output
```

Do not assume AVX-512 wins automatically. Benchmark unpacking cost, integer
dot throughput, clock behavior, cache behavior, and memory bandwidth.
Autoregressive generation may remain mostly bandwidth-bound.

## AMX direction

AMX is a two-dimensional tile/matrix facility rather than wider SIMD:

```text
Tile A × Tile B → Tile C
```

Relevant data types currently include AMX-INT8, AMX-BF16, and AMX-FP16. There
is no direct AMX-MXFP4 operation.

The central research question is:

> Can MXFP4 feed AMX-INT8 efficiently without conventional dequantization?

A promising route is:

```text
MXFP4 E2M1
    ↓
cheap integer mapping
    ↓
INT8-compatible tile representation
    ↓
AMX-INT8 matrix multiply
    ↓
INT32 accumulation
    ↓
apply MXFP4/activation scales
```

E2M1's tiny value set may permit a small signed coefficient representation.
The E8M0 power-of-two scale can potentially be applied around accumulated
sub-blocks rather than per element.

## Arithmetic question

For an MXFP4 block:

```text
w_i = q_i × 2^s
```

where `q_i` is the E2M1 value and `s` is the E8M0-derived exponent. For
quantized activations:

```text
x_i ≈ a_i × Sx
```

Then, if the accumulated region shares a weight scale:

```text
Σ(w_i × x_i) = Σ(q_i × a_i) × 2^s × Sx
```

MXFP4 changes scale every 32 weights, so an AMX kernel cannot blindly perform
a large dot and apply one final scale. It must preserve those boundaries.

### Strategy A — accumulate per MXFP4 block

Compute a separate 32-element integer dot for every block, then apply its
weight and activation scales. This is simple and exact but may underutilize
AMX's larger tiles.

### Strategy B — group equal or similar exponents

Reorder blocks by scale or transform scale differences into integer shifts.
This may damage contiguous matrix layout and is not a first implementation.

### Strategy C — pre-scale integer representations

At pack time, convert blocks to a representation compatible with a shared tile
scale. This increases packed size in exchange for simpler AMX execution. The
extra memory traffic may erase the advantage and must be measured.

### Strategy D — hybrid AVX-512 + AMX

Use AVX-512 for MXFP4 decode, scale handling, and packing; use AMX for the
dense integer matrix multiply. This is promising for prefill and larger
batches where tile conversion is amortized.

## GEMV and GEMM are different workloads

Do not assume one backend is optimal for the whole inference pass.

### Autoregressive decode

Single-sequence decode is approximately GEMV:

```text
1 activation vector × large weight matrix
```

It is often memory-bandwidth limited. A fused packed-MXFP4 AVX-512 kernel may
be extremely competitive because weights remain near 4.25 bits until use.
AMX may lose if it requires material expansion.

Likely candidate:

```text
decode batch=1 → AVX-512 MXFP4 GEMV
```

### Prefill

Prompt ingestion is GEMM-like:

```text
many token vectors × weight matrix
```

Weight reuse is much higher, so AMX has a better chance to amortize conversion
and packing.

Likely candidate:

```text
prefill → AMX
```

### Batched decode

As concurrent sequence count grows, decode becomes matrix-matrix work. AMX
may become preferable above an empirically selected threshold.

## Likely runtime policy

Dispatch on CPU capabilities, operation type, batch size, M/N/K dimensions,
packing availability, and potentially thread count or NUMA topology.

Conceptually:

```text
if no SIMD:
    scalar
else if AVX2 only:
    avx2
else if AVX512:
    if GEMV / batch very small:
        avx512_mxfp4
    else:
        avx512_gemm

if AMX available:
    if prefill or sufficiently large batch:
        amx
    else:
        benchmark/tune against avx512
```

Thresholds should eventually be selected from benchmark evidence, not product
names.

## Current development hosts

The two currently available laptop CPUs are Intel Tiger Lake systems with
AVX2 and useful AVX-512 subsets. They do not support AMX.

Use them to develop and validate scalar, AVX2, and AVX-512. Do not let their
limits define the architecture. AMX work may initially be compile-tested,
unit-tested against scalar/reference, CI-tested on suitable hardware, and
benchmarked later on a remote AMX host.

## Correctness strategy

Every optimized kernel must be checked against one source of truth:

```text
scalar/reference MXFP4
        ↓
AVX2
        ↓
AVX-512
        ↓
AMX
```

Tests should cover:

```text
all 16 E2M1 bit patterns
multiple E8M0 scales
zero blocks
positive/negative extrema
random blocks
random activation vectors
odd matrix dimensions
tail handling
multi-threaded execution
```

Use tolerances appropriate to the accumulation path. Test exact dot/GEMV/GEMM
primitives independently of full model inference wherever possible.

## Benchmark strategy

Add microbenchmarks under `crates/gpt-oss-cpu-kernels/benches/` for at least:

```text
MXFP4 decode throughput
MXFP4 × F32 dot
MXFP4 × Q8 dot
GEMV
small GEMM
large GEMM
packing/repacking cost
```

For each kernel record:

```text
ns / operation
effective GB/s of MXFP4 weight input
effective GOPS/TOPS where meaningful
cycles per weight
```

Model-level reports must separate prefill tokens/s, decode tokens/s, TTFT,
model load/packing time, and peak RSS. Do not combine prefill and decode into
one performance number.

## CPU feature dispatch

Create a capability description rather than scattering feature checks:

```rust
struct CpuFeatures {
    avx2: bool,
    fma: bool,

    avx512f: bool,
    avx512bw: bool,
    avx512vl: bool,
    avx512dq: bool,
    avx512vbmi: bool,
    avx512vnni: bool,

    amx_tile: bool,
    amx_int8: bool,
    amx_bf16: bool,
    amx_fp16: bool,
}
```

Names may differ, but detection should happen once and produce a backend plan.
Hot loops must not repeatedly run CPUID checks.

## Proposed kernel abstraction

A conceptual interface is:

```rust
trait Mxfp4Kernel {
    fn gemv(/* ... */);
    fn gemm(/* ... */);
}
```

Do not force GEMV and GEMM to share a layout if that costs performance.
Separate traits may be better:

```text
Mxfp4GemvKernel
Mxfp4GemmKernel
```

Backend-specific packed objects should be possible:

```rust
enum PackedMxfp4Weights {
    Canonical(/* ... */),
    Avx2(/* ... */),
    Avx512(/* ... */),
    Amx(/* ... */),
}
```

The exact API should follow repository conventions and preserve future
extension points.

## First implementation milestones

### Phase 1 — document and benchmark current behavior

Inspect the current CPU kernel implementation and identify storage, scalar,
AVX2, AVX-512, activation, GEMV/GEMM, packing, and dispatch behavior. Do not
refactor until current correctness and performance are captured.

### Phase 2 — clean scalar reference

Ensure there is a boring, obviously correct MXFP4 implementation. This is the
permanent oracle.

### Phase 3 — study and import proven AVX2 ideas

Use llama.cpp as reference material for nibble extraction, E2M1 lookup,
integer accumulation, and scale application. Benchmark against current code.

### Phase 4 — AVX-512 packed MXFP4 GEMV

Focus on decode/generation first. Explore 512-bit packed loads, VBMI/VBMI2,
VNNI dots, multiple blocks per iteration, prefetch, scale-vector handling, and
tails. Do not merely widen AVX2; profile each design.

### Phase 5 — repacked GEMM layout

Create an optional load-time representation optimized for matrix operations.
Study row-interleaved layouts from ik_llama.cpp. Measure packing time, size,
cache behavior, and GEMM improvement.

### Phase 6 — AMX-INT8 prototype

Build a standalone microkernel first. It must answer:

1. What integer representation should E2M1 use?
2. How are 32-element scale boundaries handled?
3. What activation format pairs best with it?
4. Can AMX avoid enough expansion to preserve the bandwidth advantage?
5. At what M/batch size does AMX beat AVX-512?
6. Does AMX help prefill much more than decode?

Integrate only after those questions have benchmark answers.

## Hypothesis worth testing

The likely optimal Intel policy is:

```text
single-token decode:
    packed MXFP4 AVX-512

prefill:
    AMX-INT8/BF16/FP16, depending conversion path

small batched decode:
    AVX-512 or AMX depending threshold

large batched decode:
    AMX

AVX2 hardware:
    packed MXFP4 AVX2

fallback:
    scalar
```

This is a hypothesis, not an architectural assumption. Benchmarks may reject
it.

## Future Intel ISA direction

Keep the backend extensible for AVX10, new AMX formats, FP8-oriented AMX, and
future low-precision conversion or matrix instructions. Do not hard-wire the
abstraction around AVX-512 as the final SIMD generation or AMX-INT8 as the
permanent MXFP4 bridge.

## Immediate implementation-oriented report

Before a large refactor, inspect the current implementation and report:

1. Files and functions involved in MXFP4 CPU execution.
2. Existing scalar, AVX2, and AVX-512 paths.
3. Exact activation formats used by current MXFP4 kernels.
4. Whether MXFP4 is eagerly dequantized or fused in the hot loop.
5. Current model weight layout and repacking behavior.
6. Current runtime CPU dispatch mechanism.
7. Existing benchmark coverage.
8. Natural insertion points for AVX-512 GEMV, backend-specific packed weights,
   and future AMX.

Compare those findings with llama.cpp, mistral.rs, and ik_llama.cpp. First
capture current architecture, correctness, and benchmarks. Then propose the
smallest patch sequence from scalar reference to strong AVX2, strong AVX-512,
repacked GEMM, and experimental AMX.

## Guiding principle

The objective is not to make GPT-OSS fast on two laptops. It is to build a
portable, feature-dispatched MXFP4 CPU kernel stack that exploits the best
available x86 ISA while preserving the compact 4.25-bit weight representation
for as long as practical.

The Tiger Lake systems are only the first AVX2/AVX-512 development hosts. The
architecture must scale from ordinary AVX2 systems through modern
AVX-512/VNNI systems and into AMX-class servers without redesigning the
inference engine.
