# Selected-expert CUDA plan

## Promotion gate

The mixed executor cannot begin real heterogeneous execution until a CUDA
primitive can execute **one selected resident expert** from the native packed
representation and reproduce the CPU semantic boundary. H2 begins with decode
`M=1`; it is correctness-first and is not required to beat CPU.

The primitive is separate from router selection. It consumes a validated
expert handle and route descriptor; H4 adds an exact BF16-weight GPU0 router
projection plus stable E=128 selection/packing. Current f32/cuBLAS router
projection is not assumed exact. The expert primitive
does not scan expert IDs, choose top-k, apply routing weights, reduce ranks,
move K/V, or publish state.

## First supported operation

```text
input:
  one GptOssRouteDescriptor
  BF16 activation [1, 2880]
  one immutable CudaNativeMxfp4BlocksScalesV1 expert handle

resident weights:
  gate/up blocks [5760, 90, 16] U8       8,294,400 B
  gate/up scales [5760, 90] U8             518,400 B
  gate/up bias [5760] BF16                  11,520 B
  down blocks [2880, 90, 16] U8          4,147,200 B
  down scales [2880, 90] U8                 259,200 B
  down bias [2880] BF16                       5,760 B

output:
  unweighted BF16 [1, 2880] into caller-owned route slot
```

The handle points only at its expert's final device allocations. An allocation
containing all experts, host `Vec` ownership, or a cached FP16 expansion does
not satisfy the interface.

## Exact kernel sequence

A new independently authored `kernels/gpt_oss_selected_expert.cu` is registered
through the existing PTX build and `KernelLoader` machinery. The first sequence
has three explicit boundaries:

1. `gpt_oss_mxfp4_gate_up_m1_exact`: one output row per logical worker; for
   each of 90 blocks it decodes 32 values, rounds each decoded scaled weight to
   BF16, multiplies by BF16 input in f32, and accumulates the same 16 lane
   positions as `accumulate_mxfp4_bf16_block`. It sums lanes 0→15, adds the
   matching bias in the CPU order, and writes 5,760 BF16 values.
2. `gpt_oss_swiglu_m1_exact`: reads interleaved gate/up BF16 pairs, applies
   gate/up clamps and exactly the CPU sequence `gate*alpha → sigmoid →
   gate*sigmoid → up+1 → product`, rounding every documented operation through
   BF16, and writes 2,880 BF16 values.
3. `gpt_oss_mxfp4_down_m1_exact`: repeats the exact 16-lane MXFP4×BF16 rule for
   2,880 rows, adds BF16-defined bias, and writes the caller's unweighted BF16
   result slot.

The MXFP4 decoder must implement all OCP E2M1 values and E8M0 special handling
used by `gpt-oss-cpu-kernels`, including scale zero and invalid `0xff`; the
current CUDA exponent-bit shift is not the authority. Compiler contraction,
parallel tree reduction, and reassociation are forbidden until oracle evidence
proves them equivalent. Explicit round-to-nearest f32 multiply/add operations
or another audited sequence must reproduce the CPU 16-lane trace.

## Scratch and residency

The conservative first-job scratch is:

```text
gate/up BF16 boundary  = 5,760 × 2 = 11,520 B
SwiGLU BF16 boundary   = 2,880 × 2 =  5,760 B
scratch total                         17,280 B
caller output slot      = 2,880 × 2 =  5,760 B
```

Small status/trace buffers and allocation alignment are separately charged.
The H2 test records actual aligned high water and sets the pool class; scratch
is leased before submit and cannot allocate within a kernel/call. A later
validated alias may reduce the bound, but the envelope does not assume it.
Weights are decoded into registers/local computation a block at a time. No
33,177,600-byte gate/up or 16,588,800-byte down FP16 matrix exists per call or
in a cache.

## Stream, event, and error contract

`prepare_selected_expert` validates device identity, representation version,
key/owner, exact shapes/byte counts, input/output alignment, route slot, scratch
capacity, and supported `M=1` before enqueue. It returns an error without
launching if any check fails.

`submit` enqueues the three kernels on the selected device's work stream and
records a terminal event. Launch errors are returned immediately; asynchronous
errors are reported when the event/stream is joined and are attached to the
prepared step. Input, handle, scratch, result, context, module, and stream stay
owned until that terminal event is drained. Cancellation suppresses later
publication but does not destroy any of them early.

The primitive emits kernel start/end event intervals, exact bytes read/written,
scratch requested/actual, expert/route identity, kernel/PTX/build identity, and
the first requested trace boundary. Enqueue time is never reported as
completion time.

## Oracle fixtures

### Synthetic

- exhaustive E2M1 nibble values and E8M0 boundary/special scales;
- signed zero, finite extremes, clamp boundaries, and values on either side of
  every BF16 rounding transition used by GPT SwiGLU;
- deterministic activation patterns that exercise every 16-lane position and
  multiple 90-block reductions;
- malformed shape, wrong representation/device, missing scratch, non-finite
  route input, launch-failure, asynchronous-failure, and cancellation/drain
  cases; and
- repeated GPU0/GPU1 runs with bit-identical outputs and no allocation growth.

### Real weight

The fixture is the Phase 1 pinned 20B layer-0 activation and a selected expert
from the retained route `[31,21,22,6]`, beginning with expert 31 and then all
four. The same immutable input/weight identity runs through
`CpuExpertProjection::ExactBf16`, GPU0, and GPU1. The fixture records:

1. source activation BF16 bits;
2. selected native block/scale/bias slice identity;
3. optional decoded-weight samples and the 16 accumulator lanes for the first
   divergent output row;
4. gate/up BF16 boundary;
5. every GPT SwiGLU BF16 boundary;
6. down BF16 output; and
7. output hash and actual scratch/allocator high water.

The fixture is forced directly on an expert; it does not claim router or whole
layer correctness. H6 combines it with real routing and all owners.

## Component-specific comparison rules

| Component | Gate |
|---|---|
| Expert ID, route row/rank, placement, representation and shape | Exact equality |
| Input, decoded-weight BF16 samples, gate/up, each SwiGLU boundary, down output | Bit-exact BF16 |
| CPU 16-lane f32 accumulators and lane sum | Bit-exact f32 when the planned operation order is implemented; any exception must use an already-authoritative ULP rule and retain first divergence |
| Final selected-expert BF16 output | Bit-exact; failure stops H2 |
| GPU0 versus GPU1 | Bit-exact outputs and descriptor identity |
| Later one-layer and continuation | Exact rank identities/BF16 boundaries and exact retained token IDs; no broad whole-layer tolerance |

An internal f32 ULP allowance cannot waive a final BF16 mismatch. A broad
relative tolerance for the layer is prohibited.

## Reuse and explicit rejection

| Current component | Planned disposition |
|---|---|
| `kernels/gpt_oss_moe.cu::mxfp4_value` and tensor indexing | Audit as a decoding/indexing clue only; independently implement BF16 weight rounding and E8M0 specials |
| `gpt_oss_dequant_expert_f16_kernel` | Reject expansion/output dtype/lifecycle; no call from the new wrapper |
| `gpt_oss_route_topk_kernel` | Reject E≤64 and unproven rounding; H4 supplies a separate E=128 exact route kernel |
| select/mask and weighted-add kernels | Reject all-expert scan, masked full rows, and expert-order reduction |
| `fused_silu_mul_split` | Reject ordinary SiLU semantics |
| `KernelLoader`, cudarc device/stream/module/error wrappers | Reuse after adding explicit event/lifetime contracts |
| `PinnedBuffer` allocation | Reuse under a hard-capped lease pool; never allocate on the route critical path |
| cuBLAS wrapper | Retain for separately validated dense operations; it is not a router/expert correctness foundation and is not used by the first native-packed expert primitive |

There is no “temporary” fallback to `GptOssMoeLayerWeights::forward` or
`forward_decode_gpu`. A shape other than `M=1` returns `UnsupportedShape` unless
the caller explicitly selects the serial-`M=1` correctness adapter.

## Prefill is separate

The first prefill-capable correctness adapter groups routes exactly as the
target interface requires but executes each CUDA route through the promoted
`M=1` primitive in stable order. It is intentionally not called a grouped
prefill kernel and receives no performance claim. After H6/H9 exactness,
H10 may add a native packed grouped kernel for observed buckets `M=7,24,33,61`
with a new scratch/crossover/oracle gate. It must not alter the decode
primitive's arithmetic or silently switch to the current host-f32 prefill.
