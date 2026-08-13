# X10 — Fused Tiger Lake Xe Expert Pipeline Research

- Status: research and implementation planning complete; production implementation not started
- Recommendation: forced-first three-kernel OpenCL expert pipeline
- Automatic Xe: unchanged and disabled
- Level Zero: L0-A; working research runtime with no compelling pipeline advantage
- Base repository revision: 0113e8214e765d168216bbee2120654555a4cfe4
- Research prototype revision: 9961137e34283e195546b3e6134e54ed7e7352cc
- Current host: Dell Latitude 5320, Core i7-1185G7, Iris Xe 8086:9a49
- Current runtime: Intel Compute Runtime 26.05.37020.3; Level Zero loader/DDI 1.28.2
- Research date: 2026-08-13

## Decision

Implement the next production candidate as a forced-first, three-kernel OpenCL
pipeline:

~~~text
host BF16 expert bucket
  -> host residual-Q8 preparation and one activation-record upload
  -> K1 existing Xe gate/up projection
  -> device FP32 gate/up intermediate
  -> K2 exact BF16 + SwiGLU + residual-Q8 preparation
  -> device ActivationRecordV2 down input
  -> K3 existing Xe down projection
  -> one final host synchronization/readback
  -> existing host route weighting and stable accumulation
~~~

Keep one in-order profiling queue. Allocate a bounded reusable workspace once
per engine and keep the gate/up intermediate and down activation records on the
device. Compile K2 as a separate OpenCL program with correctly rounded FP32
division enabled. Leave the immutable projection ABI v2 unchanged and add an
expert-pipeline ABI v1 for K2 and the multi-kernel orchestration.

Do not implement a gate/up-through-down monolithic kernel in the first pass.
The down projection consumes all 2,880 activated values, while existing gate
workgroups independently produce 32 output values. OpenCL offers no
grid-wide barrier inside a kernel. A monolith would therefore serialize the
work, duplicate computation, or require a large per-workgroup activation
working set and materially reduce occupancy. A later two-kernel
gate/up-plus-preparation experiment is a valid optimization after the
three-kernel baseline is correct and measured.

No automatic Xe policy is justified yet. M=1 decode and large-M prefill need
independent complete-expert and full-model campaigns. Weight residency remains
a tested candidate state, not an assumed prerequisite or promotion fact.

## 1. Repository and host identity

The requested reconciliation was performed before edits:

~~~text
git status --short --branch
  ## main...origin/main

git remote -v
  origin git@github.com:gracee3/gpt-oss-rs.git (fetch)
  origin git@github.com:gracee3/gpt-oss-rs.git (push)

git fetch --all --prune
git rev-parse HEAD
git rev-parse origin/main
  0113e8214e765d168216bbee2120654555a4cfe4
  0113e8214e765d168216bbee2120654555a4cfe4
~~~

The base worktree was clean. The completed Tiger Lake sprint was the fetched
main tip, not merely an assumed revision. Research work was placed on
agent/xe-fused-expert-research. The research-only semantic probe is commit
9961137e34283e195546b3e6134e54ed7e7352cc.

Current host identity:

| Property | Value |
| --- | --- |
| Host | latitude |
| System | Dell Latitude 5320, firmware 1.38.0 |
| OS | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-29-generic |
| CPU | Intel Core i7-1185G7, 4 cores / 8 threads |
| CPU ISA | AVX2, FMA, AVX-512F/BW/DQ/VL, AVX-512 VNNI |
| GPU | Intel Iris Xe 8086:9a49 |
| GPU capabilities | 96 compute units, subgroups 8/16/32, 64 KiB local memory |
| Unified memory reported | approximately 30.89 GB |
| Maximum single allocation | approximately 4.294 GB |
| System RAM | 31 GiB |

Earlier X8 documentation names its measurement host Lenovo T14. This X10
research was run on the current Dell host. Historical X8 numbers remain
historical controls bounded to their captured identity; they are not silently
relabeled as fresh Dell measurements.

## 2. Completed Tiger Lake evidence baseline

The merged sprint artifacts and all requested CPU/Xe research documents were
present. The conclusions carried forward are:

| Expert bucket | Observed share |
| --- | ---: |
| M=1 | 56.28% |
| M=2 | 2.46% |
| M=3 | 2.63% |
| M=4–7 | 8.11% |
| M=8–15 | 8.93% |
| M>=16 | 21.60% |

| Operation | Exclusive time share |
| --- | ---: |
| Gate/up projection | 65.62% |
| Down projection | 32.34% |
| Attention | 0.62% |
| Q/O dense BF16 work | 0.81% |
| SwiGLU | 0.28% |
| Residual-Q8 preparation | 0.23% |

Gate/up plus down account for approximately 97.96% after rounding. This makes
the expert projection topology the next high-leverage target.

CPU controls are settled:

- M=1 Auto is the AVX2 x8 path.
- M>1 Auto remains scalar.
- Forced AVX2 4x8 and AVX-512/VNNI 8x8 matrix implementations are available.
- M=3 forced AVX2 won its isolated measurement but an Auto promotion produced
  a repeatable mean full-request regression of 0.507%; no M>1 region was
  promoted.

Xe controls are also settled:

- explicit OpenCL Xe is supported; automatic Xe is disabled;
- an isolated 128 MiB cache case recorded 482/484 hits, 99.59%, avoided about
  3.19 GB of uploads, and held only 13,253,760 bytes;
- M=4 isolated gate/up fell from 10.868 ms streaming to 1.797 ms resident and
  down fell from 7.336 ms to 1.607 ms;
- the full-model cache campaign recorded zero hits and 846 misses at each
  128/256/512 MiB size, uploaded 5,606,340,480 bytes, and regressed from
  20.114 seconds without cache to 21.086–21.228 seconds with cache.

The profiler is retained as the only production instrumentation system. Its
disabled mean overhead was about +0.130% with an interval spanning zero, and
the completed campaign had no drops or truncations.

## 3. Exact current expert dataflow

Primary source locations are cpu_runner.rs lines 2794–3280,
gpt-oss-cpu-kernels/src/lib.rs lines 382–410 and 1051–1065,
gpt-oss-xe/src/lib.rs lines 627–760, and gpt-oss-xe/src/opencl.rs lines
580–1060 as of the base revision.

### 3.1 Routing and bucket construction

moe_batch receives row-major Vec<Vec<bf16>> post-attention normalized rows.
Each row is 2,880 BF16 hidden elements.

The router projection accumulates in FP32, then is rounded to BF16. Stable top
four selection is followed by FP32 softmax; each routing weight is immediately
rounded to BF16 and stored as an FP32 value containing an exact BF16 number.
CpuRoute values are stable-sorted by expert. This preserves source-row and
top-k rank order within each expert bucket. Bucket construction currently
clones each routed BF16 input row.

Decode uses moe_one rather than moe_batch. It prepares the input once, then
executes the four selected experts sequentially on CPU. The current production
Xe condition is prefill-only and M>=4, so current decode never enters Xe.

### 3.2 Types, layouts, and shapes

| Object | Current representation and layout | Lifetime/owner |
| --- | --- | --- |
| Expert input | BF16, row-major M x 2,880 | host bucket |
| Primary Q8 | scale FP32 + signed i8[32] | host preparation |
| Residual Q8 | second scale FP32 + signed i8[32] | host preparation |
| Xe activation record | primary i8[32], residual i8[32], two FP32 scales; 72 bytes, aligned to 8 | host staging/device buffer |
| Gate/up weights | MXFP4 E2M1 packed nibbles + E8M0 scale, N=5,760, K=2,880, 90 blocks | model/host repack/device |
| Down weights | same representation, N=2,880, K=2,880, 90 blocks | model/host repack/device |
| CPU weight layout | InterleavedSplitX8V2 | long-lived model |
| Xe v2 weight layout | output tile, K block, 17 planes, 32 lanes | streaming or cache |
| Bias | checkpoint BF16 converted and stored as FP32; gate 5,760, down 2,880 | model/device |
| Projection accumulator | FP32, initialized from bias | CPU or Xe |
| Gate/up output | FP32 until mandatory BF16 roundtrip | CPU after projection |
| SwiGLU output | FP32 values whose bytes encode BF16-rounded values | CPU |
| Down output | FP32 until mandatory BF16 roundtrip | CPU after projection |
| Route weight | FP32 containing BF16-rounded softmax weight | host |
| MoE output | rank-ordered FP32 accumulation, then BF16 | host |

A gate/up expert resident entry is 8,812,800 weight bytes plus 23,040 bias
bytes, 8,835,840 total. A down entry is 4,406,400 plus 11,520 bytes,
4,417,920 total. A gate/down pair is 13,253,760 bytes, exactly the prior
isolated cache high-water.

### 3.3 Residual-Q8 algorithm

For each 32-value BF16-derived FP32 block:

1. Find max(abs(x)).
2. Set scale = max / 127.0 in FP32.
3. Set inverse scale to 0 for zero scale, otherwise the FP32 reciprocal.
4. Quantize each value with Rust round, which is half away from zero, clamp to
   [-127, 127], and cast to i8.
5. Form each FP32 residual as x - q * scale.
6. Repeat the same algorithm for the 32 residuals.

Non-finite inputs are rejected. This byte representation is consumed by both
the CPU and Xe projection paths.

### 3.4 MXFP4 projection accumulation

Canonical checkpoint blocks store two E2M1 values per byte, low nibble first,
and one E8M0 scale per 32 K elements. The exact doubled E2M1 integer table is:

~~~text
0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
~~~

The projection starts each output with its FP32 bias. For each K block, in
ascending order, it adds the primary Q8 contribution and then the residual-Q8
contribution. The OpenCL source disables FP contraction and follows this
ordering. Current selected real-tensor X8 cases reached identical BF16
boundaries, and happened to be zero ULP in the tested projection outputs.

### 3.5 Mandatory BF16 and SwiGLU semantics

After gate/up projection, every FP32 value is converted to BF16 with
round-to-nearest-ties-to-even and expanded back to FP32.

For each interleaved gate/up pair:

~~~text
gate        = min(gate, 7)
up          = clamp(up, -7, 7)
scaled_gate = BF16(gate * 1.702) -> FP32
sigmoid     = BF16(1 / (1 + exp(-scaled_gate))) -> FP32
glu         = BF16(gate * sigmoid) -> FP32
linear      = BF16(up + 1) -> FP32
activated   = BF16(glu * linear) -> FP32
~~~

The batch code converts activated to BF16 again for down input; this is
idempotent but remains part of the current trace contract. Down projection is
then rounded to BF16.

### 3.6 Route weighting and unroute accumulation

The down BF16 value is multiplied by the BF16 route weight in FP32. Expert
outputs are stored by original top-k rank, not execution order, and added to
the row output in rank order. The accumulated MoE row is rounded to BF16
before the residual addition. The first fused implementation must return the
same per-expert down result to this existing host logic; route weighting is not
part of the first Xe pipeline.

### 3.7 Current split Xe execution graph

~~~text
HOST
  expert BF16 rows
    |
    | residual-Q8 preparation
    v
  ActivationRecordV2[M * 90] -------------------------.
    | blocking clEnqueueWriteBuffer                    |
    | optional CPU repack + blocking weight/bias writes
    v                                                  |
DEVICE                                                 |
  gate/up projection kernel                            |
    |                                                  |
    '-> terminal event -> clWaitForEvents --------------'
    |
    | blocking clEnqueueReadBuffer, FP32 M * 5,760
    v
HOST
  BF16 round gate/up
  five-boundary BF16 SwiGLU
  convert activated values to BF16
  residual-Q8 preparation
    |
    | blocking activation write, M * 90 * 72
    | optional down weight/bias writes
    v
DEVICE
  down projection kernel
    |
    '-> terminal event -> clWaitForEvents
    |
    | blocking FP32 read, M * 2,880
    v
HOST
  BF16 round down
  BF16 route weighting, stable rank accumulation, BF16 MoE output
~~~

Per row, excluding weights, the split path transfers 6,480 bytes of gate
activation records to the device, reads 23,040 bytes of gate FP32 output,
writes 6,480 bytes of down activation records, and reads 11,520 bytes of down
FP32 output. It also performs two terminal host waits.

### 3.8 Current OpenCL ownership and lifetime

The runtime owns one context, one profiling-enabled command queue, one program,
three kernel handles, fixed streaming weight/bias/activation/output buffers,
and an optional LRU of weight/bias buffer pairs. The queue does not request
out-of-order execution and is therefore in order.

The Xe engine is behind a mutex. One projection call owns the runtime until
its terminal wait and readback complete. Kernel launches pass no wait list.
Every host write and read is blocking. run_terminal_event explicitly waits on
each projection kernel. clFlush is not loaded. clFinish is used for draining,
cache eviction/fault handling, and shutdown.

Fixed device buffers live from engine attach to shutdown. Resident weight
buffers live from cache insertion to deterministic eviction or shutdown.
However, each chunk allocates a host staged activation vector and staged FP32
output vector. cpu_runner also allocates ActivationRecordV2 vectors per Xe
projection. Those per-call allocations are implementation boundaries, not
model semantics.

## 4. Semantic versus accidental boundaries

| Boundary | Semantic? | Required treatment |
| --- | --- | --- |
| Input is BF16 | yes | preserve exact bits |
| Input residual-Q8 algorithm | yes for current residual-Q8 mode | CPU preparation may remain; bytes must match |
| Bias-first, block-order primary-then-residual accumulation | yes | preserve |
| Gate/up FP32 result before conversion | diagnostic; BF16 result is authoritative | require exact BF16 bits and track FP32 ULP/bit result |
| Gate/up BF16 round | yes | explicit exact conversion |
| Five internal SwiGLU BF16 rounds | yes | explicit exact conversion after each operation |
| Down residual-Q8 bytes | yes | device K2 must match |
| Down FP32 result before conversion | diagnostic; BF16 result is authoritative | require exact BF16 bits |
| Down BF16 round | yes | preserve |
| BF16 route weights and rank order | yes | keep on host initially |
| Gate output readback | no | remove |
| CPU SwiGLU ownership | no | move to K2 |
| Down activation upload | no | remove |
| Per-projection terminal waits | no | replace with one terminal wait |
| Blocking weight/input upload | no, but safe first implementation | optimize only after baseline |
| Per-call staging allocations | no | replace with bounded reusable storage |

## 5. Candidate execution organizations

### A1: current split control

A1 is the production control just reconstructed. It is testable and has exact
full-model behavior, but necessarily reads gate/up to the host, performs K2
work on CPU, uploads the down records, and synchronizes twice.

### A2: three chained kernels — recommended

~~~text
blocking or event-owned input upload
  -> K1 projection-v2 gate/up
  -> K2 exact BF16/SwiGLU/residual-Q8
  -> K3 projection-v2 down
  -> terminal readback/wait
~~~

K1 writes M x 5,760 FP32 values. K2 uses one workgroup per
(row, 32-activation block), reads 64 interleaved gate/up values, performs the
five BF16 boundaries, reduces the primary and residual maxima, and writes one
72-byte ActivationRecordV2. K3 consumes those records directly.

K2 should use 32 work-items and subgroup/local reductions. The research probe
uses one serial work-item per block only to establish semantics; its timing is
not an optimized K2 performance result.

A2 preserves simple operator boundaries, keeps projection kernels reusable,
makes intermediate tracing straightforward, and needs no backend rewrite.

### A3: two kernels or a monolith

A two-kernel variant that combines gate/up and K2 is plausible. One workgroup
would produce two 32-output gate tiles representing 32 interleaved activation
pairs, synchronize within the group, apply SwiGLU, reduce, and emit one down
activation record. This could remove a launch and the 23,040-byte-per-row
global gate intermediate. It also changes the existing gate kernel workgroup
shape, doubles the output footprint handled together, raises registers and
private state, and complicates M2/M4 reuse. It is a second-stage experiment,
not the first production organization.

Combining down in the same kernel is rejected for the first implementation.
Down needs all 90 activated K blocks for every output tile. There is no
cross-workgroup barrier, and even one row's 6,480-byte residual-Q8 state plus
gate working state scales poorly across M. A single-workgroup M=1 design would
discard most Xe parallelism; recomputation would multiply gate cost.

| Concern | A2 three kernels | A3 gate+prep, then down | One giant kernel |
| --- | --- | --- | --- |
| Global synchronization | queue order | workgroup-local for gate pair | unavailable across gate tiles |
| Gate intermediate | 23,040 bytes/M | eliminated | eliminated |
| Launches | 3 | 2 | 1 |
| Register/private pressure | moderate and isolated | higher | very high |
| Local memory | small K2 reduction | larger paired-tile state | large/full activation state |
| Existing projection reuse | complete | down only | little |
| Numerical tracing | direct | harder | hardest |
| Compilation/codegen risk | low | medium | high |
| Recommendation | implement first | benchmark later | defer/reject |

## 6. Numerical-semantics research

### 6.1 Research probe

The research-only fused_semantics.cl and fused-semantics command do not alter
production dispatch, production source hashes, or numerical behavior. The
probe covers 34 actual-width rows: 195,840 gate/up conversions, 97,920
activations, and 3,060 residual-Q8 blocks. It includes an all-zero block,
limit/extreme anchors, and fixed-seed structured/random values.

It tests:

- explicit BF16 round-to-nearest-ties-to-even bit conversion;
- native OpenCL exp over every one of the 65,280 finite BF16 inputs;
- native-exp and immutable CPU-LUT SwiGLU variants;
- exact primary and residual Q8 value bytes;
- exact FP32 scale bits.

An exploratory dirty-tree run without correctly rounded division found the
first real semantic hazard: 95 primary scale bit mismatches, 241 residual
scale bit mismatches, four primary value mismatches, and 17 residual value
mismatches across 3,060 blocks.

The clean committed run compiled K2 with:

~~~text
-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt
~~~

It then recorded zero mismatches at every boundary. The OpenCL specification
defines this option specifically to require correctly rounded FP32 x/y and
1/x, and requires device capability before the build may succeed.

### 6.2 Boundary answers

| Boundary | Exact bytes on pinned Xe/runtime? | Mechanism |
| --- | --- | --- |
| FP32 to BF16 | yes, 0/195,840 mismatches | explicit integer bit helper matching half 2.7.1 |
| scaled gate BF16 | yes | same helper after FP32 multiply |
| sigmoid BF16 using native exp | yes, 0/65,280 finite-domain mismatches | exhaustive attach/certification test |
| sigmoid BF16 using CPU LUT | yes | immutable 65,536 x u16 table, 128 KiB |
| complete SwiGLU BF16 | yes, 0/97,920 mismatches for both variants | explicit boundary after each operation |
| primary Q8 scale/bytes | yes with correctly-rounded division | dedicated K2 build options |
| residual FP32/Q8 scale/bytes | yes with correctly-rounded division and FP contraction off | exact operation order |
| gate/down projection BF16 | existing X8 evidence says yes for selected shapes | reuse projection v2 |
| complete fused expert | not yet demonstrated | next implementation must chain and compare |

Native exp matched the complete finite BF16 domain on this driver, stronger
than a random sample. For robustness, the first implementation should retain
both exact mechanisms behind a forced research selector: CPU-authored LUT and
native exp guarded by the exhaustive startup self-test. Select one immutable
production behavior only after complete-expert performance measurement. The
LUT is the conservative correctness fallback and consumes only 128 KiB.

K2 must be a separate program so the correctly-rounded division option does
not silently alter or slow the already-certified projection program. Its
source, build options, driver identity, and native binary must enter artifact
identity.

### 6.3 Remaining numerical work

The semantic probe is intentionally not a fused projection benchmark. The
next implementation must still prove K1 output -> K2 bytes -> K3 output as one
pipeline with real checkpoint tensors. An optimized 32-work-item K2 must
repeat the exact probe because changing reduction organization can change
bytes if it changes operations.

## 7. M=1 decode research

Current production Auto is the AVX2 x8 expert path. Current production Xe
rejects decode before its M threshold, so there is no current production
complete-expert Xe decode number.

Historical X8 measured a resident-weight, transfer-inclusive layer-0/expert-0
gate/up projection:

| M | Xe tile32-m1-v2 total | Device | Inferred AVX2 control from reported ratio |
| ---: | ---: | ---: | ---: |
| 1 | 1.129 ms | 1.046 ms | approximately 1.766 ms |

The 1.564x ratio is useful evidence that K1 can be competitive. It is not a
complete expert result: it excludes down, route weighting, full-token
scheduling, and realistic cross-layer weight-cache behavior.

A fused resident M=1 pipeline could plausibly remove one wait, a 23,040-byte
gate read, a 6,480-byte down upload, and CPU SwiGLU/Q8 work. A streaming M=1
pipeline must additionally move a 13,253,760-byte gate/down pair, which is
likely dominant. The full-model cache's sequential zero-hit result warns that
a 128–512 MiB LRU may thrash across layers. Decode routing may have different
temporal locality, but this has not been measured.

The likely crossover is therefore not declared. The pre-registered decision
is:

- compare CPU Auto M=1 with current split Xe streaming, split Xe resident-hit,
  three-kernel streaming, and three-kernel resident-hit;
- measure one complete expert and a complete decode token;
- record cache working-set distance across layers and tokens;
- promote decode only if realistic bounded residency produces repeated hits
  and the full-token lower confidence bound clears the promotion gate;
- otherwise keep all M=1 decode on CPU.

## 8. Large-M prefill research

The corpus contains 260 distinct M values from 1 through 412. Exact observed
M>=16 values are listed in Appendix A. Both gate/up and down saw the same
40,188 bucket count and the same M support.

Historical selected tile32-m4-v2 gate/up controls were:

| M | Xe total | AVX2/Xe |
| ---: | ---: | ---: |
| 4 | 1.656 ms | 2.972x |
| 8 | 2.620 ms | 3.782x |
| 16 | 5.992 ms | 3.292x |
| 32 | 11.386 ms | 3.467x |
| 64 | 21.584 ms | 3.664x |
| 128 | 43.571 ms | 3.634x |

These isolated numbers show strong projection headroom but cannot select a
pipeline threshold. The completed full-model explicit Xe/cache result was
negative, and every cache size produced zero useful hits.

The future scan must use both real roles:

~~~text
gate/up: M x 2,880 -> M x 5,760
down:    M x 2,880 -> M x 2,880
~~~

Compare CPU Auto, forced AVX2 matrix, forced AVX-512/VNNI matrix, current
explicit Xe split, current resident Xe split, fused streaming Xe, and fused
resident Xe. Scan every actual observed M, then concentrate repetitions around
any crossing. A single threshold is admissible only if complete-expert
behavior is monotonic enough across both roles and relevant weight states.
Otherwise the policy may depend on phase, M, and pair-ready residency state.
Projection role should not independently dispatch once the unit of selection
is the whole expert pipeline.

No large-M crossover is selected in this research pass.

## 9. Residency and persistent memory

The negative weight-cache result does not apply to all persistence classes.

| Object | Persistence class | Recommendation |
| --- | --- | --- |
| Context, queue, programs, kernels | runtime | retain for engine lifetime |
| Gate/down packed weights and bias | weight residency | keep existing strict bounded cache; treat pair state explicitly |
| Input/down ActivationRecordV2 buffer | workspace | allocate once, reuse contents |
| Gate FP32 intermediate | workspace/intermediate | allocate once; never read to host normally |
| Down FP32 output | workspace | allocate once |
| Sigmoid BF16 LUT, if selected | immutable runtime data | one 128 KiB device buffer |
| Kernel arguments | setup metadata | stable handles; set changing scalars each dispatch |
| Profiling events | command lifetime | recycle/release after terminal query |
| Host input/output staging | host workspace | bounded reusable vectors, no per-chunk allocation |
| Gate and down intermediates | execution-only residency | device-only until expert completion |

The input activation record buffer and K2 down-record output have identical
6,480-byte-per-row extents. Because K1 has finished consuming input before K2
writes, the same cl_mem may be reused for both in an in-order queue. The
minimum device workspace per row is then:

~~~text
ActivationRecordV2 input/down reuse   6,480 bytes
gate/up FP32 intermediate            23,040 bytes
down FP32 output                     11,520 bytes
                                      ------------
total                                 41,040 bytes per row
~~~

At M=128 this is 5,253,120 bytes, plus the optional 128 KiB sigmoid table and
small metadata. A trace/debug mode may allocate separate BF16/SwiGLU buffers,
but normal execution should not.

Weight residency should become pair-aware for the pipeline: gate/up and down
must either both be addressable before submission, or the request follows a
well-defined mixed/streaming state. Do not partially commit a cache insertion
and then discover that the second role cannot fit. The existing independent
role identities remain useful, but the orchestration layer should report
pair state as cold, mixed, or hit.

Prefill's sequential working set made ordinary LRU residency ineffective.
Do not increase the default cache or claim full-model residency. M=1 decode
must measure reuse distance before choosing any auto-resident budget.

## 10. Expert scheduling

The first fused implementation should execute one expert bucket at a time.
This preserves the existing stable expert grouping, needs one workspace, and
keeps failure recomputation simple.

For each bucket:

1. Resolve or stream both role weights.
2. Upload the CPU-prepared input activation records.
3. Enqueue K1, K2, and K3.
4. Read the final down result once.
5. Commit the result to routed_outputs only after successful completion.
6. Let the unchanged host code weight and accumulate by original rank.

The internal API should return a pending/terminal result object rather than
baking an immediate host wait into every kernel helper. That preserves a seam
for later queueing of multiple buckets and delayed unroute without implementing
broad concurrency now.

A second performance phase may test:

- a two-slot workspace ring;
- preparing/prefetching the next expert pair while the current bucket runs;
- batching several complete pipelines before readback;
- delaying all readbacks until a layer's selected expert buckets finish.

Use one queue first. Multiple queues and out-of-order execution add explicit
dependency and buffer-lifetime complexity and have no current evidence.
Concurrent small CPU and large Xe expert execution remains deferred because
shared bandwidth contention is severe.

## 11. OpenCL execution mechanics

The current queue has only CL_QUEUE_PROFILING_ENABLE, so it is in order. The
OpenCL execution model guarantees that K2 sees K1's writes and K3 sees K2's
writes when the kernels are enqueued in that order on this queue. No
host-side event wait is required between them.

The driver advertises cl_intel_unified_shared_memory, but production does not
load or use its APIs. The current 16 MiB probe found shared allocation cheap
while first-write/read were about 3.594/5.071 ms, versus about 1.747/1.006 ms
for ordinary device buffers. That does not justify SVM in the baseline. The
reported extension list does not advertise cl_khr_command_buffer.

Recommended conservative submission:

~~~text
blocking input/weight writes for baseline
enqueue K1 with profiling event
enqueue K2 with profiling event
enqueue K3 with profiling event
enqueue final read
wait only for final read/event
query all completed event timestamps
release events
~~~

The initial writes may remain blocking in phase one. Converting them to
nonblocking writes requires owned/pinned staging whose lifetime extends to its
completion event; it is a later measured optimization. clFlush is only needed
if the host will continue useful work and wants prompt device start. A final
blocking read already submits prior commands. clFinish remains a drain/error/
shutdown primitive, not a per-kernel primitive.

| Mechanism | Current | Fused baseline |
| --- | --- | --- |
| Queue | one in-order profiling queue | unchanged |
| Weight/input writes | blocking | blocking first, event-owned later |
| K1 completion | host clWaitForEvents | device queue dependency |
| Gate read | blocking host read | removed |
| K2 | CPU | device kernel |
| Down activation write | blocking | removed |
| K3 completion | host clWaitForEvents | one terminal completion |
| Down read | blocking | one final read |
| Workspace | fixed projection buffers plus per-call Vec | persistent pipeline buffers |
| Kernel/program creation | attach-time | add attach-time K2 program/kernel |
| clFlush | unavailable | load only if asynchronous host work is measured |
| Mapping/SVM | unused | defer; no demonstrated advantage |
| Command buffers | unused | defer; mutable per-expert weights/rows reduce immediate value |

This is a queue/orchestration extension to the current backend, not a complete
runtime rewrite.

## 12. Package, memory, and thermal behavior

The current 128 MiB memory probe measured CPU memcpy median 7.836 ms alone and
48.955 ms while an OpenCL GPU roundtrip ran concurrently: CPU-side effective
bandwidth fell from about 17.1 GB/s to 2.74 GB/s. The Level Zero probe showed a
similar fall, 7.959 ms to 53.313 ms. Earlier X8 diagnostics reached the
configured 1,300 MHz GPU maximum and likewise found strong concurrent
bandwidth loss.

This evidence argues against CPU/Xe cooperative expert execution in the first
implementation. It also explains why eliminating host/device intermediate
traffic can matter on an integrated GPU even though the physical memory is
shared.

No optimized fused kernel exists yet, so there is no honest fused sustained
thermal result. Promotion must collect CPU and GPU frequency, CPU utilization,
GPU busy time, temperature, and package power where readable for:

- CPU complete experts;
- Xe split complete experts;
- Xe fused complete experts;
- short cold bursts;
- sustained full requests after thermal equilibrium.

Never disable thermal or power protection. Any region whose win reverses
under sustained operation remains forced-only.

## 13. Level Zero reconstruction and classification

The expected research corpus at /home/emmy/src/xe-research was missing. It was
reconstructed outside Git from immutable upstream revisions:

| Corpus | Pinned revision |
| --- | --- |
| Khronos OpenCL-Headers | c9c8ccfab584f9f7610057c4633dbd3df7e012cc |
| oneAPI Level Zero | v1.28.2 / 6369d8d642e9c7625e67f38664267f171b8e42dc |

The expected include topology was restored without changing system packages.
An external Clang/LLVM 18.1.8 sysroot and wrapper rebuilt the same OpenCL
SPIR-V. Key header hashes were:

The installed libze1 package is 1.28.2-2 and the Intel OpenCL/Level Zero GPU
runtime is 26.05.37020.3-1, so runtime versions are equivalent to or newer
than the pin. Development headers and llvm-spirv tooling were not installed
system-wide; there was therefore no system header corpus that could honestly
replace the missing immutable source pin.

| Header | SHA-256 |
| --- | --- |
| CL/cl.h | 16d09614cd7eef73b4094089cc4ce2af777181d3ffe9613473fb6d552234a4d1 |
| CL/cl_platform.h | b125cece6fe41f2e6690d88b24652c151cb6654da13b42180d69d5bd9ed526d7 |
| level_zero/ze_api.h | 252c2f3138853c632763fc381b2814118938f874e15d11c3812f683cdd59410d |
| level_zero/ze_ddi.h | b445084d996f5662901fdd0dd8cb99faebaca03c8f32fe692438c9ad15256574 |

The capability probes passed for OpenCL, regular Level Zero lists, and
immediate Level Zero lists. A 1,508-byte same-SPIR-V i32 kernel produced all
4,096 exact outputs under all three paths. OpenCL and Level Zero compiled that
SPIR-V to identical 6,872-byte native bytes with SHA-256
df14b173b2cbed54f9ffe72654d8ea872785b291518ecabdc93312986fe6290e.

Smoke timing, 30 warm samples:

| Path | Warm host median (95% bootstrap interval) | Device median |
| --- | ---: | ---: |
| OpenCL same SPIR-V | 33.692 us (32.768–35.551) | 5.833 us |
| Level Zero regular | 31.203 us (30.995–32.098) | 7.384 us |
| Level Zero immediate | 33.708 us (33.352–34.135) | 6.656 us |

The immediate manifest's internal variant label still says regular; session
identity confirms immediate=true. This label defect limits the artifact to
smoke evidence. Earlier X6 same-projection regular/immediate campaigns are the
stronger result and also found no material immediate-list advantage.

Device, host, and shared Level Zero allocations all worked. Shared allocation
functionality did not establish zero-copy or a pipeline advantage. At 16 MiB,
Level Zero shared first write/read/reuse timings were about
2.424/1.708/2.506 ms, while OpenCL device-buffer write/read/reuse was about
1.747/1.006/0.921 ms in the current probes. These are memory-class probes, not
a complete expert comparison.

Classification: L0-A. The runtime works, supports persistent/shared
allocations, regular and immediate lists, event chaining, and the same
SPIR-V. It does not materially improve the required topology over one OpenCL
in-order queue, and immediate submission was not faster in the smoke or prior
projection evidence. Level Zero is excluded from the next implementation
goal. No SYCL or oneAPI-rs dependency is warranted.

## 14. Reference-source ledger

| Source | Revision/version | License/status | Concept used | Use |
| --- | --- | --- | --- | --- |
| Khronos OpenCL specification, registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html | live unified spec consulted 2026-08-13 | specification terms | in-order dependencies, blocking transfer semantics, events, correctly-rounded divide option | studied only |
| Khronos OpenCL-Headers, github.com/KhronosGroup/OpenCL-Headers | c9c8ccfab584f9f7610057c4633dbd3df7e012cc | Apache-2.0 | reproducible FFI declarations | pinned headers |
| oneAPI Level Zero, github.com/oneapi-src/level-zero | 6369d8d642e9c7625e67f38664267f171b8e42dc | MIT | header corpus and API declarations | pinned headers |
| Level Zero core programming guide, oneapi-src.github.io/level-zero-spec | v1.28-era API, latest guide cross-check | specification | immediate/regular lists, recycling, events, shared allocations | studied only |
| Intel compute-runtime, github.com/intel/compute-runtime | installed 26.05.37020.3 stack | MIT | OpenCL/Level Zero implementation context | studied only |
| half crate | 2.7.1 | Apache-2.0 OR MIT | exact f32-to-BF16 bit behavior | algorithm mirrored in research helper |
| llama.cpp ggml-quants.c | 030ebb558a5820b444a8f836ed5cdd46c9b4bd7a | MIT | symmetric Q8 rounding contract | existing repository adaptation |
| mistral.rs GPT-OSS/MXFP4 | 8010b6a0578e416120b590ed72fd46ed5f24ee85 | MIT | SwiGLU and MXFP4 semantic cross-check | existing repository cross-check |
| ik_llama.cpp | 26ceed… as recorded in prior Xe docs | MIT | integer-dot organization reference | studied previously |

No broad dependency or source implementation was copied. The new kernel is a
small repository-native research probe.

## 15. Prototype and artifact results

External roots:

~~~text
/home/emmy/gpt-oss-rs-artifacts/xe-fused-expert-research/
  0113e8214e765d168216bbee2120654555a4cfe4/
  9961137e34283e195546b3e6134e54ed7e7352cc/
~~~

The base-revision root contains the reconstructed OpenCL/Level Zero
capability, same-SPIR-V, and memory evidence. Its selected SHA index is
SHA256SUMS, hash d13db3449898d276d1288ac3f6a8ebf4b67b2672d19e899a795168479c174387.

The committed-prototype root contains the clean F1 semantic evidence. Its
SHA256SUMS hash is
681e8484da6ba4a458ad6418af8f457dce086d9c23c5c474076f859ddc735364.

Key F1 files:

| Artifact | SHA-256 |
| --- | --- |
| F1 manifest | 99a0ea4cbf2ddde450cf5cd4fc7db7511eccaf7e0ec9aa51e0a2e539f095021c |
| Raw semantic result | dda1daf48844ec7d4280c770e084e60aae6c6dce746e87c04943e0197d57ce39 |

The F1 device timing of the deliberately serial, dual-path diagnostic K2 was
about 1.5 ms for 34 rows in one sample. It computes both native and LUT paths,
writes trace buffers, and uses one work-item per block; it must not be used as
a performance forecast.

## 16. Proposed production architecture

### 16.1 API and ABI

Keep projection ABI v2 immutable. Add:

~~~text
gpt-oss-rs.xe-expert-pipeline-abi/v1
~~~

It should identify:

- ActivationRecordV2 byte layout and alignment;
- gate intermediate FP32 layout M x 5,760;
- K2 arguments and scalar dimensions;
- down output FP32 layout M x 2,880;
- BF16 helper version;
- sigmoid mode/LUT identity;
- K2 build options;
- K1/K3 projection source and ABI hashes;
- driver/device/native artifact identity.

Add an internal ExpertPipelineRequest containing gate/down cache identities,
both packed weight/bias sources, M/K/N, prepared input records, and requested
semantic/trace mode. Return an owned result only after the terminal operation
succeeds.

### 16.2 Buffers and kernels

- K1: existing tile32-m1-v2, tile32-m2-v2, or tile32-m4-v2 selected by M.
- K2: new exact preparation kernel, 32 work-items per activation block,
  explicit BF16 helpers, FP_CONTRACT OFF, correctly rounded FP32 divide.
- K3: existing tile projection selected by M.
- Reuse one activation record buffer for K1 input and K3 input after K1.
- Keep one gate FP32 intermediate and one down FP32 output buffer.
- Allocate a separate trace buffer only for correctness tooling.
- Keep host staging bounded to maximum configured chunk rows.
- Preserve current chunk validation and checked extent arithmetic.

### 16.3 Residency and policy inputs

Reuse the existing cache key fields and add pair-state reporting. Do not make
weights resident by default. Workspace is always persistent once explicit Xe
attaches. Weight residency stays strict, byte-bounded, and user-configured
during forced research.

M=1 decode and prefill M thresholds are independent. Unknown phase, unknown
hardware/runtime identity, insufficient workspace, cache inconsistency, or any
Xe failure chooses CPU.

### 16.4 Affected repository areas

Expected files:

- crates/gpt-oss-xe/src/lib.rs: request/result types, pipeline orchestration,
  lifecycle, metrics, circuit breaker, ABI validation.
- crates/gpt-oss-xe/src/opencl.rs: K2 program/kernel, persistent workspace,
  in-order chained submission, event profiling, pair-aware weight handling.
- crates/gpt-oss-xe/kernels or the existing embedded kernel location: K2 exact
  source.
- crates/gpt-oss-xe/fixtures: expert-pipeline ABI v1 and hash tests.
- crates/gpt-oss-xe/promotion-record.json: remain automatic_enabled=false
  through forced phases; update only after certification.
- crates/gpt-oss-model-runner/src/cpu_runner.rs: forced whole-expert selection,
  transactional result integration, unchanged route accumulation.
- crates/gpt-oss-bench and tools/xe-research: complete-expert benchmark and
  trace/differential modes.
- docs/CPU_RUNTIME.md and Xe reports: forced configuration, evidence, and
  eventual promotion result.

## 17. Benchmark design

### 17.1 Candidate matrix

Decode M=1:

| Candidate | Weight state | Workspace |
| --- | --- | --- |
| CPU Auto AVX2 x8 | host | CPU |
| Current Xe split | streaming | current |
| Current Xe split | resident hit | current |
| Three-kernel Xe | streaming | persistent |
| Three-kernel Xe | resident hit | persistent |

Small prefill must retain explicit controls at M=2, M=3, every observed M in
4–7, and every observed M in 8–15. These regions remain CPU unless evidence
clearly says otherwise.

Large prefill must scan every Appendix A M for:

- CPU Auto;
- forced AVX2 matrix;
- forced AVX-512/VNNI matrix;
- current split Xe streaming;
- current split Xe resident;
- fused Xe streaming;
- fused Xe resident.

### 17.2 Per-case schema

Every retained sample must record:

~~~text
phase
projection roles and complete-expert identity
M, gate N/K, down N/K
requested implementation
effective implementation
weight state: cold/miss/mixed/hit/bypass
workspace state: cold/warm
host preparation time
weight repack time
weight/bias upload time
activation upload time
argument setup time
K1 host submission and device time
K2 host submission and device time
K3 host submission and device time
terminal synchronization time
readback time
total complete-expert time
full-token or full-request effect
bytes uploaded/read/written and intermediate bytes avoided
RSS and Xe resident/workspace high-water
CPU/GPU utilization, temperature, frequency, and package power when readable
fallback/circuit-breaker state
output comparison
~~~

### 17.3 Protocol

- Identical fixed-seed inputs and real checkpoint tensors.
- Correctness before timing.
- At least ten warmups.
- Rotated CPU/split/fused candidate order.
- Three independent trials with at least 30 retained samples per isolated
  candidate where duration permits.
- Cooldown and predeclared thermal validity bounds.
- Process-cold and process-warm workspace/program cases.
- Explicit cache cold, miss, mixed, and hit cases.
- Bootstrap median confidence intervals and paired full-request ratios.
- Dense repeats around any inferred M crossing; never derive a threshold from
  one run.
- Full decode token latency/tokens per second for M=1.
- Full-model TTFT/request timing for large prefill.
- All seven representative pinned fixtures, not one friendly prompt.
- Profiler-disabled confirmation after profiling has selected a candidate.

Current projection controls may guide experiment sizing but cannot substitute
for the complete-expert campaign.

## 18. Correctness and certification plan

### 18.1 Operator gates

Trace and compare:

1. CPU-prepared input ActivationRecordV2 bytes.
2. K1 FP32 gate/up output: bit/ULP report, mandatory BF16 bytes exact.
3. Every SwiGLU intermediate BF16 bit vector.
4. Activated BF16 bytes exact.
5. Primary and residual Q8 values and FP32 scale bits exact.
6. K3 FP32 down output: bit/ULP report, mandatory BF16 bytes exact.
7. Final expert BF16 output exact.
8. Route-weighted and rank-accumulated MoE BF16 output exact.

Debug trace reads may synchronize and allocate extra storage; they must be
compiled or selected out of performance measurements.

### 18.2 Differential matrix

Required core M values:

~~~text
1, 2, 3, 4, 8, 16, 32, 64, 128
~~~

Add every important actual corpus M, with complete correctness scans over all
actual M before threshold selection. Cover Q8 and residual Q8, bias/no bias,
zero blocks, extrema, gate/up and down tails, multiple experts/layers,
repeated execution, cache hit/miss/mixed/eviction/bypass, chunking, forced
allocation and kernel failure, circuit-breaker fallback, shutdown, and fresh
restart.

Projection dimensions have no K tail in the model, but ABI negative tests must
reject non-32 K tails and malformed extents before FFI.

### 18.3 Full-model authority

Do not add redundant Xe labels to the authoritative 42-cell CPU matrix.
Create a sibling explicit-Xe certification campaign against normal pinned
PyTorch CPU authority:

- seven fixtures x forced fused decode candidate;
- seven fixtures x forced fused prefill candidate;
- seven fixtures x combined candidate if both independently pass.

That is up to 21 candidate cells sharing seven fresh authority captures.
Streaming/resident state is tested at operator/lifecycle level and should only
be duplicated at full-model level if it changes real execution coverage.
Every generated token must match exactly. Preserve the fresh-oracle
provenance, empty-root, image, host, failure-accounting, and hash discipline.
Do not edit the published CPU oracle lock/image unless oracle inputs truly
change.

## 19. Failure, fallback, and promotion gates

The fused pipeline writes only private device/host staging until completion.
No partial K1/K2/K3 output may enter routed_outputs. On the first Xe failure:

1. drain/release affected events safely;
2. invalidate unsafe cache/workspace state;
3. recompute the complete expert once on CPU;
4. open the existing process/model circuit breaker;
5. use CPU for subsequent eligible experts;
6. expose the failure and fallback in profiler/metrics.

Decode and prefill promotion decisions are independent.

A region may become automatic only after all of these pass:

- exact intermediate bytes at every semantic boundary;
- exact full-model generated-token parity against fresh normal PyTorch CPU;
- validated ABI/source/build/native/driver identity;
- clean malformed-artifact and unsupported-device rejection;
- one CPU recomputation with no partial commit;
- bounded workspace and cache memory;
- repeat/evict/restart/shutdown lifecycle stability;
- repeated complete-expert win;
- conservative 95% lower confidence bound at or above a pre-registered 1.05x
  end-to-end token/request speedup;
- no representative fixture lower bound below parity;
- no material TTFT regression for decode promotion or decode regression for
  prefill promotion;
- no sustained thermal reversal;
- no profiler-disabled regression;
- reproducibility after process restart.

The 1.05x gate encodes the requested clear several-percent end-to-end win and
is deliberately more conservative than fractional prior regressions. If the
campaign cannot resolve that gate, the region remains forced-only.

Possible final policy, only if measured:

~~~text
decode and M==1 and validated fused profile and required pair state:
    fused Xe
prefill and M>=validated_threshold and validated fused profile:
    fused Xe
otherwise:
    CPU Auto

any Xe failure:
    one CPU recomputation, then circuit breaker
~~~

Do not add general online autotuning. Encode only predeclared, evidence-backed
regions for the exact profile; unknown state is CPU.

## 20. Deferred tracks and explicit non-goals

| Track | Decision |
| --- | --- |
| True dense BF16 matrix kernels | defer; Q/O work is only 0.81% |
| CPU attention optimization | defer; attention is 0.62% |
| Xe attention | defer; same low leverage plus transfer complexity |
| LM-head fusion | defer; not implicated by the profiler |
| Cooperative CPU/Xe expert scheduling | defer; bandwidth contention is severe |
| General autotuning | defer; first establish stable phase/M regions |
| Persistent hardware profiles | defer; one exact Tiger Lake record is enough initially |
| SYCL or oneAPI-rs | explicit non-goal |
| Broad Level Zero runtime | explicit non-goal for next sprint |
| Automatic Xe before gates | explicit non-goal |
| Route weighting/unroute on GPU | defer until expert pipeline wins |
| Published oracle/image changes | explicit non-goal absent real oracle-input changes |

## 21. Implementation risks

| Risk | Mitigation |
| --- | --- |
| Correctly-rounded division slows K2 | isolate it in K2 program; optimize workgroup organization; measure |
| Device lacks required FP flag | attach fails closed to CPU before requests |
| Native exp changes with driver | exhaustive finite-BF16 attach test or immutable LUT |
| LUT hurts cache/performance | benchmark native and LUT; retain only exact winner |
| Gate intermediate bandwidth erases launch savings | establish A2 baseline, then test A3 gate+prep |
| K2 spills/private pressure | disassemble current-driver native; record SIMD/GRF/private/scratch metadata |
| M1 weight streaming dominates | require realistic pair-hit/reuse evidence or keep CPU |
| LRU thrashes across layers | report reuse distance; no automatic capacity increase |
| Buffer aliasing corrupts in-flight input | one in-order queue, explicit lifetime state, negative tests |
| Padding/chunking changes row order | checked dimensions and exact differential cases |
| Delayed readback complicates fallback | first version waits per complete expert; no partial commit |
| Thermal win reverses | sustained, cooled, order-rotated campaign |
| Full-model win differs by fixture | require all representative fixtures and no regression bound |
| Driver/native cache staleness | include every source/ABI/build/device/driver hash |

## 22. Staged next implementation

1. Add expert-pipeline ABI v1, checked dimensions, persistent workspace, and
   forced configuration without changing dispatch.
2. Implement optimized exact K2 and repeat F1 boundary tests.
3. Add in-order K1 -> K2 -> K3 submission with one terminal read, trace mode,
   and phase metrics.
4. Integrate a forced-only complete-expert call in cpu_runner with transactional
   CPU fallback and circuit breaker.
5. Add complete-expert M=1 and all-observed-M benchmark harnesses.
6. Run operator/differential/lifecycle certification.
7. Run the sibling explicit-Xe full-model correctness campaign.
8. Run isolated and full-model performance/thermal campaigns.
9. Promote decode and/or prefill only if each independent gate passes; otherwise
   publish a negative promotion record and retain forced explicit Xe.
10. Do not add Level Zero in this sprint.

## Appendix A — exact observed large-M corpus values

~~~text
16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162,
164, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
178, 180, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193,
195, 196, 197, 199, 200, 203, 204, 206, 207, 209, 210, 212, 213,
214, 216, 218, 219, 220, 221, 222, 223, 225, 226, 229, 230, 232,
234, 238, 239, 241, 242, 243, 245, 249, 250, 251, 252, 253, 254,
256, 257, 260, 262, 263, 265, 268, 269, 270, 274, 276, 277, 278,
286, 294, 295, 297, 302, 303, 310, 314, 319, 320, 332, 340, 344,
349, 350, 357, 365, 372, 391, 399, 412
~~~

## Appendix B — recommended next Codex goal, verbatim

~~~text
# Fused Tiger Lake Xe Expert Pipeline + Decode/Prefill Crossover

Work in /home/emmy/gpt-oss-rs from the reviewed X10 fused-expert research
branch/merge. Fetch and record HEAD and origin/main before edits. Read
docs/xe-research/10-fused-expert-pipeline-research.md and all evidence it
references. Do not weaken the current CPU, oracle, profiler, Xe artifact, or
failure contracts.

Objective: implement a forced-first OpenCL expert pipeline for Tiger Lake
8086:9a49 that executes gate/up projection, exact BF16/SwiGLU/residual-Q8
preparation, and down projection as three kernels on one in-order queue,
keeping every intermediate device-side and performing one final expert-output
readback. Preserve CPU Auto as the control and transactional fallback. Measure
M=1 decode and large-M prefill independently and enable automatic Xe only for
regions that pass every correctness, lifecycle, thermal, and end-to-end
performance gate.

Required execution organization:

1. Reuse the existing projection ABI v2 and selected tile32-m1-v2,
   tile32-m2-v2, and tile32-m4-v2 kernels as K1 and K3. Do not mutate their
   immutable ABI/source identity.
2. Add gpt-oss-rs.xe-expert-pipeline-abi/v1 and a K2 kernel with 32 work-items
   per (row, 32-activation block). K2 must explicitly reproduce the gate/up
   BF16 boundary, all five GPT-OSS SwiGLU BF16 boundaries, and exact primary
   plus residual Q8 records.
3. Compile K2 in a separate program with FP_CONTRACT OFF and
   -cl-fp32-correctly-rounded-divide-sqrt so projection codegen is unchanged.
   Support the CPU-authored 65,536-entry BF16 sigmoid LUT and the native-exp
   path only when its exhaustive finite-BF16 attach self-test passes. Select a
   single behavior for promotion based on exactness and complete-pipeline
   performance.
4. Use one in-order profiling queue. Enqueue K1, K2, and K3 without host waits
   between them; retain profiling events and wait/read only at the terminal
   expert boundary. Keep clFinish for drain/error/shutdown. Start with safe
   blocking input/weight writes; test nonblocking event-owned staging only
   after the baseline.
5. Allocate bounded persistent workspace at attach: one reusable
   ActivationRecordV2 buffer shared between K1 input and K3 input after K1,
   one M x 5,760 FP32 gate intermediate, and one M x 2,880 FP32 output.
   Normal workspace is 41,040 bytes per configured row plus the optional
   128 KiB LUT. Use checked extents, row chunking, and bounded reusable host
   staging. No per-chunk Vec allocation in the steady pipeline.
6. Reuse the existing strict weight cache but make complete gate/down pair
   state observable and transactional. Gate/up plus down for one expert is
   13,253,760 bytes. Do not increase the default, claim model residency, or
   assume cache hits. Measure cold, miss, mixed, hit, bypass, eviction, and
   cross-layer/token reuse distance.
7. Integrate the pipeline in cpu_runner behind an explicit forced mode first.
   Keep route selection, BF16 route weights, stable rank-order weighting, and
   unroute accumulation on CPU. Commit no partial Xe output. On one Xe
   failure, discard staging, recompute the complete expert once on CPU, and
   open the circuit breaker.
8. First schedule one expert bucket at a time. Preserve an internal pending
   result/event seam for later queued buckets, but do not implement general
   CPU/Xe concurrency, multiple queues, or broad asynchronous scheduling in
   this goal.

Correctness gates:

- Compare CPU-prepared input records, K1 FP32/mandatory BF16 output, every
  SwiGLU BF16 intermediate, activated BF16, primary/residual Q8 value bytes
  and FP32 scale bits, K3 FP32/mandatory BF16 output, final expert BF16, and
  route-weighted MoE BF16.
- Require exact bytes wherever the current contract has a BF16/Q8 boundary.
  Report FP32 bit/ULP differences and require identical downstream BF16.
- Cover M=1,2,3,4,8,16,32,64,128 and every important actual corpus M; Q8 and
  residual Q8; bias/no bias; zero/extrema; malformed tails; multiple
  experts/layers; repeats; chunking; cache states; eviction; allocation/kernel
  failure; fallback; shutdown/restart.
- Add ABI/source/build/native/driver identity tests and reject unsupported
  correctly-rounded division at attach before request execution.
- Create a sibling explicit-Xe full-model campaign, not redundant labels in
  the 42-cell CPU matrix: seven pinned fixtures for the forced decode
  candidate, seven for the forced prefill candidate, and seven for the
  combined candidate if both independently pass. Compare every generated
  token exactly with fresh normal pinned PyTorch CPU authority.

Performance corpus:

- Decode M=1: CPU Auto AVX2 x8, current split Xe streaming/resident, fused Xe
  streaming/resident. Measure complete expert, full decode token latency, and
  tokens/sec.
- Small prefill: retain controls for M=2, M=3, observed M=4-7, and observed
  M=8-15; do not assume promotion.
- Large prefill: scan every actual observed M>=16 for CPU Auto, forced AVX2
  matrix, forced AVX-512/VNNI matrix, current split Xe, current resident Xe,
  fused streaming Xe, and fused resident Xe at both actual projection roles.
  Measure complete expert and full-model TTFT/request effect.
- Record requested/effective path, phase, M/N/K, weight/workspace/warm state,
  preparation, repack, uploads, argument setup, per-kernel submission/device
  time, terminal wait, readback, total expert/request time, bytes, RSS,
  resident/workspace high-water, CPU/GPU utilization/frequency/temperature,
  package power when readable, and exact-output status.
- Use identical inputs, at least ten warmups, rotated orders, multiple trials
  and repetitions, cooldown/thermal validity, paired full requests, and
  conservative bootstrap intervals. Confirm winners with profiling disabled.

Promotion policy:

- Evaluate decode and prefill independently.
- Require exact intermediates, exact fresh full-model tokens, validated
  artifact identity, bounded memory, clean fallback/no partial commit,
  lifecycle/restart stability, repeated complete-expert wins, no sustained
  thermal reversal, no profiler-disabled regression, and reproducibility
  across all representative fixtures.
- Require the conservative 95% lower confidence bound to be at least 1.05x for
  the relevant end-to-end token/request metric and no fixture lower bound
  below parity. A microkernel/projection win is insufficient.
- Prefer a simple phase+M threshold only if complete-expert results are
  monotonic across relevant weight states. Otherwise include only the minimum
  evidence-backed phase/M/pair-ready condition. Unknown state stays CPU.
- If neither decode nor prefill passes, retain forced Xe and publish a
  negative promotion record. If one passes, promote only that region.
- Keep crates/gpt-oss-xe/promotion-record.json automatic_enabled=false until
  every gate for a region has completed.

Level Zero is explicitly deferred as L0-A. Do not add SYCL, oneAPI-rs, Xe
attention, dense BF16 kernels, LM-head fusion, general autotuning, persistent
hardware-profile machinery, route weighting on GPU, or cooperative CPU/Xe
expert scheduling.

Expected affected areas include crates/gpt-oss-xe/src/lib.rs,
crates/gpt-oss-xe/src/opencl.rs, a new exact K2 kernel and pipeline ABI
fixture, crates/gpt-oss-model-runner/src/cpu_runner.rs, benchmark/research
tooling, promotion evidence, and runtime/research documentation.

Keep large raw evidence outside Git under a root keyed by exact
commit/runtime identity and publish SHA-256 indexes. Run formatting,
git diff --check, cargo check --workspace --locked, focused unit/OpenCL tests,
differential/lifecycle tests, and the bounded candidate campaigns. Commit
coherent phases, push the branch regularly, open a PR, address CI/review, and
merge only after the final selected or negative promotion record is
reproducible from a clean tree.

Return the implementation commits, final branch/HEAD/origin state, kernel and
buffer organization, exactness results, decode crossover, prefill crossover,
residency result, full-model correctness/performance, thermal result, fallback
result, promotion decision, evidence roots/hashes, PR, and merge commit.
~~~
