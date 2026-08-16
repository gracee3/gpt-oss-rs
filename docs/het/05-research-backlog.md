# Research backlog

This is a queue of answerable questions for Stage 2. It is not an
implementation plan and does not rank designs. Each question starts from a
specific Stage-1 fact and names the evidence needed to close it.

## Top five design blockers

| Rank | Blocking question | Why architecture selection must wait |
|---:|---|---|
| 1 | Can expert weights be independently owned and loaded once by CPU, GPU0 and GPU1 under a measured 120B memory bound? | The current CPU path maps source plus full repacks; CUDA retains multiple host/device U8 forms; TP loads then shards. A design that cannot fit or has hidden load high-water is invalid regardless of its execution API. |
| 2 | What is the narrowest semantic boundary between routing and expert execution, and where do router outputs/reduction live? | `moe_batch`, `gpt-oss-moe-semantics` and CUDA decode expose different boundaries and dtypes. Selecting the wrong seam either duplicates GPT-OSS semantics or couples generic placement to a backend layout. |
| 3 | What one-layer oracle and BF16-bit invariants gate a mixed CPU/GPU0/GPU1 execution? | Without router/expert/reduction/residual first-divergence evidence, performance experiments cannot distinguish a fast wrong path from a valid candidate. |
| 4 | What transfer/concurrency mechanisms actually work on this PCIe-only, P2P-unsupported topology, and where are the crossover points? | Static placement usefulness depends on pageable/pinned transfer cost, stream/host overlap and bucket sizes, not theoretical bandwidth or device names. |
| 5 | What is the commit, cancellation and rollback boundary after partial multi-device work? | The CPU has prepare/discard/commit; the GPU engine does not provide a cross-rank/CPU transaction. Failure semantics constrain ownership and scheduling before optimization policy can be chosen. |

## P0: resolve before comparing designs

### Q1. Narrowest existing routing/execution boundary

**Known:** `CpuModel::moe_batch` owns router projection, BF16 round trips,
stable top-k, grouping, expert calls, rank-ordered reduction and output rounding.
`project_mxfp4_batch` receives one expert and a row bucket but is CPU/Xe-shaped.
`gpt-oss-moe-semantics` is backend-neutral but does not own device buffers.

**Research question:** which smallest boundary can preserve GPT-OSS semantics
without embedding CPU repack, CUDA stream or host policy? Compare at least:

- before `CpuRoute` grouping;
- after grouping, before `project_mxfp4_batch`;
- around a single `RepackedMxfp4::expert` view; and
- the CUDA `GptOssMoeLayerWeights::forward_decode_gpu` launch boundary.

**Required evidence:** typed input/output/lifetime inventory for each boundary,
including prefill/decode shapes, reduction order and failure ownership. Do not
select one until Q2, Q3 and Q14 are answered.

### Q2. Router-output location, dtype, layout and synchronization

**Known:** CPU router logits are f32 storage after BF16 round trip; IDs are
host `usize`, weights are BF16-rounded f32, then routes are stable-sorted. CUDA
decode keeps f32 logits/weights and `i32` IDs on one GPU; CUDA prefill D2Hs the
whole input and routes on host.

**Research question:** for a mixed layer, where should logits, top-k IDs and
weights reside; which component owns BF16 rounding and stable tie behavior; and
when is a host/device synchronization semantically necessary?

**Required evidence:** bit-level CPU versus CUDA router trace for selected 20B
rows, including ties/near-ties, plus a location/lifetime table. No transfer
benchmark is meaningful until semantic equivalence is established.

### Q3. Independent expert-weight ownership

**Known:** CPU `CpuLayer` points into whole-layer repacks; CUDA
`GptOssMoeLayerWeights` owns whole-layer host vectors and optional whole-layer
device arrays. There is no per-expert device owner. TP slices projection
dimensions, not expert identity.

**Research question:** can one expert's blocks/scales/bias be held independently
on CPU, GPU0 or GPU1 without retaining the full source and another full layer
copy at each destination? What immutable metadata must remain shared?

**Required evidence:** exact ownership graph and byte counts for constructing,
moving, dropping and reopening one expert and one layer from both checkpoints.
Any experiment must use a read-only source or disposable cache, never alter the
checkpoint.

### Q4. Current CUDA MoE execution unit

**Known:** decode loops all local experts, builds dense masked rows, dequantizes
one full expert at a time, uses cuBLAS and weighted-add. Prefill MoE is scalar
host work.

**Research question:** does the current decode implementation produce correct
results for inactive experts, multi-token groups and TP slices; what work is
actually skipped; and is its useful execution unit a single expert, dense
expert scan, grouped bucket, or fused layer?

**Required evidence:** kernel/allocator trace for one decode row and
representative prefill bucket, active/inactive expert counters, scratch
high-water, and an oracle comparison. First audit global-vs-local TP dimensions
with a tiny fixture.

### Q5. MoE participation among the 29 SM86 PTX modules

**Known:** `gpt_oss_moe.cu` supplies route/top-k, select/mask, one-expert f16
dequant and weighted-add. Expert GEMMs are cuBLAS; fused SwiGLU is another
general module. The other PTX covers the transformer shell, KV and sampling.

**Research question:** which exact module/function/cuBLAS sequence executes in
prefill versus decode, which allocates scratch, and which synchronizes or
round-trips to host?

**Required evidence:** bounded launch trace mapped to source symbols and bytes,
not just the compile-time PTX list.

### Q6. CPU/CUDA layout equivalence and transformation

**Known:** checkpoint expert weights are canonical U8 block/scales. CPU combines
them into 17-byte canonical or interleaved-x8 records; CUDA keeps separate
arrays and expands a selected matrix to f16 on every decode execution.

**Research question:** are semantic indexes identical; can a placement consume
canonical bytes directly; or is one persistent, destination-specific repack
required? Can conversion occur once without a second complete checkpoint?

**Required evidence:** per-index byte mapping for one expert, round-trip/hash
test for both CPU layouts, CUDA dequant comparison, conversion time and peak
memory.

### Q7. Persistent and temporary memory at model scale

**Known:** exact checkpoint payload and static duplicate sites are documented
in `03-host-model-baseline.md`; real high-water is absent.

**Research question:** for 20B and 120B, what are checkpoint mappings, resident
pages, repacks, owned host vectors, pinned buffers, GPU weights, CUDA allocator
retention, dequant scratch, KV, NCCL and conversion high-water through load,
prefill and decode?

**Required evidence:** phase-tagged RSS/PSS/smaps, per-GPU free/used deltas,
allocator records and existing `CpuMemoryDescriptor` fields. Report persistent,
temporary and allocator-retained bytes separately. Start with metadata/tiny
fixture, then bounded 20B; no 120B load before review.

### Q8. Exact per-expert bytes in every representation

**Known:** checkpoint representation is exactly 13,236,480 bytes/expert
including BF16 biases for both models. The CPU repack payload is 13,219,200
bytes plus header; CUDA full f16 expansions are much larger and temporary.

**Research question:** what are the exact per-expert bytes for CPU canonical,
CPU x8, CUDA canonical U8, CUDA expanded f16, host metadata/bias, allocation
alignment and any destination index? Which are shared versus duplicated?

**Required evidence:** source-derived formulas checked against actual allocation
sizes for one expert. Keep checkpoint, persistent execution and workspace rows
separate.

### Q14. Minimum one-layer oracle

**Known:** `CpuLayerTrace` already contains input norm, attention/residual,
router logits, selected IDs/weights, per-expert gate/up, SwiGLU, down, weighted
output, MoE output and layer output. Restricted CUDA tools have related but not
identical fields.

**Research question:** what minimum 20B fixture/capture can run one real layer
with selected experts on CPU, GPU0 and GPU1 while comparing every semantic
boundary and final residual at the required BF16 bit boundary?

**Required evidence:** one schema, immutable source identity, forced identical
input/routes where appropriate, exact/tolerance policy per field, and first
divergence. It must distinguish prefill from decode and must not use HTTP.

### Q16. Failure, retry and commit

**Known:** CPU prepared work is discardable until atomic commit. CUDA mutates KV
and allocator/device state during rank execution; scheduler abort exists, but no
multi-device rollback protocol is present.

**Research question:** for errors at route packing, H2D, CPU worker, each GPU
kernel, D2H, reduction, KV update and cancellation, what state has changed and
which work is safe to retry? Which failures require request termination?

**Required evidence:** state transition table plus injected-failure tests on a
tiny model. No promotion without deterministic token/KV/publication outcome.

## P1: determine feasibility and useful policy

### Q9. Bytes moved for decode and grouped prefill

**Known:** a top-4 BF16 scatter-and-return lower bound is 46,080 bytes per row;
current CUDA prefill moves 23,040 f32 bytes per row because all MoE stays host.

**Research question:** for the candidate boundary/boundaries from Q1, what
router metadata, activation, result, padding and synchronization bytes move for
one decode token and representative prefill `T`/expert-bucket distributions?

**Required evidence:** formulas checked by counted copies at `T=1`, 128, 512
and 2,048, with direction/destination and sparse bucket occupancy.

### Q10. Existing overlap mechanisms

**Known:** the runner uses one stream and disables event tracking; pinned async
D2H is limited to argmax IDs. CPU serving has an async owner but does not
co-schedule CUDA work.

**Research question:** which cudarc stream/event primitives, pinned-buffer
lifetimes and host worker mechanisms can safely issue CPU, GPU0 and GPU1 work
without accidental synchronization or CUDA graph conflicts?

**Required evidence:** tiny non-model concurrency probe with event/timeline
capture, cancellation and drop tests. Keep graph and non-graph decode distinct.

### Q11. Operational topology and peer behavior

**Known:** one NUMA node, separate PCIe host branches, host maximum Gen3 x16,
no active NVLink, driver PCIe P2P reports unsupported.

**Research question:** does CUDA independently confirm peer access unavailable;
what loaded link state and pinned H2D/D2H paths are stable; and is simultaneous
traffic to both GPUs independent or host-memory-limited?

**Required evidence:** `cudaDeviceCanAccessPeer`-equivalent probe, loaded link
capture, single/bidirectional pinned transfers and dual-GPU contention. Record
unsupported separately from failed and untested.

### Q12. Installed CPU kernel choice and exact fallback

**Known:** this CPU exposes AVX2/FMA and AVX-512 VNNI but no AMX. Scalar and
ExactBF16 references remain available; auto can choose AVX-512 VNNI.

**Research question:** which expert operation shapes use AVX-512 versus AVX2
or scalar tails under real route-bucket `M`, and which reference mode is the
promotion oracle?

**Required evidence:** `CpuExecutionProfiler` operation/bucket records plus
forced scalar/AVX paths on the same one-layer input. Do not hardcode 4215R.

### Q13. CPU-execute versus GPU-transfer crossover

**Known:** no measured CPU memory, PCIe or expert-kernel bandwidth exists for
this host. Product specifications are not evidence.

**Research question:** by `M`, projection role, representation warmness and
destination, when is CPU execution faster than GPU transfer+execution+return?
How does dual-GPU contention alter it?

**Required evidence:** cold/warm, order-rotated samples with transfer-only,
kernel-only and end-to-end components; frequency/thermal/link state; exact
output gate. Report a capability/shape policy, not a CPU-name policy.

### Q15. Evidence representation

**Known:** existing manifests, timers, diagnostics, CPU traces and profiling are
strong but do not record placement, transfer or synchronization.

**Research question:** what minimal extension/artifact can represent static
placements, persistent/scratch bytes, H2D/D2H/host-staged GPU-to-GPU bytes,
overlap, waits, device/layer/expert bucket and first divergence while preserving
existing campaign identity/redaction rules?

**Required evidence:** example stable JSON and validation rules using
`RunManifestV1` artifact references. Avoid a parallel evidence system.

### Q17. Ordered 20B validation before 120B

**Research question:** what exact sequence of metadata, loader-only, one-expert,
one-layer, prefill, forced decode, retained continuation, memory and failure
gates must 20B pass on this host before a bounded 120B attempt is reviewable?

**Required evidence:** proposed research gate outcomes and cost/risk, not an
implementation work plan. The first action is likely a cheap loader preflight,
but the order remains for research review.

### Q18. Serving semantic defect isolation

**Known:** current routes use Harmony render/parse and usage accounting, but the
latest live outputs were invalid/empty/malformed.

**Research question:** is the defect model output, stopping, stream assembly,
partial/final Harmony parsing or token accounting? Can model-level
heterogeneous validation stay entirely below HTTP until final control?

**Required evidence:** one Harmony-native known token sequence through parse and
accounting, then one model output with raw token IDs. Reopen protocol only for a
demonstrated contract regression.

### Q19. Genuine model-independent mechanisms

**Research question:** after Q1-Q18, which mechanisms have no GPT-OSS/MXFP4,
CUDA, x86 or host assumption: immutable device ownership, static placement
description, bounded staging, async job/join, commit barrier, telemetry? Which
parts must stay in GPT-OSS routing/checkpoint/backend layers?

**Required evidence:** assumption table against at least CPU scalar, CPU x8,
CUDA canonical U8 and the original/transformed checkpoint namespaces. Do not
generalize for hypothetical Qwen or servers in this phase.

## P2: prior-art queue for Stage 2 only

### Q20. External projects and papers

No external repository was cloned and no broad web research was performed in
Stage 1. The following is an **unvalidated source queue**, not a statement about
quality, correctness, licensing fit or applicability:

| Candidate source | Exact question to answer during research |
|---|---|
| KTransformers | Where are router semantics separated from CPU/GPU expert ownership, and how are static offload weights loaded/staged without hidden duplication? |
| Fiddler | What measured CPU-execute versus GPU-transfer model and synchronization boundary is used for MoE offload? |
| HybriMoE | How are heterogeneous placements represented, grouped and scheduled, and what exactness/failure assumptions are made? |
| MoE-Infinity | How are expert residency, storage traffic and bounded staging accounted, especially for decode versus prefill? |
| `llama.cpp` / `ggml` | How do backend buffers, tensor ownership, graph scheduling and CPU+multi-GPU splits avoid leaking one host identity into model semantics? |
| `mistral.rs` | What Rust-native device mapping, SafeTensors lifetime, quantized layout and multi-device error patterns are reusable or cautionary? |
| xInfer | Which expert kernels/grouping layouts and memory assumptions are relevant to the exact GPT-OSS shapes here? |
| GPU expert-parallel runtimes (to identify narrowly) | How do dispatch/all-to-all/reduction streams, events and failure barriers work when P2P is unavailable? |
| Relevant heterogeneous-MoE papers (to identify narrowly) | Which claims include exact model semantics, real transfer accounting and static-placement evidence rather than simulated/adaptive policy only? |

For every source, record exact revision, license, inspected symbol/file,
question answered, applicable model/layout/hardware assumptions, and negative as
well as positive findings. Do not copy code or infer a preferred design from a
project name.

## Stage-2 stop conditions

Research should return for review once the top five blockers have evidence
sufficient to compare candidate boundaries and static placements. It must not
silently become implementation planning. In particular, no 120B full load or
generation, adaptive placement, approximate policy, Qwen work, Xe optimization,
protocol redesign, upstream contribution, or source import is authorized by
this backlog.
