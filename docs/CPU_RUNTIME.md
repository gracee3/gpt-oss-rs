# Native GPT-OSS CPU Runtime

The CPU runtime is intentionally narrow: Linux, official GPT-OSS SafeTensors,
BF16 dense weights, and MXFP4 experts. It supports experimental multi-request
batching without routing through the CUDA runner or mock architecture.

## Weight ownership

`CpuTensorStore` maps every SafeTensors shard read-only and borrows dense
tensors directly from those mappings. It does not construct the existing
GPU-shaped `ModelWeights` collection. The snapshot must remain immutable for
the lifetime of the runner; Hugging Face content-addressed snapshot blobs meet
that requirement.

Only `gate_up_proj` and `down_proj` MXFP4 tensors are repacked. Cache keys cover
the resolved model revision, every source-shard SHA-256, the tensor name and
shape, repack format version, and exact layout identifier. Scalar and
exact-BF16 projections use `CanonicalAdjacentV1`, whose record is one E8M0
scale byte followed by 16 adjacent-nibble bytes. Q8/residual-Q8 automatic,
forced-AVX2, and forced-AVX-512/VNNI projections use
`InterleavedSplitX8V2`: complete eight-row groups store eight scales and 128
split-half packed bytes per K block, while one-to-seven tail rows remain
canonical. Both layouts use exactly 17 bytes per row/block. Only the selected
layout is mapped; an existing cache for another layout is left untouched.
Writers use an exclusive lock, a synced temporary file, atomic rename, and a
directory sync. Published files are mapped read-only and never changed in
place.

The loaded resources now live in one immutable `Arc<CpuModel>`. Configuration,
the tensor store and shard mappings, repacked expert mappings, layer
descriptors, final norm, RoPE state, kernel plan, and Rayon pool are loaded
once and shared. `CpuModel` never owns or infers a current sequence.

Each sequence instead owns a `CpuSequenceModelState`: its per-layer full and
sliding KV caches, absolute next position, context cap, model token history,
abort state, and monotonic revision. `CpuExecutionContext` is worker-local and
may be used by only one preparation at a time. The existing `CpuModelRunner`
remains a batch-one compatibility facade containing one shared model, one
sequence state, and one execution context; constructing several facades from
the same model does not clone mapped weight or repack bytes.

CPU execution is prepare/commit based. `CpuStepBatch` rows explicitly name the
sequence, input token, absolute position, and whether logits are required.
`CpuModel::prepare_step` reads committed KV plus same-sequence staged rows and
returns a self-contained `PreparedCpuStep`. It does not mutate sequence state.
Commit first validates every supplied sequence ID, revision, position, layer,
and row shape, then publishes all KV rows, positions, histories, and revisions.
Sliding eviction occurs only during that successful commit. A model error,
stale commit, explicit discard, or dropped prepared value therefore leaves
committed state unchanged.

## Matrix prefill

Serving prefill constructs one `CpuStepBatch` for the prompt and marks only its
last row as requiring logits. The model gathers all embeddings, then advances
the complete row set through one transformer layer at a time. Dense Q/K/V and
output projections operate across the row set. RoPE uses each row's explicit
absolute position, while attention remains row-wise and sees committed KV plus
only earlier staged rows from the same sequence. Full and sliding cache effects
remain staged until the complete batch commits.

MoE routing produces stable records containing expert, source row, original
top-k rank, and routing weight. Records are grouped by expert for gate/up and
down matrix execution, then restored to source-row and top-k-rank order before
weighted reduction. This keeps reduction order deterministic even when expert
execution order changes. Rows without `logits_required` do not receive a
logits allocation or result.

MXFP4 expert matrices use the common `Mxfp4MatmulProblem` contract with typed
Q8/residual-Q8 row views, checked output stride, and queryable caller-owned
scratch. The scalar matrix reference is the semantic oracle. Explicit AVX2
packs up to four activation rows and computes eight output rows over the
existing x8 persistent cache; M tails stay in the bounded panel and N tails use
the canonical-row scalar path. The worker-local `CpuExecutionContext` reuses
aligned transient scratch. See [`MXFP4_MATRIX_API.md`](MXFP4_MATRIX_API.md) for
the complete contract.

## Numeric and cache behavior

- Dense BF16 and MXFP4 projections accumulate in FP32 and round at BF16 model
  operation boundaries.
- YaRN uses GPT-NeoX half rotation, the checkpoint's correction range and
  attention scale, and the official PyTorch BF16 operation boundaries for its
  trigonometric values and rotary arithmetic.
- GQA maps eight query heads to each KV head for the 20B checkpoint.
- Learned per-head sinks add a logit to the softmax denominator with an
  implicit zero value vector.
- Sliding layers retain exactly the latest 128 BF16 K/V tokens. Full-attention
  layers retain the configured CPU context cap.
- Routing is stable top-4-of-32 selected-logit softmax. Gate/up outputs use the
  official interleaved clamped GPT-OSS SwiGLU formula, including every BF16
  multiply, sigmoid, and add boundary.
- Expert activations use residual Q8 by default:
  `Q8(x) + Q8(x - dequantize(Q8(x)))`. Scalar, AVX2, and AVX-512/VNNI kernels
  unpack each MXFP4 group once and compute both integer dots from that unpack.
  The AVX2 and AVX-512/VNNI x8 kernels decode and accumulate eight output rows
  together. The AVX-512 body uses 64-byte ZMM layout loads and VNNI byte dots;
  canonical output tails retain the audited AVX-512 row kernel.
- E8M0 follows the MX specification for all paths: `0x00` is `2^-127`, normal
  encodings are exact powers of two, and invalid `0xff` propagates NaN. The
  pinned 20B checkpoint contains only normal scale bytes, so this fixes
  synthetic/special-value semantics without changing its values.
- The parity runner can select `q8`, `residual-q8`, or streaming `exact-bf16`
  expert projections. Exact BF16 decodes repacked blocks on demand into the
  deterministic FP32 reduction lanes; it is diagnostic-only and selects the
  compact canonical cache rather than expanding weights.

The runtime is an experimental mainline backend. Its preceding baseline passed
the maintained seven-scenario suite across scalar, AVX2, AVX-512/VNNI, and
automatic dispatch. The AVX2 x8 promotion uses a narrower full-model gate:
`harmony_122` on cold automatic and warm forced-AVX2 paths, and `harmony_262`
on automatic, forced-AVX2, and scalar paths. A repeat of the exhaustive 28-run
matrix, Criterion suite, llama.cpp captures, and complete API permutation
matrix is intentionally deferred until the planned AVX-512, GEMM/prefill, AMX,
and CPU scheduling features are developed. During that feature-development
phase, each new path still requires focused scalar equivalence, targeted
full-model parity, cache-integrity checks, and relevant API smoke coverage;
semantic, memory-safety, cache, and API failures remain blocking. Performance
thresholds and broad certification are not promotion gates until the later
tuning phase. A stricter end-to-end trace still shows a rare BF16
reduction-order difference before the expert projection; it is retained as
diagnostic evidence rather than hidden or compensated later. Trusted mode
continues to reject CPU serving pending a separate certification review.

The experimental AVX-512 x8 milestone passed scalar equivalence for Q8 and
residual-Q8 across x8/tail and real projection shapes. A short `harmony_122`
20B run produced first token `200005` on both automatic AVX2 x8 and forced
AVX-512 x8. This is targeted feature evidence, not the deferred exhaustive
certification or a basis for changing automatic selection.

## CPU+Xe projection attachment

The default server build includes the private `gpt-oss-xe` crate. It resolves
the audited OpenCL 3.0 ABI at runtime with `libloading`; there is no OpenCL
link-time dependency, and non-Linux or loader-absent hosts retain ordinary CPU
serving. Attachment accepts exactly one OpenCL GPU and exactly PCI
`8086:9a49`. It also requires subgroup 32, integer dot-product support,
integrated memory, checked allocation/workgroup limits, one coherent system
loader/driver/IGC generation, and either a compiler or a valid native cache.
Unsafe code is confined to the OpenCL ownership module.

The hybrid engine owns one non-cloneable context, one serialized in-order
queue, one terminal event at a time, and an idempotent reverse-order shutdown.
Its native-program cache is atomic and keyed by immutable source/ABI/build
options, PCI identity, driver version, and hashes of the loader, driver, and
IGC. A stale, corrupt, or numerically invalid cache is discarded and rebuilt
from the byte-identical embedded source when the compiler is available.

CPU retains model mapping, routing, attention, KV state, sampling, commit, and
the authoritative fallback repacks. Xe receives only selected prefill expert
projections. M=1–3 and all decode projections remain on CPU. Gate/up and down
have separate checked thresholds; the initial explicit policy selects
`tile32-m4-v2` at M>=4 with workgroup 32, pads row tails to four with zero
activation records, and discards padded outputs. Each expert operation repacks
only that expert to `[tile][K-block][17 planes][32 lanes]`, uploads it once,
and reuses it across checked row chunks.

The default `--xe-max-resident-mib 128` slab contains one largest-expert
weight/bias pair plus reusable activation/output buffers. The runtime snapshot
reports its actual device bytes, the independent worst-case host staging bound,
maximum rows per chunk, exact PCI/driver/library identity, ABI/source hashes,
thresholds, native-cache result, and fault policy. CPU request admission adds
the host-staging class without mislabeling mapped model bytes as device memory.

The forced-only `--xe-expert-cache-mib` experiment defaults to zero and is
legal only with explicit `--device xe`. A nonzero value adds a separately
accounted deterministic LRU of immutable expert weight/bias device buffers.
Cache hits avoid both expert repacking and weight/bias upload; oversize entries
bypass to the streaming path. Identity includes model/tensor, layer/expert,
role/shape/layout, kernel/ABI/build, PCI, and runtime library facts. Automatic
device selection and decode behavior are unaffected. See
[`TIGER_LAKE_XE_RESIDENCY.md`](TIGER_LAKE_XE_RESIDENCY.md).

An OpenCL runtime fault is terminal for Xe in that process: the queue is
drained, uncommitted output is discarded, that projection is recomputed once
through the configured CPU path, and a process-wide breaker stays open until
restart. The staged model step may commit only after that recomputation
succeeds. Cancellation and shutdown also drain in-flight resources before
discard; they never free a live event or replay a result after commit.

## Serving policy

The server owns `AsyncCpuBatchEngine`, which wraps one `CpuBatchEngine`, one
canonical `SequenceTable`, and the shared `Arc<CpuModel>`. CPU requests never
pass through the GPU or mock executor. Prompt chunks and decode rows from
several requests may share a `CpuStepBatch`; layer-major model execution and
row-wise causal attention preserve sequence-local KV. Chat Completions,
Responses, and text Completions, including streaming forms, share this path.

The managed CPU service exposes `Starting`, `Ready`, `Draining`, `Failed`, and
`Stopped` lifecycle states. `/health` remains liveness; `/ready` succeeds only
after the effective runtime snapshot is frozen and hashed. Admission is
bounded, logical request memory is granted after tokenization and before the
canonical sequence table, and shutdown closes admission before draining and
joining the owner. Owner failure is retained as a typed service failure rather
than being inferred from a closed channel.

Scheduling is reserve/execute/commit. Reservation records revisions and
in-flight IDs but advances no prompt, KV, RNG, token, or output state. Model
execution and sampling stage their results. Cancellation and client disconnect
are checked again after kernels return; only retained, revision-matching rows
commit. Failures and stale work are discarded atomically. Output channels are
ID-keyed delivery handles rather than a second sequence authority. The owner
publishes committed suffix events without awaiting HTTP consumers. Delivery
is byte-charged, reserves terminal control capacity, and may coalesce only
adjacent text deltas for the same choice. Stop-string prefixes are held back
until stable, so published bytes are never retracted. Disconnect or overflow
tombstones the request before its next commit; cancellation and failure are
never translated into `finish_reason=stop`. See
[`CPU_SCHEDULER.md`](CPU_SCHEDULER.md) for budgets, fairness, lifecycle, and
topology details. `CpuWorker` remains only as a batch-one compatibility/test
facade.

Service defaults are 2 MiB request bodies, 8 MiB non-streaming responses and
stored entries, 256 KiB serialized stream events, 1 MiB queued delivery per
request (including 16 KiB terminal control), `max_num_seqs` MiB global queued
delivery, and a 64 MiB/64-entry terminal Responses store with FIFO eviction.
The logical CPU request budget defaults to a checked worst-case estimate per
sequence; `--cpu-request-budget-mib` overrides it. A hybrid attachment adds its
bounded host-staging class separately from the Xe device slab. Response storage
and delivery are separately budgeted. Batch source remains in the tree, but
batch routes are not mounted by this service foundation.

`--device auto` never probes or initializes CUDA. It probes Xe before CPU only
when the checked-in exact-stack production-promotion record is enabled. The
2026-08-12 production integration record is disabled because the paired
full-model performance intervals did not clear parity, so current automatic
GPT-OSS serving selects CPU without probing OpenCL. `--device cpu` is explicit.
`--device xe` requires successful attachment and startup numerical validation;
its effective backend is `cpu_xe`, it retains the prefill-first CPU profile,
and trusted mode is rejected. A same-PCI but changed driver may be used only
explicitly after capability and startup numerical tests, and is labeled
`unvalidated_explicit`. CUDA requires `--device cuda --runtime-mode
experimental`; trusted CUDA is rejected.
Automatic selection rejects non-GPT-OSS models, while `--device mock` remains
an explicit test-only choice. CPU rejects tensor parallelism, pipeline
parallelism, CUDA graphs, best-of, beam search, and trusted mode. Automatic
GPT-OSS serving uses the `gpt-oss-cpu` profile (`max_model_len=8192`,
`max_num_seqs=1`). Explicit `--max-num-seqs > 1` enables experimental
multi-request CPU scheduling. `--max-num-batched-tokens` bounds all rows per
iteration, while `--max-prefill-chunk` bounds prompt rows per sequence; a zero
prompt chunk means only the remaining iteration token budget applies. The GPU
profile is reserved for explicit CUDA.

CPU startup reports process-allowed CPUs and memory nodes, observed
physical-core and NUMA relationships, available parallelism, and configured
worker threads. These are read-only diagnostics: the runtime applies no
affinity, placement, memory-binding, or automatic topology policy.

`--cpu-repack-cache` defaults first to `GPT_OSS_RS_CACHE`, then
`XDG_CACHE_HOME/gpt-oss-rs`, then `$HOME/.cache/gpt-oss-rs`. Model snapshots,
repacked expert tensors, build artifacts, and benchmark output remain outside
Git.

Remote model IDs serve as their own public identity. A fetched local snapshot
may recover the model ID from its manifest; any other local path requires
`--served-model-name`. The service never exposes a local path as an API ID or
metric label. `--evidence-dir` writes the canonical effective runtime snapshot,
while readiness exposes only its hash and sanitized identity. Prometheus
exposition uses bounded `gpt_oss_*` label vocabularies and is absent when
`--disable-telemetry` is set; correctness bookkeeping remains active.
`cpu_parity` and `cpu_service_probe` atomically write a raw capture plus a
`gpt-oss-rs.cpu-evidence/v1` manifest sidecar with an absolute artifact path
and SHA-256. The service probe is model-independent: it inspects liveness,
readiness, public model identity, metrics policy, and the absent batch route
against an already-running server without issuing inference.
`cpu_service_overhead` runs paired release subprocesses over the bounded
delivery fixture and the real Prometheus recorder, then emits a pass/fail E1
sidecar for the 1% median-throughput and 2% median-p99-latency gates.

`--cpu-matmul-backend` accepts `auto`, `scalar`, `avx2`, and `amx-int8` and is
serialized as `device.cpu_matmul_backend`; the default is `auto`. During this
experimental milestone, automatic M=1 expert projection uses the established
dispatched GEMV path and automatic M>1 uses the scalar matrix reference.
Optimized matrix execution therefore requires explicit `avx2`. `amx-int8`
requires the optional feature, Linux x86-64 CPUID support, Linux XSTATE kernel
support, and process tile-data permission. It is never selected automatically.
The prototype consumes transient M<=16, N=16, K=32 panels, stores INT32 after
every K block, and applies per-row activation and per-column E8M0 scales in
FP32. M=1 and N tails use scalar fallbacks only after the forced AMX gates have
succeeded. Portable emulation is tested; execution on AMX hardware remains
deferred.

## Kernel dispatch

Forced `scalar`, `avx2`, and `avx512-vnni` requests preserve their compatibility
meaning for dense operations and fail before execution when their exact ISA
requirements are unavailable. Scalar uses the canonical MXFP4 row reference,
AVX2 uses its x8 GEMV and interleaved layout, and AVX-512/VNNI uses the genuine
ZMM/VNNI x8 GEMV with that same layout. Incomplete output groups use canonical
bytes and the existing AVX-512 row kernel. The x8 body requires AVX-512F,
AVX-512BW, and AVX-512VNNI; the complete forced path also requires AVX2 and
AVX-512VL for its canonical-row and other compatibility operations. The public
compatibility `path()` remains unchanged.

Automatic dispatch resolves an immutable per-operation plan. On the validated
development host, BF16 matvec, Q8/residual-Q8 quantization, and RMS norm select
the AVX-512 implementations while MXFP4 GEMV selects AVX2 x8. Residual-Q8 uses
the same x8 kernel and decodes weights once for both activation passes. This is
a capability policy, not a CPU-model policy; startup logs and captures include
the exact GEMV kernel and packed-layout identifiers in the immutable plan.

Matrix backend selection is deliberately separate from that compatibility
kernel plan. Explicit AVX2 matrix execution selects the x8 layout and validates
host support when invoked. Automatic matrix execution does not promote the new
4x8 path: it preserves GEMV for M=1 and uses the scalar reference for M>1.
There is no shape threshold, tuning table, or second persistent weight format.

The current plan is a baseline, not the final abstraction. Future dispatch
must describe precise ISA requirements and include operation type, matrix
shape, batch size, thread count, and packed-layout availability. GEMV decode
and GEMM-like prefill may use different kernels and weight layouts. The forward
design is documented in `MXFP4_CPU_BACKEND_HANDOFF.md`.
