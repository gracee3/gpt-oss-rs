# Runtime code map

This map describes the code that exists at HEAD. It does not propose a new
architecture. Symbols called “candidate seams” are research subjects only.

## Source-to-responsibility map

| Source / symbol | Present responsibility | Device/model assumptions and maturity |
|---|---|---|
| `crates/gpt-oss-server/src/main.rs` (`Cli`, `Commands::Serve`, `DeviceChoice`) | CLI, runtime options, pre-Tokio Harmony initialization, `EngineConfig` construction | Device choice is whole-runtime (`cpu`, `xe`, `cuda`, `mock`, `auto`), not placement. GPT-OSS auto is CPU-first. |
| `crates/gpt-oss-server/src/server.rs` (`serve`, `create_cpu_engine`, `create_gpu_engine`) | Snapshot/tokenizer resolution, runtime validation, engine ownership, service lifecycle | CPU and CUDA constructors are mutually exclusive. No CPU+CUDA inference owner exists. |
| `crates/gpt-oss-engine/src/hf_snapshot.rs` | Local/HF snapshot discovery; index shard-name validation for downloads/cache | Only top-level files; does not adapt original GPT-OSS namespace. |
| `crates/gpt-oss-model-runner/src/cpu_tensor_store.rs` (`CpuTensorStore`) | Read-only mmap of every top-level SafeTensors shard; borrowed dtype-checked tensor slices | Ignores the index and maps all `*.safetensors`; no full tensor copy. CPU/GPT-OSS loader. |
| `crates/gpt-oss-model-runner/src/cpu_repack.rs` (`CpuRepackCache`, `RepackedMxfp4`) | Atomic, versioned 17-byte-record MXFP4 repack files; retained read-only mappings | CPU layouts `CanonicalAdjacentV1` and `InterleavedSplitX8V2`; full expert MXFP4 payload is represented a second time beside source mappings. |
| `crates/gpt-oss-model-runner/src/cpu_runner.rs` (`CpuModel`, `CpuModelRunner`) | Production native CPU GPT-OSS load, prefill/decode, attention, MoE, KV/state, traces, profiling | Strongly GPT-OSS- and CPU-shaped; exact transactional state boundary. |
| `crates/gpt-oss-engine/src/cpu_batch_engine.rs` (`CpuBatchEngine`) | FCFS mixed prefill/decode scheduling, memory grants, model prepare, sampling, commit | Canonical CPU serving execution owner. |
| `crates/gpt-oss-engine/src/async_cpu_engine.rs` | Async command/delivery owner, cancellation, bounded publication, failure shutdown | Drains cancellation before commit; no heterogeneous analogue. |
| `crates/gpt-oss-moe-semantics/src/lib.rs` | Stable top-k, selected softmax, grouping/order and reduction semantic contract | GPT-OSS sparse-MoE semantics but backend-neutral inputs/outputs. Candidate semantic boundary, not device ownership. |
| `crates/gpt-oss-model-runner/src/model_loader/gpu_loader.rs` | Production CUDA loader: one-shard-at-a-time mmap, dense conversion/upload, host U8 collection | Dense BF16 becomes f32 and/or f16 device storage; all MXFP4 U8 becomes owned host vectors. |
| `crates/gpt-oss-model-runner/src/model_loader/safetensors.rs` | Generic allocator-based SafeTensors loader | Reads an entire shard into a `Vec`; not the production CUDA path and not memory-bounded at model scale. |
| `crates/gpt-oss-engine/src/worker/gpu_worker.rs` (`GpuWorker`) | One CUDA context/device/runner, model load, KV cache, CUDA graph pipeline, sampling | Owns one non-default model stream; creates additional legacy stream wrappers, but event tracking is disabled because the active runner assumes one stream. |
| `crates/gpt-oss-model-runner/src/gpu_runner.rs` (`GpuModelRunner`) | GPU transformer shell, metadata upload, embeddings, layer loop, KV, logits/argmax | Experimental warnings remain; GPT-OSS f16 decode and host-MoE prefill differ materially. |
| `crates/gpt-oss-model-runner/src/gpu_layer.rs` (`GptOssMoeLayerWeights`) | GPT-OSS host scalar MoE and CUDA decode MoE | Host-owned full MXFP4 copies plus optional device copies; one layer/runner/device. |
| `kernels/gpt_oss_moe.cu` | Route/top-k, expert mask/select, one-expert MXFP4-to-f16 dequant, weighted add | Four `sm_86` PTX kernels; cuBLAS still performs both expert projections. |
| `crates/gpt-oss-model-runner/src/tensor_parallel.rs` | Local and NCCL all-reduce interface | Real CUDA NCCL comm exists; CPU is not a rank. |
| `crates/gpt-oss-engine/src/gpu_engine.rs` (`TensorParallelCoordinator`) | Builds one worker per CUDA rank, launches/collects ranks, consumes rank-0 result | Source-level multi-GPU tensor parallel; no retained real GPT-OSS TP correctness evidence here. |
| `crates/gpt-oss-model-runner/src/model_loader/shard.rs` | Name-based host/device tensor-parallel sharding | Contains GPT-OSS expert block/scale rules, but downstream MoE still uses global dimensions; compatibility is unproven and appears inconsistent. |
| `crates/gpt-oss-engine/src/executor/multi_gpu.rs` | Separate executor surface | NCCL is explicitly a stub; do not confuse it with `GpuLLMEngine`'s real NCCL code. |
| `crates/gpt-oss-evidence/src/lib.rs` | Stable manifests, campaign attempts, redaction, byte-capped diagnostics | Extensible generic evidence, no current heterogeneous placement/transfer schema. |
| `crates/gpt-oss-tokenizer/src/protocol.rs` and server routes | Harmony render, parse, partial stream parse, stop tokens and structured message boundary | Active GPT-OSS service protocol path; semantic live validation remains separate. |

## CPU execution flow: checkpoint to committed token

1. `create_cpu_engine` resolves a local snapshot, then `CpuModel::load_with_backends`
   parses the transformed Hugging Face `config.json`. `CpuGptOssConfig::validate`
   accepts only `GptOssForCausalLM`, expected attention types, valid GQA
   dimensions, top-k bounds, and YaRN settings.
2. `CpuTensorStore::open` sorts and read-only maps every top-level
   `*.safetensors`. It parses headers into tensor-to-shard offsets. Dense BF16
   matrices remain borrowed from those mappings for the model lifetime.
3. `SourceIdentity::from_store` uses a fetch manifest if present; otherwise it
   hashes every shard before repack-key construction. The local 20B snapshot has
   no such fetch manifest, so a first real CPU load would read all shard bytes
   for hashing even though tensor storage itself is mmap-backed.
4. `load_layer` validates transformed tensor names/shapes. Norms and biases are
   converted to owned f32 vectors. Each layer's gate/up and down MXFP4 source
   tensors are repacked once into versioned cache files and retained as mmap
   views. Experts are per-layer; no shared expert-weight object exists.
5. `CpuBatchEngine::reserve` chooses FCFS rows. Each `CpuStepRow` explicitly
   carries `Prefill` or `Decode`; a batch may mix phases. Reservation does not
   mutate generation, output, KV, or RNG state.
6. `CpuModel::prepare_step_inner` gathers BF16 embeddings and loops over every
   decoder layer. `forward_layer_batch` performs input RMSNorm; BF16 Q/K/V;
   YaRN RoPE; attention over committed plus step-staged KV with full/sliding
   visibility; attention output projection and residual; post-attention norm;
   MoE; then the final residual. New K/V stays in a per-step delta.
7. After final normalization and the BF16 LM-head projection, the engine samples
   from prepared logits. `CpuBatchEngine::commit` revalidates sequence revision,
   lifecycle, and cancellation, retains only valid rows, then commits all model
   deltas and publishes tokens/usage. Dropping or discarding `PreparedCpuStep`
   is a no-op.

### CPU MoE, exactly as implemented

`CpuModel::moe_batch` is the narrowest complete current routing-to-reduction
path found:

1. Project each normalized BF16 row through the layer's BF16 router; store f32
   logits after a BF16 round trip.
2. `stable_top_k` orders descending logit with expert index as tie-breaker.
3. Softmax only selected logits, then BF16-round each f32 route weight.
4. Represent routes as `CpuRoute { expert, source_row, rank, weight }`; stable
   sort by expert, preserving source-row/top-k order inside buckets.
5. For each nonempty expert bucket, copy its input rows and call
   `project_mxfp4_batch` for gate/up. BF16-round; apply GPT-OSS clipped SwiGLU;
   execute down; BF16-round.
6. Store `CpuRoutedExpertOutput`, restore source-row/rank order, multiply by the
   route weight in f32, accumulate in rank order, then BF16-round the MoE output.
7. Add the residual in the decoder-block caller. KV and sequence state remain
   uncommitted until the complete step succeeds.

`project_mxfp4_batch` is a **Hypothesis** for the narrow execution seam because
it receives one expert view and a row bucket. It is not currently device-neutral:
it is a `CpuModel` method, consumes CPU repack layouts and CPU execution context,
and contains Xe fallback/profiling behavior. Research must compare this boundary
with `gpt-oss-moe-semantics`, route grouping, and reduction ownership before any
design is selected.

### CPU kernels and installed-host distinction

`gpt-oss-cpu-kernels/src/features.rs` detects instruction capabilities and OS
state. `Kernels` exposes scalar, AVX2/FMA, and AVX-512 VNNI dispatch for BF16
matvec, activation quantization, MXFP4 dot/GEMV/matrix work, and RMSNorm.
`matmul.rs` implements scalar, AVX2, AVX-512 VNNI and feature-gated AMX-INT8
matrix backends with caller-owned scratch and scalar tail/fallback behavior.

**Verified:** the code and supplied gates test scalar/AVX2/AVX-512/AMX build
surfaces. **Verified separately:** this host exposes AVX2/FMA and the required
AVX-512 VNNI flags, but no AMX flags. Therefore the optimized Cascade
Lake-capable path is AVX-512 VNNI; AMX is unavailable at runtime here.

## CUDA execution flow

### Construction and weight lifetime

1. `GpuLLMEngine::new` validates a GPT-OSS Hugging Face config and one available
   CUDA device per requested tensor-parallel rank. It creates one `GpuWorker`
   per device, loads/profiles it, takes the minimum KV capacity, initializes
   NCCL, and gives the workers to `TensorParallelCoordinator`.
2. `GpuWorker::new` creates a CUDA context, one non-default runner stream and a
   cuBLAS handle. It disables cudarc event tracking because the active path uses
   one stream/graph. The device memory-pool release threshold is set to retain
   freed allocations for reuse. Errors propagate as typed `Result` values.
3. The production `gpu_loader` mmaps one shard at a time. Dense BF16 values are
   converted through temporary host vectors and uploaded as f32 and, for half
   mode, again as f16. U8 MXFP4 tensors are copied into an owned host map.
4. `build_gpt_oss_moe_layers` copies router/bias tensors back from device and
   clones each U8 tensor again into `GptOssMoeLayerWeights`. Decode preparation
   uploads the U8 payload and per-expert biases to the same runner device.
   The original host U8 map and layer U8 vectors remain owned. Thus the current
   CUDA path deliberately retains multiple host/device representations.
5. For TP>1, each worker first obtains complete GPU/host maps, downloads and
   slices tensors using name-based rules, then uploads rank slices. This is not
   a bounded direct-to-destination load.

### CUDA prefill

The GPU transformer shell runs embeddings, QKV, RoPE, attention, projections,
KV writes, residuals, norms, and logits on its runner device. For GPT-OSS MoE,
`gpu_layer.rs::forward` calls `clone_dtoh` for the entire normalized activation,
executes router, stable top-k, every selected MXFP4 projection, SwiGLU and
weighted reduction as scalar f32 host loops, then `clone_htod`s the whole MoE
output. The host owns routing and reduction. This path is not CPU-kernel reuse
and does not use pinned expert staging.

### CUDA decode

After `prepare_gpt_oss_graph_decode`, `forward_decode_gpu` keeps router logits,
selected IDs, selected weights and reduction on one GPU:

1. cuBLAS computes the router; bias is a CUDA kernel.
2. `gpt_oss_route_topk_kernel` produces device `i32` IDs and f32 weights.
3. The host launches a loop over **every local expert**, not only active experts.
   `gpt_oss_select_expert_inputs_kernel` creates a dense masked input and one
   route-weight row for that expert.
4. `gpt_oss_dequant_expert_f16_kernel` expands the full expert gate/up matrix to
   f16; cuBLAS executes it; a fused CUDA kernel performs SwiGLU.
5. The full expert down matrix is expanded to f16; cuBLAS executes it; the bias
   and `gpt_oss_weighted_add_kernel` update the device output.

The loop's temporary full matrices are 33,177,600 bytes for gate/up and
16,588,800 bytes for down at current GPT-OSS dimensions before allocator
alignment. Their exact overlap/high-water under the retaining CUDA pool is not
measured here. The four MoE kernels are the only MoE-specific modules among the
29 `sm_86` PTX files; expert GEMMs remain cuBLAS operations.

## Transfers, synchronization, and topology support in code

| Path | Present behavior | Status for heterogeneous experts |
|---|---|---|
| Model load H2D | Per-tensor cudarc uploads after host conversion/copy | Implemented, but not placement-aware or duplication-bounded. |
| CUDA prefill MoE D2H/H2D | Whole normalized activation D2H and whole MoE output H2D through `clone_*` | Implemented synchronous boundary; pageable, no overlap evidence. |
| Decode metadata H2D | Packed metadata buffer, one H2D update per step | Implemented for runner metadata, not expert scatter. |
| Greedy token D2H | `PinnedHostSlice<i32>` plus enqueued async D2H and later stream sync | Implemented narrow pinned output path only. |
| `PinnedBuffer` / `PinnedPool` | `gpt-oss-gpu/src/pinned_memory.rs`; real CUDA tests after the supplied context fix | Standalone/tested component; no production expert/activation consumer found. |
| CUDA KV block copy | Same-device CUDA copy kernels exist; other cache paths round-trip complete blocks through pageable host vectors | Partial, unrelated to expert placement. |
| GPU peer copy | No `memcpy_peer`, peer-enable, or placement transfer path found | Absent. |
| Streams/events | One active runner stream; event tracking deliberately disabled. Other wrapper streams/fields remain unused warnings. | No mechanism proven for safe CPU/GPU or GPU0/GPU1 expert overlap. |
| Synchronization | Explicit stream synchronizes before host reads and in debug timing; single-stream ordering otherwise | Implemented locally, no cross-device transaction barrier. |

## Multi-GPU reality

**Verified:** both devices can be enumerated and individually used by real CUDA
tests according to supplied evidence. `GpuLLMEngine` also has real source-level
multi-GPU machinery: per-device workers, NCCL communicator creation,
rank launch/collect, and all-reduce after attention output and MLP down
projection.

**Conflict / insufficient proof:** the current TP sharder splits GPT-OSS expert
gate/up block output dimensions and down block input-group dimensions, while
`GptOssMoeLayerWeights::forward_decode_gpu` uses global `hidden_size` and
`intermediate_size` to index/dequantize each local slice. Bias chunking also
uses global widths. No real TP test or retained full-model TP capture was found.
Therefore this is `partial/scaffold`, not validated multi-GPU GPT-OSS execution.
The later research stage must run a minimal dimension/index audit before any
expensive model attempt.

**Verified absent:** no runtime chooses a CPU expert and GPU experts inside one
operation; no per-layer, per-tensor, or per-expert device-placement object
exists; no peer transfer path exists; and the CPU cannot participate in NCCL.

## Memory, cleanup, and failure semantics

### CPU

`CpuMemoryDescriptor` accounts mapped checkpoint files, mapped repacks, virtual
mapped bytes, selected cache bytes, KV, staged-KV high-water, metadata, matrix
scratch and Xe bytes. The engine `MemoryBroker` grants/refunds bounded classes.
`CpuExecutionContext` owns reusable matrix scratch. Model mappings/repack maps
are immutable and shared; state deltas are request/step-owned. Prepare failures,
cancellation and stale revisions discard uncommitted deltas, and batch commit
validates all retained rows before mutation.

### CUDA

CUDA slices are RAII-owned and errors are typed, but the default device pool is
configured to retain freed memory. Capacity planning primarily profiles free
VRAM and reserves KV block counts; no unified per-rank accounting enumerates
all weight copies, MoE dequant scratch, pinned staging and NCCL workspace.
Abort handling prevents further scheduler output and recycles dead KV blocks,
but GPU layer/KV work may already have executed. No cross-worker or CPU/GPU
rollback/commit protocol was found for a failure after partial multi-device
execution. Its retry safety is **Unknown**.

## Telemetry and differential diagnosis

- CPU profiling is per phase, operation, layer and expert-bucket shape, with
  transaction and memory high-water fields. It does not record a device,
  expert ID, transfer bytes, overlap, or synchronization event.
- CUDA worker metrics cover request/forward/sample latency and throughput.
  `GPT_OSS_TRACE_LAYER_TIMINGS` forces synchronization and logs coarse layer
  stages in milliseconds. It is not an immutable per-device transfer trace.
- CPU opt-in traces expose every semantic MoE boundary needed for a one-layer
  oracle. `cpu_parity` supports prefill and a chosen retained decode step;
  forced-token/restricted tools preserve identical continuation context and
  localize the first mismatch.
- `restricted_prefill_trace` and its PyTorch comparison expose CUDA activation
  stages and selected experts for a restricted model, but do not constitute a
  20B/120B heterogeneous oracle.
- `RunManifestV1`, `TimerEvidence`, campaign indexes and bounded diagnostics can
  reference future placement/transfer artifacts. A placement-aware record
  schema is missing; research should prefer extending this system over creating
  a parallel evidence root.

## Harmony/service boundary

`HarmonyProtocol` owns GPT-OSS prompt rendering, completion parsing and partial
stream parsing. Chat, Responses, and tool routes call it, and current response
types derive prompt/completion counts from token arrays. The service lifecycle,
bounded delivery, cancellation, readiness and shutdown mechanisms are present.

**Conflict:** historical live captures prove process/HTTP lifecycle, while the
latest supplied evidence reports parse-invalid, empty/unusable or malformed
responses. No Harmony-native parse-and-token-accounting control on this host is
retained. Heterogeneous model validation can and should remain isolated from
that HTTP gap; serving must not be called semantically correct until the narrow
control passes.

## Reuse and specificity map

This classification is evidence, not permission to refactor.

| Component / boundary | Classification | Evidence and limitation |
|---|---|---|
| Stable top-k, selected softmax, route grouping/order in `gpt-oss-moe-semantics` | Reusable as-is candidate | No CUDA/x86/host ownership; semantics are intentionally sparse-MoE-shaped. GPT-OSS tie/BF16 contract still belongs at the caller boundary. |
| `RunManifestV1`, campaign index, artifact hashing/redaction | Reusable as-is candidate | Model/device-neutral evidence containers; missing heterogeneous field vocabulary can be an attached artifact. |
| `PreparedCpuStep` prepare/discard/commit discipline | Reusable after adaptation candidate | Valuable transaction semantics, but types own CPU KV/model deltas and CPU scheduler identities. |
| `CpuRoute` grouping and `project_mxfp4_batch` call boundary | Reusable after adaptation candidate | A useful bucket/expert seam exists; API, layouts, profiling, scratch and ownership are CPU/Xe-shaped. |
| `CpuTensorStore` mmap/index table | Reusable after adaptation candidate | Read-only borrowed storage is useful; top-level enumeration, tensor dtypes/names and source hashing are current CPU/GPT-OSS assumptions. |
| `PinnedBuffer` / `PinnedPool` | Reusable after adaptation candidate | Backend utility is generic over POD types, but production ownership/stream integration is absent. |
| `TensorParallelComm` | Reusable after adaptation candidate | Communication interface exists, but only local/NCCL all-reduce and gather; no CPU rank, P2P placement or rollback. |
| CPU decoder/MoE traces and comparison tools | Reusable after adaptation candidate | Rich semantic checkpoints; schema currently CPU/Harmony and lacks placement/transfer data. |
| `CpuGptOssConfig`, `load_layer`, tensor namespace | GPT-OSS-specific | Architecture, transformed names, GQA/YaRN/layer types and MXFP4 shapes are embedded. |
| `moe_batch` arithmetic and `apply_gpt_oss_swiglu` | GPT-OSS-specific | BF16 boundary, clipped SwiGLU, top-k and residual order encode model semantics. |
| Harmony render/parse routes | GPT-OSS-specific | Protocol and special-token semantics are intentional, complete scope. |
| CPU repack and `gpt-oss-cpu-kernels` | x86/backend-specific | x86 capability dispatch, Q8/residual-Q8, 17-byte records, layouts and caller scratch are embedded. |
| Xe attachment/fallback/cache | x86/backend-specific | Host/Xe prefill policy and frozen Tiger Lake evidence; deferred. |
| `GpuWorker`, `GpuModelRunner`, CUDA cache/graphs | CUDA/SM86-specific | CUDA context/stream/cuBLAS/PTX ownership; current build evidence targets `sm_86`. |
| `GptOssMoeLayerWeights::forward_decode_gpu` | CUDA/SM86-specific and GPT-OSS-specific | Device buffers, PTX names, full expert dequant, cuBLAS and GPT-OSS dimensions are embedded. |
| `model_loader/shard.rs` GPT-OSS TP semantics | Unknown | Shard rules exist but appear dimension-inconsistent with the consuming MoE; a bounded source/runtime test must resolve this. |
| Generic `bridge.rs` buffer/allocator/stream boundary | Unknown | Explicit TODOs show it is not the production CPU or CUDA owner; adopting it would require evidence, not aesthetics. |
