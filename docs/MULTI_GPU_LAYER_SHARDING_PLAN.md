# Multi-GPU Layer Sharding Plan

## Purpose

Design and prototype two-GPU layer sharding across GPU0/GPU1 without changing
per-layer math. The target layout is:

- GPU0: embeddings and layers 0..11
- GPU1: layers 12..23, final norm, and LM head

This first slice is reconnaissance only. It identifies the current
single-device assumptions, ownership boundaries, and the smallest future
`DeviceMap` design that can be introduced without changing default runtime
behavior.

## Non-goals

- No tensor parallelism design or implementation.
- No NCCL, collectives, all-reduce, or peer collectives.
- No splitting QKV, MLP, experts, or individual tensors.
- No CUDA kernel changes.
- No production default behavior changes.
- No ordered-bundle generation or new oracle artifact generation.
- No final-token or final-logit parity claims in this lane.
- No 4097-token validation work.

## Current single-device assumptions

The production CUDA path is organized around one worker-selected CUDA device.
`crates/gpt-oss-engine/src/worker/gpu_worker.rs` creates a single
`CudaContext::new(device_id)`, one primary non-default `CudaStream`, and a
`CublasHandle` bound to that stream during `GpuWorker::new`.

`GpuWorker::load_weights` loads model tensors through
`model_loader::gpu_loader::*` using `self.stream`, so every uploaded
`CudaSlice` belongs to the worker's one context/device. `GpuWorker` preserves
the raw GPU maps for deferred `GpuModelRunner` construction, then
`build_gpu_model_runner` passes the same context and stream into:

- `CudaCacheEngine::new`
- `KernelLoader::new`
- `CublasHandle::new`
- `GpuModelRunner::new`

`GpuModelRunner` then stores one `device`, one `stream`, one `blas`, one
`loader`, one `GpuModelWeights`, one `CudaCacheEngine`, and one `Vec` of
`GpuTransformerLayer`. Its forward paths iterate `self.layers` in order and
reuse `self.stream`, `self.blas`, `self.loader`, and `self.cache.gpu_cache()`
for every layer.

The runtime therefore assumes not a process-global CUDA device singleton, but a
single device per CUDA `GpuWorker` and per `GpuModelRunner` instance. The
important sharding blocker is that all tensors, layers, kernels, cuBLAS calls,
metadata buffers, RoPE buffers, scratch buffers, and KV cache slices inside a
runner share one context/stream/device.

## CUDA ownership map

- Context: `GpuWorker::new` owns one `Arc<CudaContext>` from
  `CudaContext::new(device_id)`. `GpuModelRunner` and `CudaCacheEngine` receive
  clones of that same context.
- Stream: `GpuWorker::new` owns one primary `Arc<CudaStream>` from
  `context.new_stream()`. Weight upload, metadata upload, layer execution,
  final norm, LM head, graph output, and cache allocation currently use this
  stream.
- cuBLAS: `gpt-oss-gpu/src/cublas.rs` wraps `CudaBlas` plus the stream used to
  create it. `CublasHandle::new(stream.clone())` means every GEMM issued through
  a handle is bound to that handle's stream/device.
- Kernel modules: `gpt-oss-gpu/src/kernel_loader.rs` stores one
  `Arc<CudaContext>`, one `Arc<CudaStream>`, and a module map. PTX modules are
  loaded into that context. Raw launches bind the loader context to the current
  thread before launching on the loader's stream or an explicitly supplied
  stream.
- Layers: `GpuTransformerLayer` stores one layer config, one stream clone, and
  one `Arc<KernelLoader>`. Weights are not owned by the layer; they are borrowed
  per call from `GpuModelRunner::layer_weights*`.
- Scratch: f16 layer scratch is owned once by `GpuModelRunner` and reused
  across all layers on the single stream.

## Model tensor placement map

`crates/gpt-oss-model-runner/src/model_loader/gpu_loader.rs` memory maps
safetensors, converts as needed, and uploads each tensor with
`stream.clone_htod`. `GpuModelWeights` stores the resulting `CudaSlice<f32>` and
`CudaSlice<f16>` maps plus shape metadata. The container does not record a
device id; device placement is implicit in the stream/context used for upload.

Current placement is all-or-nothing:

- embeddings: `model.embed_tokens.weight` uploaded to the worker device
- per-layer weights: all `model.layers.{i}.*` tensors uploaded to the worker
  device
- final norm: `model.norm.weight` uploaded to the worker device
- LM head: `lm_head.weight`, or tied embedding fallback, uploaded to the worker
  device
- GPT-OSS U8 expert blocks/scales: loaded to host first, then selected GPU
  copies are uploaded through `GpuModelRunner::build_gpt_oss_moe_layers` and
  `upload_gpt_oss_moe_gpu_weights` on the runner stream
- fused f16 weights and converted norm/bias/embed tables: allocated and copied
  on `GpuModelRunner::stream` during `fuse_weights`

For layer sharding, the clean future model is not tensor slicing. It is filtered
whole-tensor loading per device:

- GPU0 weight container: embeddings plus tensors for layers 0..11
- GPU1 weight container: tensors for layers 12..23 plus final norm and LM head
- optional tied embedding handling: GPU1 still needs an LM-head-compatible
  weight if `lm_head.weight` is absent and the model uses tied embeddings

## KV cache ownership / layer indexing notes

`crates/gpt-oss-model-runner/src/kv_cache/engine_cuda.rs` allocates
`gpu_cache: Vec<(CudaSlice<f16>, CudaSlice<f16>)>` with one key/value pair per
layer. The allocation loop uses local layer indices `0..num_layers`, and each
layer buffer has shape:

`[num_gpu_blocks, block_size, num_kv_heads, head_dim]` flattened.

The forward paths index cache entries with the absolute runtime layer index:

`let (key_cache, value_cache) = &gpu_cache[layer_idx];`

The attention metadata buffers (`block_tables`, `slot_mapping`, `context_lens`,
`seq_start_pos`) are uploaded once to the runner's packed metadata buffer and
are shared by all layers. They index token/block positions, not device
ownership.

For layer sharding, each device should own KV buffers only for its assigned
layers, while preserving absolute layer ids in logs, validation traces, and
weight lookup names. The cache access API should therefore avoid assuming
`gpu_cache[absolute_layer_idx]` once layers are split. A future shard-local cache
can either:

- store `(absolute_layer_idx, key, value)` entries and look up by absolute layer
  id, or
- store a shard-local vector and carry an `absolute_to_local_layer` mapping.

The second option is smaller, but the first is harder to misuse during
validation.

## Proposed DeviceMap syntax

The first inert parser only needs two forms:

- `--device-map single`
- `--device-map split:0-11@0,12-23@1`

Semantics:

- `single` is the default and must exactly preserve current behavior.
- `split:` contains comma-separated inclusive layer ranges with CUDA device ids.
- Ranges must cover every layer exactly once.
- Ranges must be ordered by layer index for Stage 1.
- Embeddings are placed with the first layer's device.
- Final norm and LM head are placed with the last layer's device.
- No tensor-level placement syntax is accepted.
- Invalid, overlapping, or incomplete maps fail before CUDA allocation.

Minimum inert data model:

```text
DeviceMap {
  devices: [0] or [0, 1],
  layer_device: Vec<DeviceId>,       // len == num_layers
  embedding_device: DeviceId,
  final_device: DeviceId,
}
```

The parser can exist behind CLI/config plumbing with `single` as the default,
but the production runner should keep constructing the current single-device
path until a later stage explicitly consumes non-single maps.

## Proposed runtime insertion points

- CLI/config: add `device_map: Option<String>` or defaulted `DeviceMapConfig`
  near existing worker/device configuration, with default `single`.
- Parsing/validation: parse after model config is known, because the parser
  needs `num_layers` to verify coverage.
- Worker construction: keep `GpuWorker::new` unchanged for `single`. A later
  split path should construct per-device CUDA resources, not overload the
  existing single `context`/`stream` fields.
- Weight loading: add a placement-aware filter above
  `load_weights_to_gpu_with_shapes_filtered` so each device uploads only whole
  tensors it owns.
- Runner ownership: introduce a sharded runner wrapper that owns per-device
  shard runtimes. Avoid teaching the existing `GpuModelRunner` to contain mixed
  device slices in one struct.
- Layer loop: the clean activation-transfer seam is between the last layer on
  one device and the first layer on the next device, immediately after the
  producing layer returns its hidden state and before the consuming layer builds
  `GpuLayerInput`.
- Final head: keep final norm and LM head on the final shard so the last
  activation handoff is only layer-to-layer, not layer-to-head.

## Activation transfer design sketch

The first split handoff can use explicit device-to-host/device-to-device
staging and synchronization. It should favor simplicity and observability over
throughput.

For a boundary such as layer 11 -> layer 12:

1. GPU0 completes layer 11 on its stream.
2. Synchronize the GPU0 stream for the boundary smoke path.
3. Copy hidden state `[num_tokens, hidden_size]` from GPU0 to host pinned or
   normal staging memory.
4. Copy that host buffer to a freshly allocated or reusable hidden-state buffer
   on GPU1's stream.
5. Synchronize GPU1 or insert the necessary stream ordering.
6. Continue layer 12 with unchanged per-layer math.

This avoids NCCL, collectives, and tensor splitting. A later optimization can
replace host staging with peer `cudaMemcpyPeerAsync` if peer access is available,
but the validation design must not depend on peer access for the first smoke.

The f16 path needs the same policy for f16 hidden states and must preserve the
existing delayed residual/MLP behavior around `prev_mlp_out`. The boundary is
cleanest after the layer API returns the hidden representation that the next
layer already consumes today.

## Validation strategy

Stage A: no-op single-device `DeviceMap` produces identical behavior. This
should be a default-behavior guard, not a parity claim about new math.

Stage B: two-device allocation smoke proves ownership and placement. It should
log or assert that embeddings/layers 0..11 are allocated on GPU0 and layers
12..23/final head are allocated on GPU1. This can run without comparing logits.

Stage C: split activation handoff smoke proves hidden-state transfer at the
layer 11 -> 12 boundary for a small prompt. It should verify transfer shape,
dtype, device placement, and successful continuation.

Stage D: compare existing layer/coarse validation outputs where available. This
reuses existing seam validation artifacts without generating ordered bundles in
this lane.

Stage E: only later attempt final-token/logit parity claims, after allocation,
handoff, and layer/coarse validation are stable.

## Stage classifications

- Stage 1 reconnaissance: docs-only ownership map and insertion-point design.
- Stage 2 inert parser: accept `single` and validate split syntax without using
  split maps for production execution.
- Stage 3 single-device plumbing: pass `DeviceMap::single` through config and
  construction with no behavior change.
- Stage 4 split allocation smoke: create per-device resources and upload
  whole-layer tensors to their owning devices.
- Stage 5 activation handoff smoke: run a small split forward through the first
  boundary transfer.
- Stage 6 validation reuse: compare existing layer/coarse outputs where
  artifacts exist.
- Stage 7 final-token/logit work: deferred until the split runtime is otherwise
  proven.

## Risks / blockers

- `GpuModelRunner` is structurally single-device. Mixing device slices into it
  would be fragile because one stream, cuBLAS handle, loader, scratch set, RoPE
  table, metadata buffer, and cache engine are assumed throughout.
- `GpuModelWeights` does not record device identity. A placement-aware loader
  needs resource ownership outside the current map type.
- `CudaCacheEngine` indexes cache by absolute layer position today. Shard-local
  cache allocation needs an explicit absolute-to-local mapping or absolute-keyed
  cache entries.
- f16 fused weights, f16 scratch, converted embeddings, final norm, and GPT-OSS
  MoE GPU uploads are late allocations that must follow device ownership too.
- Tied LM head fallback may require embedding weights on the final device even
  when embeddings execute on GPU0.
- CUDA graph capture likely remains single-device-only until split execution is
  proven without graph capture.
- Host-staged activation transfer is simple but slow. That is acceptable for
  Stage C smoke and not a performance claim.

## Next bounded step

Implement an inert `DeviceMap` parser and config surface for:

- `--device-map single`
- `--device-map split:0-11@0,12-23@1`

The implementation should default to `single`, reject invalid split maps before
CUDA allocation, and leave all existing runtime construction and execution
unchanged.

## Stage 2 status

Implemented an inert, CUDA-free parser in
`crates/gpt-oss-model-runner/src/device_map.rs` and exported it from
`gpt-oss-model-runner` as `DeviceMap`, `DeviceId`, and `DeviceMapError`.

Supported parser inputs:

- `single`
- `split:0-11@0,12-23@1`

Behavior:

- Default runtime behavior is unchanged because the parser is not consumed by
  `GpuWorker`, `GpuModelRunner`, or serve execution yet.
- `single` can be parsed into a map that assigns all layers, embeddings, final
  norm, and LM head to the selected/current device.
- `split:` can be parsed and validated as placement intent only. It assigns
  embeddings to the device of layer 0 and final norm / LM head to the device of
  the final layer.
- Split maps remain non-executable in this slice. No CUDA allocation,
  multi-context construction, tensor upload routing, activation transfer, peer
  copy, NCCL, or runtime branching was added.
- CLI/config wiring is deferred. This keeps the current serve path unchanged
  and avoids introducing a split-map flag before there is a guaranteed
  pre-allocation rejection point in the engine startup path.

Primary classification:

multi_gpu_layer_sharding_parser_complete_config_surface_deferred

## Stage 3 status

Threaded `DeviceMap::single` through the existing single-device CUDA runner
construction boundary without adding split execution.

What changed:

- `GpuWorker::build_gpu_model_runner` constructs `DeviceMap::single` after
  `mr_config.num_layers` is known, using the existing selected `self.device_id`.
- `GpuModelRunner::new` receives the map, validates it with
  `DeviceMap::validate_single_device_executable`, and stores it as inert
  metadata on `GpuModelRunner`.
- `GpuModelRunner::device_map()` exposes the stored metadata for future
  inspection.
- The validator accepts only maps where embeddings, every layer, final norm, and
  LM head are on the same device. Split maps return
  `split device maps are parsed but not executable yet`.

Behavior:

- Default runtime behavior is unchanged.
- No CLI/config surface was added.
- Runtime execution still uses the existing single context, stream, cuBLAS
  handle, kernel loader, weight container, scratch buffers, RoPE tables,
  metadata buffers, and KV cache path.
- Split maps remain parser-only and non-executable. No multi-context allocation,
  weight routing, KV cache change, activation transfer, peer copy, NCCL, or
  runtime math branching was added.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo check -p gpt-oss-engine`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_single_device_map_plumbing_complete

## Stage 3.5 status

Added a pre-CUDA config/serve boundary for device-map specs while keeping split
maps non-executable.

What changed:

- `DeviceConfig` now carries an inert `device_map` string with default
  `single`.
- The serve CLI accepts `--device-map <SPEC>` and passes it into
  `EngineConfig.device.device_map`.
- `validate_device_map_spec_for_cuda_startup` lives in
  `crates/gpt-oss-engine/src/config/device.rs`. It reuses the model-runner
  `DeviceMap` parser instead of duplicating parser logic.
- `GpuLLMEngine::new` calls the helper after reading HuggingFace `config.json`
  for `num_hidden_layers` and before constructing any `GpuWorker`.

Behavior:

- Omitted device map: resolves to `single` and preserves current behavior.
- `--device-map single`: resolves to a single-device map for the selected CUDA
  device and preserves current behavior.
- `--device-map split:0-11@0,12-23@1`: parses placement intent, then fails with
  `split device maps are parsed but not executable yet`.
- Invalid split specs fail with parser validation errors before runtime worker
  construction.
- A single-device map that does not match the selected CUDA device is rejected
  by the helper.

Pre-CUDA ordering:

- Rejection happens before `GpuWorker::new`, before `CudaContext::new`, before
  stream creation, before cuBLAS handle creation, before kernel loader creation,
  before GPU weight upload, and before KV cache allocation.

Still deferred:

- Split maps remain non-executable.
- No multiple CUDA contexts, tensor routing, KV cache changes, activation
  transfer, peer copy, NCCL, collectives, or runtime math branching were added.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-engine device_map`
- `cargo check -p gpt-oss-engine`
- `cargo check -p gpt-oss-server`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_split_map_rejected_before_cuda_allocation

## CUDA context ownership refactor design

### Design intent

Future split allocation should preserve the current single-device
`GpuWorker`/`GpuModelRunner` path and add a separate sharded ownership path.
The existing runner is coherent because every `CudaSlice`, kernel launch,
cuBLAS call, metadata buffer, RoPE table, scratch buffer, and KV cache entry is
implicitly tied to one `CudaContext` and one execution stream. The split design
should keep that invariant inside each shard instead of teaching one
`GpuModelRunner` to contain mixed-device pointers.

The first split target remains whole-layer placement only:

- GPU0 owns embeddings and layers 0..11.
- GPU1 owns layers 12..23, final norm, and LM head.
- QKV, MLP, experts, and individual tensors are never split in this lane.
- Split maps remain non-executable until a later smoke explicitly constructs
  per-device resources.

### Proposed shard resource ownership

Introduce a per-device ownership type for the split path. The clearest name is
`GpuShardResources`, with a possible wrapper named `ShardedGpuModelRunner`.

Design sketch:

```text
ShardedGpuModelRunner {
  config: ModelRunnerConfig,
  device_map: DeviceMap,
  shards: Vec<GpuShardResources>,
}

GpuShardResources {
  device_id: DeviceId,
  absolute_layers: Vec<usize>,
  owns_embeddings: bool,
  owns_final_head: bool,

  context: Arc<CudaContext>,
  stream: Arc<CudaStream>,
  blas: CublasHandle,
  loader: Arc<KernelLoader>,

  weights: GpuModelWeights,
  cache: ShardCudaCache,
  layers: Vec<GpuTransformerLayer>,

  rope_cos: CudaSlice<f32>,
  rope_sin: CudaSlice<f32>,
  meta_packed: ReusableGpuBuf,
  graph_output: Option<CudaSlice<i32>>,
  cpu_scratch: Vec<i32>,

  fused_qkv_weights: Vec<CudaSlice<f16>>,
  fused_gate_up_weights: Vec<CudaSlice<f16>>,
  fused_layernorm_f16: Vec<CudaSlice<f16>>,
  fused_post_norm_f16: Vec<CudaSlice<f16>>,
  fused_qkv_bias_f16: Vec<Option<CudaSlice<f16>>>,
  fused_o_proj_bias_f16: Vec<Option<CudaSlice<f16>>>,
  final_norm_weight_f16: Option<CudaSlice<f16>>,
  embed_tokens_f16: Option<CudaSlice<f16>>,
  f16_scratch: Option<F16LayerScratch>,

  gpt_oss_moe_layers: Vec<Option<GptOssMoeLayerWeights>>,
}
```

Objects that must be per-device:

- CUDA device id.
- `CudaContext`.
- Primary `CudaStream` and any cache/compute stream wrappers.
- `CublasHandle`, because it is created from a stream.
- `KernelLoader`, because modules are loaded into a specific context and
  launches use that loader's context/stream.
- `GpuModelWeights` or a shard-local equivalent, because `CudaSlice` placement
  is implicit in the upload stream/context.
- `CudaCacheEngine` or a shard-local cache object.
- `GpuTransformerLayer` instances for the shard's absolute layers.
- RoPE tables, metadata buffers, graph output buffers, CPU packing scratch that
  is paired with shard metadata, f16 scratch, fused f16 weights, converted
  embeddings/final norm/LM-head tensors, and GPT-OSS MoE GPU uploads.

Embedding and final-head placement should be explicit booleans or ownership
fields, not inferred from the presence of a tensor after construction. That
keeps tied-embedding fallback visible: if `lm_head.weight` is absent, the final
shard may still need an LM-head-compatible copy even though token embeddings
execute on the first shard.

### Shared/global state

The following can remain outside per-device resources:

- HuggingFace/model config and derived `ModelRunnerConfig`.
- Tokenizer, protocol types, scheduler/request state, and sampling policy.
- CPU-side safetensor headers, tensor shape metadata, and placement filters.
- Host-side raw tensor views or memory maps, if they are used only as upload
  sources and not mutated by a shard.
- Validation/logging metadata and small status reports.
- The parsed `DeviceMap` and derived layer-to-shard ownership tables.
- Split allocation planning data such as total bytes per shard and tensor-name
  manifests.

Any host staging buffer used for activation transfer can be owned by the
sharded wrapper rather than by an individual shard, because it represents the
boundary between two devices.

### Why not mixed-device GpuModelRunner

`GpuModelRunner` currently has one `weights`, one `cache`, one `blas`, one
`loader`, one `device`, one `stream`, one `layers` vector, one metadata buffer
set, and one f16 scratch set. The forward paths assume those fields are
compatible with every layer. `GpuModelWeights` also does not record device
identity, so a mixed-device runner would make pointer provenance implicit and
easy to misuse.

Refactoring `GpuModelRunner` into a mixed-device container would spread
device-routing checks through hot execution paths before split allocation has
even been smoked. A wrapper keeps the proven single-device runner model intact:
one shard is one coherent CUDA universe, and the sharded layer only decides
which shard owns a whole layer and where activation handoff occurs later.

### Proposed constructor boundaries

Keep the current path unchanged for `single`:

- `GpuLLMEngine::new` validates the default/single map.
- `GpuWorker::new` creates one CUDA context and stream.
- `GpuWorker::load_weights` uploads weights through that stream.
- `GpuWorker::build_gpu_model_runner` constructs the existing
  `GpuModelRunner`.

For future split allocation, branch before `GpuWorker::new` instead of bending
`GpuWorker` into a multi-device owner. The split path should be a new
constructor boundary, such as:

- `ShardedGpuWorker::new(config, device_map)`, or
- `ShardedGpuModelRunner::allocate_from_plan(model_dir, config, device_map)`.

That constructor should:

1. Build a shard plan from `DeviceMap`.
2. Create `GpuShardResources` independently for each device.
3. Use existing loader primitives with shard-specific whole-tensor filters.
4. Construct shard-local layers with absolute layer ids in their configs.
5. Return a non-executing allocation object or status for Stage 4.

`GpuWorker::load_weights` and `GpuWorker::build_gpu_model_runner` should remain
the single-device implementation. The first split allocation smoke can share
small helper functions, but it should not require the current worker to hold
multiple contexts.

### Future Stage 4 split allocation smoke

The minimal Stage 4 smoke should prove ownership and allocation, not inference:

- Parse `split:0-11@0,12-23@1` after model config is known.
- Create CUDA resources on GPU0 and GPU1.
- Load only whole tensors owned by each shard:
  - GPU0: embeddings and layers 0..11.
  - GPU1: layers 12..23, final norm, and LM head or tied LM-head fallback.
- Allocate shard-local RoPE tables, metadata buffers, f16 conversion outputs,
  GPT-OSS MoE uploads, and KV cache only for layers owned by the shard.
- Do not run layer forward, sampling, final norm, or LM head.
- Emit a small status JSON or structured log with device ids, absolute layer
  ranges, tensor counts, and allocation byte totals.
- Do not commit the status output or any generated traces/artifacts.

This smoke must not make final-token, logit, or parity claims. Success means
the runtime can construct two independent CUDA ownership islands according to
the map without corrupting the existing single-device path.

### KV cache ownership design

The safest first split cache design is absolute-keyed, even if the physical
storage is shard-local:

```text
ShardCudaCache {
  entries: Vec<LayerCacheEntry>,
}

LayerCacheEntry {
  absolute_layer_idx: usize,
  key: CudaSlice<f16>,
  value: CudaSlice<f16>,
}
```

Access should go through an explicit method such as
`cache_for_absolute_layer(layer_idx)`. A local vector plus
`absolute_to_local_layer` map is leaner, but it preserves the exact footgun the
current code has for sharding: `gpu_cache[layer_idx]` looks valid while using an
absolute index against local storage. Absolute-keyed entries make validation
logs and layer/coarse seam comparisons easier to reason about.

A later performance pass can add an `absolute_to_local_layer` table internally
after the accessor boundary is established and tested.

### Future activation handoff contract

The Stage 5 boundary is between the output of layer 11 on GPU0 and the input of
layer 12 on GPU1.

Contract:

- Tensor: hidden state that the next layer already consumes today.
- Shape: `[num_tokens, hidden_size]`.
- Dtype: match the active path, `f32` for the existing f32 path and `f16` for
  the f16 path.
- Producer: GPU0 stream after layer 11 finishes.
- Consumer: GPU1 stream before layer 12 starts.
- First synchronization policy: synchronize the producer stream, copy DtoH into
  host staging memory, copy HtoD into a GPU1 buffer, then synchronize or order
  the GPU1 stream before launch.
- Peer copy can be explored later if available, but the first contract is
  host-staged and requires no NCCL, collectives, or peer access.

The f16 path also has to preserve the delayed residual and `prev_mlp_out`
semantics. The handoff should therefore occur only after the producing layer
has returned the same representation that the current single-device loop would
pass to the next layer.

### Risks / blockers

- `GpuModelWeights` has no device identity, so split loading needs ownership at
  the shard container boundary or a typed shard-local weight container.
- Tied LM-head fallback can require duplicating embedding-compatible data on
  the final shard.
- GPT-OSS MoE uploads happen after initial weight loading and must follow the
  same shard ownership filter as layer tensors.
- CUDA graph capture should remain disabled for split execution until
  allocation and host-staged handoff are proven without graphs.
- The current cache API exposes layer-indexed slices directly; split execution
  needs an accessor that names absolute layer ids.
- Stage 4 requires two visible CUDA devices, so CI coverage may need a
  GPU-optional status test plus a manually run two-device smoke.

### Next bounded step

Add a non-executing per-device shard resource skeleton or allocation planner
that can build and print a split allocation plan without creating mixed-device
`GpuModelRunner` state. The planner should define tensor ownership filters
before any code uploads split weights.

### Primary classification

multi_gpu_layer_sharding_sharded_runner_wrapper_design_complete

## Stage 4 planning skeleton status

Added a CUDA-free sharded model planning skeleton in
`crates/gpt-oss-model-runner/src/shard_plan.rs` and exported it from
`gpt-oss-model-runner`.

New pure planning types:

- `ShardedModelPlan`
- `GpuShardPlan`
- `TensorPlacement`
- `TensorPlacementReason`
- `ShardPlanError`

What it proves:

- A parsed `DeviceMap` can be converted into stable per-device ownership
  metadata without constructing CUDA resources.
- Absolute layer ids are preserved in each shard plan.
- `single` produces one shard with all layers, embeddings, final norm, and
  LM-head ownership on the selected device.
- `split:0-11@0,12-23@1` produces two shard plans:
  - GPU0 owns embeddings and layers 0..11.
  - GPU1 owns layers 12..23, final norm, and LM head.
- The plan is metadata only. It is not wired into serve execution and does not
  make split maps executable.

Tensor ownership patterns recognized:

- `model.embed_tokens.weight` maps to the embedding shard.
- `model.layers.<N>.*` maps to the shard owning absolute layer `N`.
- `model.norm.weight` maps to the final shard.
- `lm_head.weight` maps to the final shard.
- Unknown tensor names are left unassigned.
- Out-of-range layer tensor names are rejected by the pure planner.
- Tied LM-head fallback is represented as future final-shard ownership, but no
  fallback upload is implemented.

Still deferred:

- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  shard-local GPU resources are created.
- No model loader upload behavior changed.
- No tensors are routed to multiple devices.
- KV cache allocation and indexing are unchanged.
- No activation transfer, peer copy, NCCL, collectives, CUDA kernel changes, or
  runtime math branching were added.
- Split maps remain rejected before CUDA allocation in the current serve path
  with `split device maps are parsed but not executable yet`.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner shard`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_shard_plan_skeleton_complete

## Filtered whole-tensor loading design

### Current loader behavior

The current CUDA loader in
`crates/gpt-oss-model-runner/src/model_loader/gpu_loader.rs` is a
stream/context-local uploader. Device placement is implicit in the
`Arc<CudaStream>` passed to the loader.

Tensor discovery:

- `load_weights_to_gpu_with_shapes` delegates to
  `load_weights_to_gpu_with_shapes_filtered(path, stream, |_| true)`.
- `load_weights_to_gpu_with_shapes_filtered` dispatches to
  `load_single_to_gpu` for one safetensors file or `load_sharded_to_gpu` for a
  directory.
- `load_single_to_gpu` memory maps the file, parses the safetensors header with
  `parse_safetensors_header`, and enumerates tensor names from that header.
- `load_sharded_to_gpu` calls `collect_shards`, sorts `*.safetensors` files,
  and merges per-file weight and shape maps.

Conversion and upload:

- The f32 path skips `U8` tensors, converts `F32`/`F16`/`BF16` data to host
  `Vec<f32>` via `convert_to_f32`, then uploads with `stream.clone_htod`.
- The f16 path uses `load_weights_to_gpu_f16_with_shapes`, converts
  `F16`/`BF16`/`F32` data to host `Vec<f16>` via `convert_to_f16`, then uploads
  with `stream.clone_htod`.
- `load_u8_weights_to_host` separately loads raw `U8` GPT-OSS expert
  block/scale tensors into host memory. Those are not uploaded by the f32/f16
  loader loops.

Shape metadata:

- Both f32 and f16 loader paths return `HashMap<String, Vec<usize>>` shape
  metadata.
- The f32 filtered path preserves shape metadata even when a tensor upload is
  skipped.
- Shape metadata is currently merged across safetensor shards and later
  inserted into `GpuModelWeights`.

Existing filtering hooks:

- The f32 path already has a `should_load: FnMut(&str) -> bool` hook through
  `load_weights_to_gpu_with_shapes_filtered`.
- The f16 path currently has no equivalent filter and uploads every non-`U8`
  tensor it sees.
- The U8 host path currently has no filter and returns all raw `U8` tensors.

Late allocations outside initial safetensor upload:

- `GpuModelRunner::new` uploads RoPE tables and creates per-runner metadata
  buffers.
- `CudaCacheEngine::new` allocates KV cache buffers from `num_layers`.
- `GpuModelRunner::build_gpt_oss_moe_layers` reads layer-owned U8 tensors and
  uploads GPT-OSS MoE bias chunks through the runner stream.
- `prepare_gpt_oss_graph_decode` can upload GPT-OSS expert blocks/scales later
  from host U8 storage.
- `GpuModelRunner::fuse_weights` creates fused QKV, fused gate/up, converted
  f16 layernorm/bias/final norm/embed tensors, and f16 scratch on the runner
  stream.

### Upload manifest concept

Future split allocation should derive a pure upload manifest before creating
CUDA resources. The manifest should be built from:

- model config, including `num_layers`, architecture, dtype, and
  `tie_word_embeddings`
- parsed `DeviceMap`
- `ShardedModelPlan`
- safetensor header names, dtypes, and shapes

Design sketch:

```text
ShardedUploadManifest {
  model_num_layers: usize,
  shards: Vec<ShardTensorManifest>,
  unassigned_tensor_names: Vec<String>,
  invalid_tensor_names: Vec<String>,
}

ShardTensorManifest {
  device_id: DeviceId,
  absolute_layers: Vec<usize>,
  required_tensor_names: Vec<String>,
  optional_tensor_names: Vec<String>,
  host_u8_tensor_names: Vec<String>,
  deferred_or_late_gpu_allocations: Vec<LateAllocationKind>,
}

LateAllocationKind {
  RopeTables,
  MetadataBuffers,
  F16FusedLayerWeights { layer_idx: usize },
  F16ConvertedEmbedding,
  F16ConvertedFinalNorm,
  GptOssMoeGpuUpload { layer_idx: usize },
  KvCache { absolute_layers: Vec<usize> },
  TiedLmHeadFallback,
}
```

The manifest is not an upload API. It is a preflight product that says what a
future shard-local loader should upload once a shard's stream exists.

### Tensor ownership rules

Whole-tensor ownership should follow the existing `ShardedModelPlan` placement
rules:

- `model.embed_tokens.weight`: first shard, using `DeviceMap.embedding_device`.
- `model.layers.<N>.*`: shard owning absolute layer `N`.
- `model.norm.weight`: final shard, using `DeviceMap.final_device`.
- `lm_head.weight`: final shard, using `DeviceMap.final_device`.
- tied LM-head fallback: if `lm_head.weight` is absent and the model uses tied
  embeddings, the final shard needs an LM-head-compatible path or copy even
  when embeddings live on GPU0.
- GPT-OSS MoE U8 expert block/scale tensors:
  `model.layers.<N>.mlp.experts.*` are layer-owned and should be retained or
  uploaded only for the shard that owns layer `N`.
- dense layer biases, router weights/biases, sink tensors, norm weights, and
  projection weights under `model.layers.<N>.*` remain whole-layer tensors and
  follow layer `N`.
- RoPE tables, metadata buffers, scratch buffers, and KV cache are not loaded
  from safetensors. They are shard-local late allocations.

Unknown tensor names should not be silently uploaded to every shard. The
manifest should classify them as unassigned unless a later model-specific rule
claims them.

### Late allocations and fused tensors

Late GPU allocations must be derived from the same ownership plan:

- Fused QKV and fused gate/up tensors are created only on the shard that owns
  the source layer.
- F16 layernorm and bias conversions are created only on the shard that owns
  the source layer.
- Final norm conversion is created only on the final shard.
- Embedding conversion is created only on the embedding shard, except for an
  explicit tied LM-head fallback plan on the final shard.
- GPT-OSS MoE GPU uploads are layer-owned. Host U8 data can remain global
  until upload, but the future upload should filter by shard-owned layer ids.
- RoPE tables should be allocated per shard because layer execution on that
  shard needs local device pointers.
- Metadata and scratch buffers should be allocated per shard because launch
  inputs and reusable workspaces are stream/context-local.
- KV cache should be allocated per shard in a later cache slice, keyed by
  absolute layer ids or protected by an absolute-to-local map.

The manifest should keep these late allocations separate from
`required_tensor_names` so the initial safetensor upload path stays
whole-tensor-only.

### Error handling / validation

Future manifest construction should validate before CUDA allocation:

- Missing required tensors should produce a manifest error that names the tensor
  and owning shard.
- Optional tensors, such as architecture-specific biases or dense MLP tensors
  absent in GPT-OSS MoE layers, should be recorded separately from required
  tensors.
- Unknown tensor names should be reported as unassigned and should not be
  uploaded by default.
- Out-of-range `model.layers.<N>.*` names should be invalid because they imply
  a model/config mismatch.
- Tensors that match no shard should be rejected for split allocation unless
  explicitly marked optional/ignored by a model-specific rule.
- Duplicate ownership should be a bug in manifest construction. A safetensor
  entry should have at most one owning shard.
- Shape metadata should remain global and complete across shards. Skipped
  uploads must still preserve shapes so later validation, tensor-parallel
  checks, and tied-head decisions can inspect the full model topology.
- Tied LM-head fallback should be explicit: if `lm_head.weight` is absent and
  `tie_word_embeddings` is true, the manifest should record
  `LateAllocationKind::TiedLmHeadFallback` for the final shard rather than
  pretending `model.embed_tokens.weight` is automatically local there.

### Future Stage 4 split allocation smoke

The split allocation smoke should become manifest-driven:

1. Read model config and safetensor headers.
2. Parse `split:0-11@0,12-23@1`.
3. Build `ShardedModelPlan`.
4. Derive `ShardTensorManifest` for GPU0 and GPU1.
5. Create CUDA resources for each shard.
6. For each shard, call loader primitives with a manifest-owned tensor filter.
7. Upload only whole tensors listed in that shard's manifest.
8. Perform only that shard's late allocations.
9. Emit a small status JSON or structured log with device id, layer range,
   required/optional/U8 tensor counts, skipped/unassigned count, and byte totals.
10. Do not run layer forward, final norm, LM head, sampling, or graph capture.
11. Do not compare logits or claim parity.

The first implementation should prefer observability over cleverness: the
status should make it obvious which shard owns every uploaded tensor and which
late allocations are still deferred.

### Preserving the current single-device path

The existing `GpuWorker::load_weights` and `GpuModelRunner` construction path
should remain unchanged for `single`. Future split loading should be separate
or explicitly gated behind a non-default split-allocation smoke path.

The current single-device f32 loader can keep using
`load_weights_to_gpu_with_shapes_filtered` for its existing GPT-OSS filtered
startup behavior. Split-specific filtering should not broaden that hook into
runtime behavior until the smoke path is isolated and tested.

The f16 and U8 loader paths need split-aware filters eventually, but this design
does not add them and does not change upload behavior today.

### Risks / blockers

- The f16 loader lacks a filter hook, so a real split smoke needs either a
  filtered f16 loader variant or a separate manifest-driven f16 upload path.
- The U8 host loader lacks a filter hook and currently returns all GPT-OSS U8
  tensors. That is acceptable for planning, but split upload should filter
  layer-owned U8 tensors before device upload.
- Required-vs-optional tensor rules are architecture-specific. GPT-OSS MoE and
  dense MLP models should not share one naive required list.
- Tied LM-head fallback can require a final-shard copy/path even when the
  embedding shard is GPU0.
- `GpuModelWeights` does not encode device ownership, so the future split path
  needs shard-local containers or clear wrapper ownership.
- Late allocations currently assume `0..num_layers` local indexing. Split
  allocation must preserve absolute layer ids before running anything.

### Next bounded step

Add a pure upload-manifest helper that consumes safetensor names plus a
`ShardedModelPlan` and emits per-shard `ShardTensorManifest` metadata without
calling CUDA or loader upload functions.

### Primary classification

multi_gpu_layer_sharding_filtered_tensor_loading_design_complete

## Upload manifest helper status

Added a CUDA-free upload manifest helper in
`crates/gpt-oss-model-runner/src/shard_plan.rs`. It consumes a
`ShardedModelPlan`, discovered safetensor tensor names, and
`UploadManifestOptions`, then emits per-shard ownership metadata without
loading, converting, uploading, allocating, or executing anything.

New pure manifest types:

- `ShardedUploadManifest`
- `ShardTensorManifest`
- `LateAllocationKind`
- `UploadManifestOptions`

Classified tensor patterns:

- `model.embed_tokens.weight` is assigned to the embedding shard.
- `model.layers.<N>.*` is assigned to the shard owning absolute layer `N`.
- `model.norm.weight` is assigned to the final shard.
- `lm_head.weight` is assigned to the final shard.
- unknown names are recorded in `unassigned_tensor_names` and are not copied to
  every shard.
- malformed or out-of-range layer names are recorded in
  `invalid_tensor_names`.

GPT-OSS U8 expert tensors:

- `model.layers.<N>.mlp.experts.gate_up_proj_blocks`
- `model.layers.<N>.mlp.experts.gate_up_proj_scales`
- `model.layers.<N>.mlp.experts.down_proj_blocks`
- `model.layers.<N>.mlp.experts.down_proj_scales`

These are assigned to the owning layer shard's `host_u8_tensor_names` rather
than `required_tensor_names`. This represents future host-side U8 retention and
shard-local GPU upload, but does not change the current U8 loader.

Tied LM-head fallback:

- Manifest construction accepts `tie_word_embeddings`.
- If `tie_word_embeddings` is true and `lm_head.weight` is absent from the
  discovered names, the final shard receives
  `LateAllocationKind::TiedLmHeadFallback`.
- If `lm_head.weight` exists, no fallback marker is added.
- No fallback copy or upload is implemented.

Late allocation markers:

- Every shard records `RopeTables`, `MetadataBuffers`, and `KvCache` for its
  absolute layers.
- The embedding shard records `F16ConvertedEmbedding`.
- The final shard records `F16ConvertedFinalNorm`.
- Each owned layer records `F16FusedLayerWeights` and `GptOssMoeGpuUpload`
  markers.

Still deferred:

- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created.
- No existing model loader runtime behavior changed.
- No f16 or U8 loader filters were added.
- No tensors are uploaded or routed to multiple devices.
- KV cache allocation and indexing are unchanged.
- No activation transfer, peer copy, NCCL, collectives, CUDA kernel changes, or
  runtime math branching were added.
- Split maps remain non-executable and continue to be rejected before CUDA
  allocation in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_upload_manifest_helper_complete

## KV cache planning skeleton status

Added a CUDA-free KV cache planning skeleton in
`crates/gpt-oss-model-runner/src/shard_plan.rs`. It derives pure cache
ownership metadata from `ShardedModelPlan` and does not instantiate
`CudaCacheEngine`, allocate `CudaSlice`, or alter runtime cache access.

New pure KV planning types:

- `ShardedKvCachePlan`
- `ShardKvCachePlan`
- `LayerKvCachePlan`

Behavior:

- `ShardedModelPlan::kv_cache_plan()` creates one KV shard plan per model shard.
- Each `LayerKvCachePlan` stores an `absolute_layer_idx` and a shard-local
  `local_cache_idx`.
- Local indices start at 0 within each shard.
- Absolute layer ids are preserved for validation, logging, and future runtime
  access.
- `ShardKvCachePlan::entry_for_absolute_layer(layer_idx)` returns the local
  entry only when that shard owns the absolute layer.
- `ShardedKvCachePlan::entry_for_absolute_layer(layer_idx)` can find the owning
  shard and local entry across the full plan.

Single-device behavior:

- A single map produces one KV shard with entries for every absolute layer.
- In the all-layer single-device case, `local_cache_idx` matches
  `absolute_layer_idx`.

Split behavior:

- `split:0-11@0,12-23@1` produces two KV shard plans.
- GPU0 owns absolute layers 0..11 with local cache indices 0..11.
- GPU1 owns absolute layers 12..23 with local cache indices 0..11.
- Looking up layer 11 succeeds on GPU0 and returns `None` on GPU1.
- Looking up layer 12 succeeds on GPU1 and returns `None` on GPU0.
- The full plan covers every model layer exactly once and has no duplicate
  absolute layer ids.

Why access stays absolute:

The current CUDA runner indexes `gpu_cache[layer_idx]` because one
`CudaCacheEngine` owns all layers. A split runtime cannot safely reuse that
assumption on shard-local vectors. The planning API therefore exposes
absolute-layer lookup first and treats `local_cache_idx` as the resolved shard
slot, making the future runtime boundary explicit.

Still deferred:

- `CudaCacheEngine` runtime behavior did not change.
- Existing KV cache allocation and indexing are unchanged.
- No CUDA contexts, streams, cache engines, or GPU buffers were created.
- No activation transfer, peer copy, NCCL, collectives, CUDA kernel changes, or
  runtime math branching were added.
- Split maps remain non-executable and continue to be rejected before CUDA
  allocation in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_kv_cache_plan_skeleton_complete

## Split allocation report status

Added a CUDA-free split allocation status/report builder in
`crates/gpt-oss-model-runner/src/shard_plan.rs`. It combines:

- `ShardedModelPlan`
- `ShardedUploadManifest`
- `ShardedKvCachePlan`

The report is pure metadata. It does not discover safetensor headers, create
CUDA resources, call loader upload functions, allocate cache buffers, or execute
layers.

New report types:

- `SplitAllocationReport`
- `ShardAllocationReport`

Report contents:

- device ids, embedding device, and final device
- per-shard absolute layers
- per-shard embedding/final-head ownership
- per-shard required tensor count and names
- per-shard optional tensor count
- per-shard host U8 tensor count and names
- per-shard late allocation count and markers
- per-shard KV cache entry count and absolute KV cache layers
- global unassigned tensor count and names
- global invalid tensor count and names

Single-report behavior:

- A single map produces one shard report.
- The shard owns all absolute layers.
- The shard owns embeddings and final head.
- KV cache layers cover every model layer.
- Known tensors are assigned to that shard, while unknown/invalid names still
  surface globally.

Split-report behavior:

- `split:0-11@0,12-23@1` produces two shard reports.
- GPU0 owns embeddings and layers 0..11.
- GPU1 owns layers 12..23 and final head.
- GPU0 KV cache layers are 0..11.
- GPU1 KV cache layers are 12..23.
- Required tensor counts and names come from the upload manifest.
- GPT-OSS U8 expert tensor counts and names remain separate from required
  tensor names.
- Tied LM-head fallback appears in the final shard's late allocation markers
  only when `tie_word_embeddings=true` and `lm_head.weight` is absent.

Still deferred:

- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created.
- No safetensor header discovery helper was added.
- No model loader upload behavior changed.
- No f16 or U8 loader filters were added.
- `CudaCacheEngine` runtime behavior did not change.
- Existing KV cache allocation and indexing are unchanged.
- No activation transfer, peer copy, NCCL, collectives, CUDA kernel changes, or
  runtime math branching were added.
- Split maps remain non-executable and continue to be rejected before CUDA
  allocation in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_split_allocation_report_complete

## Safetensor header discovery status

Added a CUDA-free safetensors header discovery helper in
`crates/gpt-oss-model-runner/src/model_loader/safetensor_headers.rs`, exported
as `SafetensorHeaderManifest` and `SafetensorTensorInfo`.

Metadata discovered:

- tensor name
- safetensors dtype string
- tensor shape
- source `.safetensors` file path
- tensor byte size from `data_offsets`

Single-file behavior:

- Reads only the first 8-byte header length and the JSON header bytes.
- Parses tensor entries without reading tensor payloads into host vectors.
- Returns tensor metadata in deterministic name order.

Sharded-directory behavior:

- Discovers `*.safetensors` files in the directory.
- Ignores non-safetensors files.
- Sorts shard file paths deterministically.
- Parses only headers from each shard.
- Merges tensor metadata and returns deterministic name order.
- Rejects duplicate tensor names across shards with a clear model error.

Planning bridge:

- `SafetensorHeaderManifest::tensor_names()` returns names that can feed
  `ShardedModelPlan::upload_manifest_for_tensor_names`.
- `contains_tensor` and `has_lm_head_weight` expose cheap header-only checks.
- Tests cover building a `ShardedUploadManifest` and `SplitAllocationReport`
  from header-discovered names without CUDA.
- Tests also cover absent `lm_head.weight` with `tie_word_embeddings=true`,
  which still triggers the existing `TiedLmHeadFallback` marker through the
  upload manifest path.

Still deferred:

- Tensor payloads are not converted, copied, or uploaded.
- Existing model loader runtime behavior did not change.
- No f16 or U8 loader filters were added.
- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created.
- `CudaCacheEngine` runtime behavior did not change.
- Existing KV cache allocation and indexing are unchanged.
- No activation transfer, peer copy, NCCL, collectives, CUDA kernel changes, or
  runtime math branching were added.
- Split maps remain non-executable and continue to be rejected before CUDA
  allocation in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_header_discovery_helper_complete

## Dry-run split allocation report command status

Added a CUDA-free operator/status binary:

`crates/gpt-oss-bench/src/bin/multi_gpu_layer_sharding_dry_run.rs`

Accepted flags:

- `--model <PATH>`
- `--device-map <SPEC>`
- `--selected-device <ID>`
- `--tie-word-embeddings <true|false>` as an optional override
- `--output <PATH>` as an optional JSON output path; stdout is used when absent

What it reads:

- `config.json` beside the model path to get `num_hidden_layers`.
- `tie_word_embeddings` from `config.json` unless the CLI override is passed.
- Safetensors header metadata through `SafetensorHeaderManifest`.
- Tensor names, dtypes, shapes, source files, and byte sizes from headers only.

What it builds:

- `DeviceMap` from the supplied spec and config layer count.
- `ShardedModelPlan`.
- `ShardedUploadManifest` from header-discovered tensor names.
- `ShardedKvCachePlan`.
- `SplitAllocationReport`.

What it emits:

- success classification:
  `multi_gpu_layer_sharding_dry_run_report_complete`
- model path, device-map spec, selected device, layer count, tensor counts, and
  total header-derived tensor bytes
- `has_lm_head_weight` and effective `tie_word_embeddings`
- per-shard device id, absolute layers, embedding/final-head ownership, tensor
  counts, U8 host tensor counts, late allocation counts, KV cache layer counts,
  KV cache layers, and observable tensor-name lists
- global unassigned and invalid tensor-name lists

Split-map behavior:

- This dry-run tool may accept split maps because it is a non-executing
  planner/status command.
- It does not call the serve/runtime pre-CUDA executable-map validator, because
  that validator intentionally rejects split maps for execution.
- Serve/runtime split maps remain non-executable and continue to be rejected
  before CUDA allocation with `split device maps are parsed but not executable
  yet`.

Still deferred:

- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created.
- No tensor payloads are converted or read into host vectors.
- No GPU loader upload functions are called.
- Existing model loader runtime behavior did not change.
- No f16 or U8 loader filters were added.
- `CudaCacheEngine` allocation and indexing behavior did not change.
- No activation transfer, peer copy, NCCL, collectives, CUDA kernel changes,
  runtime math branching, final-token, logit, or parity claims were added.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_dry_run_report_complete

## f16 and U8 filtered loader design

This slice is docs-only. It designs the remaining loader filter boundaries that
a future manifest-driven split allocation smoke needs. It does not add filtered
loader implementations or change runtime loader behavior.

### Current f32 filtered loader

The f32 CUDA loader already has the right shape:

```text
load_weights_to_gpu_with_shapes_filtered(path, stream, should_load)
```

`load_weights_to_gpu_with_shapes` is a wrapper over the filtered function with
`|_| true`, preserving the existing single-device call surface. The filtered
path dispatches to `load_single_to_gpu` for one safetensors file or
`load_sharded_to_gpu` for a directory. Sharded directories use sorted
`.safetensors` file paths, so traversal is deterministic.

For every header tensor entry, the f32 path calls `parse_tensor_meta`, computes
shape metadata, and inserts shape metadata even when the tensor is not uploaded.
It also skips `U8` tensors before conversion/upload while still recording their
shape. For non-U8 tensors, `should_load(name)` decides whether the function
converts bytes through `convert_to_f32` and uploads with `stream.clone_htod`.

This is the model for future split f32 shard uploads: a shard-local caller can
pass a manifest-derived filter and receive only that shard's owned GPU tensors,
while retaining visibility into the full model's shape topology.

### Current f16 loader gap

The f16 CUDA loader currently exposes:

```text
load_weights_to_gpu_f16_with_shapes(path, stream)
```

It dispatches to `load_single_to_gpu_f16` or `load_sharded_to_gpu_f16`. The
single-file helper memory maps the safetensors file, parses the full header, and
iterates every tensor entry. It skips `U8` tensors and records their shapes.
Every other dtype is converted through `convert_to_f16`:

- `F16` is reinterpreted into `half::f16`.
- `BF16` is converted to `f16` on the host.
- `F32` is narrowed to `f16`.

The converted `Vec<half::f16>` is uploaded immediately with
`stream.clone_htod`. Shapes are returned in a side map, but there is no
`should_load` hook today, so a future split path would upload every non-U8
tensor to every shard if it reused this function directly.

### Proposed f16 filtered API

Add an additive API in a later implementation slice:

```text
load_weights_to_gpu_f16_with_shapes_filtered(path, stream, should_load)
```

Semantics:

- Preserve complete shape metadata for every tensor in the file or sharded
  directory, including skipped tensors.
- Skip `U8` tensors exactly as the current f16 path does.
- Convert and upload only non-U8 tensors where `should_load(name)` returns
  true.
- Keep deterministic traversal for sharded directories by reusing sorted shard
  collection.
- Preserve the existing unfiltered function as a wrapper:
  `load_weights_to_gpu_f16_with_shapes(path, stream)` calls the filtered helper
  with `|_| true`.
- Preserve `load_weights_to_gpu_f16(path, stream)` as a wrapper that discards
  the returned shape map.

The future implementation should mirror the f32 filtered structure rather than
inventing a separate policy. That keeps the only behavioral difference in the
conversion function and output GPU element type.

### Current U8 host loader gap

The current U8 path is:

```text
load_u8_weights_to_host(path)
```

It dispatches to `load_single_u8_to_host` or `load_sharded_u8_to_host`. The
single-file helper memory maps the safetensors file, parses the header, and
copies payload bytes into a `HashMap<String, Vec<u8>>` only when
`dtype == "U8"`. The sharded helper collects sorted safetensors shards and
extends one combined host map.

This host map is later consumed by GPT-OSS MoE setup: U8 MXFP4 expert
blocks/scales survive initial f32/f16 loading on the host, then
`GpuModelRunner`/GPT-OSS layer construction uploads the selected MoE GPU
representation later. Today the U8 host loader has no filter hook, so a future
split allocation path would copy all U8 expert payloads into every shard setup
unless filtering is added before real split allocation.

### Proposed U8 filtered API

Add an additive host-side API in a later implementation slice:

```text
load_u8_weights_to_host_filtered(path, should_load)
```

Semantics:

- Copy only `U8` tensor payloads where `should_load(name)` returns true.
- Retain only manifest-owned layer U8 tensors for the shard.
- Preserve deterministic directory traversal by reusing sorted shard
  collection.
- Preserve enough validation metadata through the existing shape map path or
  `SafetensorHeaderManifest`; this filtered U8 helper should not become the
  sole source of shape/source metadata.
- Preserve the current unfiltered API as a wrapper:
  `load_u8_weights_to_host(path)` calls the filtered helper with `|_| true`.

For split allocation, host filtering is preferable to upload-time filtering
because it avoids copying every shard's U8 expert payloads into every shard
builder before any GPU upload occurs. Upload-time filtering can still be used as
a defensive check, but it should not be the only memory boundary.

### Manifest-to-filter integration

Future split allocation should derive loader filters from the existing
`ShardTensorManifest` produced by `ShardedModelPlan`.

Recommended caller logic:

```text
let required = HashSet<&str> from shard.required_tensor_names
let host_u8 = HashSet<&str> from shard.host_u8_tensor_names

f32/f16 should_load(name) = required.contains(name)
U8 host should_load(name) = host_u8.contains(name)
```

Small helper methods such as `required_tensor_filter_set()` and
`host_u8_tensor_filter_set()` may be added later if they keep call sites clean,
but the loader APIs should stay generic over `FnMut(&str) -> bool`.

Rules:

- f32 and f16 GPU filters use `required_tensor_names`.
- U8 host filters use `host_u8_tensor_names`.
- `unassigned_tensor_names` and `invalid_tensor_names` are not loaded by
  shard-local split allocation.
- Shape visibility remains global through `SafetensorHeaderManifest` and the
  loader shape maps; filtering GPU uploads must not hide model topology from
  validation/status reporting.
- Duplicate ownership remains a manifest/planner error before any split loader
  call.

### Tied LM-head fallback implications

If `lm_head.weight` exists, the final shard owns and loads `lm_head.weight`
normally through `required_tensor_names`.

If `lm_head.weight` is absent and `tie_word_embeddings=true`, the final shard
still needs an LM-head-compatible path. It must not assume that
`model.embed_tokens.weight` is local to the final shard, because the planned
two-GPU layout places embeddings on GPU0 and final norm/LM head on GPU1. The
existing manifest represents this as `LateAllocationKind::TiedLmHeadFallback`.
The actual copy/upload policy remains deferred: a later design must decide
whether the final shard receives a copied embedding-derived head, a separately
loaded host source, or another explicit fallback. This slice makes no claim
that tied fallback is executable.

### Future implementation order

1. Add `load_weights_to_gpu_f16_with_shapes_filtered` and focused tests. Keep it
   unused by runtime and preserve the unfiltered wrapper.
2. Add `load_u8_weights_to_host_filtered` and focused tests. Keep it unused by
   runtime and preserve the unfiltered wrapper.
3. Add manifest-to-filter helper tests or simple `HashSet` caller tests that
   prove required tensors and host U8 tensors become the intended filters.
4. Add a non-default split allocation smoke that creates per-device resources
   and calls filtered loaders from per-shard manifests.
5. Keep serve/runtime split execution rejected until allocation-only smoke is
   proven and separately reviewed.

### Risks / blockers

- The f16 filtered path must not accidentally skip shape metadata for filtered
  tensors; downstream fused f16 setup still expects model topology visibility.
- The U8 filtered path currently returns only payload maps. Validation/status
  should rely on header manifests or shape maps, not on U8 payload loading.
- Tied LM-head fallback is still not an allocation or execution policy.
- The future split allocation smoke must avoid routing unknown or invalid
  tensors into shard loaders just because they are present in safetensors.
- Adding filtered APIs is easy to make technically additive, but accidentally
  wiring them into default serve/runtime would violate this lane's guardrails.

### Next bounded step

Implement the inert f16 filtered loader API and tests, unused by runtime. Keep
the current unfiltered f16 loader as a wrapper over the filtered helper with
`|_| true`.

### Primary classification

multi_gpu_layer_sharding_f16_u8_filtered_loader_design_complete

## f16 filtered loader API status

Added an additive f16 filtered loader API in:

`crates/gpt-oss-model-runner/src/model_loader/gpu_loader.rs`

Function added:

```text
load_weights_to_gpu_f16_with_shapes_filtered(path, stream, should_load)
```

Preserved public APIs:

- `load_weights_to_gpu_f16(path, stream)`
- `load_weights_to_gpu_f16_with_shapes(path, stream)`

`load_weights_to_gpu_f16_with_shapes` now delegates to
`load_weights_to_gpu_f16_with_shapes_filtered(path, stream, |_| true)`, so the
existing unfiltered f16 behavior remains the default public behavior.
`load_weights_to_gpu_f16` continues to call the shape-returning API and discard
the shape map as before.

Shape metadata behavior:

- The filtered f16 path records shape metadata for every tensor encountered.
- Skipped non-U8 tensors still appear in the returned shape map.
- U8 tensors still appear in the returned shape map and are skipped by the f16
  GPU loader exactly as before.
- Filtering controls conversion/upload only.

Implementation shape:

- `load_weights_to_gpu_f16_with_shapes_filtered` mirrors the existing f32
  filtered API.
- Single-file and sharded-directory f16 helpers now accept `should_load`.
- Sharded-directory traversal continues to use sorted `.safetensors` paths.
- A small CUDA-free filter-decision helper is covered by unit tests so the
  shape/filter invariant is testable without requiring GPU upload execution.

Testing note:

- The focused tests exercise pure f16 filter decision semantics, not GPU
  upload execution.
- The CUDA feature path is compile-checked with
  `cargo check -p gpt-oss-model-runner --features cuda`.

Still deferred:

- U8 host filtering remains deferred.
- No split allocation smoke was added.
- No serve/runtime behavior changed.
- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created outside existing loader API behavior.
- No f32 loader behavior changed.
- No `CudaCacheEngine` behavior changed.
- Split maps remain non-executable in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_f16_filtered_loader_api_complete

## U8 host filtered loader API status

Added an additive U8 host filtered loader API in:

`crates/gpt-oss-model-runner/src/model_loader/gpu_loader.rs`

Function added:

```text
load_u8_weights_to_host_filtered(path, should_load)
```

Preserved public API:

```text
load_u8_weights_to_host(path)
```

`load_u8_weights_to_host` now delegates to
`load_u8_weights_to_host_filtered(path, |_| true)`, so the existing unfiltered
U8 host behavior remains the default public behavior.

U8-only behavior:

- The filtered loader reads safetensors headers and copies payload bytes only
  for tensors with `dtype == "U8"`.
- Non-U8 tensors are ignored even when `should_load(name)` returns true.
- The filter is applied after dtype classification, so it controls which U8
  payloads are retained in the returned host map.

Sharded-directory behavior:

- Directory traversal continues to use sorted `.safetensors` file paths.
- Non-safetensors files in a directory are ignored.
- Merge behavior remains the existing map-extension behavior; duplicate names
  are not made stricter in this slice.

Metadata responsibility:

- The U8 host loader still returns payload maps only.
- Shape, source-file, dtype, and byte-size metadata remain the responsibility of
  `SafetensorHeaderManifest` and the existing loader shape maps.
- The U8 filter API is intentionally not a metadata discovery API.

Still deferred:

- No GPU upload behavior changed.
- No split allocation smoke was added.
- No serve/runtime behavior changed.
- No f32 or f16 loader behavior changed in this slice.
- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created.
- No `CudaCacheEngine` behavior changed.
- Split maps remain non-executable in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_u8_filtered_loader_api_complete

## Manifest-to-filter helper status

Added pure manifest-to-loader filter helpers in:

`crates/gpt-oss-model-runner/src/shard_plan.rs`

Helper methods added on `ShardTensorManifest`:

```text
required_tensor_filter_set()
host_u8_tensor_filter_set()
should_load_required_tensor(name)
should_load_host_u8_tensor(name)
```

Filter derivation:

- f32/f16 GPU loader filters are derived from
  `ShardTensorManifest.required_tensor_names`.
- U8 host loader filters are derived from
  `ShardTensorManifest.host_u8_tensor_names`.
- The set-returning helpers use deterministic `BTreeSet<String>` values for
  observable/testable ordering.
- The predicate helpers avoid closure lifetime complexity for future loader
  call sites.

Exclusion behavior:

- Required tensor filters reject host U8 tensor names.
- Host U8 filters reject required non-U8 tensor names.
- Unknown/unassigned names are rejected by both filters.
- Invalid or out-of-range layer tensor names are rejected by both filters unless
  they were already present in the manifest, which the current planner avoids.

Tied LM-head fallback:

- `LateAllocationKind::TiedLmHeadFallback` does not add
  `model.embed_tokens.weight` to the final shard required-tensor filter.
- If `lm_head.weight` is absent and tied embeddings are requested, the final
  shard still needs a later explicit fallback path; this helper only reflects
  concrete manifest-owned tensor names.

Still deferred:

- No loader behavior changed.
- No loader functions are called from a split path.
- No split allocation smoke was added.
- No serve/runtime behavior changed.
- No CUDA contexts, streams, cuBLAS handles, kernel loaders, cache engines, or
  GPU buffers are created.
- Split maps remain non-executable in serve/runtime.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_manifest_to_filter_helpers_complete

## Per-device CUDA resource skeleton status

Added a non-executing CUDA shard resource skeleton in:

`crates/gpt-oss-model-runner/src/sharded_resources.rs`

Pure resource-plan types:

```text
ShardedCudaResourcePlan
CudaShardResourcePlan
ShardedCudaResourceStatus
CudaShardResourceStatus
```

CUDA-gated resource-owning types:

```text
ShardedCudaResources
CudaShardResources
```

The pure plan is built from `ShardedModelPlan` and preserves:

- device id
- absolute layer ids
- embedding ownership
- final-head ownership

When compiled with `--features cuda`, the resource constructor can create one
ownership island per shard:

- `CudaContext`
- `CudaStream`
- `CublasHandle`
- `KernelLoader`

The explicit constructor is:

```text
ShardedCudaResources::create_for_plan(plan)
ShardedCudaResources::create_for_plan_with_kernel_dir(plan, kernel_dir)
```

The first form uses an empty/nonexistent kernel directory and still binds the
context, stream, cuBLAS handle, and loader to each shard device. The second form
allows a later smoke to pass a real PTX directory without changing the default
runtime path.

Intentionally not created:

- no model tensor uploads
- no f32/f16/U8 loader calls
- no `GpuModelWeights`
- no `GpuTransformerLayer` execution path
- no `CudaCacheEngine`
- no RoPE tables
- no metadata buffers
- no f16 scratch
- no fused weights
- no MoE GPU weights
- no activation-transfer buffers
- no `GpuModelRunner`

An ignored CUDA smoke test was added:

```text
ignored_two_gpu_sharded_cuda_resource_constructor_smoke
```

Manual operator command:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo test -p gpt-oss-model-runner ignored_two_gpu_sharded_cuda_resource_constructor_smoke --features cuda -- --ignored --nocapture
```

The ignored smoke checks for at least two visible CUDA devices, then constructs
context/stream/cuBLAS/kernel-loader islands for
`split:0-11@0,12-23@1`. It does not load a model or allocate KV cache.

This differs from split execution because it only proves resource ownership can
be represented per device. Serve/runtime still reject split maps before CUDA
allocation, and this resource path is not called from engine startup.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_cuda_resource_skeleton_complete

## Bench-only CUDA resource smoke command status

Added a bench-only CUDA resource smoke/status command:

```text
multi_gpu_layer_sharding_cuda_resource_smoke
```

Binary path:

`crates/gpt-oss-bench/src/bin/multi_gpu_layer_sharding_cuda_resource_smoke.rs`

Accepted flags:

```text
--model <PATH>
--device-map <SPEC>
--selected-device <ID>
--kernel-dir <PATH>
--output <PATH>
```

The command reads:

- `config.json` from `--model`, only to get `num_hidden_layers`
- the `--device-map` placement intent
- the optional `--kernel-dir` for `KernelLoader`

The command does not read safetensor headers or payloads and does not load model
weights.

When built and run with `--features cuda`, the command can create one resource
island per shard through `ShardedCudaResources`:

- `CudaContext`
- `CudaStream`
- `CublasHandle`
- `KernelLoader`

It intentionally does not create:

- model tensor uploads
- U8 payload uploads
- KV cache
- RoPE tables
- metadata buffers
- f16 scratch
- fused weights
- MoE GPU weights
- transformer layers
- `GpuModelRunner`
- activation-transfer buffers

Status JSON includes:

- `classification`
- `model_path`
- `device_map_spec`
- `selected_device`
- `num_layers`
- `cuda_feature_enabled`
- `resource_construction_attempted`
- `resource_construction_succeeded`
- `kernel_dir`
- `shard_count`
- per-shard device id, absolute layers, embedding/final ownership, and resource
  creation booleans
- `omitted_allocations`
- `error`

Success classification:

```text
multi_gpu_layer_sharding_cuda_resource_smoke_complete
```

Error classifications:

```text
multi_gpu_layer_sharding_cuda_resource_smoke_invalid_device_map
multi_gpu_layer_sharding_cuda_resource_smoke_config_error
multi_gpu_layer_sharding_cuda_resource_smoke_cuda_unavailable
multi_gpu_layer_sharding_cuda_resource_smoke_resource_error
```

Manual operator command for the 2x RTX 3090 host:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0
```

This command may accept split maps because it is a non-executing bench-only
resource smoke. Serve/runtime split maps remain rejected before CUDA allocation
with `split device maps are parsed but not executable yet`.

Still unchanged:

- no tensor upload behavior changed
- no KV cache allocation/indexing changed
- no model loader runtime behavior changed
- no serve/runtime behavior changed
- no split execution path was added

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda -- --help`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_cuda_resource_smoke_command_complete

## Manifest-driven split allocation smoke status

Added a bench-only manifest-driven allocation smoke command:

```text
multi_gpu_layer_sharding_split_allocation_smoke
```

Binary path:

`crates/gpt-oss-bench/src/bin/multi_gpu_layer_sharding_split_allocation_smoke.rs`

Accepted flags:

```text
--model <PATH>
--device-map <SPEC>
--selected-device <ID>
--kernel-dir <PATH>
--dtype <f32|f16|both>
--tie-word-embeddings <true|false>
--output <PATH>
```

The command reads:

- `config.json` from `--model` for `num_hidden_layers` and, unless overridden,
  `tie_word_embeddings`
- safetensors headers through `SafetensorHeaderManifest`
- tensor payloads only for manifest-owned tensors selected by the smoke

The command builds:

- `DeviceMap`
- `ShardedModelPlan`
- `ShardedUploadManifest`
- `ShardedKvCachePlan`
- `SplitAllocationReport`
- `ShardedCudaResources` when run with `--features cuda`

CUDA resources created:

- per-shard `CudaContext`
- per-shard `CudaStream`
- per-shard `CublasHandle`
- per-shard `KernelLoader`

Filtered loader calls:

- `load_weights_to_gpu_with_shapes_filtered` for `--dtype f32` or `both`
- `load_weights_to_gpu_f16_with_shapes_filtered` for `--dtype f16` or `both`
- `load_u8_weights_to_host_filtered` for shard-owned GPT-OSS U8 tensors

The f32/f16 filters use `ShardTensorManifest.required_tensor_names`. The U8
host filter uses `ShardTensorManifest.host_u8_tensor_names`. Unknown,
unassigned, malformed, and out-of-range tensor names are surfaced in status and
are not selected for upload.

The smoke records counts, tensor names, shape counts, element/byte totals, U8
host byte totals, late allocation markers, and per-shard resource status, then
drops the uploaded maps. It does not build runner state from them.

Intentionally not created:

- `GpuModelRunner`
- `GpuTransformerLayer`
- `CudaCacheEngine`
- KV cache buffers
- RoPE tables
- metadata buffers
- graph output buffers
- f16 scratch
- fused QKV or gate-up weights
- converted final norm/embed tensors beyond initial loader ownership
- MoE GPU expert uploads
- activation-transfer buffers
- final norm, LM head, sampling, logits, or graph capture

Status JSON includes:

- `classification`
- `model_path`
- `device_map_spec`
- `selected_device`
- `num_layers`
- `dtype_mode`
- `has_lm_head_weight`
- `tie_word_embeddings`
- `tensor_count_from_headers`
- `total_header_tensor_bytes`
- `resource_construction_succeeded`
- `allocation_smoke_succeeded`
- `omitted_allocations`
- per-shard layer ownership, required tensors, uploaded f32/f16 counts, U8 host
  counts, shape counts, element/byte totals, late allocations, and resource
  status
- `unassigned_tensor_names`
- `invalid_tensor_names`
- `error`

Success classification:

```text
multi_gpu_layer_sharding_split_allocation_smoke_complete
```

Error classifications:

```text
multi_gpu_layer_sharding_split_allocation_smoke_invalid_device_map
multi_gpu_layer_sharding_split_allocation_smoke_config_error
multi_gpu_layer_sharding_split_allocation_smoke_header_error
multi_gpu_layer_sharding_split_allocation_smoke_resource_error
multi_gpu_layer_sharding_split_allocation_smoke_loader_error
```

Tied LM-head fallback:

- If `lm_head.weight` exists, the final shard loads it normally through
  `required_tensor_names`.
- If `lm_head.weight` is absent and `tie_word_embeddings=true`, the final shard
  reports `tied_lm_head_fallback` as deferred.
- The smoke does not copy `model.embed_tokens.weight` to the final shard and
  does not claim tied fallback is executable.

Manual operator command for the 2x RTX 3090 host:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16
```

This command may accept split maps because it is a non-executing bench-only
allocation smoke. Serve/runtime split maps remain rejected before CUDA
allocation with `split device maps are parsed but not executable yet`.

Still unchanged:

- no serve/runtime behavior changed
- no split execution path was added
- no KV cache allocation/indexing changed
- no final-token, logit, or parity claims are made

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help`
- `git diff --check`

Primary classification:

multi_gpu_layer_sharding_split_allocation_smoke_complete

## Real-model split allocation smoke status

Operator validation run date: 2026-04-29.

Command run:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_status.json
```

The status JSON and logs were written under `/tmp/multi_gpu_layer_sharding` and
are not committed.

Preflight result:

- two visible CUDA devices: RTX 3090 device 0 and RTX 3090 device 1
- model path exists
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
  passed with existing warnings

Smoke result:

- primary result classification:
  `multi_gpu_layer_sharding_real_model_split_allocation_f16_blocked`
- command status classification:
  `multi_gpu_layer_sharding_split_allocation_smoke_header_error`
- `resource_construction_succeeded`: `false`
- `allocation_smoke_succeeded`: `false`
- shard summary: not produced because header discovery failed before building the
  split allocation report
- unassigned tensor count: not reached
- invalid tensor count: not reached
- stderr contained compile warnings and the command invocation; no CUDA resource,
  loader, OOM, or execution error was reached

Failure boundary:

Header discovery rejected duplicate safetensor tensor names before CUDA resource
construction. The first reported duplicate was:

```text
model.layers.0.self_attn.sinks
```

The restricted model view contains 24 duplicate `model.layers.*.self_attn.sinks`
headers across the base model shards and `zzzz-sinks-override.safetensors`.
This is a header-merge/override-policy blocker for real-model dry allocation,
not a split CUDA resource, f16 loader, U8 host loader, or memory-pressure result.

Interpretation:

- no CUDA resource islands were created by the smoke
- no f16 tensors or U8 host payloads were loaded by the smoke
- no KV cache, RoPE, metadata, scratch, fused weights, MoE GPU uploads,
  activation-transfer buffers, runner, layers, logits, or forward pass were
  created
- no execution or parity claim is made
- serve/runtime split maps remain non-executable and rejected before CUDA
  allocation

Next bounded step:

Design and implement a narrow safetensor header discovery policy for restricted
model override shards, preserving the current duplicate rejection by default.
The likely repair is an explicit, opt-in override or index-aware merge mode that
can account for `zzzz-sinks-override.safetensors` without silently accepting
arbitrary duplicate tensor names.

Primary classification:

multi_gpu_layer_sharding_real_model_split_allocation_f16_blocked

## Restricted sinks override header policy status

Added an explicit opt-in safetensors header merge policy for the restricted
integration model view.

Default behavior remains unchanged:

- `SafetensorHeaderManifest::discover`
- `SafetensorHeaderManifest::from_dir`
- `SafetensorHeaderMergePolicy::RejectDuplicates`

These paths still reject any duplicate tensor name across safetensors shards.
There is no broad last-writer-wins policy.

New policy/API:

```text
SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride
SafetensorHeaderManifest::discover_with_merge_policy(path, policy)
SafetensorHeaderManifest::from_dir_with_merge_policy(path, policy)
```

Bench/status flags:

```text
multi_gpu_layer_sharding_dry_run --allow-restricted-sinks-override
multi_gpu_layer_sharding_split_allocation_smoke --allow-restricted-sinks-override
```

The opt-in policy permits only this override file basename:

```text
zzzz-sinks-override.safetensors
```

and only replacement tensor names matching:

```text
model.layers.<N>.self_attn.sinks
```

where `<N>` is numeric. The override tensor must replace a previously discovered
base-shard tensor with the same name; unique tensors from
`zzzz-sinks-override.safetensors` are rejected.

The header helper remains independent of model config, so it does not validate
`<N>` against `num_hidden_layers`; downstream tensor manifest planning still
surfaces out-of-range layer tensor names as invalid.

Still rejected:

- arbitrary duplicate tensor names
- duplicate non-sinks tensor names, even from `zzzz-sinks-override.safetensors`
- duplicate sinks from any later file whose basename is not
  `zzzz-sinks-override.safetensors`
- duplicate sinks across ordinary shard files
- unique tensors inside `zzzz-sinks-override.safetensors` that do not replace an
  existing base-shard header

Reporting:

- `SafetensorHeaderManifest` now records `merge_policy` and
  `overridden_tensor_names`
- bench/status JSON includes `header_merge_policy`,
  `restricted_sinks_override_enabled`, `overridden_tensor_count`, and
  `overridden_tensor_names`
- overridden tensor names are reported deterministically

No loader/upload/runtime behavior changed. The policy is header-only and is
used by the bench/status tools only when the new restricted override flag is
passed. Serve/runtime split maps remain non-executable and rejected before CUDA
allocation.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `git diff --check`

Next bounded step:

Rerun the real-model f16 split allocation smoke with
`--allow-restricted-sinks-override`. If the run advances past header discovery,
preserve the next exact failure boundary or record the successful allocation-only
status without making execution or parity claims.

Primary classification:

multi_gpu_layer_sharding_restricted_sinks_header_override_complete

## Real-model split allocation smoke with sinks override status

Operator validation run date: 2026-04-30.

Command run:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_override_status.json
```

The status JSON and logs were written under `/tmp/multi_gpu_layer_sharding` and
are not committed.

Preflight result:

- two visible CUDA devices: RTX 3090 device 0 and RTX 3090 device 1
- model path exists
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
  passed with existing warnings

Smoke result:

- primary result classification:
  `multi_gpu_layer_sharding_real_model_split_allocation_f16_override_complete`
- command status classification:
  `multi_gpu_layer_sharding_split_allocation_smoke_complete`
- header merge policy: `allow_restricted_sinks_override`
- restricted sinks override enabled: `true`
- overridden tensor count: 24
- `resource_construction_succeeded`: `true`
- `allocation_smoke_succeeded`: `true`
- unassigned tensor count: 0
- invalid tensor count: 0
- stderr contained compile warnings and the command invocation; no CUDA
  resource, loader, OOM, or execution error was reported

Shard summary:

- GPU0 owns embeddings and absolute layers 0..11
  - uploaded f16 tensor count: 181
  - uploaded f16 shape count: 459
  - uploaded f16 bytes reported by status: 1,804,456,704
  - host U8 tensor count: 48
  - host U8 bytes: 5,076,172,800
  - resource status: context, stream, cuBLAS, and kernel loader created
- GPU1 owns absolute layers 12..23 plus final-head tensors
  - uploaded f16 tensor count: 182
  - uploaded f16 shape count: 459
  - uploaded f16 bytes reported by status: 1,804,462,464
  - host U8 tensor count: 48
  - host U8 bytes: 5,076,172,800
  - resource status: context, stream, cuBLAS, and kernel loader created

Interpretation:

- the bench-only smoke advanced past header discovery using the restricted sinks
  override policy
- per-shard CUDA resource islands were created
- manifest-owned f16 tensors were uploaded through the filtered f16 loader
- manifest-owned GPT-OSS U8 expert payloads were retained through the filtered
  U8 host loader
- no KV cache, RoPE, metadata, scratch, fused weights, MoE GPU uploads,
  activation-transfer buffers, runner, transformer layers, final norm, LM head,
  sampling, logits, graph capture, or forward pass were created
- no execution or parity claim is made
- serve/runtime split maps remain non-executable and rejected before CUDA
  allocation

Next bounded step:

The allocation-only smoke has now proven the current header/policy/resource and
filtered-loader boundary for the real restricted model. The next implementation
slice can add a shard-local RoPE/metadata allocation skeleton or a KV cache
allocation skeleton behind the bench-only path, keeping serve/runtime split maps
rejected.

Primary classification:

multi_gpu_layer_sharding_real_model_split_allocation_f16_override_complete

## Shard-local RoPE/metadata allocation skeleton status

Added a bench-only shard-local runtime buffer skeleton for the split allocation
smoke path.

Model-runner helper/status types:

```text
RopeRuntimeBufferConfig
ShardedRuntimeBufferPlan
CudaShardRuntimeBufferPlan
ShardedRuntimeBufferStatus
CudaShardRuntimeBufferStatus
ShardedRuntimeBuffers             # CUDA-gated
CudaShardRuntimeBuffers           # CUDA-gated
```

Bench flag:

```text
multi_gpu_layer_sharding_split_allocation_smoke --allocate-rope-metadata
```

Default behavior remains unchanged. Without the flag, the split allocation smoke
does not attempt RoPE/metadata allocation and still reports RoPE tables as an
omitted allocation.

With the flag, the bench command keeps the existing non-executing allocation
flow, then allocates shard-local RoPE cos/sin tables on each shard's existing
`CudaStream` using the same runtime table construction helper as
`GpuModelRunner::new`:

```text
build_runtime_rope_tables(head_dim, max_position.min(8192), rope_theta)
```

Status JSON now includes, per shard:

```text
rope_allocated
rope_cos_elements
rope_sin_elements
rope_total_bytes
metadata_allocated
metadata_status
metadata_deferred_reason
runtime_buffer_error
```

and top-level:

```text
rope_metadata_allocation_attempted
rope_metadata_allocation_succeeded
```

Metadata allocation is intentionally deferred. The current runner metadata path
uses request-shaped packed metadata buffers and graph/output-adjacent state, so
this skeleton reports:

```text
metadata_status: deferred
metadata_deferred_reason: request-shaped metadata packing buffers require batch/sequence inputs
```

The successful CUDA flag path classifies as:

```text
multi_gpu_layer_sharding_rope_allocated_metadata_deferred
```

Still not created:

- `CudaCacheEngine` or KV cache buffers
- fused QKV/gate-up weights
- f16 scratch
- MoE GPU expert uploads
- activation-transfer buffers
- transformer layers
- `GpuModelRunner`
- final norm / LM-head execution
- logits, sampling, graph capture, or any forward pass

Manual operator command, not run by default:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_rope_metadata_status.json
```

No serve/runtime behavior changed. Serve/runtime split maps remain
non-executable and rejected before CUDA allocation.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help`
- `git diff --check`

Next bounded step:

Run the real-model RoPE/metadata smoke operator command and review the JSON
status, or add a KV cache allocation skeleton behind the same bench-only
boundary.

Primary classification:

multi_gpu_layer_sharding_rope_allocated_metadata_deferred

## Real-model RoPE/metadata smoke status

Ran the bench-only real-model f16 split allocation smoke with the restricted
sinks override and shard-local RoPE allocation flag enabled.

Command run:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_rope_metadata_status.json
```

The output JSON and logs were left under `/tmp/multi_gpu_layer_sharding` and
are not committed.

Primary result classification:

```text
multi_gpu_layer_sharding_real_model_rope_allocated_metadata_deferred
```

Command status classification:

```text
multi_gpu_layer_sharding_rope_allocated_metadata_deferred
```

Run summary:

- model path: `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- dtype: `f16`
- device map: `split:0-11@0,12-23@1`
- header merge policy: `allow_restricted_sinks_override`
- overridden tensor count: 24
- resource construction: succeeded
- manifest-owned f16/U8 allocation smoke: succeeded
- RoPE/metadata allocation attempted: true
- RoPE/metadata allocation succeeded: true
- unassigned tensors: 0
- invalid tensors: 0

Shard summary:

- GPU0 owns embeddings and layers 0..11.
  - f16 tensors uploaded: 181
  - U8 host tensors retained: 48
  - f16 bytes: 1,804,456,704
  - U8 host bytes: 5,076,172,800
  - RoPE cos elements: 262,144
  - RoPE sin elements: 262,144
  - RoPE bytes: 2,097,152
  - metadata status: deferred
- GPU1 owns layers 12..23 and the final head.
  - f16 tensors uploaded: 182
  - U8 host tensors retained: 48
  - f16 bytes: 1,804,462,464
  - U8 host bytes: 5,076,172,800
  - RoPE cos elements: 262,144
  - RoPE sin elements: 262,144
  - RoPE bytes: 2,097,152
  - metadata status: deferred

Metadata remains deferred for the expected request-shaped boundary:

```text
request-shaped metadata packing buffers require batch/sequence inputs
```

Stderr contained only build/runtime warnings already present in the CUDA check
path and the final command invocation line; no smoke error was reported.

This result only proves that the bench-only smoke can create per-shard CUDA
resources, complete manifest-owned f16/U8 allocation/retention, and allocate
shard-local RoPE cos/sin tables for the real restricted model. It does not prove
KV cache allocation, metadata packing, layer execution, attention, activation
transfer, final-token parity, logit parity, or serve support.

No serve/runtime behavior changed. Serve/runtime split maps remain
non-executable and rejected before CUDA allocation.

Validation:

- `nvidia-smi`
- `test -d /data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- real-model f16 smoke command above
- JSON/log status review with `jq` and `tail`
- `git diff --check`

Next bounded step:

Add a KV cache allocation skeleton behind a bench-only path, or first document
the metadata-shape boundary more deeply before attempting request-shaped
metadata allocation.

Primary classification:

multi_gpu_layer_sharding_real_model_rope_allocated_metadata_deferred

## Shard-local KV cache allocation skeleton status

Added a bench-only, CUDA-gated shard-local KV cache allocation skeleton behind
the split allocation smoke command. The skeleton allocates only f16 key/value
cache buffers for shard-owned absolute layers and does not attach those buffers
to a runner, layer, attention kernel, or cache engine execution path.

Model-runner helper/status types:

```text
KvCacheAllocationConfig
ShardedKvCacheAllocationPlan
CudaShardKvCacheAllocationPlan
CudaLayerKvCacheAllocationPlan
ShardedKvCacheAllocationStatus
CudaShardKvCacheAllocationStatus
CudaLayerKvCacheAllocationStatus
ShardedKvCacheBuffers             # CUDA-gated
CudaShardKvCacheBuffers           # CUDA-gated
CudaLayerKvCacheBuffers           # CUDA-gated
```

Bench flags:

```text
--allocate-kv-cache
--kv-num-blocks <N>
--kv-block-size <N>
```

The allocation size is explicit and bench-only. `--allocate-kv-cache` requires
both `--kv-num-blocks` and `--kv-block-size`; the command reads
`num_key_value_heads` and `head_dim` from `config.json` and uses the same f16
cache element shape as the current CUDA cache path:

```text
[num_gpu_blocks, block_size, num_kv_heads, head_dim]
```

For `split:0-11@0,12-23@1`, the plan preserves absolute layer ids while using
shard-local cache indices:

- GPU0 allocates cache entries for absolute layers 0..11 with local indices
  0..11.
- GPU1 allocates cache entries for absolute layers 12..23 with local indices
  0..11.

Status JSON additions:

```text
kv_cache_allocation_attempted
kv_cache_allocation_succeeded
kv_num_blocks
kv_block_size
```

Per shard:

```text
kv_cache_allocated
kv_cache_entry_count
kv_cache_layers
kv_cache_local_indices
kv_key_total_bytes
kv_value_total_bytes
kv_total_bytes
kv_cache_entries
kv_cache_error
```

`kv_cache_entries` reports each absolute layer id, shard-local cache index, key
bytes, and value bytes. The public status remains absolute-layer-first so a
future execution path does not accidentally index a shard-local vector with an
absolute layer id.

Default behavior remains unchanged. Without `--allocate-kv-cache`, the split
allocation smoke does not attempt KV cache allocation and still reports
`kv_cache` as omitted. With `--allocate-kv-cache`, `kv_cache` is removed from
`omitted_allocations` and the status reports planned or allocated shard-local
KV entries.

The successful CUDA flag path classifies as:

```text
multi_gpu_layer_sharding_kv_cache_allocation_smoke_complete
```

Still not created:

- request-shaped metadata packing buffers
- fused QKV/gate-up weights
- f16 scratch
- MoE GPU expert uploads
- activation-transfer buffers
- transformer layers
- `GpuModelRunner`
- attention execution
- final norm / LM-head execution
- logits, sampling, graph capture, or any forward pass

Manual operator command, not run by default:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_rope_kv_status.json
```

No serve/runtime behavior changed. Serve/runtime split maps remain
non-executable and rejected before CUDA allocation.

Validation:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help`
- `git diff --check`

Next bounded step:

Run the real-model KV cache allocation smoke operator command, then use that
result to decide whether to refine KV memory accounting or move to the
metadata-shape boundary design.

Primary classification:

multi_gpu_layer_sharding_kv_cache_allocation_skeleton_complete

## Real-model KV cache allocation smoke status

Ran the bench-only real-model f16 split allocation smoke with restricted sinks
override, shard-local RoPE allocation, and small shard-local KV cache allocation
enabled.

Command run:

```text
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_rope_kv_status.json
```

The output JSON and logs were left under `/tmp/multi_gpu_layer_sharding` and
are not committed.

Primary result classification:

```text
multi_gpu_layer_sharding_real_model_kv_cache_allocation_smoke_complete
```

Command status classification:

```text
multi_gpu_layer_sharding_kv_cache_allocation_smoke_complete
```

Run summary:

- model path: `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- dtype: `f16`
- device map: `split:0-11@0,12-23@1`
- header merge policy: `allow_restricted_sinks_override`
- overridden tensor count: 24
- resource construction: succeeded
- manifest-owned f16/U8 allocation smoke: succeeded
- RoPE allocation: succeeded
- KV cache allocation: succeeded
- KV sizing: `kv_num_blocks=1`, `kv_block_size=16`
- unassigned tensors: 0
- invalid tensors: 0

Shard summary:

- GPU0 owns embeddings and layers 0..11.
  - f16 tensors uploaded: 181
  - U8 host tensors retained: 48
  - f16 bytes: 1,804,456,704
  - U8 host bytes: 5,076,172,800
  - RoPE bytes: 2,097,152
  - KV layers: 0..11
  - KV local indices: 0..11
  - KV key bytes: 196,608
  - KV value bytes: 196,608
  - KV total bytes: 393,216
  - per-layer key/value bytes: 16,384 each
  - metadata status: deferred
- GPU1 owns layers 12..23 and the final head.
  - f16 tensors uploaded: 182
  - U8 host tensors retained: 48
  - f16 bytes: 1,804,462,464
  - U8 host bytes: 5,076,172,800
  - RoPE bytes: 2,097,152
  - KV layers: 12..23
  - KV local indices: 0..11
  - KV key bytes: 196,608
  - KV value bytes: 196,608
  - KV total bytes: 393,216
  - per-layer key/value bytes: 16,384 each
  - metadata status: deferred

Metadata remains deferred for the expected request-shaped boundary:

```text
request-shaped metadata packing buffers require batch/sequence inputs
```

Stderr contained only build/runtime warnings already present in the CUDA check
path and the final command invocation line; no smoke error was reported.

This result only proves that the bench-only smoke can create per-shard CUDA
resources, complete manifest-owned f16/U8 allocation/retention, allocate
shard-local RoPE cos/sin tables, and allocate small shard-local KV cache buffers
for the requested split plan. It does not prove metadata packing, layer
construction, attention execution, cache indexing wired into attention,
activation transfer, final-token parity, logit parity, or serve support.

No serve/runtime behavior changed. Serve/runtime split maps remain
non-executable and rejected before CUDA allocation.

Validation:

- `nvidia-smi`
- `test -d /data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- real-model f16 RoPE+KV smoke command above
- JSON/log status review with `jq` and `tail`
- `git diff --check`

Next bounded step:

Design the metadata-shape boundary for request-shaped packing buffers, or move
to tied LM-head fallback design before attempting any execution-adjacent
activation handoff work.

Primary classification:

multi_gpu_layer_sharding_real_model_kv_cache_allocation_smoke_complete

## Primary classification

multi_gpu_layer_sharding_real_model_kv_cache_allocation_smoke_complete

## Metadata-shape boundary design

This section records the request-shaped metadata boundary that remains after the
bench-only allocation path proved shard-local CUDA resources, manifest-owned
f16/U8 payloads, RoPE tables, and small shard-local KV cache buffers.

This is a design-only slice. It does not allocate request-shaped metadata,
construct layers or a runner, run attention, add activation transfer, or change
serve/runtime behavior.

### Current metadata path

Request metadata is produced before the model runner sees a batch:

- `GpuLLMEngine::build_metadata` owns sequence block allocation for the serving
  path. It maintains `seq_block_tables`, allocates physical `BlockId`s from the
  configured `cache.block_size` and `num_gpu_blocks`, and emits
  `SequenceGroupMetadata`.
- `crates/gpt-oss-engine/src/worker/input.rs` converts
  `SequenceGroupMetadata` into `ModelInput`.
  - `prepare_prefill` emits all prompt tokens for each live sequence.
  - `prepare_decode` and `prepare_decode_reuse` emit one token per sequence.
  - `merge_inputs` concatenates prefill rows first and decode rows second for
    mixed batches.
- `ModelInput` carries `token_ids`, `position_ids`, `is_prefill`, and
  `bridge::AttentionMetadata`.
- `AttentionMetadata` carries `slot_mapping`, `context_lens`, `block_tables`,
  `max_context_len`, and `query_lens`.
- For CUDA graph decode, `GraphRunner::pad_input` may add dummy decode rows and
  zero-ish metadata up to the padded graph batch size before the runner upload.

The single-device CUDA runner then packs metadata:

- `GpuModelRunner::new` computes `graph_max_blocks` as
  `(config.max_position + cache.block_size() - 1) / cache.block_size()`.
  This fixed stride is used for `block_tables` to keep graph-captured pointers
  and layout stable.
- `GpuModelRunner` owns:
  - `cpu_scratch: RefCell<Vec<i32>>`
  - `meta_packed: RefCell<ReusableGpuBuf>`
  - `meta_packed_offsets: Cell<PackedMetaOffsets>`
  - `graph_output: RefCell<Option<CudaSlice<i32>>>`
- `GpuModelRunner::upload_metadata` packs all per-step fields into
  `cpu_scratch` as `i32`, then performs one `memcpy_htod` into `meta_packed`.
  Filtering is not involved here; the packed metadata corresponds to one
  concrete request batch.
- `GpuModelRunner::forward_ex`, `forward_last_token_logits`,
  `debug_prefill_trace`, `forward_graph_body`, and `forward_gpu_only` slice
  `meta_packed` through `PackedMetaOffsets` and pass those views into
  `GpuLayerInput`.
- `GpuLayerInput` passes:
  - `positions` to RoPE kernels.
  - `slot_mapping` to cache write kernels.
  - `block_tables`, `context_lens`, and `seq_start_pos` to prefill attention.
  - `block_tables` and `context_lens` to decode attention.
  - `key_cache` and `value_cache` as explicit per-layer cache slice pointers.

`graph_output` is graph/output-adjacent, not request metadata. It is lazily
allocated by GPU-only/graph paths as `[num_tokens] i32` token ids and should
remain out of scope for the first split metadata allocation smoke.

### Request-shaped inputs

All packed metadata fields are currently uploaded as `i32` in one contiguous
GPU buffer, even when the CPU source type is `u32`.

| Object | Owner today | CPU source | GPU view / buffer | Shape / length | Depends on |
| --- | --- | --- | --- | --- | --- |
| `token_ids` | `ModelInput`, packed by `GpuModelRunner::upload_metadata` | `Vec<u32>` | slice of `meta_packed` as `CudaView<i32>` | `num_tokens` | request tokens, prefill/decode mode, graph padding |
| `positions` | `ModelInput`, packed by `GpuModelRunner::upload_metadata` | `Vec<u32>` | slice of `meta_packed` as `CudaView<i32>` | `num_tokens` | per-token absolute position, prefill/decode mode, graph padding |
| `context_lens` | `AttentionMetadata`, packed by `GpuModelRunner::upload_metadata` | `Vec<u32>` | slice of `meta_packed` as `CudaView<i32>` | `num_seqs` | sequence lengths including past/cache length; graph padding may add dummy sequences |
| `block_tables` | `SequenceGroupMetadata` -> `AttentionMetadata`, padded by `upload_metadata` | `Vec<Vec<u32>>` of physical block ids | slice of `meta_packed` as `CudaView<i32>` | `num_seqs * graph_max_blocks` | block size, max model length, allocated physical blocks, sequence lengths, graph layout |
| `slot_mapping` | `AttentionMetadata`, packed by `GpuModelRunner::upload_metadata` | `Vec<u32>` | slice of `meta_packed` as `CudaView<i32>` | `num_tokens` | block size, physical block table, token position within each sequence |
| `query_lens` | `AttentionMetadata`; not uploaded directly | `Vec<u32>` | folded into `seq_start_pos` | `num_seqs` CPU-side input | prefill/decode mode; mixed batches have prompt lengths plus decode ones |
| `seq_start_pos` | derived inside `GpuModelRunner::upload_metadata` | computed from `query_lens` | slice of `meta_packed` as `CudaView<i32>` | `num_seqs + 1` with sentinel `num_tokens` | query-token counts, mixed batch order, prefill/decode mode |
| `max_context_len` | `AttentionMetadata` | scalar `u32` | scalar argument, not in `meta_packed` | one value | max of `context_lens`; past/cache length |
| `graph_output` | `GpuModelRunner` graph path | none before forward | `CudaSlice<i32>` | `num_tokens` | graph/GPU-only output policy, not metadata allocation |

The packed buffer size in elements is:

```text
num_tokens                  // token_ids
+ num_tokens                // positions
+ num_seqs                 // context_lens
+ num_seqs * graph_max_blocks
+ num_tokens                // slot_mapping
+ num_seqs + 1              // seq_start_pos
```

The byte count is that element count times `sizeof(i32)`.

Prefill and decode differ in source shapes:

- Prefill: each sequence contributes its full prompt. `query_lens ==
  context_lens`, `num_tokens` is the sum of prompt lengths, and `positions`
  are `0..prompt_len` per sequence in the current input builder.
- Decode: each sequence contributes one token. `query_lens` is all ones,
  `num_tokens == num_seqs`, and `positions` is `seq_len - 1` per sequence.
- Mixed batches: prefill rows are concatenated before decode rows and
  `seq_start_pos` must be derived from `query_lens`, not from `context_lens`.

### Global vs shard-local metadata ownership

The future split path should treat request metadata as layer-independent and
small compared with weights/KV cache. The first implementation should duplicate
the same packed request metadata to every shard that will execute layers for
that request.

Recommended ownership:

- `token_ids`: duplicate only to the embedding shard for the first split
  execution path. It is needed for embedding lookup, not for middle/final
  shards after activation handoff exists. A metadata allocation smoke may still
  report its shape globally.
- `positions`: duplicate to every layer-owning shard because every shard's
  layers need RoPE positions.
- `context_lens`: duplicate to every layer-owning shard because attention is
  layer-local but request layout is global.
- `block_tables`: duplicate to every layer-owning shard. The physical block ids
  name slots inside each shard's own KV cache allocation; using the same block
  ids is valid as long as every shard allocates the same block count and block
  size for its local layers.
- `slot_mapping`: duplicate to every layer-owning shard because every local
  layer writes its own K/V into that shard's cache at the same token slots.
- `seq_start_pos`: duplicate to every layer-owning shard for prefill attention.
- `max_context_len`: pass unchanged to every layer-owning shard.
- `graph_output`: final-head/output-shard only, and deferred until split
  execution without graphs is already working.

Do not split request metadata by layer ownership. Layer ownership already
decides which cache pointers and weights are local. Request metadata describes
the batch, not the layer set.

### Interaction with shard-local KV cache

The existing split KV allocation plan is absolute-layer-first at the public
boundary and shard-local inside each resource island:

- GPU0 for `split:0-11@0,12-23@1` owns absolute layers 0..11 with local cache
  indices 0..11.
- GPU1 owns absolute layers 12..23 with local cache indices 0..11.

Metadata should not contain layer indices and should not know about local cache
indices. The layer execution boundary should resolve:

```text
absolute_layer_idx -> ShardKvCachePlan::entry_for_absolute_layer(...)
                   -> local_cache_idx
                   -> key/value cache slice pointers
```

before constructing `GpuLayerInput`.

Avoid any design where layer execution indexes a shard-local `gpu_cache` vector
with `absolute_layer_idx`. The metadata packet can remain identical across
shards because the layer-to-cache resolution happens outside the metadata.

### Minimal bench-only metadata allocation smoke

A future smoke should be bench-only and synthetic-request-shaped. It should not
reuse a live server request and should not run attention.

Suggested flags on `multi_gpu_layer_sharding_split_allocation_smoke`:

```text
--allocate-metadata
--metadata-mode <decode|prefill>
--metadata-num-tokens <N>
--metadata-num-seqs <N>
--metadata-context-len <N>
--metadata-block-size <N>
```

For the first slice, prefer `decode` mode only:

- Require `metadata_num_tokens == metadata_num_seqs`.
- Set `query_lens = vec![1; num_seqs]`.
- Set `context_lens = vec![metadata_context_len; num_seqs]`.
- Build one block-table row per sequence with
  `ceil(metadata_context_len / metadata_block_size)` logical block ids, then
  pad to `graph_max_blocks` during packing.
- Set `slot_mapping` for one decode token per sequence using the final token's
  physical block and offset.
- Set `positions` to `metadata_context_len - 1` for every sequence.
- Derive `seq_start_pos = 0..num_seqs` plus sentinel `num_tokens`.

Status should report, per shard:

```text
metadata_allocated
metadata_mode
metadata_num_tokens
metadata_num_seqs
metadata_graph_max_blocks
metadata_packed_elements
metadata_packed_bytes
metadata_token_ids_len
metadata_positions_len
metadata_context_lens_len
metadata_block_tables_len
metadata_slot_mapping_len
metadata_seq_start_pos_len
metadata_error
```

This smoke should allocate/copy only the packed metadata buffer on each
layer-owning shard stream. It must not build `GpuModelRunner`, run embedding,
write KV cache, launch attention, allocate graph output, or read logits.

The existing real-model allocation smoke may accept these synthetic metadata
flags later, clearly labeled as metadata allocation smoke inputs. Without those
flags it should remain allocation-only and keep metadata deferred.

### Prefill/decode scope

The first metadata smoke should target decode-like metadata only. Decode has
the smallest useful shape contract: one token per sequence, one query per
sequence, and the graph-padding path already cares about stable decode batch
sizes.

Prefill should remain deferred until decode metadata allocation is proven.
Prefill adds variable query lengths per sequence, larger `num_tokens`, and a
stronger requirement that `seq_start_pos` be built from `query_lens` rather than
from `context_lens`. Mixed prefill+decode should remain further out of scope.

### CUDA graph scope

CUDA graph capture and replay should stay out of scope for split sharding until
eager split execution works without graphs.

Reason:

- Graph replay depends on stable metadata pointers, padded decode batch sizes,
  `graph_output`, and graph-captured kernel launch topology.
- Split execution will add activation handoff and per-shard sequencing that
  should be validated eagerly before capture is considered.
- The first split metadata allocation smoke can use the same `graph_max_blocks`
  padded layout for future compatibility, but it should not begin capture,
  replay graphs, allocate graph output, or make graph-safety claims.

### Risks / blockers

- The current `upload_metadata` helper is an instance method on
  `GpuModelRunner`, which is intentionally not constructed in the split bench
  path. A future metadata smoke should either extract a narrow packing helper or
  implement a bench-only packer that mirrors the formulas above.
- Decode metadata uses physical block ids from the serving allocator. A
  synthetic smoke must ensure its generated block ids fit within the requested
  `kv_num_blocks`.
- `block_tables` are padded to `graph_max_blocks`, which derives from model
  max position and block size, not only the request's visible context. This can
  dominate metadata byte counts for larger synthetic `num_seqs`.
- Future execution must keep the absolute-layer to local-cache mapping outside
  metadata to avoid shard-local indexing bugs.
- Tied LM-head fallback remains independent from metadata allocation and still
  needs a separate final-shard design.

### Next bounded step

Add a bench-only metadata allocation skeleton with synthetic decode request
shape flags. The skeleton should duplicate the packed metadata buffer to each
layer-owning shard, report shapes and byte counts, keep graph output deferred,
and continue to make no attention/execution/parity claim.

### Primary classification

multi_gpu_layer_sharding_metadata_shape_boundary_design_complete

## Shard-local metadata allocation skeleton status

This slice adds a bench-only synthetic decode metadata allocation skeleton. It
is opt-in from `multi_gpu_layer_sharding_split_allocation_smoke` and remains
separate from serve/runtime execution.

### Module and helper names

The CUDA-free plan/status types live in
`crates/gpt-oss-model-runner/src/sharded_resources.rs`:

- `MetadataMode`
- `MetadataAllocationConfig`
- `ShardedMetadataAllocationPlan`
- `CudaShardMetadataAllocationPlan`
- `ShardedMetadataAllocationStatus`
- `CudaShardMetadataAllocationStatus`

CUDA-gated allocation wrappers in the same module allocate/copy the packed
metadata buffer through the existing per-shard resource islands:

- `ShardedMetadataBuffers`
- `CudaShardMetadataBuffers`

The model-runner crate re-exports these types from `src/lib.rs`.

### Bench flags

`crates/gpt-oss-bench/src/bin/multi_gpu_layer_sharding_split_allocation_smoke.rs`
now accepts:

```text
--allocate-metadata
--metadata-mode decode
--metadata-num-tokens <N>
--metadata-num-seqs <N>
--metadata-context-len <N>
--metadata-block-size <N>
```

Metadata allocation is disabled by default. Without `--allocate-metadata`,
the existing split allocation smoke behavior is unchanged and metadata remains
deferred.

### Decode-only scope

The first skeleton supports only `metadata-mode=decode`.

Validation rules:

- `metadata-num-tokens` must equal `metadata-num-seqs`.
- `metadata-context-len` must be non-zero.
- `metadata-block-size` must be non-zero.
- `max_position_embeddings` from `config.json` must be non-zero.
- If KV allocation is also requested, `metadata-block-size` must match
  `kv-block-size`.
- If KV allocation is also requested, the generated decode block ids must fit
  within `kv-num-blocks`.

Unsupported modes such as `prefill` are rejected by the CLI. Prefill and mixed
prefill/decode metadata remain deferred.

### Packed metadata fields

The skeleton builds deterministic synthetic decode metadata and packs all
fields as `i32`, matching the current runner metadata packet type.

For decode mode:

- `token_ids_len = metadata_num_tokens`
- `positions_len = metadata_num_tokens`
- `context_lens_len = metadata_num_seqs`
- `graph_max_blocks = ceil(max_position_embeddings / metadata_block_size)`
- `block_tables_len = metadata_num_seqs * graph_max_blocks`
- `slot_mapping_len = metadata_num_tokens`
- `seq_start_pos_len = metadata_num_seqs + 1`

Packed element count:

```text
token_ids_len
+ positions_len
+ context_lens_len
+ block_tables_len
+ slot_mapping_len
+ seq_start_pos_len
```

Packed byte count is `packed_elements * sizeof(i32)`.

### Decode metadata generation

The synthetic decode packet uses deterministic values:

- `query_lens` is represented by `seq_start_pos = [0, 1, ..., num_seqs]`.
- `context_lens = [metadata_context_len; num_seqs]`.
- `positions = [metadata_context_len - 1; num_tokens]`.
- `token_ids = 0..num_tokens`.
- `block_tables` has one row per sequence, uses physical block ids starting at
  zero, and pads each row to `graph_max_blocks`.
- `slot_mapping` uses the final token position:
  `block_index = (metadata_context_len - 1) / metadata_block_size`,
  `block_offset = (metadata_context_len - 1) % metadata_block_size`, and
  `slot = physical_block * metadata_block_size + block_offset`.

### Per-shard ownership

The packed metadata buffer is duplicated to every shard that owns layers. It is
not split by layer ownership. Metadata remains layer-independent; shard-local
layer ownership continues to determine which weights and KV cache entries are
local.

The status path preserves the existing absolute-layer-first KV policy:

```text
absolute_layer_idx -> local_cache_idx -> key/value cache slices
```

The metadata packet does not carry layer ids and does not decide local cache
indices.

### Status JSON fields

Top-level status adds:

```text
metadata_allocation_attempted
metadata_allocation_succeeded
metadata_mode
metadata_num_tokens
metadata_num_seqs
metadata_context_len
metadata_block_size
metadata_graph_max_blocks
metadata_error
```

Each shard reports:

```text
metadata_allocated
metadata_status
metadata_mode
metadata_num_tokens
metadata_num_seqs
metadata_graph_max_blocks
metadata_packed_elements
metadata_packed_bytes
metadata_token_ids_len
metadata_positions_len
metadata_context_lens_len
metadata_block_tables_len
metadata_slot_mapping_len
metadata_seq_start_pos_len
metadata_max_context_len
metadata_error
```

When metadata allocation succeeds, the command classification becomes:

```text
multi_gpu_layer_sharding_metadata_allocation_smoke_complete
```

### What remains deferred

This slice does not allocate or create:

- graph output buffers
- request-shaped metadata from a live server request
- prefill or mixed prefill/decode metadata
- KV-cache execution accessors
- f16 scratch
- fused QKV/gate-up weights
- MoE GPU expert uploads
- transformer layers
- `GpuModelRunner`
- attention execution
- activation transfer
- logits, sampling, graph capture, serving, or parity paths

Serve/runtime split maps remain non-executable.

### Manual operator command

Do not run this as part of normal validation. A future operator run can use:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_rope_kv_metadata_status.json
```

### Validation

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help`
- `git diff --check`

Neighboring bench checks were also run because the shared split allocation
smoke command shape changed:

- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda`

### Primary classification

multi_gpu_layer_sharding_metadata_allocation_skeleton_complete

### Next bounded step

Run the real-model metadata allocation smoke as an operator validation slice,
then review the emitted status JSON. No real-model metadata command was run in
this implementation slice.

## Real-model metadata allocation smoke status

The real restricted integration model was run through the bench-only split
allocation smoke with restricted sinks override, RoPE allocation, small KV cache
allocation, and synthetic decode metadata allocation enabled.

### Command run

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_rope_kv_metadata_status.json
```

The JSON status and logs were written under `/tmp/multi_gpu_layer_sharding` and
are not committed.

### Result

Primary result classification:

```text
multi_gpu_layer_sharding_real_model_metadata_allocation_smoke_complete
```

Command status classification:

```text
multi_gpu_layer_sharding_metadata_allocation_smoke_complete
```

Summary:

- Model path:
  `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`
- Device map: `split:0-11@0,12-23@1`
- Dtype: `f16`
- Header merge policy: `allow_restricted_sinks_override`
- Restricted sinks override enabled: true
- Overridden tensor count: 24
- Header tensor count: 459
- Header tensor bytes: 13,761,264,768
- `resource_construction_succeeded`: true
- `allocation_smoke_succeeded`: true
- `rope_metadata_allocation_attempted`: true
- `rope_metadata_allocation_succeeded`: true
- `kv_cache_allocation_attempted`: true
- `kv_cache_allocation_succeeded`: true
- `metadata_allocation_attempted`: true
- `metadata_allocation_succeeded`: true
- Unassigned tensor count: 0
- Invalid tensor count: 0

Metadata shape:

- `metadata_mode`: `decode`
- `metadata_num_tokens`: 1
- `metadata_num_seqs`: 1
- `metadata_context_len`: 1
- `metadata_block_size`: 16
- `metadata_graph_max_blocks`: 8192

### Shard summary

GPU0:

- Owns embeddings and absolute layers 0..11.
- Resource island created context, stream, cuBLAS handle, and kernel loader.
- f16 tensors uploaded: 181
- f16 bytes uploaded: 1,804,456,704
- U8 host tensors retained: 48
- U8 host bytes retained: 5,076,172,800
- RoPE bytes: 2,097,152
- KV layers/local indices: 0..11 / 0..11
- KV bytes: 393,216
- Metadata status: `allocated`
- Metadata packed elements: 8198
- Metadata packed bytes: 32,792
- Metadata field lengths:
  `token_ids=1`, `positions=1`, `context_lens=1`, `block_tables=8192`,
  `slot_mapping=1`, `seq_start_pos=2`

GPU1:

- Owns absolute layers 12..23 and final head.
- Resource island created context, stream, cuBLAS handle, and kernel loader.
- f16 tensors uploaded: 182
- f16 bytes uploaded: 1,804,462,464
- U8 host tensors retained: 48
- U8 host bytes retained: 5,076,172,800
- RoPE bytes: 2,097,152
- KV layers/local indices: 12..23 / 0..11
- KV bytes: 393,216
- Metadata status: `allocated`
- Metadata packed elements: 8198
- Metadata packed bytes: 32,792
- Metadata field lengths:
  `token_ids=1`, `positions=1`, `context_lens=1`, `block_tables=8192`,
  `slot_mapping=1`, `seq_start_pos=2`

### Stderr summary

The command exited successfully. `stdout` was empty because `--output` wrote the
status JSON to disk. `stderr` contained Cargo/CUDA build warnings already seen
in previous validation, followed by the command invocation. No runtime smoke
error was reported.

### Non-execution boundary

This success means only that the bench-only smoke created per-shard CUDA
resources, completed manifest-owned f16/U8 allocation/retention, allocated
shard-local RoPE tables, allocated small shard-local KV cache buffers, and
allocated/copied synthetic decode packed metadata buffers to layer-owning
shards.

It does not mean:

- metadata came from a live server request
- prefill metadata works
- mixed prefill/decode metadata works
- graph output is allocated
- transformer layers are constructed
- `GpuModelRunner` is constructed
- attention works
- cache indexing is wired into attention
- activation transfer works
- final-token or logit parity exists
- serving works with split maps

Serve/runtime split maps remain non-executable.

### Validation

- `nvidia-smi`: two RTX 3090 devices visible.
- `test -d /data/models/openai/gpt-oss-20b-full-attn-restricted-integration`:
  passed.
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`:
  passed with existing warnings.
- Real-model metadata allocation smoke command: exited 0 and emitted
  `multi_gpu_layer_sharding_metadata_allocation_smoke_complete`.
- `git diff --check`: passed after this docs update.

### Next bounded step

Use the successful metadata allocation smoke as the allocation baseline for the
next design slice. Recommended options are tied LM-head fallback design,
activation handoff design refinement, shard-local fused/f16 scratch allocation
skeleton, or a layer-construction skeleton without execution.

## Shard-local fused/f16 scratch allocation design

This is a design/recon slice for the next runtime-adjacent allocation group. No
fused weights, f16 scratch, MoE GPU expert payloads, layers, runner, attention,
or execution path were added in this slice.

### Current fused/scratch path

The current single-device f16 setup is owned by `GpuModelRunner` in
`crates/gpt-oss-model-runner/src/gpu_runner.rs`.

Key structs and fields:

- `GpuModelRunner`
  - Owns one full `GpuModelWeights` container.
  - Owns one full `Vec<GpuTransformerLayer>`.
  - Owns full-runner fused allocation fields:
    - `fused_qkv_weights: Vec<CudaSlice<f16>>`
    - `fused_gate_up_weights: Vec<CudaSlice<f16>>`
    - `fused_layernorm_f16: Vec<CudaSlice<f16>>`
    - `fused_post_norm_f16: Vec<CudaSlice<f16>>`
    - `fused_qkv_bias_f16: Vec<Option<CudaSlice<f16>>>`
    - `fused_o_proj_bias_f16: Vec<Option<CudaSlice<f16>>>`
    - `final_norm_weight_f16: Option<CudaSlice<f16>>`
    - `embed_tokens_f16: Option<CudaSlice<f16>>`
    - `f16_scratch: Option<F16LayerScratch>`
- `F16LayerScratch`
  - Current reusable buffers:
    - `qkv: [max_tokens * qkv_dim]`
    - `attn_out: [max_tokens * q_dim]`
    - `o_proj: [max_tokens * hidden]`
    - `normed: [max_tokens * hidden]`
    - `residual: [max_tokens * hidden]`
    - `gate_up: [max_tokens * intermediate * 2]`
    - `silu_out: [max_tokens * intermediate]`
    - `down: [max_tokens * hidden]`
  - Allocated by `GpuModelRunner::alloc_scratch`.
  - Current `max_tokens` is hard-coded to 32 for the max graph batch path.

Current setup sequence:

1. `GpuWorker::load_weights` loads f32 weights, U8 host tensors, and f16
   weights when the worker dtype is half.
2. `GpuWorker::build_gpu_model_runner` moves those maps into
   `model_loader::gpu_weights::GpuModelWeights`.
3. `GpuModelRunner::new` validates that only single-device maps are executable,
   clones global tensors, constructs all `GpuTransformerLayer` shells, uploads
   RoPE tables, and builds GPT-OSS MoE layer metadata.
4. `GpuWorker::ensure_runner_initialized` calls:
   - `runner.enable_fp16()`
   - `runner.fuse_weights()`
   - `runner.prepare_gpt_oss_graph_decode()` for GPT-OSS f16 graph decode

`GpuModelRunner::fuse_weights` currently does several jobs at once:

- Fused QKV weights:
  - Iterates `0..self.layers.len()`.
  - Reads f16
    `model.layers.<i>.self_attn.{q_proj,k_proj,v_proj}.weight`.
  - Allocates `[q_dim + kv_dim + kv_dim, hidden]` f16.
  - Copies Q, K, and V into one contiguous device buffer.
- Fused dense gate/up weights:
  - Reads f16 `model.layers.<i>.mlp.{gate_proj,up_proj}.weight`.
  - Allocates `[intermediate * 2, hidden]` f16.
  - Skips dense gate/up only for GPT-OSS MoE layers where those dense tensors
    are absent.
- Layernorm and post-attention norm f16 conversions:
  - Reads f32
    `model.layers.<i>.input_layernorm.weight` and
    `model.layers.<i>.post_attention_layernorm.weight`.
  - Uses `cast_fp::cast_f32_to_f16_kernel`.
- QKV bias f16 conversion:
  - If f32 Q/K/V biases exist, allocates a temporary f32 fused bias buffer,
    device-to-device copies Q/K/V into it, then casts to f16.
- O projection bias f16 conversion:
  - If f32 `o_proj.bias` exists, casts it to f16.
- Final norm f16 conversion:
  - Casts global `model.norm.weight` to f16.
- Embedding f16 conversion:
  - Clones f16 `model.embed_tokens.weight` when available, otherwise casts the
    f32 embedding table to f16.
- f16 scratch:
  - Calls `alloc_scratch()`.

`GpuLayerWeightsF16` in `gpu_layer.rs` consumes these fused/preconverted
buffers during execution. It still carries the original per-layer projection
weights and biases as fallback inputs, plus optional fused QKV, optional fused
gate/up, optional f16 norm weights, and optional fused f16 biases.

GPT-OSS MoE upload is adjacent but separate:

- `GpuModelRunner::build_gpt_oss_moe_layers` builds one optional
  `GptOssMoeLayerWeights` per absolute layer.
- It reads U8 expert block/scale payloads from `GpuModelWeights::get_u8`.
- It already clones router f32 weights/biases to host and keeps GPU references
  to router tensors.
- It uploads per-expert gate/up and down bias chunks to GPU.
- It leaves U8 gate/up and down block/scale GPU fields as `None`.
- `GpuModelRunner::prepare_gpt_oss_graph_decode` later uploads those U8
  block/scale payloads to GPU so `supports_gpu_decode()` can become true.

The first fused/scratch smoke should keep GPT-OSS MoE GPU expert uploads
deferred unless the extraction boundary makes them trivial to count. They are
not required to prove dense fused f16 allocation and scratch sizing.

### Shard-local ownership rules

Future split allocation should allocate fused/preconverted f16 state only on
the shard that owns the relevant absolute layer or global tensor.

Per-layer ownership:

- Fused QKV weights: allocate only for absolute layers owned by the shard.
- Fused dense gate/up weights: allocate only for owned absolute layers whose
  dense gate/up tensors exist. GPT-OSS MoE layers should report dense gate/up
  as not applicable, not as an error.
- Input layernorm f16: allocate only for owned absolute layers.
- Post-attention layernorm f16: allocate only for owned absolute layers.
- Fused QKV bias f16: allocate only for owned absolute layers when Q/K/V biases
  exist.
- O projection bias f16: allocate only for owned absolute layers when the bias
  exists.
- GPT-OSS MoE GPU expert uploads: remain a separate deferred per-owned-layer
  late allocation until the MoE upload boundary is designed.

Global tensor ownership:

- Embedding f16 conversion belongs to the embedding shard only.
- Final norm f16 conversion belongs to the final shard only.
- `lm_head.weight` loads normally on the final shard when present.
- If `lm_head.weight` is absent and `tie_word_embeddings=true`, the tied LM-head
  fallback must remain explicit and deferred. The final shard must not assume
  `model.embed_tokens.weight` is local.

Scratch ownership:

- Allocate f16 scratch per layer-owning shard.
- Size scratch for the shard's intended execution shape, not for the full model.
- The first smoke should use an explicit bench-only sizing flag, for example
  `--f16-scratch-max-tokens <N>`, instead of silently reusing the current
  runner's graph-batch constant.
- Scratch can be one reusable set per shard because each shard will execute its
  owned layers sequentially on that shard's stream.

### Required input state for future smoke

A future bench-only fused/scratch smoke should build on the existing
`multi_gpu_layer_sharding_split_allocation_smoke` path and retain enough
allocation state instead of immediately dropping it after counting.

Needed inputs:

- `ShardedModelPlan`
  - device ids
  - absolute layer ids
  - embedding/final-head ownership
- `ShardTensorManifest`
  - `required_tensor_names` for f16 loader filters
  - `host_u8_tensor_names` for later MoE upload design
  - `deferred_or_late_gpu_allocations` for status comparison
- Per-shard CUDA resources
  - context
  - stream
  - cuBLAS handle
  - kernel loader
- Per-shard f16 uploaded tensor map from
  `load_weights_to_gpu_f16_with_shapes_filtered`.
- Per-shard shape metadata from the f16 loader or
  `SafetensorHeaderManifest`.
- Model config
  - `hidden_size`
  - `num_heads`
  - `num_kv_heads`
  - `head_dim`
  - `intermediate_size`
  - `num_layers`
  - `architecture`
  - layer types / sliding-window data, if the later layer-construction skeleton
    needs to validate config alignment.
- Optional per-shard U8 host map from `load_u8_weights_to_host_filtered`, but
  the first fused/scratch smoke should count/report MoE upload as deferred.

The current split allocation smoke only keeps counts after loader calls. A
fused/scratch smoke will need a CUDA-gated shard-local allocation state object
that retains uploaded maps long enough to build fused buffers and count bytes.
That object should still not construct `GpuTransformerLayer` or
`GpuModelRunner`.

### Avoiding GpuModelRunner construction

Do not construct `GpuModelRunner` just to call `fuse_weights()`.

Reasons:

- `GpuModelRunner::new` validates split maps as non-executable and is designed
  around a single CUDA context.
- It constructs all `GpuTransformerLayer` shells.
- It requires a `CudaCacheEngine`.
- It owns global tensors and full layer vectors.
- Its fused vectors are indexed as if one runner owns all layers.

Recommended future approach:

1. Add a narrow shard-local fused allocation helper, CUDA-gated and bench-only
   at first.
2. Feed it a shard-local f16 map, shape metadata, model config, shard absolute
   layers, and the shard's stream/kernel loader.
3. Store outputs in a small shard-local container such as:

   ```text
   ShardedF16AllocationBuffers
   CudaShardF16AllocationBuffers
   CudaLayerF16AllocationBuffers
   ```

4. Key per-layer outputs by absolute layer id in the public status. Internal
   storage may also carry shard-local indices, but status and validation should
   stay absolute-layer-first.
5. Extract helper functions from `GpuModelRunner::fuse_weights` only when the
   extraction is clearly mechanical and no-op for the single-device path. Good
   candidates are:
   - concat Q/K/V into a provided fused f16 buffer
   - concat gate/up into a provided fused f16 buffer
   - concat Q/K/V f32 biases then cast to f16
   - cast one f32 tensor to f16 using an existing `cast_fp` kernel
   - compute f16 scratch element counts from model config and max tokens

If extraction risks changing the single-device runtime, mirror the formulas in
the bench-only helper first and leave unification as a later cleanup.

### Future bench-only fused/scratch smoke

Extend `multi_gpu_layer_sharding_split_allocation_smoke` only behind explicit
flags, for example:

```text
--allocate-fused-f16
--allocate-f16-scratch
--f16-scratch-max-tokens <N>
```

Suggested first behavior:

1. Run the existing header discovery, split plan, upload manifest, resource
   construction, filtered f16 upload, filtered U8 host retention, RoPE, KV, and
   synthetic metadata steps as requested.
2. Retain per-shard f16 tensor maps for the duration of fused allocation.
3. For each shard:
   - allocate fused QKV for each owned absolute layer when the required Q/K/V
     f16 tensors are present and the architecture path supports dense fused QKV.
   - allocate fused dense gate/up for each owned absolute layer when dense
     gate/up tensors are present.
   - cast input and post-attention layernorm weights for each owned absolute
     layer.
   - cast/fuse QKV and O projection biases when present.
   - allocate embedding f16 conversion only if `owns_embeddings`.
   - allocate final norm f16 conversion only if `owns_final_head`.
   - allocate f16 scratch only if the shard owns at least one layer and
     `--allocate-f16-scratch` is present.
4. Drop fused/scratch buffers after status collection unless a later
   layer-construction skeleton needs to inspect them.
5. Do not construct layers, runner, graph output, attention inputs, activation
   transfer buffers, final norm execution, LM-head execution, logits, or graph
   capture.

The smoke should not make `--allocate-fused-f16` imply MoE GPU upload. GPT-OSS
MoE GPU expert upload needs its own explicit flag and design because it copies
U8 block/scale tensors and carries decode-specific `supports_gpu_decode`
semantics.

### Status JSON design

Top-level additions:

```text
fused_f16_allocation_attempted
fused_f16_allocation_succeeded
f16_scratch_allocation_attempted
f16_scratch_allocation_succeeded
f16_scratch_max_tokens
fused_f16_error
```

Per-shard additions:

```text
fused_f16_allocated
fused_qkv_weight_count
fused_gate_up_weight_count
fused_layernorm_count
fused_postnorm_count
fused_qkv_bias_count
fused_o_proj_bias_count
embedding_f16_allocated
final_norm_f16_allocated
f16_scratch_allocated
fused_total_bytes
f16_scratch_bytes
fused_layer_absolute_indices
fused_allocation_error
```

Optional per-layer detail:

```text
absolute_layer_idx
local_layer_idx
fused_qkv_bytes
fused_gate_up_bytes
layernorm_f16_bytes
postnorm_f16_bytes
fused_qkv_bias_bytes
o_proj_bias_f16_bytes
moe_gpu_upload_status
```

Status byte accounting should sum actual buffer lengths, not rely only on the
current `alloc_scratch()` log formula. The current scratch struct has eight
buffers, and future status should report the bytes for each buffer category or
the exact sum across those buffers.

Success classification for the future smoke:

```text
multi_gpu_layer_sharding_fused_scratch_allocation_smoke_complete
```

If fused f16 allocation succeeds but MoE GPU upload remains deferred, keep that
deferred state explicit in `late_allocations` / per-layer status.

### Validation strategy

Future validation should be allocation/status only.

Required no-GPU tests:

- Fused/scratch plan for `single` assigns all absolute layers to one shard.
- Fused/scratch plan for `split:0-11@0,12-23@1` assigns layers 0..11 to GPU0
  and 12..23 to GPU1.
- Embedding f16 conversion appears only on the embedding shard.
- Final norm f16 conversion appears only on the final shard.
- Tied LM-head fallback does not copy embedding to the final shard.
- Dense gate/up fused status is not required for GPT-OSS MoE layers when dense
  gate/up tensors are absent.
- Scratch byte formulas use the requested `f16_scratch_max_tokens`.

Required CUDA compile/status checks:

- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- Bench command unit tests for JSON/status shape.

Future real-model operator validation should use the restricted model and small
synthetic metadata/KV settings already proven in prior slices. It should report
counts and bytes only. It must not execute attention or produce parity claims.

### Risks / blockers

- There is no shard-local `GpuModelWeights` type today. The existing container
  is a name-keyed map and can hold partial maps, but `GpuModelRunner` assumes
  full-runner ownership.
- `GpuModelRunner::fuse_weights` loops over local indices
  `0..self.layers.len()` and formats tensor names as `model.layers.<i>...`.
  Split allocation needs absolute layer ids such as 12..23 on GPU1, not local
  loop indices.
- Existing fused vectors are indexed with `get(i)`. Sparse or split storage
  should avoid depending on full-model vector positions unless the vector has
  explicit holes or a map from absolute layer id.
- `GpuModelRunner::new` constructs layers and cache state, so it is the wrong
  boundary for a non-executing fused/scratch smoke.
- `F16LayerScratch` is private to `gpu_runner.rs`. A future reusable helper
  may need a public status-only plan plus a CUDA-gated buffer wrapper without
  exposing runner internals.
- Scratch sizing is currently tied to a hard-coded max graph batch of 32. A
  split smoke should require explicit synthetic sizing.
- GPT-OSS MoE GPU uploads are related but not identical. Router/bias state,
  host U8 payloads, and U8 block/scale GPU uploads need a separate design and
  explicit opt-in.
- Tied LM-head fallback remains unresolved and should not be silently folded
  into fused allocation.
- Activation handoff is still absent, so even successful fused/scratch
  allocation cannot imply split execution readiness.

### Next bounded step

Add a bench-only fused/f16 scratch allocation skeleton, still non-executing,
or design the tied LM-head fallback before allocating final-shard LM-head
fallback state. The allocation skeleton should retain per-shard f16 maps only
long enough to build fused/preconverted buffers and report counts/bytes.

### Primary classification

multi_gpu_layer_sharding_fused_scratch_design_complete

## Shard-local fused/f16 scratch allocation skeleton status

This slice added the bench-only fused/preconverted f16 and f16 scratch
status boundary. It did not allocate fused f16 buffers or f16 scratch because
the current allocation helpers remain coupled to `GpuModelRunner`,
full-runner-owned `GpuModelWeights`, and full-runner layer indexing.

### Module/helper names

Pure model-runner plan/status helpers live in:

```text
crates/gpt-oss-model-runner/src/sharded_resources.rs
```

New CUDA-free types:

```text
F16ScratchAllocationConfig
FusedF16AllocationStatus
ShardedFusedF16AllocationPlan
CudaShardFusedF16AllocationPlan
ShardedFusedF16AllocationStatus
CudaShardFusedF16AllocationStatus
```

The bench integration lives in:

```text
crates/gpt-oss-bench/src/bin/multi_gpu_layer_sharding_split_allocation_smoke.rs
```

### Bench flags

New explicit flags:

```text
--allocate-fused-f16
--allocate-f16-scratch
--f16-scratch-max-tokens <N>
```

`--allocate-f16-scratch` requires `--f16-scratch-max-tokens <N>`.
None of these flags imply MoE GPU upload, tied LM-head fallback, layer
construction, runner construction, graph output, attention, logits, or
execution.

Default behavior remains unchanged: without these flags the split allocation
smoke does not attempt fused f16 or scratch planning, and the corresponding
JSON fields report `not_applicable`.

### Actual allocation status

Actual fused/preconverted f16 allocation remains deferred.

Deferred reason:

```text
GpuModelRunner::fuse_weights assumes full-model runner-owned weight containers and full-runner layer indexing
```

Actual f16 scratch allocation remains deferred.

Deferred reason:

```text
F16LayerScratch is private runner state and tied to GpuModelRunner allocation
```

The status boundary still reports deterministic shard-local ownership,
planned layer coverage, and planned fused input counts so a later allocation
helper can be validated without changing the JSON contract again.

### Shard-local ownership policy

The fused f16 plan derives from each `ShardTensorManifest`:

- fused QKV status is counted only when a shard owns all Q/K/V f16 tensor names
  for an absolute layer.
- fused dense gate/up status is counted only when a shard owns both dense
  gate/up tensor names for an absolute layer.
- GPT-OSS MoE layers with U8 expert tensors and no dense gate/up tensors report
  dense gate/up as not applicable rather than as an error.
- f16 layernorm and post-attention norm counts are derived from owned absolute
  layer tensor names.
- f16 QKV and O-projection bias counts are derived from owned bias tensor names.
- embedding f16 conversion is planned only on the embedding shard.
- final norm f16 conversion is planned only on the final shard.
- tied LM-head fallback remains explicit/deferred and does not copy embeddings
  to the final shard.
- scratch status is per layer-owning shard and is gated by the explicit
  `--f16-scratch-max-tokens` value.

### Status JSON fields

Top-level additions:

```text
fused_f16_allocation_attempted
fused_f16_allocation_succeeded
f16_scratch_allocation_attempted
f16_scratch_allocation_succeeded
f16_scratch_max_tokens
fused_f16_error
f16_scratch_error
```

Per-shard additions:

```text
fused_f16_allocated
fused_f16_status
fused_qkv_weight_count
fused_gate_up_weight_count
f16_layernorm_count
f16_postnorm_count
f16_qkv_bias_count
f16_o_proj_bias_count
embedding_f16_allocated
final_norm_f16_allocated
fused_total_bytes
fused_layer_absolute_indices
fused_deferred_reason
fused_error
f16_scratch_allocated
f16_scratch_status
f16_scratch_bytes
f16_scratch_deferred_reason
f16_scratch_error
```

When either fused f16 or scratch planning is requested, the command reports:

```text
multi_gpu_layer_sharding_fused_f16_plan_status_complete
```

This classification means only that the bench-only status boundary was built.
It does not mean fused buffers or scratch buffers were allocated.

### Manual operator command

Do not run automatically. Operator-only command for a future real-model
status review:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --allocate-f16-scratch \
  --f16-scratch-max-tokens 1 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_full_skeleton_status.json
```

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke
cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda
git diff --check
```

### Non-execution boundary

No layers, `GpuModelRunner`, attention inputs, graph output, final norm
execution, LM-head execution, logits, graph capture, forward pass, or parity
path was added. No GPT-OSS MoE GPU expert upload or tied LM-head fallback was
implemented. Serve/runtime split maps remain non-executable and continue to be
rejected before CUDA allocation.

### Next bounded step

Recommended next slices:

- shard-local `GpuModelWeights` design.
- narrow helper extraction from `GpuModelRunner::fuse_weights`.
- `F16LayerScratch` visibility/sizing boundary design.
- tied LM-head fallback design.

### Primary classification

multi_gpu_layer_sharding_fused_f16_plan_status_complete

## Shard-local GpuModelWeights boundary design

This slice identifies the weight-container boundary needed before future
bench-only fused allocation, GPT-OSS MoE GPU upload, or non-executing layer
construction can operate on one shard at a time. It is a design-only slice:
no runtime container, fused allocation, scratch allocation, MoE GPU upload,
layer construction, runner construction, execution, or serve path was changed.

### Current GpuModelWeights contract

`crates/gpt-oss-model-runner/src/model_loader/gpu_weights.rs` defines
`GpuModelWeights` as a name-keyed backing store with four maps:

```text
weights: HashMap<String, CudaSlice<f32>>
weights_f16: HashMap<String, CudaSlice<f16>>
weights_u8: HashMap<String, Vec<u8>>
shapes: HashMap<String, Vec<usize>>
```

Its public accessors are also name-keyed:

```text
get(name)
get_f16(name)
get_u8(name)
shape(name)
require(name)
require_shape(name)
contains(name)
names()
```

The f32 loaders populate `weights` plus `shapes` through
`load_weights_to_gpu_with_shapes` and
`load_weights_to_gpu_with_shapes_filtered`. The f16 loaders populate
`weights_f16` plus `shapes` through
`load_weights_to_gpu_f16_with_shapes` and
`load_weights_to_gpu_f16_with_shapes_filtered`. The GPT-OSS U8 host loader
populates `weights_u8` through `load_u8_weights_to_host` or
`load_u8_weights_to_host_filtered`, then callers insert those payloads with
shape metadata. The filtered f32/f16 paths preserve shape visibility for
skipped tensors; the U8 host path intentionally owns payload filtering only.

Partial maps are technically possible today because `GpuModelWeights::new`,
`empty`, `insert`, `insert_f16`, and `insert_u8` do not require all model
tensors. A partial map is not enough for split allocation, though, because it
does not explain why a tensor is absent. Missing can mean unowned, optional,
not present in the architecture, invalid, unassigned, tied-LM-head fallback,
or a real loader bug.

Downstream code is where the full-model assumptions live:

- `GpuModelRunner::new` immediately requires
  `model.embed_tokens.weight`, `model.norm.weight`, and either
  `lm_head.weight` or `model.embed_tokens.weight`, then constructs
  `GpuTransformerLayer` shells for `0..config.num_layers`.
- `GpuModelRunner::layer_weights` formats
  `model.layers.{i}.*` names from the runner-local loop index and requires
  the full set of f32 per-layer tensors for that index.
- `GpuModelRunner::layer_weights_f16` does the same for f16 projection
  weights, f32 norms/biases, optional sinks, GPT-OSS MoE layer state, and
  fused vectors indexed by `get(i)`.
- `GpuModelRunner::fuse_weights` loops `0..self.layers.len()`, builds
  fused QKV and dense gate/up buffers from `model.layers.{i}.*`, converts
  norms and biases, converts final norm and embeddings, and finally calls
  `alloc_scratch`.
- `GpuModelRunner::build_gpt_oss_moe_layers` loops over all model layers,
  reads U8 expert blocks/scales, DtoH-copies router and bias tensors, and
  returns one optional MoE state slot per full-model layer.
- `prepare_gpt_oss_graph_decode` and
  `prune_fp32_projection_weights_for_fp16` also loop full model layer ids and
  access names directly through the runner-owned `GpuModelWeights`.

The existing container therefore can back a shard-local store, but it does not
itself carry placement, ownership, layer coverage, or absolute-layer access
rules.

### Proposed shard-local weight container

The recommended boundary is a lightweight wrapper around a partial
`GpuModelWeights` backing store:

```text
ShardGpuModelWeights {
  placement: GpuShardPlan,
  weights: GpuModelWeights,
  manifest: ShardTensorManifest,
  global_shapes: Arc<BTreeMap<String, Vec<usize>>>,
  tensor_source_summary: ShardTensorSourceSummary,
}
```

Equivalent names such as `ShardWeightStore` or `GpuShardWeights` are fine if
they fit the module style better. The important property is that the wrapper
owns explicit shard placement and manifest metadata in addition to the
name-keyed CUDA/host maps.

Suggested placement fields:

```text
device_id
absolute_layers
owns_embeddings
owns_final_head
absolute_to_local_layer, optional for status/internal use only
```

Suggested source/status fields:

```text
required_tensor_names_loaded
host_u8_tensor_names_loaded
shape_tensor_count
source_files_by_tensor, optional
missing_required_tensor_names
missing_optional_tensor_names
```

Embedding an existing `GpuModelWeights` keeps the loader APIs small: future
split allocation can keep using the existing filtered f32/f16/U8 loaders,
insert those partial maps into the backing store, and add ownership-aware
accessors at the wrapper boundary. No split path should construct
`GpuModelRunner` just to obtain that behavior.

### Access policy and absolute-layer lookup

Shard-local access should be absolute-layer-first. The wrapper should validate
ownership before constructing tensor names and before translating anything to a
local index.

Recommended accessor shape:

```text
owns_layer(absolute_layer_idx) -> bool
local_layer_idx(absolute_layer_idx) -> Option<usize>
get_layer_f32(absolute_layer_idx, suffix)
get_layer_f16(absolute_layer_idx, suffix)
get_optional_layer_f32(absolute_layer_idx, suffix)
get_optional_layer_f16(absolute_layer_idx, suffix)
get_required_f16(name)
get_required_f32(name)
get_embedding_f16_if_owned()
get_embedding_f32_if_owned()
get_final_norm_f16_if_owned()
get_final_norm_f32_if_owned()
get_lm_head_if_owned()
get_u8_for_owned_layer(absolute_layer_idx, suffix)
```

Required behavior:

- reject non-owned absolute layers before lookup.
- format names with the absolute layer id, for example
  `model.layers.12.self_attn.q_proj.weight` on GPU1.
- never silently translate GPU1 layer 12 to
  `model.layers.0.self_attn.q_proj.weight`.
- distinguish missing optional tensors from missing required tensors.
- distinguish unowned tensors from owned-but-missing tensors in status/errors.
- reject unknown, unassigned, or invalid tensor names for loadable access even
  if their shapes are visible.
- keep `absolute_layer_idx -> local_layer_idx` as an internal convenience for
  shard-local vectors and KV cache entries, not as the public lookup key.

For GPT-OSS MoE tensors, accessors should also be absolute-layer-first:

```text
get_u8_for_owned_layer(layer, "mlp.experts.gate_up_proj_blocks")
get_u8_for_owned_layer(layer, "mlp.experts.gate_up_proj_scales")
get_u8_for_owned_layer(layer, "mlp.experts.down_proj_blocks")
get_u8_for_owned_layer(layer, "mlp.experts.down_proj_scales")
```

Router weights and biases should use the same ownership check before accessing
`model.layers.<absolute>.mlp.router.*`.

### Shape metadata policy

Prefer complete global shape visibility at the shard-local wrapper boundary.
The simplest implementation is either:

- store complete `shapes` in the partial `GpuModelWeights`, preserving the
  current filtered-loader shape behavior, or
- store manifest-owned payloads in `GpuModelWeights` and keep an
  `Arc<BTreeMap<String, Vec<usize>>>` or header manifest for global shape
  lookup.

The first option is convenient if the current loader return maps already
contain all shapes. The second option is cleaner if the backing store should
only describe payloads physically resident on the shard. Either way, status and
validation should be able to inspect shapes for skipped/unowned tensors without
making those tensors loadable through shard accessors.

Complete shape visibility helps:

- validate fused QKV/gate-up dimensions before allocating.
- report unowned tensor shapes in dry-run/status output.
- keep tied LM-head fallback diagnostics precise.
- distinguish absent `lm_head.weight` from a loader filter mistake.
- validate GPT-OSS U8 block/scale shape expectations before GPU upload.

Shape visibility must not override ownership. If GPU1 can see the shape of
`model.embed_tokens.weight`, that does not mean GPU1 may use the embedding as a
silent tied-LM-head fallback.

### Fused allocation integration

Future fused allocation should consume `ShardGpuModelWeights` rather than
`GpuModelRunner`:

```text
for absolute_layer_idx in shard.absolute_layers {
  q = weights.get_layer_f16(absolute_layer_idx, "self_attn.q_proj.weight")
  k = weights.get_layer_f16(absolute_layer_idx, "self_attn.k_proj.weight")
  v = weights.get_layer_f16(absolute_layer_idx, "self_attn.v_proj.weight")
  input_norm = weights.get_layer_f32(absolute_layer_idx, "input_layernorm.weight")
  post_norm = weights.get_layer_f32(absolute_layer_idx, "post_attention_layernorm.weight")
  optional biases/sinks through optional accessors
}
```

Dense gate/up fusion should be attempted only when both dense
`mlp.gate_proj.weight` and `mlp.up_proj.weight` are owned and present. For
GPT-OSS MoE layers where dense gate/up tensors are absent and U8 expert tensors
are present, dense gate/up fusion should report not applicable rather than an
error.

Embedding f16 conversion belongs only to the embedding shard. Final norm f16
conversion belongs only to the final shard. If `lm_head.weight` exists, LM-head
f16 conversion belongs only to the final shard. If `lm_head.weight` is absent
and `tie_word_embeddings=true`, the final shard needs an explicit late fallback
path; fused allocation must not silently borrow GPU0's embedding table.

This replaces the current full-runner source of truth:

```text
for i in 0..self.layers.len()
```

with:

```text
for absolute_layer_idx in shard.absolute_layers
```

Fused output storage should also avoid full-runner vector indexing. Use either
absolute-layer keyed maps or dense shard-local vectors plus an explicit
absolute-to-local map. Public status should always report absolute layer ids.

### GPT-OSS MoE upload integration

MoE GPU upload remains a separate explicit boundary. The shard-local container
should prepare for it by exposing owned GPT-OSS state without performing the
upload by default:

```text
owned_moe_u8_expert_tensors(absolute_layer_idx)
owned_router_weight(absolute_layer_idx)
owned_router_bias(absolute_layer_idx)
owned_gate_up_bias(absolute_layer_idx)
owned_down_bias(absolute_layer_idx)
```

Future MoE upload should iterate only the shard's absolute layers, create
per-layer status for router/bias/U8 expert blocks and scales, and upload U8
expert blocks/scales only when a dedicated bench flag or later execution path
requests it. It should not reuse `build_gpt_oss_moe_layers` unchanged because
that helper currently returns one slot per full-model layer and loops
`0..config.num_layers`.

### Layer-construction integration

A later non-executing layer-construction skeleton can use the same wrapper to
create layer shells for owned absolute layers only:

```text
for absolute_layer_idx in shard.absolute_layers {
  validate owned weights for this absolute layer
  build GpuLayerConfig { layer_idx: absolute_layer_idx, ... }
  create a shard-local shell/status entry
}
```

That skeleton should not attach layers to `GpuModelRunner`, run attention,
allocate graph output, or perform activation transfer. Its first job is to
prove that per-layer configs, owned weight views, RoPE/KV/metadata status, and
future fused/MoE status line up for absolute layer ids.

### Tied LM-head fallback boundary

The shard-local container should make tied fallback state explicit:

- If `lm_head.weight` exists, the final shard's manifest includes it and
  `get_lm_head_if_owned()` can return it normally.
- If `lm_head.weight` is absent and `tie_word_embeddings=true`, the final
  shard should report `TiedLmHeadFallback` as deferred/required.
- If embeddings live only on GPU0, GPU1 must not report an executable LM head
  merely because global shapes include `model.embed_tokens.weight`.

The future fallback may be a copy, a second filtered upload of the embedding
table to the final shard, or a dedicated final-head weight container entry, but
it must be an explicit operation with status and memory accounting.

### Implementation order

Recommended staged path:

A. Add a pure `ShardWeightStorePlan` / status helper if the existing fused
   status needs a clearer source for owned tensors and missing required names.
B. Add a lightweight `ShardGpuModelWeights` wrapper around partial
   `GpuModelWeights`, `GpuShardPlan`, and `ShardTensorManifest`.
C. Refactor fused allocation helpers so they take absolute layer ids, shard
   weight accessors, model config, stream, and loader/cast kernel rather than
   `&mut GpuModelRunner`.
D. Add actual bench-only fused/preconverted f16 allocation using the wrapper.
E. Design and add explicit GPT-OSS MoE GPU upload status/allocation.
F. Add a non-executing layer-construction skeleton for owned absolute layers.

### Risks / blockers

- `GpuModelRunner::new` is intentionally still a single-device executable
  boundary and rejects split maps before CUDA allocation.
- `GpuModelRunner::fuse_weights` mutates runner-owned vectors and assumes one
  full-model `GpuModelWeights`.
- `layer_weights_f16` indexes fused vectors, MoE state, and bias vectors with
  the loop index `i`.
- `build_gpt_oss_moe_layers` and `prepare_gpt_oss_graph_decode` loop all
  model layers and should not be reused as-is for a shard.
- Missing tensor errors are currently plain name lookup failures; split needs
  missing-owned versus unowned versus optional semantics.
- The tied LM-head fallback remains deferred and can require large memory on
  the final shard.
- GPT-OSS MoE GPU uploads are adjacent but still deferred behind their own
  explicit status/allocation path.
- Activation handoff and split execution are still absent.

### Next bounded step

Add the pure shard-local weight container/wrapper skeleton, or first extract a
small fused-allocation helper boundary if it can be done without constructing
`GpuModelRunner`. The wrapper should stay non-executing and should prove
absolute-layer ownership, missing-tensor classification, and shape visibility
before any fused allocation is attempted.

### Primary classification

multi_gpu_layer_sharding_shard_local_weight_container_design_complete

## Shard-local weight wrapper skeleton status

This slice added a pure, CUDA-free shard-local weight access skeleton. It
proves ownership-gated tensor naming and loadability decisions without
wrapping live CUDA slices, allocating fused buffers, allocating scratch,
uploading MoE GPU expert weights, constructing layers, or constructing
`GpuModelRunner`.

### Module/type names

The skeleton lives in:

```text
crates/gpt-oss-model-runner/src/model_loader/shard_weights.rs
```

It is exported through:

```text
crates/gpt-oss-model-runner/src/model_loader/mod.rs
crates/gpt-oss-model-runner/src/lib.rs
```

New public types:

```text
ShardWeightStorePlan
ShardWeightStoreStatus
ShardTensorAvailability
ShardWeightLookupError
ShardWeightStore
```

`ShardWeightStorePlan` records:

```text
device_id
absolute_layers
owns_embeddings
owns_final_head
required_tensor_names
host_u8_tensor_names
global_shape_names
tied_lm_head_fallback_required
```

`ShardWeightStoreStatus` reports deterministic counts plus
`missing_required_tensor_names`.

### Constructor/source plans

The constructor is:

```text
ShardWeightStorePlan::from_shard_manifest(
  shard_plan,
  shard_manifest,
  global_shapes,
  has_lm_head_weight,
  tie_word_embeddings,
)
```

It consumes existing pure planning state:

- `GpuShardPlan` for device id, absolute layers, embedding ownership, and
  final-head ownership.
- `ShardTensorManifest.required_tensor_names` for f32/f16 loadability.
- `ShardTensorManifest.host_u8_tensor_names` for U8 host loadability.
- global shape names from header/loader shape metadata for shape visibility.
- `LateAllocationKind::TiedLmHeadFallback` plus config/header inputs to keep
  tied fallback explicit.

No safetensor payloads are read and no loader functions are called by this
wrapper.

### Absolute-layer-first access policy

Public accessors validate absolute layer ownership before constructing tensor
names:

```text
owns_layer(absolute_layer_idx)
local_layer_idx(absolute_layer_idx)
layer_tensor_name(absolute_layer_idx, suffix)
require_owned_layer_tensor_name(absolute_layer_idx, suffix)
optional_owned_layer_tensor_name(absolute_layer_idx, suffix)
u8_owned_layer_tensor_name(absolute_layer_idx, suffix)
require_required_tensor_name(name)
```

For the target split `split:0-11@0,12-23@1`, GPU1 owns absolute layer 12 and
`layer_tensor_name(12, "self_attn.q_proj.weight")` returns:

```text
model.layers.12.self_attn.q_proj.weight
```

It never maps GPU1 layer 12 to `model.layers.0.*`. Non-owned layers return:

```text
ShardWeightLookupError::LayerNotOwned
```

Owned-but-missing required layer tensors return:

```text
ShardWeightLookupError::RequiredTensorMissing
```

This keeps unowned and missing-owned failures distinct for future fused
allocation.

### Shape visibility vs loadability

The wrapper records `global_shape_names` separately from loadable tensor sets.
This allows a shard to report shape visibility for unowned tensors while still
rejecting those tensors for local loading or fused allocation.

Availability classification:

```text
ShardTensorAvailability::Required
ShardTensorAvailability::HostU8
ShardTensorAvailability::ShapeOnly
ShardTensorAvailability::NotVisible
```

Shape-only visibility does not imply loadability. For example, GPU1 can see
the shape name for `model.embed_tokens.weight` when global shapes are provided,
but `should_load_required_tensor("model.embed_tokens.weight")` remains false
for the target split.

### Embedding/final-head ownership behavior

Ownership-specific accessors:

```text
embedding_tensor_name_if_owned()
final_norm_tensor_name_if_owned()
lm_head_tensor_name_if_owned_or_deferred()
```

For `split:0-11@0,12-23@1`:

- GPU0 owns `model.embed_tokens.weight`.
- GPU1 does not own embeddings.
- GPU1 owns `model.norm.weight`.
- GPU1 owns `lm_head.weight` when it exists.
- GPU0 does not own final norm or LM head.

### Tied LM-head fallback behavior

When `lm_head.weight` is absent and `tie_word_embeddings=true`, the final
shard reports:

```text
ShardWeightLookupError::TiedLmHeadFallbackDeferred
```

The wrapper does not copy `model.embed_tokens.weight` to the final shard and
does not make the embedding tensor loadable on GPU1. Tied fallback remains a
separate late allocation design.

### Fused allocation unblock

Future fused allocation can use this boundary as the source of truth:

```text
for absolute_layer_idx in shard_store.plan().absolute_layers {
  shard_store.require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.q_proj.weight")
  shard_store.require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.k_proj.weight")
  shard_store.require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.v_proj.weight")
  shard_store.optional_owned_layer_tensor_name(absolute_layer_idx, "self_attn.q_proj.bias")
  shard_store.u8_owned_layer_tensor_name(absolute_layer_idx, "mlp.experts.gate_up_proj_blocks")
}
```

This replaces full-runner local loops like `0..self.layers.len()` with
explicit absolute layer ids while preserving shard-local local indices only as
an internal/status convenience.

Wrapping actual `GpuModelWeights` / `CudaSlice` maps is still deferred. The
next implementation can layer CUDA-backed accessors under the same ownership
checks without changing serve/runtime behavior.

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
git diff --check
```

### Non-execution boundary

No loader payload behavior, tensor upload behavior, CUDA resource construction,
fused allocation, f16 scratch allocation, GPT-OSS MoE GPU upload, tied LM-head
fallback implementation, layer construction, runner construction, attention,
graph output, logits, execution, serving, or parity path was added. Serve and
runtime split maps remain non-executable and continue to reject before CUDA
allocation.

### Next bounded step

Recommended next slices:

- narrow helper extraction from `GpuModelRunner::fuse_weights`.
- `F16LayerScratch` visibility/sizing boundary design.
- actual bench-only fused f16 allocation using `ShardWeightStore`.
- tied LM-head fallback design.

### Primary classification

multi_gpu_layer_sharding_shard_weight_wrapper_skeleton_complete

## Fused f16 helper extraction status

This slice extracted narrow CUDA-free sizing helpers for fused f16 allocation
planning. It did not add split fused allocation, f16 scratch allocation from a
split path, layer construction, runner construction, execution, or serve
behavior.

### Helper names

New helper module:

```text
crates/gpt-oss-model-runner/src/fused_f16.rs
```

New public helpers:

```text
F16ScratchElementCounts
fused_qkv_dim(q_dim, kv_dim)
fused_qkv_shape(q_dim, kv_dim, hidden)
fused_qkv_num_elements(q_dim, kv_dim, hidden)
fused_gate_up_dim(intermediate)
fused_gate_up_shape(intermediate, hidden)
fused_gate_up_num_elements(intermediate, hidden)
f16_scratch_element_counts(hidden, q_dim, kv_dim, intermediate, max_tokens)
```

`F16ScratchElementCounts` exposes the eight current scratch buffers:

```text
qkv
attn_out
o_proj
normed
residual
gate_up
silu_out
down
```

It also reports `total_elements()` and `total_bytes()` using
`sizeof(f16)`.

### Extraction scope

The extraction is pure shape/count only. CUDA copy helper extraction remains
deferred.

`GpuModelRunner::fuse_weights` now uses the pure helpers for:

- fused QKV dimension and allocation element count.
- fused dense gate/up dimension and allocation element count.
- f16 scratch per-buffer element counts.

The runner still owns the existing CUDA allocations and device-to-device
copies. The existing layer loop, tensor naming, fused vector mutation, error
strings, and scratch ownership remain in `GpuModelRunner`.

### What remains runner-coupled

The following are intentionally still coupled to the single-device runner:

- `0..self.layers.len()` layer iteration.
- mutation of `self.fused_qkv_weights`.
- mutation of `self.fused_gate_up_weights`.
- mutation of `self.fused_layernorm_f16`.
- mutation of `self.fused_post_norm_f16`.
- mutation of `self.fused_qkv_bias_f16`.
- mutation of `self.fused_o_proj_bias_f16`.
- final norm f16 assignment.
- embedding f16 assignment.
- private `F16LayerScratch` construction and assignment.
- Q/K/V bias CUDA concatenation and cast.
- GPT-OSS MoE GPU upload / graph-decode preparation.

### Future ShardWeightStore integration

The next fused allocation smoke can combine this slice with
`ShardWeightStore`:

```text
for absolute_layer_idx in shard_store.plan().absolute_layers {
  validate Q/K/V ownership with ShardWeightStore
  size fused QKV with fused_qkv_num_elements
  validate dense gate/up ownership if present
  size fused gate/up with fused_gate_up_num_elements
}
```

The future helper extraction still needs CUDA copy helpers that take explicit
shard-local tensor references and absolute layer ids instead of mutating
runner-owned full-model vectors.

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
git diff --check
```

### Non-execution boundary

No split fused allocation, f16 scratch allocation from a split path, layer
construction, runner construction, attention, final norm execution, LM-head
execution, logits, graph capture, forward pass, serving, or parity path was
added. No GPT-OSS MoE GPU expert upload or tied LM-head fallback was
implemented. Serve/runtime split maps remain non-executable and continue to be
rejected before CUDA allocation.

### Next bounded step

Recommended next slices:

- actual bench-only fused f16 allocation using `ShardWeightStore` and the
  extracted sizing helpers.
- `F16LayerScratch` visibility/sizing boundary implementation.
- tied LM-head fallback design.
- GPT-OSS MoE GPU upload design.

### Primary classification

multi_gpu_layer_sharding_fused_f16_shape_helpers_complete

## Bench-only fused QKV allocation smoke status

This slice changed the existing `--allocate-fused-f16` bench flag from
plan/status-only to actual fused QKV allocation in the CUDA bench smoke when
f16 Q/K/V tensors are available. The path is still bench-only and
non-executing.

### Helper and module names

Updated model-runner status/allocation boundary:

```text
crates/gpt-oss-model-runner/src/sharded_resources.rs
```

New/extended status and CUDA-gated buffer types:

```text
CudaLayerFusedF16AllocationPlan
CudaLayerFusedF16AllocationStatus
CudaLayerFusedF16Buffers
CudaShardFusedF16Buffers
ShardedFusedF16Buffers
```

Updated bench command:

```text
crates/gpt-oss-bench/src/bin/multi_gpu_layer_sharding_split_allocation_smoke.rs
```

### ShardWeightStore usage

The CUDA bench path builds a shard-local `ShardWeightStore` from each
`ShardTensorManifest` before fused allocation. For each owned absolute layer,
it validates ownership and canonical tensor names before looking up uploaded
f16 slices.

This preserves the absolute-layer-first policy:

```text
GPU1 layer 12 -> model.layers.12.*
```

It never maps GPU1 layer 12 to `model.layers.0.*`.

### What is actually allocated

With `--allocate-fused-f16` and `--dtype f16` or `--dtype both`, the bench
smoke now retains each shard's uploaded f16 tensor map long enough to allocate:

- fused QKV f16 buffers for shard-owned layers with
  `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, and
  `self_attn.v_proj.weight`.
- optional dense fused gate/up f16 buffers when both
  `mlp.gate_proj.weight` and `mlp.up_proj.weight` are present.

The fused layout matches the current runner layout:

```text
QKV:     q_proj || k_proj || v_proj
gate/up: gate_proj || up_proj
```

Sizing uses the existing helpers:

```text
fused_qkv_num_elements
fused_gate_up_num_elements
```

The fused buffers are dropped after status collection. They are not attached to
layers, runner state, graph state, or any execution path.

### Dense gate/up policy

Dense gate/up fusion is opportunistic. If dense gate/up tensors are present,
the bench helper allocates the fused dense buffer. For GPT-OSS MoE layers where
dense gate/up tensors are absent and owned U8 expert payloads exist, dense
gate/up status is reported as `not_applicable`, not as an error.

GPT-OSS MoE GPU expert upload remains a separate deferred boundary.

### Still deferred

The following fused/preconverted f16 work remains deferred:

- f16 input layernorm conversion.
- f16 post-attention layernorm conversion.
- f16 QKV bias conversion.
- f16 O-projection bias conversion.
- embedding f16 conversion.
- final norm f16 conversion.
- tied LM-head fallback.
- f16 scratch allocation.

`--allocate-f16-scratch` remains status/deferred only and still reports the
private runner-state boundary around `F16LayerScratch`.

### Status JSON

Top-level status now uses this success classification when fused QKV allocation
succeeds:

```text
multi_gpu_layer_sharding_fused_qkv_allocation_smoke_complete
```

Useful blocked classification:

```text
multi_gpu_layer_sharding_fused_qkv_allocation_blocked
```

Per-shard status includes:

```text
fused_f16_allocated
fused_f16_status
fused_qkv_weight_count
fused_qkv_total_bytes
fused_gate_up_weight_count
fused_gate_up_total_bytes
fused_total_bytes
fused_layer_absolute_indices
fused_layer_statuses
fused_deferred_reason
f16_scratch_status
f16_scratch_deferred_reason
```

Per-layer status includes:

```text
absolute_layer_idx
local_layer_idx
fused_qkv_allocated
fused_qkv_status
fused_qkv_bytes
fused_gate_up_allocated
fused_gate_up_status
fused_gate_up_bytes
layernorm_f16_status
postnorm_f16_status
qkv_bias_f16_status
o_proj_bias_f16_status
layer_error
```

### Manual operator command

Do not run this automatically. The 2x RTX 3090 operator command is:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_qkv_status.json
```

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner fused_f16
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke
cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help
git diff --check
```

Neighboring bench checks were also run because this slice touched the shared
split allocation smoke.

### Non-execution boundary

No transformer layers, `GpuModelRunner`, attention, graph output, final norm
execution, LM-head execution, logits, graph capture, forward pass, serving, or
parity path was added. No f16 scratch allocation, GPT-OSS MoE GPU upload, tied
LM-head fallback, activation transfer, NCCL, peer copy, collectives, tensor
parallelism, or expert parallelism was added. Serve/runtime split maps remain
non-executable and continue to be rejected before CUDA allocation.

### Next bounded step

Recommended next slices:

- real-model fused QKV allocation smoke operator run.
- f16 layernorm/bias/final/embed conversion helper extraction.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.

### Primary classification

multi_gpu_layer_sharding_fused_qkv_allocation_smoke_complete

## Real-model fused QKV allocation smoke status

The real restricted integration model was run through the bench-only fused QKV
allocation smoke on the 2x RTX 3090 host. This was an operator validation run
only; the generated JSON and logs stayed under `/tmp/multi_gpu_layer_sharding`
and were not committed.

### Command

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_qkv_status.json
```

Output status path:

```text
/tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_qkv_status.json
```

### Result

Primary result classification:

```text
multi_gpu_layer_sharding_real_model_fused_qkv_allocation_smoke_complete
```

Command status classification:

```text
multi_gpu_layer_sharding_fused_qkv_allocation_smoke_complete
```

Resource/allocation results:

- resource construction: succeeded.
- manifest-owned f16/U8 allocation: succeeded.
- RoPE allocation: succeeded.
- KV allocation: succeeded.
- synthetic decode metadata allocation: succeeded.
- fused QKV allocation: succeeded.
- unassigned tensors: 0.
- invalid tensors: 0.
- restricted sinks overrides: 24.

### Shard summary

GPU0:

- owns embeddings and layers 0..11.
- f16 tensors: 181.
- U8 host tensors: 48.
- RoPE bytes: 2,097,152.
- KV bytes: 393,216.
- metadata bytes: 32,792.
- fused QKV buffers: 12.
- fused QKV bytes: 353,894,400.
- dense gate/up buffers: 0, reported as `not_applicable` for GPT-OSS MoE
  layers.
- f16 scratch: `not_applicable` because `--allocate-f16-scratch` was not
  passed.

GPU1:

- owns layers 12..23 and final head.
- f16 tensors: 182.
- U8 host tensors: 48.
- RoPE bytes: 2,097,152.
- KV bytes: 393,216.
- metadata bytes: 32,792.
- fused QKV buffers: 12.
- fused QKV bytes: 353,894,400.
- dense gate/up buffers: 0, reported as `not_applicable` for GPT-OSS MoE
  layers.
- f16 scratch: `not_applicable` because `--allocate-f16-scratch` was not
  passed.

Each fused layer status preserved absolute layer ids and shard-local indices.
GPU1 reported absolute layers 12..23 with local fused indices 0..11.

### Stderr summary

The command completed successfully. Stderr contained the expected cargo/CUDA
build warnings and the command invocation line; no smoke error, allocation
error, copy error, ownership error, OOM, or manifest error was reported.

### Non-execution boundary

This result means only that the bench-only path created per-shard CUDA
resources, completed manifest-owned f16/U8 allocation, allocated RoPE, allocated
the small synthetic KV cache, allocated synthetic decode metadata, and allocated
fused QKV f16 buffers for shard-owned absolute layers.

It does not claim f16 scratch allocation, layernorm/postnorm casts, bias casts,
embedding/final norm conversion, GPT-OSS MoE GPU upload, tied LM-head fallback,
layer construction, attention, activation transfer, final-token parity, logit
parity, graph capture, forward execution, or serving.

Serve/runtime split maps remain non-executable and are still rejected before
CUDA allocation.

### Next bounded step

Recommended next slices:

- f16 layernorm/postnorm conversion helper extraction.
- bias f16 conversion helper extraction.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.

## f16 norm conversion helper extraction status

This slice extracted the generic CUDA f32-to-f16 tensor cast used by
`GpuModelRunner::fuse_weights` into the fused f16 helper module. The extraction
is intentionally a helper boundary only; no split-path norm allocation was
added.

### Helper

New CUDA-gated helper:

```text
cast_f32_tensor_to_f16
```

Location:

```text
crates/gpt-oss-model-runner/src/fused_f16.rs
```

The helper takes the current stream, an input `CudaSlice<f32>`, an explicit
element count, and the already-loaded `cast_f32_to_f16_kernel`. It preserves the
current allocation and launch behavior, including the existing
`cast_f32_f16 alloc` and `cast_f32_f16 launch` error strings.

The helper is also re-exported from `lib.rs` behind `--features cuda` for future
shard-local use.

### Runner conversion sites

`GpuModelRunner::fuse_weights` now calls `cast_f32_tensor_to_f16` for:

- `model.layers.<N>.input_layernorm.weight`.
- `model.layers.<N>.post_attention_layernorm.weight`.
- fused QKV bias after concatenating q/k/v bias.
- O-projection bias.
- final norm weight.
- embedding f32 fallback when an uploaded f16 embedding table is absent.

The runner still loads `cast_f32_to_f16_kernel` once and passes the same kernel
handle to each cast call. Existing layer indexing, runner-owned vectors, vector
ordering, final norm/embed side effects, and scratch allocation behavior are
unchanged.

### Still runner-coupled

The following remain coupled to `GpuModelRunner::fuse_weights` or later runner
state:

- mutation of runner-owned `fused_layernorm_f16` and `fused_post_norm_f16`
  vectors.
- fused QKV/O bias vector ownership.
- final norm and embedding side-effect assignment.
- private `F16LayerScratch` allocation.
- GPT-OSS MoE GPU upload.
- layer construction and execution.

Shard-local bench norm allocation remains a later slice because it needs to
combine this helper with `ShardWeightStore` ownership validation, per-shard f32
maps, and explicit status reporting without mutating runner-owned vectors.

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner fused_f16
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
git diff --check
```

No real-model CUDA smoke was run for this helper-only slice.

### Non-execution boundary

No split norm conversion allocation, transformer layers, `GpuModelRunner`
construction from a split path, attention, graph output, final norm execution,
LM-head execution, logits, graph capture, forward pass, serving, or parity path
was added. Serve/runtime split maps remain non-executable and continue to be
rejected before CUDA allocation.

### Next bounded step

Recommended next slices:

- bench-only shard-local layernorm/postnorm f16 conversion allocation.
- bias f16 conversion helper extraction/status boundary.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.

### Primary classification

multi_gpu_layer_sharding_f16_norm_conversion_helper_complete

## Bench-only f16 norm conversion allocation status

This slice extends the existing bench-only fused f16 allocation path under
`--allocate-fused-f16` so it also allocates shard-local f16 copies of per-layer
norm weights:

- `model.layers.<N>.input_layernorm.weight`.
- `model.layers.<N>.post_attention_layernorm.weight`.

The path remains non-executing. The converted buffers are allocated only for
owned absolute layers, reported in JSON status, and dropped after status
collection.

### Helpers and modules

The allocation path uses:

- `ShardWeightStore` for absolute-layer ownership and canonical tensor-name
  validation.
- `cast_f32_tensor_to_f16` from
  `crates/gpt-oss-model-runner/src/fused_f16.rs`.
- `ShardedFusedF16Buffers`, `CudaShardFusedF16Buffers`, and
  `CudaLayerFusedF16Buffers` in
  `crates/gpt-oss-model-runner/src/sharded_resources.rs`.

### f32 norm loading policy

The real operator command can remain `--dtype f16`.

When `--allocate-fused-f16` is enabled in f16 mode, the bench smoke performs a
small additional filtered f32 upload for only the owned per-layer norm tensors
needed by the cast helper. It does not upload all f32 weights to every shard.

### Status JSON

Per layer:

- `layernorm_f16_status`.
- `layernorm_f16_bytes`.
- `postnorm_f16_status`.
- `postnorm_f16_bytes`.

Per shard:

- `f16_layernorm_count`.
- `f16_postnorm_count`.
- `f16_layernorm_total_bytes`.
- `f16_postnorm_total_bytes`.
- `fused_total_bytes`, now including fused QKV, optional dense gate/up, and
  allocated layernorm/postnorm f16 bytes.

Successful fused QKV plus norm conversion allocation reports:

```text
multi_gpu_layer_sharding_fused_qkv_norm_allocation_smoke_complete
```

### Still deferred

The split path still does not allocate:

- QKV bias f16 conversion.
- O-projection bias f16 conversion.
- final norm f16 conversion.
- embedding f16 conversion.
- tied LM-head fallback.
- `F16LayerScratch`.
- GPT-OSS MoE GPU expert weights.
- transformer layers or `GpuModelRunner`.

Dense gate/up remains allocated only when dense gate/up tensors exist; GPT-OSS
MoE U8-only layers continue to report dense gate/up as `not_applicable`.

### Manual operator command

Do not run automatically during this implementation slice:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_norm_status.json
```

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner fused_f16
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke
cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda
git diff --check
```

No real-model CUDA smoke was run for this implementation slice.

### Non-execution boundary

No split execution, layer construction, `GpuModelRunner` construction, attention,
graph output, final norm execution, LM-head execution, logits, graph capture,
forward pass, serving behavior, or parity path was added.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

### Next bounded step

Recommended next slices:

- real-model fused QKV + norm conversion allocation smoke operator run.
- bias f16 conversion allocation.
- final/embed f16 conversion allocation.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.

### Primary classification

multi_gpu_layer_sharding_fused_qkv_norm_allocation_smoke_complete

## Real-model fused QKV + norm conversion allocation smoke status

Operator run date: 2026-05-01.

Command run:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_norm_status.json
```

The generated JSON and logs are operator artifacts under
`/tmp/multi_gpu_layer_sharding/` and are not committed.

Preflight:

- `nvidia-smi` showed two visible NVIDIA GeForce RTX 3090 devices.
- `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration` existed.
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
  passed with existing warnings.

Result:

- Primary result classification:
  `multi_gpu_layer_sharding_real_model_fused_qkv_norm_allocation_cast_error`.
- Command status classification:
  `multi_gpu_layer_sharding_fused_qkv_allocation_blocked`.
- Header merge policy: `allow_restricted_sinks_override`.
- Restricted sinks overrides: 24.
- Resource construction: succeeded.
- Manifest validation: 0 unassigned tensors, 0 invalid tensors.
- f16/U8 allocation result: blocked status; U8 host payloads were retained, but
  `allocation_smoke_succeeded` was false.
- Selective f32 norm loading result: did not complete; status reports 0
  uploaded f32 tensors on each shard.
- RoPE allocation result: not reached in the success-status path.
- KV allocation result: not reached in the success-status path.
- Synthetic decode metadata allocation result: not reached in the
  success-status path.
- Fused QKV allocation result: not reached; per-layer fused QKV statuses remain
  `deferred`.
- Norm conversion allocation result: blocked on the first shard norm cast
  kernel lookup.

Exact failure boundary:

```text
gpu error: shard 0 f16 norm cast kernel lookup failed: gpu error: module 'cast_fp' not loaded
```

Shard status summary:

- GPU0 owns embeddings and layers 0..11.
  - uploaded f16 tensors reported: 0.
  - uploaded f32 norm tensors reported: 0.
  - U8 host tensors: 48, 5,076,172,800 bytes.
  - RoPE bytes: 0.
  - KV bytes: 0.
  - metadata bytes: 0.
  - fused QKV statuses: 12 deferred.
  - input/post norm statuses: 12 deferred / 12 deferred.
  - dense gate/up: 12 `not_applicable`.
  - f16 scratch: `not_applicable`.

- GPU1 owns layers 12..23 and the final head.
  - uploaded f16 tensors reported: 0.
  - uploaded f32 norm tensors reported: 0.
  - U8 host tensors: 48, 5,076,172,800 bytes.
  - RoPE bytes: 0.
  - KV bytes: 0.
  - metadata bytes: 0.
  - fused QKV statuses: 12 deferred.
  - input/post norm statuses: 12 deferred / 12 deferred.
  - dense gate/up: 12 `not_applicable`.
  - f16 scratch: `not_applicable`.

Stderr contained only existing Rust/CUDA build warnings and the cargo command
line; the precise failure was reported in the status JSON.

No split execution, layer construction, `GpuModelRunner` construction,
attention, graph output, final norm execution, LM-head execution, logits, graph
capture, forward pass, serving behavior, or parity path was added or exercised.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

Next bounded step:

- bounded repair for the norm conversion cast-kernel boundary, likely ensuring
  the bench-only per-shard kernel loader has the `cast_fp` module available
  before invoking `cast_f32_tensor_to_f16`.
- status refinement so norm conversion failures classify as a norm/cast
  boundary rather than the coarser fused QKV blocked classification.

Primary classification:

multi_gpu_layer_sharding_real_model_fused_qkv_norm_allocation_cast_error

## Bench-only f16 norm cast kernel repair status

The previous real-model fused QKV + norm conversion allocation smoke failed at:

```text
gpu error: shard 0 f16 norm cast kernel lookup failed: gpu error: module 'cast_fp' not loaded
```

The split bench path allocated fused QKV without kernels, but the new
layernorm/postnorm f32 -> f16 conversion path needs `cast_fp`. The shard
resource loader can be empty when no `--kernel-dir` is supplied, so the
bench-only norm conversion path now obtains the cast kernel explicitly.

### Helper added

`crates/gpt-oss-model-runner/src/fused_f16.rs` now provides:

```text
get_or_load_cast_f32_to_f16_kernel
```

The helper first uses `cast_fp::cast_f32_to_f16_kernel` if it is already loaded
in the shard's `KernelLoader`. If the shard loader has no `cast_fp` module, it
loads only `cast_fp.ptx` from `gpt_oss_gpu::kernel_loader::default_ptx_dir()`
into a temporary loader and returns the loaded `CudaFunction`. The returned
function owns the loaded module, so the shard resource `Arc<KernelLoader>` does
not need mutation.

`CudaShardFusedF16Buffers::create_for_resource` now calls this helper once per
shard when layernorm/postnorm casts are planned, then passes the resulting
kernel handle into the per-layer norm conversion calls.

`GpuModelRunner::fuse_weights` remains behavior-preserving and continues to use
its existing runner-owned loader path.

### Status/classification refinement

The fused allocation error mapping now distinguishes norm cast failures from
fused QKV copy/allocation failures. Norm cast kernel load, lookup, allocation,
or launch failures map to:

```text
multi_gpu_layer_sharding_fused_qkv_norm_allocation_cast_error
```

Successful fused QKV + norm conversion allocation still reports:

```text
multi_gpu_layer_sharding_fused_qkv_norm_allocation_smoke_complete
```

No new status JSON fields were added in this slice; the error string remains
the detailed boundary report.

### Manual operator command

Do not run automatically during this implementation slice:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_norm_status.json
```

### Validation commands

```bash
cargo fmt
cargo test -p gpt-oss-model-runner fused_f16
cargo test -p gpt-oss-model-runner shard
cargo test -p gpt-oss-model-runner f16
cargo test -p gpt-oss-model-runner device_map
cargo test -p gpt-oss-model-runner header
cargo test -p gpt-oss-model-runner safetensor
cargo test -p gpt-oss-model-runner u8
cargo check -p gpt-oss-model-runner --features cuda
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke
cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run
cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke
cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda
git diff --check
```

No real-model CUDA smoke was run for this implementation slice.

### Non-execution boundary

No split execution, layer construction, `GpuModelRunner` construction, attention,
graph output, final norm execution, LM-head execution, logits, graph capture,
forward pass, serving behavior, or parity path was added.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

### Next bounded step

Recommended next slices:

- rerun the real-model fused QKV + norm conversion allocation smoke.
- bias f16 conversion allocation.
- final/embed f16 conversion allocation.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.

### Primary classification

multi_gpu_layer_sharding_norm_cast_kernel_boundary_repaired

## Real-model fused QKV + norm conversion rerun status

Operator rerun date: 2026-05-01.

Command run:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_norm_rerun_status.json
```

The generated JSON and logs are operator artifacts under
`/tmp/multi_gpu_layer_sharding/` and are not committed.

Preflight:

- `nvidia-smi` showed two visible NVIDIA GeForce RTX 3090 devices.
- `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration` existed.
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
  passed with existing warnings.

Result:

- Primary result classification:
  `multi_gpu_layer_sharding_real_model_fused_qkv_norm_allocation_smoke_complete`.
- Command status classification:
  `multi_gpu_layer_sharding_fused_qkv_norm_allocation_smoke_complete`.
- Header merge policy: `allow_restricted_sinks_override`.
- Restricted sinks overrides: 24.
- Resource construction: succeeded.
- Manifest-owned f16/U8 allocation: succeeded.
- Selective f32 norm loading: succeeded with 24 f32 norm payloads per shard.
  The status retained complete shape visibility while uploading only the needed
  f32 norm payloads.
- RoPE allocation: succeeded.
- KV allocation: succeeded with `kv_num_blocks=1` and `kv_block_size=16`.
- Synthetic decode metadata allocation: succeeded with
  `tokens=1`, `seqs=1`, `context_len=1`, `block_size=16`, and
  `metadata_graph_max_blocks=8192`.
- Fused QKV allocation: succeeded.
- Layernorm/postnorm f32 -> f16 conversion: succeeded.
- Fused f16 error: none.
- Manifest validation: 0 unassigned tensors, 0 invalid tensors.

Shard status summary:

- GPU0 owns embeddings and layers 0..11.
  - uploaded f16 tensors: 181, 1,804,456,704 bytes.
  - uploaded f32 norm tensors: 24, 276,480 bytes.
  - U8 host tensors: 48, 5,076,172,800 bytes.
  - RoPE: 2,097,152 bytes.
  - KV: 393,216 bytes.
  - synthetic decode metadata: 32,792 bytes.
  - fused QKV: 12 buffers, 353,894,400 bytes.
  - input layernorm f16: 12 buffers, 69,120 bytes.
  - post-attention norm f16: 12 buffers, 69,120 bytes.
  - fused total: 354,032,640 bytes.
  - dense gate/up: `not_applicable`.
  - f16 scratch: `not_applicable`.

- GPU1 owns layers 12..23 and the final head.
  - uploaded f16 tensors: 182, 1,804,462,464 bytes.
  - uploaded f32 norm tensors: 24, 276,480 bytes.
  - U8 host tensors: 48, 5,076,172,800 bytes.
  - RoPE: 2,097,152 bytes.
  - KV: 393,216 bytes.
  - synthetic decode metadata: 32,792 bytes.
  - fused QKV: 12 buffers, 353,894,400 bytes.
  - input layernorm f16: 12 buffers, 69,120 bytes.
  - post-attention norm f16: 12 buffers, 69,120 bytes.
  - fused total: 354,032,640 bytes.
  - dense gate/up: `not_applicable`.
  - f16 scratch: `not_applicable`.

Per-layer fused status showed every owned absolute layer with:

- fused QKV: `allocated`, 29,491,200 bytes.
- input layernorm f16: `allocated`, 5,760 bytes.
- post-attention norm f16: `allocated`, 5,760 bytes.
- dense gate/up: `not_applicable`.
- QKV/O bias f16: `deferred`.

Stderr contained existing Rust/CUDA build warnings and the cargo command line;
the status JSON reported no top-level error.

No split execution, layer construction, `GpuModelRunner` construction,
attention, graph output, final norm execution, LM-head execution, logits, graph
capture, forward pass, serving behavior, or parity path was added or exercised.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

Next bounded step:

- bias f16 conversion allocation for QKV and O-projection biases.
- final/embed f16 conversion allocation.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.

Primary classification:

multi_gpu_layer_sharding_real_model_fused_qkv_norm_allocation_smoke_complete

## Bench-only f16 bias conversion allocation status

This implementation slice extends the non-executing `--allocate-fused-f16`
bench path to cover per-layer attention bias preconversion after the real-model
fused QKV + norm allocation smoke succeeded.

Helper/module names used or added:

- `fused_qkv_bias_num_elements` in
  `crates/gpt-oss-model-runner/src/fused_f16.rs`.
- `CudaLayerFusedF16Buffers::create_for_layer` in
  `crates/gpt-oss-model-runner/src/sharded_resources.rs`.
- `allocate_fused_qkv_bias_f16` and `allocate_o_proj_bias_f16` in the
  CUDA-gated sharded resource helper.
- `ShardWeightStore` remains the ownership/naming boundary before any tensor
  lookup.

Actual allocation behavior:

- For each shard-owned absolute layer, the bench path validates ownership with
  `ShardWeightStore` before resolving bias tensor names.
- If all Q/K/V bias tensors are present, it assembles one f32 CUDA buffer in
  runner-compatible `Q || K || V` layout, casts that buffer to f16 with
  `cast_f32_tensor_to_f16`, and reports `qkv_bias_f16_status=allocated`.
- If no Q/K/V bias tensors are present, QKV bias conversion reports
  `not_applicable`.
- If only some Q/K/V bias tensors are present, allocation fails at the bias
  boundary with the exact missing tensor names.
- If `self_attn.o_proj.bias` is present for an owned absolute layer, it is cast
  from f32 to f16 and reports `o_proj_bias_f16_status=allocated`; absence is
  `not_applicable`.

Selective f32 loading under `--dtype f16`:

- `multi_gpu_layer_sharding_split_allocation_smoke` now selectively loads only
  the f32 tensors needed for already-implemented f16 preconversion:
  input layernorm, post-attention layernorm, Q/K/V bias, and O-projection bias.
- The command still does not upload all f32 weights to every shard.

Status JSON behavior:

- Per layer:
  - `qkv_bias_f16_status`
  - `qkv_bias_f16_bytes`
  - `o_proj_bias_f16_status`
  - `o_proj_bias_f16_bytes`
- Per shard:
  - `f16_qkv_bias_count`
  - `f16_o_proj_bias_count`
  - `f16_qkv_bias_total_bytes`
  - `f16_o_proj_bias_total_bytes`
  - `fused_total_bytes`, now including fused QKV, optional dense gate/up,
    input/post norm f16, QKV bias f16, and O-projection bias f16 bytes.

Classification behavior:

- Successful fused QKV + norm + bias allocation reports:
  `multi_gpu_layer_sharding_fused_qkv_norm_bias_allocation_smoke_complete`.
- Bias-specific allocation failures classify as:
  `multi_gpu_layer_sharding_bias_conversion_allocation_blocked`.

Still deferred:

- final norm f16 conversion.
- embedding f16 conversion.
- tied LM-head fallback.
- `F16LayerScratch`.
- GPT-OSS MoE GPU upload.
- layer construction and `GpuModelRunner` construction.
- attention, graph output, execution, serving, logits, and parity paths.

Manual operator command, not run in this implementation slice:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_bias_status.json
```

Validation commands for this slice:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner fused_f16`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda`
- `git diff --check`

No split execution, layer construction, `GpuModelRunner` construction,
attention, graph output, final norm execution, LM-head execution, logits, graph
capture, forward pass, serving behavior, or parity path was added.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

Primary classification:

multi_gpu_layer_sharding_fused_qkv_norm_bias_allocation_smoke_complete

## Real-model fused QKV + norm + bias allocation smoke status

This operator slice reran the real restricted model with the full current
bench-only allocation baseline after adding QKV/O bias f16 conversion.

Command run:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_bias_status.json
```

The output JSON and redirected logs are under `/tmp/multi_gpu_layer_sharding`
and are not committed.

Run configuration:

- model path: `/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`.
- dtype: `f16`.
- device map: `split:0-11@0,12-23@1`.
- restricted sinks override: enabled.
- RoPE metadata allocation: enabled.
- KV cache allocation: enabled with `kv_num_blocks=1`, `kv_block_size=16`.
- synthetic decode metadata allocation: enabled with tokens/seqs/context/block
  size `1 / 1 / 1 / 16`.
- fused f16 allocation: enabled.

Result:

- primary result classification:
  `multi_gpu_layer_sharding_real_model_fused_qkv_norm_bias_allocation_smoke_complete`.
- command status classification:
  `multi_gpu_layer_sharding_fused_qkv_norm_bias_allocation_smoke_complete`.
- resource construction: succeeded.
- manifest-owned f16/U8 allocation and retention: succeeded.
- selective f32 norm/bias loading: succeeded.
- RoPE allocation: succeeded.
- KV allocation: succeeded.
- synthetic decode metadata allocation: succeeded.
- fused QKV allocation: succeeded.
- input/post-attention norm f16 conversion: succeeded.
- QKV/O bias f16 conversion: succeeded.
- unassigned tensor count: 0.
- invalid tensor count: 0.

Shard summary:

- GPU0 owns embeddings and absolute layers 0..11:
  - f16 tensors: 181.
  - f32 norm/bias tensors: 72.
  - U8 host tensors: 48.
  - RoPE: 2,097,152 bytes.
  - KV: 393,216 bytes.
  - metadata: 32,792 bytes.
  - fused QKV: 12 buffers, 353,894,400 bytes.
  - input layernorm f16: 12 buffers, 69,120 bytes.
  - post-attention norm f16: 12 buffers, 69,120 bytes.
  - fused QKV bias f16: 12 buffers, 122,880 bytes.
  - O-projection bias f16: 12 buffers, 69,120 bytes.
  - fused total: 354,224,640 bytes.
  - dense gate/up: `not_applicable`.
  - f16 scratch: `not_applicable`.
- GPU1 owns absolute layers 12..23 and the final head:
  - f16 tensors: 182.
  - f32 norm/bias tensors: 72.
  - U8 host tensors: 48.
  - RoPE: 2,097,152 bytes.
  - KV: 393,216 bytes.
  - metadata: 32,792 bytes.
  - fused QKV: 12 buffers, 353,894,400 bytes.
  - input layernorm f16: 12 buffers, 69,120 bytes.
  - post-attention norm f16: 12 buffers, 69,120 bytes.
  - fused QKV bias f16: 12 buffers, 122,880 bytes.
  - O-projection bias f16: 12 buffers, 69,120 bytes.
  - fused total: 354,224,640 bytes.
  - dense gate/up: `not_applicable`.
  - f16 scratch: `not_applicable`.

Per-layer fused status showed every owned absolute layer with:

- fused QKV: `allocated`.
- input layernorm f16: `allocated`.
- post-attention norm f16: `allocated`.
- QKV bias f16: `allocated`.
- O-projection bias f16: `allocated`.
- dense gate/up: `not_applicable`.

Stderr contained existing Rust/CUDA build warnings and the cargo command line;
the status JSON reported no top-level fused error.

No exact failure boundary remains for this slice.

No split execution, layer construction, `GpuModelRunner` construction,
attention, graph output, final norm execution, LM-head execution, logits, graph
capture, forward pass, serving behavior, or parity path was added or exercised.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

Next bounded step:

- final/embed f16 conversion allocation.
- `F16LayerScratch` visibility/sizing implementation.
- GPT-OSS MoE GPU upload design.
- layer-construction skeleton without execution.

Primary classification:

multi_gpu_layer_sharding_real_model_fused_qkv_norm_bias_allocation_smoke_complete

## Bench-only global f16 side-buffer allocation status

This implementation slice extends the non-executing `--allocate-fused-f16`
bench path to cover the remaining global f16 side-buffer boundary that is not
layer-local fused QKV/norm/bias state.

Helper/module names used or added:

- `CudaShardFusedF16Buffers::create_for_resource` in
  `crates/gpt-oss-model-runner/src/sharded_resources.rs`.
- `embedding_f16_available_from_uploaded` for embedding-shard f16 source
  reporting.
- `allocate_final_norm_f16` for final-shard `model.norm.weight` f32 to f16
  conversion.
- `ShardWeightStore` remains the ownership/naming boundary for both global
  tensors.
- `cast_f32_tensor_to_f16` remains the CUDA cast helper for final norm.

Embedding shard policy:

- `model.embed_tokens.weight` is considered only on the shard that owns
  embeddings.
- When the embedding shard already has the manifest-owned uploaded f16
  embedding tensor, status reports `embedding_f16_status=available_from_uploaded_f16`,
  `embedding_f16_source=uploaded_f16`, and the uploaded f16 byte count.
- The bench path does not create a second embedding copy in this slice and does
  not upload or cast a full f32 embedding fallback table.
- The final shard does not make `model.embed_tokens.weight` loadable for tied
  LM-head fallback.

Final shard policy:

- `model.norm.weight` is considered only on the shard that owns the final head.
- Under `--dtype f16`, the bench path selectively loads only `model.norm.weight`
  as an f32 tensor on the final shard, then casts it with
  `cast_f32_tensor_to_f16`.
- Successful allocation reports `final_norm_f16_status=allocated`,
  `final_norm_f16_source=f32_cast`, and `final_norm_f16_bytes`.
- Final norm f16 bytes are included in `fused_total_bytes` because they are a
  newly allocated fused/preconverted side buffer.

Selective f32 loading under `--dtype f16` now covers only the f32 tensors needed
for implemented f16 preconversion:

- per-layer input layernorm.
- per-layer post-attention layernorm.
- per-layer Q/K/V biases.
- per-layer O-projection bias.
- `model.norm.weight` on the final shard.

It still does not upload all f32 weights and does not selectively load the full
f32 embedding table.

Status JSON behavior:

- Per shard:
  - `embedding_f16_allocated`
  - `embedding_f16_status`
  - `embedding_f16_bytes`
  - `embedding_f16_source`
  - `final_norm_f16_allocated`
  - `final_norm_f16_status`
  - `final_norm_f16_bytes`
  - `final_norm_f16_source`
- Successful fused QKV + norm + bias + global f16 side-buffer handling reports:
  `multi_gpu_layer_sharding_fused_qkv_norm_bias_global_f16_allocation_smoke_complete`.
- Global f16-specific failures classify as:
  `multi_gpu_layer_sharding_global_f16_allocation_blocked`.

Still deferred:

- tied LM-head fallback.
- lm-head-derived fallback/copy behavior.
- f32 embedding fallback loading/casting.
- `F16LayerScratch`.
- GPT-OSS MoE GPU upload.
- layer construction and `GpuModelRunner` construction.
- attention, graph output, execution, serving, logits, and parity paths.

Manual operator command, not run in this implementation slice:

```bash
CUDA_VISIBLE_DEVICES=0,1 cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --device-map split:0-11@0,12-23@1 \
  --selected-device 0 \
  --dtype f16 \
  --allow-restricted-sinks-override \
  --allocate-rope-metadata \
  --allocate-kv-cache \
  --kv-num-blocks 1 \
  --kv-block-size 16 \
  --allocate-metadata \
  --metadata-mode decode \
  --metadata-num-tokens 1 \
  --metadata-num-seqs 1 \
  --metadata-context-len 1 \
  --metadata-block-size 16 \
  --allocate-fused-f16 \
  --output /tmp/multi_gpu_layer_sharding/split_allocation_f16_fused_global_status.json
```

Validation commands for this slice:

- `cargo fmt`
- `cargo test -p gpt-oss-model-runner fused_f16`
- `cargo test -p gpt-oss-model-runner shard`
- `cargo test -p gpt-oss-model-runner f16`
- `cargo test -p gpt-oss-model-runner device_map`
- `cargo test -p gpt-oss-model-runner header`
- `cargo test -p gpt-oss-model-runner safetensor`
- `cargo test -p gpt-oss-model-runner u8`
- `cargo check -p gpt-oss-model-runner --features cuda`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke`
- `cargo run -p gpt-oss-bench --bin multi_gpu_layer_sharding_split_allocation_smoke --features cuda -- --help`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run dry_run`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_dry_run`
- `cargo test -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke`
- `cargo check -p gpt-oss-bench --bin multi_gpu_layer_sharding_cuda_resource_smoke --features cuda`
- `git diff --check`

No split execution, layer construction, `GpuModelRunner` construction,
attention, graph output, final norm execution, LM-head execution, logits, graph
capture, forward pass, serving behavior, or parity path was added.

Serve/runtime split maps remain non-executable and continue to be rejected
before CUDA allocation.

Primary classification:

multi_gpu_layer_sharding_fused_qkv_norm_bias_global_f16_allocation_smoke_complete
