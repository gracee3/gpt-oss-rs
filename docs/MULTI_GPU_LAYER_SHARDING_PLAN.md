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

## Primary classification

multi_gpu_layer_sharding_real_model_split_allocation_f16_blocked
