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

## Primary classification

multi_gpu_layer_sharding_single_device_map_plumbing_complete
