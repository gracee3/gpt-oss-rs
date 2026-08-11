# Native GPT-OSS CPU Runtime

The initial CPU runtime is intentionally narrow: Linux, batch size one,
official GPT-OSS SafeTensors, BF16 dense weights, and MXFP4 experts. It does
not route through the CUDA runner or the mock architecture.

## Weight ownership

`CpuTensorStore` maps every SafeTensors shard read-only and borrows dense
tensors directly from those mappings. It does not construct the existing
GPU-shaped `ModelWeights` collection. The snapshot must remain immutable for
the lifetime of the runner; Hugging Face content-addressed snapshot blobs meet
that requirement.

Only `gate_up_proj` and `down_proj` MXFP4 tensors are repacked. Cache keys cover
the resolved model revision, every source-shard SHA-256, the tensor name and
shape, repack format version, and layout version. A repacked record is the E8M0
scale byte followed by the 16 adjacent-nibble bytes for one 32-value block.
Writers use an exclusive lock, a synced temporary file, atomic rename, and a
directory sync. Published files are mapped read-only and never changed in
place.

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
  official interleaved clamped GPT-OSS SwiGLU formula.

The runtime remains experimental until the separate 32 GiB i7 full-checkpoint,
cross-runtime, API, and memory gates are complete. Trusted mode must continue
to reject CPU serving until that evidence exists.

## Serving policy

The server owns a real `CpuWorker` and `CpuModelRunner`; CPU requests never
pass through the GPU or mock executor. Prefill and decode are sequential and
the existing sampler handles greedy and stochastic generation. Chat
Completions and Responses, including their streaming forms, share that engine
path and retain the existing Harmony rendering and parsing.

`--device auto` prefers usable CUDA and otherwise selects CPU for GPT-OSS.
`--device cpu` is explicit, while `--device mock` is an explicit test-only
choice. CPU rejects non-GPT-OSS models, request batching, tensor parallelism,
pipeline parallelism, CUDA graphs, and trusted mode. The `gpt-oss-cpu` profile
sets `max_model_len=8192` and `max_num_seqs=1` unless the user supplies a
stricter supported value.

`--cpu-repack-cache` defaults first to `GPT_OSS_RS_CACHE`, then
`XDG_CACHE_HOME/gpt-oss-rs`, then `$HOME/.cache/gpt-oss-rs`. Model snapshots,
repacked expert tensors, build artifacts, and benchmark output remain outside
Git.

## Kernel dispatch

Forced `scalar`, `avx2`, and `avx512-vnni` requests use that path for every
operation and fail before execution when the requested ISA is unavailable.
The public compatibility `path()` remains the highest selected host path.

Automatic dispatch resolves an immutable per-operation plan. On a host with
AVX2/FMA and AVX-512/VNNI, BF16 matvec, Q8 quantization, and RMS norm use
AVX-512/VNNI, while MXFP4/Q8 expert dot products use AVX2. This preserves the
i7's faster AVX2 expert kernel without giving up the faster AVX-512 dense and
normalization kernels. Startup logs include both the compatibility path and
the complete read-only dispatch-plan description.
