# Current-path validation and component disposition

## Local 20B control

The control used the repository's internal `cpu_parity` binary, not HTTP. It was
built `--locked --release` from the detached clean checkout at
`0113e8214e765d168216bbee2120654555a4cfe4`; binary SHA-256 is
`cb1d151b72db3a42ed2732489ba9c7f53245ec95f061e6581057de497f1bfea4`.
The model revision is `6cee5e81ee83917806bbde320786a8fb61efebee`, fixture
`cpu_harmony_parity.json`, scenario `harmony_63`, eight threads,
`residual-q8`, and `auto` dispatch.

Safety limits were 80 GiB RSS, 8 GiB minimum host available memory, and 256 MiB
maximum swap growth. All three runs returned zero, triggered no stop, used no
process or system swap, and left both GPUs idle apart from driver allocations.

| Run | Result | Time and memory | Interpretation |
|---|---|---|---|
| max-new-tokens 0 | Loaded, created the x8 cache, and completed the 63-token prefill | 125.320 s; startup 92.623 s; prompt 32.127 s; peak RSS 20,698,779,648 B (19.277 GiB) | **Verified:** bounded cold load+prefill. The binary exposes no load-only command, so this is not mislabeled as pure load. |
| max-new-tokens 8, layer-0 trace at retained step 1 | Generated `[200005, 35644, 200008, 976, 1825, 5003, 25, 392]`, exactly the pinned official sequence; trace contains four rank records | 96.285 s; startup 58.296 s; prompt 32.548 s; generation 4.099 s; peak RSS 10,893,111,296 B (10.145 GiB) | **Verified:** current-host retained-continuation and real-weight one-layer trace pass. Layer-0 selected IDs were `[31, 21, 22, 6]`. |
| max-new-tokens 1, layer-major profile | First generated token `200005` matched; 3,949 records retained with no drop/truncation | 275.346 s; startup 58.080 s; prompt 216.231 s; peak RSS 9,490,948,096 B (8.838 GiB) | **Verified:** actual route occupancy and current scalar-prefill bottleneck; not a throughput benchmark. |

The retained result/monitor/manifest hashes are indexed in
[`control-20b-summary.json`](evidence/research-2026-08/control-20b-summary.json).
The generic manifest labels this one-off run `status: insufficient_evidence`
because it is not a complete Fresh CPU Oracle Campaign capture. **Conflict
avoided:** that campaign-level label is retained, while the concrete pinned
sequence comparison above is recorded as a local pass and no broader CPU policy
is promoted.

The profile observed 6,048 prefill routes (`24 layers × 63 rows × top-4`) in 597
nonempty expert buckets: median `M=7`, p75 13, p90 24, p95 33, max 61. The decode
step yielded 96 `M=1` buckets. This supplies the representative dimensions used
by the transfer and oracle research.

## Current CUDA MoE primitives

**Conflict:** the Phase 1 handoff calls these “four MoE-specific PTX modules.”
Source contains one PTX module, `gpt_oss_moe`, generated from
`kernels/gpt_oss_moe.cu`, exporting four MoE kernel entry points. The Phase 0
code map correctly called them four kernels.

| Entry point/current operation | Shape/layout and synchronization | Correctness/reuse disposition |
|---|---|---|
| `gpt_oss_route_topk_kernel` | f32 router logits → i32 IDs/f32 weights; local arrays capped by `MAX_EXPERTS=64`; one runner stream | **Reject for 120B and exact proof:** 120B has 128 experts; rounding/tie behavior is not proven against the CPU trace. |
| `gpt_oss_select_expert_inputs_kernel` | For one expert ID, scans token top-k IDs and writes a full masked f32 input plus route weight | **Replace/bypass:** loops across all experts and materializes zeros instead of consuming a packed selected bucket. |
| `gpt_oss_dequant_expert_f16_kernel` | U8 MXFP4 blocks/scales → full FP16 projection matrix for one expert | **Primitive concept only:** bytes decode correctly in unit scope, but allocation/expansion is per expert per call and no exact selected-expert fixture exists. |
| cuBLAS gate/up and down | Full FP16 matrices; gate/up allocation 33,177,600 B and down 16,588,800 B per expert/call | **Reject current lifecycle:** transient expansion is incompatible with bounded 120B residency and excludes exact BF16 semantics. Low-level cuBLAS wrapper remains usable for separately validated dense operations. |
| `fused_silu_mul_split` | ordinary SiLU × up on FP32/FP16-shaped buffers | **Reject for GPT-OSS exactness:** does not implement alpha 1.702, clamps, `up+1`, and prescribed BF16 round points. |
| `gpt_oss_weighted_add_kernel` | f32 expert-ID loop accumulation into full output | **Reject exact reduction:** contribution order follows expert scan, not original routing rank. |

`GptOssMoeLayerWeights::forward` has two materially different paths:

- **Prefill:** clones the full f32 normalized activation to host, runs a scalar
  host f32 MoE over host-owned full weights, then uploads output. It neither
  calls the exact CPU BF16 contract nor uses pinned expert staging.
- **Decode/current CUDA path:** computes router on one GPU, then iterates every
  expert, masks routes, expands both matrices, runs cuBLAS, and accumulates.
  All expert weights are layer/runner/device-shaped.

**Conclusion:** neither is a correctness base for heterogeneous execution. The
current CUDA path compiles and its low-level tests are valuable, but full-model
exactness cannot be inferred from them.

## Tensor-parallel/NCCL conflict

The conflict is definite at source level for `tp_size=2`, not merely
undocumented:

```text
global Q weight       [4096, 2880]
rank-local Q shard    [2048, 2880]       // shard Along(0)
forward GEMM request  N=4096, K=2880     // global dimensions retained

global O weight       [2880, 4096]
rank-local O shard    [2880, 2048]       // shard Along(1)
forward O input       K=4096             // global activation retained
```

The same mismatch appears in expert gate/up sharding along output dimensions
and down sharding along input dimensions while `GptOssMoeLayerWeights` retains
global intermediate sizes. Downstream buffer sizes, fused copy ranges, and GEMM
contracts therefore disagree with the shard producer. All-reduce occurs after
O projection and MoE down projection; it cannot repair an invalid preceding
local GEMM or out-of-bounds interpretation.

The engine can construct one worker per rank, broadcast the same input, launch
all ranks, collect outputs, and consume rank 0. This proves worker scaffolding,
not model correctness. The standalone `gpt-oss-gpu::nccl` wrapper is additionally
unsafe for real use as written: it accepts Rust host byte slices and passes
their pointers to NCCL device-buffer APIs on a null/default stream. The separate
`tensor_parallel.rs` cudarc wrapper at least uses device tensors, but no retained
real-model TP parity evidence exists.

**Closed evidence statement:** the current GPT-OSS tensor-parallel model route
is dimensionally incorrect for sharded execution and must not be used as the
heterogeneous foundation. Its low-level worker/channel idea may be audited and
adapted; its model sharding and standalone NCCL wrapper are rejected.

## Reuse/reject table

| Component | Disposition | Reason / missing proof |
|---|---|---|
| `CpuModel::moe_batch`, scalar MXFP4, GPT SwiGLU, trace schema | **Trust as semantic oracle** | Real 20B retained sequence and layer trace passed; arithmetic boundaries are explicit. |
| `gpt-oss-moe-semantics` stable top-k/group/reduce | **Reuse-as-is candidate** | Backend-neutral ordering machinery; caller must preserve BF16 round points and route rank. |
| CPU x8 repack/AVX2/AVX-512 VNNI kernels | **Reuse-after-adaptation candidate** | Exact output hashes in bounded matrix runs; construction currently whole-model/cache-shaped, not owner-selective. Closed policy still governs multirow auto dispatch. |
| `PreparedCpuStep`, revision validation, discard/commit | **Reuse-after-adaptation candidate** | Proven transaction concept; types own CPU KV/request deltas and lack device events. |
| `PinnedBuffer`/`PinnedPool` | **Trust low-level allocation; adapt consumer** | Real CUDA tests and external transfer evidence pass; production expert path does not use it and needs pooling/backpressure. |
| CUDA device/stream/module/error wrappers | **Reuse-after-audit candidate** | Real CUDA tests and synchronized harnesses prove basic behavior; current runner uses one stream and disables event tracking. |
| cuBLAS wrappers | **Reuse for suitable validated projections** | Real cuBLAS tests pass; current expert expansion and activation semantics do not. |
| GPU worker commands/channels | **Reuse-after-audit candidate** | Multi-device creation/launch exists; cancellation, provisional ownership, and model dimensions are not safe yet. |
| Current CUDA prefill/decode MoE | **Reject/bypass** | Host scalar f32 prefill; all-expert, full-FP16, non-exact decode; E≤64 kernel cap. |
| TP sharding/global model route | **Reject** | Definite producer/consumer shape contradiction. |
| Standalone `gpt-oss-gpu::nccl` host-slice wrapper | **Reject** | Host pointers passed as device buffers; null/default stream; mock tests do not validate real semantics. |
| Existing memory pools/accounting | **Adapt candidate** | Useful reservation/cleanup mechanics; no owner-specific expert, pinned staging, or aggregate transaction accounting. |
| CPU trace/manifest/comparison | **Adapt candidate** | Strong first-divergence schema; needs placement, bytes, events, device timing, and commit result. |
| Harmony/HTTP | **Leave closed** | Internal 20B model correctness can be validated independently; prior HTTP liveness remains semantically insufficient. |

The pre-existing CUDA compile warnings cataloged in
[the repository baseline](01-repository-baseline.md#existing-cuda-warning-inventory-not-fixed-here)
remain untouched. No warning cleanup or production patch was attempted in this
phase.
