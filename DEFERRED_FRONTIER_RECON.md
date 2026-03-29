# Deferred Frontier Recon

Workstream A remains the active path for the first restricted live 20B run.
Workstream B is reconnaissance and planning only.
Graph, sliding/local attention, and sink attention remain intentionally deferred unless explicitly stated otherwise.

## Scope

This note records the deferred frontier after the current `main` baseline.
It is not an implementation plan for the current run-readiness lane.

## Current Baseline

- Trusted GPT-OSS currently supports the full-attention lane only.
- Trusted planning rejects sliding/local attention before execution.
- Trusted startup rejects nonzero GPT-OSS sink tensors at load time.
- Experimental graph decode plumbing exists, but trusted GPT-OSS graph replay is still forbidden.
- Reference and conformance already model some deferred semantics, but they do not yet make those frontiers honest supported claims.

## Graph

**Status**

- Deferred.
- Experimental graph decode plumbing exists.
- Trusted GPT-OSS graph replay is still intentionally forbidden.

**Why deferred**

- The runtime path exists, but the parity story is not honest yet.
- Conformance does not currently drive the real graph capture/replay path.
- Reference traces do not model graph replay invariants such as padding, capture, replay, and async readback.

**Current known coverage**

- Planner legality and graph selection in `crates/gpt-oss-runtime-plan/src/lib.rs`.
- Worker/graph runner plumbing in `crates/gpt-oss-engine/src/worker/gpu_worker.rs` and `crates/gpt-oss-engine/src/worker/graph_runner.rs`.
- CUDA graph pool/unit coverage in `crates/gpt-oss-gpu/src/cuda_graph.rs`.
- Conformance can record graph policy metadata, but not honest graph execution parity.

**Known or likely missing semantics**

- Honest observed graph execution traces in conformance.
- A replay-safe parity contract for padded decode batches.
- Clear graph-specific parity boundaries separate from eager semantics.

**Likely code seams**

- `crates/gpt-oss-runtime-plan/src/lib.rs`
- `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
- `crates/gpt-oss-engine/src/worker/graph_runner.rs`
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
- `crates/gpt-oss-conformance/src/case.rs`

**Smallest honest first case**

- Experimental mode only
- Dense full-attention only
- Decode only
- Greedy only
- No sinks
- No sliding/local attention
- No MoE
- Batch size `1` or `2` with graph-compatible padded batch

That first case should prove:
- planner chooses graph
- worker takes graph path
- first eligible request warms/captures
- subsequent request replays
- replayed decode matches eager for that narrow case

**Recommended order**

- Tackle after sliding/local eager semantics and sink eager semantics are each honest.

**Dependencies / risk to current run-readiness lane**

- High risk if started too early.
- Depends on stable eager semantics, stable metadata shape, and stable output unpadding.
- Most likely frontier to destabilize current full-attention work if reopened before the first restricted live run succeeds.

## Sliding / Local Attention

**Status**

- Deferred in trusted mode.
- Semantically modeled.
- Partially wired in GPU/kernel code.
- Not honestly supported on the trusted observed lane.

**Why deferred**

- Trusted planner explicitly rejects it today.
- CPU/mock GPT-OSS execution still rejects sliding attention.
- Conformance does not yet have an observed backend that can honestly prove sliding/local parity.

**Current known coverage**

- Semantic parsing of `sliding_attention` and `local_attention` in `crates/gpt-oss-semantics/src/lib.rs`.
- Reference visibility and trace modeling in `crates/gpt-oss-reference/src/executor.rs`.
- Decode position and sink/sliding trace tests in `crates/gpt-oss-reference/src/lib.rs`.
- Sliding window logic and tests in `crates/gpt-oss-model-runner/src/attention/sliding_window.rs`.
- CUDA layer/kernel support in `crates/gpt-oss-model-runner/src/gpu_layer.rs` and `kernels/flash_attention.cu`.

**Known or likely missing semantics**

- Honest observed-side parity for block-table-driven sliding visibility.
- A shared accepted meaning of `local_attention` vs `sliding_attention` across planner, semantics, and GPU runner.
- Trusted continuity expectations across prefill/decode with real cache metadata.

**Likely code seams**

- `crates/gpt-oss-runtime-plan/src/lib.rs`
- `crates/gpt-oss-conformance/src/case.rs`
- `crates/gpt-oss-reference/src/executor.rs`
- `crates/gpt-oss-kv-model/src/lib.rs`
- `crates/gpt-oss-model-runner/src/architectures/gpt_oss.rs`
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`
- `crates/gpt-oss-model-runner/src/attention/sliding_window.rs`

**Smallest honest first case**

- Experimental planned-reference case only
- Dense only
- One layer of `sliding_attention`
- `sliding_window = 2`
- `sink_tokens = 0`
- Prefill `[a, b, c]` at `seq_start_pos = 0`
- Decode `[d]` at `seq_start_pos = 3`

That first case should prove:
- semantic acceptance
- visible-token set in reference
- absolute-position continuity across prefill/decode
- no claim yet about trusted observed parity

**Recommended order**

- First deferred frontier to tackle after the restricted live 20B run.

**Dependencies / risk to current run-readiness lane**

- Medium risk.
- Shares cache/state and position semantics with sink attention.
- Lower risk than graph because the main missing pieces are still localizable to visibility and observed parity.

## Sink Attention

**Status**

- Deferred in trusted mode.
- Reference and CUDA kernels know about sink-aware visibility.
- Trusted startup and CPU/mock execution intentionally reject it today.

**Why deferred**

- Trusted startup rejects nonzero sink tensors.
- Trusted planner already rejects the sliding/local path that sink visibility currently depends on.
- Conformance cannot yet represent sink/sliding observed traces honestly.

**Current known coverage**

- Sink/sliding reference visibility in `crates/gpt-oss-reference/src/executor.rs`.
- Sink-aware decode/trace tests in `crates/gpt-oss-reference/src/lib.rs`.
- Runtime load-time rejection in `crates/gpt-oss-engine/src/worker/gpu_worker.rs`.
- Sink parameters in CUDA layers and kernels in `crates/gpt-oss-model-runner/src/gpu_layer.rs` and `kernels/flash_attention.cu`.
- KV visibility semantics in `crates/gpt-oss-kv-model/src/lib.rs`.

**Known or likely missing semantics**

- Honest observed sink trace shape in conformance.
- A single explicit story that connects planner legality, worker startup legality, and observed parity.
- Alignment between reference absolute positions and GPU packed query offsets.
- Proof that paged KV/block tables preserve sink-visible tokens as intended.

**Likely code seams**

- `crates/gpt-oss-runtime-plan/src/lib.rs`
- `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
- `crates/gpt-oss-engine/src/worker/input.rs`
- `crates/gpt-oss-kv-model/src/lib.rs`
- `crates/gpt-oss-reference/src/executor.rs`
- `crates/gpt-oss-conformance/src/case.rs`
- `crates/gpt-oss-model-runner/src/gpu_layer.rs`

**Smallest honest first case**

- Experimental CUDA-eager only
- Single sequence
- Single layer of `sliding_attention`
- `sink_tokens = 1`
- Small `sliding_window`
- No MoE
- One prefill case plus one decode continuation case

That first case should prove:
- sink-visible token set is correct
- prefill/decode bookkeeping is consistent
- no trusted-mode claim yet
- no multi-block or batch interaction yet

**Recommended order**

- Second deferred frontier, after sliding/local eager semantics are nailed down.

**Dependencies / risk to current run-readiness lane**

- Medium-high risk.
- Shares visibility and cache seams with sliding/local attention.
- More bookkeeping-sensitive than sliding because sink-visible prefixes must remain stable across decode continuation.

## Cross-Frontier Ordering

Recommended order after the first restricted live 20B run:

1. sliding/local attention
2. sink attention
3. graph parity

Why:

- Sliding/local has the clearest semantic/reference groundwork already in place.
- Sink attention reuses much of the same visibility and cache-state surface, but adds sink-specific bookkeeping risk.
- Graph parity depends on stable eager semantics and stable replay metadata shape, so it should stay last.

## Shared Seams And Orthogonality

Mostly shared:

- cache/state layout
- sequence position semantics
- attention visibility/mask/window logic

Partially shared:

- planner legality and trace labeling
- conformance trace shape

More orthogonal:

- graph capture/replay pool mechanics themselves
- CUDA graph warmup/replay lifecycle

Important caution:

- Graph looks deceptively close because the code surface already exists.
- In practice it is the highest-risk frontier because it sits on top of all the other semantics and replay-shape assumptions.

## What Remains Intentionally Deferred

- Trusted graph replay claims
- Trusted sliding/local attention claims
- Trusted sink-attention claims
- Multi-block sink cases
- Sliding/sink plus MoE combinations
- Any planner relaxation that would widen the current trusted surface before parity is proven
