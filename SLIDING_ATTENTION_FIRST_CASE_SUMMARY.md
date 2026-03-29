# Sliding Attention First Case — Vertical Slice Summary

## Exact Case Definition

- single sequence
- single layer
- `sliding_attention`
- `sliding_window = 128`
- prefill `128`
- decode `1` at absolute position `128`
- no sinks
- no graph
- no MoE widening
- experimental / non-trusted

## What Was Proven

- Planned-reference:
  the exact artifact-aligned `sliding_attention` case is representable and
  diagnosable on the experimental planned-reference path.
- Conformance:
  the same bounded case is plannable in `Experimental` without widening trusted
  claims.
- Mock-boundary diagnosis:
  the CPU/mock observed path rejects this case for the right reason: it does
  not expose per-layer sliding-window controls or decode visibility continuity.
- CUDA `gpu_layer`:
  the `128 -> 129` boundary behaves correctly in direct eager CUDA layer
  execution.
- CUDA `GpuModelRunner`:
  the case survives runner-level config translation and packed metadata
  continuity.
- CUDA `GpuWorker`:
  the case survives worker-level request/input shaping and worker-to-runner
  handoff on the eager CUDA path.

## Key Technical Confirmations

- Correct sliding eviction at the `128 -> 129` boundary.
- Metadata continuity for `positions`, `context_lens`, `slot_mapping`, and
  `block_tables`.
- Correct propagation of `sliding_attention` and `sliding_window = 128`.
- Successful CUDA eager execution through worker-level runtime with finite
  logits for the bounded case.

## What This Does Not Prove

- no general sliding-attention support
- no multi-layer scheduling
- no batching or scheduler behavior
- no sink support
- no graph or replay support
- no trusted-mode guarantee
- no alias normalization for `local_attention`

## First Unresolved Seams

- worker and engine scheduler surfaces
- multi-layer sliding interaction
- integration with learned sinks
- graph replay compatibility

## Conclusion

This lane establishes experimental, bounded evidence that a single
artifact-aligned `sliding_attention` case executes correctly on CUDA eager
through worker-level runtime, without implying general or trusted support.
