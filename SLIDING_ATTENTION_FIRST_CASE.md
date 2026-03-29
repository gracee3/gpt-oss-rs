# Sliding Attention First Case

## Purpose

This lane prepares the smallest honest deferred implementation target for one
concrete `sliding_attention` decode case aligned to the artifact-backed GPT-OSS
model shape.

This is a setup-and-launch-note lane only. Implementation has not started.

## Artifact-backed Facts

- `sliding_attention` is the concrete first deferred frontier. This lane should
  not frame the work as generic `local_attention`.
- `sliding_window = 128` is the exported window shape in the HF config.
- `layer_types` is artifact-backed in the HF-export config and encodes the
  alternating attention schedule.
- Learned `sinks` are a later increment on the same attention row and are not
  the same thing as repo-local `sink_tokens`.
- Graph remains a separate runtime replay frontier and is out of scope here.

## In Scope

- Preparing a future implementation lane for exactly one target case:
  - one concrete `sliding_attention` decode slice
  - single sequence
  - single layer
  - `sliding_window = 128`
  - no learned sinks enabled as the primary assertion
  - no MoE
  - no graph
  - no trusted-mode widening

## Out Of Scope

- Any semantic implementation
- Any runtime, semantics, reference, model-runner, or conformance behavior
  change
- Planner relaxation
- Sink work
- Graph work
- Shared harness refactoring
- Any interference with Workstream A

## Smallest Honest First Case

The first implementation target for this lane should be one artifact-aligned
`sliding_attention` decode slice:

- single sequence
- single layer
- real exported term: `sliding_attention`
- real exported window: `sliding_window = 128`
- no learned sinks as the primary assertion
- no MoE
- no graph capture or replay
- no trusted support claim until parity is proven

## Likely Code Seams

Inspect these first when implementation begins:

- `crates/gpt-oss-semantics/src/lib.rs`
- `crates/gpt-oss-runtime-plan/src/lib.rs`
- `crates/gpt-oss-conformance/src/case.rs`
- `crates/gpt-oss-conformance/src/harness.rs`
- `crates/gpt-oss-reference/src/executor.rs`
- `crates/gpt-oss-kv-model/src/lib.rs`
- `crates/gpt-oss-model-runner/src/architectures/gpt_oss.rs`
- `crates/gpt-oss-model-runner/src/attention/sliding_window.rs`
- `crates/gpt-oss-model-runner/src/gpu_runner.rs`

## Expected First Failure Seam

The likely first failure seam is terminology/schema translation plus decode
visibility and cache continuity:

- `sliding_attention` versus repo-local aliasing such as `local_attention`
- artifact-backed `layer_types` versus locally inferred attention scheduling
- real decode-state continuity for a sliding window of 128
- proving visibility/mask behavior without accidentally widening into sink or
  graph semantics

## What Must Not Change In This Lane

- No implementation should begin from this setup pass.
- Do not widen trusted support claims.
- Do not combine sliding work with learned sinks.
- Do not combine sliding work with graph replay.
- Do not broaden into MoE or shared harness cleanup.
- Do not touch Workstream A.

## Status

- Worktree and branch created.
- Launch note written.
- Implementation has now produced a bounded vertical slice for the exact first
  case without widening semantics.

## Sliding Attention — First Vertical Slice (Experimental)

- Exact case: single sequence, single layer, `sliding_attention`,
  `sliding_window = 128`, prefill `128`, decode `1` at absolute position `128`.
- Layers exercised: planned-reference, conformance, mock-boundary diagnosis,
  CUDA `gpu_layer`, CUDA `GpuModelRunner`, and CUDA `GpuWorker`.
- The `128 -> 129` boundary is now covered by bounded CUDA eager tests through
  worker-level runtime.
- Metadata continuity is covered for `positions`, `context_lens`,
  `slot_mapping`, and `block_tables`.
- This is bounded experimental evidence only, not general or trusted sliding
  support.
- Broader sliding-attention support remains deferred.
- Learned sinks remain the next increment on the same attention row.
- Ordering remains: sliding -> sink -> graph.
