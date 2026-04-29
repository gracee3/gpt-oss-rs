# Tensor Parallel Design Notes

## Purpose

Park a docs-only design lane for future true tensor parallelism. This branch is
not an active implementation lane.

## Likely Future Concepts

- `tp_rank` / `tp_world_size`
- Device mesh
- NCCL or collective layer
- Column/row parallel linear
- Attention head partition
- Expert/vocab partition
- All-reduce/all-gather points
- TP-aware validation artifacts

## Non-Goals

- No implementation in this branch yet.
- No production behavior changes.
