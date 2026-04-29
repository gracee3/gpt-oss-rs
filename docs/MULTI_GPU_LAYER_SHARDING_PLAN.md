# Multi-GPU Layer Sharding Plan

## Purpose

Design and prototype two-GPU layer sharding across GPU0/GPU1 without changing
per-layer math. This lane targets layer placement/device-map, per-device CUDA
handles, activation transfer, and validation against existing seam/ladder artifacts.
This is not tensor parallelism.

## Base

- Base branch: `origin/projection/layer0-validation-runtime-handoff`
- Base commit: `2680a3fdf44e401dfd8368e9388907ae81bba226`
- Validation consumer: current seam/ladder framework

## First Design Target

- Device-map parser / placement metadata
- Per-device CUDA context/handles
- Layer tensor placement
- Activation transfer between devices
- KV cache per owning layer/device

## Preferred First Sharding

- Layers 0..11 on `cuda:0`
- Layers 12..23 on `cuda:1`

## Non-Goals

- No tensor parallelism.
- No collectives.
- No sharded QKV/MLP math.
- No production default routing change.
- No CUDA kernel changes in first slice.
