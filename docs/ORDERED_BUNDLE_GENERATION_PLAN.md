# Ordered Bundle Generation Plan

## Purpose

Generate ordered attention/MLP oracle bundles for layers 2..23 with low-memory capture.
This lane produces oracle/reference artifacts consumed by validation-runtime-handoff and
does not change production runtime.

## Base

- Base branch: `origin/projection/layer0-validation-runtime-handoff`
- Base commit: `2680a3fdf44e401dfd8368e9388907ae81bba226`
- Consumer branch: `projection/layer0-validation-runtime-handoff`

## Current Blocker

True ordered seam validation is blocked because layer2+ ordered attention/MLP bundles
are missing.

The existing coarse layer2-to-final bundle is useful for coarse guards only. It
contains per-layer final-token input, attention norm, attention residual, MLP norm,
and final output boundaries for layers 2..23, but it does not include ordered
attention/MLP seams such as Q/K/V, raw-QK, attention probabilities, router/top-k,
selected outputs, or weighted sums.

## Target Artifact Contracts

- `layerN` attention ordered bundle
- `layerN` MLP ordered bundle
- Bundle manifest
- Schema checker

## Low-Memory Constraints

- Avoid full-model BF16 dequant on one 24 GB GPU.
- Prefer layer-scoped, CPU/offload, or two-GPU-aware generation if needed.

## Non-Goals

- No Rust runtime production change.
- No raw `.live` or `/tmp` artifacts committed.
- No Torch dependency in Rust.
- No final-logit or 4097 claim.
