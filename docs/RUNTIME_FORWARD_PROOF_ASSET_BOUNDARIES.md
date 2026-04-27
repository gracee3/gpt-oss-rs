# Runtime-Forward Proof Asset Boundaries

Date: 2026-04-27

This note records the current boundary between promotion-safe runtime-forward
assets and proof-only assets that should remain on `feature/runtime-forward`.

## Current Promotion Branches

`promotion/runtime-forward-final-token-oracle-parity` contains docs plus the
stdlib-only final-readout artifact validator. It does not import the proof
harness or runtime-forward proof-only candidate paths.

`promotion/runtime-forward-runtime-candidates` contains the RoPE half-split
pairing fix, a scoped layer0 pre-attention BF16 RMSNorm route, a CPU BF16
RMSNorm policy test, and optional Python tooling/`justfile` helpers. It does
not import the proof harness.

## Keep on feature/runtime-forward For Now

### `runtime_forward_layer0_qkv_bf16_candidate_status.rs` Proof Binary

Recommendation: do not promote as-is.

Reason: the proof binary is large and tightly coupled to runtime-forward-only
APIs, debug plumbing, `.live` artifacts, CUDA/model-runner proof paths, and
status modes.

Future extraction path: build a smaller integration-safe validator or
revalidation harness from scratch.

### oneDNN Q/K/V Projection Candidates

Recommendation: do not promote as runtime code.

Reason: these are CPU/oneDNN proof candidates. They are useful oracle
references, but they are not CUDA runtime fixes.

Future extraction path: use the findings to design a standalone CUDA
projection-policy strategy.

### Selected Expert Output Readout Correction

Recommendation: keep proof-only.

Reason: this is a diagnostic/readout staging correction and is not yet proven
as a production runtime behavior bug.

Future extraction path: promote only if the default runtime is shown to consume
the wrong staged buffer.

### Debug/Status Plumbing

Recommendation: do not promote broadly.

Reason: scalar capture, trace structs, debug buffers, and artifact-only paths
should not enter production runtime without explicit feature-gating and a
zero-overhead guarantee for default execution.

Future extraction path: design an explicit diagnostic feature/interface,
separate from default runtime behavior.

### Full `.live` PPP Raw Artifacts

Recommendation: do not commit raw full-value JSON artifacts to integration or
main.

Reason: raw PPP/full-value artifacts create large repository bloat.

Future extraction path: use manifest/checksum-only records, external or local
artifact storage, and small status summaries only when policy allows.

## What Integration Can Currently Validate

- Promoted RoPE/RMSNorm runtime candidates compile and test.
- The stdlib final-readout artifact validator runs without Python runtime
  dependencies.
- Artifact/digest validation can run against source runtime-forward artifacts
  when they are present locally.
- The CPU BF16 RMSNorm policy reference test documents the promoted scoped
  RMSNorm math policy.

## What Integration Cannot Yet Revalidate Natively

- Full final-token oracle proof replay.
- oneDNN Q/K/V proof candidate paths.
- Selected expert readout proof path.
- Debug/status bundle comparisons.
- Full PPP raw tensor comparisons.

This is intentional. Keeping these assets out of integration prevents
proof-only paths, debug plumbing, and large local artifacts from contaminating
production runtime or repository size.

## Recommended Next Work

- Wait for review visibility on the current runtime candidate branch.
- Prepare a standalone CUDA projection-policy design doc before any
  cuBLAS/pedantic or projection helper extraction.
- Keep full proof replay on `feature/runtime-forward` until an
  integration-safe revalidation harness is deliberately designed.
