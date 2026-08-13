# Fresh CPU Oracle Campaign Closure

This record closes the fresh CPU validation campaign for candidate
`af6c0a2e41c58983f28d0ebca4981aca8c54602b`. It does not incorporate any
capture, count, comparison, gate, or claim from an earlier oracle lineage.

The campaign acceptance matrix is complete. Native correctness and the
bounded CPU service path passed. The paired service-instrumentation overhead
budget did not pass, so this record makes no positive overhead-regression
claim.

## Identity and evidence set

| Coordinate | Value |
| --- | --- |
| Candidate A | `af6c0a2e41c58983f28d0ebca4981aca8c54602b` |
| Image-input revision | `f1937a3f53f6704e8672fa8cecb6b95be8f53e82` |
| Oracle image | `ghcr.io/gracee3/gpt-oss-rs-cpu-oracle@sha256:ed7082ac67e76fe9cd8a2d4648c304ee3e1520b688972ad02fa295a396db861c` |
| Image config | `c910ee98a0bf28b61ffe89ac0b3cf78e36291cce5f6c57c4c95d7912c36b010e` |
| Software lock | `ea4bc4b7a650da14c094ee10028ac44fb39e4875d312ce04fa79be144913db12` |
| Official source | gpt-oss v0.0.9, `599476783c6f88508dab8577808b5ead5cbee8d2` |
| Model revision | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| llama.cpp revision | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` |
| Certification host key | `900b49f3b1ce625cb7daf5e8714051b4e14bddd92f44ea44a2d050137496f8e8` |
| Container policy | `cb1762bdc25f0f25fae20ddff7d2969fa4af8829da25a1250b6c59db41af4060` |
| E1 artifact-set SHA-256 | `b3634131064d7c682c01bc4d8ffbec3f2de86d651ee1c742586ab7ec3d1417ef` |

The final summary is stored outside Git at
`/home/emmy/gpt-oss-rs-artifacts/cpu-validation/af6c0a2e41c58983f28d0ebca4981aca8c54602b/publish/final-summary.json`.
It reports one accepted C3 cell, 28 official comparisons, seven llama.cpp
captures, two service cells, one accepted performance capture, and 43
terminal attempts. An independent closeout pass rehashed all 43 terminal
manifests and all 362 artifacts declared by them.

The exact published OCI archive is retained at
`/home/emmy/gpt-oss-rs-artifacts/oracle-images/f1937a3f53f6704e8672fa8cecb6b95be8f53e82/`
with archive SHA-256
`9ff1055bc3c353e7d219a8614ea7b66350e1d7eb7fdeb7f7bdc9b115dffccd06`.
The lock also pins the SBOM and provenance hashes.

## Outcomes

- Image and host qualification passed in both modes. Native PyTorch reported
  `AVX512`; the non-authoritative generic diagnostic reported `DEFAULT`.
  Each mode produced five repeat-identical BF16 operator fingerprints, CUDA
  was invisible, and both modes resolved to the same host key.
- Fresh C3-X-001 did not reproduce. Native and official captures agreed
  through the dense-boundary scan, so no prefix correction and no candidate B
  were required.
- All 28 native/official cells passed exact generated-token comparison: seven
  scenarios crossed with automatic, scalar, AVX2, and AVX-512/VNNI dispatch.
  Normal PyTorch CPU dispatch was the sole official authority.
- Seven fresh pinned llama.cpp captures completed in one CPU-only session with
  physical `ubatch=1`, prompt-cache reuse disabled, and newly built server
  SHA-256
  `2001e9bf50fb8d7aff75ae817cd1a51b49cf957a3fada8e4445eb0c8584532f6`.
  In this pinned server revision the legacy top-level `tokens` response field
  was empty; all eight generated token IDs per scenario remain present in the
  retained raw `completion_probabilities` records. These captures are
  advisory and did not change official comparison status.
- The complete locked model-free workspace lifecycle/HTTP suite passed.
- The bounded 20B CPU service session passed readiness and lifecycle probes,
  returned HTTP 200 for a one-token chat completion, and recorded 1.53 seconds
  to readiness plus 181.69 seconds for the bounded request.
- A fresh full-model performance capture for `harmony_63` recorded 24.247
  seconds for prompt processing and 3.278 seconds for eight generated tokens
  under automatic dispatch.

## Performance limitation

The paired model-free service-instrumentation gate was run twice after all
correctness and service gates passed. Both attempts met the throughput budget
but exceeded the strict p99 latency budget:

| Attempt | Throughput regression | p99 regression | Budget result |
| --- | ---: | ---: | --- |
| Initial | 0.072% | 2.741% | fail: p99 must be below 2% |
| Recheck | 0.183% | 4.718% | fail: p99 must be below 2% |

Both failed attempts are retained as terminal `invalid` campaign evidence and
are included in E1. No retry was discarded, relabeled, or used to make a pass
claim. Follow-up tuning or threshold work requires a separately authorized
candidate and fresh performance evidence; it does not invalidate the exact
correctness or bounded-service results above.

## Closeout verification

The candidate passed the locked release workspace build, locked workspace
tests, comparator and oracle negative tests, formatting, affected-crate
warnings-denied Clippy, scalar/AVX2/AVX-512 kernel lanes, portable AMX feature
compilation and 41-test emulation suite, Markdown link validation, OCI
archive/lock verification, native-plus-generic image probes, and the final
[CPU workflow](https://github.com/gracee3/gpt-oss-rs/actions/runs/31567029998).
The exact image publication, probes, SBOM, provenance, pushed-digest export,
and imported-manifest check passed in the
[CPU oracle image workflow](https://github.com/gracee3/gpt-oss-rs/actions/runs/31565668132).

All oracle, C3, and comparison results produced before this candidate are
retired historical records. They are non-authoritative and must remain absent
from future counts, comparisons, gates, and claims.
