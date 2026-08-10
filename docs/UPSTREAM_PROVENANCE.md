# CPU Backend Upstream Provenance

This document pins the source revisions reviewed for the native CPU backend.
The implementation remains repository-native Rust and exposes no upstream
runtime dependency.

| Upstream | Pinned revision | Reviewed concepts |
| --- | --- | --- |
| mistral.rs | `8010b6a0578e416120b590ed72fd46ed5f24ee85` | GPT-OSS configuration, MXFP4 scalar meaning, YaRN/attention sinks, MoE routing and gating |
| llama.cpp | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` | Q8 activation blocks, MXFP4 dot-product organization, expert interleaving, x86 dispatch |

The source audit is intentionally narrow. Algorithms are restated against the
project's safe Rust APIs, scalar oracle, tensor shapes, and test fixtures.
Whole files or subsystems are not imported. See `THIRD_PARTY_NOTICES.md` for
license texts and attribution requirements.

When either reference is refreshed, update the pinned revision, list the exact
reviewed files/functions here, and rerun scalar/SIMD parity and model-level
conformance before accepting changed behavior.
