# CPU Backend Upstream Provenance

This document pins the source revisions reviewed for the native CPU backend.
The implementation remains repository-native Rust and exposes no upstream
runtime dependency.

| Upstream | Pinned revision | Reviewed concepts |
| --- | --- | --- |
| mistral.rs | `8010b6a0578e416120b590ed72fd46ed5f24ee85` | GPT-OSS configuration, MXFP4 scalar meaning, YaRN/attention sinks, MoE routing and gating |
| llama.cpp | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` | Q8 activation blocks, MXFP4 dot-product organization, expert interleaving, x86 dispatch |
| ik_llama.cpp | `26ceed9d4091a1696cf50e2ed87e5767d5811d81` | Row-interleaved quant packing and multi-row x86 GEMV organization |

The source audit is intentionally narrow. Algorithms are restated against the
project's safe Rust APIs, scalar oracle, tensor shapes, and test fixtures.
Whole files or subsystems are not imported. See `THIRD_PARTY_NOTICES.md` for
license texts and attribution requirements.

## CPU kernel audit map

- `mistralrs-quant/src/mxfp4/mod.rs` (`mxfp4_dequantize`): official
  SafeTensors adjacent-nibble order, E2M1 values, and E8M0 scale semantics.
- `ggml/src/ggml-quants.c` (`quantize_row_q8_0_ref`): symmetric Q8 activation
  blocks and nearest-integer clamping behavior.
- `ggml/src/ggml-cpu/arch/x86/quants.c` (`quantize_row_q8_0`,
  `ggml_vec_dot_mxfp4_q8`): x86 vector organization for activation reduction,
  FP4 table lookup, and exact integer accumulation. The repository's AVX-512
  VNNI implementation uses its own signed-to-unsigned correction because VNNI
  byte dot products have asymmetric operand signedness.
- `ggml/src/iqk/` row-interleaved quant kernels: multi-row packing and SIMD
  load organization. The repository's `InterleavedSplitX8V2` layout and AVX2
  kernel are independent Rust implementations specialized for official MXFP4.
- `mistralrs-core/src/models/gpt_oss.rs` (`Attention`, `GptOssMoE`,
  `gptoss_swiglu`): alternating sliding/full attention, selected-expert
  softmax, interleaved gate/up ordering, and clamped GPT-OSS SwiGLU.
- `mistralrs-core/src/layers.rs` (`GptOssRotaryEmbedding`): YaRN correction
  range, frequency interpolation, GPT-NeoX rotation, and attention scaling.
- `mistralrs-core/src/attention/backends/sinks.rs` (`sinks_attn_cpu`): learned
  per-head sinks as extra softmax denominator logits with no value vector.

When either reference is refreshed, update the pinned revision, list the exact
reviewed files/functions here, and rerun scalar/SIMD parity and model-level
conformance before accepting changed behavior.
