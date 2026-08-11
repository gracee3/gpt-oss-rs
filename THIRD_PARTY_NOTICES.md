# Third-Party Notices

`gpt-oss-rs` is an independent Apache-2.0 implementation. Focused CPU
algorithms and semantic behavior are informed by the following MIT-licensed
projects; none is linked, embedded, or wrapped as a runtime.

## mistral.rs

- Upstream: <https://github.com/EricLBuehler/mistral.rs>
- Audited revision: `8010b6a0578e416120b590ed72fd46ed5f24ee85`
- Use: GPT-OSS MXFP4 decoding, routing, attention-sink, YaRN, and model-config
  behavior are used as semantic cross-checks.
- License: [`LICENSES/MIT-mistral.rs.txt`](LICENSES/MIT-mistral.rs.txt)

## llama.cpp

- Upstream: <https://github.com/ggml-org/llama.cpp>
- Audited revision: `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`
- Use: Q8 activation quantization, interleaved expert storage, and x86 SIMD
  organization inform focused native Rust implementations.
- License: [`LICENSES/MIT-llama.cpp.txt`](LICENSES/MIT-llama.cpp.txt)

## ik_llama.cpp

- Upstream: <https://github.com/ikawrakow/ik_llama.cpp>
- Audited revision: `26ceed9d4091a1696cf50e2ed87e5767d5811d81`
- Use: row-interleaved quant packing and x86 multi-row GEMV organization
  inform the independent Rust x8 packing and kernel implementation.
- License: [`LICENSES/MIT-ik_llama.cpp.txt`](LICENSES/MIT-ik_llama.cpp.txt)

Every adapted function carries a nearby source note. Literal extractions, if
introduced, must retain the applicable MIT notice in the source file.
