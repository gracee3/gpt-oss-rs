# Borrowed Concepts and Design Provenance

- Status: living design-provenance ledger
- Started: 2026-08-11
- Scope: algorithms, architecture, lifecycle, testing, and research concepts

This document records ideas that influenced `gpt-oss-rs` even when no source
code was copied. Its goals are to preserve intellectual provenance, make
design decisions explainable, distinguish inspiration from adaptation, and
identify assumptions that should be revisited when upstreams change.

This is not a substitute for copyright notices. Code-level semantic audits and
adaptations are pinned in [`UPSTREAM_PROVENANCE.md`](UPSTREAM_PROVENANCE.md),
with required licenses in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)
and `LICENSES/`.

## Vocabulary

- **SEMANTIC CROSS-CHECK**: upstream behavior helped establish the expected
  numerical or model contract; local code is independently organized.
- **ADAPTED IMPLEMENTATION**: local implementation follows sufficiently
  specific upstream algorithmic or code structure to require nearby source
  attribution and license review.
- **ADOPTED CONCEPT**: a general architecture, lifecycle, or dataflow idea was
  intentionally incorporated, without translating an implementation.
- **RESEARCH REFERENCE**: the source informed alternatives or risks but did not
  determine the current design.
- **CANDIDATE CONCEPT**: worth retaining for future work; not implemented.

Do not use “borrowed” to erase the distinction between these categories.

## Current implementation ledger

| Source | Relationship | Concept retained | Local expression / record |
| --- | --- | --- | --- |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) at `8010b6a0578e416120b590ed72fd46ed5f24ee85` | Semantic cross-check and focused adapted behavior | GPT-OSS configuration, MXFP4 nibble/E2M1/E8M0 meaning, YaRN rotary behavior, attention sinks, selected-expert softmax, MoE routing, clamped SwiGLU | `crates/gpt-oss-cpu-kernels/src/lib.rs`, `crates/gpt-oss-model-runner/src/cpu_runner.rs`, and `UPSTREAM_PROVENANCE.md`; nearby comments identify focused ports/cross-checks |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) at audited revisions recorded in provenance/research docs | Adapted implementation and adopted concepts | Symmetric Q8 activation blocks, FP4 lookup/dot organization, x86 dispatch, model/context separation, stable server-slot batch mapping | CPU kernel and repack modules; `UPSTREAM_PROVENANCE.md`; research steps 1-4 |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) at `26ceed9d4091a1696cf50e2ed87e5767d5811d81` | Adapted implementation concept | Row-interleaved eight-output packing and multi-row x86 GEMV organization | `InterleavedSplitX8V2`, AVX2/AVX-512 x8 work, nearby source notes, provenance notice |
| [OpenAI gpt-oss](https://github.com/openai/gpt-oss) at `7b583341fe16729127f6d5b94a7b09ccae97e1a1` | Semantic cross-check | Canonical model shapes, MXFP4/MoE behavior, route/gather/scatter expectations | Scalar/model conformance and CPU runtime research corpus |
| [vLLM](https://github.com/vllm-project/vllm) at `52be12cfac0c5a18ba906814b2d2bcadb40a9c4b` | Adopted concept | Represent scheduled work as token catch-up under a budget; separate scheduling/reservation from output/progress update | `crates/gpt-oss-engine/src/cpu_scheduler.rs`, `crates/gpt-oss-engine/src/cpu_batch_engine.rs`, and `docs/cpu-runtime-research/04-cpu-batching-scheduling.md` |
| [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) at `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` | Adopted concept | Bound prompt chunks while preserving decode service and starvation protection | CPU scheduler policy and step-4 research record |
| [MegaBlocks](https://github.com/databricks/megablocks) at `952db33d6eac334d22c61e47a0d5d41446298784` | Adopted concept | Express MoE work as route/group-by-expert/compute/unroute while restoring stable source order | Matrix/prefill and scheduling design records; repository-specific stable reduction remains local |
| [oneDNN](https://github.com/uxlfoundation/oneDNN) at `7a6406900252f010553dda6eca442610fbedc825` | Adopted concept and research reference | Explicit packed operands, caller-visible scratch, primitive/kernel separation, runtime ISA selection, and scoped AMX hardware context | `Mxfp4MatmulProblem`, scratch contracts, AMX lifecycle, research steps 2 and 5 |
| ORCA and PagedAttention papers | Adopted concept / research reference | Iteration-level scheduling, selective batching, explicit request/KV ownership | Canonical CPU sequence scheduler and transactional batch execution |

The ledger describes influence, not identity. For example, the local scheduler
uses repository-specific reserve/execute/commit transactions and stable Rust
ownership; it is not a vLLM scheduler port. The local AMX prototype preserves
MXFP4 per-block scaling and uses its own Rust contracts; it is not oneDNN
BRGeMM integration.

## Repository-native synthesis

Several current designs combine external observations with constraints unique
to this project:

- `InterleavedSplitX8V2` is versioned around the official GPT-OSS MXFP4 model,
  mapped SafeTensors, local Rust APIs, and measured host constraints.
- `Mxfp4MatmulProblem` separates kernel data from sequence/scheduler metadata
  and keeps allocation outside kernels.
- a scheduled CPU batch is transactional: model/KV/RNG/output progress is
  committed only after execution and sampling succeed.
- AMX portable emulation, capability/permission state, and the native shim are
  deliberately separated because the available development host lacks AMX.
- derived repack caches are disposable and versioned; canonical checkpoints
  remain unchanged.

Record these as local syntheses unless a future audit finds a closer source
relationship.

## Candidate concepts not yet adopted

| Source | Candidate concept | Why it may help | Adoption gate |
| --- | --- | --- | --- |
| [`cl3`](https://github.com/kenba/cl3) at `9e2cdd8f34f09abfe49a8c2718ac58f1f762ae61` and [`opencl3`](https://github.com/kenba/opencl3) at `072410552fecfc1e3f5395856735cb8684501f74` | Functional and RAII Rust layers over the OpenCL C ABI, typed resource ownership, dynamic extension loading, and structured error propagation | Provide two Rust reference points for a deliberately smaller Iris Xe host surface | Source-map their ownership and asynchronous lifetime choices before selecting or independently expressing any host API; cloning them does not select a dependency |
| [oneAPI-rs](https://github.com/oneapi-src/oneapi-rs) at `1581663fdd0fd73e79df2900a2576d6cca8ff2a1` | RAII native handles, typed host/shared/device memory, event-bearing async ownership, typed kernel arguments, narrow unsafe launch boundary | Provides a Rust-shaped ownership model for a future Intel-GPU host interface | Implement only after a bounded integrated-GPU experiment establishes a real need; do not import SYCL/DPC++ merely for the design |
| [Level Zero](https://github.com/oneapi-src/level-zero) at `1ca51c950d97f34d9d271615af8d797836fe6974` | Explicit discovery/context/memory/module/kernel/queue/event lifecycle; optional validation/tracing; capability-first dispatch | Maps cleanly to a small forced Intel-GPU backend | Require installed loader/driver evidence, reproducible SPIR-V, one scalar-equivalent operation, and an OpenCL comparison where both paths remain viable |
| [rust-gpu](https://github.com/Rust-GPU/rust-gpu) | Rust-authored SPIR-V kernel path | Could preserve the project's Rust-first educational approach for later GPU work | Toolchain stability, artifact reproducibility, supported operations, and maintenance must be demonstrated separately |
| Candle, RTen, and Burn | Reusable Rust tensor/kernel interfaces and test practices | Possible homes for independently useful CPU results | Audit only against a concrete primitive; framework compatibility is not itself a project goal |

See [`LEVEL_ZERO_AND_ONEAPI_RS.md`](LEVEL_ZERO_AND_ONEAPI_RS.md) for the
detailed boundary between a small Rust host wrapper and the required system
driver/compiler/kernel toolchain.

## Reference-only observations

Some sources are valuable precisely because we did not adopt their whole
approach:

- oneDNN demonstrates mature primitive breadth, but `gpt-oss-rs` does not want
  a general primitive-descriptor runtime or maximum ISA/framework coverage.
- vLLM demonstrates production scheduler pressures, but its Python/C++/CUDA
  architecture and paged GPU memory manager are not the CPU runtime model.
- llama.cpp demonstrates extensive cross-platform kernels and graphs, but
  this project keeps a narrower GPT-OSS/Rust contract.
- SYCL provides portability across heterogeneous devices, but its C++ compiler
  and runtime are not justified by the present CPU-first target.
- Level Zero exposes broad low-level control, but most of its API is irrelevant
  to one experimental matrix operation.

Recording rejected breadth is useful: it explains why a studied project did
not become a dependency or architectural template.

## How to add an entry

For each material influence, record:

1. canonical project/source URL;
2. exact revision or specification version and access date;
3. source license;
4. exact path/symbol or document section inspected;
5. category from the vocabulary above;
6. concept or behavior retained;
7. local file/type/test where it appears;
8. meaningful differences from the source;
9. whether code-level notice or license text is required;
10. validation that protects the retained contract.

If source is copied, closely translated, or structurally adapted, update the
nearby code comment, `UPSTREAM_PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, and
`LICENSES/` in the same change. Do not hide code adaptation in this conceptual
ledger.

Potential destinations for independently useful results are tracked in
[`UPSTREAM_CONTRIBUTION_DISCOVERY.md`](UPSTREAM_CONTRIBUTION_DISCOVERY.md).
