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

## Next-phase pre-research reference register

These entries record why a pinned source may be inspected during a later
approved charter. They do not claim adoption or implementation derivation.

| Source | Relationship | Candidate lesson | Boundary / local question |
| --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) at `2468576f241235452013308597e6de1b78866996` | Research reference | One canonical inference owner, typed task/result queues, HTTP-side response state, bounded delivery/replay, and lifecycle tests | Preserve reserve/execute/commit and ordered API semantics; do not import router breadth, resumable generation, or the graph runtime |
| [TGI](https://github.com/huggingface/text-generation-inference) at `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` | Research reference | Separate validation/admission, concurrent/token limits, overload, streaming, readiness, telemetry, and stable failures | Upstream declares maintenance mode; use as historical service evidence, not as a deployment or compatibility authority |
| [vLLM](https://github.com/vllm-project/vllm) at `52be12cfac0c5a18ba906814b2d2bcadb40a9c4b` | Research reference and candidate concept | Per-iteration prefill/decode work descriptors, lifecycle timestamps, token-source accounting, and cache-lifecycle observations | Retain only cheap facts relevant to the CPU owner; exclude distributed/GPU memory architecture and unbounded metric breadth |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) at `8010b6a0578e416120b590ed72fd46ed5f24ee85` | Research reference | RAII in-flight accounting, bounded-cardinality HTTP metrics, request IDs in logs, and production-path CPU benchmark discipline | Do not make general model/runtime compatibility a goal; request identity must not become a metric label |
| [oneDNN](https://github.com/uxlfoundation/oneDNN) at `7a6406900252f010553dda6eca442610fbedc825` | Research reference and candidate concept | Dispatch selection/rejection evidence, canonical repro descriptors, separate correctness/bitwise/performance modes, cold-weight measurements, and insufficiently trusted validation status | Borrow evidence discipline, not a dependency, general primitive runtime, or automatic policy |
| [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) at `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` | Research reference | Separate queue, execute, stall, TTFT/decode, and batch-composition measurements plus seeded workload descriptions | GPU simulator and utilization conclusions are outside the CPU runtime boundary |
| [OpenAI gpt-oss](https://github.com/openai/gpt-oss) at `7b583341fe16729127f6d5b94a7b09ccae97e1a1` | Semantic cross-check and research reference | Stable operator phase names and readable RMSNorm, RoPE, attention-sink, routing, and SwiGLU precision boundaries | Official semantics remain authoritative; GPU execution organization is not a local runtime template |
| [MegaBlocks](https://github.com/databricks/megablocks) at `952db33d6eac334d22c61e47a0d5d41446298784` | Research reference | Expert-bucket distributions and route/group/compute/unroute stage accounting | Uniform routing is only a controlled stress shape, not evidence for real GPT-OSS routing |

No code was copied or adapted for this register. Exact inspected paths,
retained concepts, local expressions, differences, and validation obligations
must be added if a later research conclusion or implementation relies on one
of these sources.

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
| [oneAPI-rs](https://github.com/oneapi-src/oneapi-rs) at `1581663fdd0fd73e79df2900a2576d6cca8ff2a1` | RAII native handles, typed host/shared/device memory, event-bearing async ownership, typed kernel arguments, narrow unsafe launch boundary | Provides a Rust-shaped ownership model for a future Intel-GPU host interface | Implement only after a bounded integrated-GPU experiment establishes a real need; do not import SYCL/DPC++ merely for the design |
| [Level Zero](https://github.com/oneapi-src/level-zero) at `1ca51c950d97f34d9d271615af8d797836fe6974` | Explicit discovery/context/memory/module/kernel/queue/event lifecycle; optional validation/tracing; capability-first dispatch | Maps cleanly to a small forced Intel-GPU backend | Require installed loader/driver evidence, reproducible SPIR-V, one scalar-equivalent operation, and comparison with Vulkan |
| [rust-gpu](https://github.com/Rust-GPU/rust-gpu) | Rust-authored SPIR-V kernel path | Could preserve the project's Rust-first educational approach for later GPU work | Toolchain stability, artifact reproducibility, supported operations, and maintenance must be demonstrated separately |
| Candle, RTen, and Burn | Reusable Rust tensor/kernel interfaces and test practices | Possible homes for independently useful CPU results | Audit only against a concrete primitive; framework compatibility is not itself a project goal |
| llama.cpp, TGI, and local transactional ownership | Byte-bounded result delivery separated from canonical sequence authority | Prevents a slow or disconnected consumer from stalling or mutating committed inference progress | C1 must define commit, coalescing, abandonment, failure, and cleanup semantics before implementation |
| vLLM, Sarathi-Serve, mistral.rs, and oneDNN evidence patterns | One effective-runtime snapshot plus distinct production metrics, diagnostic traces, and offline run manifests | Makes dispatch, work shape, latency, memory, correctness, and negative results reproducible without a telemetry platform | E1 must set overhead, cardinality, timestamp, redaction, and source-role rules |
| Existing CPU memory ownership plus mature memory-manager references | Resource reservations with grant, expansion, refund, release, and stable rejection reasons | Can bound prompts, KV, scratch, delivery, and stored output while retaining contiguous KV initially | C1/C2 ownership and accounting research must precede policy or paged storage |
| Official GPT-OSS, oneDNN, MegaBlocks, and existing typed CPU problems | Separate typed MoE, dense-BF16, and attention contracts with explicit scratch/threading/eligibility | Preserves semantic baselines and allows measured backend experiments without one general operator framework | Three bounded C4 charters must establish real consumers and correctness strategies before implementation |
| ggml/llama.cpp graphs and oneDNN Graph | A limited inspectable execution-plan representation | Might centralize scratch liveness, operation tracing, or dispatch evidence | Require at least two real consumers or a measured maintenance/debugging problem; reject a general tensor/autodiff framework |

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
- TGI demonstrates explicit serving limits and stable failure surfaces, but is
  now maintenance-mode and its sharded deployment/compatibility topology is
  outside the project envelope.
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
