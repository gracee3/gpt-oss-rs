# CPU Runtime Research Corpus

- Status: complete for research/pre-planning; ready for per-item implementation planning
- Research date: 2026-08-11
- Repository baseline: `3600fa45d5adeca6e183488c50d5359bf7e3177a`
- Scope: CPU runtime feature design only; no feature implementation or tuning

This directory is the durable source-grounded design record for the five CPU
runtime workstreams listed in
[`CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md`](../CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md).
It records what the current repository does, what mature implementations do,
the design we intend to take into implementation planning, rejected
alternatives, risks, and focused correctness gates.

The documents are not benchmark reports or certification claims. Words such as
"likely faster" are hypotheses unless accompanied by a recorded measurement.
All new optimized paths remain forced and experimental until the later
certification and tuning phase.

## Workstream documents

1. [`01-avx512-vnni-x8.md`](01-avx512-vnni-x8.md): genuine AVX-512/VNNI
   eight-output MXFP4 GEMV.
2. [`02-gemm-prefill.md`](02-gemm-prefill.md): scalar and SIMD multi-row MXFP4
   matrix contract plus true layer-major prefill.
3. [`03-model-sequence-state.md`](03-model-sequence-state.md): immutable model,
   per-sequence state, execution scratch, and transactional KV updates.
4. [`04-cpu-batching-scheduling.md`](04-cpu-batching-scheduling.md): canonical
   sequence ownership, schedule/execute/commit, cancellation, fairness, and
   mixed CPU batches.
5. [`05-amx-prototype.md`](05-amx-prototype.md): AMX-INT8 mapping, panel layout,
   Linux/Rust lifecycle, portable tests, and the integration seam.

## Pinned source registry

Source checkouts live outside this repository. They are research inputs, not
vendored dependencies, submodules, or code donors. Observations are pinned to
the revisions below; changing a checkout does not silently update an evidence
card.

| ID | Source and local path | Revision | License | Primary use |
| --- | --- | --- | --- | --- |
| SRC-LLAMA | llama.cpp, `/home/emmy/src/llama.cpp` | `2468576f241235452013308597e6de1b78866996` | MIT | x8 MXFP4 repacking/GEMV/GEMM, model/context split, batches, server slots |
| SRC-MISTRAL | mistral.rs, `/home/emmy/src/mistral.rs` | `8010b6a0578e416120b590ed72fd46ed5f24ee85` | MIT | Rust sequence ownership, schedulers, ragged completion, MoE interfaces |
| SRC-IK | ik_llama.cpp, `/home/emmy/src/ik_llama.cpp` | `26ceed9d4091a1696cf50e2ed87e5767d5811d81` | MIT | MXFP4 r8 layout and AVX2/AVX-512 multi-row kernels |
| SRC-OPENAI | OpenAI gpt-oss, `/home/emmy/src/cpu-runtime-research/openai-gpt-oss` | `7b583341fe16729127f6d5b94a7b09ccae97e1a1` | Apache-2.0 | official numerical semantics and MoE route/gather/scatter behavior |
| SRC-VLLM | vLLM, `/home/emmy/src/cpu-runtime-research/vllm` | `52be12cfac0c5a18ba906814b2d2bcadb40a9c4b` | Apache-2.0 | token-catch-up scheduling and explicit output commit |
| SRC-ONEDNN | oneDNN, `/home/emmy/src/cpu-runtime-research/onednn` | `7a6406900252f010553dda6eca442610fbedc825` | Apache-2.0 | caller-owned packing/scratch and AMX hardware-context lifecycle |
| SRC-MEGABLOCKS | MegaBlocks, `/home/emmy/src/cpu-runtime-research/megablocks` | `952db33d6eac334d22c61e47a0d5d41446298784` | Apache-2.0 | grouped expert routing and route/unroute concepts |
| SRC-SARATHI | Sarathi-Serve, `/home/emmy/src/cpu-runtime-research/sarathi-serve` | `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` | Apache-2.0 | bounded chunked prefill and decode-maximal scheduling |

The exact licenses were read from each pinned checkout. Existing local source
trees also contain `AGENTS.md` instructions; this research only reads them and
does not prepare upstream contributions.

## Primary specifications and papers

- Open Compute Project, [Microscaling Formats (MX) Specification
  v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):
  E2M1 and E8M0 encodings, block scale semantics, and special values.
- Linux kernel, [Using XSTATE features in user-space
  applications](https://docs.kernel.org/next/x86/xstate.html): dynamic AMX
  XSTATE support and permission, signal-stack checks, fork/exec behavior, and
  first-use allocation.
- Intel, [Advanced Matrix Extensions intrinsic
  sample](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html):
  palette-one tile dimensions, INT8 dot products, permission request, tile
  configuration, and release.
- Rust, [`x86_amx_intrinsics`](https://doc.rust-lang.org/beta/unstable-book/language-features/x86-amx-intrinsics.html)
  and [the current `stdarch` AMX source](https://doc.rust-lang.org/src/core/stdarch/crates/core_arch/src/x86_64/amx.rs.html):
  AMX target features and intrinsics remain unstable; AVX-512 target features
  are stable on this repository's Rust 1.97.1 toolchain.
- Yu et al., [ORCA](https://www.usenix.org/system/files/osdi22-yu.pdf):
  iteration-level scheduling and selective batching.
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180): request-level
  KV block management and continuous batching.
- Agrawal et al., [Sarathi-Serve](https://www.usenix.org/system/files/osdi24_full_proceedings.pdf):
  chunked prefills, decode-maximal batches, and bounded per-iteration work.
- Gale et al., [MegaBlocks](https://arxiv.org/abs/2211.15841): dynamic MoE
  routing expressed as grouped/block-sparse work. This is training/GPU
  evidence; only the route/group/unroute abstraction transfers to this CPU
  design.
- oneDNN, [BRGeMM ukernel
  documentation](https://uxlfoundation.github.io/oneDNN/v3.6/group_dnnl_api_ukernel_brgemm.html):
  explicit B packing, scratch size, accumulator, and hardware-context calls.

## Common conclusions

The research produced the following cross-workstream decisions:

- Keep `InterleavedSplitX8V2` as the first persistent MXFP4 layout for AVX-512
  and GEMM. It is already model-size neutral and exposes the same eight-row
  structural grouping used by the pinned llama.cpp and ik_llama.cpp sources.
- Separate the engine/model batch descriptor from the matrix microkernel
  problem. Sequence IDs, positions, causal metadata, and logits policy do not
  belong in the MXFP4 kernel API.
- Make persistent weights, transient activation packing, output buffers, and
  scratch distinct types with explicit owners. Kernels must not allocate or
  silently change layout.
- Split immutable model state from mutable per-sequence model and generation
  state before enabling multiple active CPU requests.
- Use a single execution owner initially. The scheduler returns immutable work
  descriptors and commits progress only after model execution succeeds; core
  state does not require an `Arc<Mutex<_>>` graph.
- Treat a token step or scheduled batch as a transaction. A failed layer may
  not leave some layer caches advanced and others unchanged.
- Group MoE routes by expert for matrix work, but restore results and perform
  weighted reduction in stable source-row/top-k order.
- Make topology visible but policy-free. CPU affinity, NUMA memory placement,
  and crossover choices require later representative-host measurements.
- Build the first AMX experiment around signed INT8 dot products and the common
  GEMM contract. BF16 expansion and persistent AMX caches remain alternatives.

## Research experiments and environment

### ENV-E001 — available development host

- **EXPERIMENT:** `lscpu` on 2026-08-11 reports an Intel Core i7-1185G7 with
  four cores/eight logical CPUs, one NUMA node, AVX2, AVX-512F/BW/VL/VNNI, and
  no AMX flags.
- **Implication:** AVX-512 code can receive local focused execution coverage.
  AMX must use compile, packing, emulation, and forced-failure coverage until a
  suitable host is available.

### NUM-E001 — scale-byte survey of the pinned 20B checkpoint

- **EXPERIMENT:** a read-only SafeTensors header and byte-range scan covered 48
  scale tensors and 597,196,800 scale bytes in
  `/data/models/openai/gpt-oss-20b`.
- **Command/method:** an inline `python3` scanner read each shard's little-endian
  header length and JSON metadata, selected names ending in `_scales`, sought
  to their `data_offsets`, and counted bytes in 8 MiB chunks. It printed
  `tensors=48 bytes=597196800 min=115 max=136 zero=0 ff=0`.
- **Result:** minimum scale byte 115, maximum 136, with no `0x00` or `0xff`.
- **Implication:** the repository's current `0x00` E8M0 behavior does not affect
  this checkpoint, but the scalar and future matrix contracts still need a
  specification-correct decision for synthetic and future inputs.
- **Retention:** no model bytes or scan output are committed.

## Deferred work

The following are intentionally outside this research gate:

- implementation of any proposed type or kernel;
- automatic dispatch changes for AVX-512, GEMM, or AMX;
- microbenchmarks, percentage thresholds, tile/crossover tuning, and NUMA
  placement policy;
- the 28-run oracle matrix, long-generation certification, fresh advisory
  captures, and complete API permutations;
- AMX hardware execution and trusted-mode eligibility.

Focused scalar equivalence, memory safety, rollback, cache integrity, and API
functionality remain mandatory in the later implementation work.
