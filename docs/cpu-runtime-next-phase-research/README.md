# CPU Runtime Next-Phase Research Corpus

- Status: research closeout pending
- Research date: 2026-08-11
- Repository source baseline: `a090bb0e81457e4302deb36d6e52a0847c14bfb0`
- Intake checkpoint: `9df99d24891aa95c2bd9aa39bab9d5a1fa4b1555`
- Scope: documentation and read-only research only

This directory contains the source-grounded research record for E1 and C1-C7.
Each synthesis states its own outcome using exactly one of: **planning-ready**,
**narrow experiment warranted**, **deferred**, **rejected**, or
**inconclusive**. Planning-ready means only that a later owner-authorized plan
could be written; it is not implementation authority.

## Baseline record

The worktree was clean immediately after the intake checkpoint. Research uses
Rust `1.97.1 (8bab26f4f 2026-07-14)`, Cargo `1.97.1
(c980f4866 2026-06-30)`, LLVM 22.1.6, and `Cargo.lock` SHA-256
`1cf8db2c63a1550ff92a666e1a88421d40da974ca04e9f0d525236acbb24dd65`.
The host runs Ubuntu 24.04 on Linux `7.0.0-28-generic` and exposes one NUMA
node, four physical/eight logical Intel i7-1185G7 CPUs, AVX2, AVX-512F/BW/VL,
and AVX-512 VNNI. It does not expose AMX CPUID flags. These are capability
facts, not representative-host or performance evidence.

The static 20B checkpoint inspected at `/data/models/openai/gpt-oss-20b` has
config SHA-256
`3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce`,
index SHA-256
`0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1`,
and 13,789,246,222 directory bytes. Static inspection did not hash all
model-scale data again and did not execute the model.

## Stable identifiers

- Sources use `NX-SRC-NNN`.
- Evidence uses `<track>-E-NNN`.
- Decisions use `<track>-D-NNN`.
- Open questions use `<track>-Q-NNN`.
- Proposed experiments use `<track>-X-NNN`.
- Raw retained artifacts use `NX-ART-NNN`.

IDs are never reused. A superseding item cites the old ID and receives a new
one. Evidence labels retain the vocabulary from
[`CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md`](../CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md):
CURRENT-REPO FACT, LOCAL-SOURCE OBSERVATION, PRIMARY-SOURCE FACT, EXPERIMENT,
INFERENCE, PROVISIONAL DECISION, and OPEN QUESTION.

An evidence card must identify the question, source kind, exact revision or
document version, path and symbol/section, access date, license/provenance,
observation, implication, limitations/conflicts, and confidence. Experiment
cards additionally require command, build, host, model/workload, repetitions,
raw artifact path and SHA-256, timer inclusions, and result status.

## Pinned source registry

All local checkouts were clean when inspected. Source access date is
2026-08-11. They are research inputs, not dependencies or code donors.

| ID | Source role, checkout, revision | License | Bounded use |
| --- | --- | --- | --- |
| NX-SRC-001 | Current repository, `/home/emmy/gpt-oss-rs`, source baseline `a090bb0e81457e4302deb36d6e52a0847c14bfb0` | repository license | Authoritative current behavior |
| NX-SRC-002 | llama.cpp research checkout, `/home/emmy/src/llama.cpp`, `2468576f241235452013308597e6de1b78866996` | MIT | Lifecycle, result ownership, CPU memory/graphs |
| NX-SRC-003 | TGI, `/home/emmy/src/cpu-runtime-research/text-generation-inference`, `b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed` | Apache-2.0 | Historical admission, overload, streaming, readiness |
| NX-SRC-004 | vLLM, `/home/emmy/src/cpu-runtime-research/vllm`, `52be12cfac0c5a18ba906814b2d2bcadb40a9c4b` | Apache-2.0 | Work descriptors, lifecycle timestamps, cache pressure |
| NX-SRC-005 | mistral.rs, `/home/emmy/src/mistral.rs`, `8010b6a0578e416120b590ed72fd46ed5f24ee85` | MIT | Rust lifecycle, HTTP metrics, request logging |
| NX-SRC-006 | oneDNN, `/home/emmy/src/cpu-runtime-research/onednn`, `7a6406900252f010553dda6eca442610fbedc825` | Apache-2.0 | Explicit problem/scratch/dispatch and AMX lifecycle |
| NX-SRC-007 | MegaBlocks, `/home/emmy/src/cpu-runtime-research/megablocks`, `952db33d6eac334d22c61e47a0d5d41446298784` | Apache-2.0 | Route/group/compute/unroute abstraction |
| NX-SRC-008 | Sarathi-Serve, `/home/emmy/src/cpu-runtime-research/sarathi-serve`, `96f9911790ecc00af12ee9fae47cb8fa9ba0d199` | Apache-2.0 | Chunked-work and offline timing vocabulary |
| NX-SRC-009 | OpenAI GPT-OSS research checkout, `/home/emmy/src/cpu-runtime-research/openai-gpt-oss`, `7b583341fe16729127f6d5b94a7b09ccae97e1a1` | Apache-2.0 | Readable operator semantics only |
| NX-SRC-010 | ik_llama.cpp, `/home/emmy/src/ik_llama.cpp`, `26ceed9d4091a1696cf50e2ed87e5767d5811d81` | MIT | CPU MXFP4 organization only |
| NX-SRC-011 | Official oracle fixture checkout, `/home/emmy/src/cpu-runtime-research/openai-gpt-oss-oracle-7802bf263`, `7802bf263f902efd4c7d18fcceff3ba72f941e80` | Apache-2.0 | Blocking fixture authority; never silently refreshed |
| NX-SRC-012 | llama.cpp oracle fixture checkout, `/home/emmy/src/cpu-runtime-research/llama.cpp-oracle-030ebb558`, `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` | MIT | Advisory fixture authority; distinct from NX-SRC-002 |

The llama.cpp, mistral.rs, vLLM, and oracle checkout `AGENTS.md` files were read
before source inspection. No upstream source was edited, built, committed, or
submitted. Each full charter uses at most four external source families and
stops after two viable alternatives or an inability to discriminate.

## Primary-source register

Primary facts use official specifications and documentation: Linux kernel
XSTATE and `/proc` accounting documentation; Intel AMX documentation; Rust
`std::time`, `stdarch`, and allocation documentation; Tokio bounded channel and
cancellation documentation; Axum body-limit and graceful-shutdown
documentation; Prometheus naming/cardinality guidance; the relevant API
protocol documentation; and original PagedAttention, ORCA, Sarathi-Serve,
MegaBlocks, and speculative-decoding papers. Individual syntheses cite only the
subset they use.

## Raw artifact policy and corpus gate

Raw captures stay outside Git. A citation requires an absolute path and
SHA-256. Commands and environment values are redacted before publication, but
the private raw record must remain recoverable.

Two historical artifacts are registered only for facts already in repository
documentation:

| ID | Absolute path and SHA-256 | Role and limitation |
| --- | --- | --- |
| NX-ART-001 | `/data/models/openai/gpt-oss-rs-cpu-work/results/harmony_122-auto-cold.json`, `c2b488cad713ef9299ff12e9543ec00d97716fb7dc8b52cd2b3ab98c6666d631` | Historical M1-M5 trace. It identifies fixture/source pins and one run, but lacks a complete E1 command/host/repetition record; advisory only for this phase. |
| NX-ART-002 | `/data/models/openai/gpt-oss-rs-cpu-work/results/final-concurrent-stream.sse`, `4b438db870c8ebd0d095ad3ee0ca575754432a30d3d17b2c88b1a5d4cd668c25` | Historical API smoke showing cumulative completion text; not a benchmark or protocol certification. |

The 2026-08-11 corpus-gate inspection found no new manifest or repetition set
in `/data/models/openai/gpt-oss-rs-cpu-work/results`. This is a negative result,
`E1-NEG-001: unavailable`, not a failed benchmark. C3 and C4 must not fill the
gap with estimates.

## Corpus and outcome table

| Track | Document | Outcome |
| --- | --- | --- |
| E1 | [`00-evidence-and-observability.md`](00-evidence-and-observability.md) | planning-ready |
| C1 | [`01-service-lifecycle-api.md`](01-service-lifecycle-api.md) | planning-ready |
| C2 | [`02-memory-reservations.md`](02-memory-reservations.md) | planning-ready |
| C3 | [`03-numerical-trust.md`](03-numerical-trust.md) | narrow experiment warranted |
| C4-A | [`04a-moe-orchestration.md`](04a-moe-orchestration.md) | deferred |
| C4-B | [`04b-dense-bf16.md`](04b-dense-bf16.md) | deferred |
| C4-C | [`04c-attention.md`](04c-attention.md) | deferred |
| C5 | [`05-amx-hardware.md`](05-amx-hardware.md) | deferred |
| C6 | [`06-long-horizon-seams.md`](06-long-horizon-seams.md) | planning-ready |
| C7 | [`07-maintenance-audit.md`](07-maintenance-audit.md) | planning-ready |

## Cross-track rules

1. E1 identifiers and manifests apply to every experiment and trust claim.
2. C1 owns request, canonical model state, and delivery lifecycles; C2 owns
   resource grants. Neither may infer the other's state from channel presence.
3. C3 trust is attached to effective configurations and covered fallback
   shapes, not source-level feature names.
4. C4 descriptors may expose work and ownership without choosing scheduling,
   paging, or automatic thresholds.
5. C5 portable evidence never substitutes for native AMX evidence.
6. C6 can pressure these seams but cannot generalize them for hypothetical
   features.
7. C7 maintenance slices cannot carry semantic behavior changes.
