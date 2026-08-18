# gpt-oss-rs v0.1.0 final research report

## Abstract

This report closes a CPU-first Rust investigation of OpenAI GPT-OSS. The work
implemented native GPT-OSS 20B execution, compact MXFP4 expert kernels and
layouts, transactional model and scheduling ownership, versioned evidence
surfaces, and bounded experimental service paths. It also studied—and retained
as archives—heterogeneous expert placement and two-GPU layer sharding that did
not reach their final execution claims. Version 0.1.0 publishes source,
methods, same-host measurements, attribution, negative results, and explicit
limitations as a research artifact rather than a production-readiness claim.

## Research questions

1. Can official GPT-OSS MXFP4 expert weights remain compact while scalar and
   x86 CPU kernels preserve one exact numerical contract?
2. Which ownership seams make prompt/decode execution, KV state, repack caches,
   cancellation, and scheduling inspectable and fail-closed?
3. Can benchmarks preserve identity, thermals, run order, exact tokens, and
   negative results well enough to support bounded optimization claims?
4. Which concepts from incomplete heterogeneous and multi-GPU research are
   reusable without presenting those archived runtimes as complete?

## Contributions

- A scalar MXFP4 contract plus AVX2 and AVX-512/VNNI implementations, explicit
  packed layouts, caller-visible scratch, and exact-bit test gates.
- Native SafeTensors mapping and versioned, atomic, source-hash-keyed repack
  caches that leave canonical checkpoints unchanged.
- Separate immutable model and mutable sequence state with transactional
  prepare/execute/commit behavior and layer-major multi-row prefill.
- Bounded CPU scheduling, lifecycle, evidence, redaction, and failure surfaces.
- A fixed workload corpus and publication driver with exact source/model/binary
  identity, physical-core pinning, thermal rejection, raw samples, paired
  bootstrap analysis, and tamper-evident artifact publication.
- Archived negative and incomplete work with a standalone layer-sharding
  retrospective and explicit research-ethics disclosure.

## Methods

### Implementation and semantic method

The official GPT-OSS implementation and checkpoint are semantic authorities.
Scalar code remains the local kernel oracle. Optimized kernels are evaluated
at exact model shapes and must match scalar output bits. Full-model greedy
tokens are checked against a pinned seven-scenario Harmony fixture and its
model, tokenizer, and oracle identities.

External projects were inspected at pinned revisions. The project records
semantic cross-checks, adapted implementation concepts, adopted general
concepts, and research-only references separately in the [CPU provenance
ledger](../UPSTREAM_PROVENANCE.md), [borrowed-concepts
ledger](../BORROWED_CONCEPTS.md), notices, and [archived HET source
ledger](https://github.com/gracee3/gpt-oss-rs/blob/7bb459361c68b00eed45f56a622c061bb4b135ff/docs/het/11-source-ledger.md).

### Experimental method

The final controlled experiment is specified in the [benchmark
protocol](BENCHMARK_PROTOCOL.md). In brief, it uses one eight-core Xeon host,
CPUs 0-7, eight threads, disabled GPU backends and prompt caching, warm page
cache, fresh processes, a 65 C admission gate, throttle rejection, rotated
order, one warmup, and five measured full-model trials per lane.

MXFP4 gate/up (`N=5760`, `K=2880`) and down (`N=2880`, `K=2880`) shapes use
`M=1,3,8`, residual-Q8, four paths, seven trials, five samples per trial, exact
scalar-bit equivalence, and 10,000 paired bootstrap iterations. The full-model
workload is the exact 63-token `harmony_63` prompt plus eight greedy tokens.

The GGUF comparison input is converted locally from the already-present
SafeTensors using pinned llama.cpp. No replacement GGUF is downloaded. Source
hashes, conversion command, dependency freeze, output hash, and the comparison
with the historical GGUF hash are retained in the evidence bundle.

## Results

### MXFP4 kernels

All 12 optimized candidate/shape comparisons were scalar-bit-exact and all
matrix trials were thermally clean. The paired 10,000-iteration analysis
selected five bounded regions:

- AVX2 for the down projection (`N=2880`, `K=2880`) at `M=1` and `M=3`;
- AVX-512/VNNI for the down projection at `M=8`; and
- AVX2 for gate/up (`N=5760`, `K=2880`) at `M=1` and `M=3`.

Gate/up at `M=8` retained scalar fallback. AVX2 and AVX-512/VNNI were both much
faster than scalar there, but their paired interval against one another crossed
zero, so neither candidate met the unique-winner rule. The other seven
candidate/shape rows likewise remained non-qualifying rather than being hidden
or promoted from point estimates.

### Full-model CPU context

| Lane | Startup median (s) | Prompt / TTFT median (s) | Full request median (s) | Decode median (tok/s) | Peak RSS median (KiB) | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gpt-oss-rs Auto | 59.379 | 220.206 / 220.206 | 224.473 | 1.880 | 10,659,476 | exact 8/8 |
| gpt-oss-rs scalar | 59.191 | 183.865 / 183.865 | 210.067 | 0.310 | 10,659,520 | exact 8/8 |
| llama.cpp normal CPU | 6.040 | 1.375 / 1.375 | 1.897 | 15.429 | 23,509,208 | exact 8/8 |
| llama.cpp `ubatch=1` | 6.039 | 3.486 / 3.486 | 3.998 | 15.646 | 23,509,112 | exact 8/8 |

Every warmup and measured lane emitted the official eight-token sequence
`[200005, 35644, 200008, 976, 1825, 5003, 25, 392]`; no divergence suppression
was required. Auto's decode throughput was a supported 6.05x over scalar (95%
paired bootstrap interval 5.59x-6.23x). Auto did not earn a prompt or
full-request latency claim: the favorable-direction estimates were 0.835
(95% interval 0.832-0.850) and 0.936 (0.932-0.946), respectively, and the Auto
medians were slower. Startup was effectively equal by median.

The llama.cpp lanes are same-host context only. Their timing and memory
surfaces are retained without a cross-project ratio or universal ranking.
Fresh-process startup is separate from the request-time columns for every
lane.

### GGUF conversion identity

Pinned llama.cpp at `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`
converted the locally present SafeTensors to a 13,792,638,656-byte, 459-tensor
GGUF in 44.05 seconds with zero swap operations. Its SHA-256 is
`aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`.
That differs from the historical fixture hash
`27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
The difference is published as an identity result; no replacement GGUF was
downloaded and no historical hash is claimed for the converted file.

### Tested-model boundary

GPT-OSS 20B has end-to-end CPU execution and bounded measurement. GPT-OSS
120B received metadata, tensor-mapping, placement, and capacity study only on
the archived HET line. No 120B end-to-end execution, parity, or performance
claim is made.

## Negative and incomplete results

- A small full-model BF16 reduction-order trace difference was observed before
  expert kernels, although it did not change the maintained greedy sequences.
- Iris Xe experiments produced isolated kernel wins but failed the full-model
  automatic-promotion gate; Auto remained CPU.
- Automatic multi-row MXFP4 selection retained scalar wherever the measured
  profile did not establish a qualifying optimized region.
- In particular, gate/up at `M=8` retained scalar because the two optimized
  candidates did not establish a unique winner against one another.
- Auto substantially improved short-decode throughput over scalar on the final
  workload, but increased prompt and full-request latency; only the supported
  decode result is reported as a speedup.
- The locally converted GGUF did not reproduce the historical fixture hash.
- The HET archive passed substantial selected-expert, relay, ownership, and
  transactional sub-gates, but 120B construction/execution never began. The
  final retained-20B capacity-one comparison stopped because one native shard
  exceeded its frozen mapping window.
- The historical two-GPU layer-sharding branch never executed activation
  handoff or demonstrated end-to-end parity.

These are results, not omissions to hide. Their exact records remain in the
[HET archive](https://github.com/gracee3/gpt-oss-rs/tree/7bb459361c68b00eed45f56a622c061bb4b135ff)
and [layer-sharding archive](https://github.com/gracee3/gpt-oss-rs/tree/166c0573c970334333f3fed567e1c88bf00bfe4f).

## Threats to validity

- Final performance evidence comes from one host, CPU, memory topology,
  operating-system state, model revision, prompt, and short decode.
- Five measured full-model trials give bounded same-host context, not a broad
  workload distribution. Bootstrap intervals do not remove systematic bias.
- A warm page cache reflects repeated local use but not first-ever checkpoint
  access. Startup still includes fresh-process model construction.
- gpt-oss-rs and llama.cpp expose different internal timing surfaces; only
  explicitly defined fields are compared, and no universal ranking follows.
- CPU frequency, firmware, kernel, compiler, and memory pressure can change
  results even when source and model hashes match.
- Greedy token equality is a strong regression signal but not proof of equal
  logits, equal probabilities, model quality, or service safety.
- Archived HET synthetic and subcomponent evidence cannot establish the
  deferred 120B or multi-GPU end-to-end claims.

## Reuse guidance

The most portable outputs are narrow contracts rather than the full server:

- reuse scalar semantics and exact fixtures before adopting an optimized
  kernel;
- bind packed caches to source identity, layout version, and atomic publication;
- keep model, sequence, scratch, and delivery ownership distinct;
- make reserve/execute/commit and cancellation visibility explicit;
- use absolute model identity even when execution uses shard-local indices;
- keep benchmark identity, raw samples, analysis, and rejection reasons
  together; and
- preserve negative results when they constrain safe dispatch or future work.

Compatibility with another project still requires an independent license,
ABI, numerical, and workload review.

## Data availability and reproducibility

The [v0.1.0 evidence bundle](evidence/v0.1.0/README.md) contains the published
normalized samples, matrix raw data, analysis, environment identity, conversion
provenance, schema, and checksums. It excludes model payloads, derived repack
caches, build directories, credentials, hardware serials, hostnames, and local
personal paths. The GPT-OSS model and llama.cpp source must be obtained under
their own licenses and policies.

Source archives are pinned by full Git commit. The release has no DOI; use the
root [citation metadata](../../CITATION.cff).

## Ethics, attribution, and interests

`gracee3` is the release author and responsible maintainer. m0at and other
contributors retain lineage credit. OpenAI Codex agents assisted but are not
authors or evidence authorities. No external funding, sponsorship, or
competing interests were declared. The complete statement is in [research
ethics and disclosure](RESEARCH_ETHICS.md).

## Conclusions

The project demonstrates that a compact, inspectable CPU-first GPT-OSS runtime
can support exact kernel contracts, native model execution, transactional
ownership, and evidence-driven dispatch research in Rust. Its strongest result
is not a production or ranking claim; it is a set of reusable implementations
and methods whose successes, uncertainty, and failures remain auditable. The
v0.1.0 program is research-complete and moves to maintenance. Any renewed HET,
120B, multi-GPU, or broad service program should begin with a separately
bounded question and evidence gate.
