# gpt-oss-rs

`gpt-oss-rs` is a CPU-first Rust research implementation for OpenAI GPT-OSS.
Version 0.1.0 closes the project's planned research program as a citable
artifact: source, methods, measurements, provenance, and negative results are
published together. It is not presented as a production-ready inference
server or a universal performance winner.

The publication line is rebuilt from pre-heterogeneous commit
[`0113e82`](https://github.com/gracee3/gpt-oss-rs/commit/0113e8214e765d168216bbee2120654555a4cfe4).
Later heterogeneous and multi-GPU work is preserved in named archives, not in
the v0.1.0 runtime.

## Verified CPU-first capabilities

- Native execution of the pinned GPT-OSS 20B SafeTensors checkpoint with BF16
  dense tensors and compact MXFP4 expert weights.
- Scalar, AVX2, AVX-512/VNNI, and capability-selected CPU kernel paths, with a
  scalar oracle and exact-bit kernel equivalence gates.
- Residual-Q8 expert activations, versioned x8 repacking, atomic derived-cache
  publication, and read-only cache reopening.
- Transactional layer-major prompt prefill, separate immutable model and
  mutable sequence state, and reserve/execute/commit scheduling seams.
- Exact greedy-token parity for the maintained seven-scenario Harmony corpus
  at the pinned 20B model and oracle identities documented in the repository.
- Existing experimental Completions, Chat Completions, and Responses HTTP
  surfaces. Version 0.1.0 adds no server API.

These are bounded research claims. CPU trusted mode remains disabled, the
service surface is not production-certified, and automatic dispatch should not
be generalized beyond its checked profiles.

## Final benchmark

The release benchmark uses the real GPT-OSS expert shapes and a 63-token
Harmony prompt with eight greedy output tokens. It pins physical CPUs 0-7,
uses eight threads, disables GPU backends and prompt-cache reuse, rotates run
order, starts below a thermal gate, rejects throttling, and uses fresh
processes while leaving the page cache warm.

The results table is populated only from the checked-in v0.1.0 evidence bundle.

| Lane | Startup median (s) | Prompt / TTFT median (s) | Full request median (s) | Decode median (tok/s) | Peak RSS median (KiB) | Token result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gpt-oss-rs Auto | 59.379 | 220.206 / 220.206 | 224.473 | 1.880 | 10,659,476 | exact 8/8 |
| gpt-oss-rs scalar | 59.191 | 183.865 / 183.865 | 210.067 | 0.310 | 10,659,520 | exact 8/8 |
| llama.cpp normal CPU | 6.040 | 1.375 / 1.375 | 1.897 | 15.429 | 23,509,208 | exact 8/8 |
| llama.cpp `ubatch=1` | 6.039 | 3.486 / 3.486 | 3.998 | 15.646 | 23,509,112 | exact 8/8 |

Internal speedups are reported only when the paired 10,000-iteration bootstrap
interval supports them. llama.cpp is retained as same-host context, not as a
universal ranking. If output tokens diverge, decode and full-request ratios are
omitted and the divergence remains a result.

All four measured lanes produced the pinned official token sequence in every
trial. Auto's decode throughput was a supported 6.05x over scalar (95% paired
bootstrap interval 5.59x-6.23x), but Auto had slower prompt and full-request
medians and earned no latency-speedup claim. The locally converted GGUF hash
also differed from the historical fixture hash; both identities are retained
in the evidence instead of substituting a downloaded file.

See the [benchmark protocol](docs/research/BENCHMARK_PROTOCOL.md), [final
report](docs/research/FINAL_REPORT.md), and [versioned evidence
bundle](docs/research/evidence/v0.1.0/README.md).

## Tested models

| Model | Evidence boundary | Claim |
| --- | --- | --- |
| GPT-OSS 20B, revision `6cee5e8` | Local full checkpoint; native CPU load, prefill, and eight-token greedy decode | End-to-end CPU execution and bounded performance evidence |
| GPT-OSS 120B, revision `b5c939d` | Metadata, index, tensor mapping, and placement-envelope research on the archived HET line | No v0.1.0 execution, parity, or performance claim |

No 120B payload is required by the v0.1.0 publication workflow.

## Reusable research outputs

- `gpt-oss-cpu-kernels`: explicit MXFP4 layouts, scalar semantics, x86
  dispatch, scratch contracts, and exact equivalence tests.
- `gpt-oss-model-runner`: mapped checkpoint ownership, versioned repack caches,
  transactional model steps, and layer-major prefill.
- `gpt-oss-engine`: bounded admission, sequence scheduling, cancellation, and
  reserve/execute/commit state transitions.
- `gpt-oss-evidence`: versioned manifests, hashed artifacts, redaction, and
  fail-closed evidence publication.
- `gpt-oss-bench`: fixed prompts, controlled microbenchmarks, paired bootstrap
  analysis, and negative-result preservation.
- Research records that distinguish verified facts, inference, hypotheses,
  deferred work, and conflicts rather than collapsing them into a roadmap.

## Build and bounded use

```bash
# CPU-only build; CUDA and Xe integrations are excluded.
cargo build --release -p gpt-oss-server --no-default-features

# Direct full-model research runner used by the publication protocol.
cargo build --release -p gpt-oss-bench --bin cpu_parity --no-default-features
```

The historical server binary remains `target/release/gpt-oss-rs`:

```bash
./target/release/gpt-oss-rs serve \
  --model openai/gpt-oss-20b \
  --device cpu \
  --cpu-threads 8
```

For a local model directory not created by `fetch`, pass a stable public
identity with `--served-model-name`. Do not treat local paths as public model
IDs or evidence labels.

## Validation

```bash
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
python3 -m unittest discover -s crates/gpt-oss-bench/tools/tests -p 'test_*.py'
python3 -m unittest discover -s oracle/tests -p 'test_*.py'
python3 tools/check_markdown_links.py
cargo clippy -p gpt-oss-cpu-kernels -p gpt-oss-evidence -p gpt-oss-bench \
  --all-targets --no-deps --locked -- -D warnings
cargo build --release -p gpt-oss-server --no-default-features --locked
```

The release also validates `CITATION.cff` against CFF 1.2.0 and verifies the
evidence bundle against `SHA256SUMS`.

## Archived research

The archives are intentionally non-production and incomplete. They remain
browsable so useful designs and failed gates are not erased when `main` returns
to the CPU-first line.

| Archive | Exact tip | Preserved boundary |
| --- | --- | --- |
| [Heterogeneous research](https://github.com/gracee3/gpt-oss-rs/tree/7bb459361c68b00eed45f56a622c061bb4b135ff) | `7bb4593` | Static CPU/GPU0/GPU1 expert ownership, bounded relay, transaction and evidence work; no successful 120B execution |
| [Former HET-era main](https://github.com/gracee3/gpt-oss-rs/tree/249abfbf5f21dddb434a7975c02df396e0608dc7) | `249abfb` | Last merged HET runtime state before publication-line rebuild |
| [Layer-sharding assessment](https://github.com/gracee3/gpt-oss-rs/tree/bc8cf36f7ba79d318c9264e0f9f4198ac4135c60) | `bc8cf36` | Documentation-only salvage assessment |
| [Historical 58-commit layer-sharding branch](https://github.com/gracee3/gpt-oss-rs/tree/166c0573c970334333f3fed567e1c88bf00bfe4f) | `166c057` | Planning and allocation scaffolding; no executable activation handoff or parity proof |

The standalone [multi-GPU retrospective](docs/research/MULTI_GPU_RETROSPECTIVE.md)
states what is reusable and what was never proven.

## Sources, influence, and attribution

This repository began as a narrowed fork of
[m0at/rvllm](https://github.com/m0at/rvllm). Inherited authorship remains in
Git history and repository notices. `gracee3` is the v0.1.0 release author and
responsible maintainer.

| Source or community | Relationship | Full record |
| --- | --- | --- |
| OpenAI GPT-OSS | Model and semantic authority | [CPU provenance](docs/UPSTREAM_PROVENANCE.md) and [archived HET source ledger](https://github.com/gracee3/gpt-oss-rs/blob/7bb459361c68b00eed45f56a622c061bb4b135ff/docs/het/11-source-ledger.md) |
| m0at/rvllm | Inherited repository foundation | [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md) |
| llama.cpp and ik_llama.cpp | MXFP4/Q8/x86 comparison and adapted algorithmic concepts | [CPU provenance](docs/UPSTREAM_PROVENANCE.md) |
| mistral.rs | GPT-OSS semantic cross-checks | [CPU provenance](docs/UPSTREAM_PROVENANCE.md) |
| vLLM, Sarathi-Serve, MegaBlocks, oneDNN | Scheduling, routing, evidence, packing, and scratch concepts | [borrowed-concepts ledger](docs/BORROWED_CONCEPTS.md) |
| KTransformers, Fiddler, NCCL, CUDA references | Archived heterogeneous research references | [archived HET source ledger](https://github.com/gracee3/gpt-oss-rs/blob/7bb459361c68b00eed45f56a622c061bb4b135ff/docs/het/11-source-ledger.md) |

See [research ethics and disclosure](docs/research/RESEARCH_ETHICS.md) for
authorship, AI-assistance, funding, and competing-interest statements. Citation
metadata is in [CITATION.cff](CITATION.cff); no DOI is claimed.

## Project state

The planned v0.1.0 research program is complete. The repository is in
maintenance mode: correctness fixes, reproducibility repairs, security fixes,
and evidence clarifications are welcome, but new runtime programs require a
new explicit research scope. See [project intent](docs/PROJECT_INTENT.md),
[milestones](docs/NEXT_MILESTONES.md), and [contributing](CONTRIBUTING.md).
