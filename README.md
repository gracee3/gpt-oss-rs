# gpt-oss-rs

`gpt-oss-rs` is an experimental, CPU-first Rust inference engine for OpenAI
GPT-OSS checkpoints. It provides an OpenAI-compatible HTTP server,
Harmony-native prompt/tool handling, and a native Linux CPU backend that keeps
MXFP4 expert weights compact in memory. CUDA remains available as an explicit
experimental backend.

The project is deliberately focused on GPT-OSS. It is not a general model
runner, and the CPU backend is not yet enabled in trusted mode.

## Current status

- Native CPU serving is the default. `--device auto` resolves to CPU for
  GPT-OSS even when CUDA is present.
- CUDA serving requires the explicit combination `--device cuda
  --runtime-mode experimental`.
- CPU serving supports official GPT-OSS SafeTensors, batch size one, BF16 dense
  weights, and MXFP4 experts.
- The preceding CPU baseline has exact greedy-token parity with the pinned
  official SafeTensors/PyTorch oracle across the seven maintained Harmony
  scenarios and four kernel choices. The promoted AVX2 x8 path is gated by the
  targeted `harmony_122` and `harmony_262` vertical slice; repeating the full
  28-run matrix is deferred follow-up work.
- Residual Q8 is the current expert-activation path. A streaming exact-BF16
  projection exists for diagnostics only.
- A small end-to-end BF16 reduction-order trace difference remains before the
  expert kernels. It does not change the maintained greedy sequences, but it is
  retained as an explicit diagnostic limitation.
- CPU trusted mode, request batching, tensor/pipeline parallel CPU execution,
  and CPU CUDA-graph behavior remain unsupported.

The CPU backend now has capability-level dispatch and a size-neutral AVX2 x8
MXFP4 GEMV layout while preserving scalar and canonical row references. Later
milestones cover purpose-built AVX-512 GEMV, GEMM/prefill packing, and
experimental AMX. See
[`docs/MXFP4_CPU_BACKEND_HANDOFF.md`](docs/MXFP4_CPU_BACKEND_HANDOFF.md).

## Build

```bash
# Native CPU build
cargo build --release -p gpt-oss-server

# CUDA-enabled build
cargo build --release --features cuda -p gpt-oss-server
```

The server binary is `target/release/gpt-oss-rs`.

## Serve GPT-OSS

```bash
./target/release/gpt-oss-rs serve \
  --model openai/gpt-oss-20b
```

`--device auto` selects the native CPU backend for GPT-OSS without probing or
initializing CUDA. Select CPU explicitly and control its cache and worker count
with:

```bash
GPT_OSS_RS_CACHE=/path/to/gpt-oss-rs-cache \
./target/release/gpt-oss-rs serve \
  --model openai/gpt-oss-20b \
  --device cpu \
  --cpu-kernel auto \
  --cpu-threads 8
```

Current forced CPU kernel values are `scalar`, `avx2`, and `avx512-vnni`.
Forcing an unavailable ISA fails before model execution. `auto` detects host
capabilities once and builds an immutable per-operation dispatch plan. The
plan also reports the exact MXFP4 GEMV kernel and packed layout.

CPU serving defaults to the `gpt-oss-cpu` profile: an 8192-token context cap
and one active sequence. The first load creates a revision- and
checksum-keyed MXFP4 repack cache; later loads memory-map that cache read-only.

Select CUDA only as an explicit experimental opt-in:

```bash
./target/release/gpt-oss-rs serve \
  --model openai/gpt-oss-20b \
  --device cuda \
  --runtime-mode experimental
```

Automatic serving rejects non-GPT-OSS models. Trusted mode rejects both CPU
and CUDA serving, and `--device mock` is test-only and never an automatic
fallback.

## Fetch a pinned snapshot

```bash
./target/release/gpt-oss-rs fetch \
  --model openai/gpt-oss-20b \
  --revision main \
  --cache-dir /path/to/huggingface/hub
```

The fetch command uses resumable Hugging Face cache downloads and writes a
manifest containing the resolved revision, file sizes, and SHA-256 hashes.

## HTTP API

The server exposes:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`

Chat Completions and Responses support streaming and non-streaming Harmony
text/tool flows through the same engine path.

## CPU architecture

The CPU implementation is split so ISA-specific code does not leak into the
engine:

- `gpt-oss-cpu-kernels`: feature detection, dispatch plans, scalar and x86
  primitives, and Criterion microbenchmarks;
- `gpt-oss-model-runner`: SafeTensors mapping, MXFP4 repack ownership, the
  transformer loop, and parity tracing;
- `gpt-oss-engine`: scheduling and the batch-one CPU worker;
- `gpt-oss-bench`: pinned prompt parity tools and model-level measurements;
- `gpt-oss-reference`, `gpt-oss-conformance`, and `gpt-oss-moe-semantics`:
  independent semantic and correctness fixtures.

Canonical checkpoints remain unchanged. CPU backends may construct
load-time, versioned packed representations when a kernel can amortize the
cost. Dispatch must be based on required ISA features and workload shape—not
CPU product names.

## Development and validation

```bash
cargo fmt --all --check
cargo check --workspace --locked
cargo test --workspace --locked
python3 -m unittest discover -s crates/gpt-oss-bench/tools/tests -p 'test_*.py'
cargo clippy -p gpt-oss-cpu-kernels --all-targets --locked -- -D warnings
cargo bench -p gpt-oss-cpu-kernels --bench kernels
```

The AVX2 x8 promotion used a targeted full-model gate: cold and warm
`harmony_122`, plus automatic, AVX2, and scalar `harmony_262`, followed by
streaming and non-streaming API requests. Performance measurements are
informational for this milestone. The exhaustive oracle/API/benchmark matrix
remains documented follow-up work.

The production runtime is Rust. Narrow Python tools under
`crates/gpt-oss-bench/tools` are retained as diagnostic bridges to the pinned
official PyTorch oracle; they are not runtime dependencies.

Current documentation:

- [`docs/CPU_RUNTIME.md`](docs/CPU_RUNTIME.md): CPU storage, numerical, serving,
  and dispatch invariants;
- [`docs/MXFP4_CPU_BACKEND_HANDOFF.md`](docs/MXFP4_CPU_BACKEND_HANDOFF.md): Intel
  ISA backend direction and implementation milestones;
- [`docs/CPU_I7_CONFORMANCE.md`](docs/CPU_I7_CONFORMANCE.md): repeatable
  full-checkpoint CPU regression procedure;
- [`docs/UPSTREAM_PROVENANCE.md`](docs/UPSTREAM_PROVENANCE.md): audited upstream
  concepts and pinned revisions;
- [`docs/TIER2_FP16_CUDA_WORKFLOW.md`](docs/TIER2_FP16_CUDA_WORKFLOW.md): the
  retained restricted-fp16 CUDA investigation contract.

## Design principles

- Preserve MXFP4's compact 4.25-bit-per-weight representation for as long as
  practical.
- Keep a simple scalar oracle and test every optimized kernel against it.
- Detect CPU capabilities once; do not put CPUID checks in hot loops.
- Dispatch on ISA, operation type, matrix shape, batch size, packing
  availability, and measured thresholds.
- Treat GEMV decode and GEMM-like prefill/batched decode as different
  workloads.
- Let benchmarks contradict architectural hypotheses.

## Lineage, license, and attribution

This repository began as a narrowed fork of
[m0at/rvllm](https://github.com/m0at/rvllm) and was renamed and refocused into
a GPT-OSS inference engine. The inherited work remains Apache-2.0, with
authorship preserved in git history and repository notices.

Focused CPU work is additionally attributed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`docs/UPSTREAM_PROVENANCE.md`](docs/UPSTREAM_PROVENANCE.md). Upstream projects
are design and semantic references, not linked runtime dependencies.
