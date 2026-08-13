# gpt-oss-rs

`gpt-oss-rs` is an experimental, CPU-first Rust inference engine for OpenAI
GPT-OSS checkpoints. It provides an OpenAI-compatible HTTP server,
Harmony-native prompt/tool handling, and a native Linux CPU backend that keeps
MXFP4 expert weights compact in memory. CUDA remains available as an explicit
experimental backend.

The project is deliberately focused on GPT-OSS. It is not a general model
runner, and the CPU backend is not yet enabled in trusted mode.
It is also an educational, evidence-producing project centered on hardware
available to its owner rather than a compatibility exercise for every model,
framework, or accelerator. See
[`docs/PROJECT_INTENT.md`](docs/PROJECT_INTENT.md).

## Current status

- Native CPU serving is the default. `--device auto` resolves to CPU for
  GPT-OSS even when CUDA or OpenCL is present because the Xe performance
  promotion record is not passing.
- The default server binary includes a runtime-loaded, Linux-only Iris Xe
  projection backend. `--device xe` explicitly attaches it only on exact Intel
  `8086:9a49`; hosts without OpenCL retain the CPU path and no OpenCL link-time
  dependency.
- CUDA serving requires the explicit combination `--device cuda
  --runtime-mode experimental`.
- CPU serving supports official GPT-OSS SafeTensors, a batch-one default with
  opt-in experimental multi-request scheduling, BF16 dense weights, MXFP4
  experts, and layer-major multi-row prompt prefill.
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
- CPU trusted mode, tensor/pipeline parallel CPU execution, and CPU CUDA-graph
  behavior remain unsupported. Multi-request scheduling is experimental rather
  than a broad service-compatibility promise.

The CPU backend now has capability-level dispatch, AVX2 and experimental
AVX-512/VNNI x8 MXFP4 GEMV, plus a common matrix contract and an explicit AVX2
4x8 prefill path. Automatic multi-row matrix execution remains on the scalar
reference. Later milestones cover CPU request scheduling and experimental AMX.
See
[`docs/MXFP4_CPU_BACKEND_HANDOFF.md`](docs/MXFP4_CPU_BACKEND_HANDOFF.md).

## Build

```bash
# Native CPU build with portable runtime-loaded Xe support
cargo build --release -p gpt-oss-server

# CPU-only binary without the optional Xe integration
cargo build --release -p gpt-oss-server --no-default-features

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
initializing CUDA. It probes Xe only when a checked-in full-model promotion
record is enabled; the current record is disabled. Select CPU explicitly and
control its cache and worker count with:

```bash
GPT_OSS_RS_CACHE=/path/to/gpt-oss-rs-cache \
./target/release/gpt-oss-rs serve \
  --model openai/gpt-oss-20b \
  --device cpu \
  --cpu-kernel auto \
  --cpu-matmul-backend auto \
  --cpu-threads 8
```

Current forced CPU kernel values are `scalar`, `avx2`, and `avx512-vnni`.
Forcing an unavailable ISA fails before model execution. `auto` detects host
capabilities once and builds an immutable per-operation dispatch plan. The
plan also reports the exact MXFP4 GEMV kernel and packed layout.
Forced `avx512-vnni` uses the experimental eight-output ZMM/VNNI MXFP4 path;
automatic MXFP4 dispatch remains on the promoted AVX2 x8 baseline.

The explicit hybrid path keeps routing, attention, KV state, sampling, commit,
and fallback on CPU while offloading prefill expert buckets with four or more
rows:

```bash
./target/release/gpt-oss-rs serve \
  --model openai/gpt-oss-20b \
  --device xe \
  --xe-max-resident-mib 128 \
  --xe-expert-cache-mib 0 \
  --runtime-mode experimental
```

Decode and M=1–3 expert buckets remain on the configured CPU kernels. A runtime
OpenCL fault drains and discards uncommitted Xe output, recomputes once on CPU,
and opens a process-wide Xe breaker until restart. Explicit attachment fails
startup on an unavailable or invalid device; automatic attachment failures
fall back to CPU. Trusted mode remains blocked.

`--cpu-matmul-backend` accepts `auto`, `scalar`, `avx2`, `avx512-vnni`, and
`amx-int8`.
`auto` keeps M=1 expert work on the established GEMV path and uses the scalar
matrix reference for M>1. Select `avx2` explicitly to exercise the experimental
4-input-row by 8-output-row packed path. `amx-int8` requires the optional build
feature plus Linux x86-64 AMX hardware, XSTATE support, and process permission;
it is forced-only and never selected by `auto`. Portable packing and tile
emulation are covered without claiming local AMX-hardware execution.

CPU serving defaults to the `gpt-oss-cpu` profile: an 8192-token context cap
and one active sequence. The first load creates a revision- and
checksum-keyed MXFP4 repack cache; later loads memory-map that cache read-only.
The native CPU service also applies bounded admission, logical request-memory
reservations, byte-charged delivery, and a terminal-only bounded Responses
store. For a local model directory that was not created by `fetch`, pass a
stable public identity with `--served-model-name`; filesystem paths are never
published as API model IDs or metric labels.

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
- `GET /v1/responses/:response_id`
- `GET /v1/responses/:response_id/input_items`
- `GET /v1/models`
- `GET /health`
- `GET /ready`
- `GET /metrics` unless telemetry exposition is disabled

Chat Completions and Responses support streaming and non-streaming Harmony
text/tool flows through the same engine path. Streaming text is emitted as
committed suffix deltas; cancellation and service failures use typed errors
and are not reported as successful `stop` finishes. The mounted
`/v1/chat/completions/tools` and `/tools` routes are aliases of the chat
handler, not separate compatibility contracts. Batch route source is retained
for later work, but no `/v1/batches` route is currently mounted.

`/health` is a liveness probe. `/ready` returns 200 only after model,
tokenizer, owner, delivery, reservations, and the hashed effective runtime
snapshot are ready; otherwise it returns a typed 503. The default limits are a
2 MiB request body, 8 MiB non-streaming response/store entry, 256 KiB stream
event, 1 MiB queued delivery per request, `max_num_seqs` MiB global delivery,
and a 64 MiB/64-entry Responses store. `serve --help` lists the corresponding
overrides, `--cpu-request-budget-mib`, `--evidence-dir`, and bounded diagnostic
options. HTTP serving permits metadata and summary diagnostics only.

## CPU architecture

The CPU implementation is split so ISA-specific code does not leak into the
engine:

- `gpt-oss-cpu-kernels`: feature detection, dispatch plans, scalar and x86
  primitives, and Criterion microbenchmarks;
- `gpt-oss-model-runner`: SafeTensors mapping, MXFP4 repack ownership,
  transactional layer-major transformer execution, and parity tracing;
- `gpt-oss-xe`: internal runtime-loaded OpenCL ownership, native program cache,
  bounded streaming slab, startup numerical test, and circuit breaker;
- `gpt-oss-engine`: canonical CPU sequence scheduling, transactional batched
  execution, logical reservations, managed delivery/lifecycle, and the
  batch-one compatibility worker;
- `gpt-oss-evidence`: versioned run manifests, artifact verification,
  effective-runtime snapshots, redaction, and bounded diagnostics;
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
cargo clippy -p gpt-oss-xe --all-targets --locked -- -D warnings
cargo bench -p gpt-oss-cpu-kernels --bench kernels
```

`cpu_parity`, the model-independent `cpu_service_probe`, and the paired release
`cpu_service_overhead` gate write atomic raw captures with automatic
`gpt-oss-rs.cpu-evidence/v1` manifest sidecars. The service probe checks only
public service metadata and can be run against an already-ready server without
generating tokens.

The AVX2 x8 promotion used a targeted full-model gate: cold and warm
`harmony_122`, plus automatic, AVX2, and scalar `harmony_262`, followed by
streaming and non-streaming API requests. Performance measurements are
informational for this milestone. The exhaustive oracle/API/benchmark matrix
is intentionally deferred until the experimental CPU feature set is complete
and ready for a separate tuning/certification campaign; focused correctness
and safety checks remain mandatory for each feature slice.

The production runtime is Rust. Narrow Python tools under
`crates/gpt-oss-bench/tools` are retained as diagnostic bridges to the pinned
official PyTorch oracle; they are not runtime dependencies.

Current documentation:

- [`docs/PROJECT_INTENT.md`](docs/PROJECT_INTENT.md): project purpose, present
  hardware focus, CPU-first scope, and criteria for future targets;
- [`docs/CPU_RUNTIME.md`](docs/CPU_RUNTIME.md): CPU storage, numerical, serving,
  and dispatch invariants;
- [`docs/MXFP4_MATRIX_API.md`](docs/MXFP4_MATRIX_API.md): typed matrix views,
  scratch ownership, backend policy, and layer-major integration;
- [`docs/CPU_SCHEDULER.md`](docs/CPU_SCHEDULER.md): canonical sequence
  ownership, reserve/execute/commit, cancellation, fairness, and topology;
- [`docs/MXFP4_CPU_BACKEND_HANDOFF.md`](docs/MXFP4_CPU_BACKEND_HANDOFF.md): Intel
  ISA backend direction and implementation milestones;
- [`docs/CPU_I7_CONFORMANCE.md`](docs/CPU_I7_CONFORMANCE.md): repeatable
  full-checkpoint CPU regression procedure;
- [`docs/AMX_INT8.md`](docs/AMX_INT8.md): feature, capability, permission,
  panel, fallback, and portable-validation contract for the forced AMX backend;
- [`docs/xe-research/09-production-integration-and-auto-promotion.md`](docs/xe-research/09-production-integration-and-auto-promotion.md):
  hybrid integration, full-model evidence, and the gated automatic-dispatch
  decision;
- [`docs/UPSTREAM_PROVENANCE.md`](docs/UPSTREAM_PROVENANCE.md): audited upstream
  concepts and pinned revisions;
- [`docs/cpu-runtime-research/`](docs/cpu-runtime-research/README.md): completed
  source-grounded pre-planning for AVX-512 x8, GEMM/prefill, state separation,
  CPU scheduling, and AMX;
- [`docs/CPU_RUNTIME_NEXT_PHASE_RESEARCH.md`](docs/CPU_RUNTIME_NEXT_PHASE_RESEARCH.md):
  completed documentation-only research on evidence, service lifecycle, memory,
  numerical trust, operator seams, AMX closure, and bounded future pressure;
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
