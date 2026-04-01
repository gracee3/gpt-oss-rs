# gpt-oss-rs

`gpt-oss-rs` is a Rust-only workspace for serving OpenAI GPT-OSS checkpoints behind an OpenAI-compatible HTTP API.

## Scope

- GPT-OSS checkpoints only
- Harmony-native GPT-OSS protocol rendering only
- Rust crates, CUDA kernels, and Criterion benchmarks only
- OpenAI-compatible text generation endpoints for the server binary
- No Python bindings, Python benchmark harnesses, or fork-era comparison tooling

## Quick Start

```bash
# CPU / mock backend
cargo build --release -p gpt-oss-server

# CUDA backend
cargo build --release --features cuda -p gpt-oss-server
```

```bash
./target/release/gpt-oss-rs serve --model openai/gpt-oss-20b
```

The server exposes:

- `/v1/completions`
- `/v1/chat/completions`
- `/v1/responses`
- `/v1/models`
- `/health`
- `/metrics`

## Development

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo bench -p gpt-oss-bench --bench sampling_bench
docker build -t gpt-oss-rs -f Dockerfile .
```

Useful entry points:

- `crates/gpt-oss-server`: CLI and HTTP server binary
- `crates/gpt-oss-bench`: repository-level Rust benchmarks
- `kernels/`: CUDA kernels loaded by the GPU path

## Tier-2 Workflow

Restricted fp16 CUDA Tier 2 now uses an explicit three-step contract:

- raw global compare = telemetry
- runtime-emulated global compare = localization
- same-input local replay = ownership proof

Current docs:

- `docs/TIER2_FP16_CUDA_WORKFLOW.md`: canonical harness usage, compare modes, seed capture, and local replay
- `docs/TIER2_RESULTS_AND_STATUS.md`: current findings and what remains unresolved
- `docs/REPO_ALIGNMENT_AND_WORKSTREAMS.md`: active branch/worktree policy and forward workstreams

## Notes

- The workspace intentionally stays narrow. If a new script, test harness, or package format is not part of the Rust serving path, it should not live here by default.
- A narrow Tier-2 validation harness is intentionally retained under `crates/gpt-oss-bench` and `scripts/` because it is the current source of truth for restricted fp16 CUDA investigation and live testing.
- Historical optimization notes and fork migration collateral were removed to keep the repository easier to maintain. Add new docs only when they are current and directly useful.
- Related project: [m0at/rvllm](https://github.com/m0at/rvllm)

## Lineage

This repository began as a narrowed fork of [m0at/rvllm](https://github.com/m0at/rvllm).
It was then renamed and refocused into `gpt-oss-rs`, a GPT-OSS inference engine.

Credit for the original `rvllm` foundation and inherited upstream work goes to `m0at`
and the other upstream contributors whose authorship remains preserved in git history.

## License And Attribution

This repository contains inherited upstream work from `m0at/rvllm`, so the repository
continues to preserve Apache-2.0 licensing and attribution for that code.

The current fork intentionally credits the original upstream work in this README, in
the git history, and in the repository notice file rather than pretending the codebase
started here.
