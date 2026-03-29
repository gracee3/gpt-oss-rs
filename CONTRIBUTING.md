# Contributing to gpt-oss-rs

Keep changes aligned with the current repository scope: GPT-OSS serving in Rust, not general multi-model experimentation or Python-side tooling.

## Setup

```bash
git clone https://github.com/m0at/gpt-oss-rs.git
cd gpt-oss-rs
cargo check --workspace
cargo test --workspace
```

Use the mock backend for everyday development. CUDA is optional unless your change is specific to GPU execution.

## Expectations

- Keep PRs focused and small enough to review.
- Do not reintroduce Python packaging, bindings, or comparison harnesses unless there is a concrete runtime requirement.
- Preserve the GPT-OSS-only scope unless the project direction changes explicitly.
- Update root docs when user-facing behavior, commands, or repository layout changes.

## Before Opening a PR

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

Run any narrower checks that match your change:

- `cargo test -p <crate>`
- `cargo bench -p gpt-oss-bench --bench sampling_bench`
- `cargo build --release -p gpt-oss-server`

For CUDA-specific changes, also run the relevant `--features cuda` build or test commands on hardware if you have access to it.

## Code Guidelines

- Prefer straightforward crate boundaries and explicit data flow.
- Keep CUDA-specific behavior behind feature gates.
- Avoid adding dependencies without a clear reason.
- Add or update tests when behavior changes.
- Keep docs current; stale docs are worse than no docs.
