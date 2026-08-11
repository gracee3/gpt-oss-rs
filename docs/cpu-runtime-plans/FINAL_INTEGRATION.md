# CPU Runtime Feature-Set Final Integration Record

- Date: 2026-08-11
- Baseline: `3600fa4`
- Verified implementation head: `7f85707`
- Integration branch: `agent/cpu-runtime-feature-set`
- Model: `/data/models/openai/gpt-oss-20b`
- Artifact root: `/data/models/openai/gpt-oss-rs-cpu-work/results`

## Repository review

`origin/main` remained at `3600fa4`, and the integration branch remained a
clean, linear descendant of that commit. The reviewed range contains the
research and program plans followed by M1, M3, M2, M4, and M5 in the planned
dependency order. `git diff --check origin/main...HEAD` passed. The final range
changed 54 repository files without adding model snapshots, repack caches,
targets, traces, or result captures to Git.

## Final local gate

The following commands passed with `--locked` where shown:

```text
cargo fmt --all --check
cargo check --workspace --locked
cargo test -p gpt-oss-cpu-kernels -p gpt-oss-model-runner -p gpt-oss-engine -p gpt-oss-server --locked
cargo clippy -p gpt-oss-cpu-kernels --all-targets --locked -- -D warnings
cargo check -p gpt-oss-server --features amx-int8 --locked
cargo test -p gpt-oss-cpu-kernels --features amx-int8 --locked
cargo clippy -p gpt-oss-cpu-kernels --all-targets --features amx-int8 --locked -- -D warnings
```

The focused default run completed 36 CPU-kernel tests, 358 model-runner unit
tests plus two official-checkpoint integration tests, 253 engine unit tests
plus three engine regressions, and 102 server-library, five server-binary,
seven server end-to-end, and six HTTP-contract tests. The feature-enabled AMX
run completed 41 portable CPU-kernel tests. Both default and AMX-enabled
CPU-kernel Clippy runs denied warnings successfully.

Branch CPU workflow
[31530768556](https://github.com/gracee3/gpt-oss-rs/actions/runs/31530768556)
passed all three jobs: the ordinary locked workspace checks/tests, forced
kernel dispatch, and the new propagated-feature AMX compile/test/Clippy job.

## Short 20B smoke

The development host exposed eight allowed logical CPUs, four observed
physical cores, one observed NUMA node, AVX-512/VNNI, and no AMX CPUID support.
The release server was tested only with short greedy completion requests.

1. Default `--device auto`, automatic kernel/matrix dispatch, and the
   batch-one CPU profile loaded successfully. One non-streaming request
   completed with one finite token and `finish_reason=length`.
2. Forced `--device cpu --cpu-kernel avx512-vnni
   --cpu-matmul-backend avx2`, with two sequence slots, selected
   `avx512-vnni-x8`, `InterleavedSplitX8V2`, and the explicit AVX2 matrix
   backend. The same prompt produced the same first token (`":"`) as the
   default path.
3. Two simultaneous two-token requests, one streaming and one non-streaming,
   completed successfully. They had distinct request IDs. The streaming file
   contained two ordered events with one stable ID followed by exactly one
   `[DONE]`; the non-streaming response contained two completion tokens and
   `finish_reason=length`.
4. A feature-enabled forced `amx-int8` startup failed clearly at `CPUID does
   not report AMX-TILE`, before snapshot mapping and worker construction.

The retained result files are:

- `final-default-server.log` and `final-default-nonstream.json`;
- `final-forced-server.log` and `final-forced-nonstream.json`;
- `final-concurrent-stream.sse` and `final-concurrent-nonstream.json`.

The two agreement-response SHA-256 hashes are
`1ac20b525458c2cecce4cc950f7a882657e3b07baff4ef804ba0e1d266d3fb01`
and `71ddfca4438fc7ed26e1f9deb237266c1e3dbc71a303ddf5e06c16e1072ef394`.
The concurrent streaming/non-streaming hashes are
`4b438db870c8ebd0d095ad3ee0ca575754432a30d3d17b2c88b1a5d4cd668c25`
and `b84d275d380095bb781503fe5a8aa83dbae88bd9c55798628bff12ea18acba46`.

## Deferred certification and tuning

This closeout intentionally did not run Criterion, timing thresholds, long
generations, the 28-run oracle matrix, fresh llama.cpp captures, exhaustive
API permutations, or AMX-hardware tests. Trusted-mode promotion, automatic
selection of the new AVX-512/matrix/AMX paths, crossover thresholds,
performance tuning, AMX native equality and lifecycle stress, and broader
cross-host certification remain later work.

## Publication outcome

The evidence-only branch checkpoint passed CPU workflow `31531169406`.
`edee07e` was then fast-forwarded to `origin/main`; the remote ref was read
back at that exact SHA, and main CPU workflow
[31531447410](https://github.com/gracee3/gpt-oss-rs/actions/runs/31531447410)
passed its workspace, forced-dispatch, and portable-AMX jobs. The integration
branch became eligible for local and remote deletion only after those checks.
