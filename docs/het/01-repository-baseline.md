# Repository baseline

## Capture identity

**Verified by local Git commands before any `docs/het/` file existed:**

| Field | Value |
|---|---|
| Capture time | 2026-08-15T20:40:28-04:00 (host/toolchain capture; Git baseline was captured earlier in the same session) |
| Absolute working tree | `/home/emmy/gpt-oss-rs` |
| Branch | `main` |
| HEAD | `0113e8214e765d168216bbee2120654555a4cfe4` |
| Upstream | `origin/main`; 0 ahead, 0 behind |
| Remote | `origin`, SSH URL `github.com:gracee3/gpt-oss-rs.git` (credential-free) |
| Repository-local instructions | No repository-local `AGENTS.md` found; the supplied host instructions govern this capture. |

The exact initial `git status --short --branch` was:

```text
## main...origin/main
 M Cargo.lock
 M crates/gpt-oss-gpu/src/cublas.rs
 M crates/gpt-oss-gpu/src/cublas_ops.rs
 M crates/gpt-oss-gpu/src/device.rs
 M crates/gpt-oss-gpu/src/memory/cpu_pool.rs
 M crates/gpt-oss-gpu/src/memory/gpu_pool.rs
 M crates/gpt-oss-gpu/src/memory/swap.rs
 M crates/gpt-oss-gpu/src/nccl.rs
 M crates/gpt-oss-gpu/src/pinned_memory.rs
```

The exact initial tracked diff-stat was:

```text
 Cargo.lock                                |  4 ++--
 crates/gpt-oss-gpu/src/cublas.rs          |  6 +++++-
 crates/gpt-oss-gpu/src/cublas_ops.rs      |  4 ++++
 crates/gpt-oss-gpu/src/device.rs          |  2 +-
 crates/gpt-oss-gpu/src/memory/cpu_pool.rs |  6 ++----
 crates/gpt-oss-gpu/src/memory/gpu_pool.rs |  2 +-
 crates/gpt-oss-gpu/src/memory/swap.rs     |  4 ++--
 crates/gpt-oss-gpu/src/nccl.rs            |  2 +-
 crates/gpt-oss-gpu/src/pinned_memory.rs   | 18 +++++++++++++++++-
 9 files changed, 35 insertions(+), 13 deletions(-)
```

### Pre-existing change attribution

**Verified:** the complete diff matches the supplied workstation-readiness
report; no additional tracked change was present.

| Existing file | Observed change | Supplied readiness item |
|---|---|---|
| `Cargo.lock` | `cudarc` 0.19.4 -> 0.19.9 and checksum, at the package record beginning near line 809 | CUDA 13.3 compatibility without overrides |
| `device.rs`, `memory/gpu_pool.rs`, `memory/swap.rs`, `nccl.rs` | Mock-only tests now require `mock-gpu` and exclude `cuda` | Separate mock-only from real CUDA tests |
| `pinned_memory.rs` | Live CUDA context in tests, checked allocation-size multiplication, lifetime documentation | Fix CUDA pinned-memory setup and overflow handling |
| `cublas.rs`, `cublas_ops.rs` | Scoped `too_many_arguments` allowances for BLAS-shaped APIs and an array replacement in a test | Strict GPU Clippy cleanup without changing the cuBLAS-style API |
| `memory/cpu_pool.rs`, `memory/swap.rs` | Warning/Clippy-only cleanup | Strict GPU Clippy cleanup |

These nine files are user-owned. This phase neither formats, stages, rewrites,
nor otherwise modifies them.

## Workspace and executable topology

**Verified from root and crate manifests plus `cargo metadata --no-deps
--locked`:** the workspace has 14 library/application packages plus the bench
and server packages (16 total workspace members).

| Package | Local dependencies (condensed) | Important features / binaries |
|---|---|---|
| `gpt-oss-core` | none | Shared errors and types |
| `gpt-oss-semantics` | core | General semantic types |
| `gpt-oss-moe-semantics` | none | Canonical sparse-MoE semantic helpers |
| `gpt-oss-cpu-kernels` | none | `amx-int8`; scalar/AVX2/AVX-512 dispatch is runtime-selected |
| `gpt-oss-gpu` | core | Default `mock-gpu`; optional `cuda`, `cublaslt`, `cuda-graphs` |
| `gpt-oss-xe` | none | Frozen Tiger Lake Xe backend |
| `gpt-oss-model-runner` | core, CPU kernels, GPU, MoE semantics, semantics, Xe | Default has no feature; optional `cuda`, `cublaslt`, `amx-int8`, `xe` |
| `gpt-oss-kv-model` | core | Logical KV model |
| `gpt-oss-runtime-plan` | core | Runtime-plan/conformance surface, not heterogeneous placement |
| `gpt-oss-reference` | core, MoE semantics | Deterministic reference/scaffold executor |
| `gpt-oss-conformance` | core, model runner, MoE semantics, reference, runtime plan | Differential/scenario harness |
| `gpt-oss-evidence` | none | Stable manifests, campaign index, redaction, diagnostics |
| `gpt-oss-engine` | core, CPU kernels, GPU, KV, model runner, runtime plan, tokenizer | Optional `cuda`, `cublaslt`, `amx-int8`, `xe` |
| `gpt-oss-tokenizer` | core | Hugging Face tokenizer plus Harmony seam |
| `gpt-oss-server` | engine stack | Binary `gpt-oss-rs`; default `xe`, optional CUDA/AMX/cuBLASLt |
| `gpt-oss-bench` | runtime/evidence stack | Nine binaries including `cpu_parity`, `cpu_validation`, `live_cuda_parity`, and restricted trace/diff tools |

**Verified:** CUDA is optional throughout. The default GPU crate is a mock
backend; a successful default workspace test is not a CUDA inference proof.
Conversely, the supplied CUDA feature validation and 43 real CUDA tests prove
those bounded surfaces, not full-model generation.

## Supplied workstation-readiness evidence

The following is **Verified from the supplied sanity record**, not rerun here:

- Locked workspace check and tests passed.
- 45 Python unit tests and Markdown link validation passed.
- CPU scalar, AVX2, AVX-512 VNNI, and AMX feature gates built/validated in their
  configured lanes. This is build/test evidence, not installed-CPU ISA evidence.
- 43 real CUDA tests ran on both RTX 3090s, including cuBLAS execution.
- The CUDA release build produced all 29 PTX modules for `sm_86`.
- Strict configured Clippy lanes passed, including mock and CUDA GPU Clippy.
- Both Dockerfiles passed static validation; no large image was built.
- `/dev/nvme1n1` remained read-only and unmounted.

**Verified by cheap local artifact inspection:** `target/release/gpt-oss-rs` is
24,655,536 bytes (the supplied “24 MB” is a rounded description); 29 CUDA
sources and 29 release PTX files exist; `gpt_oss_moe.ptx` declares `.target
sm_86`.

## Documentation and evidence authority

| Status | Repository evidence | Consequence for `het` |
|---|---|---|
| **Authoritative** | `docs/CPU_FRESH_ORACLE_CAMPAIGN.md` says it is the only authoritative CPU parity procedure. | Earlier official-oracle/C3/seven-scenario/28-cell captures are retired and cannot support new counts or claims. |
| **Pinned identities** | The campaign fixes base ancestor `f86674d6acf17484899f5d17e286dcb2c6d1f850`; official `gpt-oss` v0.0.9 revision `599476783c6f88508dab8577808b5ead5cbee8d2`; source archive SHA-256 `7306d68ae017f461f2ebb82d04628f8dcba7cc7b431ef28e8786c947510c6f6b`; model revision `6cee5e81ee83917806bbde320786a8fb61efebee`; Python 3.12.12; CPU-only PyTorch 2.12.1; SafeTensors 0.8.0; llama.cpp `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`; private image `ghcr.io/gracee3/gpt-oss-rs-cpu-oracle` by immutable `name@sha256` digest only. | A later parity campaign must use the complete identity and lock procedure, never an abbreviated hash or mutable image tag. |
| **Historical/retired** | `docs/CPU_FRESH_ORACLE_CLOSURE.md` is a candidate-A closure and explicitly predates the current fresh campaign. | Useful for lineage only, not current parity totals. |
| **Completed/frozen** | `docs/TIGER_LAKE_CLOSURE.md` closes the Tiger Lake phase. It promoted diagnostics/profiling and bounded Xe residency machinery, but no broad M>1 CPU or full-model Xe default. | Retain evidence and interfaces; no new Xe optimization in this phase. |
| **Canonical CPU API** | `docs/MXFP4_MATRIX_API.md` fixes `Mxfp4MatmulProblem`, borrowed views, caller-owned scratch, scalar reference, and capability dispatch. | Preserve this seam; do not reinterpret it as a heterogeneous design. |
| **Closed, branch-scoped conflict** | The supplied policy says the fused-linear lane is closed. The detailed synthesis exists only on non-ancestor remote ref `origin/docs/fused-linear-addmm-source-attribution-closure` (`596f50d...`), not current `main`. It records 238 candidates, no global policy, and `stop_policy_lane_preserve_official_api_seam`. | Preserve the official Torch `linear`/`addmm(bias,input,weight.T)` API oracle seam. Do not reopen CUDA mirror or consumer revalidation. Resolve archival visibility separately; do not silently call the remote document current-main policy. |
| **Completed, branch-scoped** | The supplied policy says Harmony is complete/archived. Non-ancestor `origin/protocol/harmony-parity` (`fddc5c...`) calls it quiet maintenance; current main contains the active seam in `gpt-oss-tokenizer/src/protocol.rs` and server routes. | Reopen only a concrete regression or contract defect. A live semantic-control gap is validation work, not a protocol redesign invitation. |
| **Conflict** | Main closure documents report HTTP 200/one-token liveness. The supplied latest evidence says responses were parse-invalid, empty/unusable, or malformed. Current routes do call Harmony parsing and compute usage, but no retained Harmony-native semantic/token-accounting control was found for this exact host. | Serving is live at most; it is not semantically validated. Resolve with a Harmony-native response parse plus prompt/completion token accounting capture. |

## Existing evidence machinery

**Verified:** `gpt-oss-evidence/src/lib.rs` provides stable JSON,
`RunManifestV1`, immutable artifact hashes, campaign identity/indexes, strict
oracle identity validation, redaction, and byte-capped diagnostics.
`CpuPrefillTrace` and `CpuLayerTrace` in `cpu_runner.rs` capture router logits,
selected IDs/weights, each selected expert's gate/up, SwiGLU, down and weighted
outputs, layer output, final norm, and top logits. `cpu_parity` captures prefill
or a selected retained decode step; `compare_cpu_parity.py` reports generated
token first divergence and earliest traced stage mismatch.

**Verified:** `docs/CPU_EXECUTION_PROFILING.md` and the CPU profiler record
phase, layer/operation, matrix dimensions, exact expert-bucket `M`, context,
dispatch/backend, preparation/residency/fallback state, transaction state, and
scratch/resident high-water. They intentionally omit expert ID and tensor data.
The generic evidence schema can carry future artifacts and timers, but has no
existing placement/transfer-specific record contract.

## Existing CUDA warning inventory (not fixed here)

**Verified from the existing release fingerprint outputs, without rebuilding:**
the experimental CUDA release path emitted 21 model-runner and 17 engine
warnings. They group as follows:

- `gpu_loader.rs` / `gpu_weights.rs`: unused imports and a path parameter.
- `gpu_layer.rs`: unused cuBLAS/cuBLASLt/context parameters and unused helper
  functions.
- `gpu_runner.rs`: unused layer-count parameters, a private-interface warning,
  and unused fields/methods.
- CUDA attention/RoPE modules: retained but unread context/stream fields.
- `gpu_worker.rs`: unused imports, variables, mutable bindings, KV/cache fields,
  and old forward helpers.
- `async_gpu_engine.rs` / `gpu_engine.rs`: unused assignment/mutability and dead
  configuration/helper paths.

The exact retained warning records are under `target/release/.fingerprint/`.
They are inventory evidence only; this assignment does not repair them.

## Protected and non-source areas

Do not casually modify:

- `target/`: ignored Rust/CUDA outputs, including the current release binary and
  PTX evidence.
- `.live/` and `results_*.json`: ignored live/diagnostic outputs.
- `oracle/`: the Dockerfile, immutable CPU-oracle lock, requirements lock,
  generator, and negative tests.
- `crates/gpt-oss-bench/fixtures/` and `tools/xe-research/fixtures/`: checked-in
  test/evidence inputs.
- external model paths under `/data/models/`; the 20B tree also contains
  transformed, original, Metal, and Git-LFS data representing substantial
  duplicate storage.
- the standard CPU repack cache (`$GPT_OSS_RS_CACHE` or
  `/home/emmy/.cache/gpt-oss-rs`); the latter did not exist at capture time.
- any external campaign artifact root. `/home/emmy/gpt-oss-rs-artifacts` did
  not exist at capture time, so there is no local retained capture proving a
  prior run on this host.
- `/dev/nvme1n1`, which is protected, read-only, and unmounted.

There is no checked-in `vendor/` directory. Ignored Python `__pycache__` data
already exists and remains untouched.

## Relevant unresolved scaffolding

**Verified source TODOs/placeholders:** `attention/flash_attention.rs` is a
stub; `attention/paged_attention.rs` retains a CUDA placeholder;
`model-runner/src/bridge.rs` has unresolved allocator/buffer/stream/KV/weight
unification TODOs; `engine/src/executor/multi_gpu.rs` is a separate mock/NCCL
stub; `gpt-oss-gpu` memory-pool swap code contains no-op/mock transfer
placeholders; batched cuBLAS is a TODO; OTLP export is a stub. These do not by
themselves block finding a narrow heterogeneous MoE seam, but none should be
mistaken for production capability.

**Deferred:** speculative decoding, generic backend cleanup, bridge
unification, and the other unrelated placeholders are outside `het` unless a
later selected design demonstrates a direct dependency.
