# Repository Alignment and Workstreams

## Mainline policy

`main` is the integration baseline. CPU kernel work should land as small,
reviewable patches with independent correctness and benchmark evidence. Avoid
long-lived branches that mix engine restructuring, ISA kernels, cache-format
changes, and model semantics in one diff.

The earlier April Tier-2 three-worktree topology is historical. Its CUDA
validation contract remains useful, but it is no longer the repository's
active branch map.

## Active workstreams

### CPU baseline and conformance

- preserve current scalar/SIMD correctness and seven-scenario token parity;
- keep generated traces, model files, repack caches, and benchmark output out
  of Git;
- update `CPU_RUNTIME.md` when serving or numerical invariants change;
- record host-specific long runs under `cpu-agent-coordination/`.

### MXFP4 kernel architecture

- own `gpt-oss-cpu-kernels`, kernel-facing packed-weight interfaces, and
  microbenchmarks;
- dispatch by capabilities and workload shape, never CPU product names;
- keep scalar, AVX2, AVX-512, and AMX patches independently reviewable;
- require before/after microbenchmarks for optimized paths.

### Runtime integration

- own model-runner selection of GEMV/GEMM operations and packed-weight
  lifetime;
- keep ISA-specific implementation details below the model-runner boundary;
- change cache format only with explicit versioning, integrity tests, and load
  and RSS measurements;
- defer CPU batching/NUMA scheduling until kernel thresholds are measured.

### CUDA validation

- retain the Tier-2 raw/runtime-emulated/same-input replay contract;
- do not conflate CUDA numerical investigations with CPU MXFP4 dispatch work;
- promote independently validated changes through normal pull requests.

## Branch and worktree rules

- Start feature branches from current `main`.
- Prefer names that describe the capability or outcome, for example
  `agent/mxfp4-avx512-gemv`, not a CPU generation.
- One worktree should own a file while a patch is active.
- Preserve user changes and generated evidence before switching branches.
- Do not commit model artifacts, repack caches, traces, Criterion output, or
  external reference repositories.
- Keep llama.cpp, mistral.rs, and ik_llama.cpp as sibling checkouts, not git
  submodules or vendored source, unless a later design explicitly chooses a
  dependency.

## Integration sequence

The intended patch progression is:

```text
baseline inventory and benchmarks
    -> feature model and kernel-plan API
    -> AVX2 audit/improvement
    -> AVX-512 MXFP4 GEMV experiments
    -> versioned packed GEMM layout
    -> standalone AMX prototype
    -> shape-based automatic dispatch
```

Each arrow is a review boundary. Benchmark results may change the order or
reject a proposed backend without forcing an engine redesign.
