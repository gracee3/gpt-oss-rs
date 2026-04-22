# Workstream Runtime TODO

Branch/worktree:

- `feature/runtime-forward`
- `~/openai/worktrees/runtime-forward`

Purpose:

- isolated runtime/semantic implementation work
- forward engineering not yet ready for mainline

Current status:

- current tip: `a8248b7`
- clean extraction refs preserved separately:
  `extraction/conservative-semantic-cache` (`992a741`),
  `extraction/rope-scaling-parser-hardening` (`17396e2`),
  `extraction/rope-scaling-carry-forward` (`2c195e3`),
  `extraction/modelrunner-default` (`6ae2c95`),
  `extraction/modelrunner-architecture-fixture-cleanup` (`1b56c4b`),
  `extraction/modelrunner-conformance-fixture-carry` (`ade3bec`)
- auxiliary proof refs:
  `proof/sink-visibility` (`891a487`)
  `runtime/bd49d35-live-smoke` (`fa089a7`)

Immediate next steps:

- keep runtime-forward intentionally separate from aligned mainline branches
- validate any candidate fix against the Tier-2 contract before proposing promotion
- mine archived replay/debug branches selectively when specific prior evidence is needed
- record any new runtime hypotheses as explicit TODOs rather than implied settled behavior
- keep the new semantic KV-layout projection additive-only; do not wire it into default runtime/cache behavior from this lane
- treat `WorkerConfig::semantic_cache_layout()` as the safe extraction candidate
- treat `WorkerConfig::semantic_cache_layout_with_flavor(ExperimentalSinkAware { .. })` as runtime-forward only until sink semantics are validated for promotion

Current runtime-forward progress:

- branch tip `a8248b7` is a status/docs update (`Document last-position narrowing seam`) layered after the still-isolated runtime stack
- branch-local conservative semantic projection currently appears as `ee1604a` (`Add conservative semantic cache projection helpers`)
- sink-aware semantic projection remains branch-local as `838d3f8` (`Keep sink-aware semantic cache projection experimental`)
- preserved clean extraction refs now carry the safer additive sub-slices:
  - `17396e2` (`Harden rope-scaling config parsing`)
  - `2c195e3` (`Carry rope scaling config fields through worker configs`)
  - `6ae2c95` (`Add ModelRunnerConfig default`)
  - `1b56c4b` (`Add ModelRunnerConfig default for architecture tests`)
  - `ade3bec` (`Carry ModelRunnerConfig rope fields through conformance fixtures`)
- older branch-local helper attempts such as `f0afeb9` and `502fe6b` remain historical context only; prefer the preserved clean extraction refs when promoting

Targeted validation:

- `cargo test -p gpt-oss-engine worker::config -- --nocapture`
- `cargo test -p gpt-oss-model-runner architectures::gpt_oss -- --nocapture`
- `cargo test -p gpt-oss-engine --features cuda read_model_config_accepts_legacy_rope_scaling_type_key -- --nocapture`

Guardrails:

- no default compare behavior changes
- no promotion to aligned branches without validation
- no “real runtime bug” claim without same-input local replay surviving
- no sink-aware semantic projection promoted as settled behavior without additional evidence

Safe future extraction candidates:

- `extraction/conservative-semantic-cache` (`992a741`) for additive `WorkerConfig::semantic_model_spec()` plus conservative `semantic_cache_layout()`
- `extraction/rope-scaling-parser-hardening` (`17396e2`) for additive `rope_scaling.type` parser acceptance plus bounded config-field preservation coverage
- `extraction/rope-scaling-carry-forward` (`2c195e3`) for inert rope-scaling metadata carry-forward through worker configs and restricted bench constructors
- `extraction/modelrunner-default` (`6ae2c95`) for additive `ModelRunnerConfig::default()` support plumbing
- `extraction/modelrunner-architecture-fixture-cleanup` (`1b56c4b`) for the smallest architecture-test fixture cleanup that honestly restores `architectures::gpt_oss` coverage on top of `6ae2c95`
- `extraction/modelrunner-conformance-fixture-carry` (`ade3bec`) for the one-file conformance fixture carry-through needed to keep the extracted `ModelRunnerConfig` field additions test-compile-complete

Intentionally experimental / branch-local:

- `838d3f8` for `WorkerConfig::semantic_cache_layout_with_flavor(ExperimentalSinkAware { .. })`
- GPT-OSS YaRN table-generation behavior itself; config parsing is safer than the runtime semantic claim, so keep the actual table behavior labeled unsettled until stronger same-input validation exists

Frontier blocker memo:

- Primary next extraction frontier remains runtime semantics, not parser/config plumbing.
- `838d3f8` is blocked primarily on semantic-cache visibility/projection behavior.
- `bd49d35` is blocked primarily on YaRN runtime/table semantics.
- later docs/status commits through `a8248b7` do not change that blocker
- Neither branch-local area is blocked first on a harness gap alone; the missing proof is same-input semantic validation that distinguishes additive metadata carry-through from settled runtime behavior.

`838d3f8` audit:

- files/hunks involved:
  - `crates/gpt-oss-engine/src/worker/config.rs`
  - additive `SemanticCacheLayoutFlavor`
  - additive `semantic_cache_layout_with_flavor(...)`
  - sink-aware branch inside `semantic_cache_layout()`
  - sink-aware test `semantic_cache_layout_supports_explicit_experimental_sink_projection`
- current dependency chain:
  - builds directly on the already-extracted conservative semantic projection seam from `992a741`
  - depends on `semantic_model_spec()` exposing `SinkBehavior`
  - does not require new harness plumbing, but it does require confidence that `SinkBehavior::Available` plus an explicit `sink_tokens` count is an honest predictor of runtime KV visibility
- minimum honest proof before extraction:
  - same-input evidence that a sink-aware semantic projection matches the effective runtime/cache behavior for GPT-OSS sink tensors
  - bounded proof should compare conservative vs sink-aware projection against an observed runtime path, not just against model metadata
  - at minimum, one same-input case with nonzero sink tensors and one same-input case with zero/absent sink tensors
- smaller additive sub-slice extractable now:
  - no smaller slice beyond `992a741` is honestly extractable
  - the enum and helper method exist only to expose the sink-aware claim; shipping them without proof would still advertise unresolved semantics
- extraction status:
  - keep fully deferred and explicitly experimental

`bd49d35` audit:

- files/hunks involved:
  - `crates/gpt-oss-engine/src/gpu_engine.rs`
    - HF config parsing of YaRN fields and worker-config construction
  - `crates/gpt-oss-engine/src/worker/config.rs`
    - added rope/context fields and `model_runner_config()` carry-through
  - `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
    - default worker-config initialization for the new inert fields
  - `crates/gpt-oss-model-runner/src/runner.rs`
    - added `ModelRunnerConfig` rope/context fields
  - `crates/gpt-oss-model-runner/src/gpu_runner.rs`
    - `build_rope_tables(...)` and YaRN-specific RoPE table generation/concentration logic
  - `crates/gpt-oss-bench/src/bin/restricted_prefill_trace.rs`
  - `crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs`
    - restricted bench worker constructors carrying the same rope/context metadata
- current dependency chain:
  - the safe additive sub-slices from this area are already extracted as `17396e2` and `2c195e3`
  - the remaining runtime-forward piece depends on the full metadata chain being present in `ModelRunnerConfig`
  - the unresolved semantic step is specifically the GPU runner’s YaRN table construction and use, not the config acceptance or carry-forward plumbing
- minimum honest proof before extraction:
  - same-input runtime evidence that the YaRN table math in `gpu_runner.rs` reproduces expected GPT-OSS behavior
  - proof needs to localize first divergence boundaries with identical prompt/input data, not just compile/test success
  - minimum credible surface is a bounded same-input parity probe covering at least one case where YaRN scaling is active and one control case where it is inactive
  - restricted bench tools may be part of that proof workflow, but they are not by themselves proof of correctness
- smaller additive sub-slice extractable now:
  - yes, but those sub-slices are already extracted
  - safe already-extracted pieces are:
    - `17396e2` parser/config acceptance hardening
    - `2c195e3` worker-config/bench carry-forward of inert rope metadata
  - no additional smaller slice remains in `bd49d35` that is both additive and not a runtime-semantics claim
- extraction status:
  - keep the remaining YaRN table/runtime behavior fully deferred
