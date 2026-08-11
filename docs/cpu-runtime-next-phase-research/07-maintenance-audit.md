# C7: Bounded Repository and CI Maintenance Audit

- Outcome: **planning-ready**
- Scope: repository-only inventory and bounded candidate slices
- Source budget used: current repository at NX-SRC-001 only
- Changes authorized by this charter: none

## Objective, questions, and non-questions

C7 identifies stale or conflicting status wording, compiler warnings, workflow
action/version surfaces, duplicate compatibility paths, naming friction, and
large-file movement candidates. It asks whether each item can become a small
maintenance-only slice with explicit exclusions and tests.

C7 does not fix any item, remove compatibility, refresh dependencies/actions
from the network, change behavior, reorganize code, or absorb C1/E1/C3/C4 work
under a maintenance label. Closeout edits that point the documentation index
and intake ledger at this completed research are research-record maintenance,
not execution of the candidate slices below.

## Method and baseline

The audit used `rg`, tracked-file inspection, Rust source line counts, route and
consumer searches, workflow inspection, and `cargo check --workspace --locked`
on 2026-08-11. No source or configuration file was edited.

- **C7-E-001 / CURRENT-REPO FACT:** NX-SRC-001 is the only source; the audit
  baseline is `a090bb0e81457e4302deb36d6e52a0847c14bfb0` plus the documentation
  intake checkpoint. External issue trackers and action registries were not
  consulted, so latest-version claims are intentionally absent.
- **C7-E-002 / READ-ONLY CHECK:** locked workspace checking succeeds but emits
  two `dead_code` warnings in
  `crates/gpt-oss-model-runner/src/architectures/gpt_oss.rs`: field
  `GptOssMlp::semantic_spec` and method `GptOssMlp::semantic_spec` are unused.
- **C7-E-003 / CURRENT-REPO FACT:** `.github/workflows/cpu.yml` is the only
  workflow. Its three jobs use `actions/checkout@v4` and
  `dtolnay/rust-toolchain@stable`; actions are tag/channel referenced rather
  than commit-SHA pinned. Workspace fmt/check/test, forced kernel tests, and
  AMX feature check/test/Clippy are present. Warnings-denied Clippy is scoped to
  the CPU-kernel crate's AMX job, not the whole workspace.
- **C7-E-004 / CURRENT-REPO FACT:** at audit time, `README.md` said both that
  CPU serving was batch size one/request batching unsupported and that opt-in
  multi-request scheduling was implemented elsewhere. Source and
  `CPU_RUNTIME.md` establish batch-one as the default profile and multi-request
  serving as experimental. The route list also omitted mounted response
  retrieval, batch, and tool alias routes; research closeout corrects both
  status surfaces without changing a route.
- **C7-E-005 / CURRENT-REPO FACT:** at audit time, `docs/README.md` and
  `CPU_RUNTIME_NEXT_PHASE_PRE_RESEARCH_LEDGER.md` described next-phase research
  as not started. The research closeout corrects those intake entry points
  outside the maintenance slices. By contrast, the older
  `CPU_RUNTIME_RESEARCH_AND_PREPLANNING.md` contains chronological “not started”
  and “completed” journal entries for M1-M5; that history is not stale text.
- **C7-E-006 / CURRENT-REPO FACT:** `server.rs::build_router` maps
  `/v1/chat/completions/tools` and `/tools` to the same chat handler as the
  canonical chat route. Server contract tests cover `/tools`. No current source
  comment defines an independent semantic reason or removal policy.
- **C7-E-007 / CURRENT-REPO FACT:** `CpuModelRunner` is a documented batch-one
  compatibility facade. It remains consumed by the parity binary, official
  tests, `CpuWorker`, and numerous runner tests; `CpuWorker` remains publicly
  re-exported. Neither is dead solely because server CPU serving uses the batch
  engine.
- **C7-E-008 / CURRENT-REPO FACT:** stable MoE top-k and softmax logic exists in
  both `gpt-oss-moe-semantics` and private `cpu_runner.rs` helpers. Their normal
  finite-input order agrees, but nonfinite validation and BF16 boundary
  ownership differ by caller. This is a candidate duplication, not proof that
  one function can be deleted.
- **C7-E-009 / CURRENT-REPO FACT:** largest relevant source files include
  `cpu_runner.rs` (3,731 lines), `routes/responses.rs` (2,812),
  `gpt-oss-cpu-kernels/src/lib.rs` (1,959), `matmul.rs` (1,507),
  `cpu_batch_engine.rs` (980), and `cpu_scheduler.rs` (950). Size alone is not a
  refactoring requirement.

## Boundary findings

Several conspicuous placeholders are deliberately not maintenance items:

- `/metrics` and unconditional health are E1/C1 semantic work.
- response/batch storage bounds and route lifecycle are C1/C2 work.
- attention, dense, MoE, and trust changes are C3/C4 work.
- speculative, beam, paged-attention, GPU bridge, multi-GPU, and CUDA TODOs are
  outside the current CPU maintenance scope and may represent intentional
  scaffolds.
- removing a public route/type or changing JSON compatibility is an API change,
  regardless of diff size.

This boundary prevents a “cleanup” patch from bypassing evidence and lifecycle
review.

## Candidate maintenance slices

### C7-S001: status and index consistency

- **Scope:** reconcile README batch-one/default versus experimental
  multi-request wording; distinguish core documented routes from explicitly
  retained aliases; point `docs/README.md` and the intake ledger at the durable
  research closeout.
- **Must not:** claim trusted mode, production readiness, complete OpenAI
  compatibility, new route behavior, or implementation authorization.
- **Verification:** compare every status statement with `CPU_RUNTIME.md`,
  `server.rs::build_router`, runtime-policy defaults, and the research outcome
  table; run relative-link checking and `git diff --check`.
- **Disposition:** planning-ready and appropriate for research closeout only to
  the extent it records the new research state. Broader README rewriting stays
  a separate documentation review.

### C7-S002: eliminate the two generic workspace warnings

- **Scope:** determine whether `GptOssMlp::semantic_spec` should be consumed by
  an existing validation path or removed with its unused accessor. Keep the
  `MoeSemanticSpec` construction/validation authority explicit.
- **Must not:** change MoE routing, storage acceptance, model loading, GPU
  behavior, or add `allow(dead_code)` merely to silence the result.
- **Verification:** locked workspace check with zero warnings; focused
  model-runner architecture/config tests; existing conformance tests; diff
  inspection proving no numerical path changed.
- **Disposition:** planning-ready, smallest code slice. Correct choice between
  use and removal requires tracing construction consumers during that slice.

### C7-S003: workflow action and toolchain provenance

- **Scope:** in a later network-authorized maintenance session, verify the
  current supported releases and commit SHAs for the two used actions; choose
  whether to pin immutable action SHAs and an exact Rust toolchain while keeping
  readable version comments/update policy.
- **Must not:** add jobs, change test scope, update Cargo dependencies, alter
  permissions, or treat moving `stable` as an E1-reproducible toolchain pin.
- **Verification:** inspect upstream release/security provenance, run every CPU
  workflow job on `agent/**`, and compare compiler/lock/capability snapshots.
- **Disposition:** **inconclusive on versions** in this repository-only audit,
  but the slice boundary is planning-ready. Current `contents: read` permission
  is already minimal for the observed workflow.

### C7-S004: route alias inventory and policy

- **Scope:** name `/v1/chat/completions` canonical and inventory
  `/v1/chat/completions/tools` plus `/tools` as aliases with consumer/test
  evidence. Decide retain/document versus a separately approved deprecation.
- **Must not:** delete a route, change handler behavior/errors, widen protocol
  compatibility, or combine this with C1 lifecycle implementation.
- **Verification:** router enumeration plus existing HTTP contract tests for
  each retained alias; any future removal requires owner/API authorization and
  a deprecation test.
- **Disposition:** audit planning-ready; removal deferred because `/tools` has a
  direct test consumer and no usage corpus was supplied.

### C7-S005: batch-one compatibility facade audit

- **Scope:** document consumers of `CpuModelRunner` and `CpuWorker`, separate
  parity/test use from server use, and identify methods whose only consumers
  are tests.
- **Must not:** remove public exports, migrate parity tooling, change canonical
  sequence ownership, or duplicate batch-engine behavior in the worker.
- **Verification:** `rg` consumer map, model-runner official/parity tests,
  focused worker tests, and API-semver review if a public item is proposed for
  later removal.
- **Disposition:** removal rejected as “cleanup” now. The consumer inventory is
  planning-ready for a future deprecation charter only if maintenance value is
  demonstrated.

### C7-S006: shared finite-input MoE semantics helper

- **Scope:** compare private `stable_top_k`/`softmax` with
  `gpt-oss-moe-semantics::{stable_top_k_indices,softmax_weights}` and, if exact
  input/error/boundary contracts agree, call one finite-input helper from the
  CPU route stage.
- **Must not:** move BF16 rounding, accept nonfinite router values, change ties,
  change route order, or refactor expert compute.
- **Verification:** equal logits, lower-index ties, `k=0/1/E/>E`, nonfinite
  rejection at the owner, BF16 weights, routing/unrouting order, and existing
  CPU model-runner tests.
- **Disposition:** narrow experiment warranted inside a maintenance patch; the
  current audit cannot prove all numerical boundaries are identical.

### C7-S007: mechanical module extraction

- **Scope:** at most one file per patch. Viable first candidates are moving
  self-contained attention helpers/tests from `cpu_runner.rs` after C4-C's
  contract is settled, or separating response-store/endpoints from SSE event
  projection in `routes/responses.rs` after C1's lifecycle is settled.
- **Must not:** alter visibility beyond the crate, rename serialized types,
  change allocation/order/numerics, combine both files, or redesign an API.
- **Verification:** zero semantic diff by focused tests plus full locked
  workspace test; compare public API and route lists; review move-only commits
  separately from follow-up edits.
- **Disposition:** deferred. Line count is weak evidence and both candidates
  currently intersect unresolved semantic charters.

## Naming observations

- `cpu_kernel` selects primitive/ISA dispatch while `cpu_matmul_backend`
  selects the MXFP4 matrix implementation. The distinction is real and should
  not be collapsed; help text and evidence snapshots must keep “requested” and
  “effective” qualifiers.
- `compatibility_kernel_path` in `CpuPrefillTrace` describes a separate
  compatibility dispatch field, but “compatibility” alone is ambiguous. A
  future diagnostic-schema revision may use `effective_compatibility_kernel`
  only if readers/tests migrate together.
- `ScopedTimer` comments that call `Instant` duration “wall-clock” conflict
  with E1 vocabulary. A comment/field wording slice may say elapsed/monotonic,
  but changing metric names belongs to E1 implementation.
- “legacy,” “compatibility,” “placeholder,” and “experimental” are not removal
  markers. Every candidate requires a concrete consumer and authority audit.

## Alternatives and decision

| Alternative | Finding |
| --- | --- |
| Combine cleanup with C1-C5 implementation | Rejected. It obscures semantic review and rollback. |
| Silence warnings and delete aliases/facades by name | Rejected. Two are live compatibility surfaces and warning intent needs a consumer decision. |
| Execute one bounded slice with explicit non-goals and focused tests | Retained maintenance policy. S001/S002 are the smallest candidates; others have evidence gates. |

## Risks and focused verification

Primary risks are deleting a live parity/API surface, moving numerical
boundaries while deduplicating helpers, and making CI less reproducible during
an action/toolchain “refresh.” Every slice requires `git diff --check`, relative
link verification when docs move, `cargo fmt --all --check`, locked workspace
checking/testing, and the focused tests named above. Workflow changes require a
green `agent/**` CPU run. Public routes/types require explicit owner authority.

## Conclusion

C7 is **planning-ready as a bounded audit**. It records seven independently
reviewable slices and fixes none. S001 is partially consumed by this research
closeout, S002 is the smallest future code candidate, S003 cannot choose
versions without a later authorized upstream check, S004-S006 require consumer
or numerical evidence, and S007 is deferred behind semantic work. No item is a
license for opportunistic cleanup.
