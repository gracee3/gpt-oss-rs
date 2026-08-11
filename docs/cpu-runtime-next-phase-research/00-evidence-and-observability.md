# E1: Evidence and Observability Spine

- Outcome: **planning-ready**
- Scope: candidate evidence contracts only; no telemetry implementation
- Source budget used: current repository, vLLM, TGI, mistral.rs, and official
  Rust/Prometheus timing and metric guidance

## Objective, questions, and non-questions

E1 defines the minimum facts required to reproduce a run, explain an effective
runtime choice, observe a bounded service, and retain a negative result.
Questions are: which facts belong in always-on metrics, opt-in diagnostics, or
offline artifacts; how time and overhead are defined; and what makes evidence
usable for dispatch or trust decisions.

E1 does not select an observability vendor, add distributed tracing, expose
prompts, make request IDs metric labels, define performance thresholds, or
turn metrics into an execution architecture.

## Current repository baseline

- **E1-E-001 / CURRENT-REPO FACT:**
  `crates/gpt-oss-engine/src/telemetry/metrics.rs::{REQUEST_LATENCY,TTFT,ITL,...}`
  declares a GPU/vLLM-shaped metric vocabulary, while
  `crates/gpt-oss-server/src/server.rs::metrics_placeholder` returns a static
  string. Descriptions do not establish recording points or timer boundaries.
- **E1-E-002 / CURRENT-REPO FACT:**
  `worker/metrics.rs::{ScopedTimer,WorkerStepTimings}` uses monotonic
  `std::time::Instant`, but comments call the result wall-clock and the
  instrumentation is GPU-worker-specific.
- **E1-E-003 / CURRENT-REPO FACT:** `CpuPrefillTrace` records selected tensor
  payloads and dispatch strings only when requested. This is an appropriate
  diagnostic surface, but the surrounding run manifest is not one common
  schema.
- **E1-E-004 / CURRENT-REPO FACT:** startup logging in
  `gpt-oss-server/src/main.rs` records several requested/resolved settings, and
  CPU startup records topology and immutable dispatch plans. These facts are
  split across logs and result JSON rather than one capability snapshot.

## Source evidence cards

### E1-E-005 / LOCAL-SOURCE OBSERVATION

- Question: how should request phases and work be separated?
- Source: NX-SRC-003, TGI
- Pin/path: `b4adbf2...`; `router/src/server.rs`, request handlers and metric
  registration around `tgi_request_validation_duration`,
  `tgi_request_queue_duration`, and `tgi_request_inference_duration`
- Observation: validation, queue, inference, total, generated-token, queue-size,
  and batch facts are recorded independently.
- Implication: one total timer cannot diagnose admission or delivery pressure.
- Limitation: TGI's router/shard topology and broad compatibility are outside
  this service envelope; upstream describes TGI as maintenance mode.
- Confidence: high for the source observation, moderate for transfer.

### E1-E-006 / LOCAL-SOURCE OBSERVATION

- Question: may request identity be useful without becoming a metric label?
- Source: NX-SRC-005, mistral.rs
- Pin/path: `8010b6a...`;
  `mistralrs-server-core/src/metrics.rs::{request_context_middleware,InFlightGuard}`
- Observation: request IDs are used in access logs/headers while metric labels
  use bounded method/route/status dimensions.
- Implication: logs can correlate one request while metrics aggregate bounded
  behavior.
- Limitation: model label handling must be narrower here because configured
  local paths are unbounded and may reveal filesystem data.
- Confidence: high.

### E1-E-007 / LOCAL-SOURCE OBSERVATION

- Question: which lifecycle/work facts should an offline record preserve?
- Source: NX-SRC-004, vLLM
- Pin/path: `52be12c...`; `vllm/v1/metrics/stats.py` request/iteration stats and
  scheduler work accounting
- Observation: arrival, scheduled, first-token, finished, prompt/generated
  token, cached-token, preemption, and iteration work facts are distinct.
- Implication: queue, compute, commit, and client delivery milestones should
  not be collapsed into one timestamp or token counter.
- Limitation: GPU cache and distributed fields are not CPU requirements.
- Confidence: high for vocabulary, moderate for the bounded subset.

### E1-E-008 / PRIMARY-SOURCE FACT

- Question: what clocks and labels are safe?
- Sources: Rust `std::time::{Instant,SystemTime}` documentation and Prometheus
  data-model/naming guidance, accessed 2026-08-11
- Observation: `Instant` is monotonic for durations; `SystemTime` represents
  wall time and can move. Prometheus label combinations create time series, so
  unconstrained values create unbounded cardinality.
- Implication: durations use `Instant`; wall-clock timestamps are metadata only.
  Request ID, prompt, path, raw model name, arbitrary error text, expert ID,
  sequence length, and thread ID are forbidden metric labels.
- Limitation: an implementation still needs measured overhead and bucket
  selection on the real exporter.
- Confidence: high.

## Three evidence surfaces

### 1. Production metrics

Always-on metrics are aggregates with a fixed label schema. Candidate labels
are `route_class`, `delivery_mode`, `result_class`, `phase`, `backend_class`,
and a small enumerated `reason_code`. Model identity, request identity, exact
shape, and exact operation names live in snapshots/traces, not labels.

Minimum candidates:

- counters: envelopes, admission results, requests terminal by reason,
  committed/delivered/abandoned tokens and bytes, owner failures, delivery
  coalesces, reservation rejects/expansions, dispatch/fallback/rejection;
- gauges: ready/draining/failed state, admitted/runnable/in-flight requests,
  command and delivery bytes, reserved/used resource bytes;
- histograms: validation, admission wait, first commit, first delivery,
  inter-commit, inter-delivery, terminal latency, execute slice, commit, and
  delivery-stall durations; scheduled rows and prompt/decode composition.

Timers use monotonic time. Validation includes envelope parsing and bounded
semantic checks but excludes socket accept. Admission wait begins after
tokenization and ends at grant. Execute includes model kernels and required
worker-local preparation, excludes queue wait/commit/delivery. TTFC means first
committed output; TTFT means first byte-bearing client delivery. Terminal
latency ends at committed terminal state and client completion is separate.

Default overhead budget: disabled diagnostic payloads allocate no tensor
buffers; always-on instrumentation adds no per-token heap allocation and no
request-specific metric registration. A later implementation must demonstrate
less than 1% median throughput effect and less than 2% p99 request-latency
effect in a matched model-free or small-fixture A/B before always-on timing is
accepted. Failure to measure is `insufficient_evidence`, not zero overhead.

### 2. Diagnostic traces

Diagnostics are opt-in per process or bounded request sample. They may contain
request IDs and exact operation/shape facts, but prompt text, tokens, tool
arguments, environment values, and tensors are off by default. Payload modes
are `metadata`, `summary` (dtype/shape/min/max/nonfinite/hash), and explicit
`tensor`; tensor mode requires a named boundary, byte cap, output directory,
and redaction acknowledgement. Truncation is recorded, never silent.

Lifecycle traces carry monotonic offsets from a run-local origin plus an
optional wall-clock run start. Operator traces carry sequence-local row ID,
absolute position, operation class, effective backend, fallback reason, input
and output summaries, and a parent iteration ID. Production metrics never
parse diagnostic traces.

### 3. Offline benchmark/oracle manifests

Offline evidence is immutable, machine-readable, and references raw files by
absolute path plus SHA-256. A summary without its raw record cannot select
dispatch or establish trust.

## Candidate run manifest

```json
{
  "schema": "gpt-oss-rs.cpu-evidence/v1",
  "run_id": "opaque-local-id",
  "purpose": "correctness|benchmark|oracle|probe",
  "status": "pass|fail|unsupported|unavailable|invalid|incomplete|insufficient_evidence",
  "source": {
    "repository_commit": "40-hex",
    "dirty": false,
    "branch_role": "research|oracle|candidate",
    "cargo_lock_sha256": "64-hex",
    "toolchain": "rustc/cargo/LLVM",
    "profile": "debug|release|bench",
    "features": ["sorted", "feature", "names"]
  },
  "model": {
    "id": "stable alias, not a secret path",
    "revision": "content revision",
    "files": [{"role": "config|index|weight|tokenizer|template", "sha256": "64-hex"}],
    "repack": {"format": 1, "layout": "identifier", "source_hashes": ["64-hex"]}
  },
  "host_snapshot_sha256": "64-hex",
  "runtime_snapshot_sha256": "64-hex",
  "command": {"argv_redacted": ["..."], "environment_allowlist": {}},
  "workload": {"id": "stable-id", "prompt_sha256": "optional", "seed": 0, "repetitions": 1},
  "timers": [{"name": "execute", "clock": "monotonic", "includes": [], "excludes": []}],
  "artifacts": [{"role": "raw-output", "absolute_path": "/outside/git", "sha256": "64-hex", "bytes": 0}],
  "limitations": ["explicit statements"]
}
```

The private manifest may contain a host-local model path and command; the
publishable view replaces them with aliases and hashes. Environment collection
is allowlist-only. Tokens, credentials, proxy settings, home paths, hostnames,
and arbitrary `RUST_LOG` filters are not copied by default.

## Effective-runtime and capability snapshot

The immutable snapshot is emitted after validation, hardware permission
requests, model/repack resolution, and thread-pool construction, before
readiness becomes true:

```text
EffectiveRuntimeSnapshot {
  requested: { device, mode, kernel, matrix_backend, threads, context,
               concurrency, batched_tokens, prefill_chunk },
  effective: { backend, mode, compatibility_plan, matrix_backend, threads,
               affinity, context, concurrency, cache_layout },
  capability: { architecture, os, kernel, online_cpus, allowed_cpus,
                numa_nodes, isa_bits, xstate_permissions, allocator },
  identity: { repository, build, model, tokenizer, template, repack },
  decisions: [ { operation_class, eligibility, selected, reason_code,
                 possible_fallbacks } ],
  omissions: [ stable_reason_code ]
}
```

Requested and effective fields are never overwritten into one value. Capability
means observed; eligibility means all contract checks passed; selection means
the current path; fallback means a path that might actually execute. Unknown
is distinct from false.

## Negative-result taxonomy

| Status | Meaning |
| --- | --- |
| `pass` | Prespecified acceptance condition met |
| `fail` | Valid execution contradicted the acceptance condition |
| `unsupported` | Capability or declared contract excludes the case |
| `unavailable` | Required host, corpus, dependency, or artifact was absent |
| `invalid` | Inputs or provenance make the run unusable |
| `incomplete` | Execution began but the required artifact set is partial |
| `insufficient_evidence` | Valid observations exist but repetitions/coverage cannot support the claim |

Unsupported and unavailable are not correctness failures. Incomplete and
invalid cannot be averaged into passing repetitions. Negative results retain
the same manifest and artifact requirements as passes where applicable.

## Alternatives and decision

- Alternative A: one unified telemetry event stream. Rejected because
  production cardinality/privacy and diagnostic payload needs conflict.
- Alternative B: logs plus ad hoc benchmark JSON. Rejected because requested
  versus effective configuration, source roles, timer boundaries, and artifact
  integrity cannot be joined reliably.
- **E1-D-001 / PROVISIONAL DECISION:** retain the three evidence surfaces and
  common IDs/snapshots above. Metrics are bounded aggregates, diagnostics are
  opt-in, and offline manifests are authoritative for benchmark/oracle claims.

## Failure modes and focused test strategy

Tests for a later implementation must cover schema validation; sorted/stable
snapshot serialization; redaction; dirty-tree capture; monotonic duration
behavior across wall-clock changes; bounded label enumerations; disabled-trace
zero-payload allocation; trace truncation; artifact hash mismatch; missing raw
files; requested/effective mismatch; unsupported/unavailable/incomplete result
states; and uncovered fallback reporting. Model-free lifecycle probes should
establish metric transition counts and overhead before any model-scale A/B.

## Risks, questions, and conclusion

- **E1-Q-001:** histogram buckets require real observed distributions; no
  bucket set is selected here.
- **E1-Q-002:** the allocator identity and usable resident/private accounting
  fields depend on C2's chosen implementation.
- **E1-Q-003:** protocol-specific usage fields depend on C1, but computed,
  committed, delivered, and abandoned internal counters remain separate.

The contracts, privacy boundary, negative statuses, and test strategy are
sufficient for later implementation planning. Exact metric names, exporter,
buckets, and overhead results remain plan/implementation questions. Therefore
E1 is **planning-ready**.
