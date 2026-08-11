# C3: Numerical Closure and Trusted-Mode Policy

- Outcome: **narrow experiment warranted**
- Scope: numerical localization and candidate trust contracts only
- Corpus gate: `E1-NEG-001: unavailable`
- Source budget used: current repository and the two pinned official GPT-OSS
  source roles (NX-SRC-009 and NX-SRC-011)

## Objective, questions, and non-questions

C3 asks two separate questions. C3-N asks where the remaining documented
BF16 difference first appears and what evidence would close it. C3-T asks
what exact configuration may be called trusted, how fallbacks inherit no
unearned coverage, and when startup or a request must be rejected.

C3 does not enable trusted CPU service, relax an oracle tolerance, declare
token parity to be tensor parity, select a production dense kernel, refresh an
oracle, or launch another model-scale run. It also does not inspect PyTorch
source: the allowed corpus did not identify a single PyTorch operator and
matching oracle revision precisely enough to justify that checkout.

## Evidence gate and current baseline

- **C3-E-001 / CURRENT-REPO FACT:**
  `docs/cpu-agent-coordination/i7.md` records that corrected explicit BF16
  boundaries restored the pinned `harmony_262` greedy sequence. It then records
  an earliest `1e-5` trace difference before the experts: one layer-0 attention
  context BF16 value differed by `0.00048828125`; a temporary diagnostic traced
  that difference to 9 K and 11 V values out of 65,536 each and attributed the
  values to the existing RMS/dense reduction order. Trusted mode remained
  blocked.
- **C3-E-002 / CURRENT-REPO FACT:**
  `docs/CPU_I7_CONFORMANCE.md` defines the trace order and says to correct the
  earliest mismatch. It also explicitly says that exact-BF16 expert projection
  is diagnostic-only and that an unclosed trace difference must be retained
  even when official greedy tokens pass.
- **C3-E-003 / CURRENT-REPO FACT:**
  `crates/gpt-oss-model-runner/src/cpu_runner.rs::{project_bf16,
  project_bf16_batch,attention_one_staged}` accumulates dense dot products in
  FP32 and introduces explicit BF16 conversions at model boundaries. Dispatch
  can select scalar, AVX2, or AVX-512/VNNI paths, so a source-level operation
  name is not an effective numerical configuration.
- **C3-E-004 / EXPERIMENT STATUS:** the 2026-08-11 gate found no new owner
  benchmark/oracle manifest, raw tensor pair, repetitions, or hashes in
  `/data/models/openai/gpt-oss-rs-cpu-work/results`. NX-ART-001 lacks the E1
  command, host, repetition, and boundary-payload record. The historical
  localization is therefore advisory in this research phase; it cannot prove
  a dot-product lane or enable trust.

The smallest recoverably documented boundary is consequently **the layer-0 K
and V dense-projection output feeding attention, after its RMS-normalized
input and before cache consumption**. The evidence does not localize the
difference to an input normalization operation, a specific reduction lane, a
BF16 conversion, or a library instruction. Claiming any of those would exceed
the retained record.

## Official semantic anchor

### C3-E-005 / LOCAL-SOURCE OBSERVATION

- Question: which model boundaries must a diagnostic preserve?
- Source: NX-SRC-009, official GPT-OSS research checkout
- Pin/path: `7b583341...`; `gpt_oss/torch/model.py::{RMSNorm,sdpa,
  AttentionBlock,MLPBlock}`
- Observation: RMS normalization computes in float and returns the input
  dtype; attention projections and model blocks use BF16 tensors; SDPA
  explicitly expands GQA keys/values, adds the sink denominator, applies
  softmax, and contracts values; MoE routing uses sorted `topk`.
- Implication: a useful comparison must name the exact tensor boundary and
  dtype rather than compare a later logit or compensate downstream.
- Limitation/conflict: readable PyTorch code specifies model semantics but not
  the reduction order of the matching installed operator implementation.
- Confidence: high for model boundaries, low for kernel-order equivalence.

### C3-E-006 / LOCAL-SOURCE OBSERVATION

- Question: which oracle revision owns the blocking fixture?
- Source: NX-SRC-011, distinct official oracle checkout
- Pin/path: `7802bf263f902efd4c7d18fcceff3ba72f941e80`; fixture and model
  implementation used by `docs/CPU_I7_CONFORMANCE.md`
- Observation: the oracle checkout is deliberately separate from the newer
  readable-research checkout.
- Implication: an oracle capture must name this exact source role and its
  installed dependency versions; the readable checkout cannot silently replace
  it.
- Limitation: no new raw capture or environment manifest accompanied the
  corpus gate.
- Confidence: high.

## C3-N: exact future diagnostic

**C3-Q-001:** which first scalar contribution or rounding boundary makes the
native layer-0 K/V projection differ from the pinned oracle?

**C3-X-001** is the only warranted experiment. It is recorded, not executed:

1. Use the already pinned prompt and the first affected row from an
   owner-supplied E1-complete failure manifest. Capture the RMS-normalized
   projection input, the exact K or V weight row, bias if present, and both
   outputs as BF16 bit patterns. Stop at the first unequal element.
2. On the manifest's exact repository build, evaluate only that dot product
   with forced scalar, AVX2, and AVX-512/VNNI implementations. Record every
   effective dispatch decision; unavailable forced paths are `unsupported`.
3. Capture the matching oracle tensor at the same named boundary using the
   manifest's exact NX-SRC-011 environment. Do not substitute NX-SRC-009.
4. Progressively shorten the reduction prefix and record the FP32 accumulator
   bits immediately before every specified BF16 boundary. A first differing
   prefix localizes the reduction-order issue; equal FP32 with unequal BF16
   localizes conversion instead.
5. Repeat the isolated operator enough times to demonstrate determinism and
   retain command, host snapshot, inputs, outputs, and hashes under E1. This is
   a small boundary probe, not a 20B generation or benchmark campaign.

Acceptance is an identified first arithmetic boundary with bit-reproducible
inputs on both sides, or a well-formed negative result that demonstrates the
existing trace cannot reproduce. Greedy-token agreement, a later tensor
match, or a tolerance-only explanation does not close C3-Q-001. A sparse
PyTorch checkout becomes permissible only if this evidence names the exact
operator implementation and matching version; until then the source question
remains unresolved.

## C3-T: trusted-evidence state model

Correctness and performance are independent. A path can be correctness-trusted
without being preferred, or fast without being trusted. The state for one
configuration-specific coverage cell is:

```text
unseen -> exercised -> correct -> trusted
   |          |           |
   +------> unsupported   +--> rejected
              |
              +----------> insufficient_evidence
```

- `unseen`: no valid manifest covers the cell.
- `exercised`: a valid execution exists, but the acceptance set is incomplete.
- `correct`: all specified tensor/token/lifecycle assertions passed for the
  cell and its artifacts are recoverable.
- `trusted`: `correct` plus every reachable fallback and required failure case
  for the requested service configuration is covered by an approved evidence
  set.
- `unsupported`: the capability or declared descriptor excludes the cell.
- `rejected`: valid evidence contradicts correctness or safety.
- `insufficient_evidence`: observations are usable but cannot support the
  proposed scope.

States do not roll upward across shapes, tails, thread counts, host permission
states, fallbacks, model hashes, or service modes. A passing automatic run
records each effective path it actually used; it gives no coverage to paths it
did not reach.

## Candidate configuration-specific trusted-evidence tuple

```text
TrustedEvidenceKey {
  evidence_schema,
  repository_commit, cargo_lock_sha256, build_profile, feature_set,
  model_revision, model_file_hashes, tokenizer_hash, template_hash,
  oracle_role, oracle_revision, oracle_environment_hash,
  service_contract_version, delivery_mode, numerical_mode,
  host_arch, os_kernel_class, isa_bits, xstate_permissions,
  physical_cores, allowed_cpu_set, numa_policy,
  thread_count, affinity_policy, nested_parallelism_policy,
  operation_class,
  effective_backend, dispatch_plan_hash,
  preparation_format, preparation_version, preparation_source_hashes,
  shape_region: { m, n, k, stride_classes, tail_classes,
                  sequence_lengths, absolute_position_region,
                  expert_bucket_sizes },
  reachable_fallbacks: [FallbackEvidenceKey],
  workload_set_hash, assertion_set_hash, artifact_manifest_hash
}
```

Numeric fields may name finite reviewed regions, but no wildcard means "all."
A range is valid only when boundary, interior, tail, and dispatch-transition
cases justify interpolation. `effective_backend` is observed, not requested.
Preparation identity includes source tensor hashes so a repack is not trusted
for different weights. The assertion set distinguishes exact BF16 bits,
bounded tensor comparisons, exact greedy tokens, cache invariants, and service
lifecycle outcomes; one cannot silently substitute for another.

**C3-D-001 / PROVISIONAL DECISION:** the project should maintain evidence for
cells keyed by the tuple above and derive configuration eligibility from them.
It should not persist a single `cpu_trusted=true` program or model flag.

## Startup, request, and fallback requirements

1. Startup resolves requested configuration into an effective capability and
   dispatch snapshot, enumerates every predictable path and fallback reachable
   under its declared shape/resource envelope, then queries exact evidence
   cells. Trusted startup rejects if any reachable cell is rejected, unseen,
   stale, or only exercised.
2. Request validation maps bounded request properties to covered shape and
   lifecycle regions before admission. A request outside them is rejected
   before model mutation, or is assigned a reference fallback whose own tuple
   is trusted for that region.
3. A forced mode rejects before execution when unavailable or uncovered. It
   never silently becomes automatic.
4. A dynamic kernel failure may fall back only before externally visible
   mutation, only to an independently trusted cell, and only with a stable
   reason code. Otherwise the request fails and C1 performs deterministic
   cleanup.
5. Adding a thread count, tail shape, preparation version, ISA permission,
   fallback, or operator changes the tuple and invalidates inherited trust.
6. Diagnostic-only paths such as exact-BF16 expert projection cannot become
   trusted serving fallbacks merely because they aid oracle localization.

## Alternatives considered

| Alternative | Finding |
| --- | --- |
| Trust exact greedy sequences for pinned prompts | Rejected. It is a valuable assertion but does not cover operator boundaries, fallbacks, shapes, or service failure behavior. |
| Trust a named backend/model globally | Rejected. Requested names conceal effective dispatch, tails, preparation identity, threads, and fallback reachability. |
| Trust finite configuration-specific evidence cells | Retained. It is auditable and makes uncovered execution rejectable without claiming universal coverage. |

The source lane stops here: the retained evidence distinguishes the trust
models but cannot discriminate the precise BF16 arithmetic cause.

## Failure modes and focused future tests

- A supposedly forced path dispatches elsewhere; assert requested and
  effective modes separately.
- A covered main kernel reaches an uncovered tail or scalar fallback; force
  every reachable fallback and assert exact cell lookup.
- Thread/affinity changes alter reduction order; cover each supported tuple and
  reject unlisted counts.
- A repack is reused with different source tensors; corrupt or swap an identity
  hash and reject before readiness.
- A request crosses a sequence, absolute-position, or expert-bucket boundary;
  reject pre-admission or use an independently covered reference cell.
- Startup capability changes after AMX permission or CPU-set resolution;
  perform trust lookup after the final effective snapshot.
- Oracle provenance is incomplete or raw output is missing; classify the cell
  `invalid` or `incomplete`, never correct.
- Main-path success masks owner failure, cancellation, staged-KV isolation, or
  delivery failure; include C1 lifecycle and C2 reservation assertions in the
  relevant service-contract evidence set.

Minimum future coverage includes scalar/AVX2/AVX-512 forced and automatic
dispatch, every tail class that can fall back, M=1 and batched rows, empty and
skewed expert buckets, BF16 half-way/NaN/infinity boundaries, attention
context/sliding/staged-KV edges, thread counts, forced-unavailable rejection,
and pre/post-commit cancellation. These are test dimensions, not a claim that
the current repository passes them.

## Risks, conclusion, and gate

The largest risk is accidental scope inflation: a token-level success or one
host can be worded as general CPU trust. The second is provenance drift between
the readable official source and the pinned oracle. The third is silently
executed fallback code outside the evidence cell.

The trust contract is sufficiently defined for later planning, but numerical
closure is not. C3 as a whole is therefore **narrow experiment warranted**.
It can become planning-ready only after an owner-supplied E1-complete artifact
allows C3-X-001 to identify or disprove the first arithmetic boundary and the
result is reflected in configuration-specific evidence. CPU trusted mode
remains rejected in the meantime.
