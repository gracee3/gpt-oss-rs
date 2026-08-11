# C1: Service, Lifecycle, and API

- Outcome: **planning-ready**
- Scope: candidate lifecycle/API contracts; no route or runtime changes
- Source budget used: current repository, llama.cpp, TGI, and official
  Tokio/Axum documentation

## Objective and bounded questions

C1 asks how one canonical CPU owner can remain responsive when clients are
slow or disconnected; when model, generation, and usage state become
authoritative; how byte-bounded delivery preserves event order; and how
readiness, failure, shutdown, storage, and every mounted route expose those
facts.

It does not promise broad OpenAI compatibility, resumable generation, hostile
multi-tenant isolation, authentication/TLS, HA, request recovery after restart,
or more than one sequence/candidate per request.

## Current repository baseline and route audit

`crates/gpt-oss-server/src/server.rs::build_router` mounts the following
surface. The table describes current behavior and the C1 contract obligation,
not an implementation change.

| Route | Current baseline | Candidate obligation |
| --- | --- | --- |
| `POST /v1/completions` | Streaming sends cumulative `co.text`; non-streaming retains the last cumulative output | Emit text deltas; terminal finish and usage remain ordered |
| `POST /v1/chat/completions` | GPT-OSS path derives Harmony content/tool deltas; handler owns another 32-entry item channel | Preserve role/content/tool/finish ordering under one byte budget |
| `POST /v1/chat/completions/tools`, `POST /tools` | Aliases to the chat handler | Same lifecycle/error semantics; no separate compatibility promise |
| `POST /v1/responses` | Diffs cumulative text; response handler has 16/32-entry channels; optional in-memory store | Store only eligible terminal responses; `store` never keeps generation alive after disconnect |
| `GET /v1/responses/:id` and `/input_items` | Reads an unbounded process-local map | Retrieval is bounded, process-local, and only for successfully stored terminal objects |
| `GET /v1/models` | Returns exact configured `model_name`, including a local path | Return a stable served-model ID distinct from a private source path |
| `POST /v1/batches` and three batch read/cancel routes | In-memory job store, sequential child generation, JSONL persistence; completion window is accepted but not enforced | Keep outside the initial interactive readiness claim until it uses the same admission/storage budgets |
| `GET /health` | Unconditional `200 ok` | Liveness only; add a distinct engine readiness surface if implemented |
| `GET /metrics` | Static placeholder | E1 exposition only after a recorder is actually installed |

- **C1-E-001 / CURRENT-REPO FACT:**
  `AsyncCpuBatchEngine::background_loop` owns the canonical engine and command
  processing, but awaits `channel.send(output)` after commit. One full 64-item
  request channel can stop scheduling and command draining.
- **C1-E-002 / CURRENT-REPO FACT:** `CpuBatchEngine::execute_inner` clones RNG,
  generation, and output, and `commit` publishes model deltas, prompt progress,
  sampling/RNG, output, revision, and lifecycle after validating all retained
  rows. Pre-commit cancellation drops a sequence's prepared delta.
- **C1-E-003 / CURRENT-REPO FACT:** `OutputProcessor::build_request_output`
  clones the original prompt, prompt tokens, cumulative generated text/tokens,
  and logprobs for every emitted token. Channel item counts therefore do not
  bound bytes and can retain quadratic cumulative-output traffic.
- **C1-E-004 / CURRENT-REPO FACT:** `InferenceEngine` exposes only `generate`.
  The CPU `shutdown` cancellation has no server-visible readiness status or
  awaitable owner join. Dropping admission response senders or the background
  task yields stringly scheduler errors.
- **C1-E-005 / CURRENT-REPO FACT:** all inference routes compare the request
  model with `AppState::model_name`; final smoke NX-ART-002 records cumulative
  completion chunks. `FinishReason::Abort` is mapped to API `stop`, losing the
  distinction.

## Source evidence cards

### C1-E-006 / LOCAL-SOURCE OBSERVATION

- Source: NX-SRC-002, llama.cpp `2468576...`
- Paths/symbols: `tools/server/server-task.h::{server_task,
  server_task_result_cmpl_partial,server_task_result_cmpl_final}` and
  `tools/server/server-context.cpp::{process_token,send_partial_response}`
- Observation: typed tasks/results separate task identity, slot execution
  state, partial results, final results, error results, and HTTP-side result
  consumption; partial text tracks the unsent suffix.
- Implication: result/delivery state need not own canonical sequence state, and
  partial payloads can be deltas.
- Conflict/limit: llama.cpp's slot breadth, polling queue, prompt cache, and
  compatibility surface are not local requirements. Its queues do not by
  themselves establish this project's byte bound.
- Confidence: high for separation, moderate for transfer.

### C1-E-007 / LOCAL-SOURCE OBSERVATION

- Source: NX-SRC-003, TGI `b4adbf2...`
- Paths/symbols: `router/src/infer/mod.rs::{Infer,InferError}` and
  `router/src/server.rs::{health,From<InferError>}`
- Observation: validation, a concurrent-request semaphore, explicit
  `Overloaded`, backend health, incomplete streams, and HTTP status mapping are
  separate concepts.
- Implication: overload is an expected admission result, not an internal engine
  error; liveness and readiness require engine evidence.
- Conflict/limit: TGI's unbounded SSE handoff in some paths is not evidence for
  this byte-bound design, and its sharded deployment is out of scope.
- Confidence: high.

### C1-E-008 / PRIMARY-SOURCE FACT

- Source: Tokio bounded `mpsc` and cancellation-safety documentation, plus Axum
  request-body limit and graceful-shutdown documentation, accessed 2026-08-11
- Observation: bounded `mpsc` capacity counts messages rather than encoded
  bytes; async sends exert backpressure on the sending task. Axum can reject
  bodies above a configured limit and graceful shutdown stops accepting new
  work but application tasks still need explicit completion/cancellation.
- Implication: byte accounting belongs above the channel; canonical execution
  must not await delivery; shutdown needs an owner join and waiter cleanup.
- Limitation: disconnect visibility and body buffering still require a local
  deterministic probe.
- Confidence: high.

## Candidate request/engine/delivery contract

These are three linked state machines, not one overloaded enum.

### Request and canonical-owner state

```text
Received -> EnvelopeValidated -> Tokenized -> AdmissionPending
                                      |              |
                                      |              +-> Rejected
                                      v
                                   Admitted -> Runnable -> InFlight -> Prepared
                                                           |           |
                                                           | cancel    | validate
                                                           v           v
                                                       Tombstoned    Committed
                                                                         |
                                    +------------------------------------+----+
                                    |                                         |
                                 Runnable                              TerminalCommitted
                                                                         |
                                               Completed | Length | Cancelled | Failed
```

Only the canonical owner may transition `Runnable/InFlight/Prepared/Committed`
or mutate KV, token history, sampling RNG, cumulative output, committed usage,
and finish reason. Reservation identifies expected revisions. Validation of
all retained deltas precedes any mutation. Dropping `Prepared` is a no-op.

Sampling, RNG advance, sampled token, output accumulator, KV delta, prompt
progress, and **computed/committed** usage publish together. A computed token
discarded before commit is never API usage. A committed token remains committed
even if it is never delivered.

### Delivery state

```text
Open -> Queued -> Flushing -> Delivered -> (Queued | TerminalDelivered)
  |        |          |
  |        |          +-> Abandoned(disconnect/write failure)
  |        +-> Abandoned(slow_consumer/budget)
  +-> Closed(pre-commit disconnect)
```

The canonical owner publishes `DeliveryEvent` with a nonblocking, byte-charged
operation. A delivery worker owns serialization and socket progress, but never
model progress. Delivery state records committed, enqueued, delivered, and
abandoned token/byte counts independently.

```rust
enum DeliveryEvent {
    TextDelta { choice: u32, text: Bytes },
    ToolDelta { choice: u32, call: u32, fragment: Bytes },
    Usage { committed_prompt: u64, committed_completion: u64 },
    Finish { choice: u32, reason: StableFinishReason },
    Error { failure: StableFailure },
    Done,
}

struct DeliveryLimits {
    per_request_queued_bytes: usize,
    global_queued_bytes: usize,
    max_event_bytes: usize,
}
```

Adjacent ordinary text deltas for the same choice may coalesce while queued.
They may not cross a role, tool-call, usage, finish, error, or done boundary.
Tool argument fragments preserve order and identity. Usage precedes the final
done marker when the selected protocol requires usage. Finish and done are
never dropped to make room. If coalescing cannot keep a publish under the byte
grant, delivery becomes `Abandoned(slow_consumer)` and the request is
tombstoned before the next commit. No other request waits for its socket.

### Disconnect and cancellation matrix

| Moment | Canonical action | Delivery/storage result |
| --- | --- | --- |
| Before admission | Drop envelope/tokenization work | No engine request or store entry |
| Admitted, not in flight | Tombstone and release reservation | Close delivery; stable internal `client_cancelled` |
| In flight, before commit | Let the bounded non-preemptible slice finish; discard that request's prepared delta | No computed state becomes usage/output |
| After one or more commits | Tombstone before next commit; never roll committed KV/RNG/output back | Count queued/delivered/abandoned separately |
| After terminal commit, before terminal delivery | No further generation exists | Stored terminal object remains eligible; delivery is abandoned |

Cancellation is idempotent. A caller cannot infer pre/post-commit cancellation
from whether an output channel exists.

## Readiness, failure, and shutdown

```text
Starting -> Ready -> Draining -> Stopped
    |         |         |
    +-------> Failed <---+
```

- `Starting`: process live, inference rejected as `not_ready`.
- `Ready`: model, snapshot, owner task, delivery/reservation services, and
  effective-runtime snapshot are available.
- `Draining`: liveness true, readiness false, new inference rejected; admitted
  work follows the configured drain deadline.
- `Failed`: liveness may remain true for diagnostics, readiness false; every
  pending admission and active delivery receives `engine_failed` where a
  transport still exists.
- `Stopped`: owner joined, grants released, delivery senders closed.

Shutdown transitions readiness before closing admission, then tombstones or
drains requests, stops future scheduling, discards prepared work, publishes
typed terminal failures where possible, releases C2 grants, closes stores, and
awaits the canonical owner. A panic/early owner exit is observed by a supervisor
that performs the same waiter cleanup. Clearing a sender map without terminal
reasons is insufficient.

## Stable failure and finish taxonomy

Stable machine codes are separate from messages and transport mapping.

| Class/code | Meaning | Candidate HTTP before headers |
| --- | --- | --- |
| `invalid_request`, `body_too_large`, `context_limit` | Malformed or bounded-input failure | 400/413 |
| `model_not_found`, `stored_response_not_found` | Unknown served ID or local stored object | 404 |
| `unsupported_option` | Valid syntax outside the declared service envelope | 422 |
| `overloaded_requests`, `overloaded_tokens`, `overloaded_memory`, `overloaded_delivery` | Admission budget unavailable; retry may succeed | 429 |
| `not_ready`, `draining` | Process live but not admitting | 503 |
| `engine_failed`, `owner_stopped`, `shutdown` | Canonical service unavailable | 503 |
| `execution_failed`, `serialization_failed` | Admitted request failed | 500 before headers; ordered error event after headers |
| `client_cancelled`, `slow_consumer`, `delivery_failed` | Delivery-side termination; never mapped to `stop` | Usually no new response after disconnect; diagnostic/internal terminal code |

Public finish reasons remain `stop`, `length`, and route-specific tool-call
completion where supported. Cancellation/owner/delivery failure is not a model
finish. Messages may evolve; codes, retryability, and phase do not.

## Served identity, usage, and storage

`served_model_id` is a configured stable alias. `source_model` path/revision is
private snapshot metadata and E1 evidence. `/v1/models` and request validation
use the alias; a filesystem path must not be an API identifier or metric label.

Usage exposed by successful APIs counts committed prompt and completion tokens.
Internal evidence also counts computed-discarded, committed-undelivered, and
delivered. Byte counters are delivery facts, not token usage.

Response storage has its own state:

```text
NotRequested | Requested -> StorageReserved -> EligibleTerminal
                                      |              |
                                      +-> Refused    +-> Stored -> Evicted
                                                     +-> StoreFailed
```

Only a successfully terminal `completed` response is eligible initially;
cancelled, failed, abandoned-before-terminal, and protocol-`incomplete`
responses are not. Storage occurs from canonical terminal data and is not
rolled back by a later socket failure. `store=true` does not keep generation
alive after disconnect. Entries and total bytes require C2 grants, deterministic
eviction or rejection, and process-local durability wording. Previous-response
history references only a fully stored entry.

Batch storage/persistence is a separate consumer of the same memory/disk
budgets; until integrated, batch routes are not evidence that interactive
admission is bounded.

## Alternatives and provisional decisions

- Await bounded per-request channels in the owner: rejected because item bounds
  do not bound bytes and a slow client stalls unrelated inference.
- Unbounded delivery tasks: rejected because they move the stall into memory.
- Roll back committed model state when delivery fails: rejected because it
  conflates canonical computation with transport and cannot undo client bytes.
- **C1-D-001:** one canonical owner plus nonblocking byte-charged delivery
  workers is the candidate architecture.
- **C1-D-002:** all routes use one stable failure/finish taxonomy and served
  alias, but broad protocol compatibility is explicitly not a goal.
- **C1-D-003:** an awaitable owner lifecycle is required for readiness and
  deterministic shutdown.

## Focused test strategy

Model-free scripted engines and paused Tokio time should cover: every mounted
route's model/error envelope; completion deltas; role/content/tool/usage/finish
ordering; coalescing; per-request/global byte overflow; one stalled client
while another finishes; disconnect before admission, during in-flight work,
after commit, and after terminal commit; cancellation idempotence; stale
prepared rollback; admission rejection; response storage eligibility and byte
failure; owner return/panic; all waiter cleanup; readiness transitions;
drain deadline; and shutdown join. Existing focused engine tests remain the
semantic backstop for revision, cancellation, and state isolation.

## Risks and conclusion

- Event byte charging must use owned encoded payload capacity consistently,
  not `size_of` or character count.
- Stop-string truncation can retract cumulative text; delta production needs a
  tokenizer/output contract that never sends bytes later retracted.
- Once streaming headers are sent, transport status cannot change; ordered
  error events and diagnostics must carry the terminal class.
- Body, tokenizer, tool schema, batch JSONL, and stored-history limits must be
  aligned with C2 or admission remains porous.

The ownership, delivery, cancellation, readiness, failure, shutdown, identity,
usage, storage, alternatives, and focused tests are explicit enough for a later
owner-authorized implementation plan. C1 is **planning-ready**.
