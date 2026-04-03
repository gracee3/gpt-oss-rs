# Runtime Frontier Status

Branch/worktree:

- `feature/runtime-forward`
- `~/openai/worktrees/runtime-forward`

Current frontier split:

- `838d3f8` is closed/deferred on the current runtime seam
- `bd49d35` is the only live semantic frontier

## `838d3f8` status

Decision:

- keep `838d3f8` closed/deferred

Why:

- same-input CUDA prefill boundary observation on an active sliding-sink case showed identical KV/cache visibility boundary metadata between:
  - the real nonzero-sink model
  - a zero-sink control clone
- the observed boundary values stayed the same:
  - `context_len0`
  - `seq_start_pos0`
  - derived sliding-window start
  - block-table preview
- only the sink tensor payload changed
- that is direct runtime evidence for extra sink-logit terms, not extra visible KV token offsets

What would be required to reopen it honestly:

- a new runtime implementation seam that explicitly carries extra visible KV offsets, not just sink logits
- or direct same-input runtime observation at a seam closer to the actual attention read set that shows extra prefix KV slots being consumed when sinks are active

Until one of those exists:

- do not treat `WorkerConfig::semantic_cache_layout_with_flavor(ExperimentalSinkAware { .. })` as promotable
- do not use nonzero learned sink tensors alone as proof of extra visible KV offsets

## `bd49d35` status

Current positive evidence:

- YaRN table-construction math matches the local HF/vLLM-style reference on the active GPT-OSS config
- a no-YaRN control case did not perturb runtime traces
- completed same-input sink-free `>4096` runtime evidence exists for:
  - post-RoPE `q/k`
  - post-attention context
  - post-attention residual
- retained-state `restricted_logit_diff` seam evidence now confirms the safe-side retained prefill reaches all of the following boundaries on the active `4096 + 1` case:
  - retained layer-0 late-token router traversal through `mlp_done`
  - retained layer-1 attention completion
  - retained layer-1 post-attention residual handoff
  - retained layer-1 `mlp_begin`
  - retained layer-1 `router_input_ready`

Current blocker:

- the retained-state seam still does not emit a continuation-token artifact in bounded time
- the retained-state seam still does not reach `decode1`
- there is still no safe/variant retained continuation comparison for the continuation token

Current retained-seam limitation:

- none of the retained-state results above should be treated as proof of full runtime correctness
- the retained-state findings are preserved here so future work can build on the confirmed boundaries even if the workflow pivots to a bounded live-smoke lane

Current bounded live-smoke surface for `bd49d35`:

- baseline smoke reference: `~/openai/gpt-oss-rs` at `95114af`
- clean candidate smoke tree: `~/openai/worktrees/runtime-live-smoke-candidate` at `abdc358`
- relative to that baseline, the current bounded smoke surface is exactly:
  - `crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs`
  - `crates/gpt-oss-bench/src/bin/restricted_prefill_trace.rs`
  - `crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py`
  - `crates/gpt-oss-engine/src/worker/config.rs`
  - `crates/gpt-oss-engine/src/worker/gpu_worker.rs`
  - `crates/gpt-oss-model-runner/src/architectures/gpt_oss.rs`
  - `crates/gpt-oss-model-runner/src/gpu_layer.rs`
  - `crates/gpt-oss-model-runner/src/gpu_runner.rs`
- exercised binary path for the next boundary smoke remains `restricted_logit_diff` with the worker/config and model-runner chain above
- the clean candidate reconfirms that the exercised binary paths exclude retained/proof instrumentation:
  - no `RETAINED_`
  - no `GPT_OSS_RETAINED_`
  - no `PROOF_`

Decision trigger after the next bounded live-smoke:

- resume retained-seam chasing only if the boundary smoke produces a decision-forcing mismatch such as:
  - baseline passes but candidate fails on the same `>4096` boundary case
  - candidate emits a malformed or partial artifact only near the boundary
  - candidate shows candidate-only decode or output corruption tied to the boundary
- otherwise the default action remains:
  - no new runtime code
  - let bounded live-smoke drive the next decision

Current boundary-smoke timing note:

- a scratch-only timing audit of `restricted_prefill_trace` on the same restricted model and `--max-model-len 4608` showed:
  - short control prompt:
    - about `35.2s` to `worker_ready`
    - about `24.0s` inside `prefill_trace`
    - about `20ms` total for final JSON serialization and output write
  - exact 4097-token boundary prompt:
    - about `34.8s` to `worker_ready`
    - then the bounded run spent the remaining wall-clock inside `prefill_trace` until timeout
- the heaviest observed stage is therefore the monolithic prefill-trace path itself, not final JSON serialization/output
- if a lighter honest boundary-smoke surface becomes necessary later, the smallest boundary-relevant change would be a surface that reuses the same worker/config path but exits after one localized boundary capture instead of materializing the full prefill trace
- default action still remains no new runtime code until the bounded live-smoke lane clearly justifies it

Current no-trace server smoke contract note:

- inspection says plain `POST /v1/completions` with a raw text prompt is not the right semantic contract for GPT-OSS in this tree; use the native Harmony-backed `/v1/chat/completions` or `/v1/responses` path for semantic smoke
- why:
  - `crates/gpt-oss-server/src/routes/completions.rs` forwards `req.prompt` directly to `engine.generate(...)` with no Harmony normalization
  - `crates/gpt-oss-server/src/routes/chat.rs` and `crates/gpt-oss-server/src/routes/responses.rs` both normalize inputs through `gpt_oss_tokenizer::HarmonyProtocol::gpt_oss().render_prompt(...)` before inference
  - `crates/gpt-oss-tokenizer/src/protocol.rs` defines the intended seam explicitly as the "Harmony-native GPT-OSS protocol seam" and says the server should normalize API requests into `ProtocolMessage`s and treat the resulting structured parse state as the source of truth
- serving runtime mode does not change that semantic contract:
  - `trusted` vs `experimental` in `crates/gpt-oss-server/src/runtime_policy.rs` selects backend/admission behavior, not prompt-format semantics
- interpretation:
  - a garbled short raw-completions output is only a liveness signal on this model/view in this tree
  - the semantically correct smoke surface is the Harmony/native request path, not plain raw completions

Current `/v1/responses` boundary note:

- the prior `/v1/responses` `500` is plausibly confounded by an undersized output budget
- reason:
  - for GPT-OSS, non-streaming `/v1/responses` always parses returned completion tokens through `HarmonyProtocol::parse_completion_tokens(...)` before building output items
  - the rendered prompt already ends with `<|start|>assistant`, so `max_output_tokens=4` is not about emitting an assistant header; it is about whether there are enough generated tokens to form a parseable assistant body or tool-call fragment
  - with only `4` output tokens, a Harmony parse failure or structurally useless partial completion is therefore plausible on inspection alone
- exact-boundary server smoke is currently truthful only if the Harmony-rendered prompt length is verified, because a simple string `input` is first normalized to a user message and then rendered through Harmony before inference
- that means a raw `4097`-token text input is not, by itself, a truthful statement about final model-input length on `/v1/responses`
- rendered-length verification is available from existing codepaths:
  - `CreateResponseRequest::normalize_input_items`
  - `render_conversation_protocol_items(...)`
  - `HarmonyProtocol::render_prompt(...)`, which returns both rendered text and token ids
- until that rendered token count is checked for the exact server request shape, the no-trace server surface is valid for semantic control/liveness but not yet for an exact-boundary claim

Current Harmony server route note:

- first-choice semantic short-control surface should be `POST /v1/chat/completions` with `stream=true` and no tools
- why:
  - it is Harmony-backed on input because `crates/gpt-oss-server/src/routes/chat.rs` maps request messages to `ProtocolMessage`s and renders through `HarmonyProtocol::render_prompt(...)`
  - it exposes usable assistant deltas before full completion through `StreamedChatChoiceState::ingest(...)` and `visible_text_from_protocol_messages(...)`
  - by contrast, `/v1/responses` with `stream=false` is currently more parse-bound because it waits for final completion output and then requires `HarmonyProtocol::parse_completion_tokens(...)` before building response items
- exact server-side rendered-count verification is possible from existing codepaths:
  - `/v1/responses`:
    - `CreateResponseRequest::normalize_input_items`
    - `render_conversation_protocol_items(...)`
    - `HarmonyProtocol::render_prompt(...)` returning `RenderedPrompt { text, token_ids }`
  - `/v1/chat/completions`:
    - `ChatCompletionRequest.messages`
    - per-message mapping to `ProtocolMessage::new(...)` in `crates/gpt-oss-server/src/routes/chat.rs`
    - `HarmonyProtocol::render_prompt(...)` returning `RenderedPrompt { text, token_ids }`
- frontier consequence:
  - exact-boundary no-trace server smoke is still truthful only after Harmony-rendered token count is verified
  - the blocker is not missing core render/count codepaths, but that current harness/tooling does not yet expose that rendered-count verification as a normal smoke step

Guardrail:

- none of the safe extraction commits or current proof seams should be treated as proof of full GPT-OSS runtime correctness
