# Runtime-Forward Integration E2E Smoke Plan

Date: 2026-04-27

Current branch: `integration/mainline-alignment`

Current HEAD inspected: `dcc31c5b4e4b07f989f10c3552181856b9e94aee`

## Summary

Integration can validate promoted runtime candidates and validate the final-readout artifact
digest, but it does not yet have an exact integration-native final-token oracle smoke for
`developer-message-user-smoke`.

Existing integration surfaces can run model paths for a prompt string and can emit either
top-k logits or worker/runner logit comparison JSON. They do not currently accept the exact
proof-case token IDs, compute the known final-readout digest, or compare against the
runtime-forward final-readout artifact directly.

Classification: `integration_e2e_smoke_missing_needs_minimal_tool`

## Reference Case

- Case: `developer-message-user-smoke`
- Source proof branch: `feature/runtime-forward`
- Source proof commit: `5bcba1d2edcb9c15b1ed567700976dad03e12300`
- Source final-readout artifact:
  `.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json`
- Expected final logits digest:
  `67f31845dd24db26cc91954607cfae8ae7ff7b9c8954cb9d3b1610ca9c635209`

The source artifact is available locally at:

`/home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json`

The local model directory exists at:

`/data/models/openai/gpt-oss-20b-full-attn-restricted-integration`

## Available Execution Surfaces

| Surface | Command shape | GPU | Output | Oracle comparison | Notes |
| --- | --- | --- | --- | --- | --- |
| Final-readout validator | `python3 scripts/validate_runtime_forward_final_readout_status.py --artifact <artifact>` | No | Artifact classification, digests, top-20 status | Yes, artifact-only | Does not execute the model. Remains stdlib-only. |
| `restricted_prefill_topk` | `cargo run -p gpt-oss-bench --features cuda --bin restricted_prefill_topk -- --model <model> --prompt <prompt> --top-k 20 --output <json>` | Yes | Final-position top-k for a prompt string | No built-in oracle compare | Cannot run the exact proof case unless the exact prompt string is available. Does not accept token IDs. |
| `restricted_logit_diff` | `cargo run -p gpt-oss-bench --features cuda --bin restricted_logit_diff -- --model <model> --prompt <prompt> --top-k 20 --output <json>` | Yes | Worker/runner step captures including logits | No built-in final-readout digest compare | Useful differential tool, but not an artifact oracle smoke. Does not accept token IDs. |
| `restricted_prefill_trace` | `cargo run -p gpt-oss-bench --features cuda --bin restricted_prefill_trace -- ...` | Yes | Runtime trace data | No | Trace-oriented, not a final logits smoke. |
| `live_cuda_parity` | `cargo run -p gpt-oss-bench --features cuda --bin live_cuda_parity -- ...` | Yes | Engine/request liveness and decode comparison | No | Liveness/parity-oriented, not raw final logits artifact comparison. |

## Probe Commands

Help probes confirmed the available surfaces:

```bash
cargo run -p gpt-oss-bench --features cuda --bin restricted_prefill_topk -- --help
cargo run -p gpt-oss-bench --features cuda --bin restricted_logit_diff -- --help
```

Both commands completed. Existing Rust warnings were observed during compilation.

The same help probes without `--features cuda` are not usable for this purpose:

- `restricted_logit_diff` declares a CUDA feature requirement.
- `restricted_prefill_topk` reaches CUDA-gated worker APIs and must be run with
  `--features cuda`.

No exact model smoke was run in this slice because the current integration tools take a
prompt string, not the exact token IDs for `developer-message-user-smoke`. Running a
different prompt would not answer the final-token oracle question.

## Current Feasibility

Current integration supports:

- CUDA compile coverage for promoted runtime paths.
- RoPE and RMSNorm unit/reference tests.
- Artifact/digest validation against the runtime-forward final-readout artifact.
- Prompt-string top-k capture through `restricted_prefill_topk`.
- Prompt-string worker/runner logit differential capture through `restricted_logit_diff`.

Current integration does not support:

- Exact `developer-message-user-smoke` token-ID replay through an existing CLI.
- Built-in final logits digest generation for comparison with the final-readout artifact.
- Built-in top-20 comparison against the regenerated direct-module final-readout artifact.
- Full transformer-to-logits oracle proof replay without the runtime-forward proof harness.

Even if an exact current-runtime smoke is added, full digest parity is not expected to be a
guaranteed integration-native result until the CUDA BF16 projection policy is deliberately
designed and promoted. The oneDNN Q/K/V projection candidates remain proof-only.

## Recommended Minimal Future Tool

Add a small tool named `final_token_logit_smoke` in a separate slice.

Requirements:

- No dependency on `runtime_forward_layer0_qkv_bf16_candidate_status.rs`.
- No oneDNN Q/K/V candidate imports.
- No selected expert readout correction.
- No debug capture plumbing.
- No raw `.live` artifact dependency.
- Runs the current default integration runtime path.
- Accepts exact token IDs, for example `--token-ids-json <path>` or `--token-ids <csv>`.
- Optionally accepts `--prompt <text>` for convenience, but token IDs are required for exact
  proof-case replay.
- Emits final-position top-k logits and optional full logits digest.
- Accepts `--artifact <final-readout-status.json>` to compare classification, expected digest,
  vocabulary size, and top-20 token IDs/values when available.
- Reports clearly that exact parity is not expected until projection-policy work is promoted.

Suggested command shape:

```bash
cargo run -p gpt-oss-bench --features cuda --bin final_token_logit_smoke -- \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --token-ids-json /path/to/developer-message-user-smoke-token-ids.json \
  --artifact /home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json \
  --top-k 20 \
  --output /tmp/final-token-logit-smoke.json
```

## Difference From Runtime-Forward Proof Replay

The runtime-forward proof replay is a full oracle proof path. It uses proof-only status modes
and candidate mechanisms to verify intermediate seams through final block output, final norm,
and LM-head logits.

The proposed integration smoke is narrower. It should execute the promoted integration runtime
path, emit final-token evidence, and compare against the preserved oracle artifact where
possible. It is a review and regression smoke, not a replacement for the full
runtime-forward proof harness.

## Recommendation

Keep full final-token oracle proof replay on `feature/runtime-forward` for now. Use
integration for promoted runtime candidate checks and artifact/digest validation. If exact
integration-native final-token evidence is needed, add the minimal `final_token_logit_smoke`
tool as a bounded follow-up before any further projection-policy extraction.
