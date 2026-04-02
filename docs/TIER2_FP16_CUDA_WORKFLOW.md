# Tier-2 FP16 CUDA Workflow

This repository treats restricted-fp16 CUDA Tier 2 as a three-part workflow:

1. Raw global compare is telemetry.
2. Runtime-emulated global compare is localization.
3. Same-input local replay is ownership proof before calling a real runtime bug.

Do not skip step 3 when a surface still looks suspicious after runtime-emulated compare.

## Contract

### Raw Global Compare

Use raw compare when you want the unfiltered CUDA-vs-oracle picture.

- Command path: `--compare-mode raw`
- Meaning: preserve the full bf16-oracle delta surface as telemetry
- What it proves: where divergence is first visible globally
- What it does not prove: that the first visible global delta belongs to the runtime path being accused

### Runtime-Emulated Global Compare

Use runtime-emulated compare to localize whether a visible global divergence is materially explained by runtime numeric policy.

- Command path: `--compare-mode runtime-emulated`
- Meaning: keep raw deltas for telemetry, but gate the global conclusion using runtime-emulated layer-0 numeric-policy-sensitive surfaces
- What it proves: whether the current global frontier is better explained by numeric-policy differences than by an immediately local runtime bug
- What it does not prove: ownership of a later layer/path

### Same-Input Local Replay

Use local replay only after the global compare suggests a candidate layer/path worth checking.

- Required inputs:
  - seed capture from the real CUDA trace
  - `--local-replay-layer <n>`
  - `--local-replay-path <coarse|attention|mlp>`
- Meaning: replay the candidate layer/path using the exact same traced seed input
- What it proves: whether the proposed local owner survives exact/corrected same-input replay
- Standard: if the local owner does not survive same-input replay, do not call it a runtime bug

## Seed Capture

Seed capture is opt-in. Default compare behavior must remain unchanged.

- CLI: `--seed-layers <csv>`
- Env plumbing: `GPT_OSS_TRACE_SEED_LAYERS`
- Purpose: capture full-sequence seed inputs and supporting runtime surfaces needed for later same-input local replay

Example:

```bash
./scripts/probe_validation_tier.sh \
  --tier 2 \
  --compare-mode runtime-emulated \
  --seed-layers 12,23
```

Equivalent direct trace run:

```bash
CUDA_VISIBLE_DEVICES=1 \
GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
GPT_OSS_TRACE_SEED_LAYERS=12,23 \
target/release/restricted_prefill_trace \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prompt "Explain tensor parallelism in one short sentence." \
  --max-model-len 128 \
  --gpu-memory-utilization 0.75 \
  --output .live/restricted-cuda-prefill-trace.integration.json \
  --log-level info
```

## Local Replay

Local replay requires both layer and path flags together.

- `--local-replay-layer <n>`
- `--local-replay-path <coarse|attention|mlp>`

Examples:

```bash
./scripts/probe_validation_tier.sh \
  --tier 2 \
  --compare-mode runtime-emulated \
  --seed-layers 12 \
  --local-replay-layer 12 \
  --local-replay-path coarse
```

```bash
python3 crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py \
  --cuda-trace-json .live/restricted-cuda-prefill-trace.integration.json \
  --original-model /data/models/openai/gpt-oss-20b \
  --output .live/restricted-prefill-trace-diff.layer12-local-attention.json \
  --compare-mode runtime-emulated \
  --local-replay-layer 12 \
  --local-replay-path attention \
  --device cpu
```

## Opt-In Warm Oracle Reuse

Default behavior stays one-shot: one trace compare per Python process.

When you need to compare several traces or several replay variants against the same oracle checkpoint, you can opt into a warm listener session that reuses one loaded Python oracle process.

The smallest operator-facing path is now available in the shell harness:

```bash
./scripts/probe_validation_tier.sh \
  --compare-only \
  --compare-mode runtime-emulated \
  --warm-oracle
```

What the script does in warm mode:

- starts one warm oracle listener
- submits the current compare request twice in one session
- writes the primary report to `--oracle-output`
- writes a second reuse-check report beside it as `*.warm-reuse-check.json`
- writes `*.meta.json` sidecars beside generated trace and oracle artifacts
- prints whether the second request reused the already-loaded session

This is still a bounded harness path. It is meant to prove script-level session reuse without changing compare semantics or claiming broad live-run speedups.

Artifact reuse guardrail:

- trace reuse is now metadata-gated, not filename-gated
- `compare-only` fails closed if the trace sidecar does not match the current model path, prompt identity, `max-model-len`, trace schema marker, or requested local replay layer coverage
- a plain mismatch means recapture or use a compatible artifact; it does not silently widen reuse
- newly captured traces now write wrapper-owned capture metadata in the sidecar, so operators can tell current wrapper captures apart from older legacy-unversioned artifacts
- older artifacts remain explicitly labeled as legacy at reuse time; they are not silently upgraded to the new wrapper-capture contract
- `--inspect-trace-artifact` prints the reuse classification, checked provenance fields, and exact accept/reject reasons without recapturing or comparing
- `--require-current-trace-contract` is an opt-in strict mode that rejects legacy artifacts even when older provenance fields still match
- `./scripts/test_probe_validation_wrapper.sh` runs the bounded wrapper-only regression path for current metadata, legacy metadata, incompatible metadata, and warm-oracle test-mode reuse

## Operator Quick Reference

Use these commands as the default operator entrypoints:

- representative Tier 2 run:

```bash
./scripts/probe_validation_tier.sh \
  --tier 2 \
  --compare-mode runtime-emulated \
  --seed-layers 0,12,23
```

- compare an existing trace artifact without recapturing:

```bash
./scripts/probe_validation_tier.sh \
  --compare-only \
  --compare-mode runtime-emulated \
  --trace-json .live/restricted-cuda-prefill-trace.integration.json
```

- inspect whether an artifact is reusable before running compare-only:

```bash
./scripts/probe_validation_tier.sh \
  --inspect-trace-artifact \
  --trace-json .live/restricted-cuda-prefill-trace.integration.json \
  --compare-mode runtime-emulated
```

- inspect in strict mode when legacy artifacts must be refused:

```bash
./scripts/probe_validation_tier.sh \
  --inspect-trace-artifact \
  --require-current-trace-contract \
  --trace-json .live/restricted-cuda-prefill-trace.integration.json \
  --compare-mode runtime-emulated
```

- run the bounded wrapper regression script:

```bash
./scripts/test_probe_validation_wrapper.sh
```

Use `--warm-oracle` only when you intentionally want the bounded reuse-check flow against one existing trace artifact.

## Long-Context YaRN Proof Seam

For the `bd49d35` frontier, use the bounded harness seam instead of ad hoc long-context commands:

```bash
./scripts/run_yarn_long_context_proof.sh \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --output-dir .live/yarn-long-context-proof \
  --timeout 1800
```

When a lighter post-4096 proof seam is available, reuse the same runner with a different proof binary and bounded proof-only env knobs:

```bash
./scripts/run_yarn_long_context_proof.sh \
  --build-only \
  --bin restricted_logit_diff \
  --env PROOF_MODE=compact \
  --env PROOF_LABEL=gpu0-yarn \
  --output-dir .live/yarn-long-context-proof
```

For a bounded post-RoPE or post-attention seam that still uses `restricted_prefill_trace` as the outer binary, let the runner allocate a compact proof artifact path through an env var while preserving the normal `--output` trace path:

```bash
./scripts/run_yarn_long_context_proof.sh \
  --build-only \
  --bin restricted_prefill_trace \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name post_attention_probe.json \
  --env GPT_OSS_PROBE_MODE=post-attention \
  --output-dir .live/yarn-long-context-proof
```

When the compact proof artifact contains a small numeric head such as `values_head`, ask the runner to summarize the direct vector diff in `summary.json`:

```bash
./scripts/run_yarn_long_context_proof.sh \
  --run-only \
  --bin restricted_prefill_trace \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name post_attention_probe.json \
  --compare-vector-key values_head \
  --output-dir .live/yarn-long-context-proof
```

What it does:

- generates or verifies one deterministic prompt whose tokenized length is above 4096
- verifies the restricted-model view is sink-free before running
- can prebuild both worktrees once, then rerun the proof with the warmed `target/release/restricted_logit_diff` binaries
- records the selected proof binary and any repeated `--env KEY=VALUE` passthrough in the generated plan files
- can assign a per-side compact proof artifact path through `--proof-artifact-env` while keeping the outer binary's normal `--output` artifact intact
- can summarize `max_abs_diff`, `mean_abs_diff`, and `first_diff_index` directly when both compact proof artifacts expose the selected numeric vector key
- runs the same bounded `restricted_logit_diff` observation seam in the safe and variant worktrees
- records stdout, stderr, exact commands, GPU snapshot, and a compact `summary.json`

Contract for alternate proof binaries:

- keep the bounded `--model`, `--prompt`, `--max-model-len`, and `--output` surface so the same runner can warm, execute, and compare both sides
- if the proof seam emits a smaller side artifact through an env-provided path, use `--proof-artifact-env` and `--proof-artifact-name` so the runner compares that compact artifact first
- emit one compact JSON artifact to the requested `--output` path if you want generic artifact comparison in `summary.json`

What to expect:

- `state=completed` with both reports present means the seam produced a comparable same-input summary
- `state=timed_out` means the current seam is still too heavy in the allotted window, but the timeout is now reproducible and captured explicitly
- `state=failed` means the proof stopped before the bounded runtime observation completed; inspect the per-case `run.stderr`

## Bounded GPU0 Live-Smoke Workflow

Use the live-smoke runner when you want an operator-clear GPU0 command path without turning it into a proof claim yet:

```bash
./scripts/run_gpu0_live_smoke.sh \
  --build-only \
  --gpu 0 \
  --tree /home/emmy/openai/gpt-oss-rs \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prompt-file /tmp/gpu0-live-prompt.txt \
  --timeout 1800 \
  --output-dir .live/gpu0-live-smoke \
  --env GPT_OSS_PROBE_MODE=layer0-mlp
```

Then reuse the warmed binary for the bounded run:

```bash
./scripts/run_gpu0_live_smoke.sh \
  --run-only \
  --gpu 0 \
  --tree /home/emmy/openai/gpt-oss-rs \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prompt-file /tmp/gpu0-live-prompt.txt \
  --timeout 1800 \
  --output-dir .live/gpu0-live-smoke \
  --env GPT_OSS_PROBE_MODE=layer0-mlp
```

This runner only captures the bounded live-smoke surface:

- exact build/run commands
- env passthrough
- GPU snapshot
- stdout/stderr
- exit state and timeout state
- outer artifact path and `summary.json`

## Retained-Continuation Proof Workflow

For the honest retained-state continuation path, use the dedicated retained runner instead of replay-based staging. Its contract is one binary invocation per tree with both the prefix boundary and the continuation step supplied together:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_prefill_trace \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --output-dir .live/retained-continuation-proof \
  --env GPT_OSS_PROBE_MODE=retained-continuation
```

What this prepares:

- one retained-state invocation plan per tree, not two prompt replays from scratch
- one outer artifact path per tree
- one optional compact continuation proof-artifact path per tree
- the same build/run split, env capture, GPU snapshot, and compact vector-diff summary path used elsewhere in the harness
To prove the split is really the intended `4096 + 1` case before running anything live, enable tokenizer verification and strict count guards:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --output-dir .live/retained-continuation-proof
```

The resulting `summary.json` records:

- prefix token count
- continuation token count
- continuation token id list
- continuation text repr

For the retained-state `restricted_logit_diff` path, add forced-output token emission so the runner writes the exact command-ready continuation token surface into both `summary.json` and the generated run plans:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --output-dir .live/retained-continuation-proof
```

This records the exact continuation token ids and the command-ready forced-output token argument string, for example `--forced-output-tokens 6602`.

For the retained-state `restricted_logit_diff` path, also set an explicit run budget above the verified minimum. For the current `4096 + 1` split, the runner records a required minimum model length of `4097`, so the bounded operator path should keep using `--max-model-len 4608`:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --output-dir .live/retained-continuation-proof
```

With strict token verification enabled, the runner now fails closed if `--max-model-len` is omitted or smaller than the verified minimum.

If the retained child emits progress markers on stdout/stderr, prefer the built-in retained debug profile so the runner records the ordered stage sequence, the deepest marker reached, the next missing marker, and a short stall classification without manual log grepping:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --marker-profile retained-debug-v1 \
  --output-dir .live/retained-continuation-proof
```

The built-in `retained-debug-v1` profile currently covers:

- `RETAINED_CHILD_START`
- `RETAINED_CHILD_TOKENIZED`
- `RETAINED_CHILD_BUILD_WORKER_BEGIN`
- `RETAINED_CHILD_BUILD_WORKER_DONE`
- `RETAINED_STEP_BEGIN`
- `RETAINED_STEP_FORWARD_BEGIN`
- `RETAINED_STEP_FORWARD_DONE`
- `DECODE1_BEGIN`
- `RETAINED_PROOF_ENTER`
- `RETAINED_PROOF_CAPTURED`

With that profile enabled, `summary.json` records the configured marker sequence, seen markers, deepest marker reached, the next expected marker not yet seen, and a stall label such as `stalled_before=RETAINED_STEP_FORWARD_BEGIN`.

For the next layer-0 retained prefill frontier, switch to the finer-grained `retained-mlp-v1` profile:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --marker-profile retained-mlp-v1 \
  --output-dir .live/retained-continuation-proof
```

`retained-mlp-v1` extends the retained sequence through layer-0 attention, residual, router, expert, and `mlp_done` substages, so `summary.json` can report boundaries such as `stalled_before=RETAINED_PREFILL_STAGE layer=0 stage=expert_begin` directly.

For the tighter pre-router frontier, switch to `retained-mlp-prerouter-v1`:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --marker-profile retained-mlp-prerouter-v1 \
  --output-dir .live/retained-continuation-proof
```

`retained-mlp-prerouter-v1` narrows the ordered sequence further by splitting the layer-0 MLP into pre-router, router-topk, first-expert, and aggregate boundaries, so `summary.json` can report stalls such as `stalled_before=RETAINED_MLP_STAGE layer=0 stage=prerouter_begin` directly.

For the current router-call frontier, switch to `retained-router-v1`:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --marker-profile retained-router-v1 \
  --output-dir .live/retained-continuation-proof
```

`retained-router-v1` narrows the ordered sequence to the router invocation and early router/top-k stages, so `summary.json` can report boundaries such as `stalled_before=RETAINED_MLP_STAGE layer=0 stage=router_enter` directly.
For the host-transfer/router-input frontier, switch to `retained-router-host-v1`:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --marker-profile retained-router-host-v1 \
  --output-dir .live/retained-continuation-proof
```

`retained-router-host-v1` narrows the ordered sequence further to the host-side router transfer and input-preparation boundary, so `summary.json` can report stalls such as `stalled_before=RETAINED_MLP_STAGE layer=0 stage=router_dtoh_begin` directly.
For the router-score loop frontier, switch to `retained-router-score-v1`:

```bash
./scripts/run_retained_continuation_proof.sh \
  --setup-only \
  --emit-forced-output-tokens \
  --verify-tokenization \
  --python /data/models/.venv-awq/bin/python \
  --required-prefix-token-count 4096 \
  --required-continuation-token-count 1 \
  --gpu 0 \
  --safe-tree /home/emmy/openai/gpt-oss-rs \
  --variant-tree /home/emmy/openai/worktrees/runtime-forward \
  --model /data/models/openai/gpt-oss-20b-full-attn-restricted-integration \
  --prefix-prompt-file /tmp/prefix-4096.txt \
  --continuation-prompt-file /tmp/continuation-step.txt \
  --bin restricted_logit_diff \
  --max-model-len 4608 \
  --proof-artifact-env GPT_OSS_PROOF_JSON \
  --proof-artifact-name continuation-head.json \
  --compare-vector-key values_head \
  --marker-profile retained-router-score-v1 \
  --output-dir .live/retained-continuation-proof
```

`retained-router-score-v1` narrows the ordered sequence further into score-buffer allocation and first-accum boundaries, so `summary.json` can report stalls such as `stalled_before=RETAINED_MLP_STAGE layer=0 stage=router_score_begin` directly.
Start the listener:

```bash
python3 crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py \
  --listen \
  --original-model /data/models/openai/gpt-oss-20b \
  --device cpu
```

Send compare requests on stdin as NDJSON:

```json
{"op":"compare","cuda_trace_json":".live/restricted-cuda-prefill-trace.integration.json","output":".live/restricted-prefill-trace-diff.raw.json","compare_mode":"raw"}
{"op":"compare","cuda_trace_json":".live/restricted-cuda-prefill-trace.integration.json","output":".live/restricted-prefill-trace-diff.layer12-attention.json","compare_mode":"runtime-emulated","local_replay_layer":12,"local_replay_path":"attention"}
{"op":"shutdown"}
```

What this does:

- keeps today’s one-shot path unchanged
- reuses one loaded oracle session across repeated compare requests
- keeps compare semantics identical to the existing one-shot report path
- now has a shell-level opt-in path for bounded reuse verification via `--warm-oracle`

What this does not do:

- it does not change default compare behavior
- it does not change any runtime or model math
- it does not add a new correctness claim beyond reducing harness process startup cost

```bash
python3 crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py \
  --cuda-trace-json .live/restricted-cuda-prefill-trace.integration.json \
  --original-model /data/models/openai/gpt-oss-20b \
  --output .live/restricted-prefill-trace-diff.layer23-local-mlp.json \
  --compare-mode runtime-emulated \
  --local-replay-layer 23 \
  --local-replay-path mlp \
  --device cpu
```

## Evidence Standard Before Calling A Runtime Bug

Treat the following as the minimum bar:

1. Raw global compare shows the telemetry frontier.
2. Runtime-emulated compare shows the localization frontier.
3. Same-input local replay for the accused layer/path still survives exact or corrected replay.

If step 3 fails, downgrade the claim from "runtime bug" to "not yet owned."

## Operator-First Flow

Use the current workflow in this order:

1. Start with a representative Tier 2 run that captures telemetry and localization together.
2. Only escalate to same-input local replay for the specific layer/path that still looks worth owning.
3. Treat local replay as the gate before claiming a real runtime bug.

Representative layers usually start with:

- shallow: `0`
- mid: `12`
- late: `23`

Add `1` when you specifically want a second shallow check, but do not default to exhaustive per-layer reruns.

Recommended first run:

```bash
./scripts/probe_validation_tier.sh \
  --tier 2 \
  --compare-mode runtime-emulated \
  --seed-layers 0,12,23
```

What this gives you:

- raw global telemetry
- runtime-emulated localization
- exact traced runtime seeds for the representative layers you may replay later

Recommended escalation from an existing trace artifact:

```bash
./scripts/probe_validation_tier.sh \
  --compare-only \
  --compare-mode runtime-emulated \
  --local-replay-layer 12 \
  --local-replay-path attention
```

Use this only after the representative run leaves a specific layer/path worth checking. If that replay does not survive, stop calling that surface a runtime owner.

## Live-Testing Checklist

Before the first live run:

- confirm you are in `~/openai/worktrees/tier2-workflow`
- confirm the branch is `harness/tier2-workflow`
- confirm the restricted model view path is the one you intend to test
- reserve GPU1 in `~/openai/AGENT_CHANGELOG.md` if needed

Suggested operator sequence:

1. `--tier 0` after script/help-only edits
2. representative `--tier 2 --compare-mode runtime-emulated --seed-layers 0,12,23`
3. `--compare-only` local replay for the one remaining candidate layer/path
4. write down whether the result is telemetry only, localized, or actually owned

Do not escalate to broad runtime debugging until step 3 leaves a surviving same-input local owner.

## Operational Notes

- Use Tier 0 for compile-only plumbing checks.
- Use Tier 1 for trace capture and artifact reuse.
- Use Tier 2 for merge candidates, contract changes, or any run where a compare claim matters.
- Use the warm listener only when you have multiple compare requests against the same oracle checkpoint and want bounded harness-side process reuse.
- Prefer representative sentinel layers over exhaustive layer-by-layer reruns unless a concrete hypothesis needs more depth.
- Keep generated `.live/` artifacts local and out of commits unless a task explicitly asks for checked-in fixtures.
