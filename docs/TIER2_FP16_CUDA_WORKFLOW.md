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

## Operational Notes

- Use Tier 0 for compile-only plumbing checks.
- Use Tier 1 for trace capture and artifact reuse.
- Use Tier 2 for merge candidates, contract changes, or any run where a compare claim matters.
- Prefer representative sentinel layers over exhaustive layer-by-layer reruns unless a concrete hypothesis needs more depth.
- Keep generated `.live/` artifacts local and out of commits unless a task explicitly asks for checked-in fixtures.
