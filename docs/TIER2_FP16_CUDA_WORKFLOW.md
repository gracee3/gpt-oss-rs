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
