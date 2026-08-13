# GPT-OSS CPU Full-Checkpoint Regression Procedure

> The captures and counts produced by earlier revisions of this procedure are
> retired historical records. They must not be reused or counted by the fresh
> container-oracle campaign. The current authoritative procedure is
> [`CPU_FRESH_ORACLE_CAMPAIGN.md`](CPU_FRESH_ORACLE_CAMPAIGN.md).

This procedure produced the evidence used to merge the experimental CPU
serving baseline. Re-run it after numerical changes, kernel-plan changes,
packed-layout changes, or cache-format changes. The initial i5 checks cover
official tensor decoding and isolated full-attention, sliding-attention, and
MoE layers; they do not substitute for this complete-model regression.

Passing this procedure does not automatically enable CPU in trusted mode.
Trusted-mode eligibility requires a separate policy review.

## Host setup

Keep every large or generated artifact outside Git. Example host-local paths:

```bash
export HF_HOME=/data/models/gpt-oss/hf
export GPT_OSS_RS_CACHE=/data/cache/gpt-oss-rs
export CARGO_TARGET_DIR=/data/cache/gpt-oss-rs/target
export BENCH_ROOT=/data/benchmarks/gpt-oss-rs
mkdir -p "$BENCH_ROOT"
```

Build and fetch the native SafeTensors snapshot first. Download the llama.cpp
GPT-OSS GGUF only on this host, immediately before the comparison runs.

```bash
cargo build --release --locked -p gpt-oss-server
"$CARGO_TARGET_DIR/release/gpt-oss-rs" fetch \
  --model openai/gpt-oss-20b \
  --revision main \
  --cache-dir "$HF_HOME/hub"
```

Record the printed resolved revision and manifest path. Do not accept a run if
the manifest or repack cache reports corruption, a stale version, or an
interrupted publication.

## Complete-model and API runs

Run one cold-cache startup to measure repack time, then warm-cache runs for
each available kernel path. Use physical-core count for `CPU_THREADS`.

```bash
export CPU_THREADS="$(lscpu -p=core,socket | awk -F, '!/^#/ { seen[$1 FS $2]=1 } END { print length(seen) }')"
export RUST_LOG='gpt_oss_engine::worker::cpu_worker=debug,gpt_oss_server=info'
"$CARGO_TARGET_DIR/release/gpt-oss-rs" serve \
  --model openai/gpt-oss-20b \
  --device cpu \
  --profile gpt-oss-cpu \
  --cpu-kernel auto \
  --cpu-threads "$CPU_THREADS" \
  --cpu-repack-cache "$GPT_OSS_RS_CACHE" \
  --runtime-mode experimental
```

Exercise non-streaming and streaming `/v1/chat/completions` and
`/v1/responses` requests. Cover three deterministic greedy prompt classes:

- short plain text;
- a longer prompt crossing several sliding-window turnovers;
- Harmony messages containing a function tool declaration and tool history.

For the tool class, cover an initial function call, assistant/tool Chat
history, a stored Responses follow-up using `previous_response_id`, a manually
supplied `function_call_output`, and both streaming forms. Also force a
generation limit in the middle of a Harmony message. Chat must finish with
`length`; Responses must finish with `status=incomplete` and
`reason=max_output_tokens`. Neither case may return HTTP 500.

### Pinned parity workflow

The checked-in fixture manifest pins the exact rendered Harmony text and token
IDs for the 63-, 122-, 136-, 262-, 346-, and 444-token prompts plus the tool
history prompt. Generated captures, model files, traces, and benchmark output
must remain outside Git. Set `MODEL_ROOT` to the resolved SafeTensors snapshot,
`PINNED_GPT_OSS_SOURCE` to a clean checkout at the manifest's official source
revision. For new evidence, never select a host Python environment: use the
digest-locked container interface in the fresh campaign procedure.

```bash
cargo run --release --locked -p gpt-oss-bench --bin cpu_parity -- \
  --model "$MODEL_ROOT" \
  --repack-cache "$GPT_OSS_RS_CACHE" \
  --scenario harmony_122 \
  --kernel auto \
  --threads "$CPU_THREADS" \
  --max-new-tokens 8 \
  --trace-layers 0 \
  --trace-step 0 \
  --output "$BENCH_ROOT/native-harmony-122.json"

"$ORACLE_PYTHON" crates/gpt-oss-bench/tools/official_cpu_oracle.py \
  --native-capture "$BENCH_ROOT/native-harmony-122.json" \
  --model "$MODEL_ROOT" \
  --official-source "$PINNED_GPT_OSS_SOURCE" \
  --max-new-tokens 8 \
  --trace-layers 0 \
  --trace-step 0 \
  --threads "$CPU_THREADS" \
  --output "$BENCH_ROOT/official-harmony-122.json"

"$ORACLE_PYTHON" crates/gpt-oss-bench/tools/compare_cpu_parity.py \
  --native "$BENCH_ROOT/native-harmony-122.json" \
  --official "$BENCH_ROOT/official-harmony-122.json" \
  --llama "$BENCH_ROOT/llama-harmony-122.json" \
  --output "$BENCH_ROOT/compare-harmony-122.json"
```

The official SafeTensors/PyTorch capture at the source revision pinned in the
manifest is the sole blocking semantic authority. Exact native/official
generated-token parity is mandatory for every scenario. A native/official
token divergence fails the gate.

llama.cpp remains a required differential reference. Run its pinned revision
and pinned GGUF hash with the manifest's exact prompt token IDs, greedy
sampling, top-logprob capture, and physical `--ubatch-size 1`. Record its exact
tokens, first divergence, competing-token log-probability gap, and near-tie
classification for every scenario. These llama.cpp results are advisory: a
divergence cannot fail or waive the official-oracle gate. The comparator's
`--llama-near-tie` threshold remains an informational diagnostic and does not
affect its exit status. Text-only similarity is not a substitute for token
parity in either comparison.

Use the opt-in trace only to localize a failing prompt. Compare, in order,
selected-layer input norm, post-RoPE query/key, value projection, attention,
post-attention residual, router selection/weights, MoE output, layer output,
final norm, and top logits. Correct and regress the earliest mismatching
operator rather than compensating at a later layer.

`--trace-step N` is zero-based and captures the complete context and logits
used to select generated token `N`; step 0 is the final prefill token. For an
expert diagnostic, pass `--expert-projection exact-bf16` to the native parity
runner and compare each selected expert's gate/up projection, SwiGLU result,
down projection, and weighted output. `exact-bf16` is parity-only and is not a
server flag or a production fallback. Normal CPU serving uses residual Q8.

## Required report

For cold repack and each scalar, AVX2, and AVX-512/VNNI run, record:

- exact commit, model revision, source manifest hashes, and llama.cpp version;
- time to first token and decode tokens per second;
- peak RSS from `/usr/bin/time -v` or an equivalent host tool;
- cold repack duration and warm startup duration;
- exact native, official-oracle, and llama.cpp greedy token sequences for all
  seven pinned scenarios, including llama.cpp first-divergence margins;
- Chat Completions and Responses results in streaming and non-streaming modes.

A forced unavailable ISA must fail before execution. Any NaN,
native/official token divergence, cache error, or memory pressure beyond
practical 32 GiB use must be investigated before merge. llama.cpp divergence
must be retained in the report as advisory evidence. Forced scalar, AVX2, and
AVX-512/VNNI must agree on exact greedy tokens. Benchmark scalar, AVX2, forced
AVX-512/VNNI, and automatic dispatch with three warm repeats per path. The
automatic median throughput may not be worse than AVX2 by more than 2%, and
the automatic run may not regress correctness, peak RSS, or cache integrity.
There is no absolute throughput threshold for the first correct MVP.

When every item passes, attach the report to the relevant kernel/runtime pull
request. If a trace diagnostic differs while the official token gate passes,
record the earliest mismatch and its ownership explicitly rather than waiving
or hiding it. Keep CPU rejected in trusted mode until a separately reviewable
follow-up change establishes that eligibility.
