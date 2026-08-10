# GPT-OSS CPU Final Conformance Gate

Run this gate on the 32 GiB i7 before marking the CPU serving pull request
ready or enabling CPU in trusted mode. The initial i5 gate covers official
tensor decoding and isolated full-attention, sliding-attention, and MoE
layers; it does not substitute for this complete-model gate.

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

Capture the `CPU sampled token` records and compare token IDs, in order, with
the current llama.cpp GPT-OSS GGUF run using the same prompt tokens and greedy
settings. Investigate the first divergence; text-only similarity is not a
substitute for token parity.

## Required report

For cold repack and each scalar, AVX2, and AVX-512/VNNI run, record:

- exact commit, model revision, source manifest hashes, and llama.cpp version;
- time to first token and decode tokens per second;
- peak RSS from `/usr/bin/time -v` or an equivalent host tool;
- cold repack duration and warm startup duration;
- exact greedy token sequences for all three prompt classes;
- Chat Completions and Responses results in streaming and non-streaming modes.

A forced unavailable ISA must fail before execution. Any NaN, token
divergence, cache error, memory pressure beyond practical 32 GiB use, or
AVX-512 regression relative to AVX2 must be investigated before merge. There
is no absolute throughput threshold for the first correct MVP.

When every item passes, attach the report to the serving PR, mark it ready,
and make CPU eligible for trusted mode in a separately reviewable change.
