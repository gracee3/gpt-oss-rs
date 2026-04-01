#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [--tier 0|1|2] [options]"
  cat <<'USAGE'
Tier 0:
  compile only.

Tier 1:
  compile then trace (or reuse a valid existing trace artifact).

Tier 2:
  compile + trace (or reuse) + oracle compare.

Options:
  -t, --tier <0|1|2>
      validation tier to run (default: 0)
  --model <path>
      restricted model view path (default: /data/models/openai/gpt-oss-20b-full-attn-restricted-integration)
  --trace-json <path>
      trace artifact path (default: .live/restricted-cuda-prefill-trace.integration.json)
  --oracle-output <path>
      oracle diff artifact path (default: .live/restricted-prefill-trace-diff.integration.json)
  --original-model <path>
      original/oracle checkpoint path (default: /data/models/openai/gpt-oss-20b)
  --compare-mode <raw|runtime-emulated>
      comparison contract for Tier 2 oracle compare (default: raw)
  --prompt <text>
      prompt text used for trace capture
  --max-model-len <n>
      model max length for trace capture (default: 128)
  --gpu <idx>
      CUDA_VISIBLE_DEVICES index (default: 1)
  --seed-layers <csv>
      comma-separated layer indices for opt-in full-sequence seed capture during trace
  --local-replay-layer <n>
      opt-in same-input local replay layer for oracle compare
  --local-replay-path <coarse|attention|mlp>
      local replay path used with --local-replay-layer
  --no-reuse
      force trace rerun even if existing artifact is reusable
  --compare-only
      run oracle compare only (uses existing trace file, skips compile/trace)
  -h, --help
      show help

Common operator flows:
  1) Representative telemetry + localization capture:
     ./scripts/probe_validation_tier.sh \
       --tier 2 \
       --compare-mode runtime-emulated \
       --seed-layers 0,12,23

  2) Same-input local replay from an existing trace artifact:
     ./scripts/probe_validation_tier.sh \
       --compare-only \
       --compare-mode runtime-emulated \
       --local-replay-layer 12 \
       --local-replay-path attention

  3) Capture and replay the same representative layer in one run:
     ./scripts/probe_validation_tier.sh \
       --tier 2 \
       --compare-mode runtime-emulated \
       --seed-layers 12 \
       --local-replay-layer 12 \
       --local-replay-path coarse
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TIER="0"
MODEL_PATH="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
TRACE_JSON="$REPO_ROOT/.live/restricted-cuda-prefill-trace.integration.json"
ORACLE_OUTPUT="$REPO_ROOT/.live/restricted-prefill-trace-diff.integration.json"
ORIGINAL_MODEL="/data/models/openai/gpt-oss-20b"
COMPARE_MODE="raw"
PROMPT="Explain tensor parallelism in one short sentence."
MAX_MODEL_LEN="128"
GPU="1"
SEED_LAYERS=""
LOCAL_REPLAY_LAYER=""
LOCAL_REPLAY_PATH=""
REUSE_TRACE="1"
COMPARE_ONLY="0"

usage_error() {
  echo "$1" >&2
  echo >&2
  usage >&2
  exit 1
}

csv_contains_value() {
  local csv="$1"
  local needle="$2"
  local item=""

  IFS=',' read -r -a items <<< "$csv"
  for item in "${items[@]}"; do
    if [[ "$(echo "$item" | tr -d '[:space:]')" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tier)
      TIER="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --trace-json)
      TRACE_JSON="$2"
      shift 2
      ;;
    --oracle-output)
      ORACLE_OUTPUT="$2"
      shift 2
      ;;
    --original-model)
      ORIGINAL_MODEL="$2"
      shift 2
      ;;
    --compare-mode)
      COMPARE_MODE="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --seed-layers)
      SEED_LAYERS="$2"
      shift 2
      ;;
    --local-replay-layer)
      LOCAL_REPLAY_LAYER="$2"
      shift 2
      ;;
    --local-replay-path)
      LOCAL_REPLAY_PATH="$2"
      shift 2
      ;;
    --no-reuse)
      REUSE_TRACE="0"
      shift
      ;;
    --compare-only)
      COMPARE_ONLY="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$COMPARE_ONLY" == "1" ]]; then
  TIER="2"
fi

if [[ "$TIER" != "0" && "$TIER" != "1" && "$TIER" != "2" ]]; then
  usage_error "Invalid tier: $TIER"
fi

if [[ -n "$LOCAL_REPLAY_LAYER" || -n "$LOCAL_REPLAY_PATH" ]]; then
  if [[ -z "$LOCAL_REPLAY_LAYER" || -z "$LOCAL_REPLAY_PATH" ]]; then
    usage_error "--local-replay-layer and --local-replay-path must be provided together"
  fi
  if [[ "$TIER" != "2" ]]; then
    usage_error "local replay is only available for Tier 2 / compare-only runs"
  fi
  if [[ "$COMPARE_ONLY" != "1" ]]; then
    if [[ -z "$SEED_LAYERS" ]]; then
      usage_error "local replay in the same run requires --seed-layers so the trace captures an exact runtime seed"
    fi
    if ! csv_contains_value "$SEED_LAYERS" "$LOCAL_REPLAY_LAYER"; then
      usage_error "--seed-layers must include --local-replay-layer when capture and replay happen in the same run"
    fi
  fi
fi

have_python3() {
  command -v python3 >/dev/null 2>&1
}

can_reuse_trace() {
  local trace_path="$1"
  local expected_model="$2"
  local expected_prompt="$3"

  if [[ ! -f "$trace_path" ]]; then
    echo "0"
    return 0
  fi
  if ! have_python3; then
    echo "0"
    return 0
  fi
  python3 - "$trace_path" "$expected_model" "$expected_prompt" <<'PY'
import json
import sys

trace_path, expected_model, expected_prompt = sys.argv[1:4]
with open(trace_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

matched = (
    data.get("restricted_model_path") == expected_model
    and data.get("prompt") == expected_prompt
)
  print("1" if matched else "0")
PY
  return 0
}

run_builds() {
  echo "[tier] compiling workspace tools"
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-engine)
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-bench --bin restricted_prefill_trace)
}

print_run_summary() {
  echo "[tier] tier=$TIER compare_mode=$COMPARE_MODE compare_only=$COMPARE_ONLY"
  echo "[tier] model=$MODEL_PATH trace_json=$TRACE_JSON oracle_output=$ORACLE_OUTPUT"
  if [[ -n "$SEED_LAYERS" ]]; then
    echo "[tier] seed_layers=$SEED_LAYERS"
  fi
  if [[ -n "$LOCAL_REPLAY_LAYER" ]]; then
    echo "[tier] local_replay=layer:$LOCAL_REPLAY_LAYER path:$LOCAL_REPLAY_PATH"
  fi
}

run_trace() {
  local trace_output="$1"
  echo "[tier] running restricted prefill trace"
  if [[ -n "$SEED_LAYERS" ]]; then
    (cd "$REPO_ROOT" && \
      CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 GPT_OSS_TRACE_SEED_LAYERS="$SEED_LAYERS" \
      target/release/restricted_prefill_trace \
      --model "$MODEL_PATH" \
      --prompt "$PROMPT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization 0.75 \
      --output "$trace_output" \
      --log-level info)
  else
    (cd "$REPO_ROOT" && \
      CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
      target/release/restricted_prefill_trace \
      --model "$MODEL_PATH" \
      --prompt "$PROMPT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization 0.75 \
      --output "$trace_output" \
      --log-level info)
  fi
}

run_oracle_compare() {
  local trace_output="$1"
  local oracle_output="$2"
  local python_bin=""

  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    echo "python3 is required for oracle comparison and is not installed in PATH" >&2
    exit 1
  fi

  echo "[tier] running oracle compare"
  "$python_bin" "$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py" \
    --cuda-trace-json "$trace_output" \
    --original-model "$ORIGINAL_MODEL" \
    --output "$oracle_output" \
    --compare-mode "$COMPARE_MODE" \
    ${LOCAL_REPLAY_LAYER:+--local-replay-layer "$LOCAL_REPLAY_LAYER"} \
    ${LOCAL_REPLAY_PATH:+--local-replay-path "$LOCAL_REPLAY_PATH"} \
    --device cpu

  if have_python3; then
  python3 - "$oracle_output" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    report = json.load(handle)

print("first_divergence_stage=", report.get('first_divergence_stage'))
print("compare_mode=", report.get('compare_mode'))
print("diff_count=", len(report.get('stage_diffs', [])))
PY
  fi
}

if [[ "$COMPARE_ONLY" == "1" ]]; then
  if [[ ! -f "$TRACE_JSON" ]]; then
    echo "compare-only requires existing trace artifact: $TRACE_JSON" >&2
    exit 1
  fi
  print_run_summary
  run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
  exit 0
fi

print_run_summary

if [[ "$TIER" == "0" || "$TIER" == "1" || "$TIER" == "2" ]]; then
  run_builds
fi

if [[ "$TIER" == "1" || "$TIER" == "2" ]]; then
  if [[ "$REUSE_TRACE" == "1" ]]; then
    can_reuse=$(can_reuse_trace "$TRACE_JSON" "$MODEL_PATH" "$PROMPT" | tr -d '[:space:]')
    if [[ "$can_reuse" == "1" ]]; then
      echo "[tier] reusing existing trace artifact: $TRACE_JSON"
    else
      run_trace "$TRACE_JSON"
    fi
  else
    run_trace "$TRACE_JSON"
  fi
fi

if [[ "$TIER" == "2" ]]; then
  run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
fi
