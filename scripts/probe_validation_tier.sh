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
  --warm-oracle
      opt-in: run the current compare request twice through one warm oracle session and emit a reuse-check artifact
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

  4) Warm-oracle reuse check for the current compare request:
     ./scripts/probe_validation_tier.sh \
       --compare-only \
       --compare-mode runtime-emulated \
       --warm-oracle
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
WARM_ORACLE="0"

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
    --warm-oracle)
      WARM_ORACLE="1"
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

if [[ "$WARM_ORACLE" == "1" && "$TIER" != "2" ]]; then
  usage_error "--warm-oracle is only available for Tier 2 / compare-only runs"
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

resolve_oracle_python() {
  local candidate=""

  if [[ -n "${GPT_OSS_ORACLE_PYTHON:-}" ]]; then
    candidate="$GPT_OSS_ORACLE_PYTHON"
    if "$candidate" -c 'import torch' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
    echo "GPT_OSS_ORACLE_PYTHON does not have a working torch install: $candidate" >&2
    exit 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    candidate="python3"
    if "$candidate" -c 'import torch' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  fi

  candidate="/data/models/.venv-awq/bin/python"
  if [[ -x "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1; then
    echo "$candidate"
    return 0
  fi

  echo "no python interpreter with torch is available for oracle comparison; set GPT_OSS_ORACLE_PYTHON if needed" >&2
  exit 1
}

print_run_summary() {
  echo "[tier] tier=$TIER compare_mode=$COMPARE_MODE compare_only=$COMPARE_ONLY warm_oracle=$WARM_ORACLE"
  echo "[tier] model=$MODEL_PATH trace_json=$TRACE_JSON oracle_output=$ORACLE_OUTPUT"
  if [[ -n "$SEED_LAYERS" ]]; then
    echo "[tier] seed_layers=$SEED_LAYERS"
  fi
  if [[ -n "$LOCAL_REPLAY_LAYER" ]]; then
    echo "[tier] local_replay=layer:$LOCAL_REPLAY_LAYER path:$LOCAL_REPLAY_PATH"
  fi
}

reuse_check_output_path() {
  local oracle_output="$1"
  local stem="$oracle_output"
  if [[ "$oracle_output" == *.json ]]; then
    stem="${oracle_output%.json}"
  fi
  echo "${stem}.warm-reuse-check.json"
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
  local oracle_args=()

  python_bin="$(resolve_oracle_python)"

  echo "[tier] running oracle compare"
  if [[ -n "$LOCAL_REPLAY_LAYER" ]]; then
    oracle_args+=(--local-replay-layer "$LOCAL_REPLAY_LAYER")
  fi
  if [[ -n "$LOCAL_REPLAY_PATH" ]]; then
    oracle_args+=(--local-replay-path "$LOCAL_REPLAY_PATH")
  fi

  "$python_bin" "$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py" \
    --cuda-trace-json "$trace_output" \
    --original-model "$ORIGINAL_MODEL" \
    --output "$oracle_output" \
    --compare-mode "$COMPARE_MODE" \
    "${oracle_args[@]}" \
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

build_oracle_compare_request() {
  local python_bin="$1"
  local trace_output="$2"
  local oracle_output="$3"

  "$python_bin" - "$trace_output" "$oracle_output" "$COMPARE_MODE" "$LOCAL_REPLAY_LAYER" "$LOCAL_REPLAY_PATH" <<'PY'
import json
import sys

trace_output, oracle_output, compare_mode, local_replay_layer, local_replay_path = sys.argv[1:6]
request = {
    "op": "compare",
    "cuda_trace_json": trace_output,
    "output": oracle_output,
    "compare_mode": compare_mode,
}
if local_replay_layer:
    request["local_replay_layer"] = int(local_replay_layer)
if local_replay_path:
    request["local_replay_path"] = local_replay_path
print(json.dumps(request))
PY
}

compare_json_reports() {
  local python_bin="$1"
  local lhs="$2"
  local rhs="$3"

  "$python_bin" - "$lhs" "$rhs" <<'PY'
import json
import sys

lhs_path, rhs_path = sys.argv[1:3]
with open(lhs_path, "r", encoding="utf-8") as lhs_handle:
    lhs = json.load(lhs_handle)
with open(rhs_path, "r", encoding="utf-8") as rhs_handle:
    rhs = json.load(rhs_handle)
print("1" if lhs == rhs else "0")
PY
}

run_warm_oracle_compare() {
  local trace_output="$1"
  local oracle_output="$2"
  local python_bin=""
  local oracle_tool=""
  local reuse_output=""
  local request_primary=""
  local request_reuse=""
  local response_primary=""
  local response_reuse=""
  local response_shutdown=""
  local reports_match=""

  python_bin="$(resolve_oracle_python)"
  oracle_tool="$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py"
  reuse_output="$(reuse_check_output_path "$oracle_output")"
  request_primary="$(build_oracle_compare_request "$python_bin" "$trace_output" "$oracle_output")"
  request_reuse="$(build_oracle_compare_request "$python_bin" "$trace_output" "$reuse_output")"

  echo "[tier] running warm oracle compare"
  echo "[tier] warm_oracle primary_output=$oracle_output reuse_output=$reuse_output"

  coproc WARM_ORACLE_LISTENER {
    "$python_bin" "$oracle_tool" \
      --listen \
      --original-model "$ORIGINAL_MODEL" \
      --device cpu
  }

  printf '%s\n' "$request_primary" >&"${WARM_ORACLE_LISTENER[1]}"
  IFS= read -r response_primary <&"${WARM_ORACLE_LISTENER[0]}"

  printf '%s\n' "$request_reuse" >&"${WARM_ORACLE_LISTENER[1]}"
  IFS= read -r response_reuse <&"${WARM_ORACLE_LISTENER[0]}"

  printf '%s\n' '{"op":"shutdown"}' >&"${WARM_ORACLE_LISTENER[1]}"
  exec {WARM_ORACLE_LISTENER[1]}>&-
  IFS= read -r response_shutdown <&"${WARM_ORACLE_LISTENER[0]}"
  wait "$WARM_ORACLE_LISTENER_PID"

  if have_python3; then
    python3 - "$response_primary" "$response_reuse" "$response_shutdown" <<'PY'
import json
import sys

primary = json.loads(sys.argv[1])
reuse = json.loads(sys.argv[2])
shutdown = json.loads(sys.argv[3])

print(
    "[tier] warm_oracle first_request"
    f" session_reused={primary.get('session_reused')}"
    f" session_load_count={primary.get('session_load_count')}"
    f" output={primary.get('output')}"
)
print(
    "[tier] warm_oracle second_request"
    f" session_reused={reuse.get('session_reused')}"
    f" session_load_count={reuse.get('session_load_count')}"
    f" output={reuse.get('output')}"
)
print(f"[tier] warm_oracle shutdown_ok={shutdown.get('shutdown')}")
PY
  fi

  reports_match="$(compare_json_reports "$python_bin" "$oracle_output" "$reuse_output" | tr -d '[:space:]')"
  if [[ "$reports_match" != "1" ]]; then
    echo "[tier] warm_oracle repeated request reports diverged between $oracle_output and $reuse_output" >&2
    exit 1
  fi
  echo "[tier] warm_oracle repeated_request_reports_match=1"

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
  if [[ "$WARM_ORACLE" == "1" ]]; then
    run_warm_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
  else
    run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
  fi
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
  if [[ "$WARM_ORACLE" == "1" ]]; then
    run_warm_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
  else
    run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
  fi
fi
