#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/oracle_debug_loop.sh [options]

Run repeated oracle compare passes against one trace for fast iterative debugging.

Options:
  --trace-json <path>
      trace path (default: .live/restricted-cuda-prefill-trace.integration.json)
  --model <path>
      restricted model view path (default: /data/models/openai/gpt-oss-20b-full-attn-restricted-integration)
  --original-model <path>
      original/oracle checkpoint path (default: /data/models/openai/gpt-oss-20b)
  --prompt <text>
      prompt text used for trace capture
  --max-model-len <n>
      max model len used to validate/create trace
  --gpu <idx>
      CUDA_VISIBLE_DEVICES index for trace generation (default: 1)
  --iterations <n>
      number of repeated compare runs (default: 5)
  --mode <full|fast>
      compare mode for oracle compare (default: full)
  --warm-oracle
      keep python oracle process warm across compares
  --reuse-trace
      generate trace if missing instead of failing
  --run-build
      run required cargo builds before trace generation
  --python <path>
      python interpreter (default: python3)
  -h, --help
      show help
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TRACE_JSON="$REPO_ROOT/.live/restricted-cuda-prefill-trace.integration.json"
MODEL_PATH="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
ORIGINAL_MODEL="/data/models/openai/gpt-oss-20b"
PROMPT="Explain tensor parallelism in one short sentence."
MAX_MODEL_LEN="128"
GPU="1"
ITERATIONS="5"
MODE="full"
WARM_ORACLE="0"
REUSE_TRACE="0"
RUN_BUILD="0"
PYTHON_BIN="python3"
ORACLE_TOOL="$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py"
ORACLE_BINARY="$REPO_ROOT/target/release/restricted_prefill_trace"
ORACLE_DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace-json)
      TRACE_JSON="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --original-model)
      ORIGINAL_MODEL="$2"
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
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      if [[ "$MODE" != "full" && "$MODE" != "fast" ]]; then
        echo "Invalid mode: $MODE" >&2
        exit 1
      fi
      shift 2
      ;;
    --warm-oracle)
      WARM_ORACLE="1"
      shift
      ;;
    --reuse-trace)
      REUSE_TRACE="1"
      shift
      ;;
    --run-build)
      RUN_BUILD="1"
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$RUN_BUILD" == "1" ]]; then
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-engine)
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-bench --bin restricted_prefill_trace)
fi

if [[ ! -x "$ORACLE_BINARY" ]]; then
  echo "missing restricted_prefill_trace binary: $ORACLE_BINARY" >&2
  echo "Run with --run-build if needed." >&2
  exit 1
fi

run_trace() {
  echo "[oracle_debug_loop] generating trace: $TRACE_JSON"
  (cd "$REPO_ROOT" && \
    CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    "$ORACLE_BINARY" \
      --model "$MODEL_PATH" \
      --prompt "$PROMPT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization 0.75 \
      --output "$TRACE_JSON" \
      --log-level info)
}

if [[ ! -f "$TRACE_JSON" ]]; then
  if [[ "$REUSE_TRACE" == "1" ]]; then
    run_trace
  else
    echo "Trace is missing and --reuse-trace was not provided: $TRACE_JSON" >&2
    exit 1
  fi
fi

TIMING_MS=()

json_first_divergence_from_file() {
  local report_path="$1"
  "$PYTHON_BIN" - "$report_path" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    report = json.load(handle)
print(report.get("first_divergence_stage", ""))
PY
}

summarize_compare_response() {
  local report_path="$1"
  local response_json="$2"
  local mode="$3"
  local warm="$4"
  local elapsed_ms="$5"

  local run_mode
  local report_mode
  run_mode="$("$PYTHON_BIN" - <<'PY' "$response_json"
import json
import sys
data = json.loads(sys.argv[1] or "{}")
print(data.get("compare_mode", ""))
PY
)"
  report_mode="$("$PYTHON_BIN" - <<'PY' "$response_json"
import json
import sys
data = json.loads(sys.argv[1] or "{}")
print(data.get("requested_mode", ""))
PY
)"
  local first_divergence
  first_divergence="$(json_first_divergence_from_file "$report_path")"
  echo "run elapsed_ms=$elapsed_ms mode=$run_mode requested_mode=$report_mode compare_mode=$mode warm_oracle=$warm first_divergence=${first_divergence:-<none>}"
}

run_one_shot() {
  local iteration="$1"
  local output_path="$2"
  local start_ns end_ns elapsed_ms
  local response_json

  start_ns=$(date +%s%N)
  response_json="$(
    "$PYTHON_BIN" -u "$ORACLE_TOOL" \
      --cuda-trace-json "$TRACE_JSON" \
      --original-model "$ORIGINAL_MODEL" \
      --output "$output_path" \
      --device "$ORACLE_DEVICE" \
      --mode "$MODE"
  )"
  end_ns=$(date +%s%N)
  elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
  TIMING_MS+=("$elapsed_ms")

  summarize_compare_response "$output_path" "$response_json" "$MODE" "false" "$elapsed_ms"
}

start_oracle_listener() {
  coproc ORACLE_LISTENER {
    "$PYTHON_BIN" -u "$ORACLE_TOOL" --listen --original-model "$ORIGINAL_MODEL" --device "$ORACLE_DEVICE"
  }
  ORACLE_LISTENER_PID=$!
  if ! read -r -u "${ORACLE_LISTENER[0]}" ORACLE_STARTUP; then
    echo "failed to start oracle listener" >&2
    exit 1
  fi
}

stop_oracle_listener() {
  if [[ -z "${ORACLE_LISTENER_PID:-}" ]]; then
    return
  fi
  echo '{"op":"shutdown"}' >&"${ORACLE_LISTENER[1]}"
  if read -r -u "${ORACLE_LISTENER[0]}" _; then
    :
  fi
  wait "$ORACLE_LISTENER_PID" || true
  ORACLE_LISTENER_PID=""
}

listener_send() {
  local request_json="$1"
  local response
  local start_ns end_ns elapsed_ms

  printf '%s\n' "$request_json" >&"${ORACLE_LISTENER[1]}"
  if ! read -r -u "${ORACLE_LISTENER[0]}" response; then
    echo "oracle listener stopped unexpectedly" >&2
    return 1
  fi
  echo "$response"
}

run_warm_loop() {
  start_oracle_listener
  for ((i = 1; i <= ITERATIONS; i++)); do
    local output_path="${TRACE_JSON%.json}.${i}.oracle-diff.json"
    local request
    local response
    local start_ns end_ns elapsed_ms
    local first_divergence
    local err_msg

    request=$(cat <<EOF
{"op":"compare","trace_json":"$TRACE_JSON","mode":"$MODE","output":"$output_path","original_model":"$ORIGINAL_MODEL"}
EOF
    )
    start_ns=$(date +%s%N)
    if ! response="$(listener_send "$request")"; then
      err_msg="listener_error"
      echo "run $i: fail $err_msg"
      TIMING_MS+=("")
      continue
    fi
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    TIMING_MS+=("$elapsed_ms")
    summarize_compare_response "$output_path" "$response" "$MODE" "true" "$elapsed_ms"
  done
  stop_oracle_listener
}

trap 'stop_oracle_listener' EXIT

if [[ "$WARM_ORACLE" == "1" ]]; then
  run_warm_loop
else
  for ((i = 1; i <= ITERATIONS; i++)); do
    output="${TRACE_JSON%.json}.${i}.oracle-diff.json"
    run_one_shot "$i" "$output"
  done
fi

if [[ "${#TIMING_MS[@]}" -eq 0 ]]; then
  echo "No successful runs to summarize" >&2
  exit 1
fi

python3 - "$ORACLE_TOOL" - <<'PY' "${TIMING_MS[@]}"
import statistics
import sys

values = [int(x) for x in sys.argv[2:] if str(x).strip().isdigit()]
if not values:
    print("mean_ms=nan")
    print("median_ms=nan")
    print("runs=0")
    raise SystemExit
mean = statistics.mean(values)
median = statistics.median(values)
print(f"mean_ms={mean:.3f}")
print(f"median_ms={median:.3f}")
print(f"runs={len(values)}")
PY
echo "[oracle_debug_loop] mode=$MODE warm_oracle=$WARM_ORACLE trace=$TRACE_JSON"
