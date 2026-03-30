#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $(basename \"$0\") [--tier 0|1|2] [options]"
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
  --prompt <text>
      prompt text used for trace capture
  --max-model-len <n>
      model max length for trace capture (default: 128)
  --gpu <idx>
      CUDA_VISIBLE_DEVICES index (default: 1)
  --compare-mode <full|fast>
      oracle compare mode used in tier 2 compare phase (default: full)
  --reuse-key-dump <path>
      explicit reuse key metadata file path (defaults to <trace-json>.reuse-key.json)
  --no-reuse
      force trace rerun even if existing artifact is reusable
  --compare-only
      run oracle compare only (uses existing trace file, skips compile/trace)
  -h, --help
      show help
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TIER="0"
MODEL_PATH="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
TRACE_JSON="$REPO_ROOT/.live/restricted-cuda-prefill-trace.integration.json"
ORACLE_OUTPUT="$REPO_ROOT/.live/restricted-prefill-trace-diff.integration.json"
ORIGINAL_MODEL="/data/models/openai/gpt-oss-20b"
PROMPT="Explain tensor parallelism in one short sentence."
MAX_MODEL_LEN="128"
GPU="1"
REUSE_TRACE="1"
COMPARE_ONLY="0"
COMPARE_MODE="full"
REUSE_KEY_JSON=""

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
    --compare-mode)
      COMPARE_MODE="$2"
      if [[ "$COMPARE_MODE" != "full" && "$COMPARE_MODE" != "fast" ]]; then
        echo "Invalid --compare-mode: $COMPARE_MODE" >&2
        exit 1
      fi
      shift 2
      ;;
    --reuse-key-dump)
      REUSE_KEY_JSON="$2"
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
  echo "Invalid tier: $TIER" >&2
  exit 1
fi

have_python3() {
  command -v python3 >/dev/null 2>&1
}

reuse_key_path() {
  local trace_path="$1"
  if [[ -n "$REUSE_KEY_JSON" ]]; then
    printf '%s\n' "$REUSE_KEY_JSON"
    return
  fi
  printf '%s\n' "${trace_path}.reuse-key.json"
}

build_reuse_key() {
  local trace_path="$1"
  local compare_mode="$2"
  local model_path="$3"
  local prompt="$4"
  local max_model_len="$5"
  local oracle_tool="$6"

  if ! have_python3; then
    return 1
  fi

  python3 - "$trace_path" "$compare_mode" "$model_path" "$prompt" "$max_model_len" "$ORACLE_OUTPUT" "$oracle_tool" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

_prog, trace_path, compare_mode, model_path, prompt, max_model_len, _, oracle_tool = sys.argv
with open(trace_path, "r", encoding="utf-8") as handle:
    trace = json.load(handle)

def trace_schema_sig(data: dict) -> str:
    trace_obj = data.get("trace", {})
    schema = {
        "top_level_keys": sorted(data.keys()),
        "trace_type": type(trace_obj).__name__,
        "trace_has_layers": bool(trace_obj and isinstance(trace_obj, dict) and trace_obj.get("layers") is not None),
        "trace_layer_count": len(trace_obj.get("layers", [])),
        "trace_top_keys": sorted((trace_obj or {}).keys()),
    }
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]

def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""

config_path = Path(model_path) / "config.json"
model_artifact_hash = _hash_file(config_path)
weight_hash = ""
weights = sorted(
    p.as_posix()
    for p in Path(model_path).glob("*")
    if p.is_file() and p.suffix in {".safetensors", ".json", ".bin", ".pt", ".pth"}
)
if weights:
    for path in weights:
        if Path(path).name == "config.json":
            continue
        weight_hash = hashlib.sha256((weight_hash + _hash_file(Path(path))).encode()).hexdigest()

oracle_tool_path = Path(oracle_tool)
oracle_schema = hashlib.sha256(("restricted_oracle_prefill_trace.v2::" + oracle_tool_path.read_text(encoding="utf-8")).encode()).hexdigest()[:16]

key = {
    "restricted_model_path": trace.get("restricted_model_path"),
    "prompt_id": hashlib.sha256(prompt.encode()).hexdigest(),
    "prompt_token_count": len(trace.get("prompt_token_ids", [])),
    "max_model_len": int(max_model_len),
    "compare_mode": compare_mode,
    "trace_schema_version": f"restricted_prefill_trace.schema.{trace_schema_sig(trace)}",
    "restricted_model_artifact_hash": f"{model_artifact_hash[:16]}.{weight_hash[:16]}",
    "oracle_schema": oracle_schema,
    "compare_flags": {
        "compare_mode": compare_mode,
        "tier": 2,
    },
    "max_reuse_version": 1,
}

print(json.dumps(key, sort_keys=True))
PY
}

reuse_key_matches() {
  local trace_path="$1"
  local compare_mode="$2"

  if [[ ! -f "$trace_path" ]]; then
    echo "0" "reason=missing_trace"
    return 0
  fi

  local metadata_path
  metadata_path=$(reuse_key_path "$trace_path")
  if [[ ! -f "$metadata_path" ]]; then
    echo "0" "reason=reuse_key_missing"
    return 0
  fi

  if ! have_python3; then
    echo "0" "reason=no_python3"
    return 0
  fi

  local current_key
  local oracle_tool=""
  oracle_tool="$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py"
  current_key=$(build_reuse_key "$trace_path" "$compare_mode" "$MODEL_PATH" "$PROMPT" "$MAX_MODEL_LEN" "$oracle_tool")
  if [[ -z "$current_key" ]]; then
    echo "0" "reason=key_compute_failed"
    return 0
  fi

  if [[ "$(cat "$metadata_path")" != "$current_key" ]]; then
    mismatch_reason=$(python3 - "$metadata_path" "$current_key" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    stored = json.load(handle)
current = json.loads(sys.argv[2])

if stored.get("restricted_model_path") != current.get("restricted_model_path"):
    print("reuse_key_model_mismatch")
elif stored.get("restricted_model_artifact_hash") != current.get("restricted_model_artifact_hash"):
    print("reuse_key_model_hash_mismatch")
elif stored.get("prompt_id") != current.get("prompt_id"):
    print("reuse_key_prompt_mismatch")
elif stored.get("prompt_token_count") != current.get("prompt_token_count"):
    print("reuse_key_prompt_token_count_mismatch")
elif stored.get("max_model_len") != current.get("max_model_len"):
    print("reuse_key_max_model_len_mismatch")
elif stored.get("trace_schema_version") != current.get("trace_schema_version"):
    print("reuse_key_trace_schema_mismatch")
elif stored.get("oracle_schema") != current.get("oracle_schema"):
    print("reuse_key_oracle_schema_mismatch")
elif stored.get("compare_mode") != current.get("compare_mode"):
    print("reuse_key_compare_mode_mismatch")
else:
    print("full_key_match")
PY
)
    echo "0" "reason=$mismatch_reason"
    return 0
  fi

  match_reason=$(python3 - "$metadata_path" "$current_key" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    stored = json.load(handle)
current = json.loads(sys.argv[2])

for k in sorted(stored.keys()):
    if stored.get(k) != current.get(k):
        print("full_key_mismatch")
        raise SystemExit
print("full_key_match")
PY
)
  echo "1" "reason=$match_reason"
}

can_reuse_trace() {
  local trace_path="$1"
  local compare_mode="$2"

  local reuse_ok
  local reuse_reason
  read -r reuse_ok reuse_reason <<< "$(reuse_key_matches "$trace_path" "$compare_mode")"
  if [[ "$reuse_ok" == "1" ]]; then
    reuse_ok="true"
  else
    reuse_ok="false"
  fi
  echo "reuse_trace=${reuse_ok} ${reuse_reason}"
}

run_builds() {
  echo "[tier] compiling workspace tools"
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-engine)
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-bench --bin restricted_prefill_trace)
}

run_trace() {
  local trace_output="$1"
  echo "[tier] running restricted prefill trace"
  (cd "$REPO_ROOT" && \
    CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    target/release/restricted_prefill_trace \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.75 \
    --output "$trace_output" \
    --log-level info)
}

write_reuse_key() {
  local trace_output="$1"
  local compare_mode="$2"
  local metadata_path
  metadata_path=$(reuse_key_path "$trace_output")
  local key_json
  local oracle_tool="$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py"
  key_json=$(build_reuse_key "$trace_output" "$compare_mode" "$MODEL_PATH" "$PROMPT" "$MAX_MODEL_LEN" "$oracle_tool")
  printf '%s\n' "$key_json" > "$metadata_path"
}

run_oracle_compare() {
  local trace_output="$1"
  local oracle_output="$2"
  local compare_mode="$3"
  local python_bin=""

  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    echo "python3 is required for oracle comparison and is not installed in PATH" >&2
    exit 1
  fi

  echo "[tier] running oracle compare (mode=$compare_mode)"
  "$python_bin" "$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py" \
    --cuda-trace-json "$trace_output" \
    --original-model "$ORIGINAL_MODEL" \
    --output "$oracle_output" \
    --device cpu \
    --mode "$compare_mode"

  if have_python3; then
    python3 - "$oracle_output" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    report = json.load(handle)

print("first_divergence_stage=", report.get('first_divergence_stage'))
print("diff_count=", len(report.get('stage_diffs', [])))
PY
  fi
}

if [[ "$COMPARE_ONLY" == "1" ]]; then
  if [[ ! -f "$TRACE_JSON" ]]; then
    echo "compare-only requires existing trace artifact: $TRACE_JSON" >&2
    exit 1
  fi
  run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT" "$COMPARE_MODE"
  exit 0
fi

if [[ "$TIER" == "0" || "$TIER" == "1" || "$TIER" == "2" ]]; then
  run_builds
fi

if [[ "$TIER" == "1" || "$TIER" == "2" ]]; then
  if [[ "$REUSE_TRACE" == "1" ]]; then
    reuse_decision=$(can_reuse_trace "$TRACE_JSON" "$COMPARE_MODE")
    echo "[tier] $reuse_decision"
    if [[ "$reuse_decision" == reuse_trace=true* && "$reuse_decision" == *reason=full_key_match* ]]; then
      echo "[tier] reusing existing trace artifact: $TRACE_JSON"
    else
      run_trace "$TRACE_JSON"
      write_reuse_key "$TRACE_JSON" "$COMPARE_MODE"
    fi
  else
    run_trace "$TRACE_JSON"
    write_reuse_key "$TRACE_JSON" "$COMPARE_MODE"
  fi
fi

if [[ "$TIER" == "2" ]]; then
  run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT" "$COMPARE_MODE"
fi
