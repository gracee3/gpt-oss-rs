#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WRAPPER_SCRIPT="$REPO_ROOT/scripts/probe_validation_tier.sh"
ORACLE_TOOL="$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py"

resolve_test_python() {
  local candidate=""

  if [[ -n "${GPT_OSS_ORACLE_PYTHON:-}" ]] && "$GPT_OSS_ORACLE_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "$GPT_OSS_ORACLE_PYTHON"
    return 0
  fi

  candidate="/data/models/.venv-awq/bin/python"
  if [[ -x "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1; then
    echo "$candidate"
    return 0
  fi

  candidate="python3"
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import torch' >/dev/null 2>&1; then
    echo "$candidate"
    return 0
  fi

  echo "no python interpreter with torch is available for wrapper regression script" >&2
  exit 1
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"

  if [[ "$haystack" != *"$needle"* ]]; then
    echo "assertion failed: $label" >&2
    echo "missing substring: $needle" >&2
    exit 1
  fi
}

main() {
  local tmpdir=""
  local trace_current=""
  local trace_legacy=""
  local trace_bad=""
  local oracle_output=""
  local current_json=""
  local legacy_json=""
  local legacy_strict_json=""
  local bad_json=""
  local warm_stdout=""
  local python_bin=""

  python_bin="$(resolve_test_python)"
  export GPT_OSS_ORACLE_PYTHON="$python_bin"
  export GPT_OSS_ORACLE_TEST_MODE=1

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir:-}"' EXIT

  trace_current="$tmpdir/trace-current.json"
  trace_legacy="$tmpdir/trace-legacy.json"
  trace_bad="$tmpdir/trace-bad.json"
  oracle_output="$tmpdir/oracle-report.json"
  current_json="$tmpdir/current-inspect.json"
  legacy_json="$tmpdir/legacy-inspect.json"
  legacy_strict_json="$tmpdir/legacy-strict-inspect.json"
  bad_json="$tmpdir/bad-inspect.json"
  warm_stdout="$tmpdir/warm.out"

  cat > "$trace_current" <<'JSON'
{
  "prompt": "test prompt",
  "prompt_token_ids": [1, 2, 3],
  "restricted_model_path": "/tmp/restricted-model",
  "trace": {"layers": []}
}
JSON
  cp "$trace_current" "$trace_legacy"
  cp "$trace_current" "$trace_bad"

  (
    set -euo pipefail
    source "$WRAPPER_SCRIPT"
    MODEL_PATH="/tmp/restricted-model"
    PROMPT="test prompt"
    MAX_MODEL_LEN="128"
    TIER="2"
    COMPARE_MODE="runtime-emulated"
    SEED_LAYERS="12"
    LOCAL_REPLAY_LAYER="12"
    LOCAL_REPLAY_PATH="attention"

    TRACE_JSON="$trace_current"
    write_trace_metadata "$TRACE_JSON"

    TRACE_JSON="$trace_legacy"
    write_trace_metadata "$TRACE_JSON"
    python3 - "$TRACE_JSON.meta.json" <<'PY'
import json
import sys

meta_path = sys.argv[1]
with open(meta_path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)
for key in [
    "trace_capture_contract_version",
    "trace_capture_origin",
    "raw_trace_schema_status",
    "wrapper_capture_generation",
]:
    meta.pop(key, None)
with open(meta_path, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2)
    handle.write("\n")
PY

    TRACE_JSON="$trace_bad"
    write_trace_metadata "$TRACE_JSON"
    python3 - "$TRACE_JSON.meta.json" <<'PY'
import json
import sys

meta_path = sys.argv[1]
with open(meta_path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)
meta["max_model_len"] = 999
with open(meta_path, "w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2)
    handle.write("\n")
PY
  )

  "$WRAPPER_SCRIPT" \
    --inspect-trace-artifact \
    --trace-json "$trace_current" \
    --model /tmp/restricted-model \
    --prompt "test prompt" \
    --max-model-len 128 \
    --compare-mode runtime-emulated \
    --seed-layers 12 \
    --local-replay-layer 12 \
    --local-replay-path attention > "$current_json"

  "$WRAPPER_SCRIPT" \
    --inspect-trace-artifact \
    --trace-json "$trace_legacy" \
    --model /tmp/restricted-model \
    --prompt "test prompt" \
    --max-model-len 128 \
    --compare-mode runtime-emulated \
    --seed-layers 12 \
    --local-replay-layer 12 \
    --local-replay-path attention > "$legacy_json"

  if "$WRAPPER_SCRIPT" \
    --inspect-trace-artifact \
    --require-current-trace-contract \
    --trace-json "$trace_legacy" \
    --model /tmp/restricted-model \
    --prompt "test prompt" \
    --max-model-len 128 \
    --compare-mode runtime-emulated \
    --seed-layers 12 \
    --local-replay-layer 12 \
    --local-replay-path attention > "$legacy_strict_json" 2>/dev/null; then
    echo "legacy artifact unexpectedly passed strict inspection" >&2
    exit 1
  fi

  if "$WRAPPER_SCRIPT" \
    --inspect-trace-artifact \
    --trace-json "$trace_bad" \
    --model /tmp/restricted-model \
    --prompt "test prompt" \
    --max-model-len 128 \
    --compare-mode runtime-emulated \
    --seed-layers 12 \
    --local-replay-layer 12 \
    --local-replay-path attention > "$bad_json" 2>/dev/null; then
    echo "incompatible artifact unexpectedly passed inspection" >&2
    exit 1
  fi

  "$WRAPPER_SCRIPT" \
    --compare-only \
    --require-current-trace-contract \
    --trace-json "$trace_current" \
    --oracle-output "$oracle_output" \
    --original-model /tmp/original-model \
    --model /tmp/restricted-model \
    --prompt "test prompt" \
    --max-model-len 128 \
    --compare-mode runtime-emulated \
    --seed-layers 12 \
    --local-replay-layer 12 \
    --local-replay-path attention \
    --warm-oracle > "$warm_stdout" 2>&1

  python3 - "$current_json" "$legacy_json" "$legacy_strict_json" "$bad_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    current = json.load(handle)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    legacy = json.load(handle)
with open(sys.argv[3], "r", encoding="utf-8") as handle:
    legacy_strict = json.load(handle)
with open(sys.argv[4], "r", encoding="utf-8") as handle:
    bad = json.load(handle)

assert current["artifact_classification"] == "current-wrapper-captured"
assert current["reusable_for_requested_run"] is True
assert legacy["artifact_classification"] == "legacy"
assert "not upgraded" in legacy["note"]
assert legacy_strict["artifact_classification"] == "legacy"
assert legacy_strict["reusable_for_requested_run"] is False
assert any("--require-current-trace-contract" in reason for reason in legacy_strict["reasons"])
assert bad["reusable_for_requested_run"] is False
assert any("max-model-len mismatch" in reason for reason in bad["reasons"])
print("current_wrapper_captured=ok")
print("legacy_artifact_note=ok")
print("legacy_strict_rejection=ok")
print("incompatible_artifact_reason=ok")
PY

  assert_contains "$(cat "$warm_stdout")" "warm_oracle first_request session_reused=False" "warm oracle first request"
  assert_contains "$(cat "$warm_stdout")" "warm_oracle second_request session_reused=True" "warm oracle second request"
  assert_contains "$(cat "$warm_stdout")" "warm_oracle repeated_request_reports_match=1" "warm oracle report match"

  python3 - "$oracle_output" "$(dirname "$oracle_output")/oracle-report.warm-reuse-check.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["compare_mode"] == "runtime-emulated"
    assert report["local_replay"]["test_mode"] is True
print("warm_oracle_test_mode=ok")
PY

  echo "wrapper_regression: current_wrapper_captured=ok"
  echo "wrapper_regression: legacy_artifact_note=ok"
  echo "wrapper_regression: legacy_strict_rejection=ok"
  echo "wrapper_regression: incompatible_artifact_reason=ok"
  echo "wrapper_regression: warm_oracle_test_mode=ok"
}

main "$@"
