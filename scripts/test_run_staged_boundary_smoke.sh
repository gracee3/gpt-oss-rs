#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_staged_boundary_smoke.sh"

assert_file() {
  local path="$1"

  [[ -f "$path" ]] || {
    echo "expected file not found: $path" >&2
    exit 1
  }
}

main() {
  local tmpdir=""
  local tree=""
  local model=""
  local prefix_prompt=""
  local continuation_prompt=""
  local setup_output=""
  local run_output=""
  local binary_path=""

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir:-}"' EXIT

  tree="$tmpdir/tree"
  model="$tmpdir/model"
  prefix_prompt="$tmpdir/prefix.txt"
  continuation_prompt="$tmpdir/continuation.txt"
  setup_output="$tmpdir/setup-output"
  run_output="$tmpdir/run-output"
  binary_path="$tree/target/release/restricted_prefill_trace"

  mkdir -p "$tree/crates/gpt-oss-bench/src/bin" "$tree/target/release" "$model"
  printf '// fake restricted_prefill_trace bin for staged harness regression\n' \
    >"$tree/crates/gpt-oss-bench/src/bin/restricted_prefill_trace.rs"
  printf '{}\n' >"$model/config.json"
  printf '{}\n' >"$model/RESTRICTED_MODEL_VIEW.json"
  printf 'prefix boundary prompt\n' >"$prefix_prompt"
  printf 'continuation step prompt\n' >"$continuation_prompt"

  cat >"$binary_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

MODEL=""
PROMPT=""
MAX_MODEL_LEN=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:?missing value for --prompt}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?missing value for --max-model-len}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:?missing value for --output}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

python3 - "$OUTPUT" "$MODEL" "$PROMPT" "$MAX_MODEL_LEN" <<'PY'
import json
import os
import sys
from pathlib import Path

output, model, prompt, max_model_len = sys.argv[1:5]
proof_path = os.environ.get("GPT_OSS_PROOF_JSON")
payload = {
    "model": model,
    "prompt": prompt,
    "max_model_len": max_model_len,
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "probe_mode": os.environ.get("GPT_OSS_PROBE_MODE"),
    "test_label": os.environ.get("TEST_LABEL"),
    "proof_artifact_env": proof_path,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if proof_path and not Path(proof_path).exists():
    proof_payload = {
        "prompt": prompt,
        "proof_artifact_env": proof_path,
    }
    Path(proof_path).write_text(json.dumps(proof_payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
EOF
  chmod +x "$binary_path"

  "$RUNNER_SCRIPT" \
    --setup-only \
    --gpu 5 \
    --tree "$tree" \
    --model "$model" \
    --prefix-prompt-file "$prefix_prompt" \
    --continuation-prompt-file "$continuation_prompt" \
    --timeout 111 \
    --build-timeout 222 \
    --output-dir "$setup_output" \
    --bin restricted_prefill_trace \
    --max-model-len 4608 \
    --proof-artifact-env GPT_OSS_PROOF_JSON \
    --proof-artifact-name continuation-head.json \
    --env GPT_OSS_PROBE_MODE=layer0-mlp \
    --env TEST_LABEL=staged-boundary >/dev/null

  assert_file "$setup_output/summary.json"
  assert_file "$setup_output/plan/staged_env.sh"
  assert_file "$setup_output/plan/staged_env.json"
  assert_file "$setup_output/plan/prefix_command.sh"
  assert_file "$setup_output/plan/continuation_command.sh"

  python3 - "$setup_output" "$tree" "$model" "$prefix_prompt" "$continuation_prompt" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
tree = Path(sys.argv[2])
model = Path(sys.argv[3])
prefix_prompt = Path(sys.argv[4])
continuation_prompt = Path(sys.argv[5])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
assert summary["build"] is None
assert summary["stages"]["prefix"] is None
assert summary["stages"]["continuation"] is None

setup = summary["setup"]
assert setup["gpu_id"] == "5"
assert setup["tree"] == str(tree)
assert setup["model_path"] == str(model)
assert setup["stage_bin"] == "restricted_prefill_trace"
assert setup["prefix_prompt_file"] == str(prefix_prompt)
assert setup["continuation_prompt_file"] == str(continuation_prompt)
assert setup["proof_artifact_env"] == "GPT_OSS_PROOF_JSON"
assert setup["continuation_proof_artifact_path"].endswith("/continuation/continuation-head.json")
assert setup["timeout_seconds"] == 111
assert setup["build_timeout_seconds"] == 222
assert setup["max_model_len"] == 4608

envs = json.loads((output_dir / "plan" / "staged_env.json").read_text(encoding="utf-8"))
assert envs == [
    {"key": "GPT_OSS_PROBE_MODE", "value": "layer0-mlp"},
    {"key": "TEST_LABEL", "value": "staged-boundary"},
]

env_file = (output_dir / "plan" / "staged_env.sh").read_text(encoding="utf-8")
assert "export GPT_OSS_PROBE_MODE=layer0-mlp" in env_file
assert "export TEST_LABEL=staged-boundary" in env_file
assert "GPT_OSS_PROOF_JSON" not in env_file

prefix_command = (output_dir / "plan" / "prefix_command.sh").read_text(encoding="utf-8")
continuation_command = (output_dir / "plan" / "continuation_command.sh").read_text(encoding="utf-8")
assert str(prefix_prompt) in prefix_command
assert "unset GPT_OSS_PROOF_JSON" in prefix_command
assert "export GPT_OSS_PROOF_JSON=" not in prefix_command
assert str(continuation_prompt) in continuation_command
assert "unset GPT_OSS_PROOF_JSON" in continuation_command
assert "export GPT_OSS_PROOF_JSON=" in continuation_command
assert setup["continuation_proof_artifact_path"] in continuation_command

print("setup_summary=ok")
print("setup_env_scope=ok")
print("setup_command_scope=ok")
PY

  "$RUNNER_SCRIPT" \
    --run-only \
    --gpu 5 \
    --tree "$tree" \
    --model "$model" \
    --prefix-prompt-file "$prefix_prompt" \
    --continuation-prompt-file "$continuation_prompt" \
    --timeout 111 \
    --output-dir "$run_output" \
    --bin restricted_prefill_trace \
    --max-model-len 4608 \
    --proof-artifact-env GPT_OSS_PROOF_JSON \
    --proof-artifact-name continuation-head.json \
    --env GPT_OSS_PROBE_MODE=layer0-mlp \
    --env TEST_LABEL=staged-boundary >/dev/null

  assert_file "$run_output/summary.json"
  assert_file "$run_output/prefix/run.stdout"
  assert_file "$run_output/continuation/run.stdout"
  assert_file "$run_output/prefix/restricted_prefill_trace-prefix.json"
  assert_file "$run_output/continuation/restricted_prefill_trace-continuation.json"
  assert_file "$run_output/continuation/continuation-head.json"

  python3 - "$run_output" "$model" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
model = Path(sys.argv[2])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
assert summary["build"] is None
prefix = summary["stages"]["prefix"]
continuation = summary["stages"]["continuation"]

assert prefix["state"] == "completed"
assert prefix["artifact_state"] == "present"
assert prefix["proof_artifact_state"] == "not_requested"
assert prefix["proof_artifact_path"] == ""

assert continuation["state"] == "completed"
assert continuation["artifact_state"] == "present"
assert continuation["proof_artifact_state"] == "present"

prefix_artifact = json.loads(Path(prefix["artifact_path"]).read_text(encoding="utf-8"))
continuation_artifact = json.loads(Path(continuation["artifact_path"]).read_text(encoding="utf-8"))
proof_artifact = json.loads(Path(continuation["proof_artifact_path"]).read_text(encoding="utf-8"))

assert prefix_artifact["model"] == str(model)
assert prefix_artifact["prompt"] == "prefix boundary prompt"
assert prefix_artifact["proof_artifact_env"] is None
assert prefix_artifact["probe_mode"] == "layer0-mlp"
assert prefix_artifact["test_label"] == "staged-boundary"

assert continuation_artifact["model"] == str(model)
assert continuation_artifact["prompt"] == "continuation step prompt"
assert continuation_artifact["proof_artifact_env"] == continuation["proof_artifact_path"]
assert continuation_artifact["probe_mode"] == "layer0-mlp"
assert continuation_artifact["test_label"] == "staged-boundary"

assert proof_artifact["prompt"] == "continuation step prompt"
assert proof_artifact["proof_artifact_env"] == continuation["proof_artifact_path"]

prefix_stdout = (output_dir / "prefix" / "run.stdout").read_text(encoding="utf-8")
continuation_stdout = (output_dir / "continuation" / "run.stdout").read_text(encoding="utf-8")
assert '"proof_artifact_env": null' in prefix_stdout
assert continuation["proof_artifact_path"] in continuation_stdout

print("run_summary=ok")
print("run_stage_scope=ok")
print("run_proof_artifact=ok")
PY

  echo "staged_boundary_regression: setup_summary=ok"
  echo "staged_boundary_regression: setup_env_scope=ok"
  echo "staged_boundary_regression: setup_command_scope=ok"
  echo "staged_boundary_regression: run_summary=ok"
  echo "staged_boundary_regression: run_stage_scope=ok"
  echo "staged_boundary_regression: run_proof_artifact=ok"
}

main "$@"
