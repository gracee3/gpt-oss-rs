#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_gpu0_live_smoke.sh"

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
  local prompt_file=""
  local setup_output=""
  local run_output=""
  local binary_path=""

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir:-}"' EXIT

  tree="$tmpdir/tree"
  model="$tmpdir/model"
  prompt_file="$tmpdir/prompt.txt"
  setup_output="$tmpdir/setup-output"
  run_output="$tmpdir/run-output"
  binary_path="$tree/target/release/restricted_prefill_topk"

  mkdir -p "$tree/crates/gpt-oss-bench/src/bin" "$tree/target/release" "$model"
  printf '// fake restricted_prefill_topk bin for harness regression\n' \
    >"$tree/crates/gpt-oss-bench/src/bin/restricted_prefill_topk.rs"
  printf '{}\n' >"$model/config.json"
  printf '{}\n' >"$model/RESTRICTED_MODEL_VIEW.json"
  printf 'Explain tensor parallelism in one short sentence.\n' >"$prompt_file"

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
payload = {
    "model": model,
    "prompt": prompt,
    "max_model_len": max_model_len,
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "probe_mode": os.environ.get("GPT_OSS_PROBE_MODE"),
    "test_label": os.environ.get("TEST_LABEL"),
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
EOF
  chmod +x "$binary_path"

  "$RUNNER_SCRIPT" \
    --setup-only \
    --gpu 7 \
    --tree "$tree" \
    --model "$model" \
    --prompt-file "$prompt_file" \
    --timeout 111 \
    --build-timeout 222 \
    --output-dir "$setup_output" \
    --bin restricted_prefill_topk \
    --max-model-len 4608 \
    --env GPT_OSS_PROBE_MODE=layer0-mlp \
    --env TEST_LABEL=live-smoke >/dev/null

  assert_file "$setup_output/summary.json"
  assert_file "$setup_output/plan/live_env.sh"
  assert_file "$setup_output/plan/live_env.json"
  assert_file "$setup_output/plan/build_command.sh"
  assert_file "$setup_output/plan/run_command.sh"

  python3 - "$setup_output" "$tree" "$model" "$prompt_file" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
tree = Path(sys.argv[2])
model = Path(sys.argv[3])
prompt_file = Path(sys.argv[4])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
assert summary["build"] is None
assert summary["run"] is None
setup = summary["setup"]
assert setup["gpu_id"] == "7"
assert setup["tree"] == str(tree)
assert setup["model_path"] == str(model)
assert setup["prompt_file"] == str(prompt_file)
assert setup["live_bin"] == "restricted_prefill_topk"
assert setup["artifact_path"].endswith("/run/restricted_prefill_topk.json")
assert setup["timeout_seconds"] == 111
assert setup["build_timeout_seconds"] == 222
assert setup["max_model_len"] == 4608

envs = json.loads(Path(setup["env_json_file"]).read_text(encoding="utf-8"))
assert envs == [
    {"key": "GPT_OSS_PROBE_MODE", "value": "layer0-mlp"},
    {"key": "TEST_LABEL", "value": "live-smoke"},
]

env_file = (output_dir / "plan" / "live_env.sh").read_text(encoding="utf-8")
assert "export GPT_OSS_PROBE_MODE=layer0-mlp" in env_file
assert "export TEST_LABEL=live-smoke" in env_file

build_command = (output_dir / "plan" / "build_command.sh").read_text(encoding="utf-8")
assert "cargo build --release -p gpt-oss-bench --features cuda --bin restricted_prefill_topk" in build_command
assert "CUDA_VISIBLE_DEVICES=7" in build_command

run_command = (output_dir / "plan" / "run_command.sh").read_text(encoding="utf-8")
assert str(prompt_file) in run_command
assert str(model) in run_command
assert "--max-model-len 4608" in run_command
assert str(tree / "target" / "release" / "restricted_prefill_topk") in run_command

print("setup_summary=ok")
print("setup_env_plan=ok")
print("setup_command_plan=ok")
PY

  "$RUNNER_SCRIPT" \
    --run-only \
    --gpu 7 \
    --tree "$tree" \
    --model "$model" \
    --prompt-file "$prompt_file" \
    --timeout 111 \
    --output-dir "$run_output" \
    --bin restricted_prefill_topk \
    --max-model-len 4608 \
    --env GPT_OSS_PROBE_MODE=layer0-mlp \
    --env TEST_LABEL=live-smoke >/dev/null

  assert_file "$run_output/summary.json"
  assert_file "$run_output/run/run.stdout"
  assert_file "$run_output/run/run.stderr"
  assert_file "$run_output/run/restricted_prefill_topk.json"

  python3 - "$run_output" "$binary_path" "$model" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
binary_path = Path(sys.argv[2])
model = Path(sys.argv[3])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
assert summary["build"] is None
run = summary["run"]
assert run["state"] == "completed"
assert run["artifact_state"] == "present"
assert run["resolved_binary_path"] == str(binary_path)

artifact_path = Path(run["artifact_path"])
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
assert artifact["model"] == str(model)
assert artifact["prompt"] == "Explain tensor parallelism in one short sentence."
assert artifact["max_model_len"] == "4608"
assert artifact["visible_devices"] == "7"
assert artifact["probe_mode"] == "layer0-mlp"
assert artifact["test_label"] == "live-smoke"

run_stdout = (output_dir / "run" / "run.stdout").read_text(encoding="utf-8")
assert '"probe_mode": "layer0-mlp"' in run_stdout
assert '"test_label": "live-smoke"' in run_stdout

print("run_summary=ok")
print("run_artifact=ok")
print("run_stdout=ok")
PY

  echo "gpu0_live_smoke_regression: setup_summary=ok"
  echo "gpu0_live_smoke_regression: setup_env_plan=ok"
  echo "gpu0_live_smoke_regression: setup_command_plan=ok"
  echo "gpu0_live_smoke_regression: run_summary=ok"
  echo "gpu0_live_smoke_regression: run_artifact=ok"
  echo "gpu0_live_smoke_regression: run_stdout=ok"
}

main "$@"
