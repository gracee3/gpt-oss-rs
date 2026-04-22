#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_yarn_long_context_proof.sh"

assert_file() {
  local path="$1"

  [[ -f "$path" ]] || {
    echo "expected file not found: $path" >&2
    exit 1
  }
}

write_fake_binary() {
  local path="$1"
  local side="$2"

  cat >"$path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SIDE="${side}"
MODEL=""
PROMPT=""
MAX_MODEL_LEN=""
OUTPUT=""
LOG_LEVEL=""

while [[ \$# -gt 0 ]]; do
  case "\$1" in
    --model)
      MODEL="\${2:?missing value for --model}"
      shift 2
      ;;
    --prompt)
      PROMPT="\${2:?missing value for --prompt}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="\${2:?missing value for --max-model-len}"
      shift 2
      ;;
    --output)
      OUTPUT="\${2:?missing value for --output}"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="\${2:?missing value for --log-level}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

python3 - "\$OUTPUT" "\$MODEL" "\$PROMPT" "\$MAX_MODEL_LEN" "\$LOG_LEVEL" "\$SIDE" <<'PY'
import json
import os
import sys
from pathlib import Path

output, model, prompt, max_model_len, log_level, side = sys.argv[1:7]
proof_path = os.environ.get("GPT_OSS_PROOF_JSON")
outer = {
    "side": side,
    "model": model,
    "prompt": prompt,
    "max_model_len": max_model_len,
    "log_level": log_level,
    "proof_mode": os.environ.get("PROOF_MODE"),
    "test_label": os.environ.get("TEST_LABEL"),
    "proof_artifact_env": proof_path,
    "prompt_token_ids": [101, 102, 103],
    "conclusion": f"{side}-outer",
    "first_mismatch_boundary": f"{side}-boundary",
}
Path(output).write_text(json.dumps(outer, indent=2) + "\n", encoding="utf-8")
if proof_path:
    values_head = [1.0, 2.0, 3.0] if side == "safe" else [1.0, 2.5, 3.0]
    proof = {
        "side": side,
        "proof_mode": os.environ.get("PROOF_MODE"),
        "test_label": os.environ.get("TEST_LABEL"),
        "proof_artifact_env": proof_path,
        "values_head": values_head,
    }
    Path(proof_path).write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "side": side,
    "proof_artifact_env": proof_path,
    "proof_mode": os.environ.get("PROOF_MODE"),
    "test_label": os.environ.get("TEST_LABEL"),
}, indent=2))
PY
EOF
  chmod +x "$path"
}

main() {
  local tmpdir=""
  local safe_tree=""
  local variant_tree=""
  local model=""
  local pyshim=""
  local setup_output=""
  local run_output=""
  local safe_binary=""
  local variant_binary=""

  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir:-}"' EXIT

  safe_tree="$tmpdir/safe-tree"
  variant_tree="$tmpdir/variant-tree"
  model="$tmpdir/model"
  pyshim="$tmpdir/pyshim"
  setup_output="$tmpdir/setup-output"
  run_output="$tmpdir/run-output"
  safe_binary="$safe_tree/target/release/restricted_logit_diff"
  variant_binary="$variant_tree/target/release/restricted_logit_diff"

  mkdir -p \
    "$safe_tree/crates/gpt-oss-bench/src/bin" \
    "$safe_tree/target/release" \
    "$variant_tree/crates/gpt-oss-bench/src/bin" \
    "$variant_tree/target/release" \
    "$model" \
    "$pyshim/transformers"

  printf '// fake restricted_logit_diff bin for yarn harness regression\n' \
    >"$safe_tree/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs"
  cp \
    "$safe_tree/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs" \
    "$variant_tree/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs"

  cat >"$model/config.json" <<'EOF'
{
  "layer_types": ["full_attention", "full_attention"],
  "sliding_window": 0
}
EOF
  cat >"$model/RESTRICTED_MODEL_VIEW.json" <<'EOF'
{
  "source_model": "fake-source-model",
  "kind": "restricted-test-view",
  "notes": [
    "self_attn.sinks disabled for harness regression coverage"
  ]
}
EOF

  cat >"$pyshim/transformers/__init__.py" <<'EOF'
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        return list(range(5001))
EOF

  write_fake_binary "$safe_binary" safe
  write_fake_binary "$variant_binary" variant

  export PYTHONPATH="$pyshim${PYTHONPATH:+:$PYTHONPATH}"

  "$RUNNER_SCRIPT" \
    --setup-only \
    --gpu 4 \
    --safe-tree "$safe_tree" \
    --variant-tree "$variant_tree" \
    --model "$model" \
    --output-dir "$setup_output" \
    --proof-artifact-env GPT_OSS_PROOF_JSON \
    --proof-artifact-name post_attention_probe.json \
    --compare-vector-key values_head \
    --env PROOF_MODE=compact \
    --env TEST_LABEL=yarn-proof >/dev/null

  assert_file "$setup_output/prompt/deterministic_over_4096.txt"
  assert_file "$setup_output/prompt/setup_model_check.json"
  assert_file "$setup_output/prompt/prompt_token_count.json"
  assert_file "$setup_output/plan/proof_env.sh"
  assert_file "$setup_output/plan/proof_env.json"
  assert_file "$setup_output/plan/safe_command.sh"
  assert_file "$setup_output/plan/variant_command.sh"
  assert_file "$setup_output/plan/safe_build_command.sh"
  assert_file "$setup_output/plan/variant_build_command.sh"
  assert_file "$setup_output/plan/setup_summary.json"

  python3 - "$setup_output" "$safe_tree" "$variant_tree" "$model" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
safe_tree = Path(sys.argv[2])
variant_tree = Path(sys.argv[3])
model = Path(sys.argv[4])

setup = json.loads((output_dir / "plan" / "setup_summary.json").read_text(encoding="utf-8"))
assert setup["gpu_id"] == "4"
assert setup["safe_tree"] == str(safe_tree)
assert setup["variant_tree"] == str(variant_tree)
assert setup["model_path"] == str(model)
assert setup["proof_bin"] == "restricted_logit_diff"
assert setup["artifact_name"] == "restricted_logit_diff.json"
assert setup["proof_artifact_env"] == "GPT_OSS_PROOF_JSON"
assert setup["proof_artifact_name"] == "post_attention_probe.json"
assert setup["compare_vector_key"] == "values_head"
assert setup["safe_proof_artifact_path"].endswith("/safe/post_attention_probe.json")
assert setup["variant_proof_artifact_path"].endswith("/variant/post_attention_probe.json")
assert setup["timeout_seconds"] == 1800
assert setup["build_timeout_seconds"] == 1800
assert setup["max_model_len"] == 4608

prompt_file = Path(setup["prompt_file"])
prompt_text = prompt_file.read_text(encoding="utf-8")
assert prompt_text.startswith("token-0000 token-0001 token-0002")

model_check = json.loads((output_dir / "prompt" / "setup_model_check.json").read_text(encoding="utf-8"))
assert model_check["sink_free_checks"]["full_attention_only"] is True
assert model_check["sink_free_checks"]["sliding_window_zero"] is True
assert model_check["sink_free_checks"]["sink_override_note_present"] is True

token_summary = json.loads((output_dir / "prompt" / "prompt_token_count.json").read_text(encoding="utf-8"))
assert token_summary["token_count"] == 5001
assert token_summary["crosses_4096"] is True

envs = json.loads((output_dir / "plan" / "proof_env.json").read_text(encoding="utf-8"))
assert envs == [
    {"key": "PROOF_MODE", "value": "compact"},
    {"key": "TEST_LABEL", "value": "yarn-proof"},
    {"key": "GPT_OSS_PROOF_JSON_SAFE", "value": setup["safe_proof_artifact_path"]},
    {"key": "GPT_OSS_PROOF_JSON_VARIANT", "value": setup["variant_proof_artifact_path"]},
]

env_file = (output_dir / "plan" / "proof_env.sh").read_text(encoding="utf-8")
assert "export PROOF_MODE=compact" in env_file
assert "export TEST_LABEL=yarn-proof" in env_file
assert f'export GPT_OSS_PROOF_JSON_SAFE={setup["safe_proof_artifact_path"]}' in env_file
assert f'export GPT_OSS_PROOF_JSON_VARIANT={setup["variant_proof_artifact_path"]}' in env_file

safe_command = (output_dir / "plan" / "safe_command.sh").read_text(encoding="utf-8")
variant_command = (output_dir / "plan" / "variant_command.sh").read_text(encoding="utf-8")
assert str(safe_tree) in safe_command
assert str(variant_tree) in variant_command
assert f'export GPT_OSS_PROOF_JSON={setup["safe_proof_artifact_path"]}' in safe_command
assert f'export GPT_OSS_PROOF_JSON={setup["variant_proof_artifact_path"]}' in variant_command
assert str(prompt_file) in safe_command
assert str(prompt_file) in variant_command
assert "--max-model-len 4608" in safe_command
assert "--max-model-len 4608" in variant_command

print("setup_summary=ok")
print("setup_prompt_outputs=ok")
print("setup_env_scope=ok")
print("setup_command_scope=ok")
PY

  "$RUNNER_SCRIPT" \
    --run-only \
    --gpu 4 \
    --safe-tree "$safe_tree" \
    --variant-tree "$variant_tree" \
    --model "$model" \
    --output-dir "$run_output" \
    --proof-artifact-env GPT_OSS_PROOF_JSON \
    --proof-artifact-name post_attention_probe.json \
    --compare-vector-key values_head \
    --env PROOF_MODE=compact \
    --env TEST_LABEL=yarn-proof >/dev/null

  assert_file "$run_output/summary.json"
  assert_file "$run_output/safe/run.stdout"
  assert_file "$run_output/variant/run.stdout"
  assert_file "$run_output/safe/status.json"
  assert_file "$run_output/variant/status.json"
  assert_file "$run_output/safe/restricted_logit_diff.json"
  assert_file "$run_output/variant/restricted_logit_diff.json"
  assert_file "$run_output/safe/post_attention_probe.json"
  assert_file "$run_output/variant/post_attention_probe.json"

  python3 - "$run_output" "$model" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
model = Path(sys.argv[2])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
assert summary["safe_build"] is None
assert summary["variant_build"] is None

safe = summary["safe"]
variant = summary["variant"]
comparison = summary["comparison"]

assert safe["state"] == "completed"
assert variant["state"] == "completed"
assert safe["artifact_state"] == "usable_artifact"
assert variant["artifact_state"] == "usable_artifact"
assert safe["proof_artifact_env"] == "GPT_OSS_PROOF_JSON"
assert variant["proof_artifact_env"] == "GPT_OSS_PROOF_JSON"
assert safe["primary_artifact_file"] == safe["proof_artifact_file"]
assert variant["primary_artifact_file"] == variant["proof_artifact_file"]
assert safe["proof_artifact_file"].endswith("/safe/post_attention_probe.json")
assert variant["proof_artifact_file"].endswith("/variant/post_attention_probe.json")

safe_outer = json.loads(Path(safe["report_file"]).read_text(encoding="utf-8"))
variant_outer = json.loads(Path(variant["report_file"]).read_text(encoding="utf-8"))
safe_proof = json.loads(Path(safe["proof_artifact_file"]).read_text(encoding="utf-8"))
variant_proof = json.loads(Path(variant["proof_artifact_file"]).read_text(encoding="utf-8"))

assert safe_outer["model"] == str(model)
assert variant_outer["model"] == str(model)
assert safe_outer["proof_artifact_env"] == safe["proof_artifact_file"]
assert variant_outer["proof_artifact_env"] == variant["proof_artifact_file"]
assert safe_outer["proof_mode"] == "compact"
assert variant_outer["proof_mode"] == "compact"
assert safe_outer["test_label"] == "yarn-proof"
assert variant_outer["test_label"] == "yarn-proof"

assert safe_proof["proof_artifact_env"] == safe["proof_artifact_file"]
assert variant_proof["proof_artifact_env"] == variant["proof_artifact_file"]
assert safe_proof["values_head"] == [1.0, 2.0, 3.0]
assert variant_proof["values_head"] == [1.0, 2.5, 3.0]

assert comparison["comparable"] is True
assert comparison["safe_primary_artifact"] == safe["proof_artifact_file"]
assert comparison["variant_primary_artifact"] == variant["proof_artifact_file"]
assert comparison["json_equal"] is False
assert comparison["same_top_level_keys"] is True
assert comparison["vector_diff"]["vector_key"] == "values_head"
assert comparison["vector_diff"]["vector_length"] == 3
assert comparison["vector_diff"]["first_diff_index"] == 1
assert comparison["vector_diff"]["max_abs_diff"] == 0.5
assert round(comparison["vector_diff"]["mean_abs_diff"], 6) == round(1.0 / 6.0, 6)

safe_stdout = (output_dir / "safe" / "run.stdout").read_text(encoding="utf-8")
variant_stdout = (output_dir / "variant" / "run.stdout").read_text(encoding="utf-8")
assert safe["proof_artifact_file"] in safe_stdout
assert variant["proof_artifact_file"] in variant_stdout

print("run_summary=ok")
print("run_env_scope=ok")
print("run_proof_selection=ok")
print("run_vector_diff=ok")
PY

  echo "yarn_long_context_regression: setup_summary=ok"
  echo "yarn_long_context_regression: setup_prompt_outputs=ok"
  echo "yarn_long_context_regression: setup_env_scope=ok"
  echo "yarn_long_context_regression: setup_command_scope=ok"
  echo "yarn_long_context_regression: run_summary=ok"
  echo "yarn_long_context_regression: run_env_scope=ok"
  echo "yarn_long_context_regression: run_proof_selection=ok"
  echo "yarn_long_context_regression: run_vector_diff=ok"
}

main "$@"
