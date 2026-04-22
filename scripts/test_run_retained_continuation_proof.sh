#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_retained_continuation_proof.sh"

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
PREFIX_PROMPT_FILE=""
CONTINUATION_PROMPT_FILE=""
MAX_MODEL_LEN=""
OUTPUT=""
declare -a FORCED_OUTPUT_TOKENS=()

while [[ \$# -gt 0 ]]; do
  case "\$1" in
    --model)
      MODEL="\${2:?missing value for --model}"
      shift 2
      ;;
    --prefix-prompt-file)
      PREFIX_PROMPT_FILE="\${2:?missing value for --prefix-prompt-file}"
      shift 2
      ;;
    --continuation-prompt-file)
      CONTINUATION_PROMPT_FILE="\${2:?missing value for --continuation-prompt-file}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="\${2:?missing value for --max-model-len}"
      shift 2
      ;;
    --forced-output-tokens)
      FORCED_OUTPUT_TOKENS+=("\${2:?missing value for --forced-output-tokens}")
      shift 2
      ;;
    --output)
      OUTPUT="\${2:?missing value for --output}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

python3 - "\$OUTPUT" "\$MODEL" "\$PREFIX_PROMPT_FILE" "\$CONTINUATION_PROMPT_FILE" "\$MAX_MODEL_LEN" "\$SIDE" "\${FORCED_OUTPUT_TOKENS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

output, model, prefix_prompt_file, continuation_prompt_file, max_model_len, side, *forced_tokens = sys.argv[1:]
proof_path = os.environ.get("GPT_OSS_PROOF_JSON")
outer = {
    "side": side,
    "model": model,
    "prefix_prompt_file": prefix_prompt_file,
    "continuation_prompt_file": continuation_prompt_file,
    "max_model_len": max_model_len,
    "forced_output_tokens": forced_tokens,
    "proof_artifact_env": proof_path,
    "test_label": os.environ.get("TEST_LABEL"),
}
Path(output).write_text(json.dumps(outer, indent=2) + "\n", encoding="utf-8")
if proof_path:
    values_head = [1.0, 2.0] if side == "safe" else [1.0, 2.5]
    proof = {
        "side": side,
        "forced_output_tokens": forced_tokens,
        "proof_artifact_env": proof_path,
        "values_head": values_head,
    }
    Path(proof_path).write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "side": side,
    "forced_output_tokens": forced_tokens,
    "proof_artifact_env": proof_path,
}, indent=2))
PY

if [[ "\$SIDE" == "safe" ]]; then
  printf '%s\n' \
    "RETAINED_CHILD_START" \
    "RETAINED_CHILD_TOKENIZED" \
    "RETAINED_CHILD_BUILD_WORKER_BEGIN" \
    "RETAINED_CHILD_BUILD_WORKER_DONE" \
    "RETAINED_STEP_BEGIN" \
    "RETAINED_STEP_FORWARD_BEGIN" \
    "RETAINED_STEP_FORWARD_DONE" \
    "DECODE1_BEGIN" \
    "RETAINED_PROOF_ENTER" \
    "RETAINED_PROOF_CAPTURED"
else
  printf '%s\n' \
    "RETAINED_CHILD_START" \
    "RETAINED_CHILD_TOKENIZED" \
    "RETAINED_CHILD_BUILD_WORKER_BEGIN" \
    "RETAINED_CHILD_BUILD_WORKER_DONE" \
    "RETAINED_STEP_BEGIN" \
    "RETAINED_STEP_FORWARD_BEGIN" >&2
fi
EOF
  chmod +x "$path"
}

main() {
  local tmpdir=""
  local safe_tree=""
  local variant_tree=""
  local model=""
  local prefix_prompt=""
  local continuation_prompt=""
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
  prefix_prompt="$tmpdir/prefix.txt"
  continuation_prompt="$tmpdir/continuation.txt"
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

  printf '// fake retained restricted_logit_diff bin for harness regression\n' \
    >"$safe_tree/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs"
  cp \
    "$safe_tree/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs" \
    "$variant_tree/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs"

  printf '{}\n' >"$model/config.json"
  printf '{}\n' >"$model/RESTRICTED_MODEL_VIEW.json"
  printf 'prefix test prompt\n' >"$prefix_prompt"
  printf 'continuation token\n' >"$continuation_prompt"

  cat >"$pyshim/transformers/__init__.py" <<'EOF'
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        if "prefix" in text:
            return [11, 12, 13]
        if "continuation" in text:
            return [6602]
        return []
EOF

  write_fake_binary "$safe_binary" safe
  write_fake_binary "$variant_binary" variant

  export PYTHONPATH="$pyshim${PYTHONPATH:+:$PYTHONPATH}"

  "$RUNNER_SCRIPT" \
    --setup-only \
    --gpu 6 \
    --safe-tree "$safe_tree" \
    --variant-tree "$variant_tree" \
    --model "$model" \
    --prefix-prompt-file "$prefix_prompt" \
    --continuation-prompt-file "$continuation_prompt" \
    --bin restricted_logit_diff \
    --proof-artifact-env GPT_OSS_PROOF_JSON \
    --proof-artifact-name continuation-head.json \
    --compare-vector-key values_head \
    --marker-profile retained-debug-v1 \
    --emit-forced-output-tokens \
    --verify-tokenization \
    --python python3 \
    --required-prefix-token-count 3 \
    --required-continuation-token-count 1 \
    --max-model-len 8 \
    --output-dir "$setup_output" \
    --env TEST_LABEL=retained-proof >/dev/null

  assert_file "$setup_output/summary.json"
  assert_file "$setup_output/plan/setup_summary.json"
  assert_file "$setup_output/plan/tokenization_summary.json"
  assert_file "$setup_output/plan/marker_profiles.json"
  assert_file "$setup_output/plan/progress_markers.json"
  assert_file "$setup_output/plan/retained_env.sh"
  assert_file "$setup_output/plan/retained_env.json"
  assert_file "$setup_output/plan/safe_run_command.sh"
  assert_file "$setup_output/plan/variant_run_command.sh"

  python3 - "$setup_output" "$safe_tree" "$variant_tree" "$model" "$prefix_prompt" "$continuation_prompt" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
safe_tree = Path(sys.argv[2])
variant_tree = Path(sys.argv[3])
model = Path(sys.argv[4])
prefix_prompt = Path(sys.argv[5])
continuation_prompt = Path(sys.argv[6])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
assert summary["safe_build"] is None
assert summary["variant_build"] is None
assert summary["safe"] is None
assert summary["variant"] is None

setup = json.loads((output_dir / "plan" / "setup_summary.json").read_text(encoding="utf-8"))
assert setup["gpu_id"] == "6"
assert setup["safe_tree"] == str(safe_tree)
assert setup["variant_tree"] == str(variant_tree)
assert setup["model_path"] == str(model)
assert setup["proof_bin"] == "restricted_logit_diff"
assert setup["prefix_prompt_file"] == str(prefix_prompt)
assert setup["continuation_prompt_file"] == str(continuation_prompt)
assert setup["proof_artifact_env"] == "GPT_OSS_PROOF_JSON"
assert setup["proof_artifact_name"] == "continuation-head.json"
assert setup["compare_vector_key"] == "values_head"
assert setup["forced_output_tokens_arg"] == "--forced-output-tokens 6602"
assert setup["selected_marker_profiles"] == ["retained-debug-v1"]
assert setup["verify_tokenization"] == 1
assert setup["emit_forced_output_tokens"] == 1
assert setup["max_model_len"] == 8
assert setup["max_model_len_explicit"] == 1
assert setup["required_prefix_token_count"] == 3
assert setup["required_continuation_token_count"] == 1

tokenization = json.loads((output_dir / "plan" / "tokenization_summary.json").read_text(encoding="utf-8"))
assert tokenization["verified"] is True
assert tokenization["prefix_token_count"] == 3
assert tokenization["continuation_token_count"] == 1
assert tokenization["required_min_model_len"] == 4
assert tokenization["max_model_len"] == 8
assert tokenization["max_model_len_explicit"] is True
assert tokenization["continuation_token_ids"] == [6602]
assert tokenization["continuation_token_ids_csv"] == "6602"
assert tokenization["forced_output_tokens_arg"] == "--forced-output-tokens 6602"

envs = json.loads((output_dir / "plan" / "retained_env.json").read_text(encoding="utf-8"))
assert envs == [
    {"key": "TEST_LABEL", "value": "retained-proof"},
    {"key": "GPT_OSS_PROOF_JSON_SAFE", "value": setup["safe_proof_artifact_path"]},
    {"key": "GPT_OSS_PROOF_JSON_VARIANT", "value": setup["variant_proof_artifact_path"]},
]

env_file = (output_dir / "plan" / "retained_env.sh").read_text(encoding="utf-8")
assert "export TEST_LABEL=retained-proof" in env_file
assert f'export GPT_OSS_PROOF_JSON_SAFE={setup["safe_proof_artifact_path"]}' in env_file
assert f'export GPT_OSS_PROOF_JSON_VARIANT={setup["variant_proof_artifact_path"]}' in env_file

safe_run_command = (output_dir / "plan" / "safe_run_command.sh").read_text(encoding="utf-8")
variant_run_command = (output_dir / "plan" / "variant_run_command.sh").read_text(encoding="utf-8")
assert str(safe_tree) in safe_run_command
assert str(variant_tree) in variant_run_command
assert str(prefix_prompt) in safe_run_command
assert str(continuation_prompt) in safe_run_command
assert str(prefix_prompt) in variant_run_command
assert str(continuation_prompt) in variant_run_command
assert "--forced-output-tokens 6602" in safe_run_command
assert "--forced-output-tokens 6602" in variant_run_command
assert f'export GPT_OSS_PROOF_JSON={setup["safe_proof_artifact_path"]}' in safe_run_command
assert f'export GPT_OSS_PROOF_JSON={setup["variant_proof_artifact_path"]}' in variant_run_command

marker_profiles = json.loads((output_dir / "plan" / "marker_profiles.json").read_text(encoding="utf-8"))
assert marker_profiles["selected_profiles"] == ["retained-debug-v1"]
assert marker_profiles["configured_marker_sequence"][0] == "RETAINED_CHILD_START"
assert marker_profiles["configured_marker_sequence"][-1] == "RETAINED_PROOF_CAPTURED"

print("setup_summary=ok")
print("setup_tokenization=ok")
print("setup_env_scope=ok")
print("setup_command_scope=ok")
print("setup_marker_profile=ok")
PY

  "$RUNNER_SCRIPT" \
    --run-only \
    --gpu 6 \
    --safe-tree "$safe_tree" \
    --variant-tree "$variant_tree" \
    --model "$model" \
    --prefix-prompt-file "$prefix_prompt" \
    --continuation-prompt-file "$continuation_prompt" \
    --bin restricted_logit_diff \
    --proof-artifact-env GPT_OSS_PROOF_JSON \
    --proof-artifact-name continuation-head.json \
    --compare-vector-key values_head \
    --marker-profile retained-debug-v1 \
    --emit-forced-output-tokens \
    --verify-tokenization \
    --python python3 \
    --required-prefix-token-count 3 \
    --required-continuation-token-count 1 \
    --max-model-len 8 \
    --output-dir "$run_output" \
    --env TEST_LABEL=retained-proof >/dev/null

  assert_file "$run_output/summary.json"
  assert_file "$run_output/safe/status.json"
  assert_file "$run_output/variant/status.json"
  assert_file "$run_output/safe/progress-status.json"
  assert_file "$run_output/variant/progress-status.json"
  assert_file "$run_output/safe/restricted_logit_diff.json"
  assert_file "$run_output/variant/restricted_logit_diff.json"
  assert_file "$run_output/safe/continuation-head.json"
  assert_file "$run_output/variant/continuation-head.json"

  python3 - "$run_output" "$model" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
model = Path(sys.argv[2])

summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
safe = summary["safe"]
variant = summary["variant"]
comparison = summary["comparison"]

assert safe["state"] == "completed"
assert variant["state"] == "completed"
assert safe["artifact_state"] == "usable_artifact"
assert variant["artifact_state"] == "usable_artifact"
assert safe["proof_artifact_state"] == "usable_artifact"
assert variant["proof_artifact_state"] == "usable_artifact"
assert safe["primary_artifact_file"] == safe["proof_artifact_path"]
assert variant["primary_artifact_file"] == variant["proof_artifact_path"]

safe_outer = json.loads(Path(safe["outer_artifact_path"]).read_text(encoding="utf-8"))
variant_outer = json.loads(Path(variant["outer_artifact_path"]).read_text(encoding="utf-8"))
safe_proof = json.loads(Path(safe["proof_artifact_path"]).read_text(encoding="utf-8"))
variant_proof = json.loads(Path(variant["proof_artifact_path"]).read_text(encoding="utf-8"))

assert safe_outer["model"] == str(model)
assert variant_outer["model"] == str(model)
assert safe_outer["forced_output_tokens"] == ["6602"]
assert variant_outer["forced_output_tokens"] == ["6602"]
assert safe_outer["proof_artifact_env"] == safe["proof_artifact_path"]
assert variant_outer["proof_artifact_env"] == variant["proof_artifact_path"]

assert safe_proof["forced_output_tokens"] == ["6602"]
assert variant_proof["forced_output_tokens"] == ["6602"]
assert safe_proof["values_head"] == [1.0, 2.0]
assert variant_proof["values_head"] == [1.0, 2.5]

assert safe["last_progress_marker"] == "RETAINED_PROOF_CAPTURED"
assert safe["next_expected_progress_marker"] is None
assert safe["progress_stall_classification"] == "completed_through=RETAINED_PROOF_CAPTURED"

assert variant["last_progress_marker"] == "RETAINED_STEP_FORWARD_BEGIN"
assert variant["next_expected_progress_marker"] == "RETAINED_STEP_FORWARD_DONE"
assert variant["progress_stall_classification"] == "stalled_before=RETAINED_STEP_FORWARD_DONE"

assert comparison["comparable"] is True
assert comparison["safe_primary_artifact"] == safe["proof_artifact_path"]
assert comparison["variant_primary_artifact"] == variant["proof_artifact_path"]
assert comparison["json_equal"] is False
assert comparison["same_top_level_keys"] is True
assert comparison["vector_diff"]["vector_key"] == "values_head"
assert comparison["vector_diff"]["vector_length"] == 2
assert comparison["vector_diff"]["first_diff_index"] == 1
assert comparison["vector_diff"]["max_abs_diff"] == 0.5
assert comparison["vector_diff"]["mean_abs_diff"] == 0.25

safe_stdout = (output_dir / "safe" / "run.stdout").read_text(encoding="utf-8")
variant_stderr = (output_dir / "variant" / "run.stderr").read_text(encoding="utf-8")
assert '"forced_output_tokens": [' in safe_stdout
assert "RETAINED_STEP_FORWARD_BEGIN" in variant_stderr

print("run_summary=ok")
print("run_forced_output_tokens=ok")
print("run_progress_summary=ok")
print("run_proof_selection=ok")
print("run_vector_diff=ok")
PY

  echo "retained_continuation_regression: setup_summary=ok"
  echo "retained_continuation_regression: setup_tokenization=ok"
  echo "retained_continuation_regression: setup_env_scope=ok"
  echo "retained_continuation_regression: setup_command_scope=ok"
  echo "retained_continuation_regression: setup_marker_profile=ok"
  echo "retained_continuation_regression: run_summary=ok"
  echo "retained_continuation_regression: run_forced_output_tokens=ok"
  echo "retained_continuation_regression: run_progress_summary=ok"
  echo "retained_continuation_regression: run_proof_selection=ok"
  echo "retained_continuation_regression: run_vector_diff=ok"
}

main "$@"
