#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEFAULT_MODEL="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
DEFAULT_SAFE_TREE="/home/emmy/openai/gpt-oss-rs"
DEFAULT_VARIANT_TREE="/home/emmy/openai/worktrees/runtime-forward"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/.live/yarn-long-context-proof"
DEFAULT_TIMEOUT="1800"
DEFAULT_BUILD_TIMEOUT="1800"
DEFAULT_GPU="0"
DEFAULT_MAX_MODEL_LEN="4608"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_yarn_long_context_proof.sh [options]

Prepare or run a bounded same-input long-context comparison for the YaRN frontier.

Options:
  --gpu <id>                 CUDA_VISIBLE_DEVICES value to use (default: 0)
  --safe-tree <path>         worktree for the conservative baseline
  --variant-tree <path>      worktree for the candidate variant
  --model <path>             restricted sink-free model view to use
  --output-dir <path>        output directory for prompt, logs, reports, and summary
  --timeout <seconds>        per-run timeout in seconds (default: 1800)
  --build-timeout <seconds>  per-build timeout in seconds (default: 1800)
  --max-model-len <n>        requested max-model-len passed to restricted_logit_diff
  --prompt-file <path>       existing prompt file to use
  --build-only               prepare/warm both trees, then stop before executing the proof
  --run-only                 execute the proof using prebuilt binaries only
  --setup-only               stop after prompt/setup verification and command planning
  -h, --help                 show this help text
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "required directory not found: ${path}"
}

SAFE_TREE="${DEFAULT_SAFE_TREE}"
VARIANT_TREE="${DEFAULT_VARIANT_TREE}"
MODEL_PATH="${DEFAULT_MODEL}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
TIMEOUT_SECONDS="${DEFAULT_TIMEOUT}"
BUILD_TIMEOUT_SECONDS="${DEFAULT_BUILD_TIMEOUT}"
GPU_ID="${DEFAULT_GPU}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
PROMPT_FILE=""
SETUP_ONLY=0
BUILD_ONLY=0
RUN_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="${2:?missing value for --gpu}"
      shift 2
      ;;
    --safe-tree)
      SAFE_TREE="${2:?missing value for --safe-tree}"
      shift 2
      ;;
    --variant-tree)
      VARIANT_TREE="${2:?missing value for --variant-tree}"
      shift 2
      ;;
    --model)
      MODEL_PATH="${2:?missing value for --model}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="${2:?missing value for --timeout}"
      shift 2
      ;;
    --build-timeout)
      BUILD_TIMEOUT_SECONDS="${2:?missing value for --build-timeout}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?missing value for --max-model-len}"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="${2:?missing value for --prompt-file}"
      shift 2
      ;;
    --setup-only)
      SETUP_ONLY=1
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --run-only)
      RUN_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "${BUILD_ONLY}" -eq 1 && "${RUN_ONLY}" -eq 1 ]]; then
  die "--build-only and --run-only are mutually exclusive"
fi

require_dir "${SAFE_TREE}"
require_dir "${VARIANT_TREE}"
require_dir "${MODEL_PATH}"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/RESTRICTED_MODEL_VIEW.json"
require_file "${SAFE_TREE}/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs"
require_file "${VARIANT_TREE}/crates/gpt-oss-bench/src/bin/restricted_logit_diff.rs"

mkdir -p "${OUTPUT_DIR}"
PROMPT_DIR="${OUTPUT_DIR}/prompt"
PLAN_DIR="${OUTPUT_DIR}/plan"
mkdir -p "${PROMPT_DIR}" "${PLAN_DIR}"

PROMPT_FILE=${PROMPT_FILE:-"${PROMPT_DIR}/deterministic_over_4096.txt"}
PROMPT_FILE=$(cd -- "$(dirname -- "${PROMPT_FILE}")" && pwd)/$(basename -- "${PROMPT_FILE}")

export PROOF_MODEL_PATH="${MODEL_PATH}"
export PROOF_PROMPT_FILE="${PROMPT_FILE}"

python3 <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["PROOF_MODEL_PATH"])
prompt_path = Path(os.environ["PROOF_PROMPT_FILE"])
config = json.loads((model_path / "config.json").read_text())
meta = json.loads((model_path / "RESTRICTED_MODEL_VIEW.json").read_text())

errors = []
layer_types = config.get("layer_types", [])
if not layer_types or any(layer != "full_attention" for layer in layer_types):
    errors.append("restricted config is not full_attention only")
if int(config.get("sliding_window", 1)) != 0:
    errors.append("restricted config sliding_window is not zero")
notes = meta.get("notes", [])
if not any("self_attn.sinks" in note for note in notes):
    errors.append("restricted model metadata does not mention sink override")
if errors:
    raise SystemExit("\n".join(errors))

if not prompt_path.exists():
    units = []
    for idx in range(4600):
        units.append(f"token-{idx:04d}")
    prompt_path.write_text(" ".join(units) + "\n")

summary = {
    "model_path": str(model_path),
    "source_model": meta.get("source_model"),
    "restricted_view_kind": meta.get("kind"),
    "sink_free_checks": {
        "full_attention_only": True,
        "sliding_window_zero": True,
        "sink_override_note_present": True,
    },
    "prompt_file": str(prompt_path),
}
Path(prompt_path.parent / "setup_model_check.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

export PROOF_TOKENIZE_MODEL="${MODEL_PATH}"
export PROOF_TOKENIZE_PROMPT="${PROMPT_FILE}"
export PROOF_TOKEN_COUNT_JSON="${PROMPT_DIR}/prompt_token_count.json"

PATH="/data/models/.venv-awq/bin:${PATH}" python3 <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

model = os.environ["PROOF_TOKENIZE_MODEL"]
prompt_file = Path(os.environ["PROOF_TOKENIZE_PROMPT"])
output = Path(os.environ["PROOF_TOKEN_COUNT_JSON"])
prompt = prompt_file.read_text()
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
token_ids = tokenizer.encode(prompt, add_special_tokens=False)
summary = {
    "prompt_file": str(prompt_file),
    "token_count": len(token_ids),
    "crosses_4096": len(token_ids) > 4096,
    "first_tokens": token_ids[:8],
    "last_tokens": token_ids[-8:],
}
output.write_text(json.dumps(summary, indent=2) + "\n")
if len(token_ids) <= 4096:
    raise SystemExit(f"prompt does not exceed 4096 tokens: {len(token_ids)}")
PY

GPU_SNAPSHOT_FILE="${PLAN_DIR}/gpu_snapshot.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"${GPU_SNAPSHOT_FILE}" 2>&1 || true
else
  printf 'nvidia-smi unavailable\n' >"${GPU_SNAPSHOT_FILE}"
fi

SAFE_CMD_FILE="${PLAN_DIR}/safe_command.sh"
VARIANT_CMD_FILE="${PLAN_DIR}/variant_command.sh"
SAFE_BUILD_CMD_FILE="${PLAN_DIR}/safe_build_command.sh"
VARIANT_BUILD_CMD_FILE="${PLAN_DIR}/variant_build_command.sh"
SAFE_BINARY="${SAFE_TREE}/target/release/restricted_logit_diff"
VARIANT_BINARY="${VARIANT_TREE}/target/release/restricted_logit_diff"

cat >"${SAFE_CMD_FILE}" <<EOF
cd ${SAFE_TREE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${SAFE_BINARY} --model ${MODEL_PATH} --prompt "\$(cat "${PROMPT_FILE}")" --max-model-len ${MAX_MODEL_LEN} --output ${OUTPUT_DIR}/safe/restricted-logit-diff.json --log-level info
EOF

cat >"${VARIANT_CMD_FILE}" <<EOF
cd ${VARIANT_TREE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${VARIANT_BINARY} --model ${MODEL_PATH} --prompt "\$(cat "${PROMPT_FILE}")" --max-model-len ${MAX_MODEL_LEN} --output ${OUTPUT_DIR}/variant/restricted-logit-diff.json --log-level info
EOF

cat >"${SAFE_BUILD_CMD_FILE}" <<EOF
cd ${SAFE_TREE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo build --release -p gpt-oss-bench --features cuda --bin restricted_logit_diff
EOF

cat >"${VARIANT_BUILD_CMD_FILE}" <<EOF
cd ${VARIANT_TREE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo build --release -p gpt-oss-bench --features cuda --bin restricted_logit_diff
EOF

chmod +x "${SAFE_CMD_FILE}" "${VARIANT_CMD_FILE}" "${SAFE_BUILD_CMD_FILE}" "${VARIANT_BUILD_CMD_FILE}"

cat >"${PLAN_DIR}/setup_summary.json" <<EOF
{
  "gpu_id": "${GPU_ID}",
  "safe_tree": "${SAFE_TREE}",
  "variant_tree": "${VARIANT_TREE}",
  "model_path": "${MODEL_PATH}",
  "prompt_file": "${PROMPT_FILE}",
  "build_timeout_seconds": ${BUILD_TIMEOUT_SECONDS},
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "max_model_len": ${MAX_MODEL_LEN},
  "build_only": ${BUILD_ONLY},
  "run_only": ${RUN_ONLY},
  "setup_only": ${SETUP_ONLY}
}
EOF

echo "prepared long-context proof setup under ${OUTPUT_DIR}"
echo "prompt token summary: ${PROMPT_DIR}/prompt_token_count.json"
echo "safe build plan: ${SAFE_BUILD_CMD_FILE}"
echo "variant build plan: ${VARIANT_BUILD_CMD_FILE}"
echo "safe command plan: ${SAFE_CMD_FILE}"
echo "variant command plan: ${VARIANT_CMD_FILE}"

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  exit 0
fi

build_case() {
  local label="$1"
  local tree="$2"
  local binary_path="$3"
  local case_dir="${OUTPUT_DIR}/${label}"
  local stdout_file="${case_dir}/build.stdout"
  local stderr_file="${case_dir}/build.stderr"
  local command_file="${case_dir}/build-command.sh"
  local status_file="${case_dir}/build-status.json"
  local start_epoch end_epoch duration rc state binary_state
  mkdir -p "${case_dir}"

  cat >"${command_file}" <<EOF
cd ${tree}
PATH="/data/models/.venv-awq/bin:\$PATH" \
CUDA_VISIBLE_DEVICES=${GPU_ID} \
GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
cargo build --release -p gpt-oss-bench --features cuda --bin restricted_logit_diff
EOF
  chmod +x "${command_file}"

  start_epoch=$(date +%s)
  rc=0
  (
    cd "${tree}"
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${BUILD_TIMEOUT_SECONDS}" \
      cargo build --release -p gpt-oss-bench --features cuda --bin restricted_logit_diff
  ) >"${stdout_file}" 2>"${stderr_file}" || rc=$?
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))

  if [[ "${rc}" -eq 0 ]]; then
    state="completed"
  elif [[ "${rc}" -eq 124 ]]; then
    state="timed_out"
  else
    state="failed"
  fi

  if [[ -x "${binary_path}" ]]; then
    binary_state="ready"
  else
    binary_state="missing"
  fi

  cat >"${status_file}" <<EOF
{
  "label": $(json_escape "${label}"),
  "tree": $(json_escape "${tree}"),
  "state": $(json_escape "${state}"),
  "exit_code": ${rc},
  "duration_seconds": ${duration},
  "binary_state": $(json_escape "${binary_state}"),
  "binary_path": $(json_escape "${binary_path}"),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "command_file": $(json_escape "${command_file}")
}
EOF
}

run_case() {
  local label="$1"
  local tree="$2"
  local binary_path="$3"
  local case_dir="${OUTPUT_DIR}/${label}"
  local log_file="${case_dir}/run.log"
  local stderr_file="${case_dir}/run.stderr"
  local stdout_file="${case_dir}/run.stdout"
  local command_file="${case_dir}/command.sh"
  local report_file="${case_dir}/restricted-logit-diff.json"
  local status_file="${case_dir}/status.json"
  local start_epoch end_epoch duration rc state artifact_state
  mkdir -p "${case_dir}"

  [[ -x "${binary_path}" ]] || die "prebuilt binary missing for ${label}: ${binary_path}; run with --build-only first"

  cat >"${command_file}" <<EOF
cd ${tree}
PROMPT_FILE=${PROMPT_FILE}
PROMPT=\$(cat "\${PROMPT_FILE}")
PATH="/data/models/.venv-awq/bin:\$PATH" \
CUDA_VISIBLE_DEVICES=${GPU_ID} \
GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
${binary_path} \
  --model ${MODEL_PATH} \
  --prompt "\${PROMPT}" \
  --max-model-len ${MAX_MODEL_LEN} \
  --output ${report_file} \
  --log-level info
EOF
  chmod +x "${command_file}"

  start_epoch=$(date +%s)
  rc=0
  (
    cd "${tree}"
    PROMPT=$(cat "${PROMPT_FILE}")
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${TIMEOUT_SECONDS}" \
      "${binary_path}" \
      --model "${MODEL_PATH}" \
      --prompt "${PROMPT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --output "${report_file}" \
      --log-level info
  ) >"${stdout_file}" 2>"${stderr_file}" || rc=$?
  end_epoch=$(date +%s)
  duration=$((end_epoch - start_epoch))

  if [[ "${rc}" -eq 0 ]]; then
    state="completed"
  elif [[ "${rc}" -eq 124 ]]; then
    state="timed_out"
  else
    state="failed"
  fi

  if [[ -f "${report_file}" ]]; then
    artifact_state="usable_artifact"
  else
    artifact_state="no_artifact"
  fi

  {
    echo "state=${state}"
    echo "exit_code=${rc}"
    echo "duration_seconds=${duration}"
    echo "artifact_state=${artifact_state}"
  } >"${log_file}"

  cat >"${status_file}" <<EOF
{
  "label": $(json_escape "${label}"),
  "tree": $(json_escape "${tree}"),
  "state": $(json_escape "${state}"),
  "exit_code": ${rc},
  "duration_seconds": ${duration},
  "artifact_state": $(json_escape "${artifact_state}"),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "report_file": $(json_escape "${report_file}"),
  "command_file": $(json_escape "${command_file}")
}
EOF
}

if [[ "${RUN_ONLY}" -ne 1 ]]; then
  build_case safe "${SAFE_TREE}" "${SAFE_BINARY}"
  build_case variant "${VARIANT_TREE}" "${VARIANT_BINARY}"
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  export PROOF_OUTPUT_DIR="${OUTPUT_DIR}"
  python3 <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["PROOF_OUTPUT_DIR"])
summary = {
    "safe_build": json.loads((output_dir / "safe" / "build-status.json").read_text()),
    "variant_build": json.loads((output_dir / "variant" / "build-status.json").read_text()),
    "comparison": {
        "comparable": False,
        "reason": "build-only mode stops after prewarming both trees",
    },
}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
  exit 0
fi

run_case safe "${SAFE_TREE}" "${SAFE_BINARY}"
run_case variant "${VARIANT_TREE}" "${VARIANT_BINARY}"

export PROOF_OUTPUT_DIR="${OUTPUT_DIR}"
python3 <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["PROOF_OUTPUT_DIR"])
summary = {
    "safe_build": json.loads((output_dir / "safe" / "build-status.json").read_text()) if (output_dir / "safe" / "build-status.json").exists() else None,
    "variant_build": json.loads((output_dir / "variant" / "build-status.json").read_text()) if (output_dir / "variant" / "build-status.json").exists() else None,
    "safe": json.loads((output_dir / "safe" / "status.json").read_text()),
    "variant": json.loads((output_dir / "variant" / "status.json").read_text()),
}

safe_report = output_dir / "safe" / "restricted-logit-diff.json"
variant_report = output_dir / "variant" / "restricted-logit-diff.json"

if safe_report.exists() and variant_report.exists():
    safe = json.loads(safe_report.read_text())
    variant = json.loads(variant_report.read_text())
    summary["comparison"] = {
        "comparable": True,
        "safe_conclusion": safe.get("conclusion"),
        "variant_conclusion": variant.get("conclusion"),
        "safe_first_mismatch_boundary": safe.get("first_mismatch_boundary"),
        "variant_first_mismatch_boundary": variant.get("first_mismatch_boundary"),
        "safe_prompt_token_count": len(safe.get("prompt_token_ids", [])),
        "variant_prompt_token_count": len(variant.get("prompt_token_ids", [])),
        "same_prompt_token_ids": safe.get("prompt_token_ids") == variant.get("prompt_token_ids"),
        "same_conclusion": safe.get("conclusion") == variant.get("conclusion"),
    }
else:
    summary["comparison"] = {
        "comparable": False,
        "reason": "both safe and variant reports were not produced within the bounded run",
    }

(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
