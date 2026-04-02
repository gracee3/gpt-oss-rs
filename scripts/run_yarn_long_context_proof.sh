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
DEFAULT_BIN="restricted_logit_diff"

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
  --bin <name>               proof binary name to build/run (default: restricted_logit_diff)
  --proof-artifact-env <n>   env var name used to pass a compact proof artifact path
  --proof-artifact-name <f>  filename for env-driven proof artifacts inside each case dir
  --max-model-len <n>        requested max-model-len passed to restricted_logit_diff
  --prompt-file <path>       existing prompt file to use
  --env KEY=VALUE            bounded env passthrough for proof-only runs; repeatable
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

shell_escape() {
  python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$1"
}

normalize_artifact_name() {
  local name="$1"
  name=${name//\//-}
  name=${name// /-}
  printf '%s.json' "${name}"
}

SAFE_TREE="${DEFAULT_SAFE_TREE}"
VARIANT_TREE="${DEFAULT_VARIANT_TREE}"
MODEL_PATH="${DEFAULT_MODEL}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
TIMEOUT_SECONDS="${DEFAULT_TIMEOUT}"
BUILD_TIMEOUT_SECONDS="${DEFAULT_BUILD_TIMEOUT}"
GPU_ID="${DEFAULT_GPU}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
PROOF_BIN="${DEFAULT_BIN}"
PROOF_ARTIFACT_ENV=""
PROOF_ARTIFACT_NAME=""
PROMPT_FILE=""
SETUP_ONLY=0
BUILD_ONLY=0
RUN_ONLY=0
declare -a EXTRA_ENVS=()

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
    --bin)
      PROOF_BIN="${2:?missing value for --bin}"
      shift 2
      ;;
    --proof-artifact-env)
      PROOF_ARTIFACT_ENV="${2:?missing value for --proof-artifact-env}"
      shift 2
      ;;
    --proof-artifact-name)
      PROOF_ARTIFACT_NAME="${2:?missing value for --proof-artifact-name}"
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
    --env)
      [[ "${2:-}" == *=* ]] || die "--env requires KEY=VALUE"
      EXTRA_ENVS+=("${2}")
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
require_file "${SAFE_TREE}/crates/gpt-oss-bench/src/bin/${PROOF_BIN}.rs"
require_file "${VARIANT_TREE}/crates/gpt-oss-bench/src/bin/${PROOF_BIN}.rs"

OUTPUT_DIR=$(mkdir -p "${OUTPUT_DIR}" && cd -- "${OUTPUT_DIR}" && pwd)
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
ARTIFACT_NAME="$(normalize_artifact_name "${PROOF_BIN}")"
if [[ -z "${PROOF_ARTIFACT_NAME}" ]]; then
  PROOF_ARTIFACT_NAME="$(normalize_artifact_name "${PROOF_BIN}-proof")"
fi
SAFE_BINARY="${SAFE_TREE}/target/release/${PROOF_BIN}"
VARIANT_BINARY="${VARIANT_TREE}/target/release/${PROOF_BIN}"
ENV_FILE="${PLAN_DIR}/proof_env.sh"
ENV_JSON_FILE="${PLAN_DIR}/proof_env.json"
SAFE_PROOF_ARTIFACT="${OUTPUT_DIR}/safe/${PROOF_ARTIFACT_NAME}"
VARIANT_PROOF_ARTIFACT="${OUTPUT_DIR}/variant/${PROOF_ARTIFACT_NAME}"

{
  for env_kv in "${EXTRA_ENVS[@]}"; do
    key=${env_kv%%=*}
    value=${env_kv#*=}
    printf 'export %s=%s\n' "${key}" "$(shell_escape "${value}")"
  done
  if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
    printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}_SAFE" "$(shell_escape "${SAFE_PROOF_ARTIFACT}")"
    printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}_VARIANT" "$(shell_escape "${VARIANT_PROOF_ARTIFACT}")"
  fi
} >"${ENV_FILE}"

export PROOF_ENV_JSON_FILE="${ENV_JSON_FILE}"
if ((${#EXTRA_ENVS[@]} > 0)); then
  export PROOF_EXTRA_ENVS="$(printf '%s\n' "${EXTRA_ENVS[@]}")"
else
  export PROOF_EXTRA_ENVS=""
fi
export PROOF_ARTIFACT_ENV_NAME="${PROOF_ARTIFACT_ENV}"
export PROOF_ARTIFACT_SAFE_PATH="${SAFE_PROOF_ARTIFACT}"
export PROOF_ARTIFACT_VARIANT_PATH="${VARIANT_PROOF_ARTIFACT}"
python3 <<'PY'
import json
import os
from pathlib import Path

envs = []
raw = os.environ.get("PROOF_EXTRA_ENVS", "")
for line in raw.splitlines():
    if not line:
        continue
    key, value = line.split("=", 1)
    envs.append({"key": key, "value": value})
proof_artifact_env = os.environ.get("PROOF_ARTIFACT_ENV_NAME")
if proof_artifact_env:
    envs.append({"key": f"{proof_artifact_env}_SAFE", "value": os.environ["PROOF_ARTIFACT_SAFE_PATH"]})
    envs.append({"key": f"{proof_artifact_env}_VARIANT", "value": os.environ["PROOF_ARTIFACT_VARIANT_PATH"]})
Path(os.environ["PROOF_ENV_JSON_FILE"]).write_text(json.dumps(envs, indent=2) + "\n")
PY

cat >"${SAFE_CMD_FILE}" <<EOF
cd ${SAFE_TREE}
. ${ENV_FILE}
$( if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${SAFE_PROOF_ARTIFACT}")"; fi )
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${SAFE_BINARY} --model ${MODEL_PATH} --prompt "\$(cat "${PROMPT_FILE}")" --max-model-len ${MAX_MODEL_LEN} --output ${OUTPUT_DIR}/safe/${ARTIFACT_NAME} --log-level info
EOF

cat >"${VARIANT_CMD_FILE}" <<EOF
cd ${VARIANT_TREE}
. ${ENV_FILE}
$( if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${VARIANT_PROOF_ARTIFACT}")"; fi )
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${VARIANT_BINARY} --model ${MODEL_PATH} --prompt "\$(cat "${PROMPT_FILE}")" --max-model-len ${MAX_MODEL_LEN} --output ${OUTPUT_DIR}/variant/${ARTIFACT_NAME} --log-level info
EOF

cat >"${SAFE_BUILD_CMD_FILE}" <<EOF
cd ${SAFE_TREE}
. ${ENV_FILE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo build --release -p gpt-oss-bench --features cuda --bin ${PROOF_BIN}
EOF

cat >"${VARIANT_BUILD_CMD_FILE}" <<EOF
cd ${VARIANT_TREE}
. ${ENV_FILE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo build --release -p gpt-oss-bench --features cuda --bin ${PROOF_BIN}
EOF

chmod +x "${SAFE_CMD_FILE}" "${VARIANT_CMD_FILE}" "${SAFE_BUILD_CMD_FILE}" "${VARIANT_BUILD_CMD_FILE}"

cat >"${PLAN_DIR}/setup_summary.json" <<EOF
{
  "gpu_id": "${GPU_ID}",
  "safe_tree": "${SAFE_TREE}",
  "variant_tree": "${VARIANT_TREE}",
  "model_path": "${MODEL_PATH}",
  "proof_bin": "${PROOF_BIN}",
  "artifact_name": "${ARTIFACT_NAME}",
  "proof_artifact_env": "${PROOF_ARTIFACT_ENV}",
  "proof_artifact_name": "${PROOF_ARTIFACT_NAME}",
  "safe_proof_artifact_path": "${SAFE_PROOF_ARTIFACT}",
  "variant_proof_artifact_path": "${VARIANT_PROOF_ARTIFACT}",
  "prompt_file": "${PROMPT_FILE}",
  "env_file": "${ENV_FILE}",
  "env_json_file": "${ENV_JSON_FILE}",
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
echo "proof env plan: ${ENV_FILE}"
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
. ${ENV_FILE}
PATH="/data/models/.venv-awq/bin:\$PATH" \
CUDA_VISIBLE_DEVICES=${GPU_ID} \
GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
cargo build --release -p gpt-oss-bench --features cuda --bin ${PROOF_BIN}
EOF
  chmod +x "${command_file}"

  start_epoch=$(date +%s)
  rc=0
  (
    cd "${tree}"
    . "${ENV_FILE}"
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${BUILD_TIMEOUT_SECONDS}" \
      cargo build --release -p gpt-oss-bench --features cuda --bin "${PROOF_BIN}"
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
  local report_file="${case_dir}/${ARTIFACT_NAME}"
  local proof_artifact_file=""
  local status_file="${case_dir}/status.json"
  local start_epoch end_epoch duration rc state artifact_state primary_artifact
  mkdir -p "${case_dir}"

  [[ -x "${binary_path}" ]] || die "prebuilt binary missing for ${label}: ${binary_path}; run with --build-only first"

  if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
    if [[ "${label}" == "safe" ]]; then
      proof_artifact_file="${SAFE_PROOF_ARTIFACT}"
    else
      proof_artifact_file="${VARIANT_PROOF_ARTIFACT}"
    fi
  fi

  cat >"${command_file}" <<EOF
cd ${tree}
PROMPT_FILE=${PROMPT_FILE}
PROMPT=\$(cat "\${PROMPT_FILE}")
. ${ENV_FILE}
$( if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${proof_artifact_file}")"; fi )
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
    . "${ENV_FILE}"
    if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
      export "${PROOF_ARTIFACT_ENV}=${proof_artifact_file}"
    fi
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

  primary_artifact="${report_file}"
  if [[ -n "${proof_artifact_file}" && -f "${proof_artifact_file}" ]]; then
    artifact_state="usable_artifact"
    primary_artifact="${proof_artifact_file}"
  elif [[ -f "${report_file}" ]]; then
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
  "artifact_name": $(json_escape "${ARTIFACT_NAME}"),
  "proof_artifact_env": $(json_escape "${PROOF_ARTIFACT_ENV}"),
  "proof_artifact_name": $(json_escape "${PROOF_ARTIFACT_NAME}"),
  "proof_artifact_file": $(json_escape "${proof_artifact_file}"),
  "primary_artifact_file": $(json_escape "${primary_artifact}"),
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
artifact_name = None
primary_safe_artifact = None
primary_variant_artifact = None
if summary.get("safe"):
    artifact_name = summary["safe"].get("artifact_name")
    primary_safe_artifact = summary["safe"].get("primary_artifact_file")
if artifact_name is None and summary.get("variant"):
    artifact_name = summary["variant"].get("artifact_name")
if summary.get("variant"):
    primary_variant_artifact = summary["variant"].get("primary_artifact_file")
if artifact_name is None:
    artifact_name = "artifact.json"
safe_report = Path(primary_safe_artifact) if primary_safe_artifact else output_dir / "safe" / artifact_name
variant_report = Path(primary_variant_artifact) if primary_variant_artifact else output_dir / "variant" / artifact_name

if safe_report.exists() and variant_report.exists():
    safe_text = safe_report.read_text()
    variant_text = variant_report.read_text()
    safe = json.loads(safe_text)
    variant = json.loads(variant_text)
    safe_keys = sorted(safe.keys()) if isinstance(safe, dict) else None
    variant_keys = sorted(variant.keys()) if isinstance(variant, dict) else None
    summary["comparison"] = {
        "comparable": True,
        "artifact_name": artifact_name,
        "safe_primary_artifact": str(safe_report),
        "variant_primary_artifact": str(variant_report),
        "safe_sha256": __import__("hashlib").sha256(safe_text.encode("utf-8")).hexdigest(),
        "variant_sha256": __import__("hashlib").sha256(variant_text.encode("utf-8")).hexdigest(),
        "json_equal": safe == variant,
        "safe_top_level_keys": safe_keys,
        "variant_top_level_keys": variant_keys,
        "same_top_level_keys": safe_keys == variant_keys,
        "safe_conclusion": safe.get("conclusion") if isinstance(safe, dict) else None,
        "variant_conclusion": variant.get("conclusion") if isinstance(variant, dict) else None,
        "safe_first_mismatch_boundary": safe.get("first_mismatch_boundary") if isinstance(safe, dict) else None,
        "variant_first_mismatch_boundary": variant.get("first_mismatch_boundary") if isinstance(variant, dict) else None,
        "safe_prompt_token_count": len(safe.get("prompt_token_ids", [])) if isinstance(safe, dict) and isinstance(safe.get("prompt_token_ids"), list) else None,
        "variant_prompt_token_count": len(variant.get("prompt_token_ids", [])) if isinstance(variant, dict) and isinstance(variant.get("prompt_token_ids"), list) else None,
        "same_prompt_token_ids": safe.get("prompt_token_ids") == variant.get("prompt_token_ids") if isinstance(safe, dict) and isinstance(variant, dict) else None,
        "same_conclusion": safe.get("conclusion") == variant.get("conclusion") if isinstance(safe, dict) and isinstance(variant, dict) else None,
    }
else:
    summary["comparison"] = {
        "comparable": False,
        "artifact_name": artifact_name,
        "safe_primary_artifact": str(safe_report),
        "variant_primary_artifact": str(variant_report),
        "reason": "both safe and variant artifacts were not produced within the bounded run",
    }

(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
