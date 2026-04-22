#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEFAULT_TREE="/home/emmy/openai/gpt-oss-rs"
DEFAULT_MODEL="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/.live/staged-boundary-smoke"
DEFAULT_GPU="0"
DEFAULT_TIMEOUT="1800"
DEFAULT_BUILD_TIMEOUT="1800"
DEFAULT_BIN="restricted_prefill_trace"
DEFAULT_MAX_MODEL_LEN="4608"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_staged_boundary_smoke.sh [options]

Prepare or run a bounded prefix+continuation boundary-isolation workflow.

Options:
  --gpu <id>                      CUDA_VISIBLE_DEVICES value to use (default: 0)
  --tree <path>                   worktree to build/run from
  --model <path>                  model path passed to the staged binary
  --prefix-prompt-file <path>     prompt file for the prefix/prefill stage
  --continuation-prompt-file <p>  prompt file for the continuation/step stage
  --timeout <seconds>             per-stage run timeout in seconds (default: 1800)
  --build-timeout <seconds>       build timeout in seconds (default: 1800)
  --output-dir <path>             output directory for plans, logs, and summary
  --bin <name>                    binary to build/run (default: restricted_prefill_trace)
  --max-model-len <n>             max-model-len passed to the binary
  --proof-artifact-env <name>     env var used to pass a continuation proof-artifact path
  --proof-artifact-name <file>    filename for the continuation proof artifact
  --env KEY=VALUE                 bounded env passthrough; repeatable
  --build-only                    build/warm and stop
  --run-only                      run only, requiring a prebuilt binary
  --setup-only                    write staged plans/summary and stop
  -h, --help                      show this help text
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

shell_escape() {
  python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$1"
}

normalize_name() {
  local name="$1"
  name=${name//\//-}
  name=${name// /-}
  printf '%s.json' "${name}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "required directory not found: ${path}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

TREE="${DEFAULT_TREE}"
MODEL_PATH="${DEFAULT_MODEL}"
PREFIX_PROMPT_FILE=""
CONTINUATION_PROMPT_FILE=""
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
GPU_ID="${DEFAULT_GPU}"
TIMEOUT_SECONDS="${DEFAULT_TIMEOUT}"
BUILD_TIMEOUT_SECONDS="${DEFAULT_BUILD_TIMEOUT}"
STAGE_BIN="${DEFAULT_BIN}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
PROOF_ARTIFACT_ENV=""
PROOF_ARTIFACT_NAME=""
BUILD_ONLY=0
RUN_ONLY=0
SETUP_ONLY=0
declare -a EXTRA_ENVS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID="${2:?missing value for --gpu}"
      shift 2
      ;;
    --tree)
      TREE="${2:?missing value for --tree}"
      shift 2
      ;;
    --model)
      MODEL_PATH="${2:?missing value for --model}"
      shift 2
      ;;
    --prefix-prompt-file)
      PREFIX_PROMPT_FILE="${2:?missing value for --prefix-prompt-file}"
      shift 2
      ;;
    --continuation-prompt-file)
      CONTINUATION_PROMPT_FILE="${2:?missing value for --continuation-prompt-file}"
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
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --bin)
      STAGE_BIN="${2:?missing value for --bin}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?missing value for --max-model-len}"
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
    --env)
      [[ "${2:-}" == *=* ]] || die "--env requires KEY=VALUE"
      EXTRA_ENVS+=("${2}")
      shift 2
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --run-only)
      RUN_ONLY=1
      shift
      ;;
    --setup-only)
      SETUP_ONLY=1
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

[[ -n "${PREFIX_PROMPT_FILE}" ]] || die "--prefix-prompt-file is required"
[[ -n "${CONTINUATION_PROMPT_FILE}" ]] || die "--continuation-prompt-file is required"
require_dir "${TREE}"
require_dir "${MODEL_PATH}"
require_file "${PREFIX_PROMPT_FILE}"
require_file "${CONTINUATION_PROMPT_FILE}"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/RESTRICTED_MODEL_VIEW.json"
require_file "${TREE}/crates/gpt-oss-bench/src/bin/${STAGE_BIN}.rs"

OUTPUT_DIR=$(mkdir -p "${OUTPUT_DIR}" && cd -- "${OUTPUT_DIR}" && pwd)
PREFIX_PROMPT_FILE=$(cd -- "$(dirname -- "${PREFIX_PROMPT_FILE}")" && pwd)/$(basename -- "${PREFIX_PROMPT_FILE}")
CONTINUATION_PROMPT_FILE=$(cd -- "$(dirname -- "${CONTINUATION_PROMPT_FILE}")" && pwd)/$(basename -- "${CONTINUATION_PROMPT_FILE}")
PLAN_DIR="${OUTPUT_DIR}/plan"
PREFIX_DIR="${OUTPUT_DIR}/prefix"
CONTINUATION_DIR="${OUTPUT_DIR}/continuation"
mkdir -p "${PLAN_DIR}" "${PREFIX_DIR}" "${CONTINUATION_DIR}"

BINARY_PATH="${TREE}/target/release/${STAGE_BIN}"
PREFIX_ARTIFACT="${PREFIX_DIR}/$(normalize_name "${STAGE_BIN}-prefix")"
CONTINUATION_ARTIFACT="${CONTINUATION_DIR}/$(normalize_name "${STAGE_BIN}-continuation")"
if [[ -z "${PROOF_ARTIFACT_NAME}" ]]; then
  PROOF_ARTIFACT_NAME="$(normalize_name "${STAGE_BIN}-continuation-proof")"
fi
CONTINUATION_PROOF_ARTIFACT="${CONTINUATION_DIR}/${PROOF_ARTIFACT_NAME}"
ENV_FILE="${PLAN_DIR}/staged_env.sh"
ENV_JSON_FILE="${PLAN_DIR}/staged_env.json"
BUILD_CMD_FILE="${PLAN_DIR}/build_command.sh"
PREFIX_CMD_FILE="${PLAN_DIR}/prefix_command.sh"
CONTINUATION_CMD_FILE="${PLAN_DIR}/continuation_command.sh"
GPU_SNAPSHOT_FILE="${PLAN_DIR}/gpu_snapshot.txt"

{
  for env_kv in "${EXTRA_ENVS[@]}"; do
    key=${env_kv%%=*}
    value=${env_kv#*=}
    printf 'export %s=%s\n' "${key}" "$(shell_escape "${value}")"
  done
} >"${ENV_FILE}"

export STAGED_ENV_JSON_FILE="${ENV_JSON_FILE}"
export STAGED_PROOF_ARTIFACT_ENV="${PROOF_ARTIFACT_ENV}"
export STAGED_PROOF_ARTIFACT_PATH="${CONTINUATION_PROOF_ARTIFACT}"
if ((${#EXTRA_ENVS[@]} > 0)); then
  export STAGED_EXTRA_ENVS="$(printf '%s\n' "${EXTRA_ENVS[@]}")"
else
  export STAGED_EXTRA_ENVS=""
fi
python3 <<'PY'
import json
import os
from pathlib import Path

envs = []
raw = os.environ.get("STAGED_EXTRA_ENVS", "")
for line in raw.splitlines():
    if not line:
        continue
    key, value = line.split("=", 1)
    envs.append({"key": key, "value": value})
proof_key = os.environ.get("STAGED_PROOF_ARTIFACT_ENV", "")
Path(os.environ["STAGED_ENV_JSON_FILE"]).write_text(json.dumps(envs, indent=2) + "\n")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"${GPU_SNAPSHOT_FILE}" 2>&1 || true
else
  printf 'nvidia-smi unavailable\n' >"${GPU_SNAPSHOT_FILE}"
fi

cat >"${BUILD_CMD_FILE}" <<EOF
cd ${TREE}
. ${ENV_FILE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo build --release -p gpt-oss-bench --features cuda --bin ${STAGE_BIN}
EOF

cat >"${PREFIX_CMD_FILE}" <<EOF
cd ${TREE}
PROMPT=\$(cat ${PREFIX_PROMPT_FILE})
. ${ENV_FILE}
$( if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then printf 'unset %s\n' "${PROOF_ARTIFACT_ENV}"; fi )
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${BINARY_PATH} --model ${MODEL_PATH} --prompt "\${PROMPT}" --max-model-len ${MAX_MODEL_LEN} --output ${PREFIX_ARTIFACT}
EOF

{
  cat <<EOF
cd ${TREE}
PROMPT=\$(cat ${CONTINUATION_PROMPT_FILE})
. ${ENV_FILE}
EOF
  if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
    printf 'unset %s\n' "${PROOF_ARTIFACT_ENV}"
    printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${CONTINUATION_PROOF_ARTIFACT}")"
  fi
  cat <<EOF
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${BINARY_PATH} --model ${MODEL_PATH} --prompt "\${PROMPT}" --max-model-len ${MAX_MODEL_LEN} --output ${CONTINUATION_ARTIFACT}
EOF
} >"${CONTINUATION_CMD_FILE}"

chmod +x "${BUILD_CMD_FILE}" "${PREFIX_CMD_FILE}" "${CONTINUATION_CMD_FILE}"

cat >"${PLAN_DIR}/summary.json" <<EOF
{
  "gpu_id": "${GPU_ID}",
  "tree": "${TREE}",
  "model_path": "${MODEL_PATH}",
  "stage_bin": "${STAGE_BIN}",
  "prefix_prompt_file": "${PREFIX_PROMPT_FILE}",
  "continuation_prompt_file": "${CONTINUATION_PROMPT_FILE}",
  "prefix_artifact_path": "${PREFIX_ARTIFACT}",
  "continuation_artifact_path": "${CONTINUATION_ARTIFACT}",
  "proof_artifact_env": "${PROOF_ARTIFACT_ENV}",
  "continuation_proof_artifact_path": "${CONTINUATION_PROOF_ARTIFACT}",
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

echo "prepared staged boundary setup under ${OUTPUT_DIR}"
echo "env plan: ${ENV_FILE}"
echo "build command plan: ${BUILD_CMD_FILE}"
echo "prefix command plan: ${PREFIX_CMD_FILE}"
echo "continuation command plan: ${CONTINUATION_CMD_FILE}"

write_top_summary() {
  local build_json="${1}"
  local prefix_json="${2}"
  local continuation_json="${3}"
  cat >"${OUTPUT_DIR}/summary.json" <<EOF
{
  "setup": $(cat "${PLAN_DIR}/summary.json"),
  "build": ${build_json},
  "stages": {
    "prefix": ${prefix_json},
    "continuation": ${continuation_json}
  }
}
EOF
}

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  write_top_summary 'null' 'null' 'null'
  exit 0
fi

build_phase() {
  local stdout_file="${OUTPUT_DIR}/build.stdout"
  local stderr_file="${OUTPUT_DIR}/build.stderr"
  local status_file="${OUTPUT_DIR}/build-status.json"
  local start_epoch end_epoch duration rc state binary_state
  start_epoch=$(date +%s)
  rc=0
  (
    cd "${TREE}"
    . "${ENV_FILE}"
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${BUILD_TIMEOUT_SECONDS}" \
      cargo build --release -p gpt-oss-bench --features cuda --bin "${STAGE_BIN}"
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

  if [[ -x "${BINARY_PATH}" ]]; then
    binary_state="ready"
  else
    binary_state="missing"
  fi

  cat >"${status_file}" <<EOF
{
  "state": $(json_escape "${state}"),
  "exit_code": ${rc},
  "duration_seconds": ${duration},
  "binary_state": $(json_escape "${binary_state}"),
  "binary_path": $(json_escape "${BINARY_PATH}"),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "command_file": $(json_escape "${BUILD_CMD_FILE}")
}
EOF
}

run_stage() {
  local stage_name="$1"
  local prompt_file="$2"
  local output_file="$3"
  local command_file="$4"
  local stage_dir="${OUTPUT_DIR}/${stage_name}"
  local stdout_file="${stage_dir}/run.stdout"
  local stderr_file="${stage_dir}/run.stderr"
  local status_file="${stage_dir}/status.json"
  local proof_file=""
  local start_epoch end_epoch duration rc state artifact_state proof_state

  if [[ "${stage_name}" == "continuation" && -n "${PROOF_ARTIFACT_ENV}" ]]; then
    proof_file="${CONTINUATION_PROOF_ARTIFACT}"
  fi

  [[ -x "${BINARY_PATH}" ]] || die "prebuilt binary missing: ${BINARY_PATH}; run with --build-only first"

  start_epoch=$(date +%s)
  rc=0
  (
    cd "${TREE}"
    PROMPT=$(cat "${prompt_file}")
    . "${ENV_FILE}"
    if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
      unset "${PROOF_ARTIFACT_ENV}"
    fi
    if [[ -n "${proof_file}" ]]; then
      export "${PROOF_ARTIFACT_ENV}=${proof_file}"
    fi
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${TIMEOUT_SECONDS}" \
      "${BINARY_PATH}" \
      --model "${MODEL_PATH}" \
      --prompt "${PROMPT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --output "${output_file}"
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

  if [[ -f "${output_file}" ]]; then
    artifact_state="present"
  else
    artifact_state="missing"
  fi

  if [[ -n "${proof_file}" && -f "${proof_file}" ]]; then
    proof_state="present"
  elif [[ -n "${proof_file}" ]]; then
    proof_state="missing"
  else
    proof_state="not_requested"
  fi

  cat >"${status_file}" <<EOF
{
  "stage": $(json_escape "${stage_name}"),
  "state": $(json_escape "${state}"),
  "exit_code": ${rc},
  "duration_seconds": ${duration},
  "artifact_state": $(json_escape "${artifact_state}"),
  "artifact_path": $(json_escape "${output_file}"),
  "proof_artifact_env": $(json_escape "${PROOF_ARTIFACT_ENV}"),
  "proof_artifact_state": $(json_escape "${proof_state}"),
  "proof_artifact_path": $(json_escape "${proof_file}"),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "command_file": $(json_escape "${command_file}")
}
EOF
}

if [[ "${RUN_ONLY}" -ne 1 ]]; then
  build_phase
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  write_top_summary "$(cat "${OUTPUT_DIR}/build-status.json")" 'null' 'null'
  exit 0
fi

run_stage prefix "${PREFIX_PROMPT_FILE}" "${PREFIX_ARTIFACT}" "${PREFIX_CMD_FILE}"
run_stage continuation "${CONTINUATION_PROMPT_FILE}" "${CONTINUATION_ARTIFACT}" "${CONTINUATION_CMD_FILE}"

write_top_summary \
  "$( if [[ -f "${OUTPUT_DIR}/build-status.json" ]]; then cat "${OUTPUT_DIR}/build-status.json"; else printf 'null'; fi )" \
  "$(cat "${PREFIX_DIR}/status.json")" \
  "$(cat "${CONTINUATION_DIR}/status.json")"
