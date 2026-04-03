#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEFAULT_TREE="/home/emmy/openai/gpt-oss-rs"
DEFAULT_MODEL="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/.live/gpu0-live-smoke"
DEFAULT_GPU="0"
DEFAULT_TIMEOUT="1800"
DEFAULT_BUILD_TIMEOUT="1800"
DEFAULT_BIN="restricted_prefill_trace"
DEFAULT_MAX_MODEL_LEN="4608"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_gpu0_live_smoke.sh [options]

Prepare or run one bounded GPU live-smoke command with warm-build support.

Options:
  --gpu <id>                 CUDA_VISIBLE_DEVICES value to use (default: 0)
  --tree <path>              worktree to build/run from (default: /home/emmy/openai/gpt-oss-rs)
  --model <path>             model path passed to the live binary
  --prompt-file <path>       prompt file to pass to the live binary
  --timeout <seconds>        per-run timeout in seconds (default: 1800)
  --build-timeout <seconds>  build timeout in seconds (default: 1800)
  --output-dir <path>        output directory for plans, logs, and summary
  --bin <name>               live binary to build/run (default: restricted_prefill_trace)
  --max-model-len <n>        max-model-len passed to the live binary (default: 4608)
  --env KEY=VALUE            bounded env passthrough; repeatable
  --build-only               build/warm the tree and stop
  --run-only                 run only, requiring a prebuilt binary
  --setup-only               write plans/summary and stop
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
PROMPT_FILE=""
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
GPU_ID="${DEFAULT_GPU}"
TIMEOUT_SECONDS="${DEFAULT_TIMEOUT}"
BUILD_TIMEOUT_SECONDS="${DEFAULT_BUILD_TIMEOUT}"
LIVE_BIN="${DEFAULT_BIN}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
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
    --prompt-file)
      PROMPT_FILE="${2:?missing value for --prompt-file}"
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
      LIVE_BIN="${2:?missing value for --bin}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?missing value for --max-model-len}"
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

[[ -n "${PROMPT_FILE}" ]] || die "--prompt-file is required"
require_dir "${TREE}"
require_dir "${MODEL_PATH}"
require_file "${PROMPT_FILE}"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/RESTRICTED_MODEL_VIEW.json"
require_file "${TREE}/crates/gpt-oss-bench/src/bin/${LIVE_BIN}.rs"

OUTPUT_DIR=$(mkdir -p "${OUTPUT_DIR}" && cd -- "${OUTPUT_DIR}" && pwd)
PROMPT_FILE=$(cd -- "$(dirname -- "${PROMPT_FILE}")" && pwd)/$(basename -- "${PROMPT_FILE}")
PLAN_DIR="${OUTPUT_DIR}/plan"
RUN_DIR="${OUTPUT_DIR}/run"
mkdir -p "${PLAN_DIR}" "${RUN_DIR}"

ARTIFACT_NAME="$(normalize_name "${LIVE_BIN}")"
BINARY_PATH="${TREE}/target/release/${LIVE_BIN}"
ARTIFACT_PATH="${RUN_DIR}/${ARTIFACT_NAME}"
ENV_FILE="${PLAN_DIR}/live_env.sh"
ENV_JSON_FILE="${PLAN_DIR}/live_env.json"
BUILD_CMD_FILE="${PLAN_DIR}/build_command.sh"
RUN_CMD_FILE="${PLAN_DIR}/run_command.sh"
GPU_SNAPSHOT_FILE="${PLAN_DIR}/gpu_snapshot.txt"
RESOLVED_BINARY_PATH=""

resolve_binary_path() {
  local primary="${TREE}/target/release/${LIVE_BIN}"
  local deps_dir="${TREE}/target/release/deps"
  local primary_kind=""
  local candidate=""

  if [[ -x "${primary}" ]]; then
    primary_kind=$(file -b "${primary}" 2>/dev/null || true)
    if [[ "${primary_kind}" != *"shell script"* ]]; then
      printf '%s\n' "${primary}"
      return 0
    fi
  fi

  if [[ -d "${deps_dir}" ]]; then
    candidate=$(find "${deps_dir}" -maxdepth 1 -type f -name "${LIVE_BIN}-*" -perm -111 ! -name '*.d' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2- || true)
    if [[ -n "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  printf '%s\n' "${primary}"
}

{
  for env_kv in "${EXTRA_ENVS[@]}"; do
    key=${env_kv%%=*}
    value=${env_kv#*=}
    printf 'export %s=%s\n' "${key}" "$(shell_escape "${value}")"
  done
} >"${ENV_FILE}"

export LIVE_ENV_JSON_FILE="${ENV_JSON_FILE}"
if ((${#EXTRA_ENVS[@]} > 0)); then
  export LIVE_EXTRA_ENVS="$(printf '%s\n' "${EXTRA_ENVS[@]}")"
else
  export LIVE_EXTRA_ENVS=""
fi
python3 <<'PY'
import json
import os
from pathlib import Path

envs = []
raw = os.environ.get("LIVE_EXTRA_ENVS", "")
for line in raw.splitlines():
    if not line:
        continue
    key, value = line.split("=", 1)
    envs.append({"key": key, "value": value})
Path(os.environ["LIVE_ENV_JSON_FILE"]).write_text(json.dumps(envs, indent=2) + "\n")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"${GPU_SNAPSHOT_FILE}" 2>&1 || true
else
  printf 'nvidia-smi unavailable\n' >"${GPU_SNAPSHOT_FILE}"
fi

cat >"${BUILD_CMD_FILE}" <<EOF
cd ${TREE}
. ${ENV_FILE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo build --release -p gpt-oss-bench --features cuda --bin ${LIVE_BIN}
EOF

cat >"${RUN_CMD_FILE}" <<EOF
cd ${TREE}
PROMPT=\$(cat ${PROMPT_FILE})
. ${ENV_FILE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${BINARY_PATH} --model ${MODEL_PATH} --prompt "\${PROMPT}" --max-model-len ${MAX_MODEL_LEN} --output ${ARTIFACT_PATH}
EOF

chmod +x "${BUILD_CMD_FILE}" "${RUN_CMD_FILE}"

cat >"${PLAN_DIR}/summary.json" <<EOF
{
  "gpu_id": "${GPU_ID}",
  "tree": "${TREE}",
  "model_path": "${MODEL_PATH}",
  "prompt_file": "${PROMPT_FILE}",
  "live_bin": "${LIVE_BIN}",
  "artifact_path": "${ARTIFACT_PATH}",
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

echo "prepared live-smoke setup under ${OUTPUT_DIR}"
echo "env plan: ${ENV_FILE}"
echo "build command plan: ${BUILD_CMD_FILE}"
echo "run command plan: ${RUN_CMD_FILE}"

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  cat >"${OUTPUT_DIR}/summary.json" <<EOF
{
  "setup": $(cat "${PLAN_DIR}/summary.json"),
  "build": null,
  "run": null
}
EOF
  exit 0
fi

build_phase() {
  local stdout_file="${RUN_DIR}/build.stdout"
  local stderr_file="${RUN_DIR}/build.stderr"
  local status_file="${RUN_DIR}/build-status.json"
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
      cargo build --release -p gpt-oss-bench --features cuda --bin "${LIVE_BIN}"
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

  RESOLVED_BINARY_PATH=$(resolve_binary_path)
  if [[ -x "${RESOLVED_BINARY_PATH}" ]]; then
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
  "resolved_binary_path": $(json_escape "${RESOLVED_BINARY_PATH}"),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "command_file": $(json_escape "${BUILD_CMD_FILE}")
}
EOF
}

run_phase() {
  local stdout_file="${RUN_DIR}/run.stdout"
  local stderr_file="${RUN_DIR}/run.stderr"
  local status_file="${RUN_DIR}/run-status.json"
  local start_epoch end_epoch duration rc state artifact_state

  RESOLVED_BINARY_PATH=$(resolve_binary_path)
  [[ -x "${RESOLVED_BINARY_PATH}" ]] || die "prebuilt binary missing: ${RESOLVED_BINARY_PATH}; run with --build-only first"

  start_epoch=$(date +%s)
  rc=0
  (
    cd "${TREE}"
    PROMPT=$(cat "${PROMPT_FILE}")
    . "${ENV_FILE}"
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${TIMEOUT_SECONDS}" \
      "${RESOLVED_BINARY_PATH}" \
      --model "${MODEL_PATH}" \
      --prompt "${PROMPT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --output "${ARTIFACT_PATH}"
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

  if [[ -f "${ARTIFACT_PATH}" ]]; then
    artifact_state="present"
  else
    artifact_state="missing"
  fi

  cat >"${status_file}" <<EOF
{
  "state": $(json_escape "${state}"),
  "exit_code": ${rc},
  "duration_seconds": ${duration},
  "artifact_state": $(json_escape "${artifact_state}"),
  "artifact_path": $(json_escape "${ARTIFACT_PATH}"),
  "resolved_binary_path": $(json_escape "${RESOLVED_BINARY_PATH}"),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "command_file": $(json_escape "${RUN_CMD_FILE}")
}
EOF
}

if [[ "${RUN_ONLY}" -ne 1 ]]; then
  build_phase
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  cat >"${OUTPUT_DIR}/summary.json" <<EOF
{
  "setup": $(cat "${PLAN_DIR}/summary.json"),
  "build": $(cat "${RUN_DIR}/build-status.json"),
  "run": null
}
EOF
  exit 0
fi

run_phase

cat >"${OUTPUT_DIR}/summary.json" <<EOF
{
  "setup": $(cat "${PLAN_DIR}/summary.json"),
  "build": $( if [[ -f "${RUN_DIR}/build-status.json" ]]; then cat "${RUN_DIR}/build-status.json"; else printf 'null'; fi ),
  "run": $(cat "${RUN_DIR}/run-status.json")
}
EOF
