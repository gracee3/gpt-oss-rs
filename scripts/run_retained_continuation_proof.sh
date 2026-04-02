#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEFAULT_MODEL="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
DEFAULT_SAFE_TREE="/home/emmy/openai/gpt-oss-rs"
DEFAULT_VARIANT_TREE="/home/emmy/openai/worktrees/runtime-forward"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/.live/retained-continuation-proof"
DEFAULT_TIMEOUT="1800"
DEFAULT_BUILD_TIMEOUT="1800"
DEFAULT_GPU="0"
DEFAULT_BIN="restricted_prefill_trace"
DEFAULT_MAX_MODEL_LEN="4608"
DEFAULT_PYTHON="python3"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_retained_continuation_proof.sh [options]

Prepare or run a bounded retained-state continuation proof workflow.

Options:
  --gpu <id>                      CUDA_VISIBLE_DEVICES value to use (default: 0)
  --safe-tree <path>              worktree for the conservative baseline
  --variant-tree <path>           worktree for the candidate variant
  --model <path>                  restricted sink-free model view to use
  --prefix-prompt-file <path>     prefix/prefill prompt file for the retained-state invocation
  --continuation-prompt-file <p>  continuation prompt file for the retained-state invocation
  --bin <name>                    proof binary name to build/run (default: restricted_prefill_trace)
  --timeout <seconds>             per-run timeout in seconds (default: 1800)
  --build-timeout <seconds>       per-build timeout in seconds (default: 1800)
  --output-dir <path>             output directory for plans, logs, artifacts, and summary
  --env KEY=VALUE                 bounded env passthrough; repeatable
  --proof-artifact-env <name>     env var name used to pass a compact continuation proof artifact path
  --proof-artifact-name <file>    filename for per-side compact continuation proof artifacts
  --compare-vector-key <key>      compact numeric vector field to diff when present
  --marker-profile <name>         built-in ordered progress-marker profile; repeatable
  --progress-marker <string>      retained child progress marker to scan for; repeatable
  --emit-forced-output-tokens     emit command-ready forced-output token args from verified continuation ids
  --verify-tokenization           verify prefix/continuation tokenization against the selected model
  --python <path>                 python interpreter for token verification (default: python3)
  --required-prefix-token-count <n>
                                  fail closed unless the prefix token count matches
  --required-continuation-token-count <n>
                                  fail closed unless the continuation token count matches
  --max-model-len <n>             max-model-len passed to the binary
  --build-only                    prepare/warm both trees, then stop before execution
  --run-only                      execute using prebuilt binaries only
  --setup-only                    stop after writing plans and summary
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

normalize_artifact_name() {
  local name="$1"
  name=${name//\//-}
  name=${name// /-}
  printf '%s.json' "${name}"
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
PREFIX_PROMPT_FILE=""
CONTINUATION_PROMPT_FILE=""
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
TIMEOUT_SECONDS="${DEFAULT_TIMEOUT}"
BUILD_TIMEOUT_SECONDS="${DEFAULT_BUILD_TIMEOUT}"
GPU_ID="${DEFAULT_GPU}"
PROOF_BIN="${DEFAULT_BIN}"
PROOF_ARTIFACT_ENV=""
PROOF_ARTIFACT_NAME=""
COMPARE_VECTOR_KEY=""
EMIT_FORCED_OUTPUT_TOKENS=0
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
MAX_MODEL_LEN_EXPLICIT=0
PYTHON_BIN="${DEFAULT_PYTHON}"
VERIFY_TOKENIZATION=0
REQUIRED_PREFIX_TOKEN_COUNT=""
REQUIRED_CONTINUATION_TOKEN_COUNT=""
BUILD_ONLY=0
RUN_ONLY=0
SETUP_ONLY=0
declare -a EXTRA_ENVS=()
declare -a MARKER_PROFILES=()
declare -a PROGRESS_MARKERS=()

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
    --prefix-prompt-file)
      PREFIX_PROMPT_FILE="${2:?missing value for --prefix-prompt-file}"
      shift 2
      ;;
    --continuation-prompt-file)
      CONTINUATION_PROMPT_FILE="${2:?missing value for --continuation-prompt-file}"
      shift 2
      ;;
    --bin)
      PROOF_BIN="${2:?missing value for --bin}"
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
    --env)
      [[ "${2:-}" == *=* ]] || die "--env requires KEY=VALUE"
      EXTRA_ENVS+=("${2}")
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
    --compare-vector-key)
      COMPARE_VECTOR_KEY="${2:?missing value for --compare-vector-key}"
      shift 2
      ;;
    --marker-profile)
      MARKER_PROFILES+=("${2:?missing value for --marker-profile}")
      shift 2
      ;;
    --progress-marker)
      PROGRESS_MARKERS+=("${2:?missing value for --progress-marker}")
      shift 2
      ;;
    --emit-forced-output-tokens)
      EMIT_FORCED_OUTPUT_TOKENS=1
      shift
      ;;
    --verify-tokenization)
      VERIFY_TOKENIZATION=1
      shift
      ;;
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --required-prefix-token-count)
      REQUIRED_PREFIX_TOKEN_COUNT="${2:?missing value for --required-prefix-token-count}"
      shift 2
      ;;
    --required-continuation-token-count)
      REQUIRED_CONTINUATION_TOKEN_COUNT="${2:?missing value for --required-continuation-token-count}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?missing value for --max-model-len}"
      MAX_MODEL_LEN_EXPLICIT=1
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

require_dir "${SAFE_TREE}"
require_dir "${VARIANT_TREE}"
require_dir "${MODEL_PATH}"
require_file "${MODEL_PATH}/config.json"
require_file "${MODEL_PATH}/RESTRICTED_MODEL_VIEW.json"
require_file "${PREFIX_PROMPT_FILE}"
require_file "${CONTINUATION_PROMPT_FILE}"
require_file "${SAFE_TREE}/crates/gpt-oss-bench/src/bin/${PROOF_BIN}.rs"
require_file "${VARIANT_TREE}/crates/gpt-oss-bench/src/bin/${PROOF_BIN}.rs"

OUTPUT_DIR=$(mkdir -p "${OUTPUT_DIR}" && cd -- "${OUTPUT_DIR}" && pwd)
PREFIX_PROMPT_FILE=$(cd -- "$(dirname -- "${PREFIX_PROMPT_FILE}")" && pwd)/$(basename -- "${PREFIX_PROMPT_FILE}")
CONTINUATION_PROMPT_FILE=$(cd -- "$(dirname -- "${CONTINUATION_PROMPT_FILE}")" && pwd)/$(basename -- "${CONTINUATION_PROMPT_FILE}")
PLAN_DIR="${OUTPUT_DIR}/plan"
mkdir -p "${PLAN_DIR}" "${OUTPUT_DIR}/safe" "${OUTPUT_DIR}/variant"

MARKER_PROFILES_JSON_FILE="${PLAN_DIR}/marker_profiles.json"
export RETAINED_MARKER_PROFILES_JSON_FILE="${MARKER_PROFILES_JSON_FILE}"
if ((${#MARKER_PROFILES[@]} > 0)); then
  export RETAINED_MARKER_PROFILES_RAW="$(printf '%s\n' "${MARKER_PROFILES[@]}")"
else
  export RETAINED_MARKER_PROFILES_RAW=""
fi
if ((${#PROGRESS_MARKERS[@]} > 0)); then
  export RETAINED_PROGRESS_MARKERS_RAW="$(printf '%s\n' "${PROGRESS_MARKERS[@]}")"
else
  export RETAINED_PROGRESS_MARKERS_RAW=""
fi
python3 <<'PY'
import json
import os
import sys
from pathlib import Path

profiles = [line for line in os.environ.get("RETAINED_MARKER_PROFILES_RAW", "").splitlines() if line]
manual_markers = [line for line in os.environ.get("RETAINED_PROGRESS_MARKERS_RAW", "").splitlines() if line]
builtins = {
    "retained-debug-v1": [
        "RETAINED_CHILD_START",
        "RETAINED_CHILD_TOKENIZED",
        "RETAINED_CHILD_BUILD_WORKER_BEGIN",
        "RETAINED_CHILD_BUILD_WORKER_DONE",
        "RETAINED_STEP_BEGIN",
        "RETAINED_STEP_FORWARD_BEGIN",
        "RETAINED_STEP_FORWARD_DONE",
        "DECODE1_BEGIN",
        "RETAINED_PROOF_ENTER",
        "RETAINED_PROOF_CAPTURED",
    ],
    "retained-mlp-v1": [
        "RETAINED_CHILD_START",
        "RETAINED_CHILD_TOKENIZED",
        "RETAINED_CHILD_BUILD_WORKER_DONE",
        "RETAINED_STEP_BEGIN",
        "RETAINED_STEP_FORWARD_BEGIN",
        "RETAINED_PREFILL_STAGE layer=0 stage=layer_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=residual_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=router_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=router_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=expert_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=expert_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_done",
        "RETAINED_STEP_FORWARD_DONE",
        "DECODE1_BEGIN",
        "RETAINED_PROOF_ENTER",
        "RETAINED_PROOF_CAPTURED",
    ],
    "retained-mlp-prerouter-v1": [
        "RETAINED_CHILD_START",
        "RETAINED_CHILD_TOKENIZED",
        "RETAINED_CHILD_BUILD_WORKER_DONE",
        "RETAINED_STEP_BEGIN",
        "RETAINED_STEP_FORWARD_BEGIN",
        "RETAINED_PREFILL_STAGE layer=0 stage=layer_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=residual_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_done",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_begin",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_done",
        "RETAINED_MLP_STAGE layer=0 stage=aggregate_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_done",
        "RETAINED_STEP_FORWARD_DONE",
        "DECODE1_BEGIN",
        "RETAINED_PROOF_ENTER",
        "RETAINED_PROOF_CAPTURED",
    ],
    "retained-router-v1": [
        "RETAINED_CHILD_START",
        "RETAINED_CHILD_TOKENIZED",
        "RETAINED_CHILD_BUILD_WORKER_DONE",
        "RETAINED_STEP_BEGIN",
        "RETAINED_STEP_FORWARD_BEGIN",
        "RETAINED_PREFILL_STAGE layer=0 stage=layer_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=residual_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_setup_done",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_cast_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_invoke_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_enter",
        "RETAINED_MLP_STAGE layer=0 stage=router_scores_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_done",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_dispatch_begin",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_dispatch_done",
        "RETAINED_MLP_STAGE layer=0 stage=aggregate_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_done",
        "RETAINED_STEP_FORWARD_DONE",
        "DECODE1_BEGIN",
        "RETAINED_PROOF_ENTER",
        "RETAINED_PROOF_CAPTURED",
    ],
    "retained-router-host-v1": [
        "RETAINED_CHILD_START",
        "RETAINED_CHILD_TOKENIZED",
        "RETAINED_CHILD_BUILD_WORKER_DONE",
        "RETAINED_STEP_BEGIN",
        "RETAINED_STEP_FORWARD_BEGIN",
        "RETAINED_PREFILL_STAGE layer=0 stage=layer_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=residual_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_setup_done",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_cast_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_invoke_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_call_entered",
        "RETAINED_MLP_STAGE layer=0 stage=router_dtoh_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_dtoh_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_input_ready",
        "RETAINED_MLP_STAGE layer=0 stage=router_score_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_done",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_dispatch_begin",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_dispatch_done",
        "RETAINED_MLP_STAGE layer=0 stage=aggregate_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_done",
        "RETAINED_STEP_FORWARD_DONE",
        "DECODE1_BEGIN",
        "RETAINED_PROOF_ENTER",
        "RETAINED_PROOF_CAPTURED",
    ],
    "retained-router-score-v1": [
        "RETAINED_CHILD_START",
        "RETAINED_CHILD_TOKENIZED",
        "RETAINED_CHILD_BUILD_WORKER_DONE",
        "RETAINED_STEP_BEGIN",
        "RETAINED_STEP_FORWARD_BEGIN",
        "RETAINED_PREFILL_STAGE layer=0 stage=layer_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_begin",
        "RETAINED_PREFILL_STAGE layer=0 stage=attention_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=residual_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_begin",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_setup_done",
        "RETAINED_MLP_STAGE layer=0 stage=prerouter_cast_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_invoke_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_call_entered",
        "RETAINED_MLP_STAGE layer=0 stage=router_dtoh_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_dtoh_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_input_ready",
        "RETAINED_MLP_STAGE layer=0 stage=router_score_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_score_alloc_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_score_alloc_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_score_first_accum_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_score_first_accum_done",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_begin",
        "RETAINED_MLP_STAGE layer=0 stage=router_topk_done",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_dispatch_begin",
        "RETAINED_MLP_STAGE layer=0 stage=first_expert_dispatch_done",
        "RETAINED_MLP_STAGE layer=0 stage=aggregate_done",
        "RETAINED_PREFILL_STAGE layer=0 stage=mlp_done",
        "RETAINED_STEP_FORWARD_DONE",
        "DECODE1_BEGIN",
        "RETAINED_PROOF_ENTER",
        "RETAINED_PROOF_CAPTURED",
    ],
}
unknown = [name for name in profiles if name not in builtins]
if unknown:
    print(f"error: unknown --marker-profile value(s): {', '.join(unknown)}", file=sys.stderr)
    sys.exit(1)
sequence = []
for name in profiles:
    for marker in builtins[name]:
        if marker not in sequence:
            sequence.append(marker)
for marker in manual_markers:
    if marker not in sequence:
        sequence.append(marker)
Path(os.environ["RETAINED_MARKER_PROFILES_JSON_FILE"]).write_text(
    json.dumps(
        {
            "selected_profiles": profiles,
            "configured_marker_sequence": sequence,
            "builtin_profiles": {name: builtins[name] for name in profiles},
            "manual_markers": manual_markers,
        },
        indent=2,
    )
    + "\n"
)
PY

mapfile -t PROGRESS_MARKERS < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["configured_marker_sequence"]))' "${MARKER_PROFILES_JSON_FILE}")

ARTIFACT_NAME="$(normalize_artifact_name "${PROOF_BIN}")"
if [[ -z "${PROOF_ARTIFACT_NAME}" ]]; then
  PROOF_ARTIFACT_NAME="$(normalize_artifact_name "${PROOF_BIN}-continuation-proof")"
fi

SAFE_BINARY="${SAFE_TREE}/target/release/${PROOF_BIN}"
VARIANT_BINARY="${VARIANT_TREE}/target/release/${PROOF_BIN}"
SAFE_OUTER_ARTIFACT="${OUTPUT_DIR}/safe/${ARTIFACT_NAME}"
VARIANT_OUTER_ARTIFACT="${OUTPUT_DIR}/variant/${ARTIFACT_NAME}"
SAFE_PROOF_ARTIFACT="${OUTPUT_DIR}/safe/${PROOF_ARTIFACT_NAME}"
VARIANT_PROOF_ARTIFACT="${OUTPUT_DIR}/variant/${PROOF_ARTIFACT_NAME}"
ENV_FILE="${PLAN_DIR}/retained_env.sh"
ENV_JSON_FILE="${PLAN_DIR}/retained_env.json"
SAFE_BUILD_CMD_FILE="${PLAN_DIR}/safe_build_command.sh"
VARIANT_BUILD_CMD_FILE="${PLAN_DIR}/variant_build_command.sh"
SAFE_RUN_CMD_FILE="${PLAN_DIR}/safe_run_command.sh"
VARIANT_RUN_CMD_FILE="${PLAN_DIR}/variant_run_command.sh"
GPU_SNAPSHOT_FILE="${PLAN_DIR}/gpu_snapshot.txt"

{
  for env_kv in "${EXTRA_ENVS[@]}"; do
    key=${env_kv%%=*}
    value=${env_kv#*=}
    printf 'export %s=%s\n' "${key}" "$(shell_escape "${value}")"
  done
  if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
    printf 'export %s_SAFE=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${SAFE_PROOF_ARTIFACT}")"
    printf 'export %s_VARIANT=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${VARIANT_PROOF_ARTIFACT}")"
  fi
} >"${ENV_FILE}"

export RETAINED_ENV_JSON_FILE="${ENV_JSON_FILE}"
export RETAINED_PROOF_ARTIFACT_ENV="${PROOF_ARTIFACT_ENV}"
export RETAINED_SAFE_PROOF_ARTIFACT="${SAFE_PROOF_ARTIFACT}"
export RETAINED_VARIANT_PROOF_ARTIFACT="${VARIANT_PROOF_ARTIFACT}"
export RETAINED_TOKEN_MODEL_PATH="${MODEL_PATH}"
export RETAINED_PREFIX_PROMPT_FILE="${PREFIX_PROMPT_FILE}"
export RETAINED_CONTINUATION_PROMPT_FILE="${CONTINUATION_PROMPT_FILE}"
export RETAINED_EMIT_FORCED_OUTPUT_TOKENS="${EMIT_FORCED_OUTPUT_TOKENS}"
if ((${#EXTRA_ENVS[@]} > 0)); then
  export RETAINED_EXTRA_ENVS="$(printf '%s\n' "${EXTRA_ENVS[@]}")"
else
  export RETAINED_EXTRA_ENVS=""
fi
python3 <<'PY'
import json
import os
from pathlib import Path

envs = []
raw = os.environ.get("RETAINED_EXTRA_ENVS", "")
for line in raw.splitlines():
    if not line:
        continue
    key, value = line.split("=", 1)
    envs.append({"key": key, "value": value})
proof_key = os.environ.get("RETAINED_PROOF_ARTIFACT_ENV", "")
if proof_key:
    envs.append({"key": f"{proof_key}_SAFE", "value": os.environ["RETAINED_SAFE_PROOF_ARTIFACT"]})
    envs.append({"key": f"{proof_key}_VARIANT", "value": os.environ["RETAINED_VARIANT_PROOF_ARTIFACT"]})
Path(os.environ["RETAINED_ENV_JSON_FILE"]).write_text(json.dumps(envs, indent=2) + "\n")
PY

TOKENIZATION_JSON_FILE="${PLAN_DIR}/tokenization_summary.json"
if [[ "${VERIFY_TOKENIZATION}" -eq 1 ]]; then
export RETAINED_TOKENIZATION_JSON_FILE="${TOKENIZATION_JSON_FILE}"
export RETAINED_REQUIRED_PREFIX_TOKEN_COUNT="${REQUIRED_PREFIX_TOKEN_COUNT}"
export RETAINED_REQUIRED_CONTINUATION_TOKEN_COUNT="${REQUIRED_CONTINUATION_TOKEN_COUNT}"
export RETAINED_PYTHON_BIN="${PYTHON_BIN}"
export RETAINED_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export RETAINED_MAX_MODEL_LEN_EXPLICIT="${MAX_MODEL_LEN_EXPLICIT}"
  PATH="/data/models/.venv-awq/bin:${PATH}" "${PYTHON_BIN}" <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

model = os.environ["RETAINED_TOKEN_MODEL_PATH"]
prefix_path = Path(os.environ["RETAINED_PREFIX_PROMPT_FILE"])
continuation_path = Path(os.environ["RETAINED_CONTINUATION_PROMPT_FILE"])
output_path = Path(os.environ["RETAINED_TOKENIZATION_JSON_FILE"])
required_prefix = os.environ.get("RETAINED_REQUIRED_PREFIX_TOKEN_COUNT", "")
required_cont = os.environ.get("RETAINED_REQUIRED_CONTINUATION_TOKEN_COUNT", "")

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prefix_text = prefix_path.read_text()
continuation_text = continuation_path.read_text()
prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False)

summary = {
    "verified": True,
    "python": os.environ.get("RETAINED_PYTHON_BIN"),
    "prefix_token_count": len(prefix_ids),
    "continuation_token_count": len(continuation_ids),
    "required_min_model_len": len(prefix_ids) + len(continuation_ids),
    "max_model_len": int(os.environ["RETAINED_MAX_MODEL_LEN"]),
    "max_model_len_explicit": os.environ.get("RETAINED_MAX_MODEL_LEN_EXPLICIT") == "1",
    "continuation_token_ids": continuation_ids,
    "continuation_token_ids_csv": ",".join(str(token) for token in continuation_ids),
    "forced_output_tokens_arg": "",
    "continuation_text_repr": repr(continuation_text),
}
if os.environ.get("RETAINED_EMIT_FORCED_OUTPUT_TOKENS") == "1":
    summary["forced_output_tokens_arg"] = " ".join(
        f"--forced-output-tokens {token}" for token in continuation_ids
    )
output_path.write_text(json.dumps(summary, indent=2) + "\n")

if not summary["max_model_len_explicit"]:
    raise SystemExit(
        "strict token verification requires an explicit --max-model-len"
    )
if summary["max_model_len"] < summary["required_min_model_len"]:
    raise SystemExit(
        f"--max-model-len {summary['max_model_len']} is below required minimum "
        f"{summary['required_min_model_len']}"
    )
if required_prefix and len(prefix_ids) != int(required_prefix):
    raise SystemExit(
        f"prefix token count {len(prefix_ids)} did not match required {required_prefix}"
    )
if required_cont and len(continuation_ids) != int(required_cont):
    raise SystemExit(
        f"continuation token count {len(continuation_ids)} did not match required {required_cont}"
    )
PY
else
  cat >"${TOKENIZATION_JSON_FILE}" <<EOF
{
  "verified": false,
  "python": $(json_escape "${PYTHON_BIN}"),
  "prefix_token_count": null,
  "continuation_token_count": null,
  "required_min_model_len": null,
  "max_model_len": ${MAX_MODEL_LEN},
  "max_model_len_explicit": ${MAX_MODEL_LEN_EXPLICIT},
  "continuation_token_ids": [],
  "continuation_token_ids_csv": "",
  "forced_output_tokens_arg": "",
  "continuation_text_repr": null
}
EOF
fi

FORCED_OUTPUT_TOKENS_ARG=""
if [[ "${EMIT_FORCED_OUTPUT_TOKENS}" -eq 1 && "${VERIFY_TOKENIZATION}" -eq 1 ]]; then
  FORCED_OUTPUT_TOKENS_ARG=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("forced_output_tokens_arg",""))' "${TOKENIZATION_JSON_FILE}")
fi

PROGRESS_MARKERS_JSON_FILE="${PLAN_DIR}/progress_markers.json"
export RETAINED_PROGRESS_MARKERS_JSON_FILE="${PROGRESS_MARKERS_JSON_FILE}"
export RETAINED_MARKER_PROFILES_JSON_FILE="${MARKER_PROFILES_JSON_FILE}"
python3 <<'PY'
import json
import os
from pathlib import Path

profile_data = json.loads(Path(os.environ["RETAINED_MARKER_PROFILES_JSON_FILE"]).read_text())
Path(os.environ["RETAINED_PROGRESS_MARKERS_JSON_FILE"]).write_text(
    json.dumps(profile_data["configured_marker_sequence"], indent=2) + "\n"
)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"${GPU_SNAPSHOT_FILE}" 2>&1 || true
else
  printf 'nvidia-smi unavailable\n' >"${GPU_SNAPSHOT_FILE}"
fi

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

cat >"${SAFE_RUN_CMD_FILE}" <<EOF
cd ${SAFE_TREE}
. ${ENV_FILE}
$( if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${SAFE_PROOF_ARTIFACT}")"; fi )
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${SAFE_BINARY} --model ${MODEL_PATH} --prefix-prompt-file ${PREFIX_PROMPT_FILE} --continuation-prompt-file ${CONTINUATION_PROMPT_FILE} --max-model-len ${MAX_MODEL_LEN} $( if [[ "${PROOF_BIN}" == "restricted_logit_diff" && -n "${FORCED_OUTPUT_TOKENS_ARG}" ]]; then printf '%s ' "${FORCED_OUTPUT_TOKENS_ARG}"; fi )--output ${SAFE_OUTER_ARTIFACT}
EOF

cat >"${VARIANT_RUN_CMD_FILE}" <<EOF
cd ${VARIANT_TREE}
. ${ENV_FILE}
$( if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then printf 'export %s=%s\n' "${PROOF_ARTIFACT_ENV}" "$(shell_escape "${VARIANT_PROOF_ARTIFACT}")"; fi )
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 ${VARIANT_BINARY} --model ${MODEL_PATH} --prefix-prompt-file ${PREFIX_PROMPT_FILE} --continuation-prompt-file ${CONTINUATION_PROMPT_FILE} --max-model-len ${MAX_MODEL_LEN} $( if [[ "${PROOF_BIN}" == "restricted_logit_diff" && -n "${FORCED_OUTPUT_TOKENS_ARG}" ]]; then printf '%s ' "${FORCED_OUTPUT_TOKENS_ARG}"; fi )--output ${VARIANT_OUTER_ARTIFACT}
EOF

chmod +x "${SAFE_BUILD_CMD_FILE}" "${VARIANT_BUILD_CMD_FILE}" "${SAFE_RUN_CMD_FILE}" "${VARIANT_RUN_CMD_FILE}"

cat >"${PLAN_DIR}/setup_summary.json" <<EOF
{
  "gpu_id": "${GPU_ID}",
  "safe_tree": "${SAFE_TREE}",
  "variant_tree": "${VARIANT_TREE}",
  "model_path": "${MODEL_PATH}",
  "proof_bin": "${PROOF_BIN}",
  "prefix_prompt_file": "${PREFIX_PROMPT_FILE}",
  "continuation_prompt_file": "${CONTINUATION_PROMPT_FILE}",
  "outer_artifact_name": "${ARTIFACT_NAME}",
  "proof_artifact_env": "${PROOF_ARTIFACT_ENV}",
  "proof_artifact_name": "${PROOF_ARTIFACT_NAME}",
  "safe_outer_artifact_path": "${SAFE_OUTER_ARTIFACT}",
  "variant_outer_artifact_path": "${VARIANT_OUTER_ARTIFACT}",
  "safe_proof_artifact_path": "${SAFE_PROOF_ARTIFACT}",
  "variant_proof_artifact_path": "${VARIANT_PROOF_ARTIFACT}",
  "compare_vector_key": "${COMPARE_VECTOR_KEY}",
  "forced_output_tokens_arg": $(json_escape "${FORCED_OUTPUT_TOKENS_ARG}"),
  "marker_profiles_file": "${MARKER_PROFILES_JSON_FILE}",
  "selected_marker_profiles": $(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["selected_profiles"]))' "${MARKER_PROFILES_JSON_FILE}"),
  "configured_progress_markers": $(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["configured_marker_sequence"]))' "${MARKER_PROFILES_JSON_FILE}"),
  "verify_tokenization": ${VERIFY_TOKENIZATION},
  "emit_forced_output_tokens": ${EMIT_FORCED_OUTPUT_TOKENS},
  "python": "${PYTHON_BIN}",
  "max_model_len": ${MAX_MODEL_LEN},
  "max_model_len_explicit": ${MAX_MODEL_LEN_EXPLICIT},
  "required_prefix_token_count": ${REQUIRED_PREFIX_TOKEN_COUNT:-null},
  "required_continuation_token_count": ${REQUIRED_CONTINUATION_TOKEN_COUNT:-null},
  "tokenization_summary_file": "${TOKENIZATION_JSON_FILE}",
  "progress_markers_file": "${PROGRESS_MARKERS_JSON_FILE}",
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

echo "prepared retained continuation setup under ${OUTPUT_DIR}"
echo "env plan: ${ENV_FILE}"
echo "safe build plan: ${SAFE_BUILD_CMD_FILE}"
echo "variant build plan: ${VARIANT_BUILD_CMD_FILE}"
echo "safe run plan: ${SAFE_RUN_CMD_FILE}"
echo "variant run plan: ${VARIANT_RUN_CMD_FILE}"

write_summary() {
  local safe_build_json="$1"
  local variant_build_json="$2"
  local safe_run_json="$3"
  local variant_run_json="$4"
  cat >"${OUTPUT_DIR}/summary.json" <<EOF
{
  "setup": $(cat "${PLAN_DIR}/setup_summary.json"),
  "tokenization": $(cat "${TOKENIZATION_JSON_FILE}"),
  "safe_build": ${safe_build_json},
  "variant_build": ${variant_build_json},
  "safe": ${safe_run_json},
  "variant": ${variant_run_json}
}
EOF
}

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  write_summary 'null' 'null' 'null' 'null'
  exit 0
fi

build_case() {
  local label="$1"
  local tree="$2"
  local binary_path="$3"
  local command_file="$4"
  local case_dir="${OUTPUT_DIR}/${label}"
  local stdout_file="${case_dir}/build.stdout"
  local stderr_file="${case_dir}/build.stderr"
  local status_file="${case_dir}/build-status.json"
  local start_epoch end_epoch duration rc state binary_state
  mkdir -p "${case_dir}"

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
  local outer_artifact="$4"
  local proof_artifact="$5"
  local command_file="$6"
  local case_dir="${OUTPUT_DIR}/${label}"
  local stdout_file="${case_dir}/run.stdout"
  local stderr_file="${case_dir}/run.stderr"
  local status_file="${case_dir}/status.json"
  local start_epoch end_epoch duration rc state artifact_state proof_state primary_artifact
  local progress_json_file="${case_dir}/progress-status.json"
  mkdir -p "${case_dir}"

  [[ -x "${binary_path}" ]] || die "prebuilt binary missing for ${label}: ${binary_path}; run with --build-only first"

  start_epoch=$(date +%s)
  rc=0
  (
    cd "${tree}"
    . "${ENV_FILE}"
    if [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
      export "${PROOF_ARTIFACT_ENV}=${proof_artifact}"
    fi
    PATH="/data/models/.venv-awq/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    timeout "${TIMEOUT_SECONDS}" \
      "${binary_path}" \
      --model "${MODEL_PATH}" \
      --prefix-prompt-file "${PREFIX_PROMPT_FILE}" \
      --continuation-prompt-file "${CONTINUATION_PROMPT_FILE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --output "${outer_artifact}"
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

  if [[ -f "${outer_artifact}" ]]; then
    artifact_state="usable_artifact"
  else
    artifact_state="no_artifact"
  fi

  if [[ -n "${PROOF_ARTIFACT_ENV}" && -f "${proof_artifact}" ]]; then
    proof_state="usable_artifact"
    primary_artifact="${proof_artifact}"
  elif [[ -n "${PROOF_ARTIFACT_ENV}" ]]; then
    proof_state="no_artifact"
    primary_artifact="${outer_artifact}"
  else
    proof_state="not_requested"
    primary_artifact="${outer_artifact}"
  fi

  export RETAINED_PROGRESS_STDOUT_FILE="${stdout_file}"
  export RETAINED_PROGRESS_STDERR_FILE="${stderr_file}"
  export RETAINED_PROGRESS_MARKERS_RAW="$(printf '%s\n' "${PROGRESS_MARKERS[@]}")"
  export RETAINED_PROGRESS_JSON_FILE="${progress_json_file}"
  python3 <<'PY'
import json
import os
from pathlib import Path

stdout_text = Path(os.environ["RETAINED_PROGRESS_STDOUT_FILE"]).read_text() if Path(os.environ["RETAINED_PROGRESS_STDOUT_FILE"]).exists() else ""
stderr_text = Path(os.environ["RETAINED_PROGRESS_STDERR_FILE"]).read_text() if Path(os.environ["RETAINED_PROGRESS_STDERR_FILE"]).exists() else ""
combined = stdout_text + "\n" + stderr_text
markers = [line for line in os.environ.get("RETAINED_PROGRESS_MARKERS_RAW", "").splitlines() if line]
seen = [marker for marker in markers if marker in combined]
last = seen[-1] if seen else None
next_missing = None
for marker in markers:
    if marker not in seen:
        next_missing = marker
        break
stall = f"stalled_before={next_missing}" if next_missing else (f"completed_through={last}" if last else "no_markers_seen")
summary = {
    "configured_markers": markers,
    "seen_markers": seen,
    "last_progress_marker": last,
    "next_expected_marker": next_missing,
    "stall_classification": stall,
}
Path(os.environ["RETAINED_PROGRESS_JSON_FILE"]).write_text(json.dumps(summary, indent=2) + "\n")
PY

  cat >"${status_file}" <<EOF
{
  "label": $(json_escape "${label}"),
  "state": $(json_escape "${state}"),
  "exit_code": ${rc},
  "duration_seconds": ${duration},
  "artifact_state": $(json_escape "${artifact_state}"),
  "outer_artifact_path": $(json_escape "${outer_artifact}"),
  "proof_artifact_env": $(json_escape "${PROOF_ARTIFACT_ENV}"),
  "proof_artifact_state": $(json_escape "${proof_state}"),
  "proof_artifact_path": $(json_escape "${proof_artifact}"),
  "primary_artifact_file": $(json_escape "${primary_artifact}"),
  "progress_markers_file": $(json_escape "${progress_json_file}"),
  "configured_progress_markers": $(cat "${progress_json_file}" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["configured_markers"]))'),
  "seen_progress_markers": $(cat "${progress_json_file}" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["seen_markers"]))'),
  "last_progress_marker": $(cat "${progress_json_file}" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["last_progress_marker"]))'),
  "next_expected_progress_marker": $(cat "${progress_json_file}" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["next_expected_marker"]))'),
  "progress_stall_classification": $(cat "${progress_json_file}" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["stall_classification"]))'),
  "stdout_file": $(json_escape "${stdout_file}"),
  "stderr_file": $(json_escape "${stderr_file}"),
  "command_file": $(json_escape "${command_file}")
}
EOF
}

if [[ "${RUN_ONLY}" -ne 1 ]]; then
  build_case safe "${SAFE_TREE}" "${SAFE_BINARY}" "${SAFE_BUILD_CMD_FILE}"
  build_case variant "${VARIANT_TREE}" "${VARIANT_BINARY}" "${VARIANT_BUILD_CMD_FILE}"
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  write_summary \
    "$(cat "${OUTPUT_DIR}/safe/build-status.json")" \
    "$(cat "${OUTPUT_DIR}/variant/build-status.json")" \
    'null' \
    'null'
  exit 0
fi

run_case safe "${SAFE_TREE}" "${SAFE_BINARY}" "${SAFE_OUTER_ARTIFACT}" "${SAFE_PROOF_ARTIFACT}" "${SAFE_RUN_CMD_FILE}"
run_case variant "${VARIANT_TREE}" "${VARIANT_BINARY}" "${VARIANT_OUTER_ARTIFACT}" "${VARIANT_PROOF_ARTIFACT}" "${VARIANT_RUN_CMD_FILE}"

write_summary \
  "$( if [[ -f "${OUTPUT_DIR}/safe/build-status.json" ]]; then cat "${OUTPUT_DIR}/safe/build-status.json"; else printf 'null'; fi )" \
  "$( if [[ -f "${OUTPUT_DIR}/variant/build-status.json" ]]; then cat "${OUTPUT_DIR}/variant/build-status.json"; else printf 'null'; fi )" \
  "$(cat "${OUTPUT_DIR}/safe/status.json")" \
  "$(cat "${OUTPUT_DIR}/variant/status.json")"

export RETAINED_OUTPUT_DIR="${OUTPUT_DIR}"
export RETAINED_COMPARE_VECTOR_KEY="${COMPARE_VECTOR_KEY}"
python3 <<'PY'
import hashlib
import json
import os
from pathlib import Path

output_dir = Path(os.environ["RETAINED_OUTPUT_DIR"])
compare_vector_key = os.environ.get("RETAINED_COMPARE_VECTOR_KEY", "")
summary = json.loads((output_dir / "summary.json").read_text())

safe_status = json.loads((output_dir / "safe" / "status.json").read_text())
variant_status = json.loads((output_dir / "variant" / "status.json").read_text())
summary["safe"] = safe_status
summary["variant"] = variant_status

safe_artifact = Path(safe_status["primary_artifact_file"])
variant_artifact = Path(variant_status["primary_artifact_file"])

if safe_artifact.exists() and variant_artifact.exists():
    safe_text = safe_artifact.read_text()
    variant_text = variant_artifact.read_text()
    safe = json.loads(safe_text)
    variant = json.loads(variant_text)
    safe_keys = sorted(safe.keys()) if isinstance(safe, dict) else None
    variant_keys = sorted(variant.keys()) if isinstance(variant, dict) else None
    comparison = {
        "comparable": True,
        "safe_primary_artifact": str(safe_artifact),
        "variant_primary_artifact": str(variant_artifact),
        "safe_sha256": hashlib.sha256(safe_text.encode("utf-8")).hexdigest(),
        "variant_sha256": hashlib.sha256(variant_text.encode("utf-8")).hexdigest(),
        "json_equal": safe == variant,
        "safe_top_level_keys": safe_keys,
        "variant_top_level_keys": variant_keys,
        "same_top_level_keys": safe_keys == variant_keys,
    }
    if (
        compare_vector_key
        and isinstance(safe, dict)
        and isinstance(variant, dict)
        and isinstance(safe.get(compare_vector_key), list)
        and isinstance(variant.get(compare_vector_key), list)
        and len(safe[compare_vector_key]) == len(variant[compare_vector_key])
        and all(isinstance(v, (int, float)) for v in safe[compare_vector_key])
        and all(isinstance(v, (int, float)) for v in variant[compare_vector_key])
    ):
        diffs = [abs(float(a) - float(b)) for a, b in zip(safe[compare_vector_key], variant[compare_vector_key])]
        first_diff_index = next((idx for idx, value in enumerate(diffs) if value > 0.0), None)
        comparison["vector_diff"] = {
            "vector_key": compare_vector_key,
            "vector_length": len(diffs),
            "max_abs_diff": max(diffs) if diffs else 0.0,
            "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else 0.0,
            "first_diff_index": first_diff_index,
        }
    summary["comparison"] = comparison
else:
    summary["comparison"] = {
        "comparable": False,
        "safe_primary_artifact": str(safe_artifact),
        "variant_primary_artifact": str(variant_artifact),
        "reason": "both retained-continuation artifacts were not produced within the bounded run",
    }

(output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
