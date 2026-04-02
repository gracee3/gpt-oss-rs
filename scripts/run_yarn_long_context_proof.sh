#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEFAULT_MODEL="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
DEFAULT_SAFE_TREE="/home/emmy/openai/gpt-oss-rs"
DEFAULT_VARIANT_TREE="/home/emmy/openai/worktrees/runtime-forward"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/.live/yarn-long-context-proof"
DEFAULT_TIMEOUT="1800"
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
  --max-model-len <n>        requested max-model-len passed to restricted_logit_diff
  --prompt-file <path>       existing prompt file to use
  --setup-only               stop after prompt/setup verification and command planning
  -h, --help                 show this help text
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
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
GPU_ID="${DEFAULT_GPU}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
PROMPT_FILE=""
SETUP_ONLY=0

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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

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

cat >"${SAFE_CMD_FILE}" <<EOF
cd ${SAFE_TREE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo run --release -p gpt-oss-bench --bin restricted_logit_diff -- --model ${MODEL_PATH} --prompt "\$(cat "${PROMPT_FILE}")" --max-model-len ${MAX_MODEL_LEN} --output ${OUTPUT_DIR}/safe/restricted-logit-diff.json --log-level info
EOF

cat >"${VARIANT_CMD_FILE}" <<EOF
cd ${VARIANT_TREE}
PATH="/data/models/.venv-awq/bin:\$PATH" CUDA_VISIBLE_DEVICES=${GPU_ID} GPT_OSS_DISABLE_CUDA_GRAPHS=1 cargo run --release -p gpt-oss-bench --bin restricted_logit_diff -- --model ${MODEL_PATH} --prompt "\$(cat "${PROMPT_FILE}")" --max-model-len ${MAX_MODEL_LEN} --output ${OUTPUT_DIR}/variant/restricted-logit-diff.json --log-level info
EOF

chmod +x "${SAFE_CMD_FILE}" "${VARIANT_CMD_FILE}"

cat >"${PLAN_DIR}/setup_summary.json" <<EOF
{
  "gpu_id": "${GPU_ID}",
  "safe_tree": "${SAFE_TREE}",
  "variant_tree": "${VARIANT_TREE}",
  "model_path": "${MODEL_PATH}",
  "prompt_file": "${PROMPT_FILE}",
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "max_model_len": ${MAX_MODEL_LEN},
  "setup_only": ${SETUP_ONLY}
}
EOF

echo "prepared long-context proof setup under ${OUTPUT_DIR}"
echo "prompt token summary: ${PROMPT_DIR}/prompt_token_count.json"
echo "safe command plan: ${SAFE_CMD_FILE}"
echo "variant command plan: ${VARIANT_CMD_FILE}"

if [[ "${SETUP_ONLY}" -eq 1 ]]; then
  exit 0
fi

echo "run execution is not wired yet in this scaffold"
