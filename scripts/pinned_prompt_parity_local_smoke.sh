#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [options]"
  cat <<'USAGE'
Run serialized local pinned-prompt-parity smoke captures for the checked-in tiny fixture set.

This script intentionally runs captures serially on one GPU. Do not parallelize
runner and worker capture on the same 24 GB-class GPU; the exploratory PPP
probe hit CUDA OOM when both paths loaded concurrently.

Options:
  --model <path>
      restricted/local model path
      default: /data/models/openai/gpt-oss-20b-full-attn-restricted-integration
  --fixture-dir <path>
      manifest fixture directory
      default: <repo>/crates/gpt-oss-bench/fixtures/pinned-prompts
  --output-dir <path>
      output directory for artifacts and reports
      default: /tmp/pinned-prompt-parity-local-smoke
  --gpu <idx>
      CUDA_VISIBLE_DEVICES index
      default: 1
  --max-model-len <n>
      model max length passed to PPP capture
      default: 128
  --gpu-memory-utilization <f>
      GPU memory utilization passed to PPP capture
      default: 0.75
  --log-level <level>
      PPP binary log level
      default: info
  --debug
      build and run target/debug instead of target/release
  --skip-build
      reuse an existing PPP binary
  -h, --help
      show help
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
FIXTURE_DIR="$REPO_ROOT/crates/gpt-oss-bench/fixtures/pinned-prompts"
OUTPUT_DIR="/tmp/pinned-prompt-parity-local-smoke"
GPU="1"
MAX_MODEL_LEN="128"
GPU_MEMORY_UTILIZATION="0.75"
LOG_LEVEL="info"
PROFILE="release"
SKIP_BUILD="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --fixture-dir)
      FIXTURE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --debug)
      PROFILE="debug"
      shift
      ;;
    --skip-build)
      SKIP_BUILD="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$FIXTURE_DIR" ]]; then
  echo "fixture directory does not exist: $FIXTURE_DIR" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "model path does not exist: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/artifacts" "$OUTPUT_DIR/reports"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "[ppp] building pinned_prompt_parity ($PROFILE)"
  if [[ "$PROFILE" == "release" ]]; then
    (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-bench --bin pinned_prompt_parity)
  else
    (cd "$REPO_ROOT" && cargo build --features cuda -p gpt-oss-bench --bin pinned_prompt_parity)
  fi
fi

BINARY_PATH="$REPO_ROOT/target/$PROFILE/pinned_prompt_parity"
if [[ ! -x "$BINARY_PATH" ]]; then
  echo "PPP binary is missing or not executable: $BINARY_PATH" >&2
  exit 1
fi

shopt -s nullglob
manifests=("$FIXTURE_DIR"/*.json)
shopt -u nullglob

if [[ ${#manifests[@]} -eq 0 ]]; then
  echo "no manifest fixtures found in $FIXTURE_DIR" >&2
  exit 1
fi

printf '%s\n' "${manifests[@]}" | sort | while IFS= read -r manifest; do
  fixture_name="$(basename "$manifest" .json)"
  runner_artifact="$OUTPUT_DIR/artifacts/${fixture_name}.runner.json"
  worker_artifact="$OUTPUT_DIR/artifacts/${fixture_name}.worker.json"
  self_report="$OUTPUT_DIR/reports/${fixture_name}.runner-self.compare.json"
  local_report="$OUTPUT_DIR/reports/${fixture_name}.runner-vs-worker.compare.json"

  echo "[ppp] fixture=$fixture_name capture=runner"
  (
    cd "$REPO_ROOT" && \
    CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    "$BINARY_PATH" capture \
      --manifest "$manifest" \
      --model "$MODEL_PATH" \
      --output "$runner_artifact" \
      --capture-source runner \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --log-level "$LOG_LEVEL"
  )

  echo "[ppp] fixture=$fixture_name capture=worker"
  (
    cd "$REPO_ROOT" && \
    CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
    "$BINARY_PATH" capture \
      --manifest "$manifest" \
      --model "$MODEL_PATH" \
      --output "$worker_artifact" \
      --capture-source worker \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --log-level "$LOG_LEVEL"
  )

  echo "[ppp] fixture=$fixture_name compare=runner-self"
  (
    cd "$REPO_ROOT" && \
    "$BINARY_PATH" compare \
      --left "$runner_artifact" \
      --right "$runner_artifact" \
      --output "$self_report" \
      --log-level "$LOG_LEVEL"
  )

  echo "[ppp] fixture=$fixture_name compare=runner-vs-worker"
  (
    cd "$REPO_ROOT" && \
    "$BINARY_PATH" compare \
      --left "$runner_artifact" \
      --right "$worker_artifact" \
      --output "$local_report" \
      --log-level "$LOG_LEVEL"
  )

  echo "[ppp] fixture=$fixture_name complete"
  echo "  runner artifact: $runner_artifact"
  echo "  worker artifact: $worker_artifact"
  echo "  self report:     $self_report"
  echo "  local report:    $local_report"
done
