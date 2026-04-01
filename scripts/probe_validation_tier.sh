#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [--tier 0|1|2] [options]"
  cat <<'USAGE'
Tier 0:
  compile only.

Tier 1:
  compile then trace (or reuse a valid existing trace artifact).

Tier 2:
  compile + trace (or reuse) + oracle compare.

Options:
  -t, --tier <0|1|2>
      validation tier to run (default: 0)
  --model <path>
      restricted model view path (default: /data/models/openai/gpt-oss-20b-full-attn-restricted-integration)
  --trace-json <path>
      trace artifact path (default: .live/restricted-cuda-prefill-trace.integration.json)
  --oracle-output <path>
      oracle diff artifact path (default: .live/restricted-prefill-trace-diff.integration.json)
  --original-model <path>
      original/oracle checkpoint path (default: /data/models/openai/gpt-oss-20b)
  --compare-mode <raw|runtime-emulated>
      comparison contract for Tier 2 oracle compare (default: raw)
  --prompt <text>
      prompt text used for trace capture
  --max-model-len <n>
      model max length for trace capture (default: 128)
  --gpu <idx>
      CUDA_VISIBLE_DEVICES index (default: 1)
  --seed-layers <csv>
      comma-separated layer indices for opt-in full-sequence seed capture during trace
  --local-replay-layer <n>
      opt-in same-input local replay layer for oracle compare
  --local-replay-path <coarse|attention|mlp>
      local replay path used with --local-replay-layer
  --no-reuse
      force trace rerun even if existing artifact is reusable
  --compare-only
      run oracle compare only (uses existing trace file, skips compile/trace)
  --warm-oracle
      opt-in: run the current compare request twice through one warm oracle session and emit a reuse-check artifact
  --inspect-trace-artifact
      dry-run: inspect whether the trace artifact is reusable for the current requested run without recapturing or comparing
  -h, --help
      show help

Common operator flows:
  1) Representative telemetry + localization capture:
     ./scripts/probe_validation_tier.sh \
       --tier 2 \
       --compare-mode runtime-emulated \
       --seed-layers 0,12,23

  2) Same-input local replay from an existing trace artifact:
     ./scripts/probe_validation_tier.sh \
       --compare-only \
       --compare-mode runtime-emulated \
       --local-replay-layer 12 \
       --local-replay-path attention

  3) Capture and replay the same representative layer in one run:
     ./scripts/probe_validation_tier.sh \
       --tier 2 \
       --compare-mode runtime-emulated \
       --seed-layers 12 \
       --local-replay-layer 12 \
       --local-replay-path coarse

  4) Warm-oracle reuse check for the current compare request:
     ./scripts/probe_validation_tier.sh \
       --compare-only \
       --compare-mode runtime-emulated \
       --warm-oracle

  5) Inspect an existing trace artifact for reuse compatibility:
     ./scripts/probe_validation_tier.sh \
       --inspect-trace-artifact \
       --trace-json .live/restricted-cuda-prefill-trace.integration.json \
       --compare-mode runtime-emulated \
       --local-replay-layer 12 \
       --local-replay-path attention
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_METADATA_SCHEMA_VERSION="probe_validation_artifact_meta.v1"
LEGACY_TRACE_SCHEMA_VERSION="legacy-unversioned"
WRAPPER_TRACE_CAPTURE_CONTRACT_VERSION="probe_validation_trace_capture_meta.v1"

TIER="0"
MODEL_PATH="/data/models/openai/gpt-oss-20b-full-attn-restricted-integration"
TRACE_JSON="$REPO_ROOT/.live/restricted-cuda-prefill-trace.integration.json"
ORACLE_OUTPUT="$REPO_ROOT/.live/restricted-prefill-trace-diff.integration.json"
ORIGINAL_MODEL="/data/models/openai/gpt-oss-20b"
COMPARE_MODE="raw"
PROMPT="Explain tensor parallelism in one short sentence."
MAX_MODEL_LEN="128"
GPU="1"
SEED_LAYERS=""
LOCAL_REPLAY_LAYER=""
LOCAL_REPLAY_PATH=""
REUSE_TRACE="1"
COMPARE_ONLY="0"
WARM_ORACLE="0"
INSPECT_TRACE_ARTIFACT="0"

usage_error() {
  echo "$1" >&2
  echo >&2
  usage >&2
  exit 1
}

csv_contains_value() {
  local csv="$1"
  local needle="$2"
  local item=""

  IFS=',' read -r -a items <<< "$csv"
  for item in "${items[@]}"; do
    if [[ "$(echo "$item" | tr -d '[:space:]')" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

parse_cli_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--tier)
        TIER="$2"
        shift 2
        ;;
      --model)
        MODEL_PATH="$2"
        shift 2
        ;;
      --trace-json)
        TRACE_JSON="$2"
        shift 2
        ;;
      --oracle-output)
        ORACLE_OUTPUT="$2"
        shift 2
        ;;
      --original-model)
        ORIGINAL_MODEL="$2"
        shift 2
        ;;
      --compare-mode)
        COMPARE_MODE="$2"
        shift 2
        ;;
      --prompt)
        PROMPT="$2"
        shift 2
        ;;
      --max-model-len)
        MAX_MODEL_LEN="$2"
        shift 2
        ;;
      --gpu)
        GPU="$2"
        shift 2
        ;;
      --seed-layers)
        SEED_LAYERS="$2"
        shift 2
        ;;
      --local-replay-layer)
        LOCAL_REPLAY_LAYER="$2"
        shift 2
        ;;
      --local-replay-path)
        LOCAL_REPLAY_PATH="$2"
        shift 2
        ;;
      --no-reuse)
        REUSE_TRACE="0"
        shift
        ;;
      --compare-only)
        COMPARE_ONLY="1"
        shift
        ;;
      --warm-oracle)
        WARM_ORACLE="1"
        shift
        ;;
      --inspect-trace-artifact)
        INSPECT_TRACE_ARTIFACT="1"
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
}

normalize_csv() {
  local csv="$1"

  if [[ -z "$csv" ]]; then
    echo ""
    return 0
  fi

  python3 - "$csv" <<'PY'
import sys

parts = [item.strip() for item in sys.argv[1].split(",")]
parts = [item for item in parts if item]
print(",".join(parts))
PY
}

have_python3() {
  command -v python3 >/dev/null 2>&1
}

trace_metadata_path() {
  echo "$1.meta.json"
}

oracle_metadata_path() {
  echo "$1.meta.json"
}

validate_cli_args() {
  if [[ "$COMPARE_ONLY" == "1" ]]; then
    TIER="2"
  fi

  if [[ "$INSPECT_TRACE_ARTIFACT" == "1" ]]; then
    TIER="2"
  fi

  if [[ "$TIER" != "0" && "$TIER" != "1" && "$TIER" != "2" ]]; then
    usage_error "Invalid tier: $TIER"
  fi

  if [[ "$WARM_ORACLE" == "1" && "$TIER" != "2" ]]; then
    usage_error "--warm-oracle is only available for Tier 2 / compare-only runs"
  fi
  if [[ "$INSPECT_TRACE_ARTIFACT" == "1" && "$WARM_ORACLE" == "1" ]]; then
    usage_error "--inspect-trace-artifact cannot be combined with --warm-oracle"
  fi

  if [[ -n "$LOCAL_REPLAY_LAYER" || -n "$LOCAL_REPLAY_PATH" ]]; then
    if [[ -z "$LOCAL_REPLAY_LAYER" || -z "$LOCAL_REPLAY_PATH" ]]; then
      usage_error "--local-replay-layer and --local-replay-path must be provided together"
    fi
    if [[ "$TIER" != "2" ]]; then
      usage_error "local replay is only available for Tier 2 / compare-only runs"
    fi
    if [[ "$COMPARE_ONLY" != "1" ]]; then
      if [[ -z "$SEED_LAYERS" ]]; then
        usage_error "local replay in the same run requires --seed-layers so the trace captures an exact runtime seed"
      fi
      if ! csv_contains_value "$SEED_LAYERS" "$LOCAL_REPLAY_LAYER"; then
        usage_error "--seed-layers must include --local-replay-layer when capture and replay happen in the same run"
      fi
    fi
  fi
}

write_trace_metadata() {
  local trace_path="$1"
  local metadata_path=""
  local normalized_seed_layers=""

  if ! have_python3; then
    echo "[tier] trace metadata sidecar skipped because python3 is unavailable; future reuse will fail closed" >&2
    return 0
  fi

  metadata_path="$(trace_metadata_path "$trace_path")"
  normalized_seed_layers="$(normalize_csv "$SEED_LAYERS")"
  python3 - "$trace_path" "$metadata_path" "$MODEL_PATH" "$PROMPT" "$MAX_MODEL_LEN" "$normalized_seed_layers" "$TIER" "$ARTIFACT_METADATA_SCHEMA_VERSION" "$LEGACY_TRACE_SCHEMA_VERSION" "$WRAPPER_TRACE_CAPTURE_CONTRACT_VERSION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    trace_path,
    metadata_path,
    model_path,
    prompt,
    max_model_len,
    seed_layers,
    tier,
    metadata_schema_version,
    legacy_trace_schema,
    wrapper_capture_contract_version,
) = sys.argv[1:11]
with open(trace_path, "r", encoding="utf-8") as handle:
    trace = json.load(handle)

prompt_token_ids = trace.get("prompt_token_ids") or []
trace_schema_version = trace.get("trace_schema_version") or trace.get("schema_version") or legacy_trace_schema
seed_layer_values = [item for item in seed_layers.split(",") if item]

metadata = {
    "metadata_schema_version": metadata_schema_version,
    "artifact_type": "restricted_prefill_trace",
    "trace_json": str(Path(trace_path)),
    "restricted_model_path": trace.get("restricted_model_path", model_path),
    "prompt": trace.get("prompt", prompt),
    "prompt_sha256": hashlib.sha256((trace.get("prompt", prompt)).encode("utf-8")).hexdigest(),
    "prompt_token_ids_sha256": hashlib.sha256(
        json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
    "max_model_len": int(max_model_len),
    "capture_tier": tier,
    "seed_layers": seed_layer_values,
    "trace_schema_version": trace_schema_version,
    "trace_capture_contract_version": wrapper_capture_contract_version,
    "trace_capture_origin": "wrapper-captured-current",
    "raw_trace_schema_status": (
        "raw-trace-embedded-versioned" if trace.get("trace_schema_version") or trace.get("schema_version") else "raw-trace-legacy-unversioned"
    ),
    "wrapper_capture_generation": "current-wrapper-sidecar",
}

Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
with open(metadata_path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
PY
}

write_oracle_metadata() {
  local oracle_output="$1"
  local trace_path="$2"
  local metadata_path=""

  if ! have_python3; then
    return 0
  fi

  metadata_path="$(oracle_metadata_path "$oracle_output")"
  python3 - "$oracle_output" "$trace_path" "$metadata_path" "$TIER" "$COMPARE_MODE" "$LOCAL_REPLAY_LAYER" "$LOCAL_REPLAY_PATH" "$WARM_ORACLE" "$ARTIFACT_METADATA_SCHEMA_VERSION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

oracle_output, trace_path, metadata_path, tier, compare_mode, local_replay_layer, local_replay_path, warm_oracle, metadata_schema_version = sys.argv[1:10]
with open(oracle_output, "r", encoding="utf-8") as handle:
    report = json.load(handle)

prompt = report.get("prompt", "")
prompt_token_ids = report.get("prompt_token_ids") or []
metadata = {
    "metadata_schema_version": metadata_schema_version,
    "artifact_type": "restricted_oracle_report",
    "oracle_output": str(Path(oracle_output)),
    "trace_json": str(Path(trace_path)),
    "restricted_model_path": report.get("restricted_model_path"),
    "original_model_path": report.get("original_model_path"),
    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "prompt_token_ids_sha256": hashlib.sha256(
        json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
    "requested_tier": tier,
    "compare_mode": compare_mode,
    "local_replay_layer": None if not local_replay_layer else int(local_replay_layer),
    "local_replay_path": local_replay_path or None,
    "warm_oracle": warm_oracle == "1",
    "oracle_schema_version": report.get("tool_schema_version", "missing"),
}

Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
with open(metadata_path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
PY
}

can_reuse_trace() {
  local trace_path="$1"
  local metadata_path=""

  if [[ ! -f "$trace_path" ]]; then
    echo "[tier] trace reuse rejected: trace artifact is missing: $trace_path" >&2
    return 1
  fi
  if ! have_python3; then
    echo "[tier] trace reuse rejected: python3 is unavailable, so trace provenance cannot be validated" >&2
    return 1
  fi
  metadata_path="$(trace_metadata_path "$trace_path")"
  python3 - "$trace_path" "$metadata_path" "$MODEL_PATH" "$PROMPT" "$MAX_MODEL_LEN" "$(normalize_csv "$SEED_LAYERS")" "$LOCAL_REPLAY_LAYER" "$ARTIFACT_METADATA_SCHEMA_VERSION" "$LEGACY_TRACE_SCHEMA_VERSION" "$WRAPPER_TRACE_CAPTURE_CONTRACT_VERSION" <<'PY'
import hashlib
import json
import sys

(
    trace_path,
    metadata_path,
    expected_model,
    expected_prompt,
    expected_max_model_len,
    expected_seed_layers,
    expected_local_replay_layer,
    expected_metadata_schema_version,
    legacy_trace_schema,
    expected_wrapper_capture_contract_version,
) = sys.argv[1:11]

with open(trace_path, "r", encoding="utf-8") as handle:
    trace = json.load(handle)

reasons = []
try:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
except FileNotFoundError:
    reasons.append(f"missing trace metadata sidecar: {metadata_path}")
    metadata = {}

trace_schema_version = trace.get("trace_schema_version") or trace.get("schema_version") or legacy_trace_schema
prompt = trace.get("prompt", "")
prompt_token_ids = trace.get("prompt_token_ids") or []
prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
prompt_token_ids_sha256 = hashlib.sha256(
    json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
).hexdigest()
capture_contract_version = metadata.get("trace_capture_contract_version")
capture_origin = metadata.get("trace_capture_origin")
is_wrapper_captured = capture_contract_version == expected_wrapper_capture_contract_version and capture_origin == "wrapper-captured-current"
is_legacy_metadata = not capture_contract_version and not capture_origin

if trace.get("restricted_model_path") != expected_model:
    reasons.append("restricted model path mismatch")
if prompt != expected_prompt:
    reasons.append("prompt mismatch")

if metadata.get("metadata_schema_version") != expected_metadata_schema_version:
    reasons.append(
        f"trace metadata schema mismatch (have {metadata.get('metadata_schema_version')!r}, expected {expected_metadata_schema_version!r})"
    )
if metadata.get("artifact_type") != "restricted_prefill_trace":
    reasons.append("trace metadata artifact_type mismatch")
if metadata.get("restricted_model_path") != expected_model:
    reasons.append("trace metadata restricted model path mismatch")
if metadata.get("prompt_sha256") != prompt_sha256:
    reasons.append("trace metadata prompt identity mismatch")
if metadata.get("prompt_token_ids_sha256") != prompt_token_ids_sha256:
    reasons.append("trace metadata token identity mismatch")
if str(metadata.get("max_model_len")) != expected_max_model_len:
    reasons.append(
        f"trace metadata max-model-len mismatch (have {metadata.get('max_model_len')!r}, expected {expected_max_model_len!r})"
    )
if metadata.get("trace_schema_version") != trace_schema_version:
    reasons.append(
        f"trace schema marker mismatch (metadata {metadata.get('trace_schema_version')!r}, trace {trace_schema_version!r})"
    )
if is_wrapper_captured:
    raw_trace_schema_status = metadata.get("raw_trace_schema_status")
    expected_raw_trace_schema_status = (
        "raw-trace-embedded-versioned" if trace.get("trace_schema_version") or trace.get("schema_version") else "raw-trace-legacy-unversioned"
    )
    if raw_trace_schema_status != expected_raw_trace_schema_status:
        reasons.append(
            f"raw trace schema status mismatch (have {raw_trace_schema_status!r}, expected {expected_raw_trace_schema_status!r})"
        )
    if metadata.get("wrapper_capture_generation") != "current-wrapper-sidecar":
        reasons.append("wrapper capture generation mismatch")
elif is_legacy_metadata:
    pass
else:
    reasons.append(
        "trace capture contract metadata is incomplete or incompatible; treat artifact as legacy and recapture with the current wrapper if needed"
    )

metadata_seed_layers = [str(item) for item in (metadata.get("seed_layers") or [])]
expected_seed_layer_values = [item for item in expected_seed_layers.split(",") if item]
if expected_seed_layer_values and metadata_seed_layers != expected_seed_layer_values:
    reasons.append(
        f"trace metadata seed-layers mismatch (have {','.join(metadata_seed_layers) or '<none>'}, expected {','.join(expected_seed_layer_values)})"
    )
if expected_local_replay_layer and expected_local_replay_layer not in metadata_seed_layers:
    reasons.append(
        f"trace metadata does not include requested local replay layer {expected_local_replay_layer} in seed-layers"
    )

if reasons:
    print("[tier] trace reuse rejected:", file=sys.stderr)
    for reason in reasons:
        print(f"[tier]   - {reason}", file=sys.stderr)
    raise SystemExit(1)

if is_wrapper_captured:
    print(
        "[tier] trace reuse accepted: wrapper-captured metadata"
        f" contract={capture_contract_version}"
        f" raw_trace_schema_status={metadata.get('raw_trace_schema_status')}",
        file=sys.stderr,
    )
else:
    print(
        "[tier] trace reuse accepted with legacy metadata:"
        " wrapper-owned capture contract not present;"
        f" raw_trace_schema_version={trace_schema_version}",
        file=sys.stderr,
    )
PY
}

inspect_trace_artifact() {
  local trace_path="$1"
  local metadata_path=""

  if [[ ! -f "$trace_path" ]]; then
    echo "[tier] trace inspection rejected: trace artifact is missing: $trace_path" >&2
    return 1
  fi
  if ! have_python3; then
    echo "[tier] trace inspection rejected: python3 is unavailable, so trace provenance cannot be inspected" >&2
    return 1
  fi
  metadata_path="$(trace_metadata_path "$trace_path")"
  python3 - "$trace_path" "$metadata_path" "$MODEL_PATH" "$PROMPT" "$MAX_MODEL_LEN" "$(normalize_csv "$SEED_LAYERS")" "$LOCAL_REPLAY_LAYER" "$COMPARE_MODE" "$ARTIFACT_METADATA_SCHEMA_VERSION" "$LEGACY_TRACE_SCHEMA_VERSION" "$WRAPPER_TRACE_CAPTURE_CONTRACT_VERSION" <<'PY'
import hashlib
import json
import sys

(
    trace_path,
    metadata_path,
    expected_model,
    expected_prompt,
    expected_max_model_len,
    expected_seed_layers,
    expected_local_replay_layer,
    requested_compare_mode,
    expected_metadata_schema_version,
    legacy_trace_schema,
    expected_wrapper_capture_contract_version,
) = sys.argv[1:12]

with open(trace_path, "r", encoding="utf-8") as handle:
    trace = json.load(handle)

reasons = []
try:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata_present = True
except FileNotFoundError:
    reasons.append(f"missing trace metadata sidecar: {metadata_path}")
    metadata = {}
    metadata_present = False

trace_schema_version = trace.get("trace_schema_version") or trace.get("schema_version") or legacy_trace_schema
prompt = trace.get("prompt", "")
prompt_token_ids = trace.get("prompt_token_ids") or []
prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
prompt_token_ids_sha256 = hashlib.sha256(
    json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
).hexdigest()

capture_contract_version = metadata.get("trace_capture_contract_version")
capture_origin = metadata.get("trace_capture_origin")
is_wrapper_captured = capture_contract_version == expected_wrapper_capture_contract_version and capture_origin == "wrapper-captured-current"
is_legacy_metadata = metadata_present and not capture_contract_version and not capture_origin

if trace.get("restricted_model_path") != expected_model:
    reasons.append("restricted model path mismatch")
if prompt != expected_prompt:
    reasons.append("prompt mismatch")

if metadata.get("metadata_schema_version") != expected_metadata_schema_version:
    reasons.append(
        f"trace metadata schema mismatch (have {metadata.get('metadata_schema_version')!r}, expected {expected_metadata_schema_version!r})"
    )
if metadata.get("artifact_type") != "restricted_prefill_trace":
    reasons.append("trace metadata artifact_type mismatch")
if metadata.get("restricted_model_path") != expected_model:
    reasons.append("trace metadata restricted model path mismatch")
if metadata.get("prompt_sha256") != prompt_sha256:
    reasons.append("trace metadata prompt identity mismatch")
if metadata.get("prompt_token_ids_sha256") != prompt_token_ids_sha256:
    reasons.append("trace metadata token identity mismatch")
if str(metadata.get("max_model_len")) != expected_max_model_len:
    reasons.append(
        f"trace metadata max-model-len mismatch (have {metadata.get('max_model_len')!r}, expected {expected_max_model_len!r})"
    )
if metadata.get("trace_schema_version") != trace_schema_version:
    reasons.append(
        f"trace schema marker mismatch (metadata {metadata.get('trace_schema_version')!r}, trace {trace_schema_version!r})"
    )

if is_wrapper_captured:
    raw_trace_schema_status = metadata.get("raw_trace_schema_status")
    expected_raw_trace_schema_status = (
        "raw-trace-embedded-versioned" if trace.get("trace_schema_version") or trace.get("schema_version") else "raw-trace-legacy-unversioned"
    )
    if raw_trace_schema_status != expected_raw_trace_schema_status:
        reasons.append(
            f"raw trace schema status mismatch (have {raw_trace_schema_status!r}, expected {expected_raw_trace_schema_status!r})"
        )
    if metadata.get("wrapper_capture_generation") != "current-wrapper-sidecar":
        reasons.append("wrapper capture generation mismatch")
elif is_legacy_metadata:
    pass
else:
    reasons.append(
        "trace capture contract metadata is incomplete or incompatible; treat artifact as legacy and recapture with the current wrapper if needed"
    )

metadata_seed_layers = [str(item) for item in (metadata.get("seed_layers") or [])]
expected_seed_layer_values = [item for item in expected_seed_layers.split(",") if item]
if expected_seed_layer_values and metadata_seed_layers != expected_seed_layer_values:
    reasons.append(
        f"trace metadata seed-layers mismatch (have {','.join(metadata_seed_layers) or '<none>'}, expected {','.join(expected_seed_layer_values)})"
    )
if expected_local_replay_layer and expected_local_replay_layer not in metadata_seed_layers:
    reasons.append(
        f"trace metadata does not include requested local replay layer {expected_local_replay_layer} in seed-layers"
    )

if not metadata_present:
    classification = "incompatible"
elif is_wrapper_captured:
    classification = "current-wrapper-captured"
elif is_legacy_metadata:
    classification = "legacy"
else:
    classification = "incompatible"

report = {
    "trace_json": trace_path,
    "trace_metadata": metadata_path,
    "requested_run": {
        "restricted_model_path": expected_model,
        "prompt": expected_prompt,
        "prompt_sha256": hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
        "max_model_len": int(expected_max_model_len),
        "compare_mode": requested_compare_mode,
        "seed_layers": expected_seed_layer_values,
        "local_replay_layer": None if not expected_local_replay_layer else int(expected_local_replay_layer),
    },
    "artifact_classification": classification,
    "reusable_for_requested_run": not reasons,
    "wrapper_contract": {
        "metadata_schema_version": metadata.get("metadata_schema_version"),
        "trace_capture_contract_version": metadata.get("trace_capture_contract_version"),
        "trace_capture_origin": metadata.get("trace_capture_origin"),
        "wrapper_capture_generation": metadata.get("wrapper_capture_generation"),
        "raw_trace_schema_status": metadata.get("raw_trace_schema_status"),
        "trace_schema_version": metadata.get("trace_schema_version", trace_schema_version),
    },
    "provenance_checked": {
        "restricted_model_path": metadata.get("restricted_model_path"),
        "prompt_sha256": metadata.get("prompt_sha256"),
        "prompt_token_ids_sha256": metadata.get("prompt_token_ids_sha256"),
        "max_model_len": metadata.get("max_model_len"),
        "capture_tier": metadata.get("capture_tier"),
        "seed_layers": metadata.get("seed_layers"),
        "artifact_type": metadata.get("artifact_type"),
    },
    "reasons": reasons,
    "note": None,
}

if classification == "legacy":
    report["note"] = (
        "legacy artifact: wrapper-owned capture contract is missing; artifact may still be reusable if older provenance fields match, but it is not upgraded to the current wrapper-captured contract"
    )
elif classification == "current-wrapper-captured":
    report["note"] = "current wrapper-owned capture contract present"
else:
    report["note"] = "artifact is not reusable for the requested run"

print(json.dumps(report, indent=2))
if reasons:
    raise SystemExit(1)
PY
}

run_builds() {
  echo "[tier] compiling workspace tools"
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-engine)
  (cd "$REPO_ROOT" && cargo build --release --features cuda -p gpt-oss-bench --bin restricted_prefill_trace)
}

resolve_oracle_python() {
  local candidate=""

  if [[ -n "${GPT_OSS_ORACLE_PYTHON:-}" ]]; then
    candidate="$GPT_OSS_ORACLE_PYTHON"
    if "$candidate" -c 'import torch' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
    echo "GPT_OSS_ORACLE_PYTHON does not have a working torch install: $candidate" >&2
    exit 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    candidate="python3"
    if "$candidate" -c 'import torch' >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  fi

  candidate="/data/models/.venv-awq/bin/python"
  if [[ -x "$candidate" ]] && "$candidate" -c 'import torch' >/dev/null 2>&1; then
    echo "$candidate"
    return 0
  fi

  echo "no python interpreter with torch is available for oracle comparison; set GPT_OSS_ORACLE_PYTHON if needed" >&2
  exit 1
}

print_run_summary() {
  echo "[tier] tier=$TIER compare_mode=$COMPARE_MODE compare_only=$COMPARE_ONLY warm_oracle=$WARM_ORACLE"
  echo "[tier] model=$MODEL_PATH trace_json=$TRACE_JSON oracle_output=$ORACLE_OUTPUT"
  if [[ -n "$SEED_LAYERS" ]]; then
    echo "[tier] seed_layers=$SEED_LAYERS"
  fi
  if [[ -n "$LOCAL_REPLAY_LAYER" ]]; then
    echo "[tier] local_replay=layer:$LOCAL_REPLAY_LAYER path:$LOCAL_REPLAY_PATH"
  fi
}

reuse_check_output_path() {
  local oracle_output="$1"
  local stem="$oracle_output"
  if [[ "$oracle_output" == *.json ]]; then
    stem="${oracle_output%.json}"
  fi
  echo "${stem}.warm-reuse-check.json"
}

run_trace() {
  local trace_output="$1"
  echo "[tier] running restricted prefill trace"
  if [[ -n "$SEED_LAYERS" ]]; then
    (cd "$REPO_ROOT" && \
      CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 GPT_OSS_TRACE_SEED_LAYERS="$SEED_LAYERS" \
      target/release/restricted_prefill_trace \
      --model "$MODEL_PATH" \
      --prompt "$PROMPT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization 0.75 \
      --output "$trace_output" \
      --log-level info)
  else
    (cd "$REPO_ROOT" && \
      CUDA_VISIBLE_DEVICES="$GPU" GPT_OSS_DISABLE_CUDA_GRAPHS=1 \
      target/release/restricted_prefill_trace \
      --model "$MODEL_PATH" \
      --prompt "$PROMPT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization 0.75 \
      --output "$trace_output" \
      --log-level info)
  fi
  write_trace_metadata "$trace_output"
}

run_oracle_compare() {
  local trace_output="$1"
  local oracle_output="$2"
  local python_bin=""

  python_bin="$(resolve_oracle_python)"

  echo "[tier] running oracle compare"
  "$python_bin" "$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py" \
    --cuda-trace-json "$trace_output" \
    --original-model "$ORIGINAL_MODEL" \
    --output "$oracle_output" \
    --compare-mode "$COMPARE_MODE" \
    ${LOCAL_REPLAY_LAYER:+--local-replay-layer "$LOCAL_REPLAY_LAYER"} \
    ${LOCAL_REPLAY_PATH:+--local-replay-path "$LOCAL_REPLAY_PATH"} \
    --device cpu

  write_oracle_metadata "$oracle_output" "$trace_output"

  if have_python3; then
  python3 - "$oracle_output" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    report = json.load(handle)

print("first_divergence_stage=", report.get('first_divergence_stage'))
print("compare_mode=", report.get('compare_mode'))
print("diff_count=", len(report.get('stage_diffs', [])))
PY
  fi
}

build_oracle_compare_request() {
  local python_bin="$1"
  local trace_output="$2"
  local oracle_output="$3"

  "$python_bin" - "$trace_output" "$oracle_output" "$COMPARE_MODE" "$LOCAL_REPLAY_LAYER" "$LOCAL_REPLAY_PATH" <<'PY'
import json
import sys

trace_output, oracle_output, compare_mode, local_replay_layer, local_replay_path = sys.argv[1:6]
request = {
    "op": "compare",
    "cuda_trace_json": trace_output,
    "output": oracle_output,
    "compare_mode": compare_mode,
}
if local_replay_layer:
    request["local_replay_layer"] = int(local_replay_layer)
if local_replay_path:
    request["local_replay_path"] = local_replay_path
print(json.dumps(request))
PY
}

compare_json_reports() {
  local python_bin="$1"
  local lhs="$2"
  local rhs="$3"

  "$python_bin" - "$lhs" "$rhs" <<'PY'
import json
import sys

lhs_path, rhs_path = sys.argv[1:3]
with open(lhs_path, "r", encoding="utf-8") as lhs_handle:
    lhs = json.load(lhs_handle)
with open(rhs_path, "r", encoding="utf-8") as rhs_handle:
    rhs = json.load(rhs_handle)
print("1" if lhs == rhs else "0")
PY
}

run_warm_oracle_compare() {
  local trace_output="$1"
  local oracle_output="$2"
  local python_bin=""
  local oracle_tool=""
  local reuse_output=""
  local request_primary=""
  local request_reuse=""
  local response_primary=""
  local response_reuse=""
  local response_shutdown=""
  local reports_match=""

  python_bin="$(resolve_oracle_python)"
  oracle_tool="$REPO_ROOT/crates/gpt-oss-bench/tools/restricted_oracle_prefill_trace.py"
  reuse_output="$(reuse_check_output_path "$oracle_output")"
  request_primary="$(build_oracle_compare_request "$python_bin" "$trace_output" "$oracle_output")"
  request_reuse="$(build_oracle_compare_request "$python_bin" "$trace_output" "$reuse_output")"

  echo "[tier] running warm oracle compare"
  echo "[tier] warm_oracle primary_output=$oracle_output reuse_output=$reuse_output"

  coproc WARM_ORACLE_LISTENER {
    "$python_bin" "$oracle_tool" \
      --listen \
      --original-model "$ORIGINAL_MODEL" \
      --device cpu
  }

  printf '%s\n' "$request_primary" >&"${WARM_ORACLE_LISTENER[1]}"
  IFS= read -r response_primary <&"${WARM_ORACLE_LISTENER[0]}"

  printf '%s\n' "$request_reuse" >&"${WARM_ORACLE_LISTENER[1]}"
  IFS= read -r response_reuse <&"${WARM_ORACLE_LISTENER[0]}"

  printf '%s\n' '{"op":"shutdown"}' >&"${WARM_ORACLE_LISTENER[1]}"
  exec {WARM_ORACLE_LISTENER[1]}>&-
  IFS= read -r response_shutdown <&"${WARM_ORACLE_LISTENER[0]}"
  wait "$WARM_ORACLE_LISTENER_PID"

  if have_python3; then
    python3 - "$response_primary" "$response_reuse" "$response_shutdown" <<'PY'
import json
import sys

primary = json.loads(sys.argv[1])
reuse = json.loads(sys.argv[2])
shutdown = json.loads(sys.argv[3])

print(
    "[tier] warm_oracle first_request"
    f" session_reused={primary.get('session_reused')}"
    f" session_load_count={primary.get('session_load_count')}"
    f" output={primary.get('output')}"
)
print(
    "[tier] warm_oracle second_request"
    f" session_reused={reuse.get('session_reused')}"
    f" session_load_count={reuse.get('session_load_count')}"
    f" output={reuse.get('output')}"
)
print(f"[tier] warm_oracle shutdown_ok={shutdown.get('shutdown')}")
PY
  fi

  reports_match="$(compare_json_reports "$python_bin" "$oracle_output" "$reuse_output" | tr -d '[:space:]')"
  if [[ "$reports_match" != "1" ]]; then
    echo "[tier] warm_oracle repeated request reports diverged between $oracle_output and $reuse_output" >&2
    exit 1
  fi
  echo "[tier] warm_oracle repeated_request_reports_match=1"
  write_oracle_metadata "$oracle_output" "$trace_output"
  write_oracle_metadata "$reuse_output" "$trace_output"

  if have_python3; then
  python3 - "$oracle_output" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as handle:
    report = json.load(handle)

print("first_divergence_stage=", report.get('first_divergence_stage'))
print("compare_mode=", report.get('compare_mode'))
print("diff_count=", len(report.get('stage_diffs', [])))
PY
  fi
}

main() {
  parse_cli_args "$@"
  validate_cli_args

  if [[ "$INSPECT_TRACE_ARTIFACT" == "1" ]]; then
    inspect_trace_artifact "$TRACE_JSON"
    return 0
  fi

  if [[ "$COMPARE_ONLY" == "1" ]]; then
    if [[ ! -f "$TRACE_JSON" ]]; then
      echo "compare-only requires existing trace artifact: $TRACE_JSON" >&2
      exit 1
    fi
    if ! can_reuse_trace "$TRACE_JSON"; then
      echo "[tier] compare-only requires a trace artifact with matching provenance; rerun capture or use compatible metadata" >&2
      exit 1
    fi
    echo "[tier] reusing existing trace artifact with matching metadata: $TRACE_JSON"
    print_run_summary
    if [[ "$WARM_ORACLE" == "1" ]]; then
      run_warm_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
    else
      run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
    fi
    return 0
  fi

  print_run_summary

  if [[ "$TIER" == "0" || "$TIER" == "1" || "$TIER" == "2" ]]; then
    run_builds
  fi

  if [[ "$TIER" == "1" || "$TIER" == "2" ]]; then
    if [[ "$REUSE_TRACE" == "1" ]]; then
      if can_reuse_trace "$TRACE_JSON"; then
        echo "[tier] reusing existing trace artifact with matching metadata: $TRACE_JSON"
      else
        run_trace "$TRACE_JSON"
      fi
    else
      run_trace "$TRACE_JSON"
    fi
  fi

  if [[ "$TIER" == "2" ]]; then
    if [[ "$WARM_ORACLE" == "1" ]]; then
      run_warm_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
    else
      run_oracle_compare "$TRACE_JSON" "$ORACLE_OUTPUT"
    fi
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
