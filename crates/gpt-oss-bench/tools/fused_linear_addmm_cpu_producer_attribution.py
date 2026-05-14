#!/usr/bin/env python3
"""CPU-first producer attribution for fused-linear/addmm o-proj seams.

This helper is oracle/probe evidence only. It consumes existing producer/API
trace statuses, records CPU Torch environment/profiler attribution, and emits a
normalized batch status plus per-layer statuses. It does not load raw model
tensors, run CUDA, select a backend, or perform consumer revalidation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


LAYERS: dict[int, dict[str, Any]] = {
    6: {"role": "historical_context", "focus_lane": 22},
    10: {"role": "pairwise_clear_control", "focus_lane": 915},
    13: {"role": "blocked_family", "focus_lane": 151},
    16: {"role": "blocked_family", "focus_lane": 2666},
    18: {"role": "blocked_family", "focus_lane": 63},
    21: {"role": "raw_qk_solved_oproj_blocked", "focus_lane": 2807},
}

OPERATOR_KEYS = [
    "module_attn_out",
    "F_linear",
    "_C_nn_linear",
    "fused_addmm",
    "weight_at_input_plus_bias",
    "input_at_weight_t_plus_bias",
    "matmul",
    "einsum",
    "F_linear_bias_none_plus_bias",
]

OPERATOR_ALIASES = {
    "module_attn_out": "module_attn_out",
    "torch_nn_functional_linear": "F_linear",
    "torch_C_nn_linear": "_C_nn_linear",
    "torch_addmm_fused_bias": "fused_addmm",
    "weight_at_input_plus_bias": "weight_at_input_plus_bias",
    "input_at_weight_t_plus_bias": "input_at_weight_t_plus_bias",
    "torch_matmul_weight_input_plus_bias": "matmul",
    "torch_einsum_hk_k_to_h_plus_bias": "einsum",
    "flinear_no_bias_then_add_bias": "F_linear_bias_none_plus_bias",
}

FULL_CLEAR_KEYS = {"module_attn_out", "F_linear", "_C_nn_linear", "fused_addmm"}
NEGATIVE_CONTROL_KEYS = {
    "weight_at_input_plus_bias",
    "input_at_weight_t_plus_bias",
    "matmul",
    "einsum",
    "F_linear_bias_none_plus_bias",
}

ENV_KEYS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "ONEDNN_VERBOSE",
    "DNNL_VERBOSE",
    "MKL_VERBOSE",
    "ATEN_CPU_CAPABILITY",
    "CUDA_VISIBLE_DEVICES",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compact_operator_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {
            "available": False,
            "clears_full_vector": False,
            "metrics": None,
            "focus_lane": None,
            "reason_unavailable": "missing_from_source_status",
        }
    metrics = result.get("metrics", {})
    return {
        "operator": result.get("operator"),
        "available": bool(result.get("available", False)),
        "clears_full_vector": bool(result.get("clears_full_vector", False)),
        "clears_focus_lane": bool(result.get("clears_focus_lane", False)),
        "metrics": metrics.get("metrics", metrics),
        "first_mismatch": metrics.get("first_mismatch"),
        "worst_mismatch": metrics.get("worst_mismatch"),
        "focus_lane": result.get("focus_lane"),
        "dtype": result.get("dtype"),
        "device": result.get("device"),
        "shape": result.get("shape"),
        "reason_unavailable": result.get("reason_unavailable"),
    }


def metric_mismatches(result: dict[str, Any] | None) -> int | None:
    if not result:
        return None
    metrics = result.get("metrics", {})
    inner = metrics.get("metrics", metrics)
    value = inner.get("mismatches")
    return int(value) if isinstance(value, int) else None


def tensor_meta(status: dict[str, Any], name: str) -> dict[str, Any] | None:
    meta = status.get("tensor_metadata", {}).get(name)
    if not isinstance(meta, dict):
        return None
    return {
        "present": meta.get("present"),
        "shape": meta.get("shape"),
        "dtype": meta.get("dtype"),
        "device": meta.get("device"),
        "stride": meta.get("stride"),
        "contiguous": meta.get("contiguous"),
        "storage_offset": meta.get("storage_offset"),
        "summary": meta.get("summary"),
    }


def sensitivity_summary(status: dict[str, Any]) -> dict[str, Any]:
    sensitivity = status.get("sensitivity", {}) or {}
    results = status.get("sensitivity_results", {}) or {}
    mkldnn = sensitivity.get("mkldnn", {})
    threads = sensitivity.get("threads", {})
    layout = sensitivity.get("layout", {})
    fused_bias = sensitivity.get("fused_bias", {})
    return {
        "mkldnn_sensitive": bool(mkldnn.get("sensitive", status.get("interpretation", {}).get("mkldnn_sensitive", False))),
        "thread_sensitive": bool(threads.get("sensitive", status.get("interpretation", {}).get("thread_sensitive", False))),
        "layout_sensitive": bool(layout.get("sensitive", status.get("interpretation", {}).get("layout_sensitive", False))),
        "fused_bias_sensitive": bool(fused_bias.get("sensitive", status.get("interpretation", {}).get("fused_bias_sensitive", False))),
        "mkldnn_enabled_clears": clears_from_result(mkldnn.get("enabled") or results.get("mkldnn_enabled")),
        "mkldnn_disabled_clears": clears_from_result(mkldnn.get("disabled") or results.get("mkldnn_disabled")),
        "single_thread_clears": clears_from_result(threads.get("single_thread") or results.get("single_thread")),
        "default_thread_clears": clears_from_result(threads.get("default") or results.get("default_thread")),
    }


def clears_from_result(result: dict[str, Any] | None) -> bool | None:
    if not isinstance(result, dict):
        return None
    value = result.get("clears_full_vector")
    return bool(value) if isinstance(value, bool) else None


def import_torch() -> tuple[Any | None, str | None]:
    try:
        import torch  # type: ignore

        return torch, None
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, str(exc)


def torch_environment(torch: Any | None, import_error: str | None) -> dict[str, Any]:
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "venv": os.environ.get("VIRTUAL_ENV"),
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "torch_import_error": import_error,
    }
    if torch is None:
        return env

    config = None
    with contextlib.suppress(Exception):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            torch.__config__.show()
        config = buffer.getvalue()
    env.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "torch_git_version": getattr(torch.version, "git_version", None),
            "torch_file": getattr(torch, "__file__", None),
            "torch_num_threads": safe_call(torch.get_num_threads),
            "torch_num_interop_threads": safe_call(torch.get_num_interop_threads),
            "torch_backends_mkldnn_enabled": getattr(torch.backends.mkldnn, "enabled", None),
            "torch_cuda_available": safe_call(torch.cuda.is_available),
            "cuda_used": False,
            "oracle_device": "cpu",
            "torch_config_show": config,
        }
    )
    return env


def safe_call(fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": str(exc)}


def profiler_probe(torch: Any | None) -> dict[str, Any]:
    if torch is None:
        return {
            "available": False,
            "source_level_dispatch_proven": False,
            "reason": "torch_import_failed",
        }
    try:
        with torch.inference_mode():
            x = torch.zeros((1, 4096), dtype=torch.bfloat16, device="cpu")
            w = torch.zeros((2880, 4096), dtype=torch.bfloat16, device="cpu")
            b = torch.zeros((2880,), dtype=torch.bfloat16, device="cpu")
            activities = [torch.profiler.ProfilerActivity.CPU]
            with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
                torch.nn.functional.linear(x, w, b)
                torch.addmm(b, x, w.t())
        events = []
        seen = set()
        for event in prof.key_averages():
            key = getattr(event, "key", None)
            if key and key not in seen:
                seen.add(key)
                events.append(
                    {
                        "key": key,
                        "cpu_time_total": getattr(event, "cpu_time_total", None),
                        "input_shapes": getattr(event, "input_shapes", None),
                    }
                )
        return {
            "available": True,
            "device": "cpu",
            "shape": {"input": [1, 4096], "weight": [2880, 4096], "bias": [2880]},
            "observed_ops": events,
            "aten_linear_observed": any(event["key"] == "aten::linear" for event in events),
            "aten_addmm_observed": any(event["key"] == "aten::addmm" for event in events),
            "aten_mm_observed": any(event["key"] == "aten::mm" for event in events),
            "source_level_dispatch_proven": False,
            "attribution": "profiler_reports_aten_ops_not_low_level_cpu_kernel_source",
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "source_level_dispatch_proven": False,
            "reason": str(exc),
        }


def layer_status(layer: int, status: dict[str, Any], avx2_contract: dict[str, Any]) -> dict[str, Any]:
    info = LAYERS[layer]
    op_results = normalize_operator_results(status.get("operator_results", {}) or {})
    compact_ops = {key: compact_operator_result(op_results.get(key)) for key in OPERATOR_KEYS}
    full_clear_ops = [
        key
        for key, result in compact_ops.items()
        if result.get("available") and result.get("clears_full_vector")
    ]
    negative_mismatch_counts = {
        key: compact_ops[key]["metrics"].get("mismatches")
        for key in NEGATIVE_CONTROL_KEYS
        if isinstance(compact_ops.get(key, {}).get("metrics"), dict)
    }
    fused_family_clear = all(
        compact_ops[key].get("clears_full_vector", False)
        for key in ["F_linear", "_C_nn_linear", "fused_addmm"]
    )
    negative_controls_mismatch = all(
        (metric_mismatches(op_results.get(key)) or 0) > 0 for key in NEGATIVE_CONTROL_KEYS
    )
    contract = avx2_contract.get("avx2_contract", {})
    avx2_shape = contract.get("matrix_shape", {})
    avx2_dtype = contract.get("dtype_contract", {})
    avx2_contract_consistent = bool(
        avx2_shape.get("K") == 4096
        and avx2_dtype.get("input") == "BF16 weighted-V"
        and avx2_dtype.get("weight") == "BF16 o_proj weight row through weight.T view"
        and avx2_dtype.get("bias") == "BF16 o_proj bias"
        and avx2_dtype.get("output") == "BF16"
        and fused_family_clear
        and negative_controls_mismatch
    )
    output_meta = tensor_meta(status, "o_proj_output") or {}
    focus_lane = int(info["focus_lane"])
    return {
        "classification": f"layer{layer}_fused_linear_addmm_cpu_producer_attribution_recorded",
        "validation_only": True,
        "oracle_probe_only": True,
        "layer_index": layer,
        "layer_role": info["role"],
        "focus_lane": focus_lane,
        "official_focus_value": focus_value(output_meta, focus_lane, op_results),
        "oracle_device": "cpu",
        "source_status": f"/tmp/layer{layer}_attention_oproj_api_probe_status.json",
        "source_classification": status.get("classification"),
        "tensor_metadata": {
            "input": tensor_meta(status, "weighted_v"),
            "weight": tensor_meta(status, "o_proj_weight"),
            "bias": tensor_meta(status, "o_proj_bias"),
            "output": output_meta,
        },
        "api_results": compact_ops,
        "full_vector_clear_operators": full_clear_ops,
        "negative_control_mismatch_counts": negative_mismatch_counts,
        "fused_linear_addmm_family_clears": fused_family_clear,
        "matmul_einsum_unfused_bias_reproduce_mismatch_class": negative_controls_mismatch,
        "sensitivity": sensitivity_summary(status),
        "avx2_contract_consistent": avx2_contract_consistent,
        "avx2_contract_consistency_note": "consistent_with_available_contract_and_api_matrix_but_not_source_level_dispatch_proof"
        if avx2_contract_consistent
        else "insufficient_or_inconsistent_with_available_contract",
        "source_level_dispatch_proven": False,
        "backend_identity_proven": False,
        "backend_selected": False,
        "implementation_authorized": False,
        "consumer_revalidation_authorized": False,
        "runtime_behavior_changed": False,
        "production_routing_changed": False,
        "cuda_kernels_changed": False,
        "output_emitted": False,
        "ladder_continued": False,
        "correction_metadata_applied": False,
        "tolerance_pass": False,
        "final_logit_claim": False,
        "all_layer_claim": False,
        "server_claim": False,
        "context_length_claim": False,
    }


def focus_value(output_meta: dict[str, Any], focus_lane: int, op_results: dict[str, Any]) -> Any:
    for key in ("focus_lane_value", f"lane{focus_lane}_value", "output_lane_value"):
        if key in output_meta:
            return output_meta[key]
    for key in ("F_linear", "module_attn_out", "fused_addmm"):
        focus = op_results.get(key, {}).get("focus_lane", {})
        if isinstance(focus, dict) and focus.get("lane") == focus_lane:
            return focus.get("official")
    return None


def normalize_operator_results(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    normalized: dict[str, Any] = {}
    if not isinstance(raw, list):
        return normalized
    for item in raw:
        if not isinstance(item, dict):
            continue
        source_name = item.get("operator")
        key = OPERATOR_ALIASES.get(source_name, source_name)
        if isinstance(key, str) and key in OPERATOR_KEYS:
            normalized[key] = item
    return normalized


def summary_from_layers(layers: list[dict[str, Any]]) -> dict[str, Any]:
    api_summary: dict[str, Any] = {}
    for key in OPERATOR_KEYS:
        api_summary[key] = {
            "layers_available": [
                layer["layer_index"]
                for layer in layers
                if layer["api_results"][key].get("available")
            ],
            "layers_full_vector_clear": [
                layer["layer_index"]
                for layer in layers
                if layer["api_results"][key].get("clears_full_vector")
            ],
            "mismatch_counts": {
                str(layer["layer_index"]): (
                    layer["api_results"][key].get("metrics") or {}
                ).get("mismatches")
                for layer in layers
            },
        }
    return api_summary


def environment_matrix(layers: list[dict[str, Any]], current_env: dict[str, Any], profiler: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_environment": current_env,
        "profiler_probe": profiler,
        "layer_sensitivity_summary": {
            str(layer["layer_index"]): layer["sensitivity"] for layer in layers
        },
        "toggles_tested_from_source_statuses": [
            "default_environment",
            "mkldnn_enabled",
            "mkldnn_disabled",
            "single_thread",
            "default_thread_count",
            "layout_perturbation_guards",
            "fused_bias_guard",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status-output",
        "--output",
        default="/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
    )
    parser.add_argument("--layers", default="6,10,13,16,18,21")
    parser.add_argument(
        "--producer-api-13-16-10-status",
        default="/tmp/o_proj_producer_api_probes_13_16_10_status.json",
    )
    parser.add_argument(
        "--producer-api-18-21-status",
        default="/tmp/o_proj_producer_api_probes_18_21_status.json",
    )
    parser.add_argument(
        "--backend-candidate-comparator-status",
        default="/tmp/fused_linear_addmm_backend_discriminator_candidate_status.json",
    )
    parser.add_argument(
        "--avx2-contract-status",
        default="/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json",
    )
    args = parser.parse_args()

    requested_layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if requested_layers != list(LAYERS):
        raise SystemExit(f"expected layers {list(LAYERS)}, got {requested_layers}")

    source_paths = {
        "producer_api_13_16_10": Path(args.producer_api_13_16_10_status),
        "producer_api_18_21": Path(args.producer_api_18_21_status),
        "backend_candidate_comparator": Path(args.backend_candidate_comparator_status),
        "avx2_contract": Path(args.avx2_contract_status),
    }
    missing_required = [str(path) for path in source_paths.values() if not path.exists()]
    source_statuses = {name: str(path) for name, path in source_paths.items()}

    avx2_status = load_json(source_paths["avx2_contract"]) if source_paths["avx2_contract"].exists() else {}
    layers_evaluated: list[dict[str, Any]] = []
    layers_blocked: list[dict[str, Any]] = []
    if not missing_required:
        for layer in requested_layers:
            path = Path(f"/tmp/layer{layer}_attention_oproj_api_probe_status.json")
            if not path.exists():
                layers_blocked.append({"layer_index": layer, "reason": f"missing {path}"})
                continue
            status = load_json(path)
            row = layer_status(layer, status, avx2_status)
            layer_status_path = Path(f"/tmp/layer{layer}_fused_linear_addmm_cpu_producer_attribution_status.json")
            write_json(layer_status_path, row)
            layers_evaluated.append(row)

    torch, torch_import_error = import_torch()
    current_env = torch_environment(torch, torch_import_error)
    profiler = profiler_probe(torch)
    all_rows_emitted = len(layers_evaluated) == len(requested_layers) and not layers_blocked
    if missing_required:
        classification = "fused_linear_addmm_cpu_producer_attribution_blocked_by_missing_trace"
    elif all_rows_emitted:
        classification = "fused_linear_addmm_cpu_producer_attribution_recorded"
    elif layers_evaluated:
        classification = "fused_linear_addmm_cpu_producer_attribution_partial"
    else:
        classification = "fused_linear_addmm_cpu_producer_attribution_execution_failed"

    avx2_consistent_layers = [
        layer["layer_index"] for layer in layers_evaluated if layer["avx2_contract_consistent"]
    ]
    batch = {
        "classification": classification,
        "validation_only": True,
        "oracle_probe_only": True,
        "runtime_behavior_changed": False,
        "production_routing_changed": False,
        "cuda_kernels_changed": False,
        "backend_selected": False,
        "implementation_authorized": False,
        "consumer_revalidation_authorized": False,
        "torch_patched": False,
        "torch_rebuilt": False,
        "operator": "attention_o_proj",
        "reference": {
            "api": "module/F.linear/_C/addmm",
            "dtype": "torch.bfloat16",
            "fused_bias": True,
            "layout_sensitive": True,
            "full_vector_required": True,
        },
        "source_statuses": source_statuses,
        "missing_required_statuses": missing_required,
        "layers_requested": requested_layers,
        "layers_evaluated": layers_evaluated,
        "layers_blocked": layers_blocked,
        "api_paths_tested": [
            "module_attn_out(weighted_v)",
            "torch.nn.functional.linear(weighted_v, weight, bias)",
            "torch._C._nn.linear(weighted_v, weight, bias)",
            "torch.addmm(bias, input[1xK], weight.T)",
            "torch.ops.aten.addmm.default/profiler attribution",
            "explicit matmul + bias",
            "explicit einsum + bias",
            "F.linear(..., bias=None) + bias",
        ],
        "api_results_summary": summary_from_layers(layers_evaluated),
        "environment_matrix": environment_matrix(layers_evaluated, current_env, profiler),
        "avx2_contract_consistency": {
            "contract_status": str(source_paths["avx2_contract"]),
            "contract_classification": avx2_status.get("classification"),
            "layers_consistent": avx2_consistent_layers,
            "all_layers_consistent": len(avx2_consistent_layers) == len(requested_layers),
            "source_level_dispatch_proven": False,
            "note": "API behavior and shape/dtype/bias facts are consistent with the extracted AVX2 contract, but this probe does not prove source-level dispatch.",
        },
        "source_level_dispatch_proven": False,
        "backend_identity_proven": False,
        "next_bounded_step": "Review CPU producer attribution before any Rust fused-addmm helper design or implementation.",
        "output_emitted": False,
        "ladder_continued": False,
        "correction_metadata_applied": False,
        "tolerance_pass": False,
        "final_logit_claim": False,
        "all_layer_claim": False,
        "server_claim": False,
        "context_length_claim": False,
    }
    write_json(Path(args.status_output), batch)
    print(json.dumps(batch, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
