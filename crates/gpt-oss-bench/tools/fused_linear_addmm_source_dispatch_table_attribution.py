#!/usr/bin/env python3
"""Read-only PyTorch dispatch table attribution for fused-linear/addmm.

This helper is oracle/probe evidence only. It records installed Torch wheel
metadata, ATen dispatch table registrations, and CPU-only profiler events for
the sampled BF16 o-proj linear/addmm shape. It does not patch or rebuild
PyTorch, load model tensors, run CUDA, select a backend, or run consumer
revalidation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


SAMPLED_LAYERS = [6, 10, 13, 16, 18, 21]
DISPATCH_OPS = ["aten::linear", "aten::addmm", "aten::mm", "aten::matmul"]
DISPATCH_KEYS = [
    "CPU",
    "MkldnnCPU",
    "SparseCPU",
    "AutogradCPU",
    "CompositeImplicitAutograd",
    "CompositeExplicitAutograd",
    "BackendSelect",
    "Python",
    "Meta",
    "QuantizedCPU",
    "NestedTensorCPU",
]
ENV_KEYS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ONEDNN_VERBOSE",
    "DNNL_VERBOSE",
    "MKL_VERBOSE",
    "ATEN_CPU_CAPABILITY",
    "TORCH_SHOW_CPP_STACKTRACES",
    "CUDA_VISIBLE_DEVICES",
]
BACKEND_NAME_MARKERS = ["mkldnn", "onednn", "dnnl", "mkl"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def safe_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - depends on local torch build
        return {"error": str(exc)}


def import_torch() -> tuple[Any | None, str | None]:
    try:
        import torch  # type: ignore

        return torch, None
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, str(exc)


def torch_config_show(torch: Any) -> str | None:
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            torch.__config__.show()
        return buffer.getvalue()
    except Exception:
        return None


def environment(torch: Any | None, import_error: str | None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "venv": os.environ.get("VIRTUAL_ENV"),
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "torch_import_error": import_error,
        "cuda_used": False,
    }
    if torch is None:
        return data
    data.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "torch_git_version": getattr(torch.version, "git_version", None),
            "torch_file": getattr(torch, "__file__", None),
            "torch_config_show": torch_config_show(torch),
            "torch_num_threads": safe_call(torch.get_num_threads),
            "torch_num_interop_threads": safe_call(torch.get_num_interop_threads),
            "torch_backends_mkldnn_enabled": getattr(torch.backends.mkldnn, "enabled", None),
            "torch_cuda_available": safe_call(torch.cuda.is_available),
            "cuda_used": False,
            "oracle_device": "cpu",
        }
    )
    return data


def dispatch_schema(torch: Any, op: str) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    get_schema = getattr(torch._C, "_dispatch_get_schema", None)
    if callable(get_schema):
        try:
            result["dispatch_get_schema"] = str(get_schema(op))
            result["available"] = True
        except Exception as exc:
            result["dispatch_get_schema_error"] = str(exc)
    finder = getattr(torch._C, "_dispatch_find_schema_or_throw", None)
    if callable(finder):
        try:
            handle = finder(op, "")
            result["dispatch_find_schema_or_throw"] = str(handle.schema())
            result["available"] = True
        except Exception as exc:
            result["dispatch_find_schema_or_throw_error"] = str(exc)
    return result


def dispatch_table(torch: Any | None, op: str) -> dict[str, Any]:
    if torch is None:
        return {"available": False, "reason": "torch_import_failed"}
    table_fn = getattr(torch._C, "_dispatch_dump_table", None)
    if not callable(table_fn):
        return {"available": False, "reason": "_dispatch_dump_table_unavailable"}
    try:
        table = table_fn(op)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    registrations = {key: f"{key}:" in table for key in DISPATCH_KEYS}
    kernel_checks: dict[str, Any] = {}
    has_kernel = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if callable(has_kernel):
        for key in DISPATCH_KEYS:
            try:
                kernel_checks[key] = bool(has_kernel(op, key))
            except Exception as exc:
                kernel_checks[key] = {"error": str(exc)}
    return {
        "available": True,
        "op": op,
        "table": table,
        "line_count": len(table.splitlines()),
        "registrations": registrations,
        "has_kernel_for_dispatch_key": kernel_checks,
        "schema": dispatch_schema(torch, op),
    }


def collect_dispatch_tables(torch: Any | None) -> dict[str, Any]:
    return {op: dispatch_table(torch, op) for op in DISPATCH_OPS}


def summarize_dispatch_tables(tables: dict[str, Any]) -> dict[str, Any]:
    visible_by_op: dict[str, dict[str, bool]] = {}
    for op, table in tables.items():
        registrations = table.get("registrations", {}) if isinstance(table, dict) else {}
        visible_by_op[op] = {
            key: bool(registrations.get(key, False)) for key in DISPATCH_KEYS
        }
    return {
        "aten_linear_visible": bool(tables.get("aten::linear", {}).get("available")),
        "aten_addmm_visible": bool(tables.get("aten::addmm", {}).get("available")),
        "aten_mm_visible": bool(tables.get("aten::mm", {}).get("available")),
        "aten_matmul_visible": bool(tables.get("aten::matmul", {}).get("available")),
        "cpu_registration_visible": any(
            op_summary.get("CPU", False) for op_summary in visible_by_op.values()
        ),
        "mkldnn_cpu_registration_visible": any(
            op_summary.get("MkldnnCPU", False) for op_summary in visible_by_op.values()
        ),
        "visible_registrations_by_op": visible_by_op,
        "source_dispatch_unresolved": True,
        "note": "Dispatch tables expose registration labels, not the exact lower-level BF16 CPU kernel path.",
    }


def assert_cpu_tensors(tensors: dict[str, Any]) -> None:
    for name, tensor in tensors.items():
        if getattr(tensor, "is_cuda", False):
            raise RuntimeError(f"{name} unexpectedly allocated on CUDA")
        device = str(getattr(tensor, "device", ""))
        if device and device != "cpu":
            raise RuntimeError(f"{name} unexpectedly allocated on {device}")


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def call_variants(torch: Any, x: Any, w: Any, b: Any) -> list[dict[str, Any]]:
    variants: list[tuple[str, Callable[[], Any]]] = [
        ("torch.nn.functional.linear", lambda: torch.nn.functional.linear(x, w, b)),
        ("torch._C._nn.linear", lambda: torch._C._nn.linear(x, w, b)),
        ("torch.addmm", lambda: torch.addmm(b, x, w.t())),
        ("input_at_weight_t_plus_bias", lambda: x @ w.t() + b),
        ("torch.einsum_plus_bias", lambda: torch.einsum("bk,hk->bh", x, w) + b),
    ]
    results = []
    for name, fn in variants:
        try:
            output = fn()
            assert_cpu_tensors({f"{name}_output": output})
            results.append(
                {
                    "name": name,
                    "available": True,
                    "output": tensor_metadata(output),
                }
            )
        except Exception as exc:
            results.append({"name": name, "available": False, "error": str(exc)})
    return results


def profile_once(torch: Any, label: str) -> dict[str, Any]:
    started = time.time()
    with torch.inference_mode():
        x = torch.zeros((1, 4096), dtype=torch.bfloat16, device="cpu")
        w = torch.zeros((2880, 4096), dtype=torch.bfloat16, device="cpu")
        b = torch.zeros((2880,), dtype=torch.bfloat16, device="cpu")
        assert_cpu_tensors({"input": x, "weight": w, "bias": b})
        activities = [torch.profiler.ProfilerActivity.CPU]
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            variant_results = call_variants(torch, x, w, b)

    events = []
    for event in prof.key_averages():
        key = getattr(event, "key", None)
        if not key:
            continue
        events.append(
            {
                "key": key,
                "count": getattr(event, "count", None),
                "cpu_time_total": getattr(event, "cpu_time_total", None),
                "self_cpu_time_total": getattr(event, "self_cpu_time_total", None),
                "input_shapes": getattr(event, "input_shapes", None),
            }
        )
    event_keys = [event["key"] for event in events]
    backend_names = sorted(
        {
            key
            for key in event_keys
            if any(marker in key.lower() for marker in BACKEND_NAME_MARKERS)
        }
    )
    return {
        "label": label,
        "available": True,
        "elapsed_seconds": time.time() - started,
        "shape": {
            "input": [1, 4096],
            "weight": [2880, 4096],
            "bias": [2880],
        },
        "tensor_metadata": {
            "input": tensor_metadata(x),
            "weight": tensor_metadata(w),
            "bias": tensor_metadata(b),
        },
        "variant_results": variant_results,
        "events": events,
        "event_keys": event_keys,
        "aten_linear_observed": "aten::linear" in event_keys,
        "aten_addmm_observed": "aten::addmm" in event_keys,
        "aten_mm_observed": "aten::mm" in event_keys,
        "aten_matmul_observed": "aten::matmul" in event_keys,
        "profiler_observed_mkldnn_or_onednn_name": bool(backend_names),
        "profiler_deeper_backend_visible": bool(backend_names),
        "backend_names": backend_names,
        "profiler_backend_inconclusive": True,
    }


@contextlib.contextmanager
def torch_toggle(torch: Any, mkldnn_enabled: bool | None, num_threads: int | None):
    original_mkldnn = getattr(torch.backends.mkldnn, "enabled", None)
    original_threads = safe_call(torch.get_num_threads)
    try:
        if mkldnn_enabled is not None and original_mkldnn is not None:
            torch.backends.mkldnn.enabled = mkldnn_enabled
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        yield
    finally:
        if original_mkldnn is not None:
            with contextlib.suppress(Exception):
                torch.backends.mkldnn.enabled = original_mkldnn
        if isinstance(original_threads, int):
            with contextlib.suppress(Exception):
                torch.set_num_threads(original_threads)


def run_profiler_toggles(torch: Any | None) -> list[dict[str, Any]]:
    if torch is None:
        return [
            {
                "label": "torch_import_failed",
                "available": False,
                "profiler_backend_inconclusive": True,
            }
        ]
    original_threads = safe_call(torch.get_num_threads)
    original_mkldnn = getattr(torch.backends.mkldnn, "enabled", None)
    toggles = [
        ("default_environment", None, None),
        ("mkldnn_disabled", False, None),
        ("mkldnn_enabled", True, None),
        ("single_thread", None, 1),
        ("default_thread_count", None, original_threads if isinstance(original_threads, int) else None),
    ]
    results = []
    for label, mkldnn_enabled, num_threads in toggles:
        with torch_toggle(torch, mkldnn_enabled, num_threads):
            try:
                results.append(
                    {
                        "label": label,
                        "requested_mkldnn_enabled": mkldnn_enabled,
                        "requested_num_threads": num_threads,
                        "actual_mkldnn_enabled": getattr(torch.backends.mkldnn, "enabled", None),
                        "actual_num_threads": safe_call(torch.get_num_threads),
                        "profiler": profile_once(torch, label),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "label": label,
                        "requested_mkldnn_enabled": mkldnn_enabled,
                        "requested_num_threads": num_threads,
                        "actual_mkldnn_enabled": getattr(torch.backends.mkldnn, "enabled", None),
                        "actual_num_threads": safe_call(torch.get_num_threads),
                        "profiler": {
                            "available": False,
                            "profiler_backend_inconclusive": True,
                            "error": str(exc),
                        },
                    }
                )
    if original_mkldnn is not None:
        with contextlib.suppress(Exception):
            torch.backends.mkldnn.enabled = original_mkldnn
    if isinstance(original_threads, int):
        with contextlib.suppress(Exception):
            torch.set_num_threads(original_threads)
    return results


def summarize_profiler(toggle_results: list[dict[str, Any]]) -> dict[str, Any]:
    profilers = [item.get("profiler", {}) for item in toggle_results]
    available = [prof for prof in profilers if prof.get("available")]
    backend_names = sorted(
        {
            name
            for prof in available
            for name in prof.get("backend_names", [])
            if isinstance(name, str)
        }
    )
    return {
        "profiler_runs": len(toggle_results),
        "profiler_runs_available": len(available),
        "aten_linear_observed": any(prof.get("aten_linear_observed", False) for prof in available),
        "aten_addmm_observed": any(prof.get("aten_addmm_observed", False) for prof in available),
        "aten_mm_observed": any(prof.get("aten_mm_observed", False) for prof in available),
        "aten_matmul_observed": any(prof.get("aten_matmul_observed", False) for prof in available),
        "deeper_backend_observed": bool(backend_names),
        "backend_names": backend_names,
        "classifications": {
            "profiler_observed_aten_linear": any(
                prof.get("aten_linear_observed", False) for prof in available
            ),
            "profiler_observed_aten_addmm": any(
                prof.get("aten_addmm_observed", False) for prof in available
            ),
            "profiler_observed_mkldnn_or_onednn_name": bool(backend_names),
            "profiler_deeper_backend_visible": bool(backend_names),
            "profiler_backend_inconclusive": True,
        },
    }


def source_fact_summary(cpu_status: dict[str, Any], avx2_status: dict[str, Any]) -> dict[str, Any]:
    avx2 = cpu_status.get("avx2_contract_consistency", {})
    return {
        "cpu_producer_attribution_classification": cpu_status.get("classification"),
        "api_semantics_confirmed": bool(cpu_status.get("api_results_summary")),
        "cpu_attribution_avx2_contract_consistency": avx2,
        "avx2_contract_classification": avx2_status.get("classification"),
        "avx2_contract_consistency": bool(avx2.get("all_layers_consistent", False)),
        "source_level_dispatch_proven_before_this_probe": bool(
            cpu_status.get("source_level_dispatch_proven", False)
        ),
        "backend_identity_proven_before_this_probe": bool(
            cpu_status.get("backend_identity_proven", False)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status-output",
        "--output",
        default="/tmp/fused_linear_addmm_source_dispatch_table_status.json",
    )
    parser.add_argument(
        "--cpu-producer-attribution-status",
        default="/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
    )
    parser.add_argument(
        "--avx2-contract-status",
        default="/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json",
    )
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
        "--like-helper-prototype-status",
        default="/tmp/fused_linear_addmm_like_helper_prototype_status.json",
    )
    args = parser.parse_args()

    source_paths = {
        "cpu_producer_attribution": Path(args.cpu_producer_attribution_status),
        "avx2_contract": Path(args.avx2_contract_status),
        "producer_api_13_16_10": Path(args.producer_api_13_16_10_status),
        "producer_api_18_21": Path(args.producer_api_18_21_status),
        "backend_candidate_comparator": Path(args.backend_candidate_comparator_status),
        "like_helper_prototype": Path(args.like_helper_prototype_status),
    }
    missing_required = [str(path) for path in source_paths.values() if not path.exists()]

    torch, import_error = import_torch()
    env = environment(torch, import_error)
    tables = collect_dispatch_tables(torch)
    table_summary = summarize_dispatch_tables(tables)
    toggle_results = run_profiler_toggles(torch)
    profiler_summary = summarize_profiler(toggle_results)

    cpu_status = (
        load_json(source_paths["cpu_producer_attribution"])
        if source_paths["cpu_producer_attribution"].exists()
        else {}
    )
    avx2_status = load_json(source_paths["avx2_contract"]) if source_paths["avx2_contract"].exists() else {}
    facts = source_fact_summary(cpu_status, avx2_status)

    dispatch_tables_collected = bool(
        tables.get("aten::linear", {}).get("available")
        and tables.get("aten::addmm", {}).get("available")
    )
    profiler_collected = profiler_summary["profiler_runs_available"] > 0
    if missing_required:
        classification = "fused_linear_addmm_source_dispatch_table_blocked"
    elif dispatch_tables_collected and profiler_collected:
        classification = "fused_linear_addmm_source_dispatch_table_recorded"
    elif dispatch_tables_collected or profiler_collected:
        classification = "fused_linear_addmm_source_dispatch_table_inconclusive"
    else:
        classification = "fused_linear_addmm_source_dispatch_table_execution_failed"

    batch = {
        "classification": classification,
        "validation_only": True,
        "oracle_probe_only": True,
        "read_only": True,
        "pytorch_patched": False,
        "pytorch_rebuilt": False,
        "runtime_behavior_changed": False,
        "production_routing_changed": False,
        "cuda_kernels_changed": False,
        "backend_selected": False,
        "implementation_authorized": False,
        "consumer_revalidation_authorized": False,
        "operator": "attention_o_proj",
        "sampled_layers": SAMPLED_LAYERS,
        "source_statuses": {name: str(path) for name, path in source_paths.items()},
        "missing_required_statuses": missing_required,
        "environment": env,
        "dispatch_tables": tables,
        "dispatch_table_summary": table_summary,
        "profiler_summary": profiler_summary,
        "toggle_results": toggle_results,
        "preserved_cpu_attribution_facts": facts,
        "avx2_contract_consistency": facts["avx2_contract_consistency"],
        "source_level_dispatch_proven": False,
        "backend_identity_proven": False,
        "source_instrumentation_recommended_next": True,
        "next_bounded_step": "Review dispatch table attribution before deciding whether source instrumentation is needed.",
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
