#!/usr/bin/env python3
"""Installed Torch wheel dispatch attribution for fused addmm o-proj.

This Stage 1 source-attribution probe inspects the currently installed Torch
wheel for CPU BF16 addmm/linear/mm/matmul dispatch evidence. It is CPU-only
oracle evidence: it does not clone or build PyTorch, load the model, use CUDA,
or authorize runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any


DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_torch_wheel_dispatch_attribution_status.json")
DEFAULT_RESEARCH_DIR = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-wheel-dispatch")
REPO_ROOT = Path(__file__).resolve().parents[3]

RECENT_STATUS_PATHS = [
    Path("/tmp/fused_linear_addmm_cpu_producer_attribution_status.json"),
    Path("/tmp/fused_linear_addmm_addmm_boundary_localization_status.json"),
    Path("/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json"),
    Path("/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json"),
]

KNOWN_PYTHON_PATHS = [
    Path("/home/emmy/openai/gpt-oss/.venv/bin/python"),
    Path("/data/models/.venv-awq/bin/python"),
    Path("/home/emmy/openai/.venvs/torch-wheel-attribution/bin/python"),
]

OPS = {
    "addmm": "aten::addmm",
    "linear": "aten::linear",
    "mm": "aten::mm",
    "matmul": "aten::matmul",
}

GUARD_FALSE_FLAGS = {
    "runtime_behavior_changed": False,
    "production_routing_changed": False,
    "cuda_kernels_changed": False,
    "backend_selected": False,
    "implementation_authorized": False,
    "consumer_revalidation_authorized": False,
    "output_emitted": False,
    "ladder_continued": False,
    "correction_metadata_applied": False,
    "tolerance_pass": False,
    "final_logit_claim": False,
    "all_layer_claim": False,
    "server_claim": False,
    "context_length_claim": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Installed Torch wheel dispatch attribution for fused addmm o-proj."
    )
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-report", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def base_status(classification: str) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "source_attribution_probe": True,
        "torch_wheel_dispatch_attribution": True,
        "oracle_device": "cpu",
        "cuda_available": None,
        "cuda_used": False,
        "pytorch_clone_performed": False,
        "pytorch_build_performed": False,
        "pytorch_source_patched": False,
        **GUARD_FALSE_FLAGS,
    }


def as_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.name.startswith("python") or str(path).endswith("/bin/python"):
        return path
    return None


def collect_python_paths_from_json(value: Any, paths: list[Path]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {
                "selected_python_executable",
                "python_executable",
                "sys_executable",
                "python",
            }:
                path = as_path(nested)
                if path is not None:
                    paths.append(path)
            collect_python_paths_from_json(nested, paths)
    elif isinstance(value, list):
        for item in value:
            collect_python_paths_from_json(item, paths)


def discover_candidate_pythons() -> list[Path]:
    paths: list[Path] = [Path(sys.executable)]
    for status_path in RECENT_STATUS_PATHS:
        status = load_json(status_path)
        if status is not None:
            collect_python_paths_from_json(status, paths)
    paths.extend(KNOWN_PYTHON_PATHS)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def check_python_candidate(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_executable": str(path),
        "exists": path.exists(),
        "torch_importable": False,
        "selected": False,
        "error": None,
    }
    if not path.exists():
        report["error"] = "python_executable_missing"
        return report

    code = r"""
import json
import os
import sys
try:
    import torch
    result = {
        "torch_importable": True,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torch_version": getattr(torch, "__version__", None),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_import_path": getattr(torch, "__file__", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
except Exception as exc:
    result = {
        "torch_importable": False,
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "error": repr(exc),
    }
print(json.dumps(result, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(path), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report["returncode"] = completed.returncode
        report["stderr"] = completed.stderr[-2000:]
        if completed.stdout.strip():
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            report.update(payload)
        if completed.returncode != 0 and report.get("error") is None:
            report["error"] = completed.stderr[-2000:] or "candidate_check_failed"
    except Exception as exc:  # noqa: BLE001 - preserve candidate failure in status
        report["error"] = repr(exc)
    return report


def maybe_create_empty_research_venv(candidate_reports: list[dict[str, Any]]) -> dict[str, Any]:
    venv_python = Path("/home/emmy/openai/.venvs/torch-wheel-attribution/bin/python")
    result = {
        "venv_path": "/home/emmy/openai/.venvs/torch-wheel-attribution",
        "created": False,
        "reason": "not_needed",
        "python_executable": str(venv_python),
    }
    if any(report.get("torch_importable") for report in candidate_reports):
        return result
    if venv_python.exists():
        result["reason"] = "existing_empty_or_non_torch_venv_checked"
        return result
    try:
        venv.create(str(venv_python.parents[1]), with_pip=True)
        result["created"] = True
        result["reason"] = "created_empty_research_venv_without_installing_torch"
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"failed_to_create_empty_research_venv: {exc!r}"
    return result


def classify_op_dump(raw_dump: str | None, available: bool) -> dict[str, Any]:
    text = raw_dump or ""
    lower = text.lower()
    cpu_registration = "cpu:" in lower
    mkldnn_registration = "mkldnn" in lower or "onednn" in lower
    autograd_cpu = "autogradcpu" in text
    composite_implicit = "CompositeImplicitAutograd" in text
    composite_explicit = "CompositeExplicitAutograd" in text

    if not available:
        inferred = "unavailable"
    elif cpu_registration and mkldnn_registration:
        inferred = "multiple_possible"
    elif cpu_registration:
        inferred = "cpu_native_registration"
    elif mkldnn_registration:
        inferred = "mkldnn_registration_present"
    elif composite_implicit or composite_explicit:
        inferred = "composite_only"
    else:
        inferred = "inconclusive"

    source_locations = []
    for line in text.splitlines():
        if "registered at" in line or "aten/src/" in line or "torch/csrc/" in line:
            source_locations.append(line.strip())
        if len(source_locations) >= 40:
            break

    return {
        "cpu_kernel_registration_appears": cpu_registration,
        "mkldnn_or_onednn_registration_appears": mkldnn_registration,
        "autograd_cpu_appears": autograd_cpu,
        "composite_implicit_autograd_appears": composite_implicit,
        "composite_explicit_autograd_appears": composite_explicit,
        "source_registration_locations": source_locations,
        "inferred_backend_signal": inferred,
    }


def op_schema(torch: Any, op_name: str) -> dict[str, Any]:
    helper = getattr(torch._C, "_dispatch_find_schema_or_throw", None)
    if helper is None:
        return {"available": False, "executed": False, "error": "helper_unavailable"}
    base, overload = op_name, ""
    if "." in op_name.removeprefix("aten::"):
        base, overload = op_name.rsplit(".", 1)
    try:
        schema = helper(base, overload)
        return {"available": True, "executed": True, "schema": str(schema)}
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "executed": False, "error": repr(exc)}


def dispatch_key_probe(torch: Any, op_name: str) -> dict[str, Any]:
    helper = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    keys = [
        "CPU",
        "MkldnnCPU",
        "AutogradCPU",
        "CompositeImplicitAutograd",
        "CompositeExplicitAutograd",
    ]
    if helper is None:
        return {"available": False, "executed": False, "error": "helper_unavailable"}
    result: dict[str, Any] = {"available": True, "executed": True, "keys": {}}
    for key in keys:
        try:
            result["keys"][key] = {"available": True, "has_kernel": bool(helper(op_name, key))}
        except Exception as exc:  # noqa: BLE001
            result["keys"][key] = {"available": False, "error": repr(exc)}
    return result


def dispatch_dump(torch: Any, op_name: str) -> dict[str, Any]:
    helper = getattr(torch._C, "_dispatch_dump_table", None)
    if helper is None:
        return {
            "available": False,
            "executed": False,
            "error": "helper_unavailable",
            **classify_op_dump(None, False),
        }
    try:
        raw = str(helper(op_name))
        return {
            "available": True,
            "executed": True,
            "error": None,
            "raw_dump_table": raw,
            **classify_op_dump(raw, True),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": True,
            "executed": False,
            "error": repr(exc),
            "raw_dump_table": None,
            **classify_op_dump(None, False),
        }


def filtered_op_names(torch: Any) -> dict[str, Any]:
    helper = getattr(torch._C, "_dispatch_get_all_op_names", None)
    if helper is None:
        return {"available": False, "executed": False, "error": "helper_unavailable"}
    try:
        names = list(helper())
        filtered = {
            needle: sorted([name for name in names if needle in name])[:200]
            for needle in ["addmm", "linear", "mm", "matmul"]
        }
        return {"available": True, "executed": True, "filtered": filtered}
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "executed": False, "error": repr(exc)}


def environment_metadata(torch: Any) -> dict[str, Any]:
    try:
        config_show = torch.__config__.show()
    except Exception as exc:  # noqa: BLE001
        config_show = f"unavailable: {exc!r}"
    try:
        interop_threads = torch.get_num_interop_threads()
    except Exception:  # noqa: BLE001
        interop_threads = None
    mkldnn = getattr(getattr(torch, "backends", None), "mkldnn", None)
    mkldnn_enabled = getattr(mkldnn, "enabled", None) if mkldnn is not None else None
    return {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torch_import_path": getattr(torch, "__file__", None),
        "torch_version": getattr(torch, "__version__", None),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_config_show": config_show,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": interop_threads,
        "torch_backends_mkldnn_enabled": mkldnn_enabled,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "ONEDNN_VERBOSE": os.environ.get("ONEDNN_VERBOSE"),
        "DNNL_VERBOSE": os.environ.get("DNNL_VERBOSE"),
        "MKL_VERBOSE": os.environ.get("MKL_VERBOSE"),
    }


def tiny_sanity_probe(torch: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": True,
        "executed": False,
        "error": None,
        "cuda_used": False,
        "profiler": None,
    }
    try:
        input_2d = torch.tensor([[0.5, -1.25, 2.0, 0.125]], dtype=torch.bfloat16, device="cpu")
        weight_t = torch.tensor(
            [
                [0.25, -0.75, 1.5],
                [1.0, 0.5, -0.25],
                [-0.5, 0.125, 0.75],
                [0.03125, -0.0625, 0.5],
            ],
            dtype=torch.bfloat16,
            device="cpu",
        )
        bias = torch.tensor([0.125, -0.25, 0.5], dtype=torch.bfloat16, device="cpu")
        output = torch.addmm(bias, input_2d, weight_t)
        result.update(
            {
                "executed": True,
                "input_dtype": str(input_2d.dtype),
                "weight_t_dtype": str(weight_t.dtype),
                "bias_dtype": str(bias.dtype),
                "output_dtype": str(output.dtype),
                "output_device": str(output.device),
                "output_shape": list(output.shape),
                "output_is_cpu_bf16": str(output.device) == "cpu" and output.dtype is torch.bfloat16,
            }
        )
        profiler_result = {"available": False, "executed": False, "events": [], "error": None}
        profiler = getattr(torch, "profiler", None)
        if profiler is not None:
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    torch.addmm(bias, input_2d, weight_t)
                events = []
                for event in prof.key_averages()[:40]:
                    events.append(
                        {
                            "key": event.key,
                            "count": int(event.count),
                            "cpu_time_total_us": float(event.cpu_time_total),
                        }
                    )
                profiler_result.update({"available": True, "executed": True, "events": events})
            except Exception as exc:  # noqa: BLE001
                profiler_result.update({"available": True, "executed": False, "error": repr(exc)})
        result["profiler"] = profiler_result
    except Exception as exc:  # noqa: BLE001
        result.update({"executed": False, "error": repr(exc)})
    return result


def infer_overall_classification(op_results: dict[str, Any]) -> str:
    dump_executed = any(result.get("executed") for result in op_results.values())
    if not dump_executed:
        return "fused_linear_addmm_torch_wheel_dispatch_inconclusive"
    addmm_signal = op_results.get("addmm", {}).get("inferred_backend_signal")
    # A CPU registration tells us where source mapping should begin, but it is
    # not concrete active backend identity for BF16 addmm microkernel behavior.
    if addmm_signal in {"cpu_native_registration", "multiple_possible", "mkldnn_registration_present"}:
        return "fused_linear_addmm_torch_wheel_dispatch_attribution_recorded"
    return "fused_linear_addmm_torch_wheel_dispatch_inconclusive"


def run_worker(args: argparse.Namespace) -> int:
    status = base_status("fused_linear_addmm_torch_wheel_dispatch_failed")
    try:
        import torch

        research_dir = args.research_dir
        research_dir.mkdir(parents=True, exist_ok=True)

        dispatch_results: dict[str, Any] = {}
        for short_name, op_name in OPS.items():
            dump = dispatch_dump(torch, op_name)
            schema = op_schema(torch, op_name)
            key_probe = dispatch_key_probe(torch, op_name)
            dispatch_results[short_name] = {
                "op_name": op_name,
                **dump,
                "schema": schema,
                "dispatch_key_probe": key_probe,
            }
            raw = dump.get("raw_dump_table")
            if raw:
                safe_name = op_name.replace("::", "_").replace(".", "_")
                (research_dir / f"{safe_name}_dispatch_dump.txt").write_text(raw, encoding="utf-8")

        status.update(
            {
                "classification": infer_overall_classification(dispatch_results),
                "selected_python_executable": sys.executable,
                "candidate_python_executables_checked": (
                    load_json(args.candidate_report) if args.candidate_report else []
                ),
                "environment": environment_metadata(torch),
                "sys_executable": sys.executable,
                "sys_prefix": sys.prefix,
                "torch_import_path": getattr(torch, "__file__", None),
                "torch_version": getattr(torch, "__version__", None),
                "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_used": False,
                "dispatch_results": dispatch_results,
                "all_op_names_filtered": filtered_op_names(torch),
                "tiny_bf16_addmm_sanity_probe": tiny_sanity_probe(torch),
                "research_dir": str(research_dir),
                "interpretation": {
                    "dispatch_tables_available": any(
                        result.get("executed") for result in dispatch_results.values()
                    ),
                    "backend_path_identified": False,
                    "backend_identity_caveat": (
                        "Dispatch tables identify registered ATen dispatch entries, "
                        "but do not by themselves prove the concrete active CPU BF16 addmm "
                        "microkernel/arithmetic path."
                    ),
                    "recommended_next_step": (
                        "Use these wheel dispatch/source-registration hints to build the "
                        "future PyTorch source map; do not reopen Rust/CUDA policy work "
                        "without a concrete replayable rule."
                    ),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        status.update(
            {
                "classification": "fused_linear_addmm_torch_wheel_dispatch_failed",
                "error": repr(exc),
                "selected_python_executable": sys.executable,
            }
        )
    write_json(args.status_output, status)
    return 0 if status["classification"] != "fused_linear_addmm_torch_wheel_dispatch_failed" else 1


def blocked_status(
    args: argparse.Namespace,
    candidate_reports: list[dict[str, Any]],
    venv_result: dict[str, Any],
) -> dict[str, Any]:
    status = base_status("fused_linear_addmm_torch_wheel_dispatch_blocked_by_missing_torch")
    status.update(
        {
            "selected_python_executable": None,
            "candidate_python_executables_checked": candidate_reports,
            "empty_research_venv": venv_result,
            "error": "No checked Python environment could import torch.",
            "research_dir": str(args.research_dir),
        }
    )
    return status


def run_controller(args: argparse.Namespace) -> int:
    candidate_paths = discover_candidate_pythons()
    candidate_reports = [check_python_candidate(path) for path in candidate_paths]
    selected = next((report for report in candidate_reports if report.get("torch_importable")), None)
    venv_result = maybe_create_empty_research_venv(candidate_reports)

    if selected is None:
        status = blocked_status(args, candidate_reports, venv_result)
        write_json(args.status_output, status)
        return 0

    for report in candidate_reports:
        report["selected"] = report.get("python_executable") == selected.get("python_executable")

    args.research_dir.mkdir(parents=True, exist_ok=True)
    candidate_report_path = args.research_dir / "candidate_python_executables_checked.json"
    write_json(candidate_report_path, candidate_reports)

    selected_python = str(selected["python_executable"])
    command = [
        selected_python,
        str(Path(__file__).resolve()),
        "--worker",
        "--status-output",
        str(args.status_output),
        "--research-dir",
        str(args.research_dir),
        "--candidate-report",
        str(candidate_report_path),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        status = base_status("fused_linear_addmm_torch_wheel_dispatch_failed")
        status.update(
            {
                "selected_python_executable": selected_python,
                "candidate_python_executables_checked": candidate_reports,
                "worker_returncode": completed.returncode,
                "error": "selected_python_worker_failed",
                "empty_research_venv": venv_result,
            }
        )
        write_json(args.status_output, status)
        return completed.returncode

    status = load_json(args.status_output)
    if isinstance(status, dict):
        status["empty_research_venv"] = venv_result
        write_json(args.status_output, status)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
