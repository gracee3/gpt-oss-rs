#!/usr/bin/env python3
"""CPU-only PyTorch source instrumentation workflow for fused addmm.

This helper records the source-build environment, attempts a CPU-only
instrumented PyTorch build from an already-patched checkout, and, if the build
succeeds, runs a captured-tensor layer18 addmm probe under baseline and
ATEN_CPU_CAPABILITY=default. It does not use CUDA or load the full model.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path("/home/emmy/openai/pytorch")
DEFAULT_BUILD_ENV = Path("/home/emmy/openai/.venvs/pytorch-src-cpu")
DEFAULT_FORWARD_ENV = Path("/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130")
DEFAULT_MODEL = Path("/data/models/openai/gpt-oss-20b-full-attn-restricted-integration")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-pytorch-cpu-instrumentation")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_pytorch_cpu_instrumentation_status.json")
EXPECTED_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
TRACE_ENV = "GPT_OSS_TRACE_ADDMM"
INSTRUMENTED_FILES = [
    "aten/src/ATen/native/Linear.cpp",
    "aten/src/ATen/native/LinearAlgebra.cpp",
    "aten/src/ATen/native/CPUBlas.cpp",
    "aten/src/ATen/native/mkldnn/Matmul.cpp",
]
LAYER18_FOCUS_LANE = 1641
GUARD_FALSE_FLAGS = {
    "backend_selected": False,
    "implementation_authorized": False,
    "consumer_revalidation_authorized": False,
    "runtime_behavior_changed": False,
    "production_routing_changed": False,
    "cuda_kernels_changed": False,
    "output_emitted": False,
    "ladder_continued": False,
    "final_logit_claim": False,
    "all_layer_claim": False,
    "server_claim": False,
    "context_length_claim": False,
    "tolerance_pass": False,
    "correction_metadata_applied": False,
    "rebaseline_performed": False,
    "old_artifacts_replaced": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instrumented CPU PyTorch source-build addmm attribution.")
    parser.add_argument("--source-checkout-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--build-env-path", type=Path, default=DEFAULT_BUILD_ENV)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--build-timeout-seconds", type=int, default=14400)
    parser.add_argument("--runtime-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--capability-config", default="baseline", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    log_stdout: Path | None = None,
    log_stderr: Path | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if log_stdout:
            log_stdout.parent.mkdir(parents=True, exist_ok=True)
            log_stdout.write_text(completed.stdout, encoding="utf-8", errors="replace")
        if log_stderr:
            log_stderr.parent.mkdir(parents=True, exist_ok=True)
            log_stderr.write_text(completed.stderr, encoding="utf-8", errors="replace")
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout.splitlines()[-80:],
            "stderr_tail": completed.stderr.splitlines()[-80:],
            "succeeded": completed.returncode == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "stdout_tail": [],
            "stderr_tail": [repr(exc)],
            "succeeded": False,
            "error": repr(exc),
        }


def last_output_line(result: dict[str, Any]) -> str | None:
    lines = result.get("stdout_tail", []) or result.get("stderr_tail", [])
    return lines[-1] if lines else None


def torch_config_show(torch: Any) -> str:
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            torch.__config__.show()
        return stream.getvalue()
    except Exception as exc:  # noqa: BLE001
        return f"<torch.__config__.show failed: {exc!r}>"


def base_status(classification: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "source_attribution_probe": True,
        "pytorch_cpu_instrumentation": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "source_checkout_path": str(args.source_checkout_path),
        "checked_out_commit": EXPECTED_COMMIT,
        "build_env_path": str(args.build_env_path),
        "pytorch_build_performed": False,
        "pytorch_source_patched": False,
        "instrumentation_env_var": TRACE_ENV,
        "instrumented_source_files": [],
        "build_log_paths": {},
        "layers_evaluated": [],
        "configs_evaluated": [],
        "layer18_lane1641_result": {},
        "active_path_baseline": "inconclusive",
        "active_path_default": "inconclusive",
        "path_changed_under_default": None,
        "instrumentation_perturbed_numeric_behavior": None,
        "concrete_replayable_rule_found": False,
        "replayable_rule_summary": None,
        "reopen_rust_policy_synthesis": False,
        **GUARD_FALSE_FLAGS,
    }


def source_state(source: Path) -> dict[str, Any]:
    head = run_cmd(["git", "rev-parse", "HEAD"], cwd=source)
    status = run_cmd(["git", "status", "--short"], cwd=source)
    diff_files = run_cmd(["git", "diff", "--name-only"], cwd=source)
    return {
        "head": head.get("stdout_tail", [""])[-1] if head.get("stdout_tail") else "",
        "status_short": status.get("stdout_tail", []),
        "dirty_files": diff_files.get("stdout_tail", []),
        "expected_commit_match": (head.get("stdout_tail", [""])[-1] if head.get("stdout_tail") else "") == EXPECTED_COMMIT,
    }


def create_or_use_venv(args: argparse.Namespace) -> dict[str, Any]:
    env_path = args.build_env_path
    created = False
    uv_version = None
    uv_path = shutil.which("uv")
    if uv_path:
        version = run_cmd([uv_path, "--version"])
        uv_version = last_output_line(version)
    if not (env_path / "bin/python").is_file():
        if uv_path:
            result = run_cmd([uv_path, "venv", "--python", "3.12", str(env_path)], timeout=900)
        else:
            python = shutil.which("python3.12") or shutil.which("python3")
            result = run_cmd([python, "-m", "venv", str(env_path)], timeout=900) if python else {"succeeded": False, "error": "no python found"}
        created = bool(result.get("succeeded"))
        if not created:
            return {
                "venv_created": False,
                "venv_create_result": result,
                "python_executable": None,
                "uv_path": uv_path,
                "uv_version": uv_version,
            }
    python = env_path / "bin/python"
    pip = env_path / "bin/pip"
    ensurepip_result = None
    if not pip.is_file():
        ensurepip_result = run_cmd([str(python), "-m", "ensurepip", "--upgrade"], timeout=900)
    py_version = run_cmd([str(python), "--version"])
    pip_version = run_cmd([str(pip), "--version"]) if pip.is_file() else {"stdout_tail": []}
    return {
        "venv_created": created,
        "python_executable": str(python),
        "python_version": last_output_line(py_version),
        "pip_version": last_output_line(pip_version),
        "ensurepip_result": ensurepip_result,
        "uv_path": uv_path,
        "uv_version": uv_version,
    }


def install_build_requirements(args: argparse.Namespace, python: Path) -> dict[str, Any]:
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [str(python), "-m", "pip", "install", "-r", str(args.source_checkout_path / "requirements.txt")],
        [str(python), "-m", "pip", "install", "-r", str(args.source_checkout_path / "requirements-build.txt")],
        [str(python), "-m", "pip", "install", "safetensors"],
    ]
    results = []
    for index, cmd in enumerate(commands):
        result = run_cmd(
            cmd,
            cwd=args.source_checkout_path,
            timeout=1800,
            log_stdout=args.research_path / f"pip-install-{index}.log",
            log_stderr=args.research_path / f"pip-install-{index}.err.log",
        )
        results.append(result)
        if not result["succeeded"]:
            break
    return {"commands": commands, "results": results, "succeeded": all(result["succeeded"] for result in results)}


def build_pytorch(args: argparse.Namespace, python: Path) -> dict[str, Any]:
    build_log = args.research_path / "build.log"
    build_err = args.research_path / "build-error.log"
    env = os.environ.copy()
    env.update(
        {
            "USE_CUDA": "0",
            "BUILD_TEST": "0",
            "USE_DISTRIBUTED": "0",
            "USE_NNPACK": "0",
            "USE_QNNPACK": "0",
            "USE_XNNPACK": "0",
            "USE_MKLDNN": "1",
            "MAX_JOBS": env.get("MAX_JOBS", "8"),
        }
    )
    cmd = [str(python), "setup.py", "develop"]
    result = run_cmd(
        cmd,
        cwd=args.source_checkout_path,
        env=env,
        timeout=args.build_timeout_seconds,
        log_stdout=build_log,
        log_stderr=build_err,
    )
    result["build_log"] = str(build_log)
    result["build_error_log"] = str(build_err)
    result["env"] = {key: env.get(key) for key in ["USE_CUDA", "BUILD_TEST", "USE_DISTRIBUTED", "USE_MKLDNN", "MAX_JOBS"]}
    return result


def json_tensor_values(path: Path) -> list[float]:
    data = read_json(path)
    values = data.get("values")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain a values array")
    return [float(value) for value in values]


def compare_tensors(torch: Any, actual: Any, expected: Any, focus_lane: int) -> dict[str, Any]:
    actual_cpu = actual.detach().to("cpu").reshape(-1)
    expected_cpu = expected.detach().to("cpu").reshape(-1)
    diff = (actual_cpu.float() - expected_cpu.float()).abs()
    mismatch_mask = diff != 0
    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).reshape(-1)
    mismatches = int(mismatch_indices.numel())
    first = None
    worst = None
    if mismatches:
        first_idx = int(mismatch_indices[0].item())
        worst_idx = int(torch.argmax(diff).item())
        first = {
            "index": first_idx,
            "actual": float(actual_cpu[first_idx].float().item()),
            "expected": float(expected_cpu[first_idx].float().item()),
            "abs_diff": float(diff[first_idx].item()),
        }
        worst = {
            "index": worst_idx,
            "actual": float(actual_cpu[worst_idx].float().item()),
            "expected": float(expected_cpu[worst_idx].float().item()),
            "abs_diff": float(diff[worst_idx].item()),
        }
    return {
        "mismatch_count": mismatches,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "first_mismatch": first,
        "worst_mismatch": worst,
        "focus_lane": {
            "lane": focus_lane,
            "actual": float(actual_cpu[focus_lane].float().item()),
            "expected": float(expected_cpu[focus_lane].float().item()),
            "abs_diff": float(diff[focus_lane].item()),
            "matched": float(diff[focus_lane].item()) == 0.0,
        },
        "full_vector_cleared": mismatches == 0 and (float(diff.max().item()) if diff.numel() else 0.0) == 0.0,
    }


def runtime_child(args: argparse.Namespace) -> int:
    import torch

    gpt_oss_root = REPO_ROOT.parents[1] / "gpt-oss"
    if (gpt_oss_root / "gpt_oss").is_dir():
        sys.path.insert(0, str(gpt_oss_root))
    from gpt_oss.torch.weights import Checkpoint

    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    layer = 18
    weighted = torch.tensor(json_tensor_values(Path(f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json")), dtype=torch.float32, device="cpu").to(torch.bfloat16)
    official = torch.tensor(json_tensor_values(Path(f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json")), dtype=torch.float32, device="cpu").to(torch.bfloat16)
    weight = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.weight")
    bias = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.bias")
    input_2d = weighted.unsqueeze(0)
    weight_t = weight.t()
    zero_bias = torch.zeros_like(bias)
    outputs = {
        "torch_addmm": torch.addmm(bias, input_2d, weight_t).squeeze(0),
        "torch_nn_functional_linear": torch.nn.functional.linear(input_2d, weight, bias).squeeze(0),
        "torch_C_nn_linear": torch._C._nn.linear(input_2d, weight, bias).squeeze(0),
        "zero_bias_addmm_plus_bias": torch.addmm(zero_bias, input_2d, weight_t).squeeze(0) + bias,
        "explicit_matmul_plus_bias": (input_2d @ weight_t).squeeze(0) + bias,
        "explicit_einsum_plus_bias": torch.einsum("bk,hk->bh", input_2d, weight).squeeze(0) + bias,
    }
    result = {
        "config": args.capability_config,
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_import_path": str(Path(torch.__file__).resolve()),
        "torch_config_show": torch_config_show(torch),
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_used": False,
        "layer": layer,
        "lane": LAYER18_FOCUS_LANE,
        "official_lane1641": float(official[LAYER18_FOCUS_LANE].float().item()),
        "variants": {},
    }
    for name, output in outputs.items():
        if str(output.device) != "cpu":
            raise RuntimeError(f"{name} unexpectedly landed on {output.device}")
        result["variants"][name] = {
            "lane1641": float(output[LAYER18_FOCUS_LANE].float().item()),
            "comparison_vs_official": compare_tensors(torch, output, official, LAYER18_FOCUS_LANE),
        }
    if args.child_output:
        write_json(args.child_output, result)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


def run_runtime_probe(args: argparse.Namespace, python: Path) -> dict[str, Any]:
    results = {}
    traces = {}
    for config in ["baseline", "default"]:
        env = os.environ.copy()
        env[TRACE_ENV] = "1"
        if config == "baseline":
            env.pop("ATEN_CPU_CAPABILITY", None)
        else:
            env["ATEN_CPU_CAPABILITY"] = config
        child_output = args.research_path / f"layer18-{config}.json"
        trace_log = args.research_path / f"layer18-{config}-trace.log"
        cmd = [
            str(python),
            str(Path(__file__).resolve()),
            "--runtime-child",
            "--capability-config",
            config,
            "--model",
            str(args.model),
            "--child-output",
            str(child_output),
        ]
        completed = run_cmd(cmd, env=env, timeout=600, log_stdout=args.research_path / f"layer18-{config}.stdout.log", log_stderr=trace_log)
        traces[config] = str(trace_log)
        results[config] = {"run": completed, "output_path": str(child_output), "trace_log": str(trace_log)}
        if completed["succeeded"] and child_output.is_file():
            results[config]["output"] = read_json(child_output)
    return {"results": results, "trace_logs": traces}


def parse_active_path(trace_path: Path) -> str:
    if not trace_path.is_file():
        return "inconclusive"
    text = trace_path.read_text(encoding="utf-8", errors="replace")
    if "cpublas_bf16_to_bf16_path=mkldnn_bf16_gemm" in text:
        return "mkldnn_onednn_bf16_gemm"
    if "cpublas_bf16_to_bf16_path=BLAS_HAS_SBGEMM" in text or "cpublas_bf16_to_f32_path=MKL_HAS_SBGEMM" in text:
        return "mkl_blas_sbgemm"
    if "cpublas_bf16_to_bf16_path=gemm_stub" in text:
        return "native_cpublas_stub"
    if "addmm_impl_cpu_dispatch" in text:
        return "optimized_cpu_kernel"
    return "inconclusive"


def summarize_runtime(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    baseline = runtime["results"].get("baseline", {}).get("output")
    default = runtime["results"].get("default", {}).get("output")
    summary: dict[str, Any] = {
        "layers_evaluated": [],
        "configs_evaluated": [],
        "layer18_lane1641_result": {},
        "instrumentation_perturbed_numeric_behavior": None,
        "active_path_baseline": parse_active_path(Path(runtime["trace_logs"].get("baseline", ""))),
        "active_path_default": parse_active_path(Path(runtime["trace_logs"].get("default", ""))),
        "path_changed_under_default": None,
    }
    if baseline:
        summary["configs_evaluated"].append("baseline")
        summary["layers_evaluated"] = [18]
        addmm = baseline["variants"]["torch_addmm"]
        summary["instrumentation_perturbed_numeric_behavior"] = not bool(addmm["comparison_vs_official"]["full_vector_cleared"])
    if default:
        summary["configs_evaluated"].append("default")
    if baseline and default:
        b_addmm = baseline["variants"]["torch_addmm"]
        d_addmm = default["variants"]["torch_addmm"]
        summary["layer18_lane1641_result"] = {
            "official": baseline["official_lane1641"],
            "baseline": b_addmm["lane1641"],
            "default": d_addmm["lane1641"],
            "default_vs_official": d_addmm["comparison_vs_official"]["focus_lane"],
            "baseline_vs_official": b_addmm["comparison_vs_official"]["focus_lane"],
        }
        summary["path_changed_under_default"] = summary["active_path_baseline"] != summary["active_path_default"]
    return summary


def main() -> int:
    args = parse_args()
    args.research_path.mkdir(parents=True, exist_ok=True)
    if args.runtime_child:
        return runtime_child(args)

    status = base_status("fused_linear_addmm_pytorch_cpu_instrumentation_inconclusive", args)
    try:
        state = source_state(args.source_checkout_path)
        status["source_state"] = state
        status["pytorch_source_patched"] = any(path in state.get("dirty_files", []) for path in INSTRUMENTED_FILES)
        status["instrumented_source_files"] = [path for path in INSTRUMENTED_FILES if path in state.get("dirty_files", [])]
        if not state["expected_commit_match"]:
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_failed"
            status["failure_summary"] = "PyTorch checkout is not at the expected commit."
            write_json(args.status_output, status)
            return 1

        env_summary = create_or_use_venv(args)
        status["build_env_summary"] = env_summary
        write_json(args.research_path / "build-env-summary.json", env_summary)
        python_exe = env_summary.get("python_executable")
        if not python_exe:
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_build_failed"
            status["failure_summary"] = "Could not create or locate source-build Python environment."
            write_json(args.status_output, status)
            return 0
        python = Path(python_exe)

        install = install_build_requirements(args, python)
        status["packages_installed_this_run"] = bool(install["succeeded"])
        status["install_requirements_result"] = install
        if not install["succeeded"]:
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_build_failed"
            status["failure_summary"] = "Source-build Python requirements installation failed."
            status["build_log_paths"] = {
                "pip_logs": [str(args.research_path / f"pip-install-{index}.log") for index in range(len(install["results"]))],
                "pip_error_logs": [str(args.research_path / f"pip-install-{index}.err.log") for index in range(len(install["results"]))],
            }
            write_json(args.status_output, status)
            return 0

        build = build_pytorch(args, python)
        status["pytorch_build_performed"] = True
        status["build_result"] = build
        status["build_log_paths"] = {"build_log": build["build_log"], "build_error_log": build["build_error_log"]}
        if not build["succeeded"]:
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_build_failed"
            status["failure_summary"] = "CPU-only instrumented PyTorch source build failed."
            write_json(args.status_output, status)
            return 0

        runtime = run_runtime_probe(args, python)
        runtime_summary = summarize_runtime(args, runtime)
        status["runtime_probe"] = runtime
        status.update(runtime_summary)
        status["build_log_paths"].update(runtime.get("trace_logs", {}))
        status["concrete_replayable_rule_found"] = False
        status["reopen_rust_policy_synthesis"] = False
        if status["instrumentation_perturbed_numeric_behavior"]:
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_inconclusive"
        elif status["active_path_baseline"] != "inconclusive" or status["active_path_default"] != "inconclusive":
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_path_identified_not_replayable"
        else:
            status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_inconclusive"
        write_json(args.research_path / "layer18-comparison.json", runtime_summary)
        write_json(
            args.research_path / "interpretation-summary.json",
            {
                "classification": status["classification"],
                "active_path_baseline": status["active_path_baseline"],
                "active_path_default": status["active_path_default"],
                "path_changed_under_default": status["path_changed_under_default"],
                "concrete_replayable_rule_found": status["concrete_replayable_rule_found"],
                "reopen_rust_policy_synthesis": status["reopen_rust_policy_synthesis"],
            },
        )
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status["classification"] = "fused_linear_addmm_pytorch_cpu_instrumentation_failed"
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
