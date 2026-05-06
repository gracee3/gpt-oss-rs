#!/usr/bin/env python3
"""Forward Python/Torch oracle environment baseline helper.

This helper implements the fail-closed front door for the forward
oracle/source-attribution Python environment. It refuses to silently fall back
from Python 3.12, records historical/provenance environment observations, and
emits a status JSON with the standard oracle guardrails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_FORWARD_ENV = Path("/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130")
DEFAULT_STATUS = Path("/tmp/oracle_forward_python_env_baseline_status.json")
DEFAULT_RESEARCH_DIR = Path("/home/emmy/openai/pytorch-research/oracle-forward-python-env-baseline")
HISTORICAL_ENVS = [
    Path("/home/emmy/openai/gpt-oss/.venv"),
    Path("/data/models/.venv-awq"),
]
PYTHON_COMMANDS = ["python3.12"]

GUARD_FALSE_FLAGS = {
    "pytorch_clone_performed": False,
    "pytorch_build_performed": False,
    "pytorch_source_patched": False,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or fail closed for the forward oracle Python/Torch baseline."
    )
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_json_python(python: Path, code: str, timeout: int = 45) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result: dict[str, Any] = {
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
        }
        stdout = completed.stdout.strip()
        if stdout:
            result.update(json.loads(stdout.splitlines()[-1]))
        return result
    except Exception as exc:  # noqa: BLE001 - preserve diagnostics in status
        return {"returncode": None, "error": repr(exc)}


def check_python_command(command: str) -> dict[str, Any]:
    resolved = shutil.which(command)
    result: dict[str, Any] = {
        "command": command,
        "resolved": resolved,
        "found": resolved is not None,
        "version": None,
        "is_python_3_12": False,
        "error": None,
    }
    if resolved is None:
        result["error"] = "command_not_found"
        return result
    completed = subprocess.run(
        [resolved, "-c", "import json, sys; print(json.dumps({'version': sys.version, 'major': sys.version_info[0], 'minor': sys.version_info[1]}))"],
        check=False,
        capture_output=True,
        text=True,
    )
    result["returncode"] = completed.returncode
    if completed.returncode != 0:
        result["error"] = completed.stderr[-1000:] or "version_check_failed"
        return result
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    result["version"] = payload["version"]
    result["is_python_3_12"] = payload["major"] == 3 and payload["minor"] == 12
    return result


def observe_historical_env(path: Path) -> dict[str, Any]:
    python = path / "bin" / "python"
    result: dict[str, Any] = {
        "env_path": str(path),
        "python_executable": str(python),
        "env_exists": path.exists(),
        "python_exists": python.exists(),
        "provenance_only": True,
        "forward_baseline": False,
    }
    if not python.exists():
        result["error"] = "historical_python_missing"
        return result
    code = r"""
import importlib
import json
import sys

payload = {
    "python_version": sys.version,
    "sys_prefix": sys.prefix,
}
try:
    import torch
    payload.update({
        "torch_importable": True,
        "torch_version": getattr(torch, "__version__", None),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_import_path": getattr(torch, "__file__", None),
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
    })
except Exception as exc:
    payload.update({"torch_importable": False, "torch_error": repr(exc)})
try:
    import numpy
    payload.update({"numpy_importable": True, "numpy_version": getattr(numpy, "__version__", None)})
except Exception as exc:
    payload.update({"numpy_importable": False, "numpy_error": repr(exc)})
print(json.dumps(payload, sort_keys=True))
"""
    result.update(run_json_python(python, code))
    return result


def base_status(classification: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "oracle_forward_python_env_baseline": True,
        "oracle_device": "cpu",
        "forward_env_path": str(args.forward_env_path),
        "python_executable": None,
        "python_version": None,
        "sys_prefix": None,
        "pip_version": None,
        "torch_version": None,
        "torch_git_version": None,
        "torch_import_path": None,
        "torch_config_show": None,
        "torch_cuda_is_available": None,
        "cuda_used": False,
        "numpy_version": None,
        "transformers_version": None,
        "accelerate_version": None,
        "triton_version": None,
        "kernels_import_status": None,
        "safetensors_version": None,
        "huggingface_hub_version": None,
        "openai_harmony_import_status": None,
        "gpt_oss_import_status": None,
        "packaging_version": None,
        "pip_freeze_output_path": None,
        "requirements_files_written": [],
        "historical_envs_observed": [],
        **GUARD_FALSE_FLAGS,
    }


def blocked_by_python_status(args: argparse.Namespace, checks: list[dict[str, Any]]) -> dict[str, Any]:
    status = base_status("oracle_forward_python_env_baseline_blocked_by_python", args)
    status.update(
        {
            "python3.12_found": False,
            "candidate_python_commands_checked": checks,
            "forward_env_created": False,
            "package_install_attempted": False,
            "package_install_blocked": True,
            "package_install_block_reason": "python3.12_not_found",
            "venv_created": False,
            "validation_helper_executed_with_new_venv": False,
            "tiny_bf16_addmm_sanity": None,
            "pip_freeze_written": False,
            "research_dir": str(args.research_dir),
            "historical_envs_observed": [observe_historical_env(path) for path in HISTORICAL_ENVS],
            "rebaseline_policy_note": (
                "Historical artifacts remain tied to their recorded environments. "
                "No forward baseline was created because Python 3.12 is unavailable."
            ),
        }
    )
    return status


def main() -> int:
    args = parse_args()
    checks = [check_python_command(command) for command in PYTHON_COMMANDS]
    python312 = next((check for check in checks if check.get("is_python_3_12")), None)
    if python312 is None:
        write_json(args.status_output, blocked_by_python_status(args, checks))
        return 0

    # The current environment did not have Python 3.12 when this helper was
    # introduced. Keep the success path explicit and fail closed until the
    # implementation can be run and validated under Python 3.12.
    status = base_status("oracle_forward_python_env_baseline_recorded", args)
    status.update(
        {
            "python3.12_found": True,
            "candidate_python_commands_checked": checks,
            "python_executable": python312["resolved"],
            "python_version": python312["version"],
            "forward_env_created": False,
            "package_install_attempted": False,
            "package_install_blocked": True,
            "package_install_block_reason": "success_path_requires_explicit_validated_install_step",
            "historical_envs_observed": [observe_historical_env(path) for path in HISTORICAL_ENVS],
        }
    )
    write_json(args.status_output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
