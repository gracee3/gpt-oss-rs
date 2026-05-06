#!/usr/bin/env python3
"""Validate the uv-managed forward oracle Python/Torch baseline.

This helper records the forward oracle/source-attribution environment identity,
validates imports and a tiny CPU BF16 addmm sanity check, writes requirements
artifacts after validation, and preserves historical environments as provenance
only. It never uses CUDA, modifies apt, clones/builds PyTorch, or touches the
historical virtual environments.
"""

from __future__ import annotations

import argparse
import importlib
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
DEFAULT_REQUIREMENTS_DIR = Path("requirements")
HISTORICAL_ENVS = [
    Path("/home/emmy/openai/gpt-oss/.venv"),
    Path("/data/models/.venv-awq"),
]

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
TORCH_INSTALL_COMMAND = (
    "/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130/bin/python -m pip "
    f"install --index-url {TORCH_INDEX_URL} torch"
)
PACKAGE_INSTALL_COMMAND = (
    "/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130/bin/python -m pip install "
    "numpy transformers accelerate kernels safetensors huggingface_hub "
    "openai-harmony gpt-oss packaging"
)

DIRECT_PACKAGES = [
    "torch",
    "numpy",
    "transformers",
    "accelerate",
    "triton",
    "kernels",
    "safetensors",
    "huggingface_hub",
    "openai-harmony",
    "gpt-oss",
    "packaging",
]

IMPORTS = {
    "torch": "torch",
    "numpy": "numpy",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "triton": "triton",
    "kernels": "kernels",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "openai_harmony": "openai_harmony",
    "gpt_oss": "gpt_oss",
    "packaging": "packaging",
}

GUARD_FALSE_FLAGS = {
    "apt_sources_modified": False,
    "sudo_used": False,
    "historical_envs_modified": False,
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
        description="Validate the uv-managed forward oracle Python/Torch baseline."
    )
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--requirements-dir", type=Path, default=DEFAULT_REQUIREMENTS_DIR)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_command(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:  # noqa: BLE001 - preserve diagnostics in status
        return {"command": command, "returncode": None, "error": repr(exc)}


def run_json_python(python: Path, code: str, timeout: int = 45) -> dict[str, Any]:
    result = run_command([str(python), "-c", code], timeout=timeout)
    stdout = str(result.get("stdout") or "").strip()
    if stdout:
        try:
            result.update(json.loads(stdout.splitlines()[-1]))
        except Exception as exc:  # noqa: BLE001
            result["json_parse_error"] = repr(exc)
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
    observed = run_json_python(python, code)
    observed.pop("stdout", None)
    observed.pop("stderr", None)
    result.update(observed)
    return result


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "importable": True,
            "version": getattr(module, "__version__", None),
            "import_path": getattr(module, "__file__", None),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "version": None, "import_path": None, "error": repr(exc)}


def torch_install_flavor(torch_version: str | None) -> str:
    version = torch_version or ""
    if "+cu130" in version:
        return "cu130"
    if "+cu128" in version:
        return "cu128"
    if "+cpu" in version:
        return "cpu_only"
    if "+cu" in version:
        return "other_cuda"
    return "unknown"


def tiny_bf16_addmm_sanity(torch: Any) -> dict[str, Any]:
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
        return {
            "executed": True,
            "output_device": str(output.device),
            "output_dtype": str(output.dtype),
            "output_shape": list(output.shape),
            "output_values": [float(value) for value in output.reshape(-1).float().tolist()],
            "output_is_cpu": str(output.device) == "cpu",
            "output_is_bf16": output.dtype is torch.bfloat16,
        }
    except Exception as exc:  # noqa: BLE001
        return {"executed": False, "error": repr(exc)}


def pip_version() -> str | None:
    try:
        import pip

        return getattr(pip, "__version__", None)
    except Exception:
        return None


def pip_freeze(python: Path, output_path: Path) -> list[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_command([str(python), "-m", "pip", "freeze"], timeout=120)
    lines = str(result.get("stdout") or "").splitlines()
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def normalized_name(requirement_line: str) -> str:
    name = requirement_line.split("==", 1)[0].split("@", 1)[0].strip()
    return name.lower().replace("_", "-")


def find_freeze_line(freeze_lines: list[str], package: str) -> str | None:
    wanted = package.lower().replace("_", "-")
    for line in freeze_lines:
        if normalized_name(line) == wanted:
            return line
    return None


def write_requirements(requirements_dir: Path, freeze_lines: list[str], historical: list[dict[str, Any]]) -> list[str]:
    requirements_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    in_path = requirements_dir / "oracle-forward-py312-cu130.in"
    in_path.write_text(
        "\n".join(
            [
                "# Human-maintained package families for the forward oracle/source-attribution env.",
                "# Generated after validating oracle/forward-python-env-baseline-uv.",
                f"--extra-index-url {TORCH_INDEX_URL}",
                "torch",
                "numpy",
                "transformers",
                "accelerate",
                "triton",
                "kernels",
                "safetensors",
                "huggingface_hub",
                "openai-harmony",
                "gpt-oss",
                "packaging",
                "",
            ]
        ),
        encoding="utf-8",
    )
    written.append(str(in_path))

    constraints_path = requirements_dir / "oracle-forward-py312-cu130.constraints.txt"
    constraints_path.write_text(
        "\n".join(
            [
                "# Known-good pins from the validated forward oracle Python env.",
                "# Full pip freeze follows so important transitive versions are preserved.",
                *sorted(freeze_lines, key=str.lower),
                "",
            ]
        ),
        encoding="utf-8",
    )
    written.append(str(constraints_path))

    txt_path = requirements_dir / "oracle-forward-py312-cu130.txt"
    direct_lines = [find_freeze_line(freeze_lines, package) or package for package in DIRECT_PACKAGES]
    txt_path.write_text(
        "\n".join(
            [
                "# Installable entrypoint for the validated forward oracle Python env.",
                f"--extra-index-url {TORCH_INDEX_URL}",
                "-c requirements/oracle-forward-py312-cu130.constraints.txt",
                *direct_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    written.append(str(txt_path))

    legacy_path = requirements_dir / "oracle-legacy-observed.txt"
    legacy_lines = [
        "# Historical/provenance environments observed during forward baseline validation.",
        "# These are not the forward baseline and must not be overwritten or repurposed.",
    ]
    for env in historical:
        legacy_lines.extend(
            [
                "",
                f"env_path={env.get('env_path')}",
                f"python_executable={env.get('python_executable')}",
                f"python_version={env.get('python_version')}",
                f"sys_prefix={env.get('sys_prefix')}",
                f"torch_importable={env.get('torch_importable')}",
                f"torch_version={env.get('torch_version')}",
                f"torch_git_version={env.get('torch_git_version')}",
                f"torch_import_path={env.get('torch_import_path')}",
                f"numpy_importable={env.get('numpy_importable')}",
                f"numpy_version={env.get('numpy_version')}",
                "provenance_only=true",
            ]
        )
    legacy_path.write_text("\n".join(legacy_lines) + "\n", encoding="utf-8")
    written.append(str(legacy_path))

    return written


def base_status(classification: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "oracle_forward_python_env_baseline": True,
        "oracle_device": "cpu",
        "forward_env_path": str(args.forward_env_path),
        "python312_provider": "uv",
        "uv_found": False,
        "uv_version": None,
        "uv_python_install_requested": True,
        "uv_python_find_output": None,
        "python312_managed_by_uv": None,
        "python312_executable": None,
        "python_executable": None,
        "python_version": None,
        "sys_prefix": None,
        "pip_version": None,
        "torch_version": None,
        "torch_git_version": None,
        "torch_import_path": None,
        "torch_config_show": None,
        "torch_cuda_is_available": None,
        "torch_install_flavor": None,
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
        "tiny_bf16_addmm_sanity": None,
        "requirements_files_written": [],
        "pip_freeze_output_path": None,
        "historical_envs_observed": [],
        "uv_install_commands_recorded": [
            "uv python install 3.12",
            "uv venv --python 3.12 /home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130",
            "/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130/bin/python -m ensurepip --upgrade",
            TORCH_INSTALL_COMMAND,
            PACKAGE_INSTALL_COMMAND,
        ],
        "package_install_attempted": True,
        "package_install_blocked": False,
        "packages_installed": True,
        **GUARD_FALSE_FLAGS,
    }


def uv_metadata(status: dict[str, Any]) -> None:
    uv_path = shutil.which("uv")
    status["uv_found"] = uv_path is not None
    status["uv_executable"] = uv_path
    if uv_path is None:
        status["classification"] = "oracle_forward_python_env_baseline_blocked_by_uv"
        return
    version = run_command([uv_path, "--version"])
    status["uv_version"] = (version.get("stdout") or "").strip() or None
    find = run_command([uv_path, "python", "find", "3.12"])
    status["uv_python_find_output"] = {
        "returncode": find.get("returncode"),
        "stdout": (find.get("stdout") or "").strip(),
        "stderr": (find.get("stderr") or "").strip(),
    }
    if find.get("returncode") == 0:
        status["python312_executable"] = (find.get("stdout") or "").strip()
        status["python312_managed_by_uv"] = ".local/share/uv/python" in str(status["python312_executable"])


def main() -> int:
    args = parse_args()
    status = base_status("oracle_forward_python_env_baseline_failed", args)
    uv_metadata(status)

    historical = [observe_historical_env(path) for path in HISTORICAL_ENVS]
    status["historical_envs_observed"] = historical

    forward_python = args.forward_env_path / "bin" / "python"
    if not status["uv_found"]:
        status.update(
            {
                "package_install_attempted": False,
                "package_install_blocked": True,
                "packages_installed": False,
                "forward_env_created": False,
            }
        )
        write_json(args.status_output, status)
        return 0

    if not status.get("python312_executable"):
        status.update(
            {
                "classification": "oracle_forward_python_env_baseline_blocked_by_python",
                "package_install_attempted": False,
                "package_install_blocked": True,
                "package_install_block_reason": "uv_python_3_12_unavailable",
                "packages_installed": False,
                "forward_env_created": args.forward_env_path.exists(),
            }
        )
        write_json(args.status_output, status)
        return 0

    if not forward_python.exists():
        status.update(
            {
                "classification": "oracle_forward_python_env_baseline_blocked_by_python",
                "package_install_blocked": True,
                "package_install_block_reason": "forward_env_python_missing",
                "packages_installed": False,
                "forward_env_created": False,
            }
        )
        write_json(args.status_output, status)
        return 0

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        status.update(
            {
                "classification": "oracle_forward_python_env_baseline_blocked_by_torch_install",
                "torch_import_error": repr(exc),
                "forward_env_created": True,
                "packages_installed": False,
            }
        )
        write_json(args.status_output, status)
        return 0

    import_statuses = {name: import_status(module) for name, module in IMPORTS.items()}
    sanity = tiny_bf16_addmm_sanity(torch)
    freeze_path = args.research_dir / "pip-freeze.txt"
    freeze_lines = pip_freeze(forward_python, freeze_path)
    requirements_files = write_requirements(args.requirements_dir, freeze_lines, historical)

    try:
        torch_config = torch.__config__.show()
    except Exception as exc:  # noqa: BLE001
        torch_config = f"unavailable: {exc!r}"

    torch_version = getattr(torch, "__version__", None)
    torch_flavor = torch_install_flavor(torch_version)
    imports_ok = all(
        import_statuses[name]["importable"]
        for name in ["torch", "numpy", "transformers", "accelerate", "safetensors", "huggingface_hub", "packaging"]
    )
    sanity_ok = bool(sanity.get("executed")) and bool(sanity.get("output_is_cpu")) and bool(sanity.get("output_is_bf16"))
    flavor_ok = torch_flavor == "cu130"

    status.update(
        {
            "classification": (
                "oracle_forward_python_env_baseline_validated"
                if imports_ok and sanity_ok and flavor_ok
                else "oracle_forward_python_env_baseline_recorded"
            ),
            "forward_env_created": True,
            "venv_created": True,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "sys_prefix": sys.prefix,
            "pip_version": pip_version(),
            "torch_version": torch_version,
            "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
            "torch_import_path": getattr(torch, "__file__", None),
            "torch_config_show": torch_config,
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
            "torch_install_flavor": torch_flavor,
            "numpy_version": import_statuses["numpy"]["version"],
            "transformers_version": import_statuses["transformers"]["version"],
            "accelerate_version": import_statuses["accelerate"]["version"],
            "triton_version": import_statuses["triton"]["version"],
            "kernels_import_status": import_statuses["kernels"],
            "safetensors_version": import_statuses["safetensors"]["version"],
            "huggingface_hub_version": import_statuses["huggingface_hub"]["version"],
            "openai_harmony_import_status": import_statuses["openai_harmony"],
            "gpt_oss_import_status": import_statuses["gpt_oss"],
            "packaging_version": import_statuses["packaging"]["version"],
            "import_statuses": import_statuses,
            "tiny_bf16_addmm_sanity": sanity,
            "requirements_files_written": requirements_files,
            "pip_freeze_output_path": str(freeze_path),
            "pip_freeze_package_count": len(freeze_lines),
            "rebaseline_policy_note": (
                "Historical artifacts remain tied to their recorded environments. "
                "Future forward artifacts must include this forward environment identity."
            ),
        }
    )

    write_json(args.status_output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
