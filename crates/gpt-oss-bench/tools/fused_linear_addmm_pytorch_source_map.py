#!/usr/bin/env python3
"""PyTorch source-map attribution for fused addmm attention o-proj.

This Stage 2 source-attribution helper checks out PyTorch source matching the
installed Torch wheel git version and maps aten::addmm/linear/mm/matmul
registrations to source files. It reads source only: no build, no patching, no
submodule initialization, and no CUDA execution.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_FORWARD_ENV = Path("/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130")
DEFAULT_SOURCE = Path("/home/emmy/openai/pytorch")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-pytorch-source-map")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_pytorch_source_map_status.json")
REQUESTED_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
GITHUB_URL = "https://github.com/pytorch/pytorch.git"

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
}

OPS = {
    "linear": ["linear", "mkldnn_linear", "_linear"],
    "addmm": ["addmm", "addmm_impl_cpu_", "structured_addmm", "ADDMM"],
    "mm": ["mm", "mm_out", "mm_cpu", "addmm_impl_cpu_"],
    "matmul": ["matmul", "baddbmm", "_matmul_impl"],
}

OP_SEARCH_PATHS = {
    "linear": [
        "aten/src/ATen/native/Linear.cpp",
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/mkldnn/Linear.cpp",
        "aten/src/ATen/native/mkldnn/Linear.h",
    ],
    "addmm": [
        "aten/src/ATen/native/Linear.cpp",
        "aten/src/ATen/native/LinearAlgebra.cpp",
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/mkldnn/Matmul.cpp",
        "aten/src/ATen/native/mkldnn/Matmul.h",
    ],
    "mm": [
        "aten/src/ATen/native/LinearAlgebra.cpp",
        "aten/src/ATen/native/Blas.cpp",
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/mkldnn/Matmul.cpp",
        "aten/src/ATen/native/mkldnn/Matmul.h",
    ],
    "matmul": [
        "aten/src/ATen/native/Linear.cpp",
        "aten/src/ATen/native/LinearAlgebra.cpp",
        "aten/src/ATen/native/native_functions.yaml",
        "aten/src/ATen/native/mkldnn/Matmul.cpp",
        "aten/src/ATen/native/mkldnn/Matmul.h",
    ],
}

SOURCE_FILES = [
    "aten/src/ATen/native/Linear.cpp",
    "aten/src/ATen/native/native_functions.yaml",
    "aten/src/ATen/native/Blas.cpp",
    "aten/src/ATen/native/LinearAlgebra.cpp",
    "aten/src/ATen/native/cpu/BlasKernel.cpp",
    "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp",
    "c10/core/DispatchKey.h",
]

SEARCH_TERMS = [
    "addmm",
    "addmm_impl_cpu_",
    "structured_addmm",
    "ADDMM",
    "mm",
    "baddbmm",
    "linear",
    "mkldnn_linear",
    "mkl",
    "oneDNN",
    "onednn",
    "BFloat16",
    "bf16",
    "GEMM",
    "gemm",
    "brgemm",
    "packed",
    "matmul",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map PyTorch source for fused addmm o-proj seam.")
    parser.add_argument("--source-checkout", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--requested-commit", default=REQUESTED_COMMIT)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "succeeded": completed.returncode == 0,
        }
    except Exception as exc:  # noqa: BLE001 - status preserves command failure
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "succeeded": False,
        }


def base_status(classification: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "source_attribution_probe": True,
        "pytorch_source_map": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "forward_env_path": str(args.forward_env_path),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "source_checkout_path": str(args.source_checkout),
        "source_research_path": str(args.research_path),
        "requested_commit": args.requested_commit,
        "checked_out_commit": None,
        "checkout_match": "unavailable",
        "pytorch_clone_performed": False,
        "pytorch_build_performed": False,
        "pytorch_source_patched": False,
        "submodules_initialized": False,
        **GUARD_FALSE_FLAGS,
    }


def torch_metadata() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch_version": str(torch.__version__),
            "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
            "torch_import_path": str(Path(torch.__file__).resolve()) if getattr(torch, "__file__", None) else None,
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "torch_version": None,
            "torch_git_version": None,
            "torch_import_path": None,
            "torch_cuda_is_available": None,
            "torch_import_error": repr(exc),
        }


def prepare_checkout(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    source = args.source_checkout
    clone_performed = False
    commands: list[dict[str, Any]] = []
    if not source.exists():
        clone = run_cmd(["git", "clone", "--filter=blob:none", GITHUB_URL, str(source)], timeout=7200)
        commands.append(clone)
        clone_performed = clone["succeeded"]
        if not clone["succeeded"]:
            return {"clone_performed": clone_performed, "commands": commands, "error": "clone_failed"}, None
    elif not (source / ".git").exists():
        return {"clone_performed": False, "commands": commands, "error": "source_path_exists_but_is_not_git"}, None

    inside = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], cwd=source)
    commands.append(inside)
    if not inside["succeeded"]:
        return {"clone_performed": clone_performed, "commands": commands, "error": "not_a_git_work_tree"}, None

    # Fetch the exact commit first. If GitHub does not allow direct SHA fetch,
    # fall back to ordinary origin/tags fetch before trying checkout.
    commands.append(run_cmd(["git", "fetch", "--filter=blob:none", "origin", args.requested_commit], cwd=source, timeout=3600))
    rev_parse_requested = run_cmd(["git", "rev-parse", "--verify", f"{args.requested_commit}^{{commit}}"], cwd=source)
    commands.append(rev_parse_requested)
    exact_available = rev_parse_requested["succeeded"]
    if not exact_available:
        commands.append(run_cmd(["git", "fetch", "--filter=blob:none", "--tags", "origin"], cwd=source, timeout=7200))
        rev_parse_requested = run_cmd(["git", "rev-parse", "--verify", f"{args.requested_commit}^{{commit}}"], cwd=source)
        commands.append(rev_parse_requested)
        exact_available = rev_parse_requested["succeeded"]

    checkout_match = "unavailable"
    checkout_target = args.requested_commit
    if exact_available:
        checkout_match = "exact_commit"
    else:
        torch_version = torch_metadata().get("torch_version") or ""
        version_no_local = str(torch_version).split("+", 1)[0]
        tag_candidates = [f"v{version_no_local}", version_no_local, f"release/{'.'.join(version_no_local.split('.')[:2])}"]
        for candidate in tag_candidates:
            rev = run_cmd(["git", "rev-parse", "--verify", f"{candidate}^{{commit}}"], cwd=source)
            commands.append(rev)
            if rev["succeeded"]:
                checkout_target = candidate
                checkout_match = "release_tag"
                break

    checkout = run_cmd(["git", "checkout", "--detach", checkout_target], cwd=source, timeout=3600)
    commands.append(checkout)
    if not checkout["succeeded"]:
        return {
            "clone_performed": clone_performed,
            "commands": commands,
            "error": "checkout_failed",
            "checkout_match": checkout_match,
        }, None

    checked_out = run_cmd(["git", "rev-parse", "HEAD"], cwd=source)
    commands.append(checked_out)
    checked_out_commit = checked_out["stdout"].strip() if checked_out["succeeded"] else None
    if checkout_match == "release_tag" and checked_out_commit == args.requested_commit:
        checkout_match = "exact_commit"
    elif checkout_match == "unavailable" and checked_out_commit:
        checkout_match = "mismatch"

    return {
        "clone_performed": clone_performed,
        "commands": summarize_commands(commands),
        "checkout_match": checkout_match,
    }, checked_out_commit


def summarize_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized = []
    for command in commands:
        summarized.append(
            {
                "cmd": command["cmd"],
                "cwd": command["cwd"],
                "returncode": command["returncode"],
                "succeeded": command["succeeded"],
                "stdout_tail": command.get("stdout", "").splitlines()[-20:],
                "stderr_tail": command.get("stderr", "").splitlines()[-20:],
            }
        )
    return summarized


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def file_existence(source: Path) -> dict[str, Any]:
    files = {}
    for rel in SOURCE_FILES:
        path = source / rel
        files[rel] = {
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
        }
    for rel_dir in ["aten/src/ATen/native/mkldnn", "aten/src/ATen/native/onednn", "aten/src/ATen/native/cpu"]:
        path = source / rel_dir
        files[rel_dir] = {
            "exists": path.is_dir(),
            "file_count": sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0,
        }
    return files


def rg(source: Path, pattern: str, paths: list[str] | None = None, max_lines: int = 250) -> list[str]:
    cmd = ["rg", "-n", "--no-heading", "-i", pattern]
    if paths:
        cmd.extend(paths)
    result = run_cmd(cmd, cwd=source, timeout=300)
    lines = result["stdout"].splitlines()
    return lines[:max_lines]


def write_raw_searches(args: argparse.Namespace, source: Path) -> dict[str, str]:
    args.research_path.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    search_specs = {
        "addmm-grep.txt": r"addmm|addmm_impl_cpu_|structured_addmm|ADDMM",
        "linear-grep.txt": r"linear|mkldnn_linear|_linear",
        "mm-matmul-grep.txt": r"\bmm\b|matmul|baddbmm|_matmul_impl",
        "bf16-backend-grep.txt": r"BFloat16|bf16|oneDNN|onednn|mkldnn|mkl|GEMM|gemm|brgemm|packed",
    }
    for filename, pattern in search_specs.items():
        lines = rg(source, pattern, max_lines=500)
        output_path = args.research_path / filename
        write_text(output_path, "\n".join(lines) + ("\n" if lines else ""))
        outputs[filename] = str(output_path)

    native_functions = source / "aten/src/ATen/native/native_functions.yaml"
    native_text = read_file(native_functions)
    native_lines = []
    if native_text:
        for index, line in enumerate(native_text.splitlines()):
            if re.search(r"\b(addmm|linear|mm|matmul)\b", line):
                start = max(0, index - 3)
                stop = min(len(native_text.splitlines()), index + 8)
                block = native_text.splitlines()[start:stop]
                native_lines.append(f"--- native_functions.yaml:{index + 1} ---")
                native_lines.extend(block)
    native_path = args.research_path / "native-functions-addmm.txt"
    write_text(native_path, "\n".join(native_lines[:500]) + ("\n" if native_lines else ""))
    outputs["native-functions-addmm.txt"] = str(native_path)
    return outputs


def collect_source_matches(source: Path, op_name: str) -> dict[str, Any]:
    keywords = OPS[op_name]
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    preferred_paths = OP_SEARCH_PATHS[op_name]
    lines = rg(source, pattern, preferred_paths, max_lines=300)
    files: dict[str, int] = {}
    bf16_lines = []
    mkldnn_lines = []
    blas_lines = []
    bias_lines = []
    for line in lines:
        file_name = line.split(":", 1)[0]
        files[file_name] = files.get(file_name, 0) + 1
        lower = line.lower()
        if "bfloat16" in lower or "bf16" in lower:
            bf16_lines.append(line)
        if "mkldnn" in lower or "onednn" in lower or "ideep" in lower:
            mkldnn_lines.append(line)
        if "blas" in lower or "gemm" in lower or "mkl" in lower:
            blas_lines.append(line)
        if "bias" in lower or "beta" in lower or "self" in lower:
            bias_lines.append(line)

    native_entries = native_entries_for_op(source, op_name)
    file_names = sorted(files)
    cpu_candidates = [
        name
        for name in file_names
        if (
            name.endswith("Linear.cpp")
            or name.endswith("LinearAlgebra.cpp")
            or name.endswith("Blas.cpp")
            or "/native/cpu/" in name
        )
        and "/cuda/" not in name
        and "/xpu/" not in name
        and "/sparse/" not in name
    ]
    mkldnn_candidates = [
        name
        for name in file_names
        if ("mkldnn" in name.lower() or "onednn" in name.lower()) and "/xpu/" not in name
    ]
    blas_candidates = [
        name
        for name in file_names
        if ("blas" in name.lower() or "linearalgebra" in name.lower() or any("gemm" in line.lower() for line in lines if line.startswith(name + ":")))
        and "/cuda/" not in name
        and "/xpu/" not in name
        and "/sparse/" not in name
    ]
    bf16_visible = bool(bf16_lines) or bool(rg(source, r"BFloat16|bf16", file_names[:20], max_lines=20))
    bias_fused_visible = op_name in {"addmm", "linear"} and bool(bias_lines)
    confidence = "medium" if native_entries and (cpu_candidates or mkldnn_candidates or blas_candidates) else "low"
    return {
        "native_functions_yaml_entries_found": native_entries,
        "dispatch_registrations_found": lines[:80],
        "source_files_containing_implementation_symbols": file_names[:80],
        "cpu_implementation_candidates": cpu_candidates[:40],
        "mkldnn_onednn_implementation_candidates": mkldnn_candidates[:40],
        "blas_mkl_fallback_candidates": blas_candidates[:40],
        "bf16_specific_branches_visible": bf16_visible,
        "bf16_evidence_lines": bf16_lines[:20],
        "bias_fused_path_visible": bias_fused_visible,
        "bias_or_beta_evidence_lines": bias_lines[:20],
        "mkldnn_onednn_evidence_lines": mkldnn_lines[:20],
        "blas_gemm_evidence_lines": blas_lines[:20],
        "confidence": confidence,
    }


def key_source_signals(source: Path) -> dict[str, Any]:
    signal_specs = {
        "linear_2d_bias_routes_to_addmm": (
            "aten/src/ATen/native/Linear.cpp",
            [r"input_dim == 2 && bias->defined", r"return at::addmm\(\*bias, input, weight\.t\(\)\)", r"auto output = at::matmul"],
        ),
        "addmm_native_registration": (
            "aten/src/ATen/native/native_functions.yaml",
            [r"- func: addmm\.out", r"CPU: addmm_out_cpu", r"- func: addmm\("],
        ),
        "addmm_cpu_impl": (
            "aten/src/ATen/native/LinearAlgebra.cpp",
            [r"static void addmm_impl_cpu_", r"TORCH_IMPL_FUNC\(addmm_out_cpu\)", r"_AT_DISPATCH_ADDMM_TYPES"],
        ),
        "mkldnn_bf16_matmul": (
            "aten/src/ATen/native/mkldnn/Matmul.cpp",
            [r"void mkldnn_matmul", r"if \(beta != 0\.0f\).*fuse_sum", r"use_mkldnn_bf16_matmul", r"mkldnn_gemm<c10::BFloat16>"],
        ),
    }
    signals: dict[str, Any] = {}
    for name, (rel_path, patterns) in signal_specs.items():
        path = source / rel_path
        lines = read_file(path).splitlines()
        matches = []
        for index, line in enumerate(lines):
            if any(re.search(pattern, line) for pattern in patterns):
                matches.append({"file": rel_path, "line": index + 1, "text": line.strip()})
        signals[name] = {
            "file": rel_path,
            "exists": path.is_file(),
            "matches": matches[:40],
        }
    return signals


def native_entries_for_op(source: Path, op_name: str) -> list[str]:
    native_path = source / "aten/src/ATen/native/native_functions.yaml"
    lines = read_file(native_path).splitlines()
    if not lines:
        return []
    if op_name == "mm":
        func_patterns = [r"^- func: mm\(", r"^- func: mm\.", r"^- func: _?sparse_mm"]
    else:
        func_patterns = [rf"^- func: {re.escape(op_name)}\(", rf"^- func: {re.escape(op_name)}\."]
    entries: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if any(re.search(pattern, stripped) for pattern in func_patterns):
            block = lines[index : min(len(lines), index + 16)]
            entries.append("\n".join(block))
    return entries[:12]


def interpretation(op_maps: dict[str, Any], checkout_match: str, signals: dict[str, Any]) -> dict[str, Any]:
    linear_map = op_maps.get("linear", {})
    addmm_map = op_maps.get("addmm", {})
    mm_map = op_maps.get("mm", {})
    matmul_map = op_maps.get("matmul", {})
    addmm_cpu = bool(addmm_map.get("cpu_implementation_candidates"))
    addmm_mkldnn = bool(addmm_map.get("mkldnn_onednn_implementation_candidates"))
    addmm_blas = bool(addmm_map.get("blas_mkl_fallback_candidates"))
    linear_mentions_addmm = bool(signals.get("linear_2d_bias_routes_to_addmm", {}).get("matches"))
    addmm_files = set(addmm_map.get("source_files_containing_implementation_symbols", []))
    matmul_files = set(matmul_map.get("source_files_containing_implementation_symbols", []))
    mm_files = set(mm_map.get("source_files_containing_implementation_symbols", []))
    addmm_cpu_impl_visible = bool(signals.get("addmm_cpu_impl", {}).get("matches"))
    mkldnn_bf16_visible = bool(signals.get("mkldnn_bf16_matmul", {}).get("matches"))
    divergence_signal = bool(
        linear_mentions_addmm
        and addmm_cpu_impl_visible
        and (matmul_files or mm_files or matmul_map.get("native_functions_yaml_entries_found"))
    )
    concrete_rule = False
    confidence = "medium" if checkout_match == "exact_commit" and addmm_cpu and (addmm_mkldnn or addmm_blas) else "low"
    return {
        "linear_likely_routes_to_addmm_for_2d_bias": bool(linear_mentions_addmm),
        "addmm_multiple_possible_cpu_mkldnn_paths": bool((addmm_cpu or addmm_cpu_impl_visible) and (addmm_mkldnn or addmm_blas or mkldnn_bf16_visible)),
        "cpu_bf16_addmm_entry_candidates": {
            "cpu_native": addmm_map.get("cpu_implementation_candidates", []),
            "mkldnn_onednn": addmm_map.get("mkldnn_onednn_implementation_candidates", []),
            "blas_mkl": addmm_map.get("blas_mkl_fallback_candidates", []),
        },
        "source_suggests_addmm_and_matmul_diverge_before_bias_add": divergence_signal,
        "bias_fused_source_signal": bool(addmm_map.get("bias_fused_path_visible") or signals.get("mkldnn_bf16_matmul", {}).get("matches")),
        "key_source_signals": signals,
        "concrete_replayable_rule_found": concrete_rule,
        "reopen_rust_policy_synthesis": False,
        "confidence": confidence,
        "narrative": (
            "The source map links the exact Torch wheel commit to native addmm/linear/mm/matmul "
            "registrations and shows CPU plus MKLDNN/oneDNN/BLAS candidate paths for addmm. "
            "It narrows source areas for Stage 3/optional source instrumentation, but does not "
            "identify a single replayable BF16 arithmetic or microkernel rule. Rust/CUDA policy "
            "synthesis should remain closed."
        ),
    }


def build_summary_text(status: dict[str, Any]) -> str:
    lines = [
        f"classification: {status['classification']}",
        f"requested_commit: {status['requested_commit']}",
        f"checked_out_commit: {status.get('checked_out_commit')}",
        f"checkout_match: {status.get('checkout_match')}",
        "",
        "Per-op source mapping:",
    ]
    for op, mapping in status.get("op_source_maps", {}).items():
        lines.extend(
            [
                f"- aten::{op}",
                f"  confidence: {mapping.get('confidence')}",
                f"  native entries: {len(mapping.get('native_functions_yaml_entries_found', []))}",
                f"  cpu candidates: {', '.join(mapping.get('cpu_implementation_candidates', [])[:6]) or 'none'}",
                f"  mkldnn/onednn candidates: {', '.join(mapping.get('mkldnn_onednn_implementation_candidates', [])[:6]) or 'none'}",
                f"  blas/mkl candidates: {', '.join(mapping.get('blas_mkl_fallback_candidates', [])[:6]) or 'none'}",
                f"  bf16 visible: {mapping.get('bf16_specific_branches_visible')}",
                f"  fused bias visible: {mapping.get('bias_fused_path_visible')}",
            ]
        )
    lines.extend(["", "Interpretation:", json.dumps(status.get("source_map_interpretation", {}), indent=2, sort_keys=True)])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.research_path.mkdir(parents=True, exist_ok=True)
    status = base_status("fused_linear_addmm_pytorch_source_map_failed", args)
    status.update(torch_metadata())

    try:
        clone_marker = args.research_path / "pytorch-clone-performed.txt"
        checkout_info, checked_out_commit = prepare_checkout(args)
        if checkout_info.get("clone_performed"):
            write_text(clone_marker, f"PyTorch source cloned by this source-map task into {args.source_checkout}\n")
        status["pytorch_clone_performed"] = bool(checkout_info.get("clone_performed") or clone_marker.exists())
        status["pytorch_clone_performed_this_run"] = bool(checkout_info.get("clone_performed"))
        status["checkout_commands"] = checkout_info.get("commands", [])
        if checked_out_commit is None:
            classification = "fused_linear_addmm_pytorch_source_map_blocked_by_clone"
            status.update(
                {
                    "classification": classification,
                    "checkout_error": checkout_info.get("error"),
                    "checkout_match": checkout_info.get("checkout_match", "unavailable"),
                }
            )
            write_json(args.status_output, status)
            return 0

        checkout_match = checkout_info.get("checkout_match", "mismatch")
        status["checked_out_commit"] = checked_out_commit
        status["checkout_match"] = checkout_match
        status["source_files"] = file_existence(args.source_checkout)
        raw_outputs = write_raw_searches(args, args.source_checkout)
        op_maps = {op: collect_source_matches(args.source_checkout, op) for op in OPS}
        signals = key_source_signals(args.source_checkout)
        status["raw_output_paths"] = raw_outputs
        status["op_source_maps"] = op_maps
        status["key_source_signals"] = signals
        status["source_map_interpretation"] = interpretation(op_maps, checkout_match, signals)
        status["reopen_rust_policy_synthesis"] = False

        classification = (
            "fused_linear_addmm_pytorch_source_map_exact_commit_mapped"
            if checkout_match == "exact_commit"
            else "fused_linear_addmm_pytorch_source_map_release_tag_mapped"
            if checkout_match == "release_tag"
            else "fused_linear_addmm_pytorch_source_map_inconclusive"
        )
        status["classification"] = classification
        write_text(args.research_path / "source-map-summary.txt", build_summary_text(status))
        write_json(args.research_path / "source-map-status.json", status)
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status["classification"] = "fused_linear_addmm_pytorch_source_map_failed"
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
