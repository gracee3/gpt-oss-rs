#!/usr/bin/env python3
"""Lightweight PyTorch source-instrumentation lane controller.

This helper prepares and records the source-instrumentation attribution lane for
the fused-linear/addmm o-proj seam. It prefers a separate PyTorch worktree, does
not touch the dirty main PyTorch checkout, and fails closed when no usable
source build is available. It does not rebuild PyTorch by default.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SAMPLED_LAYERS = [6, 10, 13, 16, 18, 21]
SAMPLED_SHAPE = {
    "input": [1, 4096],
    "weight": [2880, 4096],
    "bias": [2880],
    "dtype": "torch.bfloat16",
    "device": "cpu",
}
DEFAULT_EXPECTED_GIT_VERSION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
TARGET_FILES = [
    "aten/src/ATen/native/Linear.cpp",
    "aten/src/ATen/native/LinearAlgebra.cpp",
    "aten/src/ATen/native/CPUBlas.cpp",
    "aten/src/ATen/native/cpu/BlasKernel.cpp",
    "aten/src/ATen/native/mkldnn/Linear.cpp",
    "aten/src/ATen/native/mkldnn/Matmul.cpp",
]
PATCH_PATH = Path("/tmp/fused_linear_addmm_source_instrumentation.patch")
TRACE_PATH = Path("/tmp/fused_linear_addmm_source_instrumentation_trace.log")
BUILD_RECOMMENDATIONS_PATH = Path(
    "/tmp/fused_linear_addmm_source_instrumentation_build_recommendations.txt"
)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "cmd": cmd,
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc), "cmd": cmd}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def import_torch() -> tuple[Any | None, str | None]:
    try:
        import torch  # type: ignore

        return torch, None
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, str(exc)


def safe_call(fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"error": str(exc)}


def torch_config_show(torch: Any) -> str | None:
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = torch.__config__.show()
        captured = buffer.getvalue()
        if isinstance(result, str) and result:
            return result
        return captured
    except Exception:
        return None


def torch_metadata() -> dict[str, Any]:
    torch, import_error = import_torch()
    data: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
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
            "torch_backends_mkldnn_enabled": getattr(torch.backends.mkldnn, "enabled", None),
            "torch_cuda_available": safe_call(torch.cuda.is_available),
            "cuda_used": False,
        }
    )
    return data


def git_metadata(path: Path, expected_git_version: str | None) -> dict[str, Any]:
    available = path.exists() and path.is_dir()
    data: dict[str, Any] = {
        "path": str(path),
        "available": available,
        "head": None,
        "expected_git_version": expected_git_version,
        "matches_installed_torch": False,
        "dirty": False,
        "status_short": [],
        "remote_url": None,
    }
    if not available:
        return data
    head = run_cmd(["git", "rev-parse", "HEAD"], cwd=path)
    if head["ok"]:
        data["head"] = head["stdout"].strip()
    status = run_cmd(["git", "status", "--short"], cwd=path)
    if status["ok"]:
        lines = [line for line in status["stdout"].splitlines() if line.strip()]
        data["status_short"] = lines
        data["dirty"] = bool(lines)
    remote = run_cmd(["git", "remote", "get-url", "origin"], cwd=path)
    if remote["ok"]:
        data["remote_url"] = remote["stdout"].strip()
    data["matches_installed_torch"] = bool(
        data.get("head") and expected_git_version and data["head"] == expected_git_version
    )
    return data


def ensure_instrumentation_worktree(
    main_path: Path, instrumentation_path: Path, expected_git_version: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(instrumentation_path),
        "created": False,
        "create_attempted": False,
        "create_result": None,
    }
    if instrumentation_path.exists():
        return result
    instrumentation_path.parent.mkdir(parents=True, exist_ok=True)
    result["create_attempted"] = True
    cmd_result = run_cmd(
        ["git", "worktree", "add", str(instrumentation_path), expected_git_version],
        cwd=main_path,
    )
    result["create_result"] = cmd_result
    result["created"] = bool(cmd_result["ok"])
    return result


def has_source_build(instrumentation_path: Path, torch_meta: dict[str, Any]) -> dict[str, Any]:
    torch_file = str(torch_meta.get("torch_file") or "")
    build_dir = instrumentation_path / "build"
    source_build = bool(torch_file and str(instrumentation_path) in torch_file)
    build_artifacts = build_dir.exists() and any(build_dir.iterdir())
    return {
        "usable_source_build": source_build or build_artifacts,
        "imported_torch_path": torch_file,
        "import_path_points_to_instrumentation_worktree": source_build,
        "build_dir_exists": build_dir.exists(),
        "build_dir_nonempty": build_artifacts,
        "source_build_detection_note": (
            "A source build is considered usable only if imported torch comes from "
            "the instrumentation worktree or a nonempty build directory exists."
        ),
    }


def insert_after(text: str, needle: str, insertion: str) -> str:
    if insertion.strip() in text:
        return text
    idx = text.find(needle)
    if idx == -1:
        return text
    idx += len(needle)
    return text[:idx] + insertion + text[idx:]


def replace_once(text: str, old: str, new: str) -> str:
    if new.strip() in text:
        return text
    return text.replace(old, new, 1)


def with_trace_helper(text: str, namespace_marker: str) -> str:
    include_insertion = ""
    if "#include <cstdio>" not in text:
        include_insertion += "#include <cstdio>\n"
    if "#include <cstdlib>" not in text:
        include_insertion += "#include <cstdlib>\n"
    if include_insertion:
        include_pos = text.find("\n\nnamespace ")
        if include_pos != -1:
            text = text[:include_pos] + "\n" + include_insertion + text[include_pos:]
    helper = r'''

static bool fused_linear_addmm_trace_enabled() {
  const char* value = std::getenv("FUSED_LINEAR_ADDMM_TRACE");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static void fused_linear_addmm_trace_marker(const char* marker, const char* detail) {
  if (fused_linear_addmm_trace_enabled()) {
    std::fprintf(stderr, "[fused_linear_addmm_trace] %s %s\n", marker, detail);
  }
}
'''
    if helper.strip() in text:
        return text
    include_pos = text.find("\n\nnamespace ")
    if include_pos != -1:
        return text[:include_pos] + helper + text[include_pos:]
    return insert_after(text, namespace_marker, helper)


def instrument_content(rel: str, text: str) -> str:
    if rel == "aten/src/ATen/native/Linear.cpp":
        text = with_trace_helper(text, "namespace at::native {")
        text = replace_once(
            text,
            "Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {\n",
            "Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {\n"
            '  fused_linear_addmm_trace_marker("FLA_TRACE_LINEAR_ENTRY", "aten_native_linear");\n',
        )
        text = replace_once(
            text,
            "  if (input.is_mkldnn()) {\n    return at::mkldnn_linear(input, weight, *bias);\n  }\n",
            "  if (input.is_mkldnn()) {\n"
            '    fused_linear_addmm_trace_marker("FLA_TRACE_MKLDNN_LINEAR", "linear_input_is_mkldnn");\n'
            "    return at::mkldnn_linear(input, weight, *bias);\n"
            "  }\n",
        )
        text = replace_once(
            text,
            "    // Fused op is marginally faster.\n    return at::addmm(*bias, input, weight.t());\n",
            "    // Fused op is marginally faster.\n"
            '    fused_linear_addmm_trace_marker("FLA_TRACE_LINEAR_ADDMM_ROUTE", "2d_bias_to_addmm_self_bias_beta_default");\n'
            "    return at::addmm(*bias, input, weight.t());\n",
        )
        return text
    if rel == "aten/src/ATen/native/LinearAlgebra.cpp":
        text = with_trace_helper(text, "namespace at::native {")
        text = replace_once(
            text,
            "static void addmm_impl_cpu_(\n",
            'static void addmm_impl_cpu_(\n',
        )
        text = replace_once(
            text,
            "    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {\n",
            "    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {\n"
            '  fused_linear_addmm_trace_marker("FLA_TRACE_ADDMM_IMPL_CPU", "entered_addmm_impl_cpu");\n',
        )
        text = replace_once(
            text,
            "TORCH_IMPL_FUNC(addmm_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor &result) {\n",
            "TORCH_IMPL_FUNC(addmm_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor &result) {\n"
            '  fused_linear_addmm_trace_marker("FLA_TRACE_ADDMM_OUT_CPU", "entered_addmm_out_cpu");\n',
        )
        text = replace_once(
            text,
            "          at::native::cpublas::gemm(\n",
            '          fused_linear_addmm_trace_marker("FLA_TRACE_CPUBLAS_GEMM", "addmm_impl_cpu_calls_cpublas_gemm");\n'
            "          at::native::cpublas::gemm(\n",
        )
        return text
    if rel == "aten/src/ATen/native/CPUBlas.cpp":
        text = with_trace_helper(text, "namespace at::native::cpublas {")
        text = replace_once(
            text,
            "   const at::BFloat16 *a, int64_t lda,\n   const at::BFloat16 *b, int64_t ldb,\n   const double beta,\n   at::BFloat16 *c, int64_t ldc) {\n",
            "   const at::BFloat16 *a, int64_t lda,\n"
            "   const at::BFloat16 *b, int64_t ldb,\n"
            "   const double beta,\n"
            "   at::BFloat16 *c, int64_t ldc) {\n"
            '   fused_linear_addmm_trace_marker("FLA_TRACE_CPUBLAS_GEMM", "bf16_to_bf16_cpublas_entry");\n',
        )
        text = replace_once(
            text,
            "   gemm_stub(\n      at::kCPU, at::kBFloat16,\n",
            '   fused_linear_addmm_trace_marker("FLA_TRACE_GEMM_STUB", "bf16_to_bf16_calls_gemm_stub");\n'
            "   gemm_stub(\n      at::kCPU, at::kBFloat16,\n",
        )
        return text
    if rel == "aten/src/ATen/native/cpu/BlasKernel.cpp":
        text = with_trace_helper(text, "namespace at::native::blas_impl {")
        text = replace_once(
            text,
            "float bf16_dot_with_fp32_arith(\n",
            "float bf16_dot_with_fp32_arith(\n",
        )
        text = replace_once(
            text,
            "float compute_dot(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {\n",
            "float compute_dot(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {\n"
            '  fused_linear_addmm_trace_marker("FLA_TRACE_BF16_VECTOR_PATH", "compute_dot_bf16_fp32_arith");\n',
        )
        text = replace_once(
            text,
            "void cpublas_gemm_impl(\n",
            'void cpublas_gemm_impl(\n',
        )
        text = replace_once(
            text,
            "    at::ScalarType type,\n",
            "    at::ScalarType type,\n",
        )
        return text
    if rel == "aten/src/ATen/native/mkldnn/Linear.cpp":
        text = with_trace_helper(text, "namespace at::native {")
        text = replace_once(
            text,
            "Tensor mkldnn_linear(\n",
            'Tensor mkldnn_linear(\n',
        )
        text = replace_once(
            text,
            "    const std::optional<Tensor>& bias_opt) {\n",
            "    const std::optional<Tensor>& bias_opt) {\n"
            '  fused_linear_addmm_trace_marker("FLA_TRACE_MKLDNN_LINEAR", "entered_mkldnn_linear");\n',
        )
        return text
    if rel == "aten/src/ATen/native/mkldnn/Matmul.cpp":
        text = with_trace_helper(text, "namespace at::native {")
        text = replace_once(
            text,
            "bool mkldnn_matmul(\n",
            "bool mkldnn_matmul(\n",
        )
        text = replace_once(
            text,
            "    const double beta) {\n",
            "    const double beta) {\n"
            '  fused_linear_addmm_trace_marker("FLA_TRACE_MKLDNN_MATMUL", "entered_mkldnn_matmul");\n',
        )
        return text
    return text


def generate_proposed_patch(source_root: Path, patch_path: Path) -> dict[str, Any]:
    hunks: list[str] = []
    files_in_patch = []
    errors = []
    for rel in TARGET_FILES:
        path = source_root / rel
        if not path.exists():
            errors.append({"file": rel, "error": "missing"})
            continue
        try:
            original = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            errors.append({"file": rel, "error": str(exc)})
            continue
        modified = instrument_content(rel, original)
        if modified == original:
            errors.append({"file": rel, "error": "no_instrumentation_pattern_matched"})
            continue
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        hunks.extend(diff)
        files_in_patch.append(rel)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text("".join(hunks), encoding="utf-8")
    return {
        "patch_path": str(patch_path),
        "patch_generated": bool(hunks),
        "files_in_patch": files_in_patch,
        "errors": errors,
        "patch_kind": "proposed_not_applied",
    }


def write_build_recommendations(path: Path, instrumentation_path: Path) -> None:
    text = f"""Fused Linear/AddMM source instrumentation was blocked by no usable source build.

Recommended future setup, only if explicitly approved:

1. Use the clean instrumentation worktree:
   {instrumentation_path}

2. Create or activate a separate PyTorch source-build environment.

3. Apply the proposed patch:
   git -C {instrumentation_path} apply /tmp/fused_linear_addmm_source_instrumentation.patch

4. Build PyTorch using the local project's accepted PyTorch source-build workflow.
   Do not run a long full build from this status-only lane without approval.

5. Run the sampled BF16 CPU shape with:
   FUSED_LINEAR_ADDMM_TRACE=1
   FUSED_LINEAR_ADDMM_TRACE_OUT=/tmp/fused_linear_addmm_source_instrumentation_trace.log

6. Preserve the trace and rerun the gpt-oss source instrumentation helper.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-source-tree", default="/home/emmy/openai/pytorch")
    parser.add_argument(
        "--instrumentation-worktree",
        default="/home/emmy/openai/pytorch-worktrees/fused-linear-addmm-source-instrumentation",
    )
    parser.add_argument(
        "--status-output",
        "--output",
        default="/tmp/fused_linear_addmm_source_instrumentation_status.json",
    )
    parser.add_argument(
        "--source-walk-status",
        default="/tmp/fused_linear_addmm_source_walk_attribution_status.json",
    )
    parser.add_argument(
        "--dispatch-table-status",
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
    args = parser.parse_args()

    source_paths = {
        "source_walk": Path(args.source_walk_status),
        "dispatch_table": Path(args.dispatch_table_status),
        "cpu_producer_attribution": Path(args.cpu_producer_attribution_status),
        "avx2_contract": Path(args.avx2_contract_status),
    }
    missing_required = [str(path) for path in source_paths.values() if not path.exists()]
    torch_meta = torch_metadata()
    expected_git_version = (
        torch_meta.get("torch_git_version")
        or DEFAULT_EXPECTED_GIT_VERSION
    )
    main_path = Path(args.main_source_tree)
    instrumentation_path = Path(args.instrumentation_worktree)
    main_meta = git_metadata(main_path, expected_git_version)

    worktree_setup = {"create_attempted": False, "created": False, "create_result": None}
    if main_meta["available"]:
        worktree_setup = ensure_instrumentation_worktree(
            main_path, instrumentation_path, str(expected_git_version)
        )
    instr_meta = git_metadata(instrumentation_path, expected_git_version)
    build_info = has_source_build(instrumentation_path, torch_meta)
    patch_info = (
        generate_proposed_patch(instrumentation_path, PATCH_PATH)
        if instr_meta["available"]
        else {
            "patch_path": str(PATCH_PATH),
            "patch_generated": False,
            "files_in_patch": [],
            "errors": [{"error": "instrumentation_worktree_unavailable"}],
            "patch_kind": "none",
        }
    )
    write_build_recommendations(BUILD_RECOMMENDATIONS_PATH, instrumentation_path)

    if missing_required:
        classification = "fused_linear_addmm_source_instrumentation_execution_failed"
        next_step = "Restore missing source statuses before retrying instrumentation."
    elif not instr_meta["available"]:
        classification = "fused_linear_addmm_source_instrumentation_blocked_by_dirty_source_tree"
        next_step = "Create a clean PyTorch instrumentation worktree before patching."
    elif instr_meta["dirty"]:
        classification = "fused_linear_addmm_source_instrumentation_blocked_by_dirty_source_tree"
        next_step = "Clean or recreate the separate PyTorch instrumentation worktree before patching."
    elif not build_info["usable_source_build"]:
        classification = "fused_linear_addmm_source_instrumentation_blocked_by_no_source_build"
        next_step = "Prepare an explicit PyTorch source build or editable install, apply the preserved instrumentation patch, then rerun instrumentation."
    else:
        classification = "fused_linear_addmm_source_instrumentation_dispatch_inconclusive"
        next_step = "Run the instrumented source build and inspect trace markers before making a dispatch claim."

    trace_markers: list[str] = []
    trace_summary: dict[str, Any] = {
        "trace_collected": TRACE_PATH.exists(),
        "trace_path": str(TRACE_PATH),
        "marker_count": 0,
    }
    if TRACE_PATH.exists():
        text = TRACE_PATH.read_text(encoding="utf-8", errors="replace")
        for marker in [
            "FLA_TRACE_LINEAR_ENTRY",
            "FLA_TRACE_LINEAR_ADDMM_ROUTE",
            "FLA_TRACE_ADDMM_OUT_CPU",
            "FLA_TRACE_ADDMM_IMPL_CPU",
            "FLA_TRACE_CPUBLAS_GEMM",
            "FLA_TRACE_GEMM_STUB",
            "FLA_TRACE_MKLDNN_LINEAR",
            "FLA_TRACE_MKLDNN_MATMUL",
            "FLA_TRACE_BF16_VECTOR_PATH",
            "FLA_TRACE_BF16_OUTPUT_CAST",
        ]:
            if marker in text:
                trace_markers.append(marker)
        trace_summary["marker_count"] = len(trace_markers)

    status = {
        "classification": classification,
        "validation_only": True,
        "oracle_probe_only": True,
        "source_instrumentation": True,
        "pytorch_patched": False,
        "pytorch_rebuilt": False,
        "source_tree_modified": False,
        "source_tree_modified_path": str(instrumentation_path),
        "main_source_tree_modified": False,
        "runtime_behavior_changed": False,
        "production_routing_changed": False,
        "cuda_kernels_changed": False,
        "backend_selected": False,
        "implementation_authorized": False,
        "consumer_revalidation_authorized": False,
        "operator": "attention_o_proj",
        "sampled_shape": SAMPLED_SHAPE,
        "sampled_layers": SAMPLED_LAYERS,
        "source_statuses": {name: str(path) for name, path in source_paths.items()},
        "missing_required_statuses": missing_required,
        "torch_metadata": torch_meta,
        "pytorch_source": {
            "main_path": str(main_path),
            "instrumentation_worktree": str(instrumentation_path),
            "head": instr_meta.get("head"),
            "expected_git_version": expected_git_version,
            "matches_installed_torch": instr_meta.get("matches_installed_torch"),
            "main_tree_dirty_preexisting": bool(main_meta.get("dirty")),
            "main_tree_status_short": main_meta.get("status_short", []),
            "instrumentation_tree_dirty_after": bool(instr_meta.get("dirty")),
            "instrumentation_tree_status_short": instr_meta.get("status_short", []),
            "worktree_setup": worktree_setup,
        },
        "source_build": build_info,
        "instrumented_files": [] if not build_info["usable_source_build"] else patch_info["files_in_patch"],
        "planned_instrumented_files": patch_info["files_in_patch"],
        "instrumentation_patch": patch_info,
        "build_recommendations_path": str(BUILD_RECOMMENDATIONS_PATH),
        "trace_path": str(TRACE_PATH),
        "trace_markers": trace_markers,
        "trace_path_summary": trace_summary,
        "source_level_dispatch_proven": False,
        "backend_identity_proven": False,
        "avx2_contract_source_confirmed": False,
        "avx2_contract_behaviorally_consistent": True,
        "next_bounded_step": next_step,
        "output_emitted": False,
        "ladder_continued": False,
        "correction_metadata_applied": False,
        "tolerance_pass": False,
        "final_logit_claim": False,
        "all_layer_claim": False,
        "server_claim": False,
        "context_length_claim": False,
    }
    write_json(Path(args.status_output), status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
