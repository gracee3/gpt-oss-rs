#!/usr/bin/env python3
"""Read-only PyTorch source-walk attribution for fused-linear/addmm.

This helper inspects a local PyTorch source checkout and records candidate
source paths for CPU BF16 linear/addmm. It does not modify PyTorch, rebuild
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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SAMPLED_LAYERS = [6, 10, 13, 16, 18, 21]

SEARCH_SYMBOLS = [
    "linear",
    "addmm",
    "aten::linear",
    "aten::addmm",
    "TORCH_LIBRARY_IMPL",
    "m.impl",
    "DispatchKey::CPU",
    "DispatchKey::MkldnnCPU",
    "CompositeImplicitAutograd",
    "CompositeExplicitAutograd",
    "addmm_impl_cpu_",
    "addmm_out_cpu",
    "addmm_cpu",
    "mm_cpu",
    "matmul",
    "linear_out",
    "gemm_stub",
    "addmm_stub",
    "cpublas",
    "cpublas::gemm",
    "gemm",
    "mkldnn_linear",
    "ideep",
    "dnnl",
    "oneDNN",
    "inner_product",
    "BFloat16",
    "bfloat16",
    "vec::Vectorized",
    "VectorizedN",
    "reduce_all",
    "vec_reduce_all",
    "CPU_CAPABILITY",
    "AVX2",
    "AVX512",
    "Vec256",
    "Vec512",
]

SOURCE_SUBTREES = [
    "aten/src/ATen/native",
    "aten/src/ATen/cpu",
    "c10/core",
    "torch/csrc/autograd/generated",
]

CANDIDATE_PATHS = [
    "aten/src/ATen/native/Linear.cpp",
    "aten/src/ATen/native/Blas.cpp",
    "aten/src/ATen/native/CPUBlas.cpp",
    "aten/src/ATen/native/CPUBlas.h",
    "aten/src/ATen/native/LinearAlgebra.cpp",
    "aten/src/ATen/native/cpu/BlasKernel.cpp",
    "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp",
    "aten/src/ATen/native/mkldnn",
    "aten/src/ATen/cpu/vec",
    "aten/src/ATen/native/cpu",
    "aten/src/ATen/native/native_functions.yaml",
    "c10/core",
    "torch/csrc/autograd/generated",
]

CURATED_EVIDENCE = [
    {
        "name": "linear_2d_bias_routes_to_addmm",
        "file": "aten/src/ATen/native/Linear.cpp",
        "patterns": ["input_dim == 2 && bias->defined()", "return at::addmm(*bias, input, weight.t())"],
        "why_it_matters": "For the sampled [1,4096] BF16 input with defined bias, aten::linear plausibly lowers to fused addmm.",
        "confidence": "high",
    },
    {
        "name": "addmm_cpu_registration",
        "file": "aten/src/ATen/native/native_functions.yaml",
        "patterns": ["func: addmm", "CPU: addmm_out_cpu"],
        "why_it_matters": "The native operator registration maps CPU addmm to addmm_out_cpu.",
        "confidence": "high",
    },
    {
        "name": "addmm_out_cpu_calls_impl",
        "file": "aten/src/ATen/native/LinearAlgebra.cpp",
        "patterns": ["TORCH_IMPL_FUNC(addmm_out_cpu)", "addmm_impl_cpu_"],
        "why_it_matters": "The CPU addmm out kernel delegates to addmm_impl_cpu_.",
        "confidence": "high",
    },
    {
        "name": "addmm_impl_calls_cpublas_gemm",
        "file": "aten/src/ATen/native/LinearAlgebra.cpp",
        "patterns": ["cpublas::gemm", "_AT_DISPATCH_ADDMM_TYPES(result.scalar_type()"],
        "why_it_matters": "The addmm implementation contains the cpublas::gemm call site used for CPU GEMM candidates.",
        "confidence": "medium",
    },
    {
        "name": "bf16_cpublas_gemm_path",
        "file": "aten/src/ATen/native/CPUBlas.cpp",
        "patterns": ["const at::BFloat16 *a", "gemm_stub", "at::kBFloat16"],
        "why_it_matters": "The BF16 cpublas GEMM overload can route to gemm_stub for BF16 input/output.",
        "confidence": "medium",
    },
    {
        "name": "gemm_stub_registration",
        "file": "aten/src/ATen/native/cpu/BlasKernel.cpp",
        "patterns": ["REGISTER_DISPATCH(cpublas::gemm_stub", "cpublas_gemm_impl"],
        "why_it_matters": "The CPU-specific BLAS kernel registers cpublas_gemm_impl for gemm_stub.",
        "confidence": "medium",
    },
    {
        "name": "bf16_dot_f32_accumulation",
        "file": "aten/src/ATen/native/cpu/BlasKernel.cpp",
        "patterns": ["bf16_dot_with_fp32_arith", "compute_dot", "gemm_transa_"],
        "why_it_matters": "The BF16 GEMM candidate source includes f32 dot accumulation helpers matching the extracted contract shape.",
        "confidence": "medium",
    },
    {
        "name": "vectorizedn_reduction_helper",
        "file": "aten/src/ATen/cpu/vec/vec_n.h",
        "patterns": ["VectorizedN", "vec_reduce_all"],
        "why_it_matters": "The vector reduction helper is a candidate match for the extracted vectorized reduction contract.",
        "confidence": "medium",
    },
    {
        "name": "mkldnn_linear_alternative",
        "file": "aten/src/ATen/native/mkldnn/Linear.cpp",
        "patterns": ["Tensor mkldnn_linear", "ideep", "inner_product"],
        "why_it_matters": "This is an alternate MKLDNN path, but the sampled dense strided profiler did not expose a deeper MKLDNN backend event.",
        "confidence": "low",
    },
]


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
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}


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
            "torch_config_show": torch_config_show(torch),
            "torch_num_threads": safe_call(torch.get_num_threads),
            "torch_backends_mkldnn_enabled": getattr(torch.backends.mkldnn, "enabled", None),
            "torch_cuda_available": safe_call(torch.cuda.is_available),
            "cuda_used": False,
        }
    )
    return data


def source_tree_metadata(source_root: Path, expected_git_version: str | None) -> dict[str, Any]:
    available = source_root.exists() and source_root.is_dir()
    data: dict[str, Any] = {
        "path": str(source_root),
        "available": available,
        "head": None,
        "expected_git_version": expected_git_version,
        "matches_installed_torch": False,
        "dirty": False,
        "status_short": [],
        "remote_url": None,
        "source_tree_modified": False,
    }
    if not available:
        data["recommended_setup"] = (
            "clone PyTorch source matching torch.version.git_version into "
            f"{source_root}"
        )
        return data
    head = run_cmd(["git", "rev-parse", "HEAD"], cwd=source_root)
    if head["ok"]:
        data["head"] = head["stdout"].strip()
    status = run_cmd(["git", "status", "--short"], cwd=source_root)
    if status["ok"]:
        lines = [line for line in status["stdout"].splitlines() if line.strip()]
        data["status_short"] = lines
        data["dirty"] = bool(lines)
    remote = run_cmd(["git", "remote", "get-url", "origin"], cwd=source_root)
    if remote["ok"]:
        data["remote_url"] = remote["stdout"].strip()
    data["matches_installed_torch"] = bool(
        data.get("head") and expected_git_version and data["head"] == expected_git_version
    )
    return data


def source_paths(source_root: Path) -> list[Path]:
    paths = [source_root / rel for rel in SOURCE_SUBTREES if (source_root / rel).exists()]
    return paths or [source_root]


def parse_rg_line(line: str, source_root: Path) -> dict[str, Any] | None:
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    raw_path, raw_line, text = parts
    try:
        line_no = int(raw_line)
    except ValueError:
        return None
    path = Path(raw_path)
    try:
        rel = str(path.relative_to(source_root))
    except ValueError:
        rel = str(path)
    return {"file": rel, "line": line_no, "snippet": text.strip()}


def rg_search(source_root: Path, symbol: str, max_results: int = 20) -> list[dict[str, Any]]:
    if not shutil.which("rg"):
        return []
    cmd = [
        "rg",
        "-n",
        "--fixed-strings",
        "--no-heading",
        "--glob",
        "*.cpp",
        "--glob",
        "*.h",
        "--glob",
        "*.hpp",
        "--glob",
        "*.yaml",
        "--glob",
        "*.py",
        symbol,
        *[str(path) for path in source_paths(source_root)],
    ]
    result = run_cmd(cmd)
    matches = []
    for line in result["stdout"].splitlines():
        parsed = parse_rg_line(line, source_root)
        if parsed:
            matches.append(parsed)
        if len(matches) >= max_results:
            break
    return matches


def line_snippet(path: Path, line_no: int, context: int = 1) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    start = max(1, line_no - context)
    end = min(len(lines), line_no + context)
    return "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1))


def find_pattern(path: Path, pattern: str) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for idx, line in enumerate(lines, start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": idx,
                "snippet": line_snippet(path, idx, context=1),
            }
    return None


def curated_evidence(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for item in CURATED_EVIDENCE:
        path = source_root / item["file"]
        evidence = []
        for pattern in item["patterns"]:
            found = find_pattern(path, pattern)
            if found:
                evidence.append(found)
        rows.append(
            {
                "name": item["name"],
                "file": item["file"],
                "exists": path.exists(),
                "confidence": item["confidence"],
                "why_it_matters": item["why_it_matters"],
                "matched_patterns": evidence,
                "all_patterns_found": len(evidence) == len(item["patterns"]),
            }
        )
    return rows


def candidate_files(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for rel in CANDIDATE_PATHS:
        path = source_root / rel
        if path.is_dir():
            match_count = 0
            sample_matches = []
            if shutil.which("rg"):
                result = run_cmd(
                    [
                        "rg",
                        "-n",
                        "--fixed-strings",
                        "--no-heading",
                        "--glob",
                        "*.cpp",
                        "--glob",
                        "*.h",
                        "--glob",
                        "*.hpp",
                        "addmm",
                        str(path),
                    ]
                )
                parsed = [
                    parse_rg_line(line, source_root)
                    for line in result["stdout"].splitlines()
                ]
                sample_matches = [row for row in parsed if row][:8]
                match_count = len([row for row in parsed if row])
            rows.append(
                {
                    "path": rel,
                    "kind": "directory",
                    "exists": True,
                    "sample_match_count": match_count,
                    "sample_matches": sample_matches,
                }
            )
        else:
            exists = path.exists()
            rows.append(
                {
                    "path": rel,
                    "kind": "file",
                    "exists": exists,
                    "sample_match_count": None,
                    "sample_matches": [],
                }
            )
    return rows


def source_search(source_root: Path) -> list[dict[str, Any]]:
    return [
        {"symbol": symbol, "matches": rg_search(source_root, symbol)}
        for symbol in SEARCH_SYMBOLS
    ]


def first_evidence(evidence: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in evidence:
        if row["name"] == name:
            return row
    return None


def dispatch_path_graph(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    def node(name: str, summary: str, evidence_name: str | None, confidence: str) -> dict[str, Any]:
        return {
            "name": name,
            "summary": summary,
            "evidence": first_evidence(evidence, evidence_name) if evidence_name else None,
            "confidence": confidence,
        }

    return {
        "python_api": [
            {
                "name": "torch.nn.functional.linear",
                "aten_operator": "aten::linear",
                "confidence": "medium",
            },
            {
                "name": "torch._C._nn.linear",
                "aten_operator": "aten::linear",
                "confidence": "medium",
            },
            {
                "name": "torch.addmm",
                "aten_operator": "aten::addmm",
                "confidence": "high",
            },
        ],
        "aten_operators": [
            node(
                "aten::linear",
                "For 2D input with a defined bias, Linear.cpp routes through at::addmm.",
                "linear_2d_bias_routes_to_addmm",
                "high",
            ),
            node(
                "aten::addmm",
                "native_functions.yaml maps CPU addmm to addmm_out_cpu.",
                "addmm_cpu_registration",
                "high",
            ),
            {
                "name": "aten::mm",
                "summary": "Observed in profiler for explicit matmul negative-control path.",
                "confidence": "medium",
            },
            {
                "name": "aten::matmul",
                "summary": "Observed in profiler for explicit matmul negative-control path.",
                "confidence": "medium",
            },
        ],
        "native_backend_candidates": [
            node(
                "addmm_out_cpu -> addmm_impl_cpu_",
                "CPU addmm delegates to addmm_impl_cpu_.",
                "addmm_out_cpu_calls_impl",
                "high",
            ),
            node(
                "addmm_impl_cpu_ -> cpublas::gemm",
                "The implementation has a cpublas::gemm call site for scalar-dispatched GEMM.",
                "addmm_impl_calls_cpublas_gemm",
                "medium",
            ),
            node(
                "cpublas BF16 GEMM overload",
                "BF16 input/output GEMM can route through mkldnn/BLAS fallbacks or gemm_stub.",
                "bf16_cpublas_gemm_path",
                "medium",
            ),
            node(
                "gemm_stub -> cpublas_gemm_impl",
                "CPU BlasKernel registers cpublas_gemm_impl for gemm_stub.",
                "gemm_stub_registration",
                "medium",
            ),
            node(
                "BF16 dot/f32 accumulation in BlasKernel",
                "The BF16 GEMM candidate includes compute_dot/bf16_dot_with_fp32_arith and gemm_transa_.",
                "bf16_dot_f32_accumulation",
                "medium",
            ),
            node(
                "mkldnn_linear alternative",
                "Potential MKLDNN linear path for mkldnn-layout inputs, not proven for sampled dense strided tensors.",
                "mkldnn_linear_alternative",
                "low",
            ),
        ],
        "graph_conclusion": (
            "The source walk identifies a plausible linear -> addmm -> addmm_impl_cpu_ -> "
            "cpublas::gemm/gemm_stub/BlasKernel path, but it does not prove the runtime-selected "
            "lower-level kernel for the sampled wheel."
        ),
    }


def avx2_candidates(source_root: Path, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = [
        (
            "K=4096",
            "aten/src/ATen/native/LinearAlgebra.cpp",
            "addmm_impl_cpu_",
            "Shape-dependent call path reaches addmm_impl_cpu_; K is runtime metadata rather than a hard-coded source constant.",
            "medium",
        ),
        (
            "BF16 inputs",
            "aten/src/ATen/native/CPUBlas.cpp",
            "const at::BFloat16 *a",
            "BF16 cpublas overloads accept BF16 input pointers.",
            "medium",
        ),
        (
            "f32 accumulation",
            "aten/src/ATen/native/cpu/BlasKernel.cpp",
            "bf16_dot_with_fp32_arith",
            "BF16 dot helpers with fp32 arithmetic appear in BlasKernel.",
            "medium",
        ),
        (
            "fused bias beta=1",
            "aten/src/ATen/native/Linear.cpp",
            "return at::addmm(*bias, input, weight.t())",
            "linear passes bias as addmm self; addmm beta defaults to 1.",
            "medium",
        ),
        (
            "vectorized reduction",
            "aten/src/ATen/cpu/vec/vec_n.h",
            "vec_reduce_all",
            "VectorizedN reduction helper exists and matches the extracted contract vocabulary.",
            "medium",
        ),
        (
            "final BF16 cast",
            "aten/src/ATen/native/CPUBlas.cpp",
            "c10::convert<at::BFloat16>",
            "BF16 conversion appears in CPUBlas fallback code.",
            "low",
        ),
    ]
    rows = []
    for element, file, symbol, why, confidence in mapping:
        matched = [
            row
            for row in evidence
            if row["file"] == file
            and any(symbol in item.get("pattern", "") or symbol in (item.get("snippet") or "") for item in row["matched_patterns"])
        ]
        rows.append(
            {
                "contract_element_matched": element,
                "source_file": file,
                "function_or_symbol": symbol,
                "why_it_matters": why,
                "confidence": confidence,
                "matched": bool(matched) or bool(rg_search(source_root, symbol, max_results=1)),
                "source_level_confirmation": False,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/home/emmy/openai/pytorch")
    parser.add_argument(
        "--status-output",
        "--output",
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
    parser.add_argument(
        "--producer-api-13-16-10-status",
        default="/tmp/o_proj_producer_api_probes_13_16_10_status.json",
    )
    parser.add_argument(
        "--producer-api-18-21-status",
        default="/tmp/o_proj_producer_api_probes_18_21_status.json",
    )
    args = parser.parse_args()

    torch_meta = torch_metadata()
    source_paths = {
        "dispatch_table": Path(args.dispatch_table_status),
        "cpu_producer_attribution": Path(args.cpu_producer_attribution_status),
        "avx2_contract": Path(args.avx2_contract_status),
        "producer_api_13_16_10": Path(args.producer_api_13_16_10_status),
        "producer_api_18_21": Path(args.producer_api_18_21_status),
    }
    missing_required = [str(path) for path in source_paths.values() if not path.exists()]
    source_root = Path(args.source_root)
    source_tree = source_tree_metadata(source_root, torch_meta.get("torch_git_version"))

    dispatch_status = load_json(source_paths["dispatch_table"]) if source_paths["dispatch_table"].exists() else {}
    cpu_status = (
        load_json(source_paths["cpu_producer_attribution"])
        if source_paths["cpu_producer_attribution"].exists()
        else {}
    )

    if not source_tree["available"]:
        searched_symbols: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        graph: dict[str, Any] = {}
        avx2_source_candidates: list[dict[str, Any]] = []
        classification = "fused_linear_addmm_source_walk_attribution_partial_source_tree_missing"
    else:
        searched_symbols = source_search(source_root)
        candidates = curated_evidence(source_root)
        files = candidate_files(source_root)
        graph = dispatch_path_graph(candidates)
        avx2_source_candidates = avx2_candidates(source_root, candidates)
        if missing_required:
            classification = "fused_linear_addmm_source_walk_attribution_inconclusive"
        elif not source_tree.get("matches_installed_torch"):
            classification = "fused_linear_addmm_source_walk_attribution_source_commit_mismatch"
        elif graph:
            classification = "fused_linear_addmm_source_walk_attribution_recorded"
        else:
            classification = "fused_linear_addmm_source_walk_attribution_inconclusive"

    behaviorally_consistent = bool(
        cpu_status.get("avx2_contract_consistency", {}).get("all_layers_consistent")
        or dispatch_status.get("avx2_contract_consistency")
    )
    batch = {
        "classification": classification,
        "validation_only": True,
        "oracle_probe_only": True,
        "read_only": True,
        "source_tree_modified": False,
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
        "torch_metadata": torch_meta,
        "source_tree": source_tree,
        "searched_symbols": searched_symbols,
        "candidate_source_files": files,
        "curated_source_evidence": candidates,
        "dispatch_path_graph": graph,
        "avx2_contract_source_candidates": avx2_source_candidates,
        "source_level_dispatch_proven": False,
        "backend_identity_proven": False,
        "avx2_contract_source_confirmed": False,
        "avx2_contract_behaviorally_consistent": behaviorally_consistent,
        "source_instrumentation_recommended": True,
        "next_bounded_step": "Review source-walk attribution before authorizing lightweight PyTorch source instrumentation.",
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
