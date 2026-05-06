#!/usr/bin/env python3
"""Lower-GEMM attribution for the fused addmm attention o-proj seam.

This source-attribution probe maps PyTorch's CPU addmm implementation down to
cpublas::gemm and gathers read-only wheel/runtime evidence for likely lower
GEMM families. It does not build or patch PyTorch, and it does not reopen
Rust/CUDA policy synthesis without a concrete replayable rule.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from fused_linear_addmm_cpu_producer_attribution import (
    DEFAULT_MODEL,
    GUARD_FALSE_FLAGS,
    SAMPLED_LAYERS,
    assert_cpu_tensor,
    compare_tensors,
    import_torch_and_checkpoint,
    json_tensor_values,
    load_json,
    tensor_metadata,
    write_json,
)


DEFAULT_FORWARD_ENV = Path("/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130")
DEFAULT_SOURCE = Path("/home/emmy/openai/pytorch")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-cpublas-gemm-attribution")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_cpublas_gemm_attribution_status.json")
CHECKED_OUT_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
SAMPLED = [6, 10, 13, 16, 18, 21]
SYMBOL_PATTERN = r"gemm|dnnl|onednn|mkldnn|bf16|bfloat|sbgemm|cpublas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lower-GEMM attribution for fused addmm attention o-proj.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--source-checkout-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--layers", default=",".join(str(layer) for layer in SAMPLED))
    parser.add_argument("--telemetry-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--telemetry-config", default="baseline", help=argparse.SUPPRESS)
    return parser.parse_args()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> dict[str, Any]:
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
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "cwd": str(cwd) if cwd else None, "returncode": None, "stdout": "", "stderr": repr(exc), "succeeded": False}


def tail_lines(text: str, limit: int = 80) -> list[str]:
    return text.splitlines()[-limit:]


def torch_config_show(torch: Any) -> str:
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            torch.__config__.show()
        return stream.getvalue()
    except Exception as exc:  # noqa: BLE001
        return f"<torch.__config__.show failed: {exc!r}>"


def torch_metadata(torch: Any) -> dict[str, Any]:
    interop = None
    try:
        interop = int(torch.get_num_interop_threads())
    except Exception:
        pass
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "sys_prefix": sys.prefix,
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_import_path": str(Path(torch.__file__).resolve()) if getattr(torch, "__file__", None) else None,
        "torch_config_show": torch_config_show(torch),
        "torch_backends_mkldnn_enabled": bool(getattr(torch.backends.mkldnn, "enabled", False)),
        "torch_get_num_threads": int(torch.get_num_threads()),
        "torch_get_num_interop_threads": interop,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
    }


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
    }


def missing_required_paths(model: Path, source: Path, layers: list[int]) -> list[str]:
    paths = [model, source / "aten/src/ATen/native/LinearAlgebra.cpp", source / "aten/src/ATen/native/CPUBlas.cpp"]
    for layer in layers:
        layer_source = layer_paths(layer)
        paths.extend([Path(layer_source["attention_bundle_dir"]), Path(layer_source["weighted_v"]), Path(layer_source["official_o_proj"])])
    return [str(path) for path in paths if not path.exists()]


def load_layer_tensors(torch: Any, checkpoint: Any, layer: int) -> dict[str, Any]:
    paths = layer_paths(layer)
    weighted_v_json = load_json(Path(paths["weighted_v"]))
    official_json = load_json(Path(paths["official_o_proj"]))
    weighted_v = torch.tensor(json_tensor_values(Path(paths["weighted_v"])), dtype=torch.float32, device="cpu").to(torch.bfloat16)
    official = torch.tensor(json_tensor_values(Path(paths["official_o_proj"])), dtype=torch.float32, device="cpu").to(torch.bfloat16)
    weight_name = f"model.layers.{layer}.self_attn.o_proj.weight"
    bias_name = f"model.layers.{layer}.self_attn.o_proj.bias"
    weight = checkpoint.get(weight_name)
    bias = checkpoint.get(bias_name)
    input_2d = weighted_v.unsqueeze(0)
    weight_t_2d = weight.t()
    for name, tensor in {
        "weighted_v": weighted_v,
        "official_o_proj": official,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "input_2d": input_2d,
        "weight_t_2d": weight_t_2d,
    }.items():
        assert_cpu_tensor(f"layer{layer}.{name}", tensor)
    return {
        "paths": paths,
        "weighted_v_json": weighted_v_json,
        "official_json": official_json,
        "weighted_v": weighted_v,
        "official_o_proj": official,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "input_2d": input_2d,
        "weight_t_2d": weight_t_2d,
        "weight_name": weight_name,
        "bias_name": bias_name,
    }


def source_text(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def snippets_for_patterns(source: Path, rel_path: str, patterns: list[str], context: int = 10) -> list[dict[str, Any]]:
    path = source / rel_path
    if not path.is_file():
        return [{"file": rel_path, "exists": False, "snippets": []}]
    lines = source_text(path)
    snippets = []
    for pattern in patterns:
        for index, line in enumerate(lines):
            if pattern.lower() in line.lower():
                start = max(0, index - context)
                stop = min(len(lines), index + context + 1)
                snippets.append(
                    {
                        "pattern": pattern,
                        "file": rel_path,
                        "line": index + 1,
                        "snippet": [
                            {"line": line_no + 1, "text": lines[line_no]} for line_no in range(start, stop)
                        ],
                    }
                )
                break
    return snippets


def collect_source_maps(source: Path) -> dict[str, Any]:
    source_targets = {
        "addmm_source": {
            "aten/src/ATen/native/Linear.cpp": ["input_dim == 2 && bias->defined", "return at::addmm(*bias, input, weight.t())"],
            "aten/src/ATen/native/native_functions.yaml": ["- func: addmm.out", "CPU: addmm_out_cpu", "- func: addmm("],
            "aten/src/ATen/native/LinearAlgebra.cpp": [
                "_AT_DISPATCH_ADDMM_TYPES",
                "static void addmm_impl_cpu_",
                "result.copy_(self)",
                "transpose_c",
                "transpose_a",
                "transpose_b",
                "cpublas::gemm",
                "TORCH_IMPL_FUNC(addmm_out_cpu)",
                "TORCH_IMPL_FUNC(mm_out_cpu)",
            ],
        },
        "cpublas_source": {
            "aten/src/ATen/native/CPUBlas.h": ["DECLARE_DISPATCH(gemm_fn, gemm_stub)", "void gemm(", "BFloat16", "brgemm"],
            "aten/src/ATen/native/CPUBlas.cpp": [
                "BLAS_HAS_SBGEMM",
                "sbgemm_",
                "DEFINE_DISPATCH(gemm_stub)",
                "mkldnn_bf16_gemm",
                "mkldnn_bf16f32_gemm",
                "gemm_stub(",
                "gemm_no_downcast_stub",
                "MKL_HAS_SBGEMM",
                "Brgemm::call",
            ],
        },
        "mkldnn_source": {
            "aten/src/ATen/native/mkldnn/Matmul.cpp": [
                "mkldnn_gemm<c10::BFloat16>",
                "use_mkldnn_bf16_matmul",
                "if (beta != 0.0f) op_attr = ideep::attr_t::fuse_sum()",
                "ideep::matmul_forward::compute",
                "dnnl::matmul",
            ],
            "aten/src/ATen/native/mkldnn/Matmul.h": ["mkldnn_bf16_gemm", "mkldnn_bf16f32_gemm", "use_mkldnn_matmul"],
            "aten/src/ATen/native/mkldnn/MKLDNNCommon.h": ["mkldnn_bf16_device_check", "oneDNN", "ideep"],
        },
    }
    maps: dict[str, Any] = {}
    for group, files in source_targets.items():
        maps[group] = {}
        for rel_path, patterns in files.items():
            maps[group][rel_path] = snippets_for_patterns(source, rel_path, patterns)
    return maps


def grep_source(source: Path, patterns: dict[str, str], research: Path) -> dict[str, str]:
    outputs = {}
    for filename, pattern in patterns.items():
        cmd = ["rg", "-n", "--no-heading", "-i", pattern, "aten/src/ATen/native"]
        result = run_cmd(cmd, cwd=source, timeout=300)
        text = result["stdout"]
        path = research / filename
        write_text(path, "\n".join(text.splitlines()[:700]) + ("\n" if text else ""))
        outputs[filename] = str(path)
    return outputs


def locate_libtorch_cpu(torch: Any) -> dict[str, Any]:
    torch_path = Path(torch.__file__).resolve()
    lib_dir = torch_path.parent / "lib"
    candidates = [lib_dir / "libtorch_cpu.so"]
    found = next((path for path in candidates if path.is_file()), None)
    return {
        "torch_import_path": str(torch_path),
        "torch_lib_dir": str(lib_dir),
        "libtorch_cpu_so": str(found) if found else None,
        "libtorch_cpu_exists": bool(found),
    }


def filter_command(tool: str, lib_path: Path, pattern: str, extra: str = "") -> dict[str, Any]:
    available = shutil.which(tool) is not None
    if not available:
        return {"tool": tool, "available": False, "executed": False, "matches": [], "reason": "tool_not_found"}
    rg_available = shutil.which("rg") is not None
    grep_cmd = f"rg -i {shlex.quote(pattern)}" if rg_available else f"grep -Ei {shlex.quote(pattern)}"
    cmd_text = f"{tool} {extra} {shlex.quote(str(lib_path))} 2>&1 | {grep_cmd} | head -n 240"
    result = run_cmd(["bash", "-lc", cmd_text], timeout=600)
    return {
        "tool": tool,
        "available": True,
        "executed": True,
        "returncode": result["returncode"],
        "matches": result["stdout"].splitlines(),
        "stderr_tail": tail_lines(result["stderr"]),
    }


def inspect_libtorch(torch: Any, research: Path) -> dict[str, Any]:
    location = locate_libtorch_cpu(torch)
    if not location["libtorch_cpu_so"]:
        return {**location, "tools": {}, "linked_libraries": [], "symbol_evidence_confidence": "low"}
    lib = Path(location["libtorch_cpu_so"])
    tools = {
        "ldd": filter_command("ldd", lib, r"mkl|dnnl|onednn|mkldnn|blas|iomp|gomp"),
        "readelf": filter_command("readelf", lib, SYMBOL_PATTERN, "-Ws"),
        "nm": filter_command("nm", lib, SYMBOL_PATTERN, "-D"),
        "strings": filter_command("strings", lib, SYMBOL_PATTERN),
    }
    linked = tools["ldd"].get("matches", [])
    evidence_lines = []
    for result in tools.values():
        evidence_lines.extend(result.get("matches", []))
    confidence = "medium" if evidence_lines else "low"
    summary = {**location, "tools": tools, "linked_libraries": linked, "symbol_evidence_confidence": confidence}
    write_json(research / "libtorch-symbols-summary.json", summary)
    write_text(
        research / "libtorch-symbols-summary.txt",
        "\n".join(
            [
                f"libtorch_cpu.so: {location['libtorch_cpu_so']}",
                "",
                "linked libraries:",
                *linked,
                "",
                "symbol evidence:",
                *evidence_lines[:400],
            ]
        )
        + "\n",
    )
    return summary


def derive_source_branch_inputs(torch: Any, tensors: dict[str, Any], addmm_output: Any) -> dict[str, Any]:
    input_2d = tensors["input_2d"]
    weight_t = tensors["weight_t_2d"]
    bias = tensors["o_proj_bias"]
    result = addmm_output
    m1_sizes = list(input_2d.shape)
    m2_sizes = list(weight_t.shape)
    result_sizes = list(result.shape)
    result_strides = list(result.stride())
    m1_strides = list(input_2d.stride())
    m2_strides = list(weight_t.stride())

    transpose_c = False
    c_source = "result"
    m1_swapped = False
    if result_strides[0] == 1 and (result_sizes[1] == 1 or result_strides[1] >= max(1, result_sizes[0])):
        transpose_c = False
    elif result_strides[1] == 1 and (result_sizes[0] == 1 or result_strides[0] >= max(1, result_sizes[1])):
        transpose_c = True
        m1_sizes, m2_sizes = m2_sizes, m1_sizes
        m1_strides, m2_strides = m2_strides, m1_strides
        m1_swapped = True
    else:
        c_source = "fortran_contiguous_copy"

    m = result_sizes[1 if transpose_c else 0]
    n = result_sizes[0 if transpose_c else 1]
    k = m1_sizes[0 if transpose_c else 1]

    def cast_matrix(strides: list[int], sizes: list[int], first_extent: int, second_extent: int) -> dict[str, Any]:
        if strides[1 if transpose_c else 0] == 1 and strides[0 if transpose_c else 1] >= max(1, first_extent):
            return {"transpose": False, "source": "resolve_conj"}
        if strides[0 if transpose_c else 1] == 1 and strides[1 if transpose_c else 0] >= max(1, second_extent):
            return {"transpose": True, "source": "view"}
        return {"transpose": not transpose_c, "source": "contiguous_clone"}

    a = cast_matrix(m1_strides, m1_sizes, m, k)
    b = cast_matrix(m2_strides, m2_sizes, k, n)
    lda_index = 1 if a["transpose"] == transpose_c else 0
    ldb_index = 1 if b["transpose"] == transpose_c else 0
    ldc_index = 0 if transpose_c else 1
    return {
        "gemm_dimensions": {"m": int(m), "n": int(n), "k": int(k)},
        "result_shape": list(result.shape),
        "result_stride": list(result.stride()),
        "bias_expansion_needed": list(bias.shape) != list(result.shape),
        "bias_shape": list(bias.shape),
        "bias_expanded_shape": list(result.shape),
        "transpose_c": bool(transpose_c),
        "transpose_a": bool(a["transpose"]),
        "transpose_b": bool(b["transpose"]),
        "m1_m2_swapped_by_transpose_c": bool(m1_swapped),
        "a_source": a["source"],
        "b_source": b["source"],
        "c_source": c_source,
        "lda": int(m1_strides[lda_index]) if lda_index < len(m1_strides) else None,
        "ldb": int(m2_strides[ldb_index]) if ldb_index < len(m2_strides) else None,
        "ldc": int(result_strides[ldc_index]) if ldc_index < len(result_strides) else None,
        "weight_t_is_view": bool(weight_t._base is not None),
        "blas_compatible_by_source_logic": a["source"] != "contiguous_clone" and b["source"] != "contiguous_clone",
        "result_expected_contiguous": bool(result.is_contiguous()),
        "tensor_metadata": {
            "input_2d": tensor_metadata(torch, input_2d, include_summary=False),
            "weight": tensor_metadata(torch, tensors["o_proj_weight"], include_summary=False),
            "weight_t": tensor_metadata(torch, weight_t, include_summary=False),
            "bias": tensor_metadata(torch, bias, include_summary=False),
            "official_o_proj": tensor_metadata(torch, tensors["official_o_proj"], include_summary=False),
            "addmm_output": tensor_metadata(torch, result, include_summary=False),
        },
    }


def run_layer_replay(torch: Any, checkpoint: Any, layer: int) -> dict[str, Any]:
    focus_lane = int(SAMPLED_LAYERS[layer]["focus_lane"])
    tensors = load_layer_tensors(torch, checkpoint, layer)
    output = torch.addmm(tensors["o_proj_bias"], tensors["input_2d"], tensors["weight_t_2d"])
    assert_cpu_tensor(f"layer{layer}.addmm_output", output)
    comparison = compare_tensors(torch, output.squeeze(0), tensors["official_o_proj"], focus_lane)
    return {
        "layer_index": layer,
        "role": SAMPLED_LAYERS[layer]["role"],
        "source_artifacts": tensors["paths"],
        "model_tensors_loaded": [tensors["weight_name"], tensors["bias_name"]],
        "comparison_vs_official": comparison,
        "full_vector_cleared": comparison["full_vector_cleared"],
        "branch_inputs": derive_source_branch_inputs(torch, tensors, output),
    }


def run_runtime_replay(args: argparse.Namespace, layers: list[int]) -> dict[str, Any]:
    torch, Checkpoint = import_torch_and_checkpoint()
    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    layer_results = [run_layer_replay(torch, checkpoint, layer) for layer in layers]
    return {
        "environment": torch_metadata(torch),
        "sampled_layers_evaluated": [layer["layer_index"] for layer in layer_results],
        "layers": layer_results,
        "all_layers_full_vector_clear": all(layer["full_vector_cleared"] for layer in layer_results),
        "cuda_used": False,
    }


def telemetry_child(args: argparse.Namespace) -> int:
    torch, Checkpoint = import_torch_and_checkpoint()
    if args.telemetry_config == "mkldnn_false":
        torch.backends.mkldnn.enabled = False
    elif args.telemetry_config == "mkldnn_true":
        torch.backends.mkldnn.enabled = True
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    results = []
    for layer in layers:
        result = run_layer_replay(torch, checkpoint, layer)
        results.append({"layer_index": layer, "full_vector_cleared": result["full_vector_cleared"]})
    print(json.dumps({"config": args.telemetry_config, "mkldnn_enabled": bool(torch.backends.mkldnn.enabled), "layers": results, "cuda_used": False}))
    return 0


def telemetry_variants(args: argparse.Namespace) -> dict[str, Any]:
    variants = {
        "baseline": ({}, "baseline"),
        "mkldnn_false": ({}, "mkldnn_false"),
        "mkldnn_true": ({}, "mkldnn_true"),
        "ONEDNN_VERBOSE_all": ({"ONEDNN_VERBOSE": "all"}, "baseline"),
        "DNNL_VERBOSE_all": ({"DNNL_VERBOSE": "all"}, "baseline"),
        "MKL_VERBOSE_1": ({"MKL_VERBOSE": "1"}, "baseline"),
        "ATEN_CPU_CAPABILITY_default": ({"ATEN_CPU_CAPABILITY": "default"}, "baseline"),
        "ONEDNN_MAX_CPU_ISA_AVX2": ({"ONEDNN_MAX_CPU_ISA": "AVX2"}, "baseline"),
        "ONEDNN_MAX_CPU_ISA_AVX512_CORE_BF16": ({"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}, "baseline"),
    }
    results = {}
    for name, (env_updates, config) in variants.items():
        env = os.environ.copy()
        env.update(env_updates)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--telemetry-child",
            "--telemetry-config",
            config,
            "--model",
            str(args.model),
            "--layers",
            args.layers,
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=240, check=False)
        combined = "\n".join(completed.stdout.splitlines() + completed.stderr.splitlines()).lower()
        parsed = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        results[name] = {
            "env_updates": env_updates,
            "telemetry_config": config,
            "returncode": completed.returncode,
            "stdout_tail": tail_lines(completed.stdout),
            "stderr_tail": tail_lines(completed.stderr),
            "parsed_result": parsed,
            "all_layers_full_vector_clear": bool(parsed and all(layer["full_vector_cleared"] for layer in parsed.get("layers", []))),
            "mkldnn_onednn_signal": "onednn" in combined or "dnnl" in combined or "mkldnn" in combined,
            "mkl_blas_signal": "mkl" in combined or "sgemm" in combined or "sbgemm" in combined,
            "unsupported_or_skipped": completed.returncode != 0,
        }
    return results


def infer_backend(source_maps: dict[str, Any], symbols: dict[str, Any], telemetry: dict[str, Any]) -> tuple[str, str]:
    symbol_text = json.dumps(symbols).lower()
    telemetry_text = json.dumps(telemetry).lower()
    source_text_blob = json.dumps(source_maps).lower()
    mkldnn_source = "mkldnn_bf16_gemm" in source_text_blob or "mkldnn_bf16_matmul" in source_text_blob
    cpublas_source = "cpublas::gemm" in source_text_blob or "gemm_stub" in source_text_blob
    mkl_symbol = "mkl" in symbol_text or "sbgemm" in symbol_text
    dnnl_symbol = "dnnl" in symbol_text or "onednn" in symbol_text or "mkldnn" in symbol_text
    mkl_runtime = any(item.get("mkl_blas_signal") for item in telemetry.values())
    dnnl_runtime = any(item.get("mkldnn_onednn_signal") for item in telemetry.values())
    if dnnl_runtime and not mkl_runtime:
        return "mkldnn_onednn_likely", "medium"
    if mkl_runtime and not dnnl_runtime:
        return "mkl_blas_likely", "medium"
    if (mkl_symbol or dnnl_symbol) and cpublas_source and mkldnn_source:
        return "multiple_possible", "medium"
    if cpublas_source:
        return "native_cpublas_likely", "low"
    return "inconclusive", "low"


def base_status(classification: str, args: argparse.Namespace, layers: list[int]) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "source_attribution_probe": True,
        "cpublas_gemm_attribution": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "forward_env_path": str(args.forward_env_path),
        "source_checkout_path": str(args.source_checkout_path),
        "checked_out_commit": CHECKED_OUT_COMMIT,
        "sampled_layers_requested": layers,
        "sampled_layers_evaluated": [],
        "full_model_loaded": False,
        "gpu_tensors_created": False,
        "pytorch_build_performed": False,
        "pytorch_source_patched": False,
        "active_backend_inference": "inconclusive",
        "active_backend_confidence": "low",
        "concrete_replayable_rule_found": False,
        "replayable_rule_summary": None,
        "reopen_rust_policy_synthesis": False,
        "tolerance_pass": False,
        "correction_metadata_applied": False,
        **GUARD_FALSE_FLAGS,
    }


def write_research_outputs(args: argparse.Namespace, status: dict[str, Any]) -> None:
    args.research_path.mkdir(parents=True, exist_ok=True)
    snippets = status["source_snippets"]
    write_text(args.research_path / "addmm-impl-snippet.txt", json.dumps(snippets["addmm_source"], indent=2, sort_keys=True) + "\n")
    write_text(args.research_path / "cpublas-gemm-source-map.txt", json.dumps(snippets["cpublas_source"], indent=2, sort_keys=True) + "\n")
    write_text(args.research_path / "mkldnn-gemm-source-map.txt", json.dumps(snippets["mkldnn_source"], indent=2, sort_keys=True) + "\n")
    write_json(args.research_path / "runtime-branch-inputs.json", status["runtime_branch_inputs"])
    write_json(args.research_path / "runtime-telemetry-summary.json", status["runtime_telemetry_summary"])
    write_text(
        args.research_path / "interpretation-summary.txt",
        json.dumps(
            {
                "active_backend_inference": status["active_backend_inference"],
                "active_backend_confidence": status["active_backend_confidence"],
                "concrete_replayable_rule_found": status["concrete_replayable_rule_found"],
                "reopen_rust_policy_synthesis": status["reopen_rust_policy_synthesis"],
                "interpretation": status["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if args.telemetry_child:
        return telemetry_child(args)

    args.research_path.mkdir(parents=True, exist_ok=True)
    missing = missing_required_paths(args.model, args.source_checkout_path, layers)
    if missing:
        status = base_status("fused_linear_addmm_cpublas_gemm_blocked_by_missing_source", args, layers)
        status["missing_required_paths"] = missing
        write_json(args.status_output, status)
        return 0

    try:
        torch, _Checkpoint = import_torch_and_checkpoint()
        source_commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=args.source_checkout_path)
        checked_out_commit = source_commit["stdout"].strip() if source_commit["succeeded"] else None
        metadata = torch_metadata(torch)
        source_maps = collect_source_maps(args.source_checkout_path)
        grep_outputs = grep_source(
            args.source_checkout_path,
            {
                "addmm-source-grep.txt": r"addmm_impl_cpu_|addmm_out_cpu|cpublas::gemm|transpose_a|transpose_b|transpose_c|result.copy_|_AT_DISPATCH_ADDMM_TYPES",
                "cpublas-gemm-grep.txt": r"cpublas::gemm|gemm_stub|gemm_no_downcast|mkldnn_bf16_gemm|mkldnn_bf16f32_gemm|BLAS_HAS_SBGEMM|MKL_HAS_SBGEMM|sbgemm|BFloat16",
                "mkldnn-gemm-grep.txt": r"use_mkldnn_bf16_matmul|mkldnn_gemm|fuse_sum|ideep::matmul_forward::compute|dnnl::matmul|beta",
            },
            args.research_path,
        )
        symbols = inspect_libtorch(torch, args.research_path)
        runtime_replay = run_runtime_replay(args, layers)
        telemetry = telemetry_variants(args)
        active_backend, confidence = infer_backend(source_maps, symbols, telemetry)
        addmm_source_chain = bool(
            source_maps["addmm_source"]["aten/src/ATen/native/Linear.cpp"]
            and source_maps["addmm_source"]["aten/src/ATen/native/LinearAlgebra.cpp"]
            and source_maps["addmm_source"]["aten/src/ATen/native/native_functions.yaml"]
        )
        cpublas_callsite = "cpublas::gemm" in json.dumps(source_maps["addmm_source"])
        lower_candidates = {
            "native_cpu": bool(cpublas_callsite),
            "mkl_blas": "sbgemm" in json.dumps(source_maps).lower() or any(item.get("mkl_blas_signal") for item in telemetry.values()),
            "mkldnn_onednn": "mkldnn_bf16" in json.dumps(source_maps).lower() or any(item.get("mkldnn_onednn_signal") for item in telemetry.values()),
            "unknown": True,
        }
        classification = (
            "fused_linear_addmm_cpublas_gemm_backend_likely_identified"
            if active_backend in {"native_cpublas_likely", "mkl_blas_likely", "mkldnn_onednn_likely"}
            else "fused_linear_addmm_cpublas_gemm_attribution_recorded"
            if active_backend == "multiple_possible"
            else "fused_linear_addmm_cpublas_gemm_inconclusive"
        )
        status = base_status(classification, args, layers)
        status.update(
            {
                **metadata,
                "source_checkout_commit_observed": checked_out_commit,
                "sampled_layers_evaluated": runtime_replay["sampled_layers_evaluated"],
                "addmm_source_chain_confirmed": bool(addmm_source_chain),
                "cpublas_gemm_callsite_confirmed": bool(cpublas_callsite),
                "lower_gemm_candidates": lower_candidates,
                "active_backend_inference": active_backend,
                "active_backend_confidence": confidence,
                "concrete_replayable_rule_found": False,
                "replayable_rule_summary": None,
                "reopen_rust_policy_synthesis": False,
                "source_snippets": source_maps,
                "raw_source_output_paths": grep_outputs,
                "binary_symbol_inspection": symbols,
                "runtime_branch_inputs": runtime_replay,
                "runtime_telemetry_summary": telemetry,
                "interpretation": {
                    "source_chain": "linear 2D+bias -> addmm -> addmm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm",
                    "runtime_outputs_changed_under_any_tested_setting": not all(item.get("all_layers_full_vector_clear") for item in telemetry.values()),
                    "source_and_symbols_identify_replayable_rule": False,
                    "likely_backend_is_not_enough_to_reopen_policy": True,
                    "next_bounded_step": "Preserve official Torch API seam; only reopen Rust/CUDA policy if a concrete global replayable rule is identified.",
                },
            }
        )
        write_research_outputs(args, status)
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = base_status("fused_linear_addmm_cpublas_gemm_failed", args, layers)
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
