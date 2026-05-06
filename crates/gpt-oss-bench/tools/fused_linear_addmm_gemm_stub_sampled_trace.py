#!/usr/bin/env python3
"""Sampled Workstream A GEMM-stub trace using instrumented CPU PyTorch.

This helper is source-attribution only. It archives the current external
PyTorch instrumentation diff, reuses the existing CPU-only source build, and
runs captured-tensor addmm traces for sampled o-proj layers. It does not load
the full model, run model forward, use CUDA, or implement a Rust/CUDA policy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path("/home/emmy/openai/pytorch")
DEFAULT_BUILD_ENV = Path("/home/emmy/openai/.venvs/pytorch-src-cpu")
DEFAULT_MODEL = Path("/data/models/openai/gpt-oss-20b-full-attn-restricted-integration")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-sampled-trace")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_gemm_stub_sampled_trace_status.json")
EXPECTED_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
PRE_EXISTING_PATCH = Path(
    "/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-dispatch-internals/"
    "pre_gemm_stub_internals.patch"
)
SAMPLED_LAYERS = [6, 10, 13, 16, 18, 21]
FOCUS_LANES = {6: 22, 10: 915, 13: 151, 16: 2666, 18: 63, 21: 2807}
REQUIRED_CONFIGS = ["baseline", "default"]
OPTIONAL_CONFIGS = ["avx2", "avx512", "avx512_bf16", "avx512_vnni"]
TRACE_ENV_VARS = {
    "GPT_OSS_TRACE_ADDMM": "1",
    "GPT_OSS_TRACE_GEMM_STUB": "1",
}
INSTRUMENTED_SOURCE_FILES = [
    "aten/src/ATen/native/CPUBlas.cpp",
    "aten/src/ATen/native/Linear.cpp",
    "aten/src/ATen/native/LinearAlgebra.cpp",
    "aten/src/ATen/native/cpu/BlasKernel.cpp",
    "aten/src/ATen/native/mkldnn/Matmul.cpp",
]
FALSE_GUARDS = {
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
    "full_model_loaded": False,
    "model_forward_run": False,
    "gpu_tensors_created": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sampled GEMM-stub trace for fused addmm.")
    parser.add_argument("--source-checkout-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--build-env-path", type=Path, default=DEFAULT_BUILD_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--layer", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--lane", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--config", default="baseline", help=argparse.SUPPRESS)
    parser.add_argument("--full-run", action="store_true", help=argparse.SUPPRESS)
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
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
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
        if stdout_path:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        if stderr_path:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "succeeded": completed.returncode == 0,
            "stdout_tail": completed.stdout.splitlines()[-80:],
            "stderr_tail": completed.stderr.splitlines()[-80:],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "succeeded": False,
            "stdout_tail": [],
            "stderr_tail": [repr(exc)],
            "error": repr(exc),
        }


def json_tensor_values(path: Path) -> list[float]:
    data = read_json(path)
    values = data.get("values")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain a values array")
    return [float(value) for value in values]


def source_state(source: Path) -> dict[str, Any]:
    head = run_cmd(["git", "rev-parse", "HEAD"], cwd=source)
    status = run_cmd(["git", "status", "--short"], cwd=source)
    diff_files = run_cmd(["git", "diff", "--name-only"], cwd=source)
    head_value = head.get("stdout_tail", [""])[-1] if head.get("stdout_tail") else ""
    return {
        "head": head_value,
        "expected_commit_match": head_value == EXPECTED_COMMIT,
        "status_short": status.get("stdout_tail", []),
        "dirty_files": diff_files.get("stdout_tail", []),
    }


def archive_patch(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(["git", "diff"], cwd=source, timeout=300)
    text = "\n".join(result.get("stdout_tail", []))
    # Preserve the complete diff, not only the summarized tail.
    completed = subprocess.run(["git", "diff"], cwd=str(source), text=True, capture_output=True, check=False)
    output.write_text(completed.stdout, encoding="utf-8", errors="replace")
    return {
        "path": str(output),
        "succeeded": completed.returncode == 0,
        "size_bytes": output.stat().st_size if output.is_file() else 0,
        "summary_tail": text,
    }


def collect_lanes_from_status(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "available": path.is_file(),
        "residual_lanes_by_layer": {str(layer): [] for layer in SAMPLED_LAYERS},
        "missing_reason": None,
    }
    if not path.is_file():
        result["missing_reason"] = "missing_status_file"
        return result
    try:
        data = read_json(path)
    except Exception as exc:  # noqa: BLE001
        result["missing_reason"] = repr(exc)
        return result

    lanes = {layer: set() for layer in SAMPLED_LAYERS}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            layer = value.get("layer_index", value.get("layer"))
            if layer in lanes:
                for key in ("first_mismatch", "worst_mismatch", "focus_lane"):
                    item = value.get(key)
                    if isinstance(item, dict):
                        lane = item.get("hidden_lane", item.get("index"))
                        if isinstance(lane, int):
                            lanes[layer].add(lane)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    result["residual_lanes_by_layer"] = {str(layer): sorted(values) for layer, values in lanes.items()}
    return result


def build_lane_plan() -> dict[str, Any]:
    sources = [
        collect_lanes_from_status(Path("/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json")),
        collect_lanes_from_status(Path("/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json")),
    ]
    residuals = {layer: set() for layer in SAMPLED_LAYERS}
    for source in sources:
        for layer, values in source.get("residual_lanes_by_layer", {}).items():
            if int(layer) in residuals:
                residuals[int(layer)].update(int(value) for value in values)
    residuals[18].add(1641)
    focus = {str(layer): [lane] for layer, lane in FOCUS_LANES.items()}
    lane_plan = {
        "residual_lane_sources": sources,
        "residual_lanes_by_layer": {str(layer): sorted(values) for layer, values in residuals.items()},
        "focus_lanes_by_layer": focus,
        "missing_lane_metadata": [
            source for source in sources if not source.get("available") or source.get("missing_reason")
        ],
    }
    all_lanes = {}
    for layer in SAMPLED_LAYERS:
        merged = set(residuals[layer])
        merged.add(FOCUS_LANES[layer])
        all_lanes[str(layer)] = sorted(merged)
    lane_plan["trace_lanes_by_layer"] = all_lanes
    return lane_plan


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
            "hidden_lane": first_idx,
            "actual": float(actual_cpu[first_idx].float().item()),
            "expected": float(expected_cpu[first_idx].float().item()),
            "abs_diff": float(diff[first_idx].item()),
        }
        worst = {
            "hidden_lane": worst_idx,
            "actual": float(actual_cpu[worst_idx].float().item()),
            "expected": float(expected_cpu[worst_idx].float().item()),
            "abs_diff": float(diff[worst_idx].item()),
        }
    return {
        "full_vector_mismatches": mismatches,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "first_mismatch": first,
        "worst_mismatch": worst,
        "focus_lane": {
            "hidden_lane": focus_lane,
            "actual": float(actual_cpu[focus_lane].float().item()),
            "expected": float(expected_cpu[focus_lane].float().item()),
            "abs_diff": float(diff[focus_lane].item()),
            "cleared": float(diff[focus_lane].item()) == 0.0,
            "diagnostic_only": True,
        },
        "full_vector_cleared": mismatches == 0 and (float(diff.max().item()) if diff.numel() else 0.0) == 0.0,
    }


def bf16_neighbors(torch: Any, value: Any) -> dict[str, Any]:
    try:
        tensor = torch.tensor(float(value), dtype=torch.bfloat16, device="cpu")
        down = torch.nextafter(tensor, torch.tensor(float("-inf"), dtype=torch.bfloat16, device="cpu"))
        up = torch.nextafter(tensor, torch.tensor(float("inf"), dtype=torch.bfloat16, device="cpu"))
        return {
            "value": float(tensor.float().item()),
            "previous": float(down.float().item()),
            "next": float(up.float().item()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def runtime_child(args: argparse.Namespace) -> int:
    import torch

    gpt_oss_root = REPO_ROOT.parents[1] / "gpt-oss"
    if (gpt_oss_root / "gpt_oss").is_dir():
        sys.path.insert(0, str(gpt_oss_root))
    from gpt_oss.torch.weights import Checkpoint

    if args.layer not in SAMPLED_LAYERS:
        raise ValueError(f"unexpected layer {args.layer}")
    if args.lane is None:
        raise ValueError("--lane is required for child mode")

    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    weighted = torch.tensor(
        json_tensor_values(Path(f"/tmp/layer{args.layer}_ordered_attention_bundle/weighted_v.json")),
        dtype=torch.float32,
        device="cpu",
    ).to(torch.bfloat16)
    official = torch.tensor(
        json_tensor_values(Path(f"/tmp/layer{args.layer}_ordered_attention_bundle/o_proj.json")),
        dtype=torch.float32,
        device="cpu",
    ).to(torch.bfloat16)
    weight = checkpoint.get(f"model.layers.{args.layer}.self_attn.o_proj.weight")
    bias = checkpoint.get(f"model.layers.{args.layer}.self_attn.o_proj.bias")
    for name, tensor in {
        "weighted_v": weighted,
        "official_o_proj": official,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
    }.items():
        if str(tensor.device) != "cpu":
            raise RuntimeError(f"{name} unexpectedly landed on {tensor.device}")

    input_2d = weighted.unsqueeze(0)
    weight_t = weight.t()
    zero_bias = torch.zeros_like(bias)
    addmm = torch.addmm(bias, input_2d, weight_t).squeeze(0)
    result: dict[str, Any] = {
        "layer": args.layer,
        "lane": args.lane,
        "config": args.config,
        "full_run": bool(args.full_run),
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_import_path": str(Path(torch.__file__).resolve()),
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_used": False,
        "official_lane": float(official[args.lane].float().item()),
        "official_lane_neighbors": bf16_neighbors(torch, official[args.lane].float().item()),
        "torch_addmm_lane": float(addmm[args.lane].float().item()),
        "torch_addmm_vs_official": compare_tensors(torch, addmm, official, args.lane),
        "variants": {},
    }
    if args.full_run:
        variants = {
            "torch_addmm": addmm,
            "torch_nn_functional_linear": torch.nn.functional.linear(input_2d, weight, bias).squeeze(0),
            "torch_C_nn_linear": torch._C._nn.linear(input_2d, weight, bias).squeeze(0),
            "zero_bias_addmm_plus_bias": torch.addmm(zero_bias, input_2d, weight_t).squeeze(0) + bias,
            "explicit_matmul_plus_bias": (input_2d @ weight_t).squeeze(0) + bias,
            "explicit_einsum_plus_bias": torch.einsum("bk,hk->bh", input_2d, weight).squeeze(0) + bias,
        }
        for name, output in variants.items():
            if str(output.device) != "cpu":
                raise RuntimeError(f"{name} unexpectedly landed on {output.device}")
            result["variants"][name] = {
                "dtype": str(output.dtype),
                "device": str(output.device),
                "shape": list(output.shape),
                "lane_value": float(output[args.lane].float().item()),
                "comparison_vs_official": compare_tensors(torch, output, official, args.lane),
            }
    if args.child_output:
        write_json(args.child_output, result)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


CALLSITE_RE = re.compile(
    r"callsite=bf16_to_bf16 runtime_capability=(?P<runtime>\S+) "
    r"env_ATEN_CPU_CAPABILITY=(?P<env>\S*) DEFAULT=(?P<default>\S+) "
    r"AVX2=(?P<avx2>\S+) AVX512=(?P<avx512>\S+)"
)
TARGET_RE = re.compile(
    r"target=cpublas_gemm_impl compile_capability=(?P<compile>\S+) "
    r"env_ATEN_CPU_CAPABILITY=(?P<env>\S*) fn=(?P<fn>\S+).*?"
    r"m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+).*?output_downcast=(?P<downcast>\d+)"
)
LANE_RE = re.compile(
    r"lane_trace compile_capability=(?P<compile>\S+) env_ATEN_CPU_CAPABILITY=(?P<env>\S*) "
    r"i=(?P<i>\d+) j=(?P<j>\d+) k=(?P<k>\d+) alpha=(?P<alpha>\S+) beta=(?P<beta>\S+) "
    r"prior=(?P<prior>\S+) dot=(?P<dot>\S+) combined=(?P<combined>\S+) output=(?P<output>\S+)"
)


def parse_trace(trace_path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "trace_path": str(trace_path),
        "available": trace_path.is_file(),
        "callsite": None,
        "target": None,
        "lane_trace": None,
        "invalid_capability_warning": False,
    }
    if not trace_path.is_file():
        return parsed
    text = trace_path.read_text(encoding="utf-8", errors="replace")
    parsed["invalid_capability_warning"] = "Ignoring invalid value for ATEN_CPU_CAPABILITY" in text
    for line in text.splitlines():
        if parsed["callsite"] is None:
            match = CALLSITE_RE.search(line)
            if match:
                parsed["callsite"] = {
                    "runtime_capability": match.group("runtime"),
                    "env_ATEN_CPU_CAPABILITY": match.group("env"),
                    "dispatch_DEFAULT": match.group("default"),
                    "dispatch_AVX2": match.group("avx2"),
                    "dispatch_AVX512": match.group("avx512"),
                }
        if parsed["target"] is None:
            match = TARGET_RE.search(line)
            if match:
                parsed["target"] = {
                    "selected_compile_capability": match.group("compile"),
                    "env_ATEN_CPU_CAPABILITY": match.group("env"),
                    "selected_function_pointer": match.group("fn"),
                    "m": int(match.group("m")),
                    "n": int(match.group("n")),
                    "k": int(match.group("k")),
                    "output_downcast": match.group("downcast") == "1",
                }
        if parsed["lane_trace"] is None:
            match = LANE_RE.search(line)
            if match:
                parsed["lane_trace"] = {
                    "compile_capability": match.group("compile"),
                    "env_ATEN_CPU_CAPABILITY": match.group("env"),
                    "hidden_lane": int(match.group("i")),
                    "column": int(match.group("j")),
                    "k": int(match.group("k")),
                    "alpha": float(match.group("alpha")),
                    "beta": float(match.group("beta")),
                    "bias_prior": float(match.group("prior")),
                    "dot": float(match.group("dot")),
                    "pre_bf16_combined": float(match.group("combined")),
                    "output": float(match.group("output")),
                }
    return parsed


def child_env_for_config(config: str, lane: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(TRACE_ENV_VARS)
    env["GPT_OSS_TRACE_ADDMM_LANE"] = str(lane)
    if config == "baseline":
        env.pop("ATEN_CPU_CAPABILITY", None)
    else:
        env["ATEN_CPU_CAPABILITY"] = config
    return env


def run_trace_child(
    args: argparse.Namespace,
    python: Path,
    *,
    layer: int,
    lane: int,
    config: str,
    full_run: bool,
) -> dict[str, Any]:
    stem = f"layer{layer}-{config}-lane{lane}{'-full' if full_run else ''}"
    child_output = args.research_path / f"{stem}.json"
    stdout_path = args.research_path / f"{stem}.stdout.log"
    trace_path = args.research_path / f"{stem}.trace.log"
    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--child",
        "--layer",
        str(layer),
        "--lane",
        str(lane),
        "--config",
        config,
        "--model",
        str(args.model),
        "--child-output",
        str(child_output),
    ]
    if full_run:
        cmd.append("--full-run")
    completed = run_cmd(
        cmd,
        env=child_env_for_config(config, lane),
        timeout=900,
        stdout_path=stdout_path,
        stderr_path=trace_path,
    )
    output = read_json(child_output) if completed["succeeded"] and child_output.is_file() else None
    return {
        "layer": layer,
        "lane": lane,
        "config": config,
        "full_run": full_run,
        "run": completed,
        "output_path": str(child_output),
        "stdout_path": str(stdout_path),
        "trace_path": str(trace_path),
        "output": output,
        "trace": parse_trace(trace_path),
    }


def expected_target_for_config(config: str) -> str:
    if config in {"baseline", "avx2", "avx512", "avx512_bf16", "avx512_vnni"}:
        return "AVX2"
    if config == "default":
        return "DEFAULT"
    return "unknown"


def summarize_runs(runs: list[dict[str, Any]], lane_plan: dict[str, Any]) -> dict[str, Any]:
    full_runs = [run for run in runs if run["full_run"] and run["output"]]
    trace_runs = [run for run in runs if run["output"]]
    trace_results_by_layer: dict[str, Any] = {str(layer): {} for layer in SAMPLED_LAYERS}
    target_summary: dict[str, Any] = {str(layer): {} for layer in SAMPLED_LAYERS}
    official_comparisons: dict[str, Any] = {str(layer): {} for layer in SAMPLED_LAYERS}
    negative_control_lanes: dict[str, list[int]] = {str(layer): [] for layer in SAMPLED_LAYERS}

    for run in trace_runs:
        layer_key = str(run["layer"])
        config = run["config"]
        lane_key = str(run["lane"])
        trace_results_by_layer[layer_key].setdefault(config, {})[lane_key] = {
            "target": run["trace"].get("target"),
            "callsite": run["trace"].get("callsite"),
            "lane_trace": run["trace"].get("lane_trace"),
            "official_lane": run["output"].get("official_lane"),
            "torch_addmm_lane": run["output"].get("torch_addmm_lane"),
            "torch_addmm_vs_official_focus": run["output"].get("torch_addmm_vs_official", {}).get("focus_lane"),
            "trace_path": run["trace_path"],
        }
        if run["full_run"]:
            target_summary[layer_key][config] = {
                "callsite": run["trace"].get("callsite"),
                "target": run["trace"].get("target"),
                "invalid_capability_warning": run["trace"].get("invalid_capability_warning"),
                "target_rule_matched": (
                    (run["trace"].get("target") or {}).get("selected_compile_capability")
                    == expected_target_for_config(config)
                ),
            }
            official_comparisons[layer_key][config] = run["output"].get("variants", {})
            for variant_name in (
                "zero_bias_addmm_plus_bias",
                "explicit_matmul_plus_bias",
                "explicit_einsum_plus_bias",
            ):
                comparison = run["output"].get("variants", {}).get(variant_name, {}).get("comparison_vs_official", {})
                for key in ("first_mismatch", "worst_mismatch"):
                    mismatch = comparison.get(key)
                    if isinstance(mismatch, dict) and isinstance(mismatch.get("hidden_lane"), int):
                        negative_control_lanes[layer_key].append(mismatch["hidden_lane"])

    baseline_target_rule_holds = all(
        target_summary[str(layer)].get("baseline", {}).get("target_rule_matched") is True
        for layer in SAMPLED_LAYERS
    )
    baseline_full_vector_matches = all(
        all(
            official_comparisons[str(layer)]
            .get("baseline", {})
            .get(name, {})
            .get("comparison_vs_official", {})
            .get("full_vector_cleared")
            is True
            for name in ("torch_addmm", "torch_nn_functional_linear", "torch_C_nn_linear")
        )
        for layer in SAMPLED_LAYERS
    )
    negative_controls_negative = all(
        all(
            official_comparisons[str(layer)]
            .get("baseline", {})
            .get(name, {})
            .get("comparison_vs_official", {})
            .get("full_vector_cleared")
            is False
            for name in ("zero_bias_addmm_plus_bias", "explicit_matmul_plus_bias", "explicit_einsum_plus_bias")
        )
        for layer in SAMPLED_LAYERS
    )
    all_layers_traced = all(
        str(layer) in trace_results_by_layer
        and "baseline" in trace_results_by_layer[str(layer)]
        and "default" in trace_results_by_layer[str(layer)]
        for layer in SAMPLED_LAYERS
    )

    residual_traced = 0
    residual_explained = 0
    residual_unexplained = 0
    residual_explanation_by_layer: dict[str, list[dict[str, Any]]] = {str(layer): [] for layer in SAMPLED_LAYERS}
    for layer in SAMPLED_LAYERS:
        for lane in lane_plan["residual_lanes_by_layer"].get(str(layer), []):
            baseline_lane = (
                trace_results_by_layer[str(layer)]
                .get("baseline", {})
                .get(str(lane), {})
                .get("lane_trace")
            )
            default_lane = (
                trace_results_by_layer[str(layer)]
                .get("default", {})
                .get(str(lane), {})
                .get("lane_trace")
            )
            item = {
                "hidden_lane": lane,
                "baseline_traced": baseline_lane is not None,
                "default_traced": default_lane is not None,
                "explanation": "missing_trace",
            }
            if baseline_lane:
                residual_traced += 1
                if layer == 18 and lane == 1641 and default_lane:
                    residual_explained += 1
                    item["explanation"] = "target_selection_and_bf16_rounding_boundary"
                else:
                    residual_unexplained += 1
                    item["explanation"] = "official_gemm_stub_value_traced_replay_design_needed"
                item["baseline"] = baseline_lane
                if default_lane:
                    item["default"] = default_lane
                    item["avx2_default_dot_diff"] = abs(
                        float(baseline_lane["dot"]) - float(default_lane["dot"])
                    )
                    item["avx2_default_pre_bf16_diff"] = abs(
                        float(baseline_lane["pre_bf16_combined"])
                        - float(default_lane["pre_bf16_combined"])
                    )
            residual_explanation_by_layer[str(layer)].append(item)

    default_summary = {
        str(layer): {
            "target": target_summary[str(layer)].get("default", {}).get("target"),
            "official_variants": {
                name: official_comparisons[str(layer)]
                .get("default", {})
                .get(name, {})
                .get("comparison_vs_official", {})
                for name in ("torch_addmm", "torch_nn_functional_linear", "torch_C_nn_linear")
            },
        }
        for layer in SAMPLED_LAYERS
    }
    supports_design = bool(
        all_layers_traced
        and baseline_target_rule_holds
        and baseline_full_vector_matches
        and negative_controls_negative
        and residual_traced > 0
    )
    return {
        "all_sampled_layers_traced": all_layers_traced,
        "baseline_target_rule_holds_all_layers": baseline_target_rule_holds,
        "baseline_full_vector_matches_official_all_layers": baseline_full_vector_matches,
        "negative_controls_remained_negative": negative_controls_negative,
        "sampled_trace_supports_source_replay_design": supports_design,
        "trace_results_by_layer": trace_results_by_layer,
        "per_layer_target_summary": target_summary,
        "per_layer_official_comparisons": official_comparisons,
        "negative_control_lanes_by_layer": {
            layer: sorted(set(values)) for layer, values in negative_control_lanes.items()
        },
        "default_diagnostic_behavior_summary": default_summary,
        "residual_lanes_traced_count": residual_traced,
        "residual_lanes_explained_count": residual_explained,
        "residual_lanes_unexplained_count": residual_unexplained,
        "residual_explanation_by_layer": residual_explanation_by_layer,
    }


def main() -> int:
    args = parse_args()
    if args.child:
        return runtime_child(args)

    args.research_path.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "classification": "fused_linear_addmm_gemm_stub_sampled_trace_failed",
        "validation_only": True,
        "source_attribution_probe": True,
        "gemm_stub_sampled_trace": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "source_checkout_path": str(args.source_checkout_path),
        "checked_out_commit": EXPECTED_COMMIT,
        "build_env_path": str(args.build_env_path),
        "pre_existing_patch_verified": PRE_EXISTING_PATCH.is_file(),
        "pre_sampled_trace_patch_path": str(args.research_path / "pre_sampled_trace.patch"),
        "pytorch_source_patched_this_branch": False,
        "pytorch_rebuilt_this_branch": False,
        "instrumented_source_files": [],
        "instrumentation_env_vars": {**TRACE_ENV_VARS, "GPT_OSS_TRACE_ADDMM_LANE": "<per-lane>"},
        "sampled_layers_requested": SAMPLED_LAYERS,
        "sampled_layers_evaluated": [],
        "configs_evaluated": [],
        "concrete_global_replay_policy_found": False,
        "concrete_replayable_rule_found": False,
        "replayable_rule_scope": "none",
        "reopen_rust_policy_synthesis": False,
        **FALSE_GUARDS,
    }
    try:
        state = source_state(args.source_checkout_path)
        status["source_state"] = state
        status["instrumented_source_files"] = [
            path for path in INSTRUMENTED_SOURCE_FILES if path in state.get("dirty_files", [])
        ]
        write_json(args.research_path / "sampled-trace-source-state.json", state)
        if not state["expected_commit_match"] or not status["pre_existing_patch_verified"]:
            status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_build_or_patch_failed"
            status["failure_summary"] = "PyTorch checkout commit or pre-existing patch archive check failed."
            write_json(args.status_output, status)
            return 0
        patch_archive = archive_patch(args.source_checkout_path, args.research_path / "pre_sampled_trace.patch")
        status["pre_sampled_trace_patch"] = patch_archive

        python = args.build_env_path / "bin/python"
        if not python.is_file():
            status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_build_or_patch_failed"
            status["failure_summary"] = f"missing source-build Python: {python}"
            write_json(args.status_output, status)
            return 0

        lane_plan = build_lane_plan()
        status.update(
            {
                "residual_lane_sources": lane_plan["residual_lane_sources"],
                "residual_lanes_by_layer": lane_plan["residual_lanes_by_layer"],
                "focus_lanes_by_layer": lane_plan["focus_lanes_by_layer"],
                "missing_lane_metadata": lane_plan["missing_lane_metadata"],
            }
        )
        write_json(args.research_path / "residual-lanes-input.json", lane_plan)

        runs: list[dict[str, Any]] = []
        for layer in SAMPLED_LAYERS:
            primary_lane = FOCUS_LANES[layer]
            for config in REQUIRED_CONFIGS + OPTIONAL_CONFIGS:
                runs.append(
                    run_trace_child(args, python, layer=layer, lane=primary_lane, config=config, full_run=True)
                )
            for lane in lane_plan["trace_lanes_by_layer"][str(layer)]:
                if lane == primary_lane:
                    continue
                for config in REQUIRED_CONFIGS:
                    runs.append(
                        run_trace_child(args, python, layer=layer, lane=lane, config=config, full_run=False)
                    )

        successful_runs = [run for run in runs if run["run"].get("succeeded") and run.get("output")]
        status["sampled_layers_evaluated"] = sorted({run["layer"] for run in successful_runs})
        status["configs_evaluated"] = sorted({run["config"] for run in successful_runs})
        status["runtime_run_count"] = len(runs)
        status["runtime_success_count"] = len(successful_runs)
        status["runtime_failed_runs"] = [
            {
                "layer": run["layer"],
                "lane": run["lane"],
                "config": run["config"],
                "full_run": run["full_run"],
                "trace_path": run["trace_path"],
                "returncode": run["run"].get("returncode"),
                "stderr_tail": run["run"].get("stderr_tail"),
            }
            for run in runs
            if not (run["run"].get("succeeded") and run.get("output"))
        ]

        summary = summarize_runs(runs, lane_plan)
        status.update(summary)
        status["concrete_replayable_rule_found"] = summary["residual_lanes_explained_count"] > 0
        status["replayable_rule_scope"] = (
            "residual_lane_set" if summary["residual_lanes_explained_count"] > 1 else "lane_level"
            if status["concrete_replayable_rule_found"]
            else "none"
        )
        status["concrete_global_replay_policy_found"] = False
        status["reopen_rust_policy_synthesis"] = False

        if status["runtime_failed_runs"]:
            status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_partial_only"
        elif summary["sampled_trace_supports_source_replay_design"]:
            status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_supports_replay_design"
        elif summary["all_sampled_layers_traced"]:
            status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_all_layers_traced"
        else:
            status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_partial_only"

        write_json(args.research_path / "per-layer-target-summary.json", summary["per_layer_target_summary"])
        write_json(args.research_path / "per-layer-official-comparisons.json", summary["per_layer_official_comparisons"])
        write_json(args.research_path / "residual-lane-traces.json", summary["residual_explanation_by_layer"])
        layer18 = summary["trace_results_by_layer"].get("18", {})
        write_json(
            args.research_path / "layer18-lane1641-confirmation.json",
            {
                "baseline": layer18.get("baseline", {}).get("1641"),
                "default": layer18.get("default", {}).get("1641"),
            },
        )
        write_json(args.research_path / "default-diagnostic-summary.json", summary["default_diagnostic_behavior_summary"])
        interpretation = {
            "classification": status["classification"],
            "baseline_target_rule_holds_all_layers": status["baseline_target_rule_holds_all_layers"],
            "baseline_full_vector_matches_official_all_layers": status[
                "baseline_full_vector_matches_official_all_layers"
            ],
            "negative_controls_remained_negative": status["negative_controls_remained_negative"],
            "residual_lanes_traced_count": status["residual_lanes_traced_count"],
            "residual_lanes_explained_count": status["residual_lanes_explained_count"],
            "sampled_trace_supports_source_replay_design": status[
                "sampled_trace_supports_source_replay_design"
            ],
            "concrete_global_replay_policy_found": False,
            "reopen_rust_policy_synthesis": False,
        }
        write_json(args.research_path / "interpretation-summary.json", interpretation)
        (args.research_path / "interpretation-summary.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in interpretation.items()) + "\n",
            encoding="utf-8",
        )
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status["classification"] = "fused_linear_addmm_gemm_stub_sampled_trace_failed"
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
