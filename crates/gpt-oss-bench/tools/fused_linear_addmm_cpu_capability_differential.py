#!/usr/bin/env python3
"""CPU capability differential probe for the fused addmm o-proj seam.

This source-attribution probe runs fresh CPU-only Python workers under selected
ATEN_CPU_CAPABILITY settings and compares the official fused addmm seam against
both historical official artifacts and the no-override baseline output.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
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
    make_result,
    tensor_metadata,
    write_json,
)


DEFAULT_FORWARD_ENV = Path("/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-cpu-capability-differential")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_cpu_capability_differential_status.json")
PRIOR_RUST_STATUS = Path("/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json")
SAMPLED = [6, 10, 13, 16, 18, 21]
REQUIRED_CONFIGS = ["baseline", "default"]
OPTIONAL_CONFIGS = ["avx2", "avx512", "avx512_bf16", "avx512_vnni"]
VARIANT_NAMES = [
    "torch_addmm_fused_bias",
    "torch_nn_functional_linear",
    "torch_C_nn_linear",
    "zero_bias_addmm_plus_bias",
    "explicit_matmul_plus_bias",
    "explicit_einsum_plus_bias",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU capability differential for fused addmm o-proj.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--layers", default=",".join(str(layer) for layer in SAMPLED))
    parser.add_argument("--capability-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--capability-config", default="baseline", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


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
        "torch_config_show_abbrev": "\n".join(torch_config_show(torch).splitlines()[:40]),
        "torch_get_num_threads": int(torch.get_num_threads()),
        "torch_get_num_interop_threads": interop,
        "torch_backends_mkldnn_enabled": bool(getattr(torch.backends.mkldnn, "enabled", False)),
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "cuda_used": False,
    }


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
    }


def missing_required_paths(model: Path, layers: list[int]) -> list[str]:
    paths = [model]
    for layer in layers:
        source = layer_paths(layer)
        paths.extend([Path(source["attention_bundle_dir"]), Path(source["weighted_v"]), Path(source["official_o_proj"])])
    return [str(path) for path in paths if not path.exists()]


def tensor_values_float(tensor: Any) -> list[float]:
    return [float(value) for value in tensor.detach().to("cpu").float().reshape(-1).tolist()]


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
    tensors = {
        "weighted_v": weighted_v,
        "official_o_proj": official,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "input_2d": input_2d,
        "weight_t_2d": weight_t_2d,
    }
    for name, tensor in tensors.items():
        assert_cpu_tensor(f"layer{layer}.{name}", tensor)
    return {
        **tensors,
        "zero_bias": torch.zeros_like(bias),
        "paths": paths,
        "weighted_v_json": weighted_v_json,
        "official_json": official_json,
        "weight_name": weight_name,
        "bias_name": bias_name,
    }


def run_variant(
    torch: Any,
    name: str,
    thunk: Any,
    official: Any,
    focus_lane: int,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    try:
        output = thunk()
        if output.dim() == 2 and output.shape[0] == 1:
            output = output.squeeze(0)
        assert_cpu_tensor(name, output)
        result = make_result(torch, name, True, True, output, official, focus_lane, diagnostic_only=diagnostic_only)
        result["output_values"] = tensor_values_float(output)
        result["output_metadata"] = tensor_metadata(torch, output, include_summary=False)
        return result
    except Exception as exc:  # noqa: BLE001
        result = make_result(torch, name, False, False, None, official, focus_lane, repr(exc), diagnostic_only)
        result["output_values"] = None
        result["output_metadata"] = None
        return result


def run_profiler(torch: Any, layer_tensors: dict[int, dict[str, Any]]) -> dict[str, Any]:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "succeeded": False, "reason": repr(exc), "event_keys": []}
    try:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            with record_function("cpu_capability_addmm"):
                for tensors in layer_tensors.values():
                    torch.addmm(tensors["o_proj_bias"], tensors["input_2d"], tensors["weight_t_2d"])
        keys = [event.key for event in prof.key_averages()]
        return {
            "attempted": True,
            "succeeded": True,
            "event_keys": keys[:60],
            "aten_addmm_seen": any("aten::addmm" in key for key in keys),
            "mkldnn_or_onednn_seen": any(("mkldnn" in key.lower() or "onednn" in key.lower()) for key in keys),
        }
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "succeeded": False, "reason": repr(exc), "event_keys": []}


def run_layer(torch: Any, tensors: dict[str, Any], layer: int) -> dict[str, Any]:
    focus_lane = int(SAMPLED_LAYERS[layer]["focus_lane"])
    official = tensors["official_o_proj"]
    weight = tensors["o_proj_weight"]
    bias = tensors["o_proj_bias"]
    input_2d = tensors["input_2d"]
    weight_t_2d = tensors["weight_t_2d"]
    zero_bias = tensors["zero_bias"]
    variants = {
        "torch_addmm_fused_bias": run_variant(
            torch,
            "torch_addmm_fused_bias",
            lambda: torch.addmm(bias, input_2d, weight_t_2d),
            official,
            focus_lane,
        ),
        "torch_nn_functional_linear": run_variant(
            torch,
            "torch_nn_functional_linear",
            lambda: torch.nn.functional.linear(input_2d, weight, bias),
            official,
            focus_lane,
        ),
        "torch_C_nn_linear": run_variant(
            torch,
            "torch_C_nn_linear",
            lambda: torch._C._nn.linear(input_2d, weight, bias),
            official,
            focus_lane,
        ),
        "zero_bias_addmm_plus_bias": run_variant(
            torch,
            "zero_bias_addmm_plus_bias",
            lambda: torch.addmm(zero_bias, input_2d, weight_t_2d).squeeze(0) + bias,
            official,
            focus_lane,
            diagnostic_only=True,
        ),
        "explicit_matmul_plus_bias": run_variant(
            torch,
            "explicit_matmul_plus_bias",
            lambda: input_2d @ weight_t_2d + bias,
            official,
            focus_lane,
            diagnostic_only=True,
        ),
        "explicit_einsum_plus_bias": run_variant(
            torch,
            "explicit_einsum_plus_bias",
            lambda: torch.einsum("bk,hk->bh", input_2d, weight) + bias,
            official,
            focus_lane,
            diagnostic_only=True,
        ),
    }
    official_variants = ["torch_addmm_fused_bias", "torch_nn_functional_linear", "torch_C_nn_linear"]
    negative_controls = ["zero_bias_addmm_plus_bias", "explicit_matmul_plus_bias", "explicit_einsum_plus_bias"]
    return {
        "layer_index": layer,
        "role": SAMPLED_LAYERS[layer]["role"],
        "focus_lane": focus_lane,
        "source_artifacts": tensors["paths"],
        "model_tensors_loaded": [tensors["weight_name"], tensors["bias_name"]],
        "tensor_metadata": {
            "weighted_v": tensor_metadata(torch, tensors["weighted_v"]),
            "o_proj_weight": tensor_metadata(torch, weight),
            "o_proj_bias": tensor_metadata(torch, bias),
            "official_o_proj": tensor_metadata(torch, official),
            "input_2d": tensor_metadata(torch, input_2d, include_summary=False),
            "weight_t_2d": tensor_metadata(torch, weight_t_2d, include_summary=False),
        },
        "official_output_values": tensor_values_float(official),
        "variants": variants,
        "official_variants_cleared": all(variants[name].get("full_vector_cleared") for name in official_variants),
        "negative_controls_remained_negative": all(not variants[name].get("full_vector_cleared") for name in negative_controls),
    }


def capability_child(args: argparse.Namespace, layers: list[int]) -> int:
    torch, Checkpoint = import_torch_and_checkpoint()
    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    layer_tensors = {layer: load_layer_tensors(torch, checkpoint, layer) for layer in layers}
    layer_results = [run_layer(torch, layer_tensors[layer], layer) for layer in layers]
    profiler = run_profiler(torch, layer_tensors)
    status = {
        "config_name": args.capability_config,
        "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
        "environment": torch_metadata(torch),
        "layers": layer_results,
        "profiler_event_keys": profiler,
        "cuda_used": False,
        "all_official_variants_cleared": all(layer["official_variants_cleared"] for layer in layer_results),
        "negative_controls_remained_negative": all(layer["negative_controls_remained_negative"] for layer in layer_results),
    }
    if args.child_output:
        write_json(args.child_output, status)
    else:
        print(json.dumps(status, sort_keys=True))
    return 0


def run_child_config(args: argparse.Namespace, config_name: str) -> dict[str, Any]:
    env = os.environ.copy()
    if config_name == "baseline":
        env.pop("ATEN_CPU_CAPABILITY", None)
    else:
        env["ATEN_CPU_CAPABILITY"] = config_name
    child_output = args.research_path / f"raw-{config_name}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--capability-child",
        "--capability-config",
        config_name,
        "--model",
        str(args.model),
        "--layers",
        args.layers,
        "--child-output",
        str(child_output),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=360, check=False)
    result: dict[str, Any] = {
        "config_name": config_name,
        "env_ATEN_CPU_CAPABILITY": env.get("ATEN_CPU_CAPABILITY"),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-80:],
        "stderr_tail": completed.stderr.splitlines()[-80:],
        "raw_output_path": str(child_output),
        "executed": completed.returncode == 0 and child_output.is_file(),
        "skipped_or_unavailable": completed.returncode != 0 or not child_output.is_file(),
    }
    if result["executed"]:
        try:
            result["raw"] = load_json(child_output)
        except Exception as exc:  # noqa: BLE001
            result["executed"] = False
            result["skipped_or_unavailable"] = True
            result["parse_error"] = repr(exc)
    return result


def tensor_from_values(torch: Any, values: list[float]) -> Any:
    return torch.tensor(values, dtype=torch.float32, device="cpu").to(torch.bfloat16)


def comparison_from_values(torch: Any, actual_values: list[float] | None, expected_values: list[float] | None, focus_lane: int) -> dict[str, Any] | None:
    if actual_values is None or expected_values is None:
        return None
    return compare_tensors(torch, tensor_from_values(torch, actual_values), tensor_from_values(torch, expected_values), focus_lane)


def one_bf16_ulp_or_less(diff: float, expected: float) -> bool:
    if diff == 0.0:
        return True
    # BF16 spacing is 2^(exponent - 7) for normalized values. This intentionally
    # stays approximate; it is an attribution hint, not a proof gate.
    import math

    magnitude = abs(expected)
    if magnitude == 0.0:
        return diff <= 2.0 ** -133
    exponent = math.floor(math.log2(magnitude))
    ulp = 2.0 ** (exponent - 7)
    return diff <= ulp + 1e-30


def load_prior_rust_residuals() -> dict[str, Any]:
    if not PRIOR_RUST_STATUS.is_file():
        return {"available": False, "path": str(PRIOR_RUST_STATUS), "top_candidates": [], "residual_lanes_by_layer": {}}
    try:
        data = load_json(PRIOR_RUST_STATUS)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(PRIOR_RUST_STATUS), "error": repr(exc), "top_candidates": [], "residual_lanes_by_layer": {}}
    residuals: dict[int, set[int]] = {}
    top_candidates = data.get("top_near_global_candidates", [])[:10]
    for candidate in top_candidates:
        for layer in candidate.get("per_layer", []):
            layer_index = int(layer.get("layer_index"))
            for key in ["first_mismatch", "worst_mismatch"]:
                mismatch = layer.get(key)
                if isinstance(mismatch, dict):
                    lane = mismatch.get("hidden_lane", mismatch.get("index"))
                    if lane is not None:
                        residuals.setdefault(layer_index, set()).add(int(lane))
    return {
        "available": True,
        "path": str(PRIOR_RUST_STATUS),
        "classification": data.get("classification"),
        "next_recommended_state": data.get("next_recommended_state"),
        "top_candidates": [
            {
                "candidate_name": candidate.get("candidate_name"),
                "cleared_layer_count": candidate.get("cleared_layer_count"),
                "total_mismatches": candidate.get("total_mismatches"),
                "max_abs_diff": candidate.get("max_abs_diff"),
            }
            for candidate in top_candidates
        ],
        "residual_lanes_by_layer": {str(layer): sorted(lanes) for layer, lanes in residuals.items()},
    }


def compare_config_to_baseline(torch: Any, config: dict[str, Any], baseline: dict[str, Any], prior_residuals: dict[str, Any]) -> dict[str, Any]:
    config_raw = config["raw"]
    baseline_raw = baseline["raw"]
    baseline_by_layer = {int(layer["layer_index"]): layer for layer in baseline_raw["layers"]}
    layers = []
    changed_layers: set[int] = set()
    changed_variant_layers: dict[str, list[int]] = {name: [] for name in VARIANT_NAMES}
    for layer in config_raw["layers"]:
        layer_index = int(layer["layer_index"])
        baseline_layer = baseline_by_layer[layer_index]
        focus_lane = int(layer["focus_lane"])
        layer_summary = {
            "layer_index": layer_index,
            "role": layer["role"],
            "focus_lane": focus_lane,
            "variants": {},
        }
        for variant_name, variant in layer["variants"].items():
            baseline_variant = baseline_layer["variants"].get(variant_name, {})
            official_values = layer["official_output_values"]
            actual_values = variant.get("output_values")
            baseline_values = baseline_variant.get("output_values")
            vs_official = comparison_from_values(torch, actual_values, official_values, focus_lane)
            vs_baseline = comparison_from_values(torch, actual_values, baseline_values, focus_lane)
            mismatch_samples = (vs_baseline or {}).get("mismatch_samples", [])
            changed_lanes = [int(sample["index"]) for sample in mismatch_samples]
            prior_lanes = set(prior_residuals.get("residual_lanes_by_layer", {}).get(str(layer_index), []))
            overlap = sorted(set(changed_lanes) & prior_lanes)
            full_variant = {
                "available": variant.get("available"),
                "executed": variant.get("executed"),
                "diagnostic_only": variant.get("diagnostic_only"),
                "full_vector_mismatches_vs_official": (vs_official or {}).get("mismatch_count"),
                "max_abs_diff_vs_official": (vs_official or {}).get("max_abs_diff"),
                "mean_abs_diff_vs_official": (vs_official or {}).get("mean_abs_diff"),
                "first_mismatch_vs_official": (vs_official or {}).get("first_mismatch"),
                "worst_mismatch_vs_official": (vs_official or {}).get("worst_mismatch"),
                "full_vector_mismatches_vs_baseline": (vs_baseline or {}).get("mismatch_count"),
                "max_abs_diff_vs_baseline": (vs_baseline or {}).get("max_abs_diff"),
                "mean_abs_diff_vs_baseline": (vs_baseline or {}).get("mean_abs_diff"),
                "first_mismatch_vs_baseline": (vs_baseline or {}).get("first_mismatch"),
                "worst_mismatch_vs_baseline": (vs_baseline or {}).get("worst_mismatch"),
                "full_vector_cleared_vs_official": (vs_official or {}).get("full_vector_cleared"),
                "full_vector_cleared_vs_baseline": (vs_baseline or {}).get("full_vector_cleared"),
                "changed_lane_samples_vs_baseline": changed_lanes,
                "changed_lanes_overlap_prior_rust_residual_lanes": overlap,
            }
            layer_summary["variants"][variant_name] = full_variant
            if full_variant["full_vector_cleared_vs_baseline"] is False:
                changed_layers.add(layer_index)
                changed_variant_layers.setdefault(variant_name, []).append(layer_index)
        layers.append(layer_summary)
    return {
        "config_name": config["config_name"],
        "ATEN_CPU_CAPABILITY": config["raw"].get("ATEN_CPU_CAPABILITY"),
        "environment": config["raw"].get("environment"),
        "profiler_event_keys": config["raw"].get("profiler_event_keys"),
        "layers": layers,
        "layers_changed_vs_baseline": sorted(changed_layers),
        "changed_variant_layers": {name: values for name, values in changed_variant_layers.items() if values},
    }


def build_layer18_summary(config_summaries: list[dict[str, Any]], prior_residuals: dict[str, Any]) -> dict[str, Any]:
    configs = {}
    for config in config_summaries:
        layer = next((item for item in config["layers"] if item["layer_index"] == 18), None)
        if not layer:
            continue
        addmm = layer["variants"]["torch_addmm_fused_bias"]
        first = addmm.get("first_mismatch_vs_baseline")
        worst = addmm.get("worst_mismatch_vs_baseline")
        max_diff = addmm.get("max_abs_diff_vs_baseline")
        expected_for_ulp = worst.get("expected") if isinstance(worst, dict) else 0.0
        configs[config["config_name"]] = {
            "addmm_mismatches_vs_official": addmm.get("full_vector_mismatches_vs_official"),
            "addmm_mismatches_vs_baseline": addmm.get("full_vector_mismatches_vs_baseline"),
            "max_abs_diff_vs_baseline": max_diff,
            "first_mismatch_vs_baseline": first,
            "worst_mismatch_vs_baseline": worst,
            "changed_output_is_one_bf16_ulp_or_less": one_bf16_ulp_or_less(float(max_diff or 0.0), float(expected_for_ulp or 0.0)),
            "changed_lanes_overlap_prior_rust_residual_lanes": addmm.get("changed_lanes_overlap_prior_rust_residual_lanes"),
        }
    return {
        "prior_rust_residual_lanes": prior_residuals.get("residual_lanes_by_layer", {}).get("18", []),
        "configs": configs,
    }


def compact_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    compact_layers = []
    for layer in config["layers"]:
        compact_layers.append(
            {
                "layer_index": layer["layer_index"],
                "focus_lane": layer["focus_lane"],
                "variant_summary": {
                    name: {
                        "mismatches_vs_official": result.get("full_vector_mismatches_vs_official"),
                        "mismatches_vs_baseline": result.get("full_vector_mismatches_vs_baseline"),
                        "max_abs_diff_vs_baseline": result.get("max_abs_diff_vs_baseline"),
                        "cleared_vs_official": result.get("full_vector_cleared_vs_official"),
                        "cleared_vs_baseline": result.get("full_vector_cleared_vs_baseline"),
                    }
                    for name, result in layer["variants"].items()
                },
            }
        )
    return {
        "config_name": config["config_name"],
        "ATEN_CPU_CAPABILITY": config["ATEN_CPU_CAPABILITY"],
        "layers_changed_vs_baseline": config["layers_changed_vs_baseline"],
        "changed_variant_layers": config["changed_variant_layers"],
        "layers": compact_layers,
        "profiler_event_keys": config.get("profiler_event_keys"),
    }


def base_status(classification: str, args: argparse.Namespace, layers: list[int]) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "source_attribution_probe": True,
        "cpu_capability_differential": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "forward_env_path": str(args.forward_env_path),
        "sampled_layers_requested": layers,
        "sampled_layers_evaluated": [],
        "configs_attempted": [],
        "configs_executed": [],
        "configs_skipped": [],
        "full_model_loaded": False,
        "gpu_tensors_created": False,
        "pytorch_build_performed": False,
        "pytorch_source_patched": False,
        "rebaseline_performed": False,
        "old_artifacts_replaced": False,
        "active_backend_inference": "inconclusive",
        "concrete_replayable_rule_found": False,
        "reopen_rust_policy_synthesis": False,
        "tolerance_pass": False,
        "correction_metadata_applied": False,
        **GUARD_FALSE_FLAGS,
    }


def write_research_outputs(args: argparse.Namespace, status: dict[str, Any]) -> None:
    args.research_path.mkdir(parents=True, exist_ok=True)
    write_json(args.research_path / "per-config-summary.json", status.get("per_config_summary", []))
    write_json(args.research_path / "layer18-differential.json", status.get("layer18_differential_summary", {}))
    write_json(args.research_path / "profiler-events.json", status.get("profiler_events", {}))
    interpretation = {
        "classification": status["classification"],
        "cpu_capability_changes_any_official_output": status["cpu_capability_changes_any_official_output"],
        "layers_changed_by_cpu_capability": status["layers_changed_by_cpu_capability"],
        "layer18_changed_under_default": status["layer18_changed_under_default"],
        "official_baseline_requires_optimized_cpu_capability": status["official_baseline_requires_optimized_cpu_capability"],
        "active_backend_inference": status["active_backend_inference"],
        "concrete_replayable_rule_found": status["concrete_replayable_rule_found"],
        "reopen_rust_policy_synthesis": status["reopen_rust_policy_synthesis"],
    }
    (args.research_path / "interpretation-summary.txt").write_text(json.dumps(interpretation, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if args.capability_child:
        return capability_child(args, layers)

    args.research_path.mkdir(parents=True, exist_ok=True)
    missing = missing_required_paths(args.model, layers)
    if missing:
        status = base_status("fused_linear_addmm_cpu_capability_differential_failed", args, layers)
        status["missing_required_artifacts"] = missing
        write_json(args.status_output, status)
        return 0

    try:
        torch, _Checkpoint = import_torch_and_checkpoint()
        metadata = torch_metadata(torch)
        prior_residuals = load_prior_rust_residuals()
        attempted = REQUIRED_CONFIGS + OPTIONAL_CONFIGS
        raw_results = [run_child_config(args, name) for name in attempted]
        executed = [result for result in raw_results if result.get("executed")]
        skipped = [
            {
                "config_name": result["config_name"],
                "returncode": result["returncode"],
                "stdout_tail": result["stdout_tail"],
                "stderr_tail": result["stderr_tail"],
                "reason": result.get("parse_error", "worker_failed_or_output_missing"),
            }
            for result in raw_results
            if not result.get("executed")
        ]
        baseline = next((result for result in executed if result["config_name"] == "baseline"), None)
        default = next((result for result in executed if result["config_name"] == "default"), None)
        if baseline is None or default is None:
            classification = "fused_linear_addmm_cpu_capability_differential_inconclusive"
            status = base_status(classification, args, layers)
            status.update(
                {
                    **metadata,
                    "configs_attempted": attempted,
                    "configs_executed": [result["config_name"] for result in executed],
                    "configs_skipped": skipped,
                    "prior_rust_policy_diagnostics": prior_residuals,
                    "reason": "required baseline/default configuration did not execute",
                }
            )
            write_json(args.status_output, status)
            return 0

        config_summaries = [compare_config_to_baseline(torch, result, baseline, prior_residuals) for result in executed]
        per_config_compact = [compact_config_summary(config) for config in config_summaries]
        layers_changed: set[int] = set()
        for config in config_summaries:
            if config["config_name"] == "baseline":
                continue
            for layer in config["layers_changed_vs_baseline"]:
                layers_changed.add(int(layer))
        default_summary = next(config for config in config_summaries if config["config_name"] == "default")
        default_addmm_changed_layers = default_summary.get("changed_variant_layers", {}).get("torch_addmm_fused_bias", [])
        layer18_changed_under_default = 18 in default_addmm_changed_layers
        baseline_addmm_official_ok = all(
            layer["variants"]["torch_addmm_fused_bias"]["full_vector_cleared_vs_official"] is True
            for layer in next(config for config in config_summaries if config["config_name"] == "baseline")["layers"]
        )
        default_addmm_official_ok = all(
            layer["variants"]["torch_addmm_fused_bias"]["full_vector_cleared_vs_official"] is True
            for layer in default_summary["layers"]
        )
        capability_changes_any = bool(layers_changed)
        official_baseline_requires = (
            True
            if baseline_addmm_official_ok and not default_addmm_official_ok
            else False
            if baseline_addmm_official_ok and default_addmm_official_ok and not default_addmm_changed_layers
            else "inconclusive"
        )
        if capability_changes_any:
            classification = "fused_linear_addmm_cpu_capability_differential_official_depends_on_cpu_capability"
            active_backend = "optimized_cpu_kernel_likely"
        else:
            classification = "fused_linear_addmm_cpu_capability_differential_no_material_change"
            active_backend = "multiple_possible"
        layer18_summary = build_layer18_summary(config_summaries, prior_residuals)
        profiler_events = {
            result["config_name"]: result["raw"].get("profiler_event_keys", {}) for result in executed
        }
        status = base_status(classification, args, layers)
        status.update(
            {
                **metadata,
                "sampled_layers_evaluated": layers,
                "configs_attempted": attempted,
                "configs_executed": [result["config_name"] for result in executed],
                "configs_skipped": skipped,
                "per_config_summary": per_config_compact,
                "full_config_differentials": config_summaries,
                "layer18_differential_summary": layer18_summary,
                "prior_rust_policy_diagnostics": prior_residuals,
                "profiler_events": profiler_events,
                "cpu_capability_changes_any_official_output": capability_changes_any,
                "layers_changed_by_cpu_capability": sorted(layers_changed),
                "layer18_changed_under_default": bool(layer18_changed_under_default),
                "official_baseline_requires_optimized_cpu_capability": official_baseline_requires,
                "active_backend_inference": active_backend,
                "concrete_replayable_rule_found": False,
                "reopen_rust_policy_synthesis": False,
                "interpretation": {
                    "default_differs_from_official_or_baseline": bool(layer18_changed_under_default),
                    "capability_change_is_not_rebaseline": True,
                    "capability_change_identifies_replayable_rule": False,
                    "next_bounded_step": "Preserve official Torch API seam; do not reopen Rust/CUDA policy without a concrete replayable rule.",
                },
            }
        )
        write_research_outputs(args, status)
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = base_status("fused_linear_addmm_cpu_capability_differential_failed", args, layers)
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
