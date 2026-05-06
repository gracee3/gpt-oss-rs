#!/usr/bin/env python3
"""CPU Torch dispatch-stability probe for fused addmm attention o-proj.

This Gate A oracle evidence probe checks whether
``torch.addmm(bias, input_2d, weight_t_2d)`` is stable across CPU thread and
backend settings for the sampled official fused-linear/addmm seam.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fused_linear_addmm_cpu_producer_attribution import (
    DEFAULT_MODEL,
    GUARD_FALSE_FLAGS,
    REPO_ROOT,
    SAMPLED_LAYERS,
    assert_cpu_tensor,
    compare_tensors,
    environment_metadata,
    json_tensor_values,
    load_json,
    make_result,
    tensor_metadata,
    write_json,
)


DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_cpu_dispatch_stability_status.json")
DEFAULT_OUTPUT_DIR = Path("/tmp/fused_linear_addmm_cpu_dispatch_stability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU Torch dispatch-stability probe for fused addmm attention o-proj."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="6,10,13,16,18,21")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle": f"/tmp/layer{layer}_ordered_attention_bundle_status.json",
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
    }


def required_source_paths(layers: list[int], model: Path) -> list[Path]:
    paths = [
        model,
        Path("/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json"),
        Path("/tmp/fused_linear_addmm_addmm_boundary_localization_status.json"),
        Path("/tmp/fused_linear_addmm_cpu_producer_attribution_status.json"),
    ]
    for layer in layers:
        layer_source = layer_paths(layer)
        paths.extend(
            [
                Path(layer_source["attention_bundle_dir"]),
                Path(layer_source["weighted_v"]),
                Path(layer_source["official_o_proj"]),
            ]
        )
    return paths


def base_status(classification: str) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "producer_probe": True,
        "dispatch_stability_probe": True,
        "oracle_device": "cpu",
        "cuda_available": None,
        "cuda_used": False,
        **GUARD_FALSE_FLAGS,
    }


def import_torch_then_checkpoint(config: dict[str, Any]) -> tuple[Any, Any, list[str]]:
    import torch

    notes: list[str] = []
    if config.get("num_threads") is not None:
        torch.set_num_threads(int(config["num_threads"]))
    if config.get("num_interop_threads") is not None:
        torch.set_num_interop_threads(int(config["num_interop_threads"]))
    if config.get("mkldnn_enabled") is not None:
        mkldnn = getattr(torch.backends, "mkldnn", None)
        if mkldnn is None or not hasattr(mkldnn, "enabled"):
            raise RuntimeError("torch.backends.mkldnn.enabled is unavailable")
        mkldnn.enabled = bool(config["mkldnn_enabled"])
        notes.append(f"mkldnn.enabled set to {mkldnn.enabled}")

    candidates = [
        REPO_ROOT.parent / "gpt-oss",
        REPO_ROOT.parents[1] / "gpt-oss",
    ]
    for candidate in candidates:
        if (candidate / "gpt_oss").is_dir():
            sys.path.insert(0, str(candidate))
            break
    from gpt_oss.torch.weights import Checkpoint

    return torch, Checkpoint, notes


def run_negative_controls(torch: Any, weighted_v: Any, weight: Any, bias: Any, official: Any, focus_lane: int) -> dict[str, Any]:
    controls = {
        "explicit_matmul_plus_bias": lambda: weighted_v @ weight.t() + bias,
        "explicit_einsum_plus_bias": lambda: torch.einsum("k,hk->h", weighted_v, weight) + bias,
        "explicit_unfused_bf16_bias": lambda: (
            torch.nn.functional.linear(weighted_v, weight, None) + bias
        ).to(torch.bfloat16),
    }
    results: dict[str, Any] = {}
    for name, thunk in controls.items():
        try:
            output = thunk()
            results[name] = make_result(torch, name, True, True, output, official, focus_lane, diagnostic_only=True)
        except Exception as exc:  # noqa: BLE001 - preserve diagnostic failure in status
            results[name] = make_result(torch, name, False, False, None, official, focus_lane, repr(exc), True)
    return results


def run_worker(args: argparse.Namespace) -> int:
    if args.worker_config is None or args.worker_output is None:
        raise ValueError("--worker requires --worker-config and --worker-output")
    config = load_json(args.worker_config)
    layers = [int(part) for part in config["layers"]]
    try:
        torch, Checkpoint, setting_notes = import_torch_then_checkpoint(config)
        checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
        layer_results = []
        for layer in layers:
            focus_lane = int(SAMPLED_LAYERS[layer]["focus_lane"])
            paths = layer_paths(layer)
            weighted_v_json = load_json(Path(paths["weighted_v"]))
            official_json = load_json(Path(paths["official_o_proj"]))
            weighted_v = torch.tensor(
                json_tensor_values(Path(paths["weighted_v"])), dtype=torch.float32, device="cpu"
            ).to(torch.bfloat16)
            official = torch.tensor(
                json_tensor_values(Path(paths["official_o_proj"])), dtype=torch.float32, device="cpu"
            ).to(torch.bfloat16)
            weight = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.weight")
            bias = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.bias")
            input_2d = weighted_v.unsqueeze(0)
            weight_t_2d = weight.t()

            for name, tensor in {
                "weighted_v": weighted_v,
                "o_proj_weight": weight,
                "o_proj_bias": bias,
                "official_output": official,
                "input_2d": input_2d,
                "weight_t_2d": weight_t_2d,
            }.items():
                assert_cpu_tensor(f"layer{layer}.{name}", tensor)

            addmm_output = torch.addmm(bias, input_2d, weight_t_2d).squeeze(0)
            assert_cpu_tensor(f"layer{layer}.addmm_output", addmm_output)
            layer_result: dict[str, Any] = {
                "layer_index": layer,
                "focus_lane": focus_lane,
                "source_paths": paths,
                "tensor_metadata": {
                    "weighted_v": tensor_metadata(torch, weighted_v),
                    "o_proj_weight": tensor_metadata(torch, weight),
                    "o_proj_bias": tensor_metadata(torch, bias),
                    "official_output": tensor_metadata(torch, official),
                    "input_2d": tensor_metadata(torch, input_2d, include_summary=False),
                    "weight_t_2d": tensor_metadata(torch, weight_t_2d, include_summary=False),
                },
                "addmm_output_metadata": tensor_metadata(torch, addmm_output),
                "comparison_vs_official": compare_tensors(torch, addmm_output, official, focus_lane),
                "addmm_output_values": [float(value) for value in addmm_output.detach().to("cpu").float().tolist()],
                "official_values": [float(value) for value in official.detach().to("cpu").float().tolist()],
            }
            if config.get("include_negative_controls"):
                layer_result["negative_controls"] = run_negative_controls(
                    torch, weighted_v, weight, bias, official, focus_lane
                )
            layer_results.append(layer_result)

        status = {
            "config_name": config["name"],
            "process_mode": "fresh_process",
            "executed": True,
            "available": True,
            "reason_unavailable": None,
            "requested_settings": config,
            "setting_notes": setting_notes,
            "environment": environment_metadata(torch),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_used": False,
            "layers": layer_results,
        }
        write_json(args.worker_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = {
            "config_name": config.get("name"),
            "process_mode": "fresh_process",
            "executed": False,
            "available": False,
            "reason_unavailable": repr(exc),
            "requested_settings": config,
            "cuda_used": False,
            "layers": [],
        }
        write_json(args.worker_output, status)
        return 1


def compare_value_lists(actual: list[float], expected: list[float], focus_lane: int) -> dict[str, Any]:
    mismatches = []
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    worst = None
    for index, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        diff = abs(actual_value - expected_value)
        total_abs_diff += diff
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst = {
                "index": index,
                "actual": actual_value,
                "expected": expected_value,
                "abs_diff": diff,
            }
        if diff != 0.0:
            if len(mismatches) < 8:
                mismatches.append(
                    {
                        "index": index,
                        "actual": actual_value,
                        "expected": expected_value,
                        "abs_diff": diff,
                    }
                )
    mismatch_count = sum(1 for actual_value, expected_value in zip(actual, expected, strict=True) if actual_value != expected_value)
    first = mismatches[0] if mismatches else None
    focus_diff = abs(actual[focus_lane] - expected[focus_lane])
    return {
        "value_count": len(expected),
        "mismatch_count": mismatch_count,
        "full_vector_mismatches": mismatch_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": total_abs_diff / len(expected) if expected else 0.0,
        "first_mismatch": first,
        "worst_mismatch": worst,
        "mismatch_samples": mismatches,
        "focus_lane": {
            "lane": focus_lane,
            "actual": actual[focus_lane],
            "expected": expected[focus_lane],
            "abs_diff": focus_diff,
            "matched": focus_diff == 0.0,
            "diagnostic_only": True,
        },
        "full_vector_cleared": mismatch_count == 0 and max_abs_diff == 0.0,
    }


def config_specs(layers: list[int]) -> list[dict[str, Any]]:
    base = {
        "num_threads": None,
        "num_interop_threads": None,
        "mkldnn_enabled": None,
        "env_overrides": {},
        "layers": layers,
        "include_negative_controls": False,
    }
    specs = [
        {"name": "baseline_default", "include_negative_controls": True},
        {"name": "torch_num_threads_1", "num_threads": 1},
        {"name": "torch_num_threads_8", "num_threads": 8},
        {"name": "torch_num_threads_1_interop_1", "num_threads": 1, "num_interop_threads": 1},
        {"name": "torch_num_threads_8_interop_8", "num_threads": 8, "num_interop_threads": 8},
        {"name": "mkldnn_enabled_true", "mkldnn_enabled": True},
        {"name": "mkldnn_enabled_false", "mkldnn_enabled": False},
        {"name": "env_omp_num_threads_1", "env_overrides": {"OMP_NUM_THREADS": "1"}},
        {"name": "env_omp_num_threads_8", "env_overrides": {"OMP_NUM_THREADS": "8"}},
        {"name": "env_mkl_num_threads_1", "env_overrides": {"MKL_NUM_THREADS": "1"}},
        {"name": "env_mkl_num_threads_8", "env_overrides": {"MKL_NUM_THREADS": "8"}},
        {
            "name": "env_omp_mkl_num_threads_1",
            "env_overrides": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        },
        {
            "name": "env_omp_mkl_num_threads_8",
            "env_overrides": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
        },
    ]
    merged = []
    for spec in specs:
        item = dict(base)
        item.update(spec)
        item["layers"] = layers
        merged.append(item)
    return merged


def run_config_worker(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / f"{config['name']}_config.json"
    worker_output = args.output_dir / f"{config['name']}_worker_status.json"
    write_json(config_path, config)
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in config.get("env_overrides", {}).items()})
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-config",
        str(config_path),
        "--worker-output",
        str(worker_output),
        "--model",
        str(args.model),
    ]
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=300, check=False)
    if worker_output.is_file():
        status = load_json(worker_output)
    else:
        status = {
            "config_name": config["name"],
            "process_mode": "fresh_process",
            "executed": False,
            "available": False,
            "reason_unavailable": "worker did not write status",
            "requested_settings": config,
            "cuda_used": False,
            "layers": [],
        }
    status["subprocess"] = {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }
    return status


def strip_values_from_layer(layer: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(layer)
    stripped.pop("addmm_output_values", None)
    stripped.pop("official_values", None)
    return stripped


def attach_baseline_comparisons(config_result: dict[str, Any], baseline_by_layer: dict[int, dict[str, Any]]) -> dict[str, Any]:
    result = dict(config_result)
    layers = []
    if not result.get("executed"):
        result["stable_vs_baseline"] = False
        result["stable_vs_official"] = False
        result["changed_layers"] = []
        result["skipped_or_unavailable"] = True
        return result

    changed_layers = []
    official_fail_layers = []
    for layer in result["layers"]:
        layer_index = int(layer["layer_index"])
        focus_lane = int(layer["focus_lane"])
        baseline_layer = baseline_by_layer[layer_index]
        comparison_vs_baseline = compare_value_lists(
            layer["addmm_output_values"], baseline_layer["addmm_output_values"], focus_lane
        )
        comparison_vs_official = compare_value_lists(layer["addmm_output_values"], layer["official_values"], focus_lane)
        stripped = strip_values_from_layer(layer)
        stripped["comparison_vs_baseline"] = comparison_vs_baseline
        stripped["comparison_vs_official"] = comparison_vs_official
        layers.append(stripped)
        if not comparison_vs_baseline["full_vector_cleared"]:
            changed_layers.append(layer_index)
        if not comparison_vs_official["full_vector_cleared"]:
            official_fail_layers.append(layer_index)

    result["layers"] = layers
    result["stable_vs_baseline"] = not changed_layers
    result["stable_vs_official"] = not official_fail_layers
    result["changed_layers"] = changed_layers
    result["official_fail_layers"] = official_fail_layers
    result["skipped_or_unavailable"] = False
    return result


def classify(config_results: list[dict[str, Any]]) -> str:
    if any(result.get("executed") and result.get("changed_layers") for result in config_results):
        return "fused_linear_addmm_cpu_dispatch_stability_unstable"
    if any(result.get("executed") and result.get("official_fail_layers") for result in config_results):
        return "fused_linear_addmm_cpu_dispatch_stability_unstable"
    if any(not result.get("executed") for result in config_results):
        return "fused_linear_addmm_cpu_dispatch_stability_inconclusive"
    return "fused_linear_addmm_cpu_dispatch_stability_stable"


def missing_artifacts_status(missing: list[str], layers: list[int], model: Path) -> dict[str, Any]:
    status = base_status("fused_linear_addmm_cpu_dispatch_stability_blocked_by_missing_artifacts")
    status.update(
        {
            "model": str(model),
            "sampled_layers": layers,
            "missing_artifacts": missing,
            "source_statuses": {
                "arithmetic_contract": "/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json",
                "addmm_boundary_localization": "/tmp/fused_linear_addmm_addmm_boundary_localization_status.json",
                "cpu_producer_attribution": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
            },
        }
    )
    return status


def failure_status(error: str) -> dict[str, Any]:
    status = base_status("fused_linear_addmm_cpu_dispatch_stability_failed")
    status["error"] = error
    return status


def run_controller(args: argparse.Namespace) -> int:
    try:
        layers = [int(part) for part in args.layers.split(",") if part.strip()]
        unknown = [layer for layer in layers if layer not in SAMPLED_LAYERS]
        if unknown:
            raise ValueError(f"unsupported sampled layers: {unknown}")

        missing = [str(path) for path in required_source_paths(layers, args.model) if not path.exists()]
        if missing:
            status = missing_artifacts_status(missing, layers, args.model)
            write_json(args.status_output, status)
            return 1

        specs = config_specs(layers)
        raw_results = [run_config_worker(args, config) for config in specs]
        baseline = next((result for result in raw_results if result.get("config_name") == "baseline_default"), None)
        if not baseline or not baseline.get("executed"):
            status = failure_status("baseline_default did not execute")
            status["configuration_results"] = raw_results
            write_json(args.status_output, status)
            return 1

        baseline_by_layer = {int(layer["layer_index"]): layer for layer in baseline["layers"]}
        config_results = [attach_baseline_comparisons(result, baseline_by_layer) for result in raw_results]
        classification = classify(config_results)
        baseline_sanitized = next(result for result in config_results if result["config_name"] == "baseline_default")
        cuda_available = bool(baseline_sanitized.get("cuda_available"))
        changed_configs = [
            {
                "config_name": result["config_name"],
                "changed_layers": result.get("changed_layers", []),
                "official_fail_layers": result.get("official_fail_layers", []),
            }
            for result in config_results
            if result.get("changed_layers") or result.get("official_fail_layers")
        ]
        skipped = [
            {
                "config_name": result.get("config_name"),
                "reason": result.get("reason_unavailable"),
                "returncode": result.get("subprocess", {}).get("returncode"),
                "stderr_tail": result.get("subprocess", {}).get("stderr_tail", []),
            }
            for result in config_results
            if not result.get("executed")
        ]
        negative_controls = {}
        for layer in baseline_sanitized.get("layers", []):
            if "negative_controls" in layer:
                negative_controls[str(layer["layer_index"])] = {
                    name: {
                        "full_vector_cleared": value.get("full_vector_cleared"),
                        "mismatch_count": value.get("comparison", {}).get("mismatch_count"),
                        "max_abs_diff": value.get("comparison", {}).get("max_abs_diff"),
                    }
                    for name, value in layer["negative_controls"].items()
                }

        status = base_status(classification)
        status.update(
            {
                "cuda_available": cuda_available,
                "model": str(args.model),
                "operator": "attention_o_proj",
                "reference_operation": "torch.addmm(bias, input_2d, weight_t_2d)",
                "sampled_layers": layers,
                "source_statuses": {
                    "arithmetic_contract": "/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json",
                    "addmm_boundary_localization": "/tmp/fused_linear_addmm_addmm_boundary_localization_status.json",
                    "cpu_producer_attribution": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
                },
                "configurations_requested": [
                    {
                        "config_name": spec["name"],
                        "process_mode": "fresh_process",
                        "num_threads": spec.get("num_threads"),
                        "num_interop_threads": spec.get("num_interop_threads"),
                        "mkldnn_enabled": spec.get("mkldnn_enabled"),
                        "env_overrides": spec.get("env_overrides", {}),
                    }
                    for spec in specs
                ],
                "configuration_results": config_results,
                "negative_controls_baseline": negative_controls,
                "summary": {
                    "configurations_tested": [result["config_name"] for result in config_results if result.get("executed")],
                    "skipped_or_unavailable_configurations": skipped,
                    "changed_configs": changed_configs,
                    "addmm_output_changed_under_any_tested_setting": bool(changed_configs),
                    "gate_a_result": "passed"
                    if classification == "fused_linear_addmm_cpu_dispatch_stability_stable"
                    else "failed"
                    if classification == "fused_linear_addmm_cpu_dispatch_stability_unstable"
                    else "inconclusive",
                    "baseline_negative_controls_remain_negative": all(
                        not control["full_vector_cleared"]
                        for layer_controls in negative_controls.values()
                        for control in layer_controls.values()
                    ),
                },
                "interpretation": {
                    "cpu_dispatch_stable": classification == "fused_linear_addmm_cpu_dispatch_stability_stable",
                    "rust_cpu_policy_synthesis_authorized": False,
                    "cuda_mirror_authorized": False,
                    "backend_identity_claim": "none",
                    "next_bounded_step": "Review Gate A dispatch stability before considering Rust CPU policy synthesis.",
                },
            }
        )
        write_json(args.status_output, status)
        write_json(args.output_dir / "cpu_dispatch_stability_status.json", status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = failure_status(repr(exc))
        write_json(args.status_output, status)
        return 1


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
