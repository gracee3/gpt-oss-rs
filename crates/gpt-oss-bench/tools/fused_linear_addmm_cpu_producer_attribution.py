#!/usr/bin/env python3
"""CPU-first producer attribution for attention o-proj fused linear/addmm.

This is an oracle evidence probe. It reconstructs the sampled final-token
attention o-proj seam from existing ordered attention bundle JSON artifacts and
per-layer checkpoint tensors, then compares CPU Torch API variants against the
official producer/API output artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/data/models/openai/gpt-oss-20b-full-attn-restricted-integration")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_cpu_producer_attribution_status.json")
SAMPLED_LAYERS = {
    6: {"role": "historical_blocker", "focus_lane": 22},
    10: {"role": "pairwise_clear_control", "focus_lane": 915},
    13: {"role": "blocked_family", "focus_lane": 151},
    16: {"role": "blocked_family", "focus_lane": 2666},
    18: {"role": "blocked_family", "focus_lane": 63},
    21: {"role": "raw_qk_solved_oproj_blocked", "focus_lane": 2807},
}
GUARD_FALSE_FLAGS = {
    "runtime_behavior_changed": False,
    "production_routing_changed": False,
    "cuda_kernels_changed": False,
    "backend_selected": False,
    "implementation_authorized": False,
    "consumer_revalidation_authorized": False,
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
        description="CPU producer attribution for fused linear/addmm attention o-proj."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/fused_linear_addmm_cpu_producer_attribution"))
    parser.add_argument("--layers", default="6,10,13,16,18,21")
    parser.add_argument("--skip-backend-verbose", action="store_true")
    parser.add_argument("--backend-verbose-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--layer-index", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def import_torch_and_checkpoint() -> tuple[Any, Any]:
    import torch

    candidates = [
        REPO_ROOT.parent / "gpt-oss",
        REPO_ROOT.parents[1] / "gpt-oss",
    ]
    for candidate in candidates:
        if (candidate / "gpt_oss").is_dir():
            sys.path.insert(0, str(candidate))
            break
    from gpt_oss.torch.weights import Checkpoint

    return torch, Checkpoint


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def json_tensor_values(path: Path) -> list[float]:
    data = load_json(path)
    values = data.get("values")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain a values array")
    return [float(v) for v in values]


def finite_summary(torch: Any, tensor: Any) -> dict[str, Any]:
    flat = tensor.detach().to("cpu").float().reshape(-1)
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    summary: dict[str, Any] = {
        "count": int(flat.numel()),
        "finite_count": finite_count,
        "all_finite": finite_count == int(flat.numel()),
    }
    if finite_count:
        finite_values = flat[finite]
        summary.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
            }
        )
    return summary


def tensor_metadata(torch: Any, tensor: Any, include_summary: bool = True) -> dict[str, Any]:
    meta = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()) if hasattr(tensor, "storage_offset") else None,
    }
    if include_summary:
        meta["finite_summary"] = finite_summary(torch, tensor)
    return meta


def assert_cpu_tensor(name: str, tensor: Any) -> None:
    if str(tensor.device) != "cpu":
        raise RuntimeError(f"{name} unexpectedly landed on {tensor.device}; CPU oracle probe fails closed")


def compare_tensors(torch: Any, actual: Any, expected: Any, focus_lane: int) -> dict[str, Any]:
    actual_cpu = actual.detach().to("cpu")
    expected_cpu = expected.detach().to("cpu")
    diff = (actual_cpu.float() - expected_cpu.float()).abs()
    mismatch_mask = diff != 0
    mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).reshape(-1)
    mismatches = int(mismatch_indices.numel())
    first_mismatch = None
    worst_mismatch = None
    samples = []
    if mismatches:
        first_index = int(mismatch_indices[0].item())
        worst_index = int(torch.argmax(diff).item())
        first_mismatch = {
            "index": first_index,
            "actual": float(actual_cpu[first_index].float().item()),
            "expected": float(expected_cpu[first_index].float().item()),
            "abs_diff": float(diff[first_index].item()),
        }
        worst_mismatch = {
            "index": worst_index,
            "actual": float(actual_cpu[worst_index].float().item()),
            "expected": float(expected_cpu[worst_index].float().item()),
            "abs_diff": float(diff[worst_index].item()),
        }
        for index_tensor in mismatch_indices[:8]:
            index = int(index_tensor.item())
            samples.append(
                {
                    "index": index,
                    "actual": float(actual_cpu[index].float().item()),
                    "expected": float(expected_cpu[index].float().item()),
                    "abs_diff": float(diff[index].item()),
                }
            )
    focus_diff = float(diff[focus_lane].item())
    return {
        "value_count": int(expected_cpu.numel()),
        "mismatch_count": mismatches,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "first_mismatch": first_mismatch,
        "worst_mismatch": worst_mismatch,
        "mismatch_samples": samples,
        "focus_lane": {
            "lane": focus_lane,
            "actual": float(actual_cpu[focus_lane].float().item()),
            "official": float(expected_cpu[focus_lane].float().item()),
            "abs_diff": focus_diff,
            "matched": focus_diff == 0.0,
            "diagnostic_only": True,
        },
        "full_vector_cleared": mismatches == 0 and float(diff.max().item()) == 0.0,
    }


def make_result(
    torch: Any,
    name: str,
    available: bool,
    executed: bool,
    output: Any | None,
    official: Any,
    focus_lane: int,
    reason: str | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "operator": name,
        "available": available,
        "executed": executed,
        "diagnostic_only": diagnostic_only,
        "reason_unavailable": reason,
    }
    if output is None:
        result.update(
            {
                "output": None,
                "full_vector_cleared": False,
                "focus_lane_result_diagnostic_only": True,
            }
        )
        return result
    assert_cpu_tensor(name, output)
    result["output"] = tensor_metadata(torch, output, include_summary=False)
    result["comparison"] = compare_tensors(torch, output, official, focus_lane)
    result["full_vector_cleared"] = result["comparison"]["full_vector_cleared"]
    result["focus_lane_result_diagnostic_only"] = True
    return result


def run_api_variants(torch: Any, weighted_v: Any, weight: Any, bias: Any, official: Any, focus_lane: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True, dtype=weight.dtype, device="cpu")
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    variants = [
        ("module_call", lambda: linear(weighted_v)),
        ("torch_nn_functional_linear", lambda: torch.nn.functional.linear(weighted_v, weight, bias)),
        ("torch_C_nn_linear", lambda: torch._C._nn.linear(weighted_v, weight, bias)),
        ("torch_addmm_fused_bias", lambda: torch.addmm(bias, weighted_v.unsqueeze(0), weight.t()).squeeze(0)),
        ("explicit_matmul", lambda: weight @ weighted_v + bias),
        ("explicit_einsum", lambda: torch.einsum("hk,k->h", weight, weighted_v) + bias),
        ("explicit_unfused_bias", lambda: torch.nn.functional.linear(weighted_v, weight, None) + bias),
    ]
    for name, thunk in variants:
        try:
            output = thunk()
            results[name] = make_result(torch, name, True, True, output, official, focus_lane)
        except Exception as exc:  # noqa: BLE001 - status JSON should preserve probe failures
            results[name] = make_result(torch, name, False, False, None, official, focus_lane, repr(exc))
    return results


def run_layout_guards(torch: Any, weighted_v: Any, weight: Any, bias: Any, official: Any, focus_lane: int) -> dict[str, Any]:
    guards = [
        ("original_layout", weighted_v, weight, bias),
        ("input_contiguous_clone", weighted_v.contiguous().clone(), weight, bias),
        ("weight_contiguous_clone", weighted_v, weight.contiguous().clone(), bias),
        ("bias_clone", weighted_v, weight, bias.clone()),
        ("all_contiguous_clones", weighted_v.contiguous().clone(), weight.contiguous().clone(), bias.clone()),
        ("input_clone_weight_clone", weighted_v.clone(), weight.clone(), bias),
    ]
    try:
        noncontig_weight = weight.t().contiguous().t()
        if list(noncontig_weight.shape) == list(weight.shape) and not noncontig_weight.is_contiguous():
            guards.append(("weight_noncontiguous_same_shape", weighted_v, noncontig_weight, bias))
    except Exception:
        pass

    results: dict[str, Any] = {}
    for name, input_tensor, weight_tensor, bias_tensor in guards:
        try:
            output = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
            results[name] = make_result(torch, f"layout_guard_{name}", True, True, output, official, focus_lane)
            results[name]["input_metadata"] = {
                "weighted_v": tensor_metadata(torch, input_tensor, include_summary=False),
                "weight": tensor_metadata(torch, weight_tensor, include_summary=False),
                "bias": tensor_metadata(torch, bias_tensor, include_summary=False),
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = make_result(torch, f"layout_guard_{name}", False, False, None, official, focus_lane, repr(exc))
    return results


def profile_cpu_ops(torch: Any, weighted_v: Any, weight: Any, bias: Any) -> dict[str, Any]:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except Exception as exc:  # noqa: BLE001
        return {"succeeded": False, "reason": repr(exc), "events": [], "backend_identity": "blocked_by_profiler"}

    try:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("cpu_fused_linear_attribution"):
                torch.nn.functional.linear(weighted_v, weight, bias)
                torch.addmm(bias, weighted_v.unsqueeze(0), weight.t()).squeeze(0)
                weight @ weighted_v + bias
                torch.einsum("hk,k->h", weight, weighted_v) + bias
        events = []
        for event in prof.key_averages()[:40]:
            events.append(
                {
                    "key": event.key,
                    "count": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "input_shapes": str(getattr(event, "input_shapes", "")),
                }
            )
        return {
            "succeeded": True,
            "events": events,
            "operator_names": [event["key"] for event in events],
            "backend_identity": "inconclusive",
            "note": "ATen/profiler operator names are informative but do not by themselves identify the CPU kernel backend.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"succeeded": False, "reason": repr(exc), "events": [], "backend_identity": "blocked_by_profiler"}


def run_backend_verbose_child(args: argparse.Namespace) -> None:
    torch, Checkpoint = import_torch_and_checkpoint()
    layer = args.layer_index or 6
    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    weight = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.weight")
    bias = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.bias")
    weighted_v_values = json_tensor_values(Path(f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json"))
    weighted_v = torch.tensor(weighted_v_values, dtype=torch.float32, device="cpu").to(torch.bfloat16)
    with torch.no_grad():
        torch.nn.functional.linear(weighted_v, weight, bias)
        torch.addmm(bias, weighted_v.unsqueeze(0), weight.t()).squeeze(0)
    print(json.dumps({"child_layer": layer, "completed": True}))


def run_verbose_subprocess(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({"ONEDNN_VERBOSE": "1", "DNNL_VERBOSE": "1", "MKL_VERBOSE": "1"})
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend-verbose-child",
        "--layer-index",
        str(layer),
        "--model",
        str(args.model),
    ]
    try:
        completed = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120, check=False)
        return {
            "attempted": True,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout.splitlines()[-40:],
            "stderr_tail": completed.stderr.splitlines()[-40:],
            "backend_identity": "inconclusive",
            "note": "Verbose output is captured as attribution evidence only; it is not treated as backend selection.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "failed": True, "reason": repr(exc), "backend_identity": "inconclusive"}


def torch_config(torch: Any) -> str:
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            torch.__config__.show()
        return stream.getvalue()
    except Exception as exc:  # noqa: BLE001
        return f"<torch.__config__.show failed: {exc!r}>"


def environment_metadata(torch: Any) -> dict[str, Any]:
    interop_threads = None
    try:
        interop_threads = int(torch.get_num_interop_threads())
    except Exception:
        pass
    return {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "torch_version": str(torch.__version__),
        "torch_config": torch_config(torch),
        "torch_get_num_threads": int(torch.get_num_threads()),
        "torch_get_num_interop_threads": interop_threads,
        "torch_backends_mkldnn_enabled": bool(getattr(torch.backends.mkldnn, "enabled", False)),
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "env": {
            name: os.environ.get(name)
            for name in [
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "ONEDNN_VERBOSE",
                "DNNL_VERBOSE",
                "MKL_VERBOSE",
            ]
        },
    }


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle": f"/tmp/layer{layer}_ordered_attention_bundle_status.json",
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
        "api_probe_status": f"/tmp/layer{layer}_attention_oproj_api_probe_status.json",
        "api_probe_dir": f"/tmp/layer{layer}_attention_oproj_api_probe",
    }


def run_layer(torch: Any, checkpoint: Any, layer: int, args: argparse.Namespace) -> dict[str, Any]:
    focus_lane = int(SAMPLED_LAYERS[layer]["focus_lane"])
    paths = layer_paths(layer)
    weighted_v_json = load_json(Path(paths["weighted_v"]))
    official_json = load_json(Path(paths["official_o_proj"]))
    weighted_v = torch.tensor(weighted_v_json["values"], dtype=torch.float32, device="cpu").to(torch.bfloat16)
    official = torch.tensor(official_json["values"], dtype=torch.float32, device="cpu").to(torch.bfloat16)
    weight = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.weight")
    bias = checkpoint.get(f"model.layers.{layer}.self_attn.o_proj.bias")

    for name, tensor in {
        "weighted_v": weighted_v,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "official_output": official,
    }.items():
        assert_cpu_tensor(f"layer{layer}.{name}", tensor)

    api_results = run_api_variants(torch, weighted_v, weight, bias, official, focus_lane)
    layout_results = run_layout_guards(torch, weighted_v, weight, bias, official, focus_lane)
    profiler = profile_cpu_ops(torch, weighted_v, weight, bias)

    api_probe_status = None
    api_probe_path = Path(paths["api_probe_status"])
    if api_probe_path.is_file():
        api_probe_status = load_json(api_probe_path)

    clears = [
        name
        for name, result in api_results.items()
        if result.get("full_vector_cleared") is True
    ]
    negative_controls = [
        name
        for name in ["explicit_matmul", "explicit_einsum", "explicit_unfused_bias"]
        if not api_results.get(name, {}).get("full_vector_cleared", False)
    ]

    return {
        "layer_index": layer,
        "role": SAMPLED_LAYERS[layer]["role"],
        "operator": "attention_o_proj",
        "focus_lane": focus_lane,
        "source_paths": paths,
        "source_status_classification": api_probe_status.get("classification") if isinstance(api_probe_status, dict) else None,
        "tensor_metadata": {
            "weighted_v": tensor_metadata(torch, weighted_v),
            "o_proj_weight": tensor_metadata(torch, weight),
            "o_proj_bias": tensor_metadata(torch, bias),
            "official_output": tensor_metadata(torch, official),
            "provenance": {
                "weighted_v_boundary": weighted_v_json.get("boundary"),
                "official_o_proj_boundary": official_json.get("boundary"),
            },
        },
        "api_variant_results": api_results,
        "layout_perturbation_results": layout_results,
        "profiler": profiler,
        "summary": {
            "full_vector_clear_api_variants": clears,
            "negative_controls_rejected": negative_controls,
            "explicit_matmul_einsum_unfused_bias_differ": len(negative_controls) == 3,
            "focus_lane_only_promoted": False,
        },
    }


def failure_status(classification: str, error: str) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "producer_probe": True,
        "oracle_device": "cpu",
        "cuda_available": None,
        "cuda_used": False,
        **GUARD_FALSE_FLAGS,
        "error": error,
    }


def main() -> int:
    args = parse_args()
    if args.backend_verbose_child:
        run_backend_verbose_child(args)
        return 0

    try:
        torch, Checkpoint = import_torch_and_checkpoint()
        layers = [int(part) for part in args.layers.split(",") if part.strip()]
        unknown = [layer for layer in layers if layer not in SAMPLED_LAYERS]
        if unknown:
            raise ValueError(f"unsupported sampled layers: {unknown}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        env = environment_metadata(torch)
        checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
        layer_results = []
        for layer in layers:
            layer_results.append(run_layer(torch, checkpoint, layer, args))

        profiler_success = any(layer["profiler"].get("succeeded") for layer in layer_results)
        verbose = None
        if not args.skip_backend_verbose and layers:
            verbose = run_verbose_subprocess(args, layers[0])

        all_module_clear = all(
            result["api_variant_results"].get("torch_nn_functional_linear", {}).get("full_vector_cleared")
            for result in layer_results
        )
        all_negative_differ = all(
            result["summary"].get("explicit_matmul_einsum_unfused_bias_differ")
            for result in layer_results
        )
        classification = (
            "fused_linear_addmm_cpu_backend_attribution_inconclusive"
            if profiler_success
            else "fused_linear_addmm_cpu_producer_attribution_blocked_by_profiler"
        )
        status = {
            "classification": classification,
            "validation_only": True,
            "producer_probe": True,
            "oracle_device": "cpu",
            "cuda_available": bool(env["torch_cuda_is_available"]),
            "cuda_used": False,
            **GUARD_FALSE_FLAGS,
            "model": str(args.model),
            "operator": "attention_o_proj",
            "dtype": "torch.bfloat16",
            "prompt_case": "developer-message-user-smoke",
            "sampled_layers": layers,
            "environment": env,
            "source_statuses": {
                "producer_api_13_16_10": "/tmp/o_proj_producer_api_probes_13_16_10_status.json",
                "producer_api_18_21": "/tmp/o_proj_producer_api_probes_18_21_status.json",
                "fused_linear_addmm_status_scaffold": "/tmp/fused_linear_addmm_status_scaffold.json",
                "layer6_api_probe": "/tmp/layer6_attention_oproj_api_probe_status.json",
            },
            "layers": layer_results,
            "backend_attribution": {
                "profiler_succeeded": profiler_success,
                "backend_verbose": verbose,
                "backend_identity": "inconclusive",
                "backend_selected": False,
                "note": "CPU profiler/verbose evidence is attribution telemetry only. It does not prove backend identity strongly enough to select a backend.",
            },
            "interpretation": {
                "producer_api_reference_reproduced_on_cpu": bool(all_module_clear),
                "negative_controls_remain_negative": bool(all_negative_differ),
                "explains_matmul_einsum_unfused_bias_difference": bool(all_module_clear and all_negative_differ),
                "explains_rust_cuda_helper_difference": "partially: the CPU Torch API reference still differs from explicit matmul/einsum/unfused-bias forms, but profiler/backend attribution is inconclusive and no backend is selected.",
                "focus_lane_only_clears_rejected": True,
                "layout_perturbations_are_guards_only": True,
                "backend_identity_claim": "inconclusive",
                "next_bounded_step": "Review CPU producer attribution before any status-only backend discriminator or candidate execution.",
            },
        }
        write_json(args.status_output, status)
        write_json(args.output_dir / "cpu_producer_attribution_status.json", status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = failure_status("fused_linear_addmm_cpu_producer_attribution_failed", repr(exc))
        try:
            write_json(args.status_output, status)
        finally:
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
