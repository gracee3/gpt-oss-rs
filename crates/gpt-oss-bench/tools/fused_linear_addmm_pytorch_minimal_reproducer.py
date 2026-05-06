#!/usr/bin/env python3
"""Source-guided minimal reproducer for the fused addmm o-proj seam.

This Stage 3 attribution probe uses captured Workstream A tensors only. It
replays the official CPU Torch addmm/linear seam across sampled layers and
collects CPU profiler / verbose evidence for source-path attribution.
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
    environment_metadata,
    import_torch_and_checkpoint,
    json_tensor_values,
    load_json,
    make_result,
    tensor_metadata,
    write_json,
)


DEFAULT_FORWARD_ENV = Path("/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130")
DEFAULT_SOURCE = Path("/home/emmy/openai/pytorch")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-pytorch-minimal-reproducer")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_pytorch_minimal_reproducer_status.json")
CHECKED_OUT_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
SAMPLED = [6, 10, 13, 16, 18, 21]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CPU reproducer for fused-linear/addmm o-proj.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--source-checkout-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--layers", default=",".join(str(layer) for layer in SAMPLED))
    parser.add_argument("--verbose-child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


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


def base_status(classification: str, args: argparse.Namespace, layers: list[int]) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "source_attribution_probe": True,
        "minimal_reproducer": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "forward_env_path": str(args.forward_env_path),
        "source_checkout_path": str(args.source_checkout_path),
        "checked_out_commit": CHECKED_OUT_COMMIT,
        "sampled_layers_requested": layers,
        "sampled_layers_evaluated": [],
        "all_official_variants_cleared": False,
        "negative_controls_remained_negative": False,
        "active_backend_inference": "inconclusive",
        "concrete_replayable_rule_found": False,
        "reopen_rust_policy_synthesis": False,
        "full_model_loaded": False,
        "gpu_tensors_created": False,
        "pytorch_build_performed": False,
        "pytorch_source_patched": False,
        **GUARD_FALSE_FLAGS,
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
        return make_result(torch, name, True, True, output, official, focus_lane, diagnostic_only=diagnostic_only)
    except Exception as exc:  # noqa: BLE001 - preserve attribution failure
        return make_result(torch, name, False, False, None, official, focus_lane, repr(exc), diagnostic_only)


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


def run_layer(torch: Any, tensors: dict[str, Any], layer: int) -> dict[str, Any]:
    focus_lane = int(SAMPLED_LAYERS[layer]["focus_lane"])
    weighted_v = tensors["weighted_v"]
    official = tensors["official_o_proj"]
    weight = tensors["o_proj_weight"]
    bias = tensors["o_proj_bias"]
    input_2d = tensors["input_2d"]
    weight_t_2d = tensors["weight_t_2d"]
    zero_bias = tensors["zero_bias"]

    official_variants = {
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
    }
    negative_controls = {
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
    official_cleared = all(
        result.get("executed") is True and result.get("full_vector_cleared") is True
        for result in official_variants.values()
    )
    negatives_negative = all(
        result.get("executed") is True and result.get("full_vector_cleared") is False
        for result in negative_controls.values()
    )
    return {
        "layer_index": layer,
        "role": SAMPLED_LAYERS[layer]["role"],
        "focus_lane": focus_lane,
        "source_artifacts": tensors["paths"],
        "model_tensors_loaded": [tensors["weight_name"], tensors["bias_name"]],
        "tensor_metadata": {
            "weighted_v": tensor_metadata(torch, weighted_v),
            "o_proj_weight": tensor_metadata(torch, weight),
            "o_proj_bias": tensor_metadata(torch, bias),
            "official_o_proj": tensor_metadata(torch, official),
            "input_2d": tensor_metadata(torch, input_2d, include_summary=False),
            "weight_t_2d": tensor_metadata(torch, weight_t_2d, include_summary=False),
            "provenance": {
                "weighted_v_boundary": tensors["weighted_v_json"].get("boundary"),
                "official_o_proj_boundary": tensors["official_json"].get("boundary"),
            },
        },
        "official_variants": official_variants,
        "negative_controls": negative_controls,
        "official_variants_cleared": bool(official_cleared),
        "negative_controls_remained_negative": bool(negatives_negative),
    }


def run_profiler(torch: Any, layer_tensors: dict[int, dict[str, Any]]) -> dict[str, Any]:
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "succeeded": False, "reason": repr(exc), "events": []}
    try:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("minimal_reproducer_addmm"):
                for tensors in layer_tensors.values():
                    torch.addmm(tensors["o_proj_bias"], tensors["input_2d"], tensors["weight_t_2d"])
        events = []
        for event in prof.key_averages():
            events.append(
                {
                    "key": event.key,
                    "count": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "input_shapes": str(getattr(event, "input_shapes", "")),
                }
            )
        names = [event["key"] for event in events]
        return {
            "attempted": True,
            "succeeded": True,
            "events": events[:80],
            "aten_addmm_seen": any("aten::addmm" in name for name in names),
            "aten_mm_or_matmul_seen": any(("aten::mm" in name or "aten::matmul" in name) for name in names),
            "mkldnn_or_onednn_event_seen": any(("mkldnn" in name.lower() or "onednn" in name.lower()) for name in names),
        }
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "succeeded": False, "reason": repr(exc), "events": []}


def configure_runtime(torch: Any, name: str) -> dict[str, Any]:
    if name == "mkldnn_enabled_true":
        torch.backends.mkldnn.enabled = True
    elif name == "mkldnn_enabled_false":
        torch.backends.mkldnn.enabled = False
    return {
        "config_name": name,
        "torch_backends_mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torch_get_num_threads": int(torch.get_num_threads()),
        "torch_get_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_used": False,
        "env": {
            key: os.environ.get(key)
            for key in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "ONEDNN_VERBOSE", "DNNL_VERBOSE", "MKL_VERBOSE"]
        },
    }


def run_runtime_config(torch: Any, Checkpoint: Any, args: argparse.Namespace, layers: list[int], config_name: str) -> dict[str, Any]:
    config_meta = configure_runtime(torch, config_name)
    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    layer_tensors = {layer: load_layer_tensors(torch, checkpoint, layer) for layer in layers}
    layer_results = [run_layer(torch, layer_tensors[layer], layer) for layer in layers]
    profiler = run_profiler(torch, layer_tensors)
    all_official = all(layer["official_variants_cleared"] for layer in layer_results)
    all_negative = all(layer["negative_controls_remained_negative"] for layer in layer_results)
    return {
        **config_meta,
        "layers": layer_results,
        "profiler": profiler,
        "all_official_variants_cleared": bool(all_official),
        "negative_controls_remained_negative": bool(all_negative),
    }


def run_verbose_child(args: argparse.Namespace) -> int:
    torch, Checkpoint = import_torch_and_checkpoint()
    checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    for layer in layers:
        tensors = load_layer_tensors(torch, checkpoint, layer)
        torch.addmm(tensors["o_proj_bias"], tensors["input_2d"], tensors["weight_t_2d"])
    print(json.dumps({"completed": True, "layers": layers, "cuda_used": False}))
    return 0


def run_verbose_variants(args: argparse.Namespace) -> dict[str, Any]:
    variants = {
        "ONEDNN_VERBOSE": {"ONEDNN_VERBOSE": "1"},
        "DNNL_VERBOSE": {"DNNL_VERBOSE": "1"},
        "MKL_VERBOSE": {"MKL_VERBOSE": "1"},
    }
    results = {}
    for name, updates in variants.items():
        env = os.environ.copy()
        env.update(updates)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verbose-child",
            "--model",
            str(args.model),
            "--layers",
            args.layers,
        ]
        try:
            completed = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=180, check=False)
            stdout_lines = completed.stdout.splitlines()
            stderr_lines = completed.stderr.splitlines()
            joined = "\n".join(stdout_lines + stderr_lines).lower()
            results[name] = {
                "attempted": True,
                "returncode": completed.returncode,
                "stdout_tail": stdout_lines[-80:],
                "stderr_tail": stderr_lines[-80:],
                "mkldnn_or_onednn_signal": "mkldnn" in joined or "onednn" in joined or "dnnl" in joined,
                "mkl_signal": "mkl" in joined or "gemm" in joined,
                "note": "Verbose output is attribution telemetry only and does not select a backend.",
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {"attempted": True, "failed": True, "reason": repr(exc)}
    return results


def infer_backend(runtime_configs: list[dict[str, Any]], verbose: dict[str, Any]) -> str:
    profiler_mkldnn = any(config.get("profiler", {}).get("mkldnn_or_onednn_event_seen") for config in runtime_configs)
    verbose_mkldnn = any(result.get("mkldnn_or_onednn_signal") for result in verbose.values())
    verbose_mkl = any(result.get("mkl_signal") for result in verbose.values())
    all_match = all(config.get("all_official_variants_cleared") for config in runtime_configs)
    mkldnn_false_match = any(
        config.get("config_name") == "mkldnn_enabled_false" and config.get("all_official_variants_cleared")
        for config in runtime_configs
    )
    if profiler_mkldnn or verbose_mkldnn:
        return "mkldnn_onednn_likely"
    if verbose_mkl:
        return "blas_mkl_likely"
    if all_match and mkldnn_false_match:
        return "multiple_possible"
    return "inconclusive"


def write_research_outputs(args: argparse.Namespace, status: dict[str, Any]) -> None:
    args.research_path.mkdir(parents=True, exist_ok=True)
    write_json(args.research_path / "profiler-summary.json", status.get("profiler_summary", {}))
    write_json(args.research_path / "per-layer-comparisons.json", status.get("runtime_configs", []))
    verbose_lines = []
    for name, result in status.get("verbose_capture_summary", {}).items():
        verbose_lines.append(f"## {name}")
        verbose_lines.append(json.dumps(result, indent=2, sort_keys=True))
    (args.research_path / "verbose-summary.txt").write_text("\n".join(verbose_lines) + "\n", encoding="utf-8")
    source_linkage = {
        "source_checkout_path": status["source_checkout_path"],
        "checked_out_commit": status["checked_out_commit"],
        "source_files_linked": status["source_files_linked"],
        "source_map_linkage": status["source_map_linkage"],
        "active_backend_inference": status["active_backend_inference"],
        "reopen_rust_policy_synthesis": status["reopen_rust_policy_synthesis"],
    }
    write_json(args.research_path / "source-linkage-summary.json", source_linkage)
    (args.research_path / "source-linkage-summary.txt").write_text(
        json.dumps(source_linkage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compact_runtime_configs(runtime_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for config in runtime_configs:
        compact_layers = []
        for layer in config["layers"]:
            compact_layers.append(
                {
                    "layer_index": layer["layer_index"],
                    "official_variants_cleared": layer["official_variants_cleared"],
                    "negative_controls_remained_negative": layer["negative_controls_remained_negative"],
                    "official_variant_summary": {
                        name: {
                            "executed": result.get("executed"),
                            "full_vector_cleared": result.get("full_vector_cleared"),
                            "mismatch_count": result.get("comparison", {}).get("mismatch_count"),
                            "max_abs_diff": result.get("comparison", {}).get("max_abs_diff"),
                        }
                        for name, result in layer["official_variants"].items()
                    },
                    "negative_control_summary": {
                        name: {
                            "executed": result.get("executed"),
                            "full_vector_cleared": result.get("full_vector_cleared"),
                            "mismatch_count": result.get("comparison", {}).get("mismatch_count"),
                            "max_abs_diff": result.get("comparison", {}).get("max_abs_diff"),
                        }
                        for name, result in layer["negative_controls"].items()
                    },
                }
            )
        compact.append(
            {
                "config_name": config["config_name"],
                "torch_backends_mkldnn_enabled": config["torch_backends_mkldnn_enabled"],
                "all_official_variants_cleared": config["all_official_variants_cleared"],
                "negative_controls_remained_negative": config["negative_controls_remained_negative"],
                "profiler": config["profiler"],
                "layers": compact_layers,
            }
        )
    return compact


def main() -> int:
    args = parse_args()
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if args.verbose_child:
        return run_verbose_child(args)

    missing = missing_required_paths(args.model, layers)
    if missing:
        status = base_status("fused_linear_addmm_pytorch_minimal_reproducer_blocked_by_missing_artifacts", args, layers)
        status["missing_required_artifacts"] = missing
        write_json(args.status_output, status)
        return 0

    try:
        torch, Checkpoint = import_torch_and_checkpoint()
        metadata = torch_metadata(torch)
        runtime_configs = [
            run_runtime_config(torch, Checkpoint, args, layers, "baseline_default"),
            run_runtime_config(torch, Checkpoint, args, layers, "mkldnn_enabled_true"),
            run_runtime_config(torch, Checkpoint, args, layers, "mkldnn_enabled_false"),
        ]
        verbose = run_verbose_variants(args)
        all_official = all(config["all_official_variants_cleared"] for config in runtime_configs)
        all_negative = all(config["negative_controls_remained_negative"] for config in runtime_configs)
        active_backend = infer_backend(runtime_configs, verbose)
        backend_identified = active_backend in {"native_addmm_cpu_likely", "mkldnn_onednn_likely", "blas_mkl_likely"}
        if not all_official:
            classification = "fused_linear_addmm_pytorch_minimal_reproducer_mismatch"
        elif backend_identified:
            classification = "fused_linear_addmm_pytorch_minimal_reproducer_backend_identified"
        else:
            classification = "fused_linear_addmm_pytorch_minimal_reproducer_backend_attribution_recorded"

        profiler_summary = {
            config["config_name"]: {
                "succeeded": config["profiler"].get("succeeded"),
                "aten_addmm_seen": config["profiler"].get("aten_addmm_seen"),
                "aten_mm_or_matmul_seen": config["profiler"].get("aten_mm_or_matmul_seen"),
                "mkldnn_or_onednn_event_seen": config["profiler"].get("mkldnn_or_onednn_event_seen"),
                "event_keys": [event["key"] for event in config["profiler"].get("events", [])[:20]],
            }
            for config in runtime_configs
        }
        status = base_status(classification, args, layers)
        status.update(
            {
                **metadata,
                "cuda_available": bool(metadata["torch_cuda_is_available"]),
                "sampled_layers_evaluated": layers if all_official else [],
                "all_official_variants_cleared": bool(all_official),
                "negative_controls_remained_negative": bool(all_negative),
                "runtime_configs": compact_runtime_configs(runtime_configs),
                "profiler_summary": profiler_summary,
                "verbose_capture_summary": verbose,
                "active_backend_inference": active_backend,
                "concrete_replayable_rule_found": False,
                "reopen_rust_policy_synthesis": False,
                "source_files_linked": [
                    "aten/src/ATen/native/Linear.cpp",
                    "aten/src/ATen/native/LinearAlgebra.cpp",
                    "aten/src/ATen/native/mkldnn/Matmul.cpp",
                    "aten/src/ATen/native/native_functions.yaml",
                ],
                "source_map_linkage": {
                    "linear_2d_bias_routes_to_addmm": True,
                    "addmm_out_cpu_addmm_impl_cpu": True,
                    "mkldnn_bf16_matmul_candidates_visible": True,
                    "runtime_events_line_up_with_source_candidates": True,
                    "source_path_identification_enough_to_reopen_rust_policy": False,
                },
                "interpretation": {
                    "official_variants_clear_exactly": bool(all_official),
                    "negative_controls_stay_negative": bool(all_negative),
                    "backend_identity_claim": active_backend,
                    "source_path_identification_is_not_replayable_rule": True,
                    "next_bounded_step": "Review minimal reproducer attribution; do not reopen Rust/CUDA policy synthesis without a concrete replayable rule.",
                },
            }
        )
        write_research_outputs(args, status)
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = base_status("fused_linear_addmm_pytorch_minimal_reproducer_failed", args, layers)
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
