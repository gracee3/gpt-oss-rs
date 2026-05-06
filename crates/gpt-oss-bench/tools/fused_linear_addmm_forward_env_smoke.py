#!/usr/bin/env python3
"""Forward-env smoke for the fused addmm attention o-proj seam.

This validation-only smoke checks whether the uv-managed forward Python/Torch
environment reproduces a minimal existing Workstream A producer/API artifact
subset with CPU ``torch.addmm(bias, input_2d, weight_t_2d)``.
"""

from __future__ import annotations

import argparse
import platform
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
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_forward_env_smoke_status.json")
REQUIRED_LAYERS = [6, 18]
OPTIONAL_LAYERS = [10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward Python env smoke for fused-linear/addmm attention o-proj."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--forward-env-path", type=Path, default=DEFAULT_FORWARD_ENV)
    parser.add_argument("--include-optional-layer10", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
    }


def base_status(classification: str) -> dict[str, Any]:
    return {
        "classification": classification,
        "validation_only": True,
        "forward_env_smoke": True,
        "oracle_device": "cpu",
        "forward_env_path": str(DEFAULT_FORWARD_ENV),
        "cuda_available": None,
        "cuda_used": False,
        "full_model_loaded": False,
        "gpu_tensors_created": False,
        "historical_envs_modified": False,
        "rebaseline_performed": False,
        "old_artifacts_replaced": False,
        "cross_env_comparison": True,
        "cross_env_comparison_note": (
            "Forward-env smoke compares new-env addmm outputs to existing historical official artifacts; "
            "this does not replace historical artifacts."
        ),
        **GUARD_FALSE_FLAGS,
    }


def missing_required_paths(model: Path, required_layers: list[int]) -> list[str]:
    paths = [model]
    for layer in required_layers:
        layer_source = layer_paths(layer)
        paths.extend(
            [
                Path(layer_source["attention_bundle_dir"]),
                Path(layer_source["weighted_v"]),
                Path(layer_source["official_o_proj"]),
            ]
        )
    return [str(path) for path in paths if not path.exists()]


def import_metadata(torch: Any) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "sys_prefix": sys.prefix,
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "torch_import_path": str(Path(torch.__file__).resolve()) if getattr(torch, "__file__", None) else None,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
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
        return make_result(
            torch,
            name,
            available=True,
            executed=True,
            output=output,
            official=official,
            focus_lane=focus_lane,
            diagnostic_only=diagnostic_only,
        )
    except Exception as exc:  # noqa: BLE001 - preserve smoke failures in status
        return make_result(
            torch,
            name,
            available=False,
            executed=False,
            output=None,
            official=official,
            focus_lane=focus_lane,
            reason=repr(exc),
            diagnostic_only=diagnostic_only,
        )


def run_layer(torch: Any, checkpoint: Any, layer: int, optional: bool) -> dict[str, Any]:
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
    weight_name = f"model.layers.{layer}.self_attn.o_proj.weight"
    bias_name = f"model.layers.{layer}.self_attn.o_proj.bias"
    weight = checkpoint.get(weight_name)
    bias = checkpoint.get(bias_name)
    zero_bias = torch.zeros_like(bias)
    input_2d = weighted_v.unsqueeze(0)
    weight_t_2d = weight.t()

    tensors = {
        "weighted_v": weighted_v,
        "official_o_proj": official,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "zero_bias": zero_bias,
        "input_2d": input_2d,
        "weight_t_2d": weight_t_2d,
    }
    for name, tensor in tensors.items():
        assert_cpu_tensor(f"layer{layer}.{name}", tensor)

    addmm_result = run_variant(
        torch,
        "torch_addmm_fused_bias",
        lambda: torch.addmm(bias, input_2d, weight_t_2d).squeeze(0),
        official,
        focus_lane,
    )
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
            lambda: weighted_v @ weight.t() + bias,
            official,
            focus_lane,
            diagnostic_only=True,
        ),
        "explicit_einsum_plus_bias": run_variant(
            torch,
            "explicit_einsum_plus_bias",
            lambda: torch.einsum("k,hk->h", weighted_v, weight) + bias,
            official,
            focus_lane,
            diagnostic_only=True,
        ),
    }
    negative_summary = {
        name: {
            "executed": result.get("executed"),
            "full_vector_cleared": result.get("full_vector_cleared"),
            "mismatch_count": result.get("comparison", {}).get("mismatch_count"),
            "max_abs_diff": result.get("comparison", {}).get("max_abs_diff"),
        }
        for name, result in negative_controls.items()
    }
    return {
        "layer_index": layer,
        "role": SAMPLED_LAYERS[layer]["role"],
        "optional": optional,
        "operator": "attention_o_proj",
        "api": "torch.addmm(bias, input_2d, weight_t_2d)",
        "focus_lane": focus_lane,
        "source_artifacts": paths,
        "model_tensors_loaded": [weight_name, bias_name],
        "tensor_metadata": {
            "weighted_v": tensor_metadata(torch, weighted_v),
            "o_proj_weight": tensor_metadata(torch, weight),
            "o_proj_bias": tensor_metadata(torch, bias),
            "official_o_proj": tensor_metadata(torch, official),
            "input_2d": tensor_metadata(torch, input_2d, include_summary=False),
            "weight_t_2d": tensor_metadata(torch, weight_t_2d, include_summary=False),
            "provenance": {
                "weighted_v_boundary": weighted_v_json.get("boundary"),
                "official_o_proj_boundary": official_json.get("boundary"),
            },
        },
        "addmm_result": addmm_result,
        "comparison_vs_official": addmm_result.get("comparison"),
        "full_vector_cleared": bool(addmm_result.get("full_vector_cleared")),
        "negative_controls": negative_controls,
        "negative_control_summary": negative_summary,
    }


def main() -> int:
    args = parse_args()
    required_layers = REQUIRED_LAYERS
    optional_layers = OPTIONAL_LAYERS if args.include_optional_layer10 else []
    missing = missing_required_paths(args.model, required_layers)
    if missing:
        status = base_status("fused_linear_addmm_forward_env_smoke_blocked_by_missing_artifacts")
        status.update(
            {
                "forward_env_path": str(args.forward_env_path),
                "sampled_layers_requested": required_layers,
                "optional_layers_requested": optional_layers,
                "sampled_layers_evaluated": [],
                "optional_layers_evaluated": [],
                "missing_required_artifacts": missing,
                "model": str(args.model),
            }
        )
        write_json(args.status_output, status)
        return 0

    try:
        torch, Checkpoint = import_torch_and_checkpoint()
        import_meta = import_metadata(torch)
        checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
        layer_results = []
        optional_skipped = []
        for layer in required_layers:
            layer_results.append(run_layer(torch, checkpoint, layer, optional=False))
        for layer in optional_layers:
            paths = layer_paths(layer)
            if Path(paths["weighted_v"]).is_file() and Path(paths["official_o_proj"]).is_file():
                layer_results.append(run_layer(torch, checkpoint, layer, optional=True))
            else:
                optional_skipped.append({"layer_index": layer, "reason": "missing optional artifacts", "paths": paths})

        required_results = [result for result in layer_results if result["layer_index"] in required_layers]
        required_matched = all(result.get("full_vector_cleared") is True for result in required_results)
        classification = (
            "fused_linear_addmm_forward_env_smoke_matched"
            if required_matched
            else "fused_linear_addmm_forward_env_smoke_mismatch"
        )
        source_artifacts = {
            f"layer{result['layer_index']}": result["source_artifacts"] for result in layer_results
        }
        model_tensors_loaded = {
            f"layer{result['layer_index']}": result["model_tensors_loaded"] for result in layer_results
        }
        status = base_status(classification)
        status.update(
            {
                "forward_env_path": str(args.forward_env_path),
                **import_meta,
                "cuda_available": bool(import_meta["torch_cuda_is_available"]),
                "model": str(args.model),
                "operator": "attention_o_proj",
                "api": "torch.addmm(bias, input_2d, weight_t_2d)",
                "dtype_contract": {
                    "input_dtype": "torch.bfloat16",
                    "weight_dtype": "torch.bfloat16",
                    "bias_dtype": "torch.bfloat16",
                    "output_dtype": "torch.bfloat16",
                    "device": "cpu",
                    "full_vector_exactness_required": True,
                },
                "sampled_layers_requested": required_layers,
                "sampled_layers_evaluated": [result["layer_index"] for result in required_results],
                "optional_layers_requested": optional_layers,
                "optional_layers_evaluated": [
                    result["layer_index"] for result in layer_results if result.get("optional")
                ],
                "optional_layers_skipped": optional_skipped,
                "source_artifacts": source_artifacts,
                "model_tensors_loaded": model_tensors_loaded,
                "environment": environment_metadata(torch),
                "layers": layer_results,
                "summary": {
                    "required_layers_matched": bool(required_matched),
                    "forward_env_matched_required_historical_official_artifacts": bool(required_matched),
                    "rebaseline_performed": False,
                    "negative_controls_diagnostic_only": True,
                    "optional_layer10_does_not_drive_classification": True,
                },
            }
        )
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = base_status("fused_linear_addmm_forward_env_smoke_failed")
        status.update(
            {
                "forward_env_path": str(args.forward_env_path),
                "sampled_layers_requested": required_layers,
                "optional_layers_requested": optional_layers,
                "sampled_layers_evaluated": [],
                "optional_layers_evaluated": [],
                "model": str(args.model),
                "error": repr(exc),
            }
        )
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
