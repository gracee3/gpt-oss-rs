#!/usr/bin/env python3
"""CPU-only addmm boundary localization for attention o-proj.

This oracle evidence probe localizes why fused ``torch.addmm`` matches the
official producer/API o-proj reference while explicit matmul/einsum and
unfused-bias forms do not.
"""

from __future__ import annotations

import argparse
import json
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


DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_addmm_boundary_localization_status.json")
DEFAULT_OUTPUT_DIR = Path("/tmp/fused_linear_addmm_addmm_boundary_localization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only addmm boundary localization for fused-linear/addmm o-proj."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="6,10,13,16,18,21")
    return parser.parse_args()


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle": f"/tmp/layer{layer}_ordered_attention_bundle_status.json",
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
        "api_probe_status": f"/tmp/layer{layer}_attention_oproj_api_probe_status.json",
        "cpu_producer_attribution_status": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
    }


def compare_between(torch: Any, actual: Any, expected: Any, focus_lane: int) -> dict[str, Any]:
    return compare_tensors(torch, actual, expected, focus_lane)


def unavailable(torch: Any, name: str, official: Any, focus_lane: int, reason: str) -> dict[str, Any]:
    return make_result(torch, name, False, False, None, official, focus_lane, reason)


def run_variant(thunk: Any) -> tuple[bool, Any | str]:
    try:
        return True, thunk()
    except Exception as exc:  # noqa: BLE001 - probe status should preserve failures
        return False, repr(exc)


def result_from_variant(
    torch: Any,
    name: str,
    thunk: Any,
    official: Any,
    focus_lane: int,
    diagnostic_only: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    ok, output_or_reason = run_variant(thunk)
    if not ok:
        return unavailable(torch, name, official, focus_lane, str(output_or_reason)), None
    output = output_or_reason
    return make_result(torch, name, True, True, output, official, focus_lane, diagnostic_only=diagnostic_only), output


def run_layer(torch: Any, checkpoint: Any, layer: int) -> dict[str, Any]:
    role = SAMPLED_LAYERS[layer]["role"]
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
    zero_bias = torch.zeros_like(bias)
    input_2d = weighted_v.unsqueeze(0)
    weight_t_2d = weight.t()

    tensors = {
        "weighted_v": weighted_v,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "official_output": official,
        "zero_bias": zero_bias,
        "input_2d": input_2d,
        "weight_t_2d": weight_t_2d,
    }
    for name, tensor in tensors.items():
        assert_cpu_tensor(f"layer{layer}.{name}", tensor)

    variant_specs = [
        ("torch_nn_functional_linear", lambda: torch.nn.functional.linear(weighted_v, weight, bias)),
        ("torch_C_nn_linear", lambda: torch._C._nn.linear(weighted_v, weight, bias)),
        ("torch_addmm_bias", lambda: torch.addmm(bias, input_2d, weight_t_2d).squeeze(0)),
        ("torch_addmm_bias_clone", lambda: torch.addmm(bias.clone(), input_2d, weight_t_2d).squeeze(0)),
        ("torch_addmm_zero_bias", lambda: torch.addmm(zero_bias, input_2d, weight_t_2d).squeeze(0)),
        (
            "torch_addmm_zero_bias_plus_bias",
            lambda: (torch.addmm(zero_bias, input_2d, weight_t_2d).squeeze(0) + bias),
        ),
        (
            "torch_addmm_zero_bias_beta0_alpha1",
            lambda: torch.addmm(zero_bias, input_2d, weight_t_2d, beta=0, alpha=1).squeeze(0),
        ),
        (
            "torch_addmm_bias_beta1_alpha1",
            lambda: torch.addmm(bias, input_2d, weight_t_2d, beta=1, alpha=1).squeeze(0),
        ),
        ("input_at_weight_t", lambda: weighted_v @ weight.t()),
        ("torch_matmul_input_weight_t", lambda: torch.matmul(weighted_v, weight.t())),
        ("torch_einsum_core", lambda: torch.einsum("k,hk->h", weighted_v, weight)),
        ("explicit_matmul_plus_bias", lambda: weighted_v @ weight.t() + bias),
        ("explicit_einsum_plus_bias", lambda: torch.einsum("k,hk->h", weighted_v, weight) + bias),
        (
            "explicit_unfused_bias_bf16_output",
            lambda: (torch.nn.functional.linear(weighted_v, weight, None) + bias).to(torch.bfloat16),
        ),
        (
            "zero_addmm_core_float_plus_bias_float_cast_bf16",
            lambda: (
                torch.addmm(zero_bias, input_2d, weight_t_2d).squeeze(0).float() + bias.float()
            ).to(torch.bfloat16),
        ),
        (
            "addmm_f32_bias_only",
            lambda: torch.addmm(bias.float(), input_2d, weight_t_2d).squeeze(0),
        ),
    ]

    variants: dict[str, dict[str, Any]] = {}
    outputs: dict[str, Any] = {}
    for name, thunk in variant_specs:
        result, output = result_from_variant(torch, name, thunk, official, focus_lane)
        variants[name] = result
        if output is not None:
            outputs[name] = output

    layout_specs = [
        ("original_layout", weighted_v, weight, bias),
        ("input_contiguous_clone", weighted_v.contiguous().clone(), weight, bias),
        ("weight_contiguous_clone", weighted_v, weight.contiguous().clone(), bias),
        ("bias_clone", weighted_v, weight, bias.clone()),
        ("all_contiguous_clone", weighted_v.contiguous().clone(), weight.contiguous().clone(), bias.clone()),
    ]
    try:
        noncontig_weight = weight.t().contiguous().t()
        if list(noncontig_weight.shape) == list(weight.shape) and not noncontig_weight.is_contiguous():
            layout_specs.append(("weight_noncontiguous_same_shape", weighted_v, noncontig_weight, bias))
    except Exception:
        pass

    layout_results: dict[str, dict[str, Any]] = {}
    for name, input_tensor, weight_tensor, bias_tensor in layout_specs:
        result, _ = result_from_variant(
            torch,
            f"layout_{name}_addmm_bias",
            lambda i=input_tensor, w=weight_tensor, b=bias_tensor: torch.addmm(
                b, i.unsqueeze(0), w.t()
            ).squeeze(0),
            official,
            focus_lane,
        )
        result["input_metadata"] = {
            "weighted_v": tensor_metadata(torch, input_tensor, include_summary=False),
            "weight": tensor_metadata(torch, weight_tensor, include_summary=False),
            "bias": tensor_metadata(torch, bias_tensor, include_summary=False),
        }
        layout_results[name] = result

    cross_comparisons: dict[str, dict[str, Any] | None] = {}
    if "torch_addmm_zero_bias" in outputs and "input_at_weight_t" in outputs:
        cross_comparisons["addmm_zero_bias_vs_input_at_weight_t"] = compare_between(
            torch, outputs["torch_addmm_zero_bias"], outputs["input_at_weight_t"], focus_lane
        )
    if "torch_addmm_zero_bias" in outputs and "torch_einsum_core" in outputs:
        cross_comparisons["addmm_zero_bias_vs_einsum_core"] = compare_between(
            torch, outputs["torch_addmm_zero_bias"], outputs["torch_einsum_core"], focus_lane
        )
    if "torch_addmm_bias" in outputs and "torch_addmm_zero_bias_plus_bias" in outputs:
        cross_comparisons["addmm_bias_vs_zero_bias_plus_bias"] = compare_between(
            torch, outputs["torch_addmm_bias"], outputs["torch_addmm_zero_bias_plus_bias"], focus_lane
        )
    if "input_at_weight_t" in outputs and "torch_einsum_core" in outputs:
        cross_comparisons["input_at_weight_t_vs_einsum_core"] = compare_between(
            torch, outputs["input_at_weight_t"], outputs["torch_einsum_core"], focus_lane
        )

    def clears(name: str) -> bool:
        return variants.get(name, {}).get("full_vector_cleared") is True

    def cross_clears(name: str) -> bool:
        comparison = cross_comparisons.get(name)
        return bool(comparison and comparison.get("full_vector_cleared") is True)

    localization = {
        "addmm_with_bias_clears": clears("torch_addmm_bias"),
        "addmm_zero_bias_plus_bias_clears": clears("torch_addmm_zero_bias_plus_bias"),
        "addmm_zero_bias_matches_matmul": cross_clears("addmm_zero_bias_vs_input_at_weight_t"),
        "addmm_zero_bias_matches_einsum": cross_clears("addmm_zero_bias_vs_einsum_core"),
        "addmm_with_bias_matches_zero_bias_plus_bias": cross_clears("addmm_bias_vs_zero_bias_plus_bias"),
        "explicit_matmul_plus_bias_clears": clears("explicit_matmul_plus_bias"),
        "explicit_einsum_plus_bias_clears": clears("explicit_einsum_plus_bias"),
        "fused_bias_signal": clears("torch_addmm_bias") and not clears("torch_addmm_zero_bias_plus_bias"),
        "matmul_core_signal": not cross_clears("addmm_zero_bias_vs_input_at_weight_t")
        or not cross_clears("addmm_zero_bias_vs_einsum_core"),
        "output_cast_signal": clears("zero_addmm_core_float_plus_bias_float_cast_bf16"),
        "layout_signal": any(
            result.get("full_vector_cleared") is False for result in layout_results.values()
        ),
    }

    source_probe = None
    source_probe_path = Path(paths["api_probe_status"])
    if source_probe_path.is_file():
        source_probe = load_json(source_probe_path)

    return {
        "layer_index": layer,
        "role": role,
        "focus_lane": focus_lane,
        "source_paths": paths,
        "source_status_classification": source_probe.get("classification") if isinstance(source_probe, dict) else None,
        "tensor_metadata": {
            "weighted_v": tensor_metadata(torch, weighted_v),
            "o_proj_weight": tensor_metadata(torch, weight),
            "o_proj_bias": tensor_metadata(torch, bias),
            "official_output": tensor_metadata(torch, official),
            "input_2d": tensor_metadata(torch, input_2d, include_summary=False),
            "weight_t_2d": tensor_metadata(torch, weight_t_2d, include_summary=False),
            "zero_bias": tensor_metadata(torch, zero_bias),
            "provenance": {
                "weighted_v_boundary": weighted_v_json.get("boundary"),
                "official_o_proj_boundary": official_json.get("boundary"),
            },
        },
        "variant_results": variants,
        "cross_comparisons": cross_comparisons,
        "layout_perturbation_results": layout_results,
        "localization": localization,
    }


def classify(layer_results: list[dict[str, Any]]) -> str:
    fused_bias = any(result["localization"]["fused_bias_signal"] for result in layer_results)
    matmul_core = any(result["localization"]["matmul_core_signal"] for result in layer_results)
    output_cast = any(result["localization"]["output_cast_signal"] for result in layer_results)
    layout = any(result["localization"]["layout_signal"] for result in layer_results)
    if fused_bias and not matmul_core and not output_cast:
        return "fused_linear_addmm_addmm_boundary_fused_bias_localized"
    if matmul_core and not fused_bias and not output_cast:
        return "fused_linear_addmm_addmm_boundary_matmul_core_localized"
    if output_cast and not fused_bias and not matmul_core:
        return "fused_linear_addmm_addmm_boundary_output_cast_localized"
    if layout and not fused_bias and not matmul_core and not output_cast:
        return "fused_linear_addmm_addmm_boundary_layout_localized"
    if fused_bias or matmul_core or output_cast or layout:
        return "fused_linear_addmm_addmm_boundary_inconclusive"
    return "fused_linear_addmm_addmm_boundary_localization_recorded"


def failure_status(error: str) -> dict[str, Any]:
    return {
        "classification": "fused_linear_addmm_addmm_boundary_failed",
        "validation_only": True,
        "producer_probe": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        **GUARD_FALSE_FLAGS,
        "error": error,
    }


def main() -> int:
    args = parse_args()
    try:
        torch, Checkpoint = import_torch_and_checkpoint()
        layers = [int(part) for part in args.layers.split(",") if part.strip()]
        unknown = [layer for layer in layers if layer not in SAMPLED_LAYERS]
        if unknown:
            raise ValueError(f"unsupported sampled layers: {unknown}")
        checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
        args.output_dir.mkdir(parents=True, exist_ok=True)

        layer_results = [run_layer(torch, checkpoint, layer) for layer in layers]
        classification = classify(layer_results)
        status = {
            "classification": classification,
            "validation_only": True,
            "producer_probe": True,
            "oracle_device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_used": False,
            **GUARD_FALSE_FLAGS,
            "model": str(args.model),
            "operator": "attention_o_proj",
            "dtype": "torch.bfloat16",
            "source_cpu_producer_attribution_status": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
            "sampled_layers": layers,
            "environment": environment_metadata(torch),
            "layers": layer_results,
            "summary": {
                "layers_with_fused_bias_signal": [
                    result["layer_index"] for result in layer_results if result["localization"]["fused_bias_signal"]
                ],
                "layers_with_matmul_core_signal": [
                    result["layer_index"] for result in layer_results if result["localization"]["matmul_core_signal"]
                ],
                "layers_with_output_cast_signal": [
                    result["layer_index"] for result in layer_results if result["localization"]["output_cast_signal"]
                ],
                "layers_with_layout_signal": [
                    result["layer_index"] for result in layer_results if result["localization"]["layout_signal"]
                ],
                "focus_lane_only_promoted": False,
                "backend_selected": False,
            },
            "interpretation": {
                "primary_localization": classification,
                "explains_negative_controls": True,
                "backend_identity_claim": "none",
                "next_bounded_step": "Review addmm boundary localization before any status-only discriminator or candidate execution.",
            },
        }
        write_json(args.status_output, status)
        write_json(args.output_dir / "addmm_boundary_localization_status.json", status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = failure_status(repr(exc))
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
