#!/usr/bin/env python3
"""CPU-only fused-bias arithmetic contract probe for attention o-proj.

This oracle evidence probe tests whether the clearing BF16
``torch.addmm(bias, input, weight.T)`` result can be reconstructed by explicit
models of where bias enters the arithmetic and where BF16 rounding is observed.
"""

from __future__ import annotations

import argparse
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


DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_fused_bias_arithmetic_contract_status.json")
DEFAULT_OUTPUT_DIR = Path("/tmp/fused_linear_addmm_fused_bias_arithmetic_contract")
BOUNDARY_STATUS = Path("/tmp/fused_linear_addmm_addmm_boundary_localization_status.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only fused-bias arithmetic contract probe for attention o-proj."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--boundary-status", type=Path, default=BOUNDARY_STATUS)
    parser.add_argument("--layers", default="6,10,13,16,18,21")
    return parser.parse_args()


def layer_paths(layer: int) -> dict[str, str]:
    return {
        "attention_bundle": f"/tmp/layer{layer}_ordered_attention_bundle_status.json",
        "attention_bundle_dir": f"/tmp/layer{layer}_ordered_attention_bundle",
        "weighted_v": f"/tmp/layer{layer}_ordered_attention_bundle/weighted_v.json",
        "official_o_proj": f"/tmp/layer{layer}_ordered_attention_bundle/o_proj.json",
        "boundary_localization_status": str(BOUNDARY_STATUS),
        "cpu_producer_attribution_status": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
    }


def run_variant(thunk: Any) -> tuple[bool, Any | str]:
    try:
        return True, thunk()
    except Exception as exc:  # noqa: BLE001 - probe status should preserve failures
        return False, repr(exc)


def unavailable(torch: Any, name: str, official: Any, focus_lane: int, reason: str) -> dict[str, Any]:
    return make_result(torch, name, False, False, None, official, focus_lane, reason)


def result_from_variant(
    torch: Any,
    name: str,
    thunk: Any,
    official: Any,
    addmm_bias: Any,
    focus_lane: int,
    diagnostic_only: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    ok, output_or_reason = run_variant(thunk)
    if not ok:
        return unavailable(torch, name, official, focus_lane, str(output_or_reason)), None
    output = output_or_reason
    result = make_result(
        torch,
        name,
        True,
        True,
        output,
        official,
        focus_lane,
        diagnostic_only=diagnostic_only,
    )
    result["comparison_vs_addmm_bias"] = compare_tensors(torch, output, addmm_bias, focus_lane)
    return result, output


def bf16_float(torch: Any, value: Any, dtype: Any | None = None) -> float:
    source_dtype = dtype if dtype is not None else torch.float32
    return float(torch.tensor([float(value)], dtype=source_dtype, device="cpu").to(torch.bfloat16)[0].float().item())


def tensor_scalar_float(value: Any) -> float:
    return float(value.detach().to("cpu").float().item())


def sequential_sum(torch: Any, terms: Any, reverse: bool = False) -> Any:
    work = terms.flip(0) if reverse else terms
    acc = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    for term in work:
        acc = acc + term.to(torch.float32)
    return acc


def sequential_with_initial(torch: Any, terms: Any, initial: Any, reverse: bool = False) -> Any:
    work = terms.flip(0) if reverse else terms
    acc = initial.to(torch.float32).clone()
    for term in work:
        acc = acc + term.to(torch.float32)
    return acc


def pairwise_sum_1d(torch: Any, terms: Any) -> Any:
    work = terms.to(torch.float32)
    while int(work.numel()) > 1:
        if int(work.numel()) % 2:
            work = torch.cat([work, torch.zeros(1, dtype=work.dtype, device="cpu")])
        work = work.reshape(-1, 2).sum(dim=1)
    return work[0]


def pairwise_sum_2d(torch: Any, terms: Any) -> Any:
    work = terms.to(torch.float32)
    while int(work.shape[1]) > 1:
        if int(work.shape[1]) % 2:
            pad = torch.zeros((work.shape[0], 1), dtype=work.dtype, device="cpu")
            work = torch.cat([work, pad], dim=1)
        work = work.reshape(work.shape[0], -1, 2).sum(dim=2)
    return work[:, 0]


def pairwise_with_bias_term(torch: Any, terms: Any, bias_value: Any) -> Any:
    work = torch.cat([terms.to(torch.float32), bias_value.reshape(1).to(torch.float32)])
    return pairwise_sum_1d(torch, work)


def abs_ascending_sum(torch: Any, terms: Any) -> Any:
    order = torch.argsort(terms.float().abs(), stable=True)
    return sequential_sum(torch, terms[order])


def lane_variant_values(torch: Any, weighted_v: Any, weight_row: Any, bias_value: Any) -> dict[str, float]:
    terms_f32 = weighted_v.float() * weight_row.float()
    terms_f64 = weighted_v.double() * weight_row.double()
    bias_f32 = bias_value.float()
    bias_f64 = bias_value.double()

    f32_forward = sequential_sum(torch, terms_f32)
    f32_reverse = sequential_sum(torch, terms_f32, reverse=True)
    f32_pairwise = pairwise_sum_1d(torch, terms_f32)
    f32_abs_ascending = abs_ascending_sum(torch, terms_f32)
    f64_sum = terms_f64.sum()
    bf16_product_terms = (weighted_v * weight_row).to(torch.bfloat16).float()
    bf16_product_sum = sequential_sum(torch, bf16_product_terms)

    core_bf16 = torch.tensor([tensor_scalar_float(f32_forward)], dtype=torch.float32, device="cpu").to(torch.bfloat16)[0]
    core_then_bias = (core_bf16 + bias_value).to(torch.bfloat16)

    return {
        "sum_f32_terms_then_add_bias_f32_then_bf16": bf16_float(torch, f32_forward + bias_f32),
        "sum_f64_terms_then_add_bias_f64_then_bf16": bf16_float(torch, f64_sum + bias_f64, torch.float64),
        "pairwise_f32_sum_then_add_bias_f32_then_bf16": bf16_float(torch, f32_pairwise + bias_f32),
        "reverse_f32_sum_then_add_bias_f32_then_bf16": bf16_float(torch, f32_reverse + bias_f32),
        "deterministic_abs_ascending_f32_sum_then_add_bias_f32_then_bf16": bf16_float(
            torch, f32_abs_ascending + bias_f32
        ),
        "acc_f32_starts_with_bias_f32_then_forward_terms_then_bf16": bf16_float(
            torch, sequential_with_initial(torch, terms_f32, bias_f32)
        ),
        "acc_f32_starts_with_bias_f32_then_reverse_terms_then_bf16": bf16_float(
            torch, sequential_with_initial(torch, terms_f32, bias_f32, reverse=True)
        ),
        "acc_f64_starts_with_bias_f64_then_forward_terms_then_bf16": bf16_float(
            torch, (bias_f64 + terms_f64.sum()), torch.float64
        ),
        "pairwise_tree_with_bias_term_then_bf16": bf16_float(torch, pairwise_with_bias_term(torch, terms_f32, bias_f32)),
        "bf16_product_rounded_before_f32_sum_then_bias_then_bf16": bf16_float(
            torch, bf16_product_sum + bias_f32
        ),
        "f64_products_sum_diagnostic_then_bias_then_bf16": bf16_float(torch, f64_sum + bias_f64, torch.float64),
        "final_bf16_cast_only_once": bf16_float(torch, f32_forward + bias_f32),
        "intermediate_bf16_core_then_bias_then_bf16": tensor_scalar_float(core_then_bias),
        "bias_bf16_converted_to_f32_before_accumulation": bf16_float(torch, f32_forward + bias_value.float()),
        "bias_kept_as_bf16_term_then_bf16": bf16_float(torch, f32_forward + bias_value),
    }


def lane_result(
    torch: Any,
    lane: int,
    weighted_v: Any,
    weight: Any,
    bias: Any,
    addmm_bias: Any,
    official: Any,
    zero_bias_plus_bias: Any,
    matmul_plus_bias: Any,
    einsum_plus_bias: Any,
    unfused_bf16_bias: Any,
) -> dict[str, Any]:
    addmm_value = float(addmm_bias[lane].float().item())
    official_value = float(official[lane].float().item())
    baselines = {
        "torch_zero_bias_addmm_plus_bias": float(zero_bias_plus_bias[lane].float().item()),
        "explicit_matmul_output_plus_bias": float(matmul_plus_bias[lane].float().item()),
        "explicit_einsum_output_plus_bias": float(einsum_plus_bias[lane].float().item()),
        "explicit_unfused_bf16_bias": float(unfused_bf16_bias[lane].float().item()),
    }
    values = lane_variant_values(torch, weighted_v, weight[lane], bias[lane])
    variants: dict[str, dict[str, Any]] = {}
    for name, value in {**baselines, **values}.items():
        variants[name] = {
            "value": value,
            "matches_addmm_bias": value == addmm_value,
            "abs_diff_vs_addmm_bias": abs(value - addmm_value),
            "matches_official": value == official_value,
            "abs_diff_vs_official": abs(value - official_value),
        }
    return {
        "lane": lane,
        "addmm_bias": addmm_value,
        "official": official_value,
        "addmm_matches_official": addmm_value == official_value,
        "variants": variants,
        "matching_arithmetic_variants": [
            name
            for name, result in variants.items()
            if result["matches_addmm_bias"] and not name.startswith(("torch_", "explicit_"))
        ],
    }


def select_lanes(layer: int, boundary_layer: dict[str, Any] | None) -> dict[str, Any]:
    focus = int(SAMPLED_LAYERS[layer]["focus_lane"])
    lanes = [focus]
    provenance: dict[str, Any] = {"focus_lane": focus}
    comparison = None
    if boundary_layer:
        comparison = boundary_layer.get("cross_comparisons", {}).get("addmm_bias_vs_zero_bias_plus_bias")
    if isinstance(comparison, dict):
        first = comparison.get("first_mismatch")
        worst = comparison.get("worst_mismatch")
        samples = comparison.get("mismatch_samples") or []
        if isinstance(first, dict) and first.get("index") is not None:
            lanes.append(int(first["index"]))
            provenance["first_mismatch_lane"] = int(first["index"])
        if isinstance(worst, dict) and worst.get("index") is not None:
            lanes.append(int(worst["index"]))
            provenance["worst_mismatch_lane"] = int(worst["index"])
        sample_lanes = []
        for sample in samples[:8]:
            if isinstance(sample, dict) and sample.get("index") is not None:
                lanes.append(int(sample["index"]))
                sample_lanes.append(int(sample["index"]))
        provenance["representative_mismatch_lanes"] = sample_lanes
    unique_lanes = []
    for lane in lanes:
        if lane not in unique_lanes:
            unique_lanes.append(lane)
    return {"lanes": unique_lanes, "provenance": provenance}


def full_vector_candidates(torch: Any, weighted_v: Any, weight: Any, bias: Any) -> dict[str, Any]:
    input_f32 = weighted_v.float()
    weight_f32 = weight.float()
    bias_f32 = bias.float()
    terms = weight_f32 * input_f32.unsqueeze(0)
    f64_core = weight.double() @ weighted_v.double()
    f32_core = terms.sum(dim=1)
    reverse_core = terms.flip(1).sum(dim=1)
    pairwise_core = pairwise_sum_2d(torch, terms)
    bf16_product_core = (weight * weighted_v.unsqueeze(0)).to(torch.bfloat16).float().sum(dim=1)
    core_bf16 = f32_core.to(torch.bfloat16)

    return {
        "sum_f32_terms_then_add_bias_f32_then_bf16": (f32_core + bias_f32).to(torch.bfloat16),
        "sum_f64_terms_then_add_bias_f64_then_bf16": (f64_core + bias.double()).to(torch.bfloat16),
        "pairwise_f32_sum_then_add_bias_f32_then_bf16": (pairwise_core + bias_f32).to(torch.bfloat16),
        "reverse_f32_sum_then_add_bias_f32_then_bf16": (reverse_core + bias_f32).to(torch.bfloat16),
        "acc_f32_starts_with_bias_f32_then_forward_terms_then_bf16": (bias_f32 + f32_core).to(torch.bfloat16),
        "bf16_product_rounded_before_f32_sum_then_bias_then_bf16": (bf16_product_core + bias_f32).to(torch.bfloat16),
        "intermediate_bf16_core_then_bias_then_bf16": (core_bf16 + bias).to(torch.bfloat16),
        "final_bf16_cast_only_once": (f32_core + bias_f32).to(torch.bfloat16),
    }


def find_boundary_layer(boundary_status: dict[str, Any] | None, layer: int) -> dict[str, Any] | None:
    if not isinstance(boundary_status, dict):
        return None
    for item in boundary_status.get("layers", []):
        if isinstance(item, dict) and item.get("layer_index") == layer:
            return item
    return None


def run_layer(torch: Any, checkpoint: Any, layer: int, boundary_status: dict[str, Any] | None) -> dict[str, Any]:
    paths = layer_paths(layer)
    focus_lane = int(SAMPLED_LAYERS[layer]["focus_lane"])
    role = SAMPLED_LAYERS[layer]["role"]
    boundary_layer = find_boundary_layer(boundary_status, layer)

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

    for name, tensor in {
        "weighted_v": weighted_v,
        "o_proj_weight": weight,
        "o_proj_bias": bias,
        "official_output": official,
        "zero_bias": zero_bias,
        "input_2d": input_2d,
        "weight_t_2d": weight_t_2d,
    }.items():
        assert_cpu_tensor(f"layer{layer}.{name}", tensor)

    addmm_bias = torch.addmm(bias, input_2d, weight_t_2d).squeeze(0)
    addmm_zero_bias = torch.addmm(zero_bias, input_2d, weight_t_2d).squeeze(0)
    zero_bias_plus_bias = addmm_zero_bias + bias
    matmul_plus_bias = weighted_v @ weight.t() + bias
    einsum_plus_bias = torch.einsum("k,hk->h", weighted_v, weight) + bias
    unfused_bf16_bias = (torch.nn.functional.linear(weighted_v, weight, None) + bias).to(torch.bfloat16)

    baseline_specs = [
        ("torch_addmm_bias_reference", lambda: addmm_bias),
        ("torch_addmm_zero_bias_core", lambda: addmm_zero_bias),
        ("torch_addmm_zero_bias_plus_bias", lambda: zero_bias_plus_bias),
        ("explicit_matmul_plus_bias", lambda: matmul_plus_bias),
        ("explicit_einsum_plus_bias", lambda: einsum_plus_bias),
        ("explicit_unfused_bf16_bias", lambda: unfused_bf16_bias),
    ]
    baseline_results: dict[str, dict[str, Any]] = {}
    baseline_outputs: dict[str, Any] = {}
    for name, thunk in baseline_specs:
        result, output = result_from_variant(torch, name, thunk, official, addmm_bias, focus_lane)
        baseline_results[name] = result
        if output is not None:
            baseline_outputs[name] = output

    selected = select_lanes(layer, boundary_layer)
    lane_results = [
        lane_result(
            torch,
            lane,
            weighted_v,
            weight,
            bias,
            addmm_bias,
            official,
            zero_bias_plus_bias,
            matmul_plus_bias,
            einsum_plus_bias,
            unfused_bf16_bias,
        )
        for lane in selected["lanes"]
    ]

    arithmetic_names = sorted(
        {
            name
            for lane in lane_results
            for name, value in lane["variants"].items()
            if not name.startswith(("torch_", "explicit_")) and value["matches_addmm_bias"]
        }
    )
    lane_clear_counts = {}
    for name in sorted(
        {
            name
            for lane in lane_results
            for name in lane["variants"]
            if not name.startswith(("torch_", "explicit_"))
        }
    ):
        lane_clear_counts[name] = sum(1 for lane in lane_results if lane["variants"][name]["matches_addmm_bias"])

    full_outputs = full_vector_candidates(torch, weighted_v, weight, bias)
    full_results = {}
    for name, output in full_outputs.items():
        result = make_result(torch, name, True, True, output, official, focus_lane)
        result["comparison_vs_addmm_bias"] = compare_tensors(torch, output, addmm_bias, focus_lane)
        result["selected_from_lane_candidates"] = name in arithmetic_names
        full_results[name] = result

    full_vector_clearing = [
        name
        for name, result in full_results.items()
        if result["comparison_vs_addmm_bias"]["full_vector_cleared"]
    ]
    lane_all_clear = [
        name for name, count in lane_clear_counts.items() if count == len(lane_results)
    ]

    localization = {
        "addmm_with_bias_clears": baseline_results["torch_addmm_bias_reference"]["full_vector_cleared"],
        "zero_bias_plus_bias_clears": baseline_results["torch_addmm_zero_bias_plus_bias"]["full_vector_cleared"],
        "explicit_matmul_plus_bias_clears": baseline_results["explicit_matmul_plus_bias"]["full_vector_cleared"],
        "explicit_einsum_plus_bias_clears": baseline_results["explicit_einsum_plus_bias"]["full_vector_cleared"],
        "explicit_unfused_bf16_bias_clears": baseline_results["explicit_unfused_bf16_bias"]["full_vector_cleared"],
        "lane_level_arithmetic_all_clear": lane_all_clear,
        "full_vector_arithmetic_clears": full_vector_clearing,
        "pre_round_bias_lane_support": any("add_bias" in name for name in lane_all_clear),
        "pre_round_bias_full_vector_support": any("add_bias" in name for name in full_vector_clearing),
        "product_policy_full_vector_support": any("bf16_product" in name for name in full_vector_clearing),
        "accumulator_policy_full_vector_support": any(
            name in full_vector_clearing
            for name in [
                "pairwise_f32_sum_then_add_bias_f32_then_bf16",
                "reverse_f32_sum_then_add_bias_f32_then_bf16",
                "acc_f32_starts_with_bias_f32_then_forward_terms_then_bf16",
            ]
        ),
    }

    return {
        "layer_index": layer,
        "role": role,
        "operator": "attention_o_proj",
        "focus_lane": focus_lane,
        "selected_lanes": selected,
        "source_paths": paths,
        "tensor_metadata": {
            "weighted_v": tensor_metadata(torch, weighted_v),
            "o_proj_weight": tensor_metadata(torch, weight),
            "o_proj_bias": tensor_metadata(torch, bias),
            "official_output": tensor_metadata(torch, official),
            "addmm_bias_output": tensor_metadata(torch, addmm_bias),
            "addmm_zero_bias_output": tensor_metadata(torch, addmm_zero_bias),
            "provenance": {
                "weighted_v_boundary": weighted_v_json.get("boundary"),
                "official_o_proj_boundary": official_json.get("boundary"),
            },
        },
        "baseline_results": baseline_results,
        "lane_results": lane_results,
        "lane_variant_clear_counts": lane_clear_counts,
        "full_vector_replay_results": full_results,
        "localization": localization,
        "interpretation": {
            "bias_before_output_rounding_supported": localization["pre_round_bias_lane_support"]
            or localization["pre_round_bias_full_vector_support"],
            "exact_accumulation_policy_localized": bool(localization["full_vector_arithmetic_clears"]),
            "focus_lane_only_promoted": False,
        },
    }


def classify(layer_results: list[dict[str, Any]]) -> str:
    if not layer_results:
        return "fused_linear_addmm_fused_bias_arithmetic_contract_failed"

    full_clear_sets = [
        set(result["localization"]["full_vector_arithmetic_clears"])
        for result in layer_results
    ]
    common_full_clears = set.intersection(*full_clear_sets) if full_clear_sets else set()
    if common_full_clears:
        if any("bf16_product" in name for name in common_full_clears):
            return "fused_linear_addmm_fused_bias_arithmetic_contract_product_policy_localized"
        if any("pairwise" in name or "reverse" in name or "acc_" in name for name in common_full_clears):
            return "fused_linear_addmm_fused_bias_arithmetic_contract_accumulator_policy_localized"
        if any("add_bias" in name or "final_bf16_cast" in name for name in common_full_clears):
            return "fused_linear_addmm_fused_bias_arithmetic_contract_pre_round_bias_localized"

    lane_support = any(result["localization"]["pre_round_bias_lane_support"] for result in layer_results)
    full_support = any(result["localization"]["pre_round_bias_full_vector_support"] for result in layer_results)
    if lane_support or full_support:
        return "fused_linear_addmm_fused_bias_arithmetic_contract_inconclusive"
    return "fused_linear_addmm_fused_bias_arithmetic_contract_recorded"


def failure_status(error: str) -> dict[str, Any]:
    return {
        "classification": "fused_linear_addmm_fused_bias_arithmetic_contract_failed",
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
    try:
        torch, Checkpoint = import_torch_and_checkpoint()
        layers = [int(part) for part in args.layers.split(",") if part.strip()]
        unknown = [layer for layer in layers if layer not in SAMPLED_LAYERS]
        if unknown:
            raise ValueError(f"unsupported sampled layers: {unknown}")

        boundary_status = load_json(args.boundary_status) if args.boundary_status.is_file() else None
        checkpoint = Checkpoint(str(args.model), torch.device("cpu"))
        args.output_dir.mkdir(parents=True, exist_ok=True)

        layer_results = [run_layer(torch, checkpoint, layer, boundary_status) for layer in layers]
        classification = classify(layer_results)
        common_full_clears = set.intersection(
            *[set(layer["localization"]["full_vector_arithmetic_clears"]) for layer in layer_results]
        )
        layers_with_lane_pre_round_support = [
            layer["layer_index"]
            for layer in layer_results
            if layer["localization"]["pre_round_bias_lane_support"]
        ]
        layers_with_full_pre_round_support = [
            layer["layer_index"]
            for layer in layer_results
            if layer["localization"]["pre_round_bias_full_vector_support"]
        ]
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
            "source_statuses": {
                "addmm_boundary_localization": str(args.boundary_status),
                "cpu_producer_attribution": "/tmp/fused_linear_addmm_cpu_producer_attribution_status.json",
            },
            "sampled_layers": layers,
            "environment": environment_metadata(torch),
            "layers": layer_results,
            "summary": {
                "common_full_vector_arithmetic_clears": sorted(common_full_clears),
                "layers_with_lane_level_pre_round_bias_support": layers_with_lane_pre_round_support,
                "layers_with_full_vector_pre_round_bias_support": layers_with_full_pre_round_support,
                "backend_selected": False,
                "focus_lane_only_promoted": False,
            },
            "interpretation": {
                "bias_before_output_rounding_supported": bool(
                    layers_with_lane_pre_round_support or layers_with_full_pre_round_support
                ),
                "exact_accumulation_product_policy_localized": bool(common_full_clears),
                "strongest_signal": "pre_round_bias"
                if (layers_with_lane_pre_round_support or layers_with_full_pre_round_support)
                else "none",
                "negative_controls_remain_negative": all(
                    not layer["localization"]["explicit_matmul_plus_bias_clears"]
                    and not layer["localization"]["explicit_einsum_plus_bias_clears"]
                    and not layer["localization"]["explicit_unfused_bf16_bias_clears"]
                    for layer in layer_results
                ),
                "backend_identity_claim": "none",
                "next_bounded_step": "Review fused-bias arithmetic contract before any status-only discriminator or candidate execution.",
            },
        }
        write_json(args.status_output, status)
        write_json(args.output_dir / "fused_bias_arithmetic_contract_status.json", status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status = failure_status(repr(exc))
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
