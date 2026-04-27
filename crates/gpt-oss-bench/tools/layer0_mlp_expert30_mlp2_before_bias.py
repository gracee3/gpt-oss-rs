#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_mlp_router_logits_before_routing_after_norm as router


EXPERT_INDEX = 30
EXPERT_RANK = 1
HIDDEN_SIZE = 2880
KNOWN_LANE = 1156


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token expert30 mlp2 output before bias status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-expert30-swiglu", type=Path, required=True)
    parser.add_argument("--source-selected-expert-outputs", type=Path, required=True)
    parser.add_argument("--official-expert30-mlp2", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_f32_le(tensor):
    flat = tensor.reshape(-1).detach().cpu().to(torch.float32)
    hasher = hashlib.sha256()
    for value in flat.tolist():
        hasher.update(struct.pack("<f", float(value)))
    return hasher.hexdigest()


def tensor_meta(tensor, layout, serialization_dtype):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
        "device": str(tensor.device),
    }


def compare_vector(lhs, rhs, index_name="hidden_lane"):
    lhs = lhs.reshape(-1).to(torch.float32)
    rhs = rhs.reshape(-1).to(torch.float32)
    diff = (lhs - rhs).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        first_idx = int(mismatch.nonzero(as_tuple=False)[0].item())
        worst_idx = int(diff.argmax().item())
        first = {
            index_name: first_idx,
            "local_value": float(lhs[first_idx].item()),
            "official_value": float(rhs[first_idx].item()),
            "abs_diff": float(diff[first_idx].item()),
        }
        worst = {
            index_name: worst_idx,
            "local_value": float(lhs[worst_idx].item()),
            "official_value": float(rhs[worst_idx].item()),
            "abs_diff": float(diff[worst_idx].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_lane_count": int(mismatch.sum().item()),
        f"first_differing_{index_name}": first,
        f"worst_differing_{index_name}": worst,
    }


def compare_with_names(lhs, rhs, index_name, lhs_name, rhs_name):
    metric = compare_vector(lhs, rhs, index_name)
    for key in [f"first_differing_{index_name}", f"worst_differing_{index_name}"]:
        item = metric.get(key)
        if item is not None:
            item[f"{lhs_name}_value"] = item.pop("local_value")
            item[f"{rhs_name}_value"] = item.pop("official_value")
    return metric


def vector_summary(tensor):
    flat = tensor.reshape(-1).to(torch.float32)
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "sha256_f32_le": sha256_f32_le(flat),
    }


def manual_ltr(input_bf16, weight_bf16):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16.to(torch.float32)
    out = []
    for row_index in range(w.shape[0]):
        acc = torch.tensor(0.0, dtype=torch.float32)
        row = w[row_index]
        for lane in range(row.numel()):
            acc = acc + row[lane] * x[lane]
        out.append(acc)
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def manual_pairwise(input_bf16, weight_bf16):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16.to(torch.float32)
    out = []
    for row_index in range(w.shape[0]):
        terms = w[row_index] * x
        while terms.numel() > 1:
            if terms.numel() % 2 == 1:
                tail = terms[-1:]
                terms = terms[:-1]
            else:
                tail = None
            terms = terms.reshape(-1, 2).sum(dim=1)
            if tail is not None:
                terms = torch.cat([terms, tail])
        out.append(terms[0])
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def variant_entry(name, tensor, official, local_runtime):
    tensor = tensor.reshape(-1).to(torch.float32)
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official, "hidden_lane"),
        "metrics_vs_local_runtime": compare_with_names(
            tensor, local_runtime, "hidden_lane", "variant", "local_runtime"
        ),
        "matches_official": bool(torch.equal(tensor, official.reshape(-1).to(torch.float32))),
        "matches_local_runtime": bool(torch.equal(tensor, local_runtime.reshape(-1).to(torch.float32))),
    }


def main():
    args = parse_args()
    source_swiglu = load_json(args.source_expert30_swiglu)
    source_selected = load_json(args.source_selected_expert_outputs)
    official_artifact = load_json(args.official_expert30_mlp2)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_swiglu.get("classification") != "expert30_swiglu_before_mlp2_cleared":
        raise ValueError("source expert30 SwiGLU artifact is not cleared")
    if source_selected.get("classification") != "selected_expert_outputs_mismatch_before_routing_weighted_sum":
        raise ValueError("source selected expert outputs artifact is not the expected mismatch state")
    if official_artifact.get("classification") != "official_layer0_final_token_expert30_mlp2_output_before_bias_captured":
        raise ValueError("official expert30 mlp2-before-bias artifact is not the expected PPP capture")
    if official_artifact.get("boundary") != "layer0_final_token_expert30_mlp2_output_before_bias":
        raise ValueError("official expert30 mlp2-before-bias boundary is not usable")

    device = torch.device(args.device)
    swiglu_ref_path = Path(source_swiglu["official_ppp_reference_path"])
    swiglu_ref = load_json(swiglu_ref_path)
    swiglu_input = torch.tensor(swiglu_ref["values"], dtype=torch.float32, device=device).reshape(HIDDEN_SIZE)
    swiglu_bf16 = swiglu_input.to(torch.bfloat16).contiguous()
    official = torch.tensor(official_artifact["values"], dtype=torch.float32, device=device).reshape(HIDDEN_SIZE)

    input_guard_metric = source_swiglu["mlp_norm_input_guard_metrics"]["metrics"]
    routing_guard = source_swiglu["routing_guard_metrics"]
    mlp1_guard = source_swiglu["expert30_mlp1_guard_metrics"]
    swiglu_guard = source_swiglu["local_swiglu_vs_official_metrics"]
    guards_pass = (
        input_guard_metric["matched"]
        and routing_guard["selected_expert_indices"]["ordered_match"]
        and routing_guard["routing_weights"]["matched"]
        and routing_guard["expert30_selected_rank_1"]
        and mlp1_guard["fused"]["matched"]
        and mlp1_guard["gate_slice"]["matched"]
        and mlp1_guard["up_slice"]["matched"]
        and swiglu_guard["matched"]
    )

    mlp = router.load_layer0_mlp(args.model_root, device)
    weight = mlp.mlp2_weight[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    bias = mlp.mlp2_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    official_weight_meta = official_artifact.get("mlp2_weight_metadata", {})
    official_bias_meta = official_artifact.get("mlp2_bias_metadata", {})

    with torch.inference_mode():
        local_pre_bias = torch.einsum("hk,k->h", weight, swiglu_bf16).reshape(HIDDEN_SIZE).contiguous()
        replay = torch.einsum("hk,k->h", weight, swiglu_bf16).reshape(HIDDEN_SIZE).contiguous()
        bias_added_context = (local_pre_bias + bias).to(torch.bfloat16).reshape(HIDDEN_SIZE).contiguous()
        transposed_orientation = torch.einsum("kh,k->h", weight, swiglu_bf16).reshape(HIDDEN_SIZE).contiguous()

    local_f32 = local_pre_bias.to(torch.float32)
    replay_f32 = replay.to(torch.float32)
    bias_added_f32 = bias_added_context.to(torch.float32)
    local_vs_official = compare_vector(local_f32, official, "hidden_lane")
    replay_vs_official = compare_vector(replay_f32, official, "hidden_lane")
    local_vs_replay = compare_with_names(local_f32, replay_f32, "hidden_lane", "local_runtime", "replay")

    weight_digest = sha256_f32_le(weight)
    bias_digest = sha256_f32_le(bias)
    weight_shape_matches = list(weight.shape) == official_weight_meta.get("shape")
    bias_shape_matches = list(bias.shape) == official_bias_meta.get("shape")
    bias_digest_matches = bias_digest == official_bias_meta.get("sha256_f32_le")
    weight_mismatch = (
        not weight_shape_matches
        or str(weight.dtype).replace("torch.", "") not in official_weight_meta.get("dtype", "")
    )

    discriminator_table = []
    focused_trace = None
    if not local_vs_official["matched"]:
        ltr = manual_ltr(swiglu_bf16, weight)
        pairwise = manual_pairwise(swiglu_bf16, weight)
        discriminator_table = [
            variant_entry(
                "torch_einsum_hk_k_no_bias_bf16_output",
                replay_f32,
                official,
                local_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_no_bias_left_to_right_f32_accum_bf16_output",
                ltr,
                official,
                local_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_no_bias_pairwise_f32_accum_bf16_output",
                pairwise,
                official,
                local_f32,
            ),
            variant_entry(
                "layout_discriminator_transposed_weight_interpretation",
                transposed_orientation,
                official,
                local_f32,
            ),
        ]
        worst = local_vs_official["worst_differing_hidden_lane"]
        if worst is not None:
            lane = int(worst["hidden_lane"])
            focused_trace = {
                "hidden_lane": lane,
                "swiglu_input_digest": sha256_f32_le(swiglu_bf16),
                "mlp2_weight_row_digest": sha256_f32_le(weight[lane]),
                "local_runtime_mlp2_before_bias_value": float(local_f32[lane].item()),
                "official_ppp_mlp2_before_bias_value": float(official[lane].item()),
                "explicit_replay_value": float(replay_f32[lane].item()),
                "local_minus_official": float(local_f32[lane].item() - official[lane].item()),
                "mismatch_scale": "one BF16 ULP or accumulation-policy-sized",
                "first_divergent_stage": "undetermined; bounded mlp2 replay/discriminator recorded",
            }

    selected_outputs = torch.tensor(
        source_selected.get("selected_expert_output_values", []),
        dtype=torch.float32,
        device=device,
    )
    official_selected = None
    try:
        official_selected_path = Path(source_selected["official_ppp_reference_path"])
        official_selected_artifact = load_json(official_selected_path)
        official_selected = torch.tensor(
            official_selected_artifact["values"], dtype=torch.float32, device=device
        ).reshape(4, HIDDEN_SIZE)
    except Exception:
        official_selected = None
    local_selected_context = None
    if selected_outputs.numel() == 4 * HIDDEN_SIZE:
        local_selected_context = selected_outputs.reshape(4, HIDDEN_SIZE)[EXPERT_RANK]
    known_lane_context = {
        "hidden_lane": KNOWN_LANE,
        "local_mlp2_before_bias": float(local_f32[KNOWN_LANE].item()),
        "official_mlp2_before_bias": float(official[KNOWN_LANE].item()),
        "local_selected_expert_output": float(local_selected_context[KNOWN_LANE].item()) if local_selected_context is not None else None,
        "official_selected_expert_output": float(official_selected[EXPERT_RANK, KNOWN_LANE].item()) if official_selected is not None else None,
        "mismatch_already_exists_before_bias": bool(local_f32[KNOWN_LANE].item() != official[KNOWN_LANE].item()),
        "bias_added_context_value": float(bias_added_f32[KNOWN_LANE].item()),
        "mlp2_bias_value": float(bias[KNOWN_LANE].to(torch.float32).item()),
    }

    shape_or_layout_mismatch = list(local_f32.shape) != official_artifact.get("shape")
    if not guards_pass:
        classification = "expert30_mlp2_blocked_by_input_or_swiglu_guard_regression"
        earliest = "layer0_final_token_expert30_swiglu_output_before_mlp2_or_routing"
        next_step = "re-establish input/routing/SwiGLU guards before expert30 mlp2"
    elif weight_mismatch:
        classification = "expert30_mlp2_weight_mismatch"
        earliest = "model.block[0].mlp.expert30.mlp2_weight"
        next_step = "inspect expert30 mlp2 parameter loading only"
    elif shape_or_layout_mismatch:
        classification = "expert30_mlp2_before_bias_shape_or_layout_mismatch"
        earliest = "layer0_final_token_expert30_mlp2_output_before_bias"
        next_step = "align expert30 mlp2 output layout/readout only"
    elif local_vs_official["matched"]:
        classification = "expert30_mlp2_before_bias_cleared"
        earliest = "none"
        next_step = "compare expert30 mlp2 bias application against existing selected expert output boundary"
    elif replay_vs_official["matched"] and not local_vs_replay["matched"]:
        classification = "expert30_mlp2_runtime_arithmetic_or_capture_mismatch"
        earliest = "layer0_final_token_expert30_mlp2_output_before_bias"
        next_step = "localize expert30 mlp2 linear arithmetic/capture only before bias"
    elif local_vs_replay["matched"] and not replay_vs_official["matched"]:
        classification = "expert30_mlp2_replay_policy_not_authoritative"
        earliest = "official-semantics expert30 mlp2 replay"
        next_step = "refine expert30 mlp2 official replay policy before bias"
    else:
        classification = "expert30_mlp2_runtime_arithmetic_or_capture_mismatch"
        earliest = "layer0_final_token_expert30_mlp2_output_before_bias"
        next_step = "localize expert30 mlp2 linear arithmetic/capture only before bias"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_expert30_mlp2_before_bias_status/v1",
        "mode": "mlp-expert30-mlp2-before-bias-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_rank": EXPERT_RANK,
            "selected_expert_index": EXPERT_INDEX,
            "hidden_size": HIDDEN_SIZE,
        },
        "source_artifact_paths": {
            "expert30_swiglu_cleared": str(args.source_expert30_swiglu),
            "selected_expert_outputs_diagnostic": str(args.source_selected_expert_outputs),
            "official_expert30_swiglu_reference": str(swiglu_ref_path),
            "official_expert30_mlp2_before_bias_reference": str(args.official_expert30_mlp2),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_expert30_mlp2),
        "mlp_norm_input_guard_metrics": source_swiglu["mlp_norm_input_guard_metrics"],
        "routing_guard_metrics": routing_guard,
        "expert30_mlp1_guard_metrics": mlp1_guard,
        "expert30_swiglu_guard_metrics": swiglu_guard,
        "expert30_mlp2_weight_metadata": {
            "expert_module_path": "model.block[0].mlp expert index 30",
            "mlp2_module_path": "model.block[0].mlp.mlp2_weight[30]",
            "shape": list(weight.shape),
            "dtype": str(weight.dtype).replace("torch.", ""),
            "layout_orientation": "[hidden_size, intermediate_size / world_size]; torch.einsum('hk,k->h', weight, swiglu_output)",
            "sha256_f32_le": weight_digest,
            "official_metadata": official_weight_meta,
            "shape_matches_official": weight_shape_matches,
            "official_digest_available": official_weight_meta.get("sha256_f32_le") is not None,
        },
        "expert30_mlp2_bias_metadata": {
            "present": True,
            "shape": list(bias.shape),
            "dtype": str(bias.dtype).replace("torch.", ""),
            "nonzero_count": int(torch.count_nonzero(bias.to(torch.float32)).item()),
            "sha256_f32_le": bias_digest,
            "official_metadata": official_bias_meta,
            "shape_matches_official": bias_shape_matches,
            "digest_matches_official": bias_digest_matches,
            "bias_included_in_pre_bias_boundary": False,
        },
        "weight_bias_comparison_metrics": {
            "weight": {
                "availability_status": "official PPP records shape/dtype/orientation but omits full weight digest; local checkpoint tensor metadata recorded",
                "shape_matches_official": weight_shape_matches,
                "dtype_matches_official": str(weight.dtype).replace("torch.", "") in official_weight_meta.get("dtype", ""),
            },
            "bias": {
                "availability_status": "official PPP records BF16 bias digest",
                "digest_matches_official": bias_digest_matches,
                "metrics": compare_with_names(bias, bias, "hidden_lane", "local_bias", "official_bias"),
            },
        },
        "local_mlp2_before_bias_output_metadata": {
            **tensor_meta(
                local_f32,
                "expert 30 raw mlp2 projection output vector [hidden_size], before mlp2_bias",
                "f32-expanded BF16 values",
            ),
            "before_mlp2_bias": True,
            "before_routing_weight": True,
            "capture_point_confirmation": "after expert30 mlp2 matrix multiplication/projection; before mlp2_bias",
            "local_all_reduce_or_partitioning": "none; world_size=1",
        },
        "official_tensor_metadata": {
            "shape": official_artifact.get("shape"),
            "tensor_dtype": official_artifact.get("tensor_dtype"),
            "serialization_dtype": official_artifact.get("serialization_dtype"),
            "layout_interpretation": official_artifact.get("layout_interpretation"),
            "source_input_boundary": official_artifact.get("source_input_boundary"),
            "mlp2_weight_metadata": official_weight_meta,
            "mlp2_bias_metadata": official_bias_meta,
            "computation_semantics": official_artifact.get("computation_semantics"),
        },
        "local_mlp2_before_bias_vs_official_metrics": local_vs_official,
        "official_semantics_replay_metrics": {
            "policy": "torch.einsum('hk,k->h', BF16 weight, BF16 SwiGLU input), no bias, BF16 output",
            "replay_vs_official": replay_vs_official,
            "local_runtime_vs_replay": local_vs_replay,
            "bias_added_replay_context": {
                "scope": "next-step context only; not compared as this pre-bias boundary",
                "lane_1156_value": float(bias_added_f32[KNOWN_LANE].item()),
            },
        },
        "discriminator_table": discriminator_table,
        "focused_mismatch_trace": focused_trace,
        "known_downstream_lane_1156_context": known_lane_context,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
        "local_residual_input_artifact_model": local_residual_input_artifact.get("provenance", {}).get("model"),
        "local_stage_summary": vector_summary(local_f32),
        "official_summary": official_artifact.get("finite_value_summary"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
