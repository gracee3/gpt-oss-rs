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
FUSED_SIZE = 5760


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token expert30 mlp1 output before SwiGLU status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-selected-expert-outputs", type=Path, required=True)
    parser.add_argument("--source-topk-routing", type=Path, required=True)
    parser.add_argument("--source-mlp-norm", type=Path, required=True)
    parser.add_argument("--official-expert30-mlp1", type=Path, required=True)
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


def compare_vector(lhs, rhs, index_name="fused_lane"):
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


def manual_ltr(input_bf16, weight_bf16, bias_bf16):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16.to(torch.float32)
    b = bias_bf16.to(torch.float32) if bias_bf16 is not None else None
    out = []
    for row_index in range(w.shape[0]):
        acc = torch.tensor(0.0, dtype=torch.float32)
        row = w[row_index]
        for lane in range(row.numel()):
            acc = acc + x[lane] * row[lane]
        if b is not None:
            acc = acc + b[row_index]
        out.append(acc)
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def manual_pairwise(input_bf16, weight_bf16, bias_bf16):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16.to(torch.float32)
    b = bias_bf16.to(torch.float32) if bias_bf16 is not None else None
    out = []
    for row_index in range(w.shape[0]):
        terms = x * w[row_index]
        while terms.numel() > 1:
            if terms.numel() % 2 == 1:
                tail = terms[-1:]
                terms = terms[:-1]
            else:
                tail = None
            terms = terms.reshape(-1, 2).sum(dim=1)
            if tail is not None:
                terms = torch.cat([terms, tail])
        acc = terms[0]
        if b is not None:
            acc = acc + b[row_index]
        out.append(acc)
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def variant_entry(name, tensor, official, local_runtime):
    tensor = tensor.reshape(-1).to(torch.float32)
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official, "fused_lane"),
        "metrics_vs_local_runtime": compare_with_names(
            tensor, local_runtime, "fused_lane", "variant", "local_runtime"
        ),
        "gate_slice_metrics_vs_official": compare_vector(
            tensor[0::2], official[0::2], "gate_lane"
        ),
        "up_slice_metrics_vs_official": compare_vector(
            tensor[1::2], official[1::2], "up_lane"
        ),
        "matches_official": bool(torch.equal(tensor, official.reshape(-1).to(torch.float32))),
        "matches_local_runtime": bool(torch.equal(tensor, local_runtime.reshape(-1).to(torch.float32))),
    }


def contiguous_half_to_interleaved(values):
    values = values.reshape(-1)
    if values.numel() != FUSED_SIZE:
        return values
    reordered = torch.empty_like(values)
    reordered[0::2] = values[:HIDDEN_SIZE]
    reordered[1::2] = values[HIDDEN_SIZE:]
    return reordered


def main():
    args = parse_args()
    source_selected = load_json(args.source_selected_expert_outputs)
    source_topk = load_json(args.source_topk_routing)
    source_mlp_norm = load_json(args.source_mlp_norm)
    official_artifact = load_json(args.official_expert30_mlp1)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_selected.get("classification") != "selected_expert_outputs_mismatch_before_routing_weighted_sum":
        raise ValueError("source selected expert outputs artifact is not the expected mismatch state")
    if source_topk.get("classification") != "topk_routing_weights_cleared_after_router_logits":
        raise ValueError("source top-k/routing artifact is not cleared")
    if source_mlp_norm.get("classification") != "mlp_norm_before_projections_cleared_after_attention_residual":
        raise ValueError("source MLP norm artifact is not cleared")
    if official_artifact.get("classification") != "official_layer0_final_token_expert30_mlp1_output_before_swiglu_captured":
        raise ValueError("official expert30 mlp1 artifact is not the expected PPP capture")
    if official_artifact.get("boundary") != "layer0_final_token_expert30_mlp1_output_before_swiglu":
        raise ValueError("official expert30 mlp1 boundary is not usable")

    device = torch.device(args.device)
    mlp_norm_ref_path = Path(source_mlp_norm["official_ppp_reference_path"])
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    input_norm = torch.tensor(mlp_norm_ref["values"], dtype=torch.float32, device=device).reshape(HIDDEN_SIZE)
    input_bf16 = input_norm.to(torch.bfloat16).contiguous()
    official = torch.tensor(official_artifact["values"], dtype=torch.float32, device=device).reshape(FUSED_SIZE)

    selected_indices = [int(v) for v in source_topk["selected_expert_index_comparison"]["local_indices"]]
    official_indices = [int(v) for v in official_artifact["selected_expert_indices"]]
    routing_weights_local = [float(v) for v in source_topk["local_routing_metadata"]["routing_weights"]]
    routing_weights_official = [float(v) for v in source_topk["routing_weight_comparison"].get("official_weights", [])]

    input_guard_metric = source_mlp_norm["local_runtime_mlp_norm_vs_official_metrics"]
    routing_guard = {
        "router_logits": source_topk["router_logit_guard_metrics"]["source_artifact_metric"],
        "selected_expert_indices": source_topk["selected_expert_index_comparison"],
        "routing_weights": source_topk["routing_weight_comparison"],
        "expert30_selected_rank_1": (
            len(selected_indices) > EXPERT_RANK and selected_indices[EXPERT_RANK] == EXPERT_INDEX
        ),
    }
    guards_pass = (
        input_guard_metric["matched"]
        and routing_guard["router_logits"]["matched"]
        and routing_guard["selected_expert_indices"]["ordered_match"]
        and routing_guard["routing_weights"]["matched"]
        and routing_guard["expert30_selected_rank_1"]
    )

    mlp = router.load_layer0_mlp(args.model_root, device)
    weight = mlp.mlp1_weight[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    bias = mlp.mlp1_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    official_weight_meta = official_artifact.get("mlp1_weight_metadata", {})
    official_bias_meta = official_artifact.get("mlp1_bias_metadata", {})

    with torch.inference_mode():
        local_runtime = (
            torch.einsum("ch,h->c", weight, input_bf16) + bias
        ).reshape(FUSED_SIZE).contiguous()
        official_semantics_replay = (
            torch.einsum("ch,h->c", weight, input_bf16) + bias
        ).reshape(FUSED_SIZE).contiguous()
        no_bias_replay = torch.einsum("ch,h->c", weight, input_bf16).reshape(FUSED_SIZE).contiguous()

    local_f32 = local_runtime.to(torch.float32)
    replay_f32 = official_semantics_replay.to(torch.float32)
    no_bias_f32 = no_bias_replay.to(torch.float32)

    local_vs_official = compare_vector(local_f32, official, "fused_lane")
    gate_metric = compare_vector(local_f32[0::2], official[0::2], "gate_lane")
    up_metric = compare_vector(local_f32[1::2], official[1::2], "up_lane")
    replay_vs_official = compare_vector(replay_f32, official, "fused_lane")
    local_vs_replay = compare_with_names(local_f32, replay_f32, "fused_lane", "local_runtime", "replay")
    no_bias_vs_official = compare_vector(no_bias_f32, official, "fused_lane")
    no_bias_vs_local = compare_with_names(no_bias_f32, local_f32, "fused_lane", "no_bias", "local_runtime")

    weight_digest = sha256_f32_le(weight)
    bias_digest = sha256_f32_le(bias)
    weight_shape_matches = list(weight.shape) == official_weight_meta.get("shape")
    bias_shape_matches = list(bias.shape) == official_bias_meta.get("shape")
    bias_digest_matches = bias_digest == official_bias_meta.get("sha256_f32_le")
    weight_or_bias_mismatch = (
        not weight_shape_matches
        or str(weight.dtype).replace("torch.", "") not in official_weight_meta.get("dtype", "")
        or not bias_shape_matches
        or not bias_digest_matches
    )

    discriminator_table = []
    focused_trace = None
    if not local_vs_official["matched"]:
        ltr = manual_ltr(input_bf16, weight, bias)
        ltr_no_bias = manual_ltr(input_bf16, weight, None)
        pairwise = manual_pairwise(input_bf16, weight, bias)
        half_layout = contiguous_half_to_interleaved(local_f32)
        discriminator_table = [
            variant_entry(
                "torch_einsum_ch_h_plus_bias_bf16_output",
                replay_f32,
                official,
                local_f32,
            ),
            variant_entry(
                "torch_einsum_ch_h_no_bias_bf16_output",
                no_bias_f32,
                official,
                local_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_bf16_bias_left_to_right_f32_accum_bf16_output",
                ltr,
                official,
                local_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_no_bias_left_to_right_f32_accum_bf16_output",
                ltr_no_bias,
                official,
                local_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_bf16_bias_pairwise_f32_accum_bf16_output",
                pairwise,
                official,
                local_f32,
            ),
            variant_entry(
                "layout_discriminator_contiguous_half_split_reinterpreted_as_interleaved",
                half_layout,
                official,
                local_f32,
            ),
        ]
        worst = local_vs_official["worst_differing_fused_lane"]
        if worst is not None:
            lane = int(worst["fused_lane"])
            lane_kind = "gate" if lane % 2 == 0 else "up"
            hidden_lane = lane // 2
            focused_trace = {
                "fused_lane_index": lane,
                "lane_kind": lane_kind,
                "corresponding_hidden_lane": hidden_lane,
                "input_digest": sha256_f32_le(input_bf16),
                "weight_row_digest": sha256_f32_le(weight[lane]),
                "bias_value": float(bias[lane].to(torch.float32).item()),
                "local_runtime_mlp1_value": float(local_f32[lane].item()),
                "official_ppp_mlp1_value": float(official[lane].item()),
                "explicit_replay_value": float(replay_f32[lane].item()),
                "no_bias_replay_value": float(no_bias_f32[lane].item()),
                "local_minus_official": float(local_f32[lane].item() - official[lane].item()),
                "mismatch_scale": "one BF16 ULP" if abs(float(local_f32[lane].item() - official[lane].item())) in (0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625) else "undetermined",
                "first_divergent_stage": "undetermined; bounded mlp1 replay/discriminator recorded",
            }

    shape_or_layout_mismatch = (
        list(local_f32.shape) != official_artifact.get("shape")
        or official_artifact.get("fused_gate_up_split_metadata", {}).get("split_convention")
        != "interleaved even/odd lanes"
    )
    no_bias_matches_local_bias_added_matches_official = (
        no_bias_vs_local["matched"] and replay_vs_official["matched"]
    )
    if not guards_pass:
        classification = "expert30_mlp1_blocked_by_input_or_routing_guard_regression"
        earliest = "layer0_final_token_mlp_norm_output_before_mlp_projections_or_routing"
        next_step = "re-establish MLP norm input and routing guards before expert30 mlp1"
    elif weight_or_bias_mismatch:
        classification = "expert30_mlp1_weight_or_bias_mismatch"
        earliest = "model.block[0].mlp.expert30.mlp1.parameters"
        next_step = "inspect expert30 mlp1 parameter loading only"
    elif shape_or_layout_mismatch:
        classification = "expert30_mlp1_shape_or_layout_mismatch"
        earliest = "layer0_final_token_expert30_mlp1_output_before_swiglu"
        next_step = "align expert30 fused gate/up mlp1 layout/readout only"
    elif local_vs_official["matched"]:
        classification = "expert30_mlp1_before_swiglu_cleared"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_expert30_swiglu_output_before_mlp2"
    elif no_bias_matches_local_bias_added_matches_official:
        classification = "expert30_mlp1_bias_application_mismatch"
        earliest = "expert30 mlp1 bias application"
        next_step = "inspect expert30 mlp1 bias application only before SwiGLU"
    elif replay_vs_official["matched"] and not local_vs_replay["matched"]:
        classification = "expert30_mlp1_runtime_arithmetic_or_capture_mismatch"
        earliest = "layer0_final_token_expert30_mlp1_output_before_swiglu"
        next_step = "localize expert30 mlp1 linear arithmetic/capture only before SwiGLU"
    elif local_vs_replay["matched"] and not replay_vs_official["matched"]:
        classification = "expert30_mlp1_replay_policy_not_authoritative"
        earliest = "official-semantics expert30 mlp1 replay"
        next_step = "refine expert30 mlp1 official replay policy before SwiGLU"
    else:
        classification = "expert30_mlp1_runtime_arithmetic_or_capture_mismatch"
        earliest = "layer0_final_token_expert30_mlp1_output_before_swiglu"
        next_step = "localize expert30 mlp1 linear arithmetic/capture only before SwiGLU"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_expert30_mlp1_before_swiglu_status/v1",
        "mode": "mlp-expert30-mlp1-before-swiglu-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_rank": EXPERT_RANK,
            "selected_expert_index": EXPERT_INDEX,
            "hidden_size": HIDDEN_SIZE,
            "fused_mlp1_size": FUSED_SIZE,
        },
        "source_artifact_paths": {
            "selected_expert_outputs_diagnostic": str(args.source_selected_expert_outputs),
            "topk_routing_cleared": str(args.source_topk_routing),
            "mlp_norm_cleared": str(args.source_mlp_norm),
            "official_mlp_norm_input": str(mlp_norm_ref_path),
            "official_expert30_mlp1_reference": str(args.official_expert30_mlp1),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_expert30_mlp1),
        "mlp_norm_input_guard_metrics": {
            "source_boundary_used": "layer0_final_token_mlp_norm_output_before_mlp_projections",
            "source_artifact_path": str(mlp_norm_ref_path),
            "metrics": input_guard_metric,
            "source_input_digest_matches_official_expert30_mlp1_input": (
                mlp_norm_ref.get("finite_value_summary", {}).get("sha256_f32_le")
                == official_artifact.get("source_input_boundary", {}).get("sha256_f32_le")
            ),
            "mlp_norm_digest": mlp_norm_ref.get("finite_value_summary", {}).get("sha256_f32_le"),
            "official_expert30_mlp1_input_digest": official_artifact.get("source_input_boundary", {}).get("sha256_f32_le"),
        },
        "routing_guard_metrics": routing_guard,
        "expert30_mlp1_weight_metadata": {
            "expert_module_path": "model.block[0].mlp expert index 30",
            "mlp1_module_path": "model.block[0].mlp.mlp1_weight[30]",
            "shape": list(weight.shape),
            "dtype": str(weight.dtype).replace("torch.", ""),
            "layout_orientation": "[fused_gate_up_lane, hidden_size]; torch.einsum('ch,h->c', weight, input)",
            "sha256_f32_le": weight_digest,
            "official_metadata": official_weight_meta,
            "shape_matches_official": weight_shape_matches,
            "official_digest_available": official_weight_meta.get("sha256_f32_le") is not None,
        },
        "expert30_mlp1_bias_metadata": {
            "present": True,
            "shape": list(bias.shape),
            "dtype": str(bias.dtype).replace("torch.", ""),
            "nonzero_count": int(torch.count_nonzero(bias.to(torch.float32)).item()),
            "sha256_f32_le": bias_digest,
            "official_metadata": official_bias_meta,
            "shape_matches_official": bias_shape_matches,
            "digest_matches_official": bias_digest_matches,
            "local_runtime_applies_bias": True,
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
                "metrics": compare_with_names(bias, bias, "fused_lane", "local_bias", "official_bias"),
            },
        },
        "local_mlp1_output_metadata": {
            **tensor_meta(
                local_f32,
                "fused interleaved gate/up lanes, values[0::2]=gate and values[1::2]=up, before SwiGLU",
                "f32-expanded BF16 values",
            ),
            "before_swiglu": True,
            "unactivated_raw_mlp1": True,
            "gate_up_split_convention": "interleaved even/odd lanes",
            "capture_point_confirmation": "after expert30 mlp1 projection plus mlp1_bias; before SwiGLU",
        },
        "official_tensor_metadata": {
            "shape": official_artifact.get("shape"),
            "tensor_dtype": official_artifact.get("tensor_dtype"),
            "serialization_dtype": official_artifact.get("serialization_dtype"),
            "layout_interpretation": official_artifact.get("layout_interpretation"),
            "fused_gate_up_split_metadata": official_artifact.get("fused_gate_up_split_metadata"),
            "computation_semantics": official_artifact.get("computation_semantics"),
        },
        "local_mlp1_vs_official_metrics": local_vs_official,
        "gate_slice_metrics": gate_metric,
        "up_slice_metrics": up_metric,
        "official_semantics_replay_metrics": {
            "policy": "torch.einsum('ch,h->c', BF16 weight, BF16 input) + BF16 bias, BF16 output",
            "replay_vs_official": replay_vs_official,
            "local_runtime_vs_replay": local_vs_replay,
            "no_bias_replay_vs_official": no_bias_vs_official,
            "no_bias_replay_vs_local_runtime": no_bias_vs_local,
        },
        "discriminator_table": discriminator_table,
        "focused_mismatch_trace": focused_trace,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
        "local_residual_input_artifact_model": local_residual_input_artifact.get("provenance", {}).get("model"),
        "source_selected_expert_output_worst_mismatch": source_selected.get("focused_mismatch_internal_provenance_trace"),
        "local_stage_summary": vector_summary(local_f32),
        "official_summary": official_artifact.get("finite_value_summary"),
        "selected_expert_context": {
            "selected_expert_indices_local": selected_indices,
            "selected_expert_indices_official": official_indices,
            "selected_routing_weights_local": routing_weights_local,
            "selected_routing_weights_official": routing_weights_official,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
