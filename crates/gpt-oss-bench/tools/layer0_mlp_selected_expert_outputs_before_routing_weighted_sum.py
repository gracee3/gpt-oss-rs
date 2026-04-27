#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_mlp_router_logits_before_routing_after_norm as router


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token selected expert outputs before routing weighted sum status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-topk-routing", type=Path, required=True)
    parser.add_argument("--official-selected-expert-outputs", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_flat(lhs, rhs):
    lhs = lhs.reshape(-1).to(torch.float32)
    rhs = rhs.reshape(-1).to(torch.float32)
    diff = (lhs - rhs).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        first_flat = int(mismatch.nonzero(as_tuple=False)[0].item())
        worst_flat = int(diff.argmax().item())
        first = {
            "rank": first_flat // 2880,
            "hidden_lane": first_flat % 2880,
            "local_value": float(lhs[first_flat].item()),
            "official_value": float(rhs[first_flat].item()),
            "abs_diff": float(diff[first_flat].item()),
        }
        worst = {
            "rank": worst_flat // 2880,
            "hidden_lane": worst_flat % 2880,
            "local_value": float(lhs[worst_flat].item()),
            "official_value": float(rhs[worst_flat].item()),
            "abs_diff": float(diff[worst_flat].item()),
        }
    if lhs.numel() == 0:
        max_abs = 0.0
        mean_abs = 0.0
    else:
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
    rank_mismatch_count = 0
    if lhs.numel():
        rank_mismatch_count = int(mismatch.reshape(-1, 2880).any(dim=1).sum().item())
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_expert_rank_count": rank_mismatch_count,
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_rank_lane": first,
        "worst_differing_rank_lane": worst,
    }


def compare_rank(lhs, rhs, rank, expert_index):
    lhs = lhs.reshape(2880).to(torch.float32)
    rhs = rhs.reshape(2880).to(torch.float32)
    diff = (lhs - rhs).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        first_lane = int(mismatch.nonzero(as_tuple=False)[0].item())
        worst_lane = int(diff.argmax().item())
        first = {
            "hidden_lane": first_lane,
            "local_value": float(lhs[first_lane].item()),
            "official_value": float(rhs[first_lane].item()),
            "abs_diff": float(diff[first_lane].item()),
        }
        worst = {
            "hidden_lane": worst_lane,
            "local_value": float(lhs[worst_lane].item()),
            "official_value": float(rhs[worst_lane].item()),
            "abs_diff": float(diff[worst_lane].item()),
        }
    return {
        "rank": int(rank),
        "expert_index": int(expert_index),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_hidden_lane": first,
        "worst_differing_hidden_lane": worst,
    }


def tensor_meta(tensor, layout, serialization_dtype):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
        "device": str(tensor.device),
    }


def sha256_f32_le(tensor):
    flat = tensor.reshape(-1).detach().cpu().to(torch.float32)
    hasher = hashlib.sha256()
    for value in flat.tolist():
        hasher.update(struct.pack("<f", float(value)))
    return hasher.hexdigest()


def vector_summary(tensor):
    flat = tensor.reshape(-1).to(torch.float32)
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "sha256_f32_le": sha256_f32_le(flat),
    }


def compute_selected_expert_outputs(mlp, norm_input_bf16, selected_indices):
    from gpt_oss.torch.model import swiglu

    expert_indices = torch.tensor(selected_indices, dtype=torch.long, device=norm_input_bf16.device).reshape(1, -1)
    t = norm_input_bf16.reshape(1, -1)
    mlp1_weight = mlp.mlp1_weight[expert_indices, ...]
    mlp1_bias = mlp.mlp1_bias[expert_indices, ...]
    mlp1 = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    swiglu_out = swiglu(mlp1, limit=mlp.swiglu_limit)
    mlp2_weight = mlp.mlp2_weight[expert_indices, ...]
    mlp2_bias = mlp.mlp2_bias[expert_indices, ...]
    mlp2_pre_bias = torch.einsum("beck,bek->bec", mlp2_weight, swiglu_out)
    final = (mlp2_pre_bias + mlp2_bias).reshape(len(selected_indices), 2880).contiguous()
    return {
        "mlp1_pre_activation": mlp1.reshape(len(selected_indices), -1).contiguous(),
        "swiglu_output": swiglu_out.reshape(len(selected_indices), -1).contiguous(),
        "mlp2_pre_bias": mlp2_pre_bias.reshape(len(selected_indices), 2880).contiguous(),
        "final_unweighted": final,
    }


def main():
    args = parse_args()
    source_topk = load_json(args.source_topk_routing)
    official_selected_artifact = load_json(args.official_selected_expert_outputs)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_topk.get("classification") != "topk_routing_weights_cleared_after_router_logits":
        raise ValueError("source top-k/routing artifact is not cleared")
    if official_selected_artifact.get("classification") != "official_layer0_final_token_selected_expert_outputs_before_routing_weighted_sum_captured":
        raise ValueError("official selected expert outputs artifact is not the expected PPP capture")
    if official_selected_artifact.get("boundary") != "layer0_final_token_selected_expert_outputs_before_routing_weighted_sum":
        raise ValueError("official selected expert outputs boundary is not usable")

    device = torch.device(args.device)
    mlp_norm_ref_path = Path(".live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json")
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    norm_input = torch.tensor(mlp_norm_ref["values"], dtype=torch.float32, device=device).reshape(2880)
    norm_input_bf16 = norm_input.to(torch.bfloat16).contiguous()
    selected_indices = [int(v) for v in source_topk["selected_expert_index_comparison"]["local_indices"]]
    official_indices = [int(v) for v in official_selected_artifact["selected_expert_indices"]]
    routing_weights = [float(v) for v in source_topk["local_routing_metadata"]["routing_weights"]]
    official_weights = [float(v) for v in official_selected_artifact["selected_routing_weights"]]

    mlp = router.load_layer0_mlp(args.model_root, device)
    with torch.inference_mode():
        stages = compute_selected_expert_outputs(mlp, norm_input_bf16, selected_indices)
    local_outputs = stages["final_unweighted"].to(torch.float32)
    official_outputs = torch.tensor(
        official_selected_artifact["values"], dtype=torch.float32, device=device
    ).reshape(4, 2880)
    all_metric = compare_flat(local_outputs, official_outputs)
    per_rank = [
        compare_rank(local_outputs[rank], official_outputs[rank], rank, selected_indices[rank])
        for rank in range(len(selected_indices))
    ]

    weighted_hypothesis = local_outputs * torch.tensor(routing_weights, dtype=torch.float32, device=device).reshape(-1, 1)
    weighted_metric = compare_flat(weighted_hypothesis, official_outputs)

    focused_trace = None
    if not all_metric["matched"]:
        worst = all_metric["worst_differing_rank_lane"]
        rank = int(worst["rank"])
        lane = int(worst["hidden_lane"])
        expert = selected_indices[rank]
        focused_trace = {
            "rank": rank,
            "expert_index": expert,
            "hidden_lane": lane,
            "expert_input_metadata": {
                "shape": [2880],
                "dtype": "torch.bfloat16",
                "digest": sha256_f32_le(norm_input_bf16),
            },
            "mlp1_pre_activation_metadata": tensor_meta(
                stages["mlp1_pre_activation"][rank],
                "[intermediate_size * 2] before SwiGLU",
                "f32-expanded BF16 values",
            ),
            "swiglu_output_metadata": tensor_meta(
                stages["swiglu_output"][rank],
                "[intermediate_size] after SwiGLU",
                "f32-expanded BF16 values",
            ),
            "mlp2_pre_bias_metadata": tensor_meta(
                stages["mlp2_pre_bias"][rank],
                "[hidden_size] before mlp2_bias",
                "f32-expanded BF16 values",
            ),
            "mlp2_bias_value": float(mlp.mlp2_bias[expert, lane].to(torch.float32).item()),
            "local_final_unweighted_value": float(local_outputs[rank, lane].item()),
            "official_final_unweighted_value": float(official_outputs[rank, lane].item()),
            "local_minus_official": float(local_outputs[rank, lane].item() - official_outputs[rank, lane].item()),
            "mlp1_weight_shape": list(mlp.mlp1_weight[expert].shape),
            "mlp1_bias_shape": list(mlp.mlp1_bias[expert].shape),
            "mlp2_weight_shape": list(mlp.mlp2_weight[expert].shape),
            "mlp2_bias_shape": list(mlp.mlp2_bias[expert].shape),
            "mlp1_weight_digest": sha256_f32_le(mlp.mlp1_weight[expert]),
            "mlp2_weight_digest": sha256_f32_le(mlp.mlp2_weight[expert]),
            "earliest_available_local_stage": "final selected expert output; no official internal selected-expert substage provided in this mode",
        }

    routing_guards_pass = (
        source_topk["router_logit_guard_metrics"]["source_artifact_metric"]["matched"]
        and source_topk["selected_expert_index_comparison"]["ordered_match"]
        and source_topk["routing_weight_comparison"]["matched"]
    )
    order_matches = selected_indices == official_indices
    layout_matches = list(local_outputs.shape) == official_selected_artifact.get("shape")
    appears_weighted = (not all_metric["matched"]) and weighted_metric["matched"]

    if not routing_guards_pass:
        classification = "selected_expert_outputs_blocked_by_routing_guard_regression"
        earliest = "layer0_final_token_mlp_topk_expert_indices_and_routing_weights"
        next_step = "re-establish top-k/routing guards before selected expert outputs"
    elif not order_matches or not layout_matches:
        classification = "selected_expert_outputs_order_or_layout_mismatch"
        earliest = "layer0_final_token_selected_expert_outputs_before_routing_weighted_sum"
        next_step = "inspect selected expert output order/layout only"
    elif appears_weighted:
        classification = "selected_expert_outputs_are_weighted_locally"
        earliest = "selected expert output capture point"
        next_step = "inspect local selected expert output capture point only"
    elif not all_metric["matched"]:
        classification = "selected_expert_outputs_mismatch_before_routing_weighted_sum"
        earliest = "layer0_final_token_selected_expert_outputs_before_routing_weighted_sum"
        next_step = "ask PPP for or inspect exactly the first mismatching selected expert internal boundary, starting with expert mlp1 output before SwiGLU for the worst expert"
    else:
        classification = "selected_expert_outputs_cleared_before_routing_weighted_sum"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_selected_expert_outputs_before_routing_weighted_sum_status/v1",
        "mode": "mlp-selected-expert-outputs-before-routing-weighted-sum-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_indices": selected_indices,
            "top_k": len(selected_indices),
            "hidden_size": 2880,
        },
        "source_artifact_paths": {
            "topk_routing_cleared": str(args.source_topk_routing),
            "official_selected_expert_outputs_reference": str(args.official_selected_expert_outputs),
            "official_mlp_norm_input": str(mlp_norm_ref_path),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_selected_expert_outputs),
        "routing_guard_metrics": {
            "router_logits": source_topk["router_logit_guard_metrics"]["source_artifact_metric"],
            "selected_expert_indices": source_topk["selected_expert_index_comparison"],
            "routing_weights": source_topk["routing_weight_comparison"],
        },
        "local_selected_expert_output_metadata": {
            **tensor_meta(
                local_outputs,
                "[top_k_rank, hidden_size] sorted top-k order, unweighted expert outputs",
                "f32-expanded BF16 values",
            ),
            "selected_expert_indices_order": selected_indices,
            "outputs_are_unweighted": True,
            "routing_weights_already_applied": False,
            "expert_module_path_pattern": "model.block[0].mlp expert parameter rows indexed by selected experts",
            "output_capture_point": "after mlp1 -> SwiGLU -> mlp2 -> mlp2_bias; before routing weight",
        },
        "official_tensor_metadata": {
            "shape": official_selected_artifact.get("shape"),
            "tensor_dtype": official_selected_artifact.get("tensor_dtype"),
            "serialization_dtype": official_selected_artifact.get("serialization_dtype"),
            "layout_interpretation": official_selected_artifact.get("layout_interpretation"),
            "outputs_are_unweighted": official_selected_artifact.get("outputs_are_unweighted"),
            "routing_weights_already_applied": official_selected_artifact.get("routing_weights_already_applied"),
            "selected_expert_order_convention": official_selected_artifact.get("selected_expert_order_convention"),
            "selected_expert_computation_metadata": official_selected_artifact.get("selected_expert_computation_metadata"),
        },
        "selected_expert_output_comparison_metrics": all_metric,
        "per_rank_metrics": per_rank,
        "weighted_output_hypothesis_metric": weighted_metric,
        "focused_mismatch_internal_provenance_trace": focused_trace,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
        "local_residual_input_artifact_model": local_residual_input_artifact.get("provenance", {}).get("model"),
        "local_stage_summaries": {
            "mlp1_pre_activation": vector_summary(stages["mlp1_pre_activation"]),
            "swiglu_output": vector_summary(stages["swiglu_output"]),
            "mlp2_pre_bias": vector_summary(stages["mlp2_pre_bias"]),
            "final_unweighted": vector_summary(local_outputs),
        },
        "official_summary": official_selected_artifact.get("finite_value_summary"),
        "selected_routing_weights": {
            "local": routing_weights,
            "official": official_weights,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
