#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

import layer0_mlp_router_logits_before_routing_after_norm as router


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token MLP top-k indices and routing weights status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-router-logits", type=Path, required=True)
    parser.add_argument("--official-topk-routing", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_vector(lhs, rhs, index_name="rank"):
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
            "lhs_value": float(lhs[first_idx].item()),
            "rhs_value": float(rhs[first_idx].item()),
            "abs_diff": float(diff[first_idx].item()),
        }
        worst = {
            index_name: worst_idx,
            "lhs_value": float(lhs[worst_idx].item()),
            "rhs_value": float(rhs[worst_idx].item()),
            "abs_diff": float(diff[worst_idx].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_count": int(mismatch.sum().item()),
        f"first_differing_{index_name}": first,
        f"worst_differing_{index_name}": worst,
    }


def compare_indices(local_indices, official_indices):
    local = [int(v) for v in local_indices]
    official = [int(v) for v in official_indices]
    first = None
    for rank, (lhs, rhs) in enumerate(zip(local, official)):
        if lhs != rhs:
            first = {"rank": rank, "local_expert_index": lhs, "official_expert_index": rhs}
            break
    return {
        "local_indices": local,
        "official_indices": official,
        "ordered_match": local == official,
        "set_match_ignoring_order": sorted(local) == sorted(official),
        "first_differing_rank": first,
    }


def routing_from_logits(logits_bf16, top_k, sorted_flag=True, all_softmax=False, cast_output=True):
    top = torch.topk(logits_bf16, k=top_k, dim=0, sorted=sorted_flag)
    indices = top.indices.to(torch.int64)
    values = top.values
    if all_softmax:
        weights = torch.nn.functional.softmax(logits_bf16, dim=0).gather(0, indices)
    else:
        weights = torch.nn.functional.softmax(values, dim=0)
    if cast_output:
        weights = weights.to(torch.bfloat16)
    return {
        "indices": indices,
        "selected_logits": values.to(torch.float32),
        "routing_weights": weights.to(torch.float32),
    }


def variant_entry(name, result, official_indices, official_logits, official_weights):
    index_metric = compare_indices(result["indices"].tolist(), official_indices)
    weight_metric = compare_vector(result["routing_weights"], official_weights, "rank")
    logit_metric = compare_vector(result["selected_logits"], official_logits, "rank")
    return {
        "name": name,
        "selected_indices": [int(v) for v in result["indices"].tolist()],
        "selected_logits": [float(v) for v in result["selected_logits"].tolist()],
        "routing_weights": [float(v) for v in result["routing_weights"].tolist()],
        "routing_weight_sum": float(result["routing_weights"].sum().item()),
        "ordered_index_match_vs_official": index_metric["ordered_match"],
        "selected_logit_metric_vs_official": logit_metric,
        "routing_weight_metric_vs_official": weight_metric,
        "matches_official": bool(
            index_metric["ordered_match"]
            and logit_metric["matched"]
            and weight_metric["matched"]
        ),
    }


def main():
    args = parse_args()
    source_router = load_json(args.source_router_logits)
    official_topk = load_json(args.official_topk_routing)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_router.get("classification") != "router_logits_before_routing_cleared_after_mlp_norm":
        raise ValueError("source router logits artifact is not cleared")
    if official_topk.get("classification") != "official_layer0_final_token_mlp_topk_indices_and_routing_weights_captured":
        raise ValueError("official top-k/routing artifact is not the expected PPP capture")
    if official_topk.get("boundary") != "layer0_final_token_mlp_topk_expert_indices_and_routing_weights":
        raise ValueError("official top-k/routing boundary is not usable")

    device = torch.device(args.device)
    mlp_norm_ref_path = Path(source_router["source_artifact_paths"]["official_mlp_norm_input"])
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    input_norm = torch.tensor(mlp_norm_ref["values"], dtype=torch.float32, device=device).reshape(2880)
    input_bf16 = input_norm.to(torch.bfloat16).contiguous()

    mlp = router.load_layer0_mlp(args.model_root, device)
    gate = mlp.gate
    with torch.inference_mode():
        local_logits = gate(input_bf16.reshape(1, 2880)).reshape(32).contiguous()
    local_logits_f32 = local_logits.to(torch.float32)
    official_router_logits_path = Path(source_router["official_ppp_reference_path"])
    official_router_logits = torch.tensor(
        load_json(official_router_logits_path)["values"], dtype=torch.float32, device=device
    ).reshape(32)
    router_guard = source_router["local_runtime_router_logits_vs_official_metrics"]
    recomputed_router_guard = compare_vector(local_logits_f32, official_router_logits, "expert_index")

    top_k = int(official_topk["top_k"])
    official_indices = [int(v) for v in official_topk["selected_expert_indices"]["values"]]
    official_logits = torch.tensor(
        official_topk["selected_expert_logits"]["values"], dtype=torch.float32, device=device
    )
    official_weights = torch.tensor(
        official_topk["routing_weights"]["values"], dtype=torch.float32, device=device
    )

    local_result = routing_from_logits(local_logits.to(torch.bfloat16), top_k, True, False, True)
    index_cmp = compare_indices(local_result["indices"].tolist(), official_indices)
    logit_cmp = compare_vector(local_result["selected_logits"], official_logits, "rank")
    weight_cmp = compare_vector(local_result["routing_weights"], official_weights, "rank")

    policy_table = []
    focused_trace = None
    if not (index_cmp["ordered_match"] and logit_cmp["matched"] and weight_cmp["matched"]):
        variants = [
            (
                "torch_topk_sorted_true_softmax_selected_logits_bf16_output",
                routing_from_logits(local_logits.to(torch.bfloat16), top_k, True, False, True),
            ),
            (
                "torch_topk_sorted_false_softmax_selected_logits_bf16_output",
                routing_from_logits(local_logits.to(torch.bfloat16), top_k, False, False, True),
            ),
            (
                "torch_topk_sorted_true_softmax_all_32_then_gather_bf16_output",
                routing_from_logits(local_logits.to(torch.bfloat16), top_k, True, True, True),
            ),
            (
                "torch_topk_sorted_true_softmax_selected_logits_f32_compare_before_bf16_cast",
                routing_from_logits(local_logits.to(torch.bfloat16), top_k, True, False, False),
            ),
            (
                "torch_topk_sorted_true_softmax_selected_logits_f32_then_bf16_output",
                routing_from_logits(local_logits.to(torch.float32), top_k, True, False, True),
            ),
        ]
        policy_table = [
            variant_entry(name, result, official_indices, official_logits, official_weights)
            for name, result in variants
        ]
        rank = 0
        if not index_cmp["ordered_match"] and index_cmp["first_differing_rank"] is not None:
            rank = int(index_cmp["first_differing_rank"]["rank"])
        elif not weight_cmp["matched"] and weight_cmp["worst_differing_rank"] is not None:
            rank = int(weight_cmp["worst_differing_rank"]["rank"])
        elif not logit_cmp["matched"] and logit_cmp["worst_differing_rank"] is not None:
            rank = int(logit_cmp["worst_differing_rank"]["rank"])
        focused_trace = {
            "rank": rank,
            "local_expert_index": int(local_result["indices"][rank].item()),
            "official_expert_index": official_indices[rank],
            "local_selected_logit": float(local_result["selected_logits"][rank].item()),
            "official_selected_logit": float(official_logits[rank].item()),
            "local_routing_weight": float(local_result["routing_weights"][rank].item()),
            "official_routing_weight": float(official_weights[rank].item()),
            "local_minus_official_weight": float(
                local_result["routing_weights"][rank].item() - official_weights[rank].item()
            ),
            "likely_cause": "bounded routing policy discriminator recorded",
        }

    if not router_guard["matched"] or not recomputed_router_guard["matched"]:
        classification = "topk_routing_blocked_by_router_logits_regression"
        earliest = "layer0_final_token_mlp_router_logits_before_routing"
        next_step = "re-establish router logits before top-k/routing"
    elif not index_cmp["ordered_match"]:
        classification = "topk_expert_indices_mismatch_after_router_logits_clear"
        earliest = "layer0_final_token_mlp_topk_expert_indices"
        next_step = "inspect top-k ordering/tie-breaking only"
    elif not logit_cmp["matched"]:
        classification = "topk_selected_logits_readout_mismatch"
        earliest = "layer0_final_token_mlp_topk_selected_logits"
        next_step = "inspect selected-logit readout only"
    elif not weight_cmp["matched"]:
        classification = "routing_weight_softmax_or_dtype_policy_mismatch"
        earliest = "layer0_final_token_mlp_routing_weights"
        next_step = "prove scoped routing softmax dtype/output policy before expert dispatch"
    else:
        classification = "topk_routing_weights_cleared_after_router_logits"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_selected_expert_outputs_before_routing_weighted_sum"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_topk_routing_after_router_logits_status/v1",
        "mode": "mlp-topk-routing-after-router-logits-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "num_experts": int(official_topk["number_of_experts"]),
            "top_k": top_k,
        },
        "source_artifact_paths": {
            "router_logits_cleared": str(args.source_router_logits),
            "official_router_logits": str(official_router_logits_path),
            "official_topk_routing_reference": str(args.official_topk_routing),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_topk_routing),
        "router_logit_guard_metrics": {
            "source_boundary_used": "layer0_final_token_mlp_router_logits_before_routing",
            "source_artifact_metric": router_guard,
            "recomputed_local_metric": recomputed_router_guard,
        },
        "local_routing_metadata": {
            "number_of_experts": int(official_topk["number_of_experts"]),
            "top_k": top_k,
            "selected_expert_indices": [int(v) for v in local_result["indices"].tolist()],
            "selected_expert_logits": [float(v) for v in local_result["selected_logits"].tolist()],
            "routing_weights": [float(v) for v in local_result["routing_weights"].tolist()],
            "routing_weight_sum": float(local_result["routing_weights"].sum().item()),
            "index_order_convention": "torch.topk(..., sorted=True), descending selected logit",
            "routing_normalization_function": "softmax over selected logits only",
            "softmax_axis": 0,
            "weight_dtype": "torch.bfloat16",
            "serialization_dtype": "json_f32_values",
            "routing_weights_bf16_rounded_before_comparison": True,
        },
        "official_routing_metadata": {
            "selected_expert_indices": official_topk["selected_expert_indices"],
            "selected_expert_logits": official_topk["selected_expert_logits"],
            "routing_weights": official_topk["routing_weights"],
            "routing_weight_semantics": official_topk.get("routing_weight_semantics"),
            "tie_breaking_behavior": official_topk.get("tie_breaking_behavior"),
        },
        "selected_expert_index_comparison": index_cmp,
        "selected_expert_logit_comparison": logit_cmp,
        "routing_weight_comparison": {
            **weight_cmp,
            "local_weight_sum": float(local_result["routing_weights"].sum().item()),
            "official_weight_sum": float(official_weights.sum().item()),
            "weight_sum_abs_diff": float(abs(local_result["routing_weights"].sum().item() - official_weights.sum().item())),
        },
        "policy_discriminator_table": policy_table,
        "focused_mismatch_trace": focused_trace,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
        "local_residual_input_artifact_model": local_residual_input_artifact.get("provenance", {}).get("model"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
