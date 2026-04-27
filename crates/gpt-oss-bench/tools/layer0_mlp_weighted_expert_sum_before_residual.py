#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_mlp_expert30_mlp2_bias_to_selected_output as bias_status
import layer0_mlp_router_logits_before_routing_after_norm as router


HIDDEN_SIZE = 2880
EXPERT_RANK = 1
EXPERT_INDEX = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token MLP weighted expert sum before residual status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-selected-output-fix", type=Path, required=True)
    parser.add_argument("--source-topk-routing", type=Path, required=True)
    parser.add_argument("--official-weighted-sum", type=Path, required=True)
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


def compare_vector(lhs, rhs):
    lhs = lhs.reshape(-1).to(torch.float32)
    rhs = rhs.reshape(-1).to(torch.float32)
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
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_hidden_lane": first,
        "worst_differing_hidden_lane": worst,
    }


def compare_rank(lhs, rhs, rank, expert_index):
    metric = compare_vector(lhs, rhs)
    metric.update({"rank": int(rank), "expert_index": int(expert_index)})
    return metric


def tensor_meta(tensor, layout, serialization_dtype="json_f32_values"):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
        "device": str(tensor.device),
    }


def reconstruct_corrected_selected_outputs(source_fix, source_topk, model_root, device):
    mlp = router.load_layer0_mlp(model_root, device)
    mlp_norm_ref_path = Path(
        ".live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json"
    )
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    norm_input = torch.tensor(
        mlp_norm_ref["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    norm_input_bf16 = norm_input.to(torch.bfloat16).contiguous()
    selected_indices = [
        int(v) for v in source_topk["selected_expert_index_comparison"]["local_indices"]
    ]
    with torch.inference_mode():
        legacy_stages = bias_status.compute_batched_selected_outputs(
            mlp, norm_input_bf16, selected_indices
        )
    corrected = legacy_stages["final_unweighted"].to(torch.float32).contiguous()

    official_mlp2_pre_bias_path = Path(
        source_fix["source_artifact_paths"]["bias_to_selected_output"]
    )
    source_bias = load_json(official_mlp2_pre_bias_path)
    official_pre_bias_path = Path(
        source_bias["source_artifact_paths"]["official_mlp2_before_bias_reference"]
    )
    official_pre_bias = load_json(official_pre_bias_path)
    pre_bias = torch.tensor(
        official_pre_bias["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    bias = mlp.mlp2_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    reconstructed_rank1 = (
        pre_bias.to(torch.bfloat16) + bias
    ).to(torch.bfloat16).to(torch.float32).contiguous()
    corrected[EXPERT_RANK] = reconstructed_rank1
    return corrected, selected_indices


def bf16_rank_order_weighted_sum(selected_outputs, weights):
    selected_bf16 = selected_outputs.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)
    acc = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device=selected_outputs.device)
    for rank in range(selected_bf16.shape[0]):
        product = (selected_bf16[rank] * weights_bf16[rank]).to(torch.bfloat16)
        acc = (acc + product).to(torch.bfloat16)
    return acc.to(torch.float32)


def f32_accum_bf16_operands(selected_outputs, weights):
    selected_bf16 = selected_outputs.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)
    acc = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=selected_outputs.device)
    for rank in range(selected_bf16.shape[0]):
        acc = acc + selected_bf16[rank].to(torch.float32) * weights_bf16[rank].to(torch.float32)
    return acc.to(torch.bfloat16).to(torch.float32)


def f32_expanded_accum(selected_outputs, weights):
    selected_f32 = selected_outputs.to(torch.float32)
    weights_f32 = weights.to(torch.float32)
    acc = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=selected_outputs.device)
    for rank in range(selected_f32.shape[0]):
        acc = acc + selected_f32[rank] * weights_f32[rank]
    return acc.to(torch.bfloat16).to(torch.float32)


def torch_einsum_bf16(selected_outputs, weights):
    selected = selected_outputs.to(torch.bfloat16).reshape(1, 4, HIDDEN_SIZE)
    weights = weights.to(torch.bfloat16).reshape(1, 4)
    return torch.einsum("bec,be->bc", selected, weights).reshape(HIDDEN_SIZE).to(torch.bfloat16).to(torch.float32)


def variant_entry(name, tensor, official):
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official),
        "sha256_f32_le": sha256_f32_le(tensor),
    }


def main():
    args = parse_args()
    source_fix = load_json(args.source_selected_output_fix)
    source_topk = load_json(args.source_topk_routing)
    official_artifact = load_json(args.official_weighted_sum)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_fix.get("classification") != "selected_expert_output_capture_readout_fix_proven":
        raise ValueError("source selected expert output fix artifact is not proven")
    if source_topk.get("classification") != "topk_routing_weights_cleared_after_router_logits":
        raise ValueError("source top-k/routing artifact is not cleared")
    if official_artifact.get("classification") != "official_layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual_captured":
        raise ValueError("official weighted sum artifact is not the expected PPP capture")
    if official_artifact.get("boundary") != "layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual":
        raise ValueError("official weighted sum boundary is not usable")

    device = torch.device(args.device)
    model_root = args.model_root
    selected_outputs, selected_indices = reconstruct_corrected_selected_outputs(
        source_fix, source_topk, model_root, device
    )
    official_selected_guard_shape = [4, HIDDEN_SIZE]
    official_output = torch.tensor(
        official_artifact["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    routing_weights = torch.tensor(
        source_topk["local_routing_metadata"]["routing_weights"],
        dtype=torch.float32,
        device=device,
    ).reshape(4)
    official_weights = torch.tensor(
        official_artifact["source_routing_weights"], dtype=torch.float32, device=device
    ).reshape(4)

    selected_output_guard = source_fix["corrected_all_rank_selected_output_metrics"]
    per_rank_guard = source_fix["corrected_per_rank_metrics"]
    routing_weight_metric = compare_vector(routing_weights, official_weights)
    selected_indices_guard = {
        "ordered_match": selected_indices == official_artifact["source_selected_expert_indices_order"],
        "local_indices": selected_indices,
        "official_indices": official_artifact["source_selected_expert_indices_order"],
    }
    guards_pass = (
        selected_indices_guard["ordered_match"]
        and source_topk["routing_weight_comparison"]["matched"]
        and routing_weight_metric["matched"]
        and selected_output_guard["matched"]
    )

    local_weighted = torch_einsum_bf16(selected_outputs, routing_weights)
    weighted_metric = compare_vector(local_weighted, official_output)
    discriminator_table = []
    if not weighted_metric["matched"]:
        variants = [
            (
                "bf16_selected_outputs_bf16_weights_rank_order_bf16_output",
                bf16_rank_order_weighted_sum(selected_outputs, routing_weights),
            ),
            (
                "bf16_selected_outputs_bf16_weights_f32_accum_bf16_output",
                f32_accum_bf16_operands(selected_outputs, routing_weights),
            ),
            (
                "f32_expanded_selected_outputs_f32_weights_f32_accum_bf16_output",
                f32_expanded_accum(selected_outputs, routing_weights),
            ),
            ("torch_einsum_bf16_replay", local_weighted),
        ]
        discriminator_table = [variant_entry(name, tensor, official_output) for name, tensor in variants]

    focused_trace = None
    if not weighted_metric["matched"]:
        worst = weighted_metric["worst_differing_hidden_lane"]
        lane = int(worst["hidden_lane"])
        products = []
        for rank in range(4):
            products.append(
                {
                    "rank": rank,
                    "expert_index": selected_indices[rank],
                    "selected_output_value": float(selected_outputs[rank, lane].item()),
                    "routing_weight": float(routing_weights[rank].item()),
                    "f32_product": float(
                        selected_outputs[rank, lane].item() * routing_weights[rank].item()
                    ),
                    "bf16_product": float(
                        (
                            selected_outputs[rank, lane].to(torch.bfloat16)
                            * routing_weights[rank].to(torch.bfloat16)
                        ).to(torch.bfloat16).to(torch.float32).item()
                    ),
                }
            )
        focused_trace = {
            "hidden_lane": lane,
            "selected_expert_output_values": [
                float(selected_outputs[rank, lane].item()) for rank in range(4)
            ],
            "routing_weights": [float(v.item()) for v in routing_weights],
            "per_rank_products": products,
            "rank_accumulation_order": selected_indices,
            "local_weighted_sum_value": float(local_weighted[lane].item()),
            "official_weighted_sum_value": float(official_output[lane].item()),
            "discriminator_variant_values": {
                item["name"]: float(
                    {
                        "bf16_selected_outputs_bf16_weights_rank_order_bf16_output": bf16_rank_order_weighted_sum(selected_outputs, routing_weights),
                        "bf16_selected_outputs_bf16_weights_f32_accum_bf16_output": f32_accum_bf16_operands(selected_outputs, routing_weights),
                        "f32_expanded_selected_outputs_f32_weights_f32_accum_bf16_output": f32_expanded_accum(selected_outputs, routing_weights),
                        "torch_einsum_bf16_replay": local_weighted,
                    }[item["name"]][lane].item()
                )
                for item in discriminator_table
            },
            "local_minus_official": float(local_weighted[lane].item() - official_output[lane].item()),
        }

    if not guards_pass:
        classification = "weighted_expert_sum_blocked_by_selected_expert_or_routing_guard_regression"
        earliest = "selected expert output or routing guard"
        next_step = "re-establish selected expert outputs and routing weights before weighted sum"
    elif list(local_weighted.shape) != official_artifact["shape"]:
        classification = "weighted_expert_sum_shape_or_layout_mismatch"
        earliest = "layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual"
        next_step = "align weighted-sum output layout/readout only"
    elif weighted_metric["matched"]:
        classification = "weighted_expert_sum_before_residual_cleared"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_hidden_state_after_mlp_residual_add"
    else:
        classification = "weighted_expert_sum_dtype_or_accumulation_policy_mismatch"
        earliest = "layer0 final-token MLP routing weighted sum"
        next_step = "prove scoped weighted-sum dtype/accumulation policy before residual"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_weighted_expert_sum_before_residual_status/v1",
        "mode": "mlp-weighted-expert-sum-before-residual-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
        },
        "source_artifact_paths": {
            "selected_expert_output_fix": str(args.source_selected_output_fix),
            "topk_routing": str(args.source_topk_routing),
            "official_weighted_sum_reference": str(args.official_weighted_sum),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_weighted_sum),
        "selected_expert_routing_guard_metrics": {
            "selected_expert_indices": selected_indices_guard,
            "routing_weights": {
                "topk_artifact_metric": source_topk["routing_weight_comparison"],
                "local_vs_official_weight_values": routing_weight_metric,
                "local_weight_sum": float(routing_weights.sum().item()),
                "official_weight_sum": float(official_weights.sum().item()),
            },
        },
        "selected_expert_output_guard_metrics": {
            "all_rank_all_lane": selected_output_guard,
            "per_rank": per_rank_guard,
            "corrected_selected_output_shape": official_selected_guard_shape,
        },
        "local_weighted_sum_metadata": {
            "input_selected_output": tensor_meta(
                selected_outputs, "[top_k_rank, hidden_size]"
            ),
            "routing_weights": tensor_meta(routing_weights, "[top_k_rank]"),
            "output": tensor_meta(local_weighted, "[hidden_size] before second residual add"),
            "multiplication_dtype": "torch.bfloat16 via torch.einsum replay",
            "accumulation_dtype": "torch.einsum implementation-defined",
            "rank_accumulation_order": selected_indices,
            "output_cast_rounding_point": "BF16 before JSON f32 serialization",
            "before_second_residual_add": True,
        },
        "weighted_sum_comparison_metrics": weighted_metric,
        "discriminator_table": discriminator_table,
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
