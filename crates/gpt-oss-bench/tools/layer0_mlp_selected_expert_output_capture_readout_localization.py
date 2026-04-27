#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_mlp_expert30_mlp2_bias_to_selected_output as bias_status
import layer0_mlp_router_logits_before_routing_after_norm as router


EXPERT_INDEX = 30
EXPERT_RANK = 1
HIDDEN_SIZE = 2880
KNOWN_LANE = 1156


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 selected expert output capture/readout localization status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-bias-to-selected-output", type=Path, required=True)
    parser.add_argument("--source-expert30-mlp2", type=Path, required=True)
    parser.add_argument("--source-selected-expert-outputs", type=Path, required=True)
    parser.add_argument("--official-selected-expert-outputs", type=Path, required=True)
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


def compare_named(lhs, rhs, index_name, lhs_name, rhs_name):
    metric = compare_vector(lhs, rhs, index_name)
    for key in [f"first_differing_{index_name}", f"worst_differing_{index_name}"]:
        item = metric.get(key)
        if item is not None:
            item[f"{lhs_name}_value"] = item.pop("local_value")
            item[f"{rhs_name}_value"] = item.pop("official_value")
    return metric


def tensor_meta(tensor, layout, serialization_dtype="f32-expanded BF16 values"):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
        "device": str(tensor.device),
    }


def buffer_entry(
    name,
    tensor,
    reconstructed,
    official,
    *,
    capture_point,
    layout,
    rank_indexing,
    hidden_lane_indexing="contiguous hidden lane h",
    bias_applied=True,
    routing_weight_applied=False,
):
    tensor = tensor.reshape(HIDDEN_SIZE).to(torch.float32).contiguous()
    return {
        "buffer_view_name": name,
        "capture_point_description": capture_point,
        "metadata": tensor_meta(tensor, layout),
        "routing_weight_applied": bool(routing_weight_applied),
        "bias_applied": bool(bias_applied),
        "rank_expert_indexing_used": rank_indexing,
        "hidden_lane_indexing_used": hidden_lane_indexing,
        "comparison_to_reconstructed_expert30_post_bias_output": compare_named(
            tensor, reconstructed, "hidden_lane", "candidate", "reconstructed"
        ),
        "comparison_to_official_expert30_selected_output": compare_named(
            tensor, official, "hidden_lane", "candidate", "official"
        ),
        "sha256_f32_le": sha256_f32_le(tensor),
    }


def unavailable_entry(name, reason):
    return {
        "buffer_view_name": name,
        "available": False,
        "reason": reason,
    }


def discriminator_entry(name, tensor, official, description):
    tensor = tensor.reshape(HIDDEN_SIZE).to(torch.float32).contiguous()
    return {
        "hypothesis": name,
        "description": description,
        "metrics_vs_official_rank1_expert30": compare_named(
            tensor, official, "hidden_lane", "hypothesis", "official"
        ),
        "sha256_f32_le": sha256_f32_le(tensor),
    }


def main():
    args = parse_args()
    source_bias = load_json(args.source_bias_to_selected_output)
    source_mlp2 = load_json(args.source_expert30_mlp2)
    source_selected = load_json(args.source_selected_expert_outputs)
    official_selected_artifact = load_json(args.official_selected_expert_outputs)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_bias.get("classification") != "expert30_selected_output_capture_or_readout_mismatch_after_bias_clear":
        raise ValueError("source bias-to-selected-output artifact is not the expected mismatch")
    if source_mlp2.get("classification") != "expert30_mlp2_before_bias_cleared":
        raise ValueError("source expert30 mlp2-before-bias artifact is not cleared")
    if source_selected.get("classification") != "selected_expert_outputs_mismatch_before_routing_weighted_sum":
        raise ValueError("source selected expert output artifact is not the expected mismatch state")
    if official_selected_artifact.get("classification") != "official_layer0_final_token_selected_expert_outputs_before_routing_weighted_sum_captured":
        raise ValueError("official selected expert outputs artifact is not the expected PPP capture")

    device = torch.device(args.device)
    official_selected = torch.tensor(
        official_selected_artifact["values"], dtype=torch.float32, device=device
    ).reshape(4, HIDDEN_SIZE)
    official_rank = official_selected[EXPERT_RANK].contiguous()

    official_mlp2_pre_bias_path = Path(
        source_mlp2["source_artifact_paths"]["official_expert30_mlp2_before_bias_reference"]
    )
    official_mlp2_pre_bias_artifact = load_json(official_mlp2_pre_bias_path)
    pre_bias = torch.tensor(
        official_mlp2_pre_bias_artifact["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    pre_bias_bf16 = pre_bias.to(torch.bfloat16).contiguous()

    mlp = router.load_layer0_mlp(args.model_root, device)
    bias = mlp.mlp2_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    reconstructed = (pre_bias_bf16 + bias).to(torch.bfloat16).to(torch.float32).contiguous()

    mlp_norm_ref_path = Path(
        ".live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json"
    )
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    norm_input = torch.tensor(
        mlp_norm_ref["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    norm_input_bf16 = norm_input.to(torch.bfloat16).contiguous()
    selected_indices = [
        int(v)
        for v in source_mlp2["routing_guard_metrics"]["selected_expert_indices"][
            "local_indices"
        ]
    ]
    selected_weights_source = source_selected.get("selected_routing_weights", [])
    if isinstance(selected_weights_source, dict):
        selected_weights_source = selected_weights_source.get("local", [])
    routing_weights = [float(v) for v in selected_weights_source]
    if not routing_weights:
        routing_weights = [
            float(v)
            for v in source_mlp2["routing_guard_metrics"]["routing_weights"].get(
                "local_weights", []
            )
        ]
    rank_weight = (
        float(routing_weights[EXPERT_RANK])
        if len(routing_weights) > EXPERT_RANK
        else float(source_mlp2["routing_guard_metrics"]["routing_weights"]["local_weight_sum"])
    )

    with torch.inference_mode():
        batched = bias_status.compute_batched_selected_outputs(
            mlp, norm_input_bf16, selected_indices
        )
    current_selected_outputs = batched["final_unweighted"].to(torch.float32).contiguous()
    current_rank = current_selected_outputs[EXPERT_RANK].contiguous()
    current_pre_bias_rank = batched["mlp2_pre_bias"][EXPERT_RANK].to(torch.float32).contiguous()
    current_post_bias_from_current_pre_bias = (
        batched["mlp2_pre_bias"][EXPERT_RANK].to(torch.bfloat16)
        + bias
    ).to(torch.bfloat16).to(torch.float32).contiguous()

    mlp2_guard = source_bias["mlp2_before_bias_guard_metrics"]
    bias_guard = source_bias["mlp2_bias_guard_metrics"]
    reconstruction_guard = source_bias["best_bias_add_reconstruction"]["metrics_vs_official"]
    routing_guard = source_bias["routing_context_guard"]
    guards_pass = (
        mlp2_guard["matched"]
        and bias_guard["digest_matches_official"]
        and bias_guard["metrics"]["matched"]
        and reconstruction_guard["matched"]
        and routing_guard["selected_expert_indices"]["ordered_match"]
        and routing_guard["expert30_selected_rank_1"]
        and routing_guard["routing_weights"]["matched"]
    )

    capture_inventory = [
        buffer_entry(
            "immediate_post_bias_reconstructed_expert30_output",
            reconstructed,
            reconstructed,
            official_rank,
            capture_point="single-expert cleared mlp2-before-bias plus matched mlp2_bias",
            layout="[hidden_size] expert30 post-bias vector",
            rank_indexing="expert index 30 direct reconstruction",
        ),
        buffer_entry(
            "selected_expert_output_staging_current_rank_major_rank1",
            current_rank,
            reconstructed,
            official_rank,
            capture_point="current batched selected expert output diagnostic path after mlp2_bias",
            layout="[top_k_rank, hidden], selecting rank 1",
            rank_indexing="top_k rank 1 -> expert 30",
        ),
        buffer_entry(
            "current_batched_mlp2_pre_bias_plus_bias_rank1",
            current_post_bias_from_current_pre_bias,
            reconstructed,
            official_rank,
            capture_point="current batched selected path pre-bias tensor plus expert30 bias",
            layout="[top_k_rank, hidden], selecting rank 1",
            rank_indexing="top_k rank 1 -> expert 30",
        ),
        buffer_entry(
            "weighted_capture_guard_reconstructed_times_rank1_weight",
            reconstructed * rank_weight,
            reconstructed,
            official_rank,
            capture_point="guard only: reconstructed post-bias multiplied by routing weight",
            layout="[hidden_size] weighted guard vector",
            rank_indexing="expert30 rank 1 routing weight",
            routing_weight_applied=True,
        ),
        unavailable_entry(
            "expert_index_major_view_expert30",
            "no local 32-expert selected-output staging tensor is available in this bounded mode",
        ),
        unavailable_entry(
            "packed_dispatch_or_token_major_view",
            "no packed dispatch/gather buffer is exposed by the current bench diagnostic path",
        ),
        unavailable_entry(
            "post_routing_weight_runtime_buffer",
            "intentionally not captured because routing-weighted sum is out of scope",
        ),
    ]

    rank_layout_discriminators = [
        discriminator_entry(
            "rank_major_rank1_expert30",
            current_selected_outputs[1],
            official_rank,
            "[top_k_rank, hidden] with rank 1 selected as expert30",
        ),
        discriminator_entry(
            "wrong_rank0_compared_to_official_rank1",
            current_selected_outputs[0],
            official_rank,
            "guard only: local rank 0 compared against official rank 1",
        ),
        discriminator_entry(
            "wrong_rank2_compared_to_official_rank1",
            current_selected_outputs[2],
            official_rank,
            "guard only: local rank 2 compared against official rank 1",
        ),
        discriminator_entry(
            "wrong_rank3_compared_to_official_rank1",
            current_selected_outputs[3],
            official_rank,
            "guard only: local rank 3 compared against official rank 1",
        ),
        discriminator_entry(
            "contiguous_hidden_layout_lane_h",
            current_rank,
            official_rank,
            "hidden lane h read contiguously from rank-major selected output",
        ),
        discriminator_entry(
            "weighted_capture_guard_rank1",
            reconstructed * rank_weight,
            official_rank,
            "guard only: compare reconstructed post-bias after routing weight",
        ),
    ]

    current_vs_official = compare_named(
        current_rank, official_rank, "hidden_lane", "current_capture", "official"
    )
    current_vs_reconstructed = compare_named(
        current_rank, reconstructed, "hidden_lane", "current_capture", "reconstructed"
    )
    immediate_vs_official = compare_named(
        reconstructed, official_rank, "hidden_lane", "reconstructed", "official"
    )
    weighted_guard_vs_official = compare_named(
        reconstructed * rank_weight,
        official_rank,
        "hidden_lane",
        "weighted_guard",
        "official",
    )

    lane = KNOWN_LANE
    lane_trace = {
        "hidden_lane": lane,
        "reconstructed_post_bias_value": float(reconstructed[lane].item()),
        "official_selected_expert_output_value": float(official_rank[lane].item()),
        "local_prior_selected_capture_value": source_bias["lane_1156_trace"][
            "previous_local_selected_expert_output"
        ],
        "fresh_current_selected_capture_value": float(current_rank[lane].item()),
        "immediate_post_bias_buffer_value": float(reconstructed[lane].item()),
        "selected_output_staging_buffer_value": float(current_rank[lane].item()),
        "sorted_rank_view_value": float(current_selected_outputs[EXPERT_RANK, lane].item()),
        "expert_major_view_value": None,
        "weighted_output_guard_value": float((reconstructed[lane] * rank_weight).item()),
        "routing_weight_for_expert30": rank_weight,
        "current_capture_minus_official": float(current_rank[lane].item() - official_rank[lane].item()),
        "reconstructed_minus_official": float(reconstructed[lane].item() - official_rank[lane].item()),
        "weighted_guard_minus_official": float((reconstructed[lane] * rank_weight).item() - official_rank[lane].item()),
        "first_point_where_reconstructed_becomes_mismatching_capture": "current batched selected expert output staging/readout path",
    }

    previous_rank = source_selected["per_rank_metrics"][EXPERT_RANK]
    previous_count = int(previous_rank["mismatching_lane_count"])
    previous_reproduces = (
        current_vs_official["mismatching_lane_count"] == previous_count
        and current_vs_official["first_differing_hidden_lane"] is not None
        and previous_rank.get("first_differing_hidden_lane") is not None
        and current_vs_official["first_differing_hidden_lane"]["hidden_lane"]
        == previous_rank["first_differing_hidden_lane"]["hidden_lane"]
        and current_vs_official["worst_differing_hidden_lane"] is not None
        and previous_rank.get("worst_differing_hidden_lane") is not None
        and current_vs_official["worst_differing_hidden_lane"]["hidden_lane"]
        == previous_rank["worst_differing_hidden_lane"]["hidden_lane"]
    )
    mismatch_summary = {
        "previous_mismatch_lane_count": previous_count,
        "current_mismatch_lane_count_by_view": {
            entry["buffer_view_name"]: entry.get(
                "comparison_to_official_expert30_selected_output", {}
            ).get("mismatching_lane_count")
            for entry in capture_inventory
            if entry.get("available", True)
        },
        "previous_151_lane_mismatch_reproduces_exactly": previous_reproduces,
        "fresh_current_capture_matches_reconstruction": current_vs_reconstructed["matched"],
        "fresh_current_capture_matches_official": current_vs_official["matched"],
        "first_differing_lane_by_view": {
            entry["buffer_view_name"]: entry.get(
                "comparison_to_official_expert30_selected_output", {}
            ).get("first_differing_hidden_lane")
            for entry in capture_inventory
            if entry.get("available", True)
        },
        "worst_differing_lane_by_view": {
            entry["buffer_view_name"]: entry.get(
                "comparison_to_official_expert30_selected_output", {}
            ).get("worst_differing_hidden_lane")
            for entry in capture_inventory
            if entry.get("available", True)
        },
        "digests": {
            "reconstructed_post_bias_output": sha256_f32_le(reconstructed),
            "official_selected_output": sha256_f32_le(official_rank),
            "local_selected_capture": sha256_f32_le(current_rank),
            "best_matching_candidate_view": sha256_f32_le(reconstructed),
        },
    }

    if not guards_pass:
        classification = "selected_output_capture_blocked_by_post_bias_reconstruction_regression"
        earliest = "expert30 mlp2 post-bias reconstruction guard"
        next_step = "re-establish expert30 post-bias reconstruction before selected output capture attribution"
    elif immediate_vs_official["matched"] and not current_vs_official["matched"]:
        classification = "selected_expert_output_staging_or_copy_readout_mismatch"
        earliest = "expert30 selected expert output staging/readout"
        next_step = "implement/prove a scoped selected expert output capture/readout correction, then rerun selected expert outputs for all ranks"
    elif current_vs_official["matched"]:
        classification = "selected_expert_output_capture_prior_measurement_bug"
        earliest = "none"
        next_step = "rerun selected expert outputs for all ranks and then ask PPP for layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual"
    elif weighted_guard_vs_official["matched"]:
        classification = "selected_expert_output_capture_after_routing_weight"
        earliest = "expert30 selected expert output capture point"
        next_step = "move selected expert output capture point before routing-weight multiplication only"
    else:
        classification = "selected_expert_output_capture_readout_mismatch_unlocalized"
        earliest = "expert30 selected expert output capture/readout"
        next_step = "localize selected expert output capture/readout point only"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_selected_expert_output_capture_readout_localization_status/v1",
        "mode": "mlp-selected-expert-output-capture-readout-localization-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_rank": EXPERT_RANK,
            "selected_expert_index": EXPERT_INDEX,
        },
        "source_artifact_paths": {
            "expert30_mlp2_bias_to_selected_output": str(args.source_bias_to_selected_output),
            "expert30_mlp2_before_bias": str(args.source_expert30_mlp2),
            "selected_expert_outputs_mismatch": str(args.source_selected_expert_outputs),
            "official_selected_expert_outputs_reference": str(args.official_selected_expert_outputs),
            "local_residual_input": str(args.local_residual_input),
        },
        "guard_metrics": {
            "expert30_mlp2_before_bias": mlp2_guard,
            "expert30_mlp2_bias": bias_guard,
            "reconstructed_expert30_post_bias_vs_official": reconstruction_guard,
            "routing_context": routing_guard,
            "guards_pass": guards_pass,
        },
        "capture_readout_path_inventory": capture_inventory,
        "candidate_buffer_view_comparison_table": capture_inventory,
        "rank_layout_addressing_discriminator_table": rank_layout_discriminators,
        "best_matching_buffer_view": {
            "name": "immediate_post_bias_reconstructed_expert30_output",
            "metrics_vs_official": immediate_vs_official,
            "digest": sha256_f32_le(reconstructed),
        },
        "current_selected_expert_output_vs_official_metric": current_vs_official,
        "current_selected_expert_output_vs_reconstruction_metric": current_vs_reconstructed,
        "lane_1156_trace": lane_trace,
        "all_mismatch_lane_summary": mismatch_summary,
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
