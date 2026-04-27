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
        description="Layer0 final-token expert30 mlp2 bias to selected output status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
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


def tensor_summary(tensor):
    flat = tensor.reshape(-1).to(torch.float32)
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "sha256_f32_le": sha256_f32_le(flat),
    }


def variant_entry(name, tensor, official, local_capture):
    tensor = tensor.reshape(-1).to(torch.float32)
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official, "hidden_lane"),
        "metrics_vs_local_selected_capture": compare_named(
            tensor, local_capture, "hidden_lane", "variant", "local_capture"
        ),
        "digest": sha256_f32_le(tensor),
    }


def compute_batched_selected_outputs(mlp, norm_input_bf16, selected_indices):
    from gpt_oss.torch.model import swiglu

    expert_indices = torch.tensor(
        selected_indices, dtype=torch.long, device=norm_input_bf16.device
    ).reshape(1, -1)
    t = norm_input_bf16.reshape(1, -1)
    mlp1_weight = mlp.mlp1_weight[expert_indices, ...]
    mlp1_bias = mlp.mlp1_bias[expert_indices, ...]
    mlp1 = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    swiglu_out = swiglu(mlp1, limit=mlp.swiglu_limit)
    mlp2_weight = mlp.mlp2_weight[expert_indices, ...]
    mlp2_bias = mlp.mlp2_bias[expert_indices, ...]
    mlp2_pre_bias = torch.einsum("beck,bek->bec", mlp2_weight, swiglu_out)
    final = (mlp2_pre_bias + mlp2_bias).reshape(len(selected_indices), HIDDEN_SIZE).contiguous()
    return {
        "mlp2_pre_bias": mlp2_pre_bias.reshape(len(selected_indices), HIDDEN_SIZE).contiguous(),
        "final_unweighted": final,
    }


def main():
    args = parse_args()
    source_mlp2 = load_json(args.source_expert30_mlp2)
    source_selected = load_json(args.source_selected_expert_outputs)
    official_selected_artifact = load_json(args.official_selected_expert_outputs)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_mlp2.get("classification") != "expert30_mlp2_before_bias_cleared":
        raise ValueError("source expert30 mlp2-before-bias artifact is not cleared")
    if source_selected.get("classification") != "selected_expert_outputs_mismatch_before_routing_weighted_sum":
        raise ValueError("source selected expert output artifact is not the expected mismatch state")
    if official_selected_artifact.get("classification") != "official_layer0_final_token_selected_expert_outputs_before_routing_weighted_sum_captured":
        raise ValueError("official selected expert outputs artifact is not the expected PPP capture")
    if official_selected_artifact.get("boundary") != "layer0_final_token_selected_expert_outputs_before_routing_weighted_sum":
        raise ValueError("official selected expert outputs boundary is not usable")

    device = torch.device(args.device)
    official_mlp2_pre_bias_path = Path(
        source_mlp2["source_artifact_paths"]["official_expert30_mlp2_before_bias_reference"]
    )
    official_mlp2_pre_bias_artifact = load_json(official_mlp2_pre_bias_path)
    pre_bias = torch.tensor(
        official_mlp2_pre_bias_artifact["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    pre_bias_bf16 = pre_bias.to(torch.bfloat16).contiguous()
    official_selected = torch.tensor(
        official_selected_artifact["values"], dtype=torch.float32, device=device
    ).reshape(4, HIDDEN_SIZE)
    official_rank = official_selected[EXPERT_RANK].contiguous()

    mlp = router.load_layer0_mlp(args.model_root, device)
    bias = mlp.mlp2_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    bias_f32 = bias.to(torch.float32)

    with torch.inference_mode():
        bf16_add = (pre_bias_bf16 + bias).to(torch.bfloat16).contiguous()
        f32_add_bf16_output = (pre_bias_bf16.to(torch.float32) + bias.to(torch.float32)).to(torch.bfloat16).contiguous()
        f32_expanded_add = (pre_bias.to(torch.float32) + bias.to(torch.float32)).to(torch.bfloat16).contiguous()

    source_swiglu_path = Path(source_mlp2["source_artifact_paths"]["expert30_swiglu_cleared"])
    source_swiglu = load_json(source_swiglu_path)
    mlp_norm_ref_path = Path(source_swiglu["source_artifact_paths"]["official_expert30_mlp1_reference"]).parent / "developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json"
    if not mlp_norm_ref_path.exists():
        mlp_norm_ref_path = Path(".live/pinned-prompt-parity-official-reference-20260424/developer-message.ppp-layer0-final-token-mlp-norm-output-before-mlp-projections-status.json")
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    norm_input = torch.tensor(mlp_norm_ref["values"], dtype=torch.float32, device=device).reshape(HIDDEN_SIZE)
    norm_input_bf16 = norm_input.to(torch.bfloat16).contiguous()
    selected_indices = [int(v) for v in source_mlp2["routing_guard_metrics"]["selected_expert_indices"]["local_indices"]]
    batched = compute_batched_selected_outputs(mlp, norm_input_bf16, selected_indices)
    local_capture = batched["final_unweighted"][EXPERT_RANK].to(torch.float32).contiguous()
    local_batched_pre_bias = batched["mlp2_pre_bias"][EXPERT_RANK].to(torch.float32).contiguous()

    bf16_metric = compare_vector(bf16_add, official_rank, "hidden_lane")
    f32_bf16_metric = compare_vector(f32_add_bf16_output, official_rank, "hidden_lane")
    f32_expanded_metric = compare_vector(f32_expanded_add, official_rank, "hidden_lane")
    local_vs_official = compare_vector(local_capture, official_rank, "hidden_lane")
    local_vs_bf16 = compare_named(local_capture, bf16_add, "hidden_lane", "local_capture", "bf16_reconstruction")
    local_vs_f32_bf16 = compare_named(local_capture, f32_add_bf16_output, "hidden_lane", "local_capture", "f32_add_reconstruction")
    weighted_probe = local_capture * float(source_mlp2["routing_guard_metrics"]["routing_weights"]["local_weight_sum"])
    weighted_probe_metric = compare_vector(weighted_probe, official_rank, "hidden_lane")

    bias_guard = {
        "digest_matches_official": source_mlp2["expert30_mlp2_bias_metadata"]["digest_matches_official"],
        "sha256_f32_le": sha256_f32_le(bias),
        "official_sha256_f32_le": source_mlp2["expert30_mlp2_bias_metadata"]["official_metadata"].get("sha256_f32_le"),
        "metrics": compare_named(bias, bias, "hidden_lane", "local_bias", "official_bias"),
        "nonzero_count": int(torch.count_nonzero(bias_f32).item()),
    }
    mlp2_guard = source_mlp2["local_mlp2_before_bias_vs_official_metrics"]
    routing_context = {
        "selected_expert_indices": source_mlp2["routing_guard_metrics"]["selected_expert_indices"],
        "expert30_selected_rank_1": source_mlp2["routing_guard_metrics"]["expert30_selected_rank_1"],
        "routing_weights": source_mlp2["routing_guard_metrics"]["routing_weights"],
        "selected_expert_outputs_are_unweighted": official_selected_artifact.get("outputs_are_unweighted"),
    }
    guards_pass = (
        mlp2_guard["matched"]
        and bias_guard["digest_matches_official"]
        and bias_guard["metrics"]["matched"]
        and routing_context["selected_expert_indices"]["ordered_match"]
        and routing_context["expert30_selected_rank_1"]
        and routing_context["routing_weights"]["matched"]
    )

    reconstruction_table = [
        variant_entry("bf16_pre_bias_plus_bf16_bias_bf16_add_output", bf16_add, official_rank, local_capture),
        variant_entry("bf16_pre_bias_plus_bf16_bias_f32_add_then_bf16_output", f32_add_bf16_output, official_rank, local_capture),
        variant_entry("f32_expanded_pre_bias_plus_f32_expanded_bias_f32_add_then_bf16_output", f32_expanded_add, official_rank, local_capture),
        variant_entry("current_legacy_batched_selected_output_capture", local_capture, official_rank, local_capture),
    ]
    best_variant = min(
        reconstruction_table[:3],
        key=lambda item: (
            item["metrics_vs_official"]["max_abs_diff"],
            item["metrics_vs_official"]["mean_abs_diff"],
        ),
    )

    previous_rank_metric = source_selected["per_rank_metrics"][EXPERT_RANK]
    previous_first = previous_rank_metric.get("first_differing_hidden_lane")
    previous_worst = previous_rank_metric.get("worst_differing_hidden_lane")
    current_matches_previous_summary = (
        local_vs_official["max_abs_diff"] == previous_rank_metric["max_abs_diff"]
        and local_vs_official["mismatching_lane_count"] == previous_rank_metric["mismatching_lane_count"]
        and previous_first is not None
        and local_vs_official["first_differing_hidden_lane"] is not None
        and previous_first["hidden_lane"] == local_vs_official["first_differing_hidden_lane"]["hidden_lane"]
        and previous_worst is not None
        and local_vs_official["worst_differing_hidden_lane"] is not None
        and previous_worst["hidden_lane"] == local_vs_official["worst_differing_hidden_lane"]["hidden_lane"]
    )

    lane = KNOWN_LANE
    lane_trace = {
        "hidden_lane": lane,
        "official_mlp2_pre_bias": float(pre_bias[lane].item()),
        "local_mlp2_pre_bias": float(pre_bias[lane].item()),
        "bias_value": float(bias_f32[lane].item()),
        "bf16_add_output_value": float(bf16_add.to(torch.float32)[lane].item()),
        "f32_add_then_bf16_output_value": float(f32_add_bf16_output.to(torch.float32)[lane].item()),
        "local_selected_expert_output_current_capture": float(local_capture[lane].item()),
        "official_selected_expert_output": float(official_rank[lane].item()),
        "local_minus_official": float(local_capture[lane].item() - official_rank[lane].item()),
        "reconstructed_minus_official": float(bf16_add.to(torch.float32)[lane].item() - official_rank[lane].item()),
        "local_batched_pre_bias_capture": float(local_batched_pre_bias[lane].item()),
        "previous_local_selected_expert_output": previous_worst.get("local_value") if previous_worst and previous_worst.get("hidden_lane") == lane else None,
        "previous_official_selected_expert_output": previous_worst.get("official_value") if previous_worst and previous_worst.get("hidden_lane") == lane else None,
        "explained_by_bias_add_policy": bool(best_variant["metrics_vs_official"]["matched"] and local_vs_official["matched"]),
        "explained_by_capture_or_readout": bool(best_variant["metrics_vs_official"]["matched"] and not local_vs_official["matched"]),
    }

    local_capture_appears_weighted = weighted_probe_metric["matched"]
    if not guards_pass:
        classification = "expert30_selected_output_blocked_by_mlp2_or_bias_guard_regression"
        earliest = "expert30_mlp2_before_bias_or_mlp2_bias"
        next_step = "re-establish expert30 mlp2-before-bias and bias guards before selected output"
    elif local_capture_appears_weighted:
        classification = "expert30_selected_output_capture_is_after_routing_weight"
        earliest = "expert30 selected expert output capture point"
        next_step = "localize selected expert output capture/readout point only"
    elif best_variant["metrics_vs_official"]["matched"] and not local_vs_official["matched"]:
        classification = "expert30_selected_output_capture_or_readout_mismatch_after_bias_clear"
        earliest = "expert30 selected expert output capture/readout"
        next_step = "localize selected expert output capture/readout point only"
    elif best_variant["metrics_vs_official"]["matched"] and local_vs_official["matched"]:
        classification = "expert30_selected_output_cleared_after_mlp2_bias"
        earliest = "none"
        next_step = "rerun selected expert outputs for all ranks and then ask PPP for layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual"
    elif best_variant["name"] == "bf16_pre_bias_plus_bf16_bias_bf16_add_output" and best_variant["metrics_vs_official"]["matched"] and not f32_bf16_metric["matched"]:
        classification = "expert30_mlp2_bias_add_bf16_policy_identified"
        earliest = "expert30 mlp2 bias add dtype policy"
        next_step = "prove scoped expert30 mlp2 bias-add dtype policy before selected expert output"
    elif best_variant["name"] in (
        "bf16_pre_bias_plus_bf16_bias_f32_add_then_bf16_output",
        "f32_expanded_pre_bias_plus_f32_expanded_bias_f32_add_then_bf16_output",
    ) and best_variant["metrics_vs_official"]["matched"] and not bf16_metric["matched"]:
        classification = "expert30_mlp2_bias_add_f32_policy_identified"
        earliest = "expert30 mlp2 bias add dtype policy"
        next_step = "prove scoped expert30 mlp2 bias-add dtype policy before selected expert output"
    else:
        classification = "expert30_mlp2_bias_add_policy_still_unmodeled"
        earliest = "expert30 mlp2 bias add"
        next_step = "prove scoped expert30 mlp2 bias-add dtype policy before selected expert output"

    rank_summary = {
        "official_selected_output_digest": sha256_f32_le(official_rank),
        "best_reconstructed_selected_output_digest": best_variant["digest"],
        "local_selected_output_digest": sha256_f32_le(local_capture),
        "previous_mismatching_lane_count": previous_rank_metric["mismatching_lane_count"],
        "current_mismatching_lane_count": local_vs_official["mismatching_lane_count"],
        "previous_first_differing_lane": previous_first,
        "previous_worst_differing_lane": previous_worst,
        "current_first_differing_lane": local_vs_official["first_differing_hidden_lane"],
        "current_worst_differing_lane": local_vs_official["worst_differing_hidden_lane"],
        "previous_151_lanes_still_mismatch_in_current_run": current_matches_previous_summary,
    }

    output = {
        "schema_version": "runtime_forward_layer0_mlp_expert30_mlp2_bias_to_selected_output_status/v1",
        "mode": "mlp-expert30-mlp2-bias-to-selected-output-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_rank": EXPERT_RANK,
            "selected_expert_index": EXPERT_INDEX,
        },
        "source_artifact_paths": {
            "expert30_mlp2_before_bias_cleared": str(args.source_expert30_mlp2),
            "selected_expert_outputs_mismatch": str(args.source_selected_expert_outputs),
            "official_selected_expert_outputs_reference": str(args.official_selected_expert_outputs),
            "official_mlp2_before_bias_reference": str(official_mlp2_pre_bias_path),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_selected_expert_output_reference_path": str(args.official_selected_expert_outputs),
        "mlp2_before_bias_guard_metrics": mlp2_guard,
        "mlp2_bias_guard_metrics": bias_guard,
        "routing_context_guard": routing_context,
        "bias_add_reconstruction_table": reconstruction_table,
        "best_bias_add_reconstruction": best_variant,
        "local_selected_expert_output_capture_metrics": {
            "current_capture_vs_official": local_vs_official,
            "current_capture_vs_bf16_reconstruction": local_vs_bf16,
            "current_capture_vs_f32_add_reconstruction": local_vs_f32_bf16,
            "current_batched_mlp2_pre_bias_vs_official_pre_bias": compare_vector(
                local_batched_pre_bias, pre_bias, "hidden_lane"
            ),
            "capture_assessment": (
                "stale artifact / prior measurement issue or batched selected-output path differs from cleared single-expert path"
                if best_variant["metrics_vs_official"]["matched"] and local_vs_official["matched"]
                else "read from wrong buffer/layout or batched selected-output capture path"
                if best_variant["metrics_vs_official"]["matched"] and not local_vs_official["matched"]
                else "bias dtype/cast policy unresolved"
            ),
        },
        "lane_1156_trace": lane_trace,
        "rank_wide_summary": rank_summary,
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
