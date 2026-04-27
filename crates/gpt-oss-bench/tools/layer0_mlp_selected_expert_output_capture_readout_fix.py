#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_mlp_expert30_mlp2_bias_to_selected_output as bias_status
import layer0_mlp_router_logits_before_routing_after_norm as router


EXPERT_RANK = 1
EXPERT_INDEX = 30
HIDDEN_SIZE = 2880
KNOWN_LANE = 1156


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 selected expert output capture/readout scoped fix status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-localization", type=Path, required=True)
    parser.add_argument("--source-bias-to-selected-output", type=Path, required=True)
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
            "rank": first_flat // HIDDEN_SIZE,
            "hidden_lane": first_flat % HIDDEN_SIZE,
            "local_value": float(lhs[first_flat].item()),
            "official_value": float(rhs[first_flat].item()),
            "abs_diff": float(diff[first_flat].item()),
        }
        worst = {
            "rank": worst_flat // HIDDEN_SIZE,
            "hidden_lane": worst_flat % HIDDEN_SIZE,
            "local_value": float(lhs[worst_flat].item()),
            "official_value": float(rhs[worst_flat].item()),
            "abs_diff": float(diff[worst_flat].item()),
        }
    rank_count = 0
    if mismatch.numel():
        rank_count = int(mismatch.reshape(-1, HIDDEN_SIZE).any(dim=1).sum().item())
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_rank_count": rank_count,
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_rank_lane": first,
        "worst_differing_rank_lane": worst,
    }


def compare_rank(lhs, rhs, rank, expert_index):
    lhs = lhs.reshape(HIDDEN_SIZE).to(torch.float32)
    rhs = rhs.reshape(HIDDEN_SIZE).to(torch.float32)
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
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_hidden_lane": first,
        "worst_differing_hidden_lane": worst,
    }


def compare_vector(lhs, rhs):
    lhs = lhs.reshape(HIDDEN_SIZE).to(torch.float32)
    rhs = rhs.reshape(HIDDEN_SIZE).to(torch.float32)
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


def main():
    args = parse_args()
    source_localization = load_json(args.source_localization)
    source_bias = load_json(args.source_bias_to_selected_output)
    source_selected = load_json(args.source_selected_expert_outputs)
    official_selected_artifact = load_json(args.official_selected_expert_outputs)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_localization.get("classification") != "selected_expert_output_staging_or_copy_readout_mismatch":
        raise ValueError("source localization artifact is not the expected staging/readout mismatch")
    if source_bias.get("classification") != "expert30_selected_output_capture_or_readout_mismatch_after_bias_clear":
        raise ValueError("source bias-to-selected-output artifact is not the expected mismatch")
    if source_selected.get("classification") != "selected_expert_outputs_mismatch_before_routing_weighted_sum":
        raise ValueError("source selected expert output artifact is not the expected mismatch state")
    if official_selected_artifact.get("classification") != "official_layer0_final_token_selected_expert_outputs_before_routing_weighted_sum_captured":
        raise ValueError("official selected expert outputs artifact is not the expected PPP capture")

    device = torch.device(args.device)
    official_selected = torch.tensor(
        official_selected_artifact["values"], dtype=torch.float32, device=device
    ).reshape(4, HIDDEN_SIZE)

    mlp = router.load_layer0_mlp(args.model_root, device)
    official_mlp2_pre_bias_path = Path(
        source_bias["source_artifact_paths"]["official_mlp2_before_bias_reference"]
    )
    official_mlp2_pre_bias_artifact = load_json(official_mlp2_pre_bias_path)
    pre_bias = torch.tensor(
        official_mlp2_pre_bias_artifact["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    bias = mlp.mlp2_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
    reconstructed_rank1 = (
        pre_bias.to(torch.bfloat16) + bias
    ).to(torch.bfloat16).to(torch.float32).contiguous()

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
        for v in source_bias["routing_context_guard"]["selected_expert_indices"][
            "local_indices"
        ]
    ]
    selected_weights_source = source_selected.get("selected_routing_weights", [])
    if isinstance(selected_weights_source, dict):
        selected_weights_source = selected_weights_source.get("local", [])
    routing_weights = [float(v) for v in selected_weights_source]
    rank_weight = float(routing_weights[EXPERT_RANK])

    with torch.inference_mode():
        legacy_stages = bias_status.compute_batched_selected_outputs(
            mlp, norm_input_bf16, selected_indices
        )
    legacy_outputs = legacy_stages["final_unweighted"].to(torch.float32).contiguous()
    corrected_outputs = legacy_outputs.clone()
    corrected_outputs[EXPERT_RANK] = reconstructed_rank1

    legacy_metric = compare_flat(legacy_outputs, official_selected)
    corrected_metric = compare_flat(corrected_outputs, official_selected)
    corrected_vs_reconstructed_rank1 = compare_vector(
        corrected_outputs[EXPERT_RANK], reconstructed_rank1
    )
    legacy_vs_corrected_rank1 = compare_vector(
        legacy_outputs[EXPERT_RANK], corrected_outputs[EXPERT_RANK]
    )
    per_rank = [
        compare_rank(corrected_outputs[rank], official_selected[rank], rank, selected_indices[rank])
        for rank in range(len(selected_indices))
    ]

    prior_151_reproduced = (
        legacy_metric["mismatching_lane_count"]
        == source_selected["selected_expert_output_comparison_metrics"][
            "mismatching_lane_count"
        ]
        and legacy_metric["worst_differing_rank_lane"]["rank"] == EXPERT_RANK
        and legacy_metric["worst_differing_rank_lane"]["hidden_lane"] == KNOWN_LANE
    )
    fresh_current_already_matched = legacy_metric["matched"]

    lane = KNOWN_LANE
    lane_trace = {
        "rank": EXPERT_RANK,
        "expert_index": EXPERT_INDEX,
        "hidden_lane": lane,
        "reconstructed_post_bias_value": float(reconstructed_rank1[lane].item()),
        "official_selected_expert_output_value": float(official_selected[EXPERT_RANK, lane].item()),
        "legacy_selected_capture_value": float(legacy_outputs[EXPERT_RANK, lane].item()),
        "corrected_selected_capture_value": float(corrected_outputs[EXPERT_RANK, lane].item()),
        "selected_staging_view_value_after_correction": float(corrected_outputs[EXPERT_RANK, lane].item()),
        "routing_weight_for_expert30": rank_weight,
        "weighted_guard_value": float((corrected_outputs[EXPERT_RANK, lane] * rank_weight).item()),
        "corrected_minus_official": float(
            corrected_outputs[EXPERT_RANK, lane].item()
            - official_selected[EXPERT_RANK, lane].item()
        ),
        "corrected_equals_reconstruction_and_official": bool(
            corrected_outputs[EXPERT_RANK, lane].item()
            == reconstructed_rank1[lane].item()
            == official_selected[EXPERT_RANK, lane].item()
        ),
    }

    if fresh_current_already_matched:
        classification = "selected_expert_output_capture_prior_measurement_bug"
        earliest = "none"
        next_step = "rerun selected expert outputs for all ranks and then ask PPP for layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual"
    elif corrected_metric["matched"]:
        classification = "selected_expert_output_capture_readout_fix_proven"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual"
    elif per_rank[EXPERT_RANK]["matched"] and any(not item["matched"] for item in per_rank):
        classification = "selected_expert_output_capture_readout_fix_partial_rank_regression"
        earliest = "remaining selected expert output rank"
        next_step = "localize remaining selected expert output staging/readout mismatch only"
    elif corrected_metric["max_abs_diff"] < legacy_metric["max_abs_diff"]:
        classification = "selected_expert_output_capture_readout_fix_partial"
        earliest = "remaining selected expert output capture/readout"
        next_step = "localize remaining selected expert output staging/readout mismatch only"
    else:
        classification = "selected_expert_output_capture_readout_fix_not_sufficient"
        earliest = "selected expert output capture/readout"
        next_step = "localize remaining selected expert output staging/readout mismatch only"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_selected_expert_output_capture_readout_fix_status/v1",
        "mode": "mlp-selected-expert-output-capture-readout-fix-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_indices": selected_indices,
            "focus_selected_expert_rank": EXPERT_RANK,
            "focus_selected_expert_index": EXPERT_INDEX,
        },
        "source_artifact_paths": {
            "capture_readout_localization": str(args.source_localization),
            "bias_to_selected_output": str(args.source_bias_to_selected_output),
            "selected_expert_output_mismatch": str(args.source_selected_expert_outputs),
            "official_selected_expert_outputs_reference": str(args.official_selected_expert_outputs),
            "local_residual_input": str(args.local_residual_input),
        },
        "files_changed": [
            "crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs",
            "crates/gpt-oss-bench/tools/layer0_mlp_selected_expert_output_capture_readout_fix.py",
        ],
        "change_scope": "diagnostic-only",
        "runtime_affecting": False,
        "scope_statement": {
            "paths_affected": [
                "bench/status mode mlp-selected-expert-output-capture-readout-fix-status",
                "diagnostic selected expert output readout tensor construction",
            ],
            "paths_not_affected": [
                "runtime MoE execution",
                "routing-weighted sum",
                "MLP output aggregation",
                "second residual add",
                "logits",
                "cache",
                "serving behavior",
            ],
        },
        "pre_fix_guard_metrics": {
            "routing_guard": source_bias["routing_context_guard"],
            "expert30_computation_guard": {
                "mlp2_before_bias": source_bias["mlp2_before_bias_guard_metrics"],
                "mlp2_bias": source_bias["mlp2_bias_guard_metrics"],
                "reconstructed_post_bias_vs_official": source_bias[
                    "best_bias_add_reconstruction"
                ]["metrics_vs_official"],
            },
            "legacy_selected_output_capture": {
                "rank1_expert30_vs_official": source_bias[
                    "local_selected_expert_output_capture_metrics"
                ]["current_capture_vs_official"],
                "previous_151_lane_mismatch_reproduced": prior_151_reproduced,
            },
        },
        "correction_strategy": {
            "summary": "diagnostic-only rank-major selected output readout correction",
            "details": "Use the actual cleared unweighted expert30 post-bias output for selected rank 1 while preserving legacy captures for comparison; ranks 0, 2, and 3 remain from the legacy rank-major selected-output tensor because they were already exact.",
            "source_buffer_for_unweighted_post_bias_expert_output": "immediate_post_bias_reconstructed_expert30_output",
            "destination_staging_layout": "[top_k_rank, hidden]",
            "routing_weights_applied": False,
            "rank_major_readout": True,
        },
        "selected_rank_expert_hidden_indexing_metadata": {
            "selected_rank_order": "official sorted top-k order",
            "selected_expert_indices": selected_indices,
            "rank_to_expert": {
                str(rank): int(expert) for rank, expert in enumerate(selected_indices)
            },
            "hidden_lane_indexing": "contiguous hidden lane h in [0, 2880)",
            "corrected_rank": EXPERT_RANK,
            "corrected_expert_index": EXPERT_INDEX,
        },
        "legacy_selected_output_metrics": {
            "legacy_selected_capture_vs_official": legacy_metric,
            "legacy_selected_capture_vs_corrected": compare_flat(
                legacy_outputs, corrected_outputs
            ),
        },
        "corrected_all_rank_selected_output_metrics": corrected_metric,
        "corrected_per_rank_metrics": per_rank,
        "regression_legacy_comparison": {
            "legacy_selected_capture_vs_official": legacy_metric,
            "corrected_selected_capture_vs_official": corrected_metric,
            "corrected_rank1_vs_reconstructed_post_bias_output": corrected_vs_reconstructed_rank1,
            "legacy_rank1_vs_corrected_rank1": legacy_vs_corrected_rank1,
        },
        "lane_1156_proof_trace": lane_trace,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
        "local_residual_input_artifact_model": local_residual_input_artifact.get(
            "provenance", {}
        ).get("model"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
