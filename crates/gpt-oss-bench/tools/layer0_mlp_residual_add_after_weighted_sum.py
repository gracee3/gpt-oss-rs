#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch


HIDDEN_SIZE = 2880


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token MLP residual add after weighted expert sum status."
    )
    parser.add_argument("--source-weighted-sum", type=Path, required=True)
    parser.add_argument("--source-attention-residual-add", type=Path, required=True)
    parser.add_argument("--official-after-mlp-residual", type=Path, required=True)
    parser.add_argument("--official-post-attention-residual", type=Path, required=True)
    parser.add_argument("--official-mlp-weighted-sum", type=Path, required=True)
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


def tensor_meta(tensor, layout, serialization_dtype="json_f32_values"):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
        "device": str(tensor.device),
    }


def load_vector_values(artifact, path, expected_boundary):
    if artifact.get("boundary") != expected_boundary:
        raise ValueError(f"{path} is not boundary {expected_boundary}")
    values = artifact.get("values")
    if not isinstance(values, list) or len(values) != HIDDEN_SIZE:
        raise ValueError(f"{path} does not contain {HIDDEN_SIZE} values")
    return values


def bf16_residual_add(residual, mlp_output):
    return (
        residual.to(torch.bfloat16) + mlp_output.to(torch.bfloat16)
    ).to(torch.bfloat16).to(torch.float32).contiguous()


def f32_add_then_bf16(residual, mlp_output):
    return (
        residual.to(torch.bfloat16).to(torch.float32)
        + mlp_output.to(torch.bfloat16).to(torch.float32)
    ).to(torch.bfloat16).to(torch.float32).contiguous()


def f32_expanded_add_then_bf16(residual, mlp_output):
    return (
        residual.to(torch.float32) + mlp_output.to(torch.float32)
    ).to(torch.bfloat16).to(torch.float32).contiguous()


def variant_entry(name, tensor, official):
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official),
        "sha256_f32_le": sha256_f32_le(tensor),
    }


def main():
    args = parse_args()
    source_weighted_sum = load_json(args.source_weighted_sum)
    source_attention_residual = load_json(args.source_attention_residual_add)
    official_after_mlp = load_json(args.official_after_mlp_residual)
    official_post_attention = load_json(args.official_post_attention_residual)
    official_weighted_sum = load_json(args.official_mlp_weighted_sum)

    if source_weighted_sum.get("classification") != "weighted_expert_sum_before_residual_cleared":
        raise ValueError("source weighted MLP output artifact is not cleared")
    if source_attention_residual.get("classification") != "attention_residual_add_before_mlp_cleared_after_o_proj_candidates":
        raise ValueError("source post-attention residual artifact is not cleared")
    if official_after_mlp.get("classification") != "official_layer0_final_token_hidden_state_after_mlp_residual_add_captured":
        raise ValueError("official after-MLP residual artifact is not the expected PPP capture")

    post_attention_values = load_vector_values(
        official_post_attention,
        args.official_post_attention_residual,
        "layer0_final_token_hidden_state_after_attention_residual_add_before_mlp",
    )
    weighted_sum_values = load_vector_values(
        official_weighted_sum,
        args.official_mlp_weighted_sum,
        "layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual",
    )
    after_mlp_values = load_vector_values(
        official_after_mlp,
        args.official_after_mlp_residual,
        "layer0_final_token_hidden_state_after_mlp_residual_add",
    )

    device = torch.device(args.device)
    post_attention = torch.tensor(
        post_attention_values, dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    weighted_sum = torch.tensor(
        weighted_sum_values, dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    official_after = torch.tensor(
        after_mlp_values, dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)

    post_attention_guard = source_attention_residual["residual_add_comparison_metrics"]
    weighted_sum_guard = source_weighted_sum["weighted_sum_comparison_metrics"]
    guards_pass = post_attention_guard["matched"] and weighted_sum_guard["matched"]

    reconstructed = bf16_residual_add(post_attention, weighted_sum)
    residual_metric = compare_vector(reconstructed, official_after)
    discriminator_table = []
    if not residual_metric["matched"]:
        variants = [
            (
                "bf16_post_attention_residual_plus_bf16_mlp_output_bf16_add_output",
                reconstructed,
            ),
            (
                "bf16_inputs_f32_add_then_bf16_output",
                f32_add_then_bf16(post_attention, weighted_sum),
            ),
            (
                "f32_expanded_inputs_f32_add_then_bf16_output",
                f32_expanded_add_then_bf16(post_attention, weighted_sum),
            ),
        ]
        discriminator_table = [
            variant_entry(name, tensor, official_after) for name, tensor in variants
        ]

    focused_trace = None
    if not residual_metric["matched"]:
        lane = int(residual_metric["worst_differing_hidden_lane"]["hidden_lane"])
        focused_trace = {
            "hidden_lane": lane,
            "post_attention_residual_value": float(post_attention[lane].item()),
            "mlp_weighted_expert_sum_value": float(weighted_sum[lane].item()),
            "reconstructed_add_value": float(reconstructed[lane].item()),
            "official_after_mlp_residual_value": float(official_after[lane].item()),
            "local_runtime_after_mlp_residual_value": None,
            "local_minus_official": float(
                reconstructed[lane].item() - official_after[lane].item()
            ),
            "mismatch_attribution": {
                "residual_input_regression": not post_attention_guard["matched"],
                "mlp_output_regression": not weighted_sum_guard["matched"],
                "add_dtype_or_rounding_policy": guards_pass,
                "layout_or_capture": False,
            },
        }

    if not guards_pass:
        classification = "mlp_residual_add_blocked_by_input_guard_regression"
        earliest = "post-attention residual or MLP weighted sum input guard"
        next_step = "revalidate the regressed input seam only"
    elif list(reconstructed.shape) != official_after_mlp["shape"]:
        classification = "mlp_residual_add_shape_or_layout_mismatch"
        earliest = "layer0_final_token_hidden_state_after_mlp_residual_add"
        next_step = "align after-MLP residual output layout/readout only"
    elif residual_metric["matched"]:
        classification = "layer0_hidden_after_mlp_residual_add_cleared"
        earliest = "none"
        next_step = "ask PPP for exactly layer1_final_token_attention_norm_output_before_qkv"
    else:
        classification = "mlp_residual_add_dtype_policy_mismatch"
        earliest = "layer0 final-token MLP residual add"
        next_step = "prove scoped BF16 MLP residual-add dtype policy before layer1"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_residual_add_after_weighted_sum_status/v1",
        "mode": "mlp-residual-add-after-weighted-sum-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
        },
        "source_artifact_paths": {
            "mlp_weighted_expert_sum_cleared": str(args.source_weighted_sum),
            "attention_residual_add_cleared": str(args.source_attention_residual_add),
            "official_after_mlp_residual_reference": str(args.official_after_mlp_residual),
            "official_post_attention_residual_reference": str(args.official_post_attention_residual),
            "official_mlp_weighted_sum_reference": str(args.official_mlp_weighted_sum),
        },
        "official_ppp_reference_path": str(args.official_after_mlp_residual),
        "official_tensor_metadata": {
            "shape": official_after_mlp.get("shape"),
            "layout": official_after_mlp.get("layout")
            or official_after_mlp.get("layout_interpretation"),
            "tensor_dtype": official_after_mlp.get("tensor_dtype"),
            "serialization_dtype": official_after_mlp.get("serialization_dtype"),
            "boundary": official_after_mlp.get("boundary"),
            "residual_add_semantics": official_after_mlp.get("residual_add_semantics"),
        },
        "post_attention_residual_guard_metrics": {
            "metrics": post_attention_guard,
            "source_artifact": str(args.source_attention_residual_add),
            "source_boundary_used": "layer0_final_token_hidden_state_after_attention_residual_add_before_mlp",
        },
        "mlp_weighted_sum_guard_metrics": {
            "metrics": weighted_sum_guard,
            "source_artifact": str(args.source_weighted_sum),
            "source_boundary_used": "layer0_final_token_mlp_output_after_routing_weighted_sum_before_residual",
        },
        "local_reconstructed_after_mlp_residual_metadata": {
            "post_attention_residual_input": tensor_meta(post_attention, "[hidden_size]"),
            "mlp_weighted_expert_sum_input": tensor_meta(weighted_sum, "[hidden_size]"),
            "output": tensor_meta(
                reconstructed, "[hidden_size] after MLP residual add before layer1"
            ),
            "addend_order": "post_attention_residual + mlp_weighted_expert_sum",
            "computation_dtype": "torch.bfloat16",
            "output_dtype_before_serialization": "torch.bfloat16",
            "rounded_to_bf16_after_add": True,
            "before_layer1": True,
        },
        "after_mlp_residual_comparison_metrics": residual_metric,
        "dtype_add_policy_finding": (
            "not needed; BF16 residual add/output matches official"
            if residual_metric["matched"]
            else "bounded discriminator required because reconstructed residual add mismatches official"
        ),
        "dtype_add_discriminator_table": discriminator_table,
        "focused_mismatch_trace": focused_trace,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
