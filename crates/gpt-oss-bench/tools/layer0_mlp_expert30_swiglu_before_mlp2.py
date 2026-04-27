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
SIGMOID_SCALE = 1.702
SWIGLU_LIMIT = 7.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token expert30 SwiGLU output before mlp2 status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-expert30-mlp1", type=Path, required=True)
    parser.add_argument("--source-selected-expert-outputs", type=Path, required=True)
    parser.add_argument("--source-topk-routing", type=Path, required=True)
    parser.add_argument("--official-expert30-swiglu", type=Path, required=True)
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


def swiglu_formula(fused, gate_clamp=True, up_clamp=True, sigmoid_scale=SIGMOID_SCALE, bf16_output=True):
    fused = fused.reshape(FUSED_SIZE)
    gate = fused[0::2]
    up = fused[1::2]
    if gate_clamp:
        gate = gate.clamp(max=SWIGLU_LIMIT)
    if up_clamp:
        up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    out = gate * torch.sigmoid(sigmoid_scale * gate) * (up + 1)
    if bf16_output:
        out = out.to(torch.bfloat16)
    return out.reshape(HIDDEN_SIZE).contiguous()


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


def vector_summary(tensor):
    flat = tensor.reshape(-1).to(torch.float32)
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "sha256_f32_le": sha256_f32_le(flat),
    }


def bf16_bits_from_float(value):
    packed = struct.pack("<f", float(value))
    bits = struct.unpack("<I", packed)[0]
    return f"0x{bits >> 16:04x}"


def main():
    args = parse_args()
    source_mlp1 = load_json(args.source_expert30_mlp1)
    source_selected = load_json(args.source_selected_expert_outputs)
    source_topk = load_json(args.source_topk_routing)
    official_artifact = load_json(args.official_expert30_swiglu)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_mlp1.get("classification") != "expert30_mlp1_before_swiglu_cleared":
        raise ValueError("source expert30 mlp1 artifact is not cleared")
    if source_selected.get("classification") != "selected_expert_outputs_mismatch_before_routing_weighted_sum":
        raise ValueError("source selected expert outputs artifact is not the expected mismatch state")
    if source_topk.get("classification") != "topk_routing_weights_cleared_after_router_logits":
        raise ValueError("source top-k/routing artifact is not cleared")
    if official_artifact.get("classification") != "official_layer0_final_token_expert30_swiglu_output_before_mlp2_captured":
        raise ValueError("official expert30 SwiGLU artifact is not the expected PPP capture")
    if official_artifact.get("boundary") != "layer0_final_token_expert30_swiglu_output_before_mlp2":
        raise ValueError("official expert30 SwiGLU boundary is not usable")

    device = torch.device(args.device)
    mlp_norm_ref_path = Path(source_mlp1["source_artifact_paths"]["official_mlp_norm_input"])
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    mlp1_ref_path = Path(source_mlp1["official_ppp_reference_path"])
    mlp1_ref = load_json(mlp1_ref_path)
    input_norm = torch.tensor(mlp_norm_ref["values"], dtype=torch.float32, device=device).reshape(HIDDEN_SIZE)
    input_bf16 = input_norm.to(torch.bfloat16).contiguous()
    official_mlp1 = torch.tensor(mlp1_ref["values"], dtype=torch.float32, device=device).reshape(FUSED_SIZE)
    official_mlp1_bf16 = official_mlp1.to(torch.bfloat16).contiguous()
    official_swiglu = torch.tensor(official_artifact["values"], dtype=torch.float32, device=device).reshape(HIDDEN_SIZE)

    selected_indices = [int(v) for v in source_topk["selected_expert_index_comparison"]["local_indices"]]
    routing_guard = {
        "router_logits": source_topk["router_logit_guard_metrics"]["source_artifact_metric"],
        "selected_expert_indices": source_topk["selected_expert_index_comparison"],
        "routing_weights": source_topk["routing_weight_comparison"],
        "expert30_selected_rank_1": (
            len(selected_indices) > EXPERT_RANK and selected_indices[EXPERT_RANK] == EXPERT_INDEX
        ),
    }
    mlp1_guard_metrics = {
        "fused": source_mlp1["local_mlp1_vs_official_metrics"],
        "gate_slice": source_mlp1["gate_slice_metrics"],
        "up_slice": source_mlp1["up_slice_metrics"],
    }
    input_guard_metric = source_mlp1["mlp_norm_input_guard_metrics"]["metrics"]
    guards_pass = (
        input_guard_metric["matched"]
        and routing_guard["router_logits"]["matched"]
        and routing_guard["selected_expert_indices"]["ordered_match"]
        and routing_guard["routing_weights"]["matched"]
        and routing_guard["expert30_selected_rank_1"]
        and mlp1_guard_metrics["fused"]["matched"]
        and mlp1_guard_metrics["gate_slice"]["matched"]
        and mlp1_guard_metrics["up_slice"]["matched"]
    )

    mlp = router.load_layer0_mlp(args.model_root, device)
    with torch.inference_mode():
        weight = mlp.mlp1_weight[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
        bias = mlp.mlp1_bias[EXPERT_INDEX].detach().to(torch.bfloat16).contiguous()
        local_mlp1 = (
            torch.einsum("ch,h->c", weight, input_bf16) + bias
        ).reshape(FUSED_SIZE).contiguous()
        from gpt_oss.torch.model import swiglu

        local_swiglu = swiglu(local_mlp1.reshape(1, 1, FUSED_SIZE), limit=mlp.swiglu_limit).reshape(HIDDEN_SIZE).contiguous()
        replay = swiglu_formula(official_mlp1_bf16, gate_clamp=True, up_clamp=True, sigmoid_scale=SIGMOID_SCALE, bf16_output=True)

    local_f32 = local_swiglu.to(torch.float32)
    replay_f32 = replay.to(torch.float32)
    local_vs_official = compare_vector(local_f32, official_swiglu, "hidden_lane")
    replay_vs_official = compare_vector(replay_f32, official_swiglu, "hidden_lane")
    local_vs_replay = compare_with_names(local_f32, replay_f32, "hidden_lane", "local_runtime", "replay")

    discriminator_table = []
    focused_trace = None
    if not local_vs_official["matched"]:
        no_gate_clamp = swiglu_formula(official_mlp1_bf16, gate_clamp=False, up_clamp=True, sigmoid_scale=SIGMOID_SCALE, bf16_output=True)
        scale_one = swiglu_formula(official_mlp1_bf16, gate_clamp=True, up_clamp=True, sigmoid_scale=1.0, bf16_output=True)
        f32_output = swiglu_formula(official_mlp1_bf16, gate_clamp=True, up_clamp=True, sigmoid_scale=SIGMOID_SCALE, bf16_output=False)
        discriminator_table = [
            variant_entry(
                "official_formula_gate_clamp_max7_up_clamp_pm7_scale1p702_bf16_output",
                replay_f32,
                official_swiglu,
                local_f32,
            ),
            variant_entry(
                "no_gate_clamp_up_clamp_only_scale1p702_bf16_output",
                no_gate_clamp,
                official_swiglu,
                local_f32,
            ),
            variant_entry(
                "gate_up_clamp_sigmoid_scale1p0_bf16_output",
                scale_one,
                official_swiglu,
                local_f32,
            ),
            variant_entry(
                "gate_up_clamp_sigmoid_scale1p702_f32_output_before_bf16_cast",
                f32_output,
                official_swiglu,
                local_f32,
            ),
        ]
        worst = local_vs_official["worst_differing_hidden_lane"]
        if worst is not None:
            lane = int(worst["hidden_lane"])
            gate_pre = official_mlp1_bf16[2 * lane].to(torch.float32)
            up_pre = official_mlp1_bf16[2 * lane + 1].to(torch.float32)
            gate_post = gate_pre.clamp(max=SWIGLU_LIMIT)
            up_post = up_pre.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            sigmoid_arg = SIGMOID_SCALE * gate_post
            sigmoid_value = torch.sigmoid(sigmoid_arg)
            f32_value = gate_post * sigmoid_value * (up_post + 1)
            bf16_value = f32_value.to(torch.bfloat16).to(torch.float32)
            focused_trace = {
                "hidden_lane": lane,
                "gate_pre_clamp_value": float(gate_pre.item()),
                "gate_post_clamp_value": float(gate_post.item()),
                "up_pre_clamp_value": float(up_pre.item()),
                "up_post_clamp_value": float(up_post.item()),
                "sigmoid_argument": float(sigmoid_arg.item()),
                "sigmoid_value": float(sigmoid_value.item()),
                "f32_swiglu_value_before_bf16_cast": float(f32_value.item()),
                "bf16_output_bits": bf16_bits_from_float(float(bf16_value.item())),
                "bf16_output_value": float(bf16_value.item()),
                "local_runtime_output_value": float(local_f32[lane].item()),
                "official_ppp_output_value": float(official_swiglu[lane].item()),
                "official_semantics_replay_value": float(replay_f32[lane].item()),
                "local_minus_official": float(local_f32[lane].item() - official_swiglu[lane].item()),
                "first_divergent_stage": "undetermined; bounded SwiGLU discriminator recorded",
            }

    shape_or_layout_mismatch = list(local_f32.shape) != official_artifact.get("shape")
    if not guards_pass:
        classification = "expert30_swiglu_blocked_by_input_or_mlp1_guard_regression"
        earliest = "layer0_final_token_expert30_mlp1_output_before_swiglu_or_routing"
        next_step = "re-establish input/routing/mlp1 guards before expert30 SwiGLU"
    elif shape_or_layout_mismatch:
        classification = "expert30_swiglu_shape_or_layout_mismatch"
        earliest = "layer0_final_token_expert30_swiglu_output_before_mlp2"
        next_step = "align expert30 SwiGLU output layout/readout only"
    elif local_vs_official["matched"]:
        classification = "expert30_swiglu_before_mlp2_cleared"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_expert30_mlp2_output_before_bias"
    elif replay_vs_official["matched"] and not local_vs_replay["matched"]:
        classification = "expert30_swiglu_runtime_formula_or_dtype_mismatch"
        earliest = "layer0_final_token_expert30_swiglu_output_before_mlp2"
        next_step = "localize/prove expert30 SwiGLU clamp/formula/dtype only before mlp2"
    elif local_vs_replay["matched"] and not replay_vs_official["matched"]:
        classification = "expert30_swiglu_replay_policy_not_authoritative"
        earliest = "official-semantics expert30 SwiGLU replay"
        next_step = "refine expert30 SwiGLU official replay policy before mlp2"
    else:
        classification = "expert30_swiglu_runtime_formula_or_dtype_mismatch"
        earliest = "layer0_final_token_expert30_swiglu_output_before_mlp2"
        next_step = "localize/prove expert30 SwiGLU clamp/formula/dtype only before mlp2"

    swiglu_formula_meta = official_artifact.get("swiglu_formula", {})
    output = {
        "schema_version": "runtime_forward_layer0_mlp_expert30_swiglu_before_mlp2_status/v1",
        "mode": "mlp-expert30-swiglu-before-mlp2-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "selected_expert_rank": EXPERT_RANK,
            "selected_expert_index": EXPERT_INDEX,
            "hidden_size": HIDDEN_SIZE,
        },
        "source_artifact_paths": {
            "expert30_mlp1_cleared": str(args.source_expert30_mlp1),
            "selected_expert_outputs_diagnostic": str(args.source_selected_expert_outputs),
            "topk_routing_cleared": str(args.source_topk_routing),
            "official_expert30_mlp1_reference": str(mlp1_ref_path),
            "official_expert30_swiglu_reference": str(args.official_expert30_swiglu),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_expert30_swiglu),
        "mlp_norm_input_guard_metrics": {
            "metrics": input_guard_metric,
            "source_input_digest_matches_expert30_mlp1": source_mlp1["mlp_norm_input_guard_metrics"].get("source_input_digest_matches_official_expert30_mlp1_input"),
        },
        "routing_guard_metrics": routing_guard,
        "expert30_mlp1_guard_metrics": mlp1_guard_metrics,
        "local_swiglu_metadata": {
            **tensor_meta(
                local_f32,
                "expert 30 SwiGLU output vector [intermediate_size / world_size], before mlp2",
                "f32-expanded BF16 values",
            ),
            "before_mlp2": True,
            "routing_weights_applied": False,
            "gate_clamp_min": None,
            "gate_clamp_max": SWIGLU_LIMIT,
            "up_clamp_min": -SWIGLU_LIMIT,
            "up_clamp_max": SWIGLU_LIMIT,
            "sigmoid_scale": SIGMOID_SCALE,
            "formula": "gate * sigmoid(1.702 * gate) * (up + 1)",
            "output_cast_rounding_point": "BF16 output after SwiGLU",
        },
        "official_tensor_metadata": {
            "shape": official_artifact.get("shape"),
            "tensor_dtype": official_artifact.get("tensor_dtype"),
            "serialization_dtype": official_artifact.get("serialization_dtype"),
            "layout_interpretation": official_artifact.get("layout_interpretation"),
            "source_input_boundary": official_artifact.get("source_input_boundary"),
            "fused_gate_up_split_metadata": official_artifact.get("fused_gate_up_split_metadata"),
            "swiglu_formula": swiglu_formula_meta,
            "computation_dtype": official_artifact.get("computation_dtype"),
            "output_dtype_before_serialization": official_artifact.get("output_dtype_before_serialization"),
            "rounded_or_cast_to_bf16_after_swiglu": official_artifact.get("rounded_or_cast_to_bf16_after_swiglu"),
        },
        "local_swiglu_vs_official_metrics": local_vs_official,
        "official_semantics_replay_metrics": {
            "policy": "interleaved gate/up, gate clamp max 7, up clamp [-7, 7], sigmoid scale 1.702, BF16 output",
            "replay_vs_official": replay_vs_official,
            "local_runtime_vs_replay": local_vs_replay,
        },
        "discriminator_table": discriminator_table,
        "focused_mismatch_trace": focused_trace,
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
