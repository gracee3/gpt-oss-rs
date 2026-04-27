#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base
import layer0_q_pre_post_rope_runtime_localization as qloc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token MLP RMSNorm output before MLP projections status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-residual-add", type=Path, required=True)
    parser.add_argument("--official-mlp-norm", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_vector(lhs, rhs):
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
            "hidden_lane": first_idx,
            "lhs_value": float(lhs[first_idx].item()),
            "rhs_value": float(rhs[first_idx].item()),
            "abs_diff": float(diff[first_idx].item()),
        }
        worst = {
            "hidden_lane": worst_idx,
            "lhs_value": float(lhs[worst_idx].item()),
            "rhs_value": float(rhs[worst_idx].item()),
            "abs_diff": float(diff[worst_idx].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
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


def bf16_bits(value):
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


def load_layer0_mlp(model_root: Path, device):
    torch_mod, _AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    global torch
    torch = torch_mod
    from gpt_oss.torch.model import MLPBlock

    config = base.load_restricted_config(model_root, ModelConfig)
    mlp = MLPBlock(config=config, device=device)
    mlp.eval()
    checkpoint = Checkpoint(str(base.resolve_oracle_checkpoint_dir(model_root)), device)
    for name, param in dict(mlp.named_parameters()).items():
        param.data.copy_(checkpoint.get(f"block.0.mlp.{name}"))
    return mlp


def rmsnorm_replay(input_bf16, weight, eps):
    t = input_bf16.to(torch.float32)
    sum_squares = torch.sum(t * t)
    mean_square = torch.mean(t * t)
    inverse_rms = torch.rsqrt(mean_square + float(eps))
    output = (t * inverse_rms * weight.to(torch.float32)).to(torch.bfloat16).to(torch.float32)
    return output, {
        "sum_of_squares": float(sum_squares.item()),
        "mean_square": float(mean_square.item()),
        "epsilon": float(eps),
        "inverse_rms": float(inverse_rms.item()),
    }


def replay_variant(name, input_tensor, weight_tensor, official, eps):
    replay, scalar = rmsnorm_replay(input_tensor, weight_tensor, eps)
    return {
        "name": name,
        "scalar_trace": scalar,
        "metrics_vs_official": compare_vector(replay, official),
        "matches_official": bool(torch.equal(replay.reshape(-1), official.reshape(-1).to(torch.float32))),
    }


def main():
    args = parse_args()
    source_residual_add = load_json(args.source_residual_add)
    official_mlp_norm_artifact = load_json(args.official_mlp_norm)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_residual_add.get("classification") != "attention_residual_add_before_mlp_cleared_after_o_proj_candidates":
        raise ValueError("source residual-add artifact is not cleared")
    if official_mlp_norm_artifact.get("classification") != "official_layer0_final_token_mlp_norm_output_before_mlp_projections_captured":
        raise ValueError("official MLP norm artifact is not the expected PPP capture")
    if official_mlp_norm_artifact.get("boundary") != "layer0_final_token_mlp_norm_output_before_mlp_projections":
        raise ValueError("official MLP norm artifact boundary is not usable")

    device = torch.device(args.device)
    residual_add_ref_path = Path(source_residual_add["official_residual_add_reference_path"])
    residual_add_ref = load_json(residual_add_ref_path)
    hidden_after_attn = torch.tensor(residual_add_ref["values"], dtype=torch.float32, device=device).reshape(2880)
    input_bf16 = hidden_after_attn.to(torch.bfloat16).contiguous()
    official_mlp_norm = torch.tensor(
        official_mlp_norm_artifact["values"], dtype=torch.float32, device=device
    ).reshape(2880)

    mlp = load_layer0_mlp(args.model_root, device)
    norm = mlp.norm
    weight = norm.scale.detach().to(torch.float32).contiguous()
    eps = float(norm.eps)

    with torch.inference_mode():
        local_runtime = norm(input_bf16.reshape(1, 2880)).reshape(2880).contiguous()
    local_runtime_f32 = local_runtime.to(torch.float32)
    replay, replay_scalar = rmsnorm_replay(input_bf16, weight, eps)

    input_guard = {
        "source_boundary_used": "layer0_final_token_hidden_state_after_attention_residual_add_before_mlp",
        "source_artifact_path": str(residual_add_ref_path),
        "metrics": source_residual_add["residual_add_comparison_metrics"],
        "source_input_digest_matches_official_mlp_norm_source": (
            residual_add_ref.get("finite_value_summary", {}).get("sha256_f32_le")
            == official_mlp_norm_artifact.get("source_input_boundary", {}).get("sha256_f32_le")
        ),
        "residual_add_digest": residual_add_ref.get("finite_value_summary", {}).get("sha256_f32_le"),
        "official_mlp_norm_source_digest": official_mlp_norm_artifact.get("source_input_boundary", {}).get("sha256_f32_le"),
    }

    official_parameters = official_mlp_norm_artifact.get("norm_parameters", {})
    parameter_mismatch = (
        list(weight.shape) != official_parameters.get("weight_shape")
        or abs(eps - float(official_parameters.get("epsilon", eps))) > 0.0
        or bool(official_parameters.get("bias_exists", False))
    )
    weight_metric = compare_vector(weight, weight)

    local_vs_official = compare_vector(local_runtime_f32, official_mlp_norm)
    replay_vs_official = compare_vector(replay, official_mlp_norm)
    local_vs_replay = compare_vector(local_runtime_f32, replay)

    dtype_table = []
    focused_trace = None
    if not local_vs_official["matched"]:
        dtype_table = [
            replay_variant(
                "bf16_input_fp32_weight_fp32_reduction_bf16_output",
                input_bf16,
                weight,
                official_mlp_norm,
                eps,
            ),
            replay_variant(
                "bf16_input_bf16_cast_weight_fp32_reduction_bf16_output",
                input_bf16,
                weight.to(torch.bfloat16),
                official_mlp_norm,
                eps,
            ),
            replay_variant(
                "fp32_expanded_input_fp32_weight_fp32_reduction_bf16_output",
                hidden_after_attn.to(torch.float32),
                weight,
                official_mlp_norm,
                eps,
            ),
            {
                "name": "local_runtime_captured_policy",
                "metrics_vs_official": local_vs_official,
                "matches_official": bool(local_vs_official["matched"]),
            },
        ]
        worst = local_vs_official["worst_differing_hidden_lane"]
        if worst is not None:
            lane = int(worst["hidden_lane"])
            t = input_bf16.to(torch.float32)
            pre_output = t[lane] * replay_scalar["inverse_rms"] * weight[lane]
            focused_trace = {
                "hidden_lane": lane,
                "input_value": float(t[lane].item()),
                "norm_weight_value": float(weight[lane].item()),
                "epsilon": eps,
                "sum_of_squares": replay_scalar["sum_of_squares"],
                "mean_square": replay_scalar["mean_square"],
                "inverse_rms_scalar": replay_scalar["inverse_rms"],
                "pre_output_weighted_value": float(pre_output.item()),
                "replay_bf16_output_bits": bf16_bits(float(replay[lane].item())),
                "replay_bf16_output_value": float(replay[lane].item()),
                "local_runtime_output_value": float(local_runtime_f32[lane].item()),
                "official_output_value": float(official_mlp_norm[lane].item()),
                "local_minus_official": float(local_runtime_f32[lane].item() - official_mlp_norm[lane].item()),
                "first_divergent_stage": "undetermined; bounded dtype discriminator recorded",
            }

    if not input_guard["metrics"]["matched"]:
        classification = "mlp_norm_blocked_by_attention_residual_add_input_regression"
        earliest = "layer0_final_token_hidden_state_after_attention_residual_add_before_mlp"
        next_step = "re-establish attention residual-add input before MLP"
    elif parameter_mismatch:
        classification = "mlp_norm_parameter_mismatch"
        earliest = "model.block[0].mlp.norm.parameters"
        next_step = "inspect block[0].mlp.norm parameter loading only"
    elif list(local_runtime_f32.shape) != list(official_mlp_norm.shape):
        classification = "mlp_norm_shape_or_layout_mismatch"
        earliest = "layer0_final_token_mlp_norm_output_before_mlp_projections"
        next_step = "align MLP norm output shape/layout to official flat hidden vector"
    elif local_vs_official["matched"]:
        classification = "mlp_norm_before_projections_cleared_after_attention_residual"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_mlp_router_logits_before_routing"
    elif replay_vs_official["matched"]:
        classification = "mlp_norm_runtime_dtype_or_kernel_policy_mismatch"
        earliest = "layer0_final_token_mlp_norm_output_before_mlp_projections"
        next_step = "isolate MLP RMSNorm dtype/kernel policy only before MLP projections"
    elif local_vs_replay["matched"] and not replay_vs_official["matched"]:
        classification = "mlp_norm_replay_policy_not_authoritative"
        earliest = "official-semantics MLP norm replay"
        next_step = "refine MLP RMSNorm replay policy before runtime attribution"
    else:
        classification = "mlp_norm_runtime_dtype_or_kernel_policy_mismatch"
        earliest = "layer0_final_token_mlp_norm_output_before_mlp_projections"
        next_step = "isolate MLP RMSNorm dtype/kernel policy only before MLP projections"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_norm_before_projections_after_attn_residual_status/v1",
        "mode": "mlp-norm-before-projections-after-attn-residual-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "hidden_size": 2880,
        },
        "source_artifact_paths": {
            "attention_residual_add_cleared": str(args.source_residual_add),
            "official_residual_add_source_input": str(residual_add_ref_path),
            "official_mlp_norm_reference": str(args.official_mlp_norm),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_mlp_norm),
        "input_provenance_guard_metrics": input_guard,
        "mlp_norm_parameter_metadata": {
            "module_path": "model.block[0].mlp.norm",
            "local_module_type": type(norm).__name__,
            "official_module_type": official_mlp_norm_artifact.get("mlp_norm_module", {}).get("type"),
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype).replace("torch.", ""),
            "storage_source": "Checkpoint block.0.mlp.norm.scale copied into RMSNorm FP32 scale parameter",
            "epsilon": eps,
            "bias_exists": False,
            "mean_subtraction": False,
            "variance_rule": "mean(x^2) over hidden dimension",
            "parameter_mismatch": bool(parameter_mismatch),
            "official_norm_parameters": official_parameters,
            "weight_summary": vector_summary(weight),
        },
        "mlp_norm_weight_comparison_metrics": {
            "availability_status": "official weight is accessible through the same checkpoint tensor block.0.mlp.norm.scale used for the local module",
            "metrics": weight_metric,
        },
        "local_runtime_mlp_norm_output_metadata": tensor_meta(
            local_runtime_f32,
            "flat hidden dimension vector [hidden_size] for final token before MLP projections",
            "f32-expanded BF16 values",
        ),
        "official_tensor_metadata": {
            "shape": official_mlp_norm_artifact.get("shape"),
            "tensor_dtype": official_mlp_norm_artifact.get("tensor_dtype"),
            "serialization_dtype": official_mlp_norm_artifact.get("serialization_dtype"),
            "layout_interpretation": official_mlp_norm_artifact.get("layout_interpretation"),
            "norm_computation_semantics": official_mlp_norm_artifact.get("norm_computation_semantics"),
        },
        "local_runtime_mlp_norm_vs_official_metrics": local_vs_official,
        "official_semantics_replay_metrics": {
            "policy": "BF16 input, FP32 weight, FP32 RMS reduction, x * inverse_rms * weight, BF16 output round-trip",
            "scalar_trace": replay_scalar,
            "replay_vs_official": replay_vs_official,
            "local_runtime_vs_replay": local_vs_replay,
        },
        "dtype_policy_discriminator_table": dtype_table,
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
