#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


HIDDEN_SIZE = 2880


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer1 final-token attention RMSNorm output before QKV status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-layer0-residual", type=Path, required=True)
    parser.add_argument("--official-layer1-attn-norm", type=Path, required=True)
    parser.add_argument("--official-layer0-after-mlp-residual", type=Path, required=True)
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


def load_layer1_attention(model_root: Path, device):
    torch_mod, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    global torch
    torch = torch_mod
    config = base.load_restricted_config(model_root, ModelConfig)
    attn = AttentionBlock(config=config, layer_idx=1, device=device)
    attn.eval()
    checkpoint = Checkpoint(str(base.resolve_oracle_checkpoint_dir(model_root)), device)
    for name, param in dict(attn.named_parameters()).items():
        param.data.copy_(checkpoint.get(f"block.1.attn.{name}"))
    return attn


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
    source_layer0 = load_json(args.source_layer0_residual)
    official_layer1 = load_json(args.official_layer1_attn_norm)
    official_layer0 = load_json(args.official_layer0_after_mlp_residual)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_layer0.get("classification") != "layer0_hidden_after_mlp_residual_add_cleared":
        raise ValueError("source layer0 after-MLP residual artifact is not cleared")
    if official_layer1.get("classification") != "official_layer1_final_token_attention_norm_output_before_qkv_captured":
        raise ValueError("official layer1 attention norm artifact is not the expected PPP capture")
    if official_layer1.get("boundary") != "layer1_final_token_attention_norm_output_before_qkv":
        raise ValueError("official layer1 attention norm boundary is not usable")
    if official_layer0.get("boundary") != "layer0_final_token_hidden_state_after_mlp_residual_add":
        raise ValueError("official layer0 after-MLP residual source boundary is not usable")

    device = torch.device(args.device)
    hidden_after_layer0 = torch.tensor(
        official_layer0["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)
    input_bf16 = hidden_after_layer0.to(torch.bfloat16).contiguous()
    official_norm = torch.tensor(
        official_layer1["values"], dtype=torch.float32, device=device
    ).reshape(HIDDEN_SIZE)

    attn = load_layer1_attention(args.model_root, device)
    norm = attn.norm
    weight = norm.scale.detach().to(torch.float32).contiguous()
    eps = float(norm.eps)

    with torch.inference_mode():
        local_runtime = norm(input_bf16.reshape(1, HIDDEN_SIZE)).reshape(HIDDEN_SIZE).contiguous()
    local_runtime_f32 = local_runtime.to(torch.float32)
    replay, replay_scalar = rmsnorm_replay(input_bf16, weight, eps)

    layer0_metric = source_layer0["after_mlp_residual_comparison_metrics"]
    layer0_source_digest = official_layer0.get("finite_value_summary", {}).get("sha256_f32_le")
    official_source_digest = official_layer1.get("source_input_boundary", {}).get("sha256_f32_le")
    input_guard = {
        "source_boundary_used": "layer0_final_token_hidden_state_after_mlp_residual_add",
        "source_artifact_path": str(args.source_layer0_residual),
        "official_source_boundary_path": str(args.official_layer0_after_mlp_residual),
        "metrics": layer0_metric,
        "source_input_digest_matches_official_layer1_source": layer0_source_digest == official_source_digest,
        "layer0_after_mlp_residual_digest": layer0_source_digest,
        "official_layer1_source_digest": official_source_digest,
    }

    official_parameters = official_layer1.get("norm_parameters", {})
    weight_digest = sha256_f32_le(weight)
    official_weight_digest = official_parameters.get("weight_sha256_f32_le")
    parameter_mismatch = (
        list(weight.shape) != official_parameters.get("weight_shape")
        or abs(eps - float(official_parameters.get("epsilon", eps))) > 0.0
        or bool(official_parameters.get("bias_exists", False))
        or (official_weight_digest is not None and weight_digest != official_weight_digest)
    )
    weight_metric = {
        "availability_status": "official weight digest available from PPP; local weight loaded from checkpoint block.1.attn.norm.scale",
        "max_abs_diff": 0.0 if weight_digest == official_weight_digest else None,
        "mean_abs_diff": 0.0 if weight_digest == official_weight_digest else None,
        "matched": bool(weight_digest == official_weight_digest),
        "local_sha256_f32_le": weight_digest,
        "official_sha256_f32_le": official_weight_digest,
    }

    local_vs_official = compare_vector(local_runtime_f32, official_norm)
    replay_vs_official = compare_vector(replay, official_norm)
    local_vs_replay = compare_vector(local_runtime_f32, replay)

    dtype_table = []
    focused_trace = None
    if not local_vs_official["matched"]:
        dtype_table = [
            replay_variant(
                "bf16_input_fp32_weight_fp32_reduction_bf16_output",
                input_bf16,
                weight,
                official_norm,
                eps,
            ),
            replay_variant(
                "bf16_input_bf16_cast_weight_fp32_reduction_bf16_output",
                input_bf16,
                weight.to(torch.bfloat16),
                official_norm,
                eps,
            ),
            replay_variant(
                "fp32_expanded_input_fp32_weight_fp32_reduction_bf16_output",
                hidden_after_layer0.to(torch.float32),
                weight,
                official_norm,
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
                "official_output_value": float(official_norm[lane].item()),
                "local_minus_official": float(local_runtime_f32[lane].item() - official_norm[lane].item()),
                "first_divergent_stage": "undetermined; bounded dtype discriminator recorded",
            }

    if not input_guard["metrics"]["matched"] or not input_guard["source_input_digest_matches_official_layer1_source"]:
        classification = "layer1_attn_norm_blocked_by_layer0_output_regression"
        earliest = "layer0_final_token_hidden_state_after_mlp_residual_add"
        next_step = "revalidate the regressed input seam only"
    elif parameter_mismatch:
        classification = "layer1_attn_norm_parameter_mismatch"
        earliest = "model.block[1].attn.norm.parameters"
        next_step = "inspect block[1].attn.norm parameter loading only"
    elif list(local_runtime_f32.shape) != list(official_norm.shape):
        classification = "layer1_attn_norm_shape_or_layout_mismatch"
        earliest = "layer1_final_token_attention_norm_output_before_qkv"
        next_step = "align layer1 attention norm output shape/layout to official flat hidden vector"
    elif local_vs_official["matched"]:
        classification = "layer1_attn_norm_before_qkv_cleared_after_layer0"
        earliest = "none"
        next_step = "ask PPP for exactly layer1_final_token_q_projection_output_before_rope"
    elif replay_vs_official["matched"]:
        classification = "layer1_attn_norm_runtime_dtype_or_kernel_policy_mismatch"
        earliest = "layer1_final_token_attention_norm_output_before_qkv"
        next_step = "isolate layer1 attention RMSNorm dtype/kernel policy only before QKV"
    elif local_vs_replay["matched"] and not replay_vs_official["matched"]:
        classification = "layer1_attn_norm_replay_policy_not_authoritative"
        earliest = "official-semantics layer1 attention norm replay"
        next_step = "refine layer1 attention RMSNorm replay policy before runtime attribution"
    else:
        classification = "layer1_attn_norm_runtime_dtype_or_kernel_policy_mismatch"
        earliest = "layer1_final_token_attention_norm_output_before_qkv"
        next_step = "isolate layer1 attention RMSNorm dtype/kernel policy only before QKV"

    output = {
        "schema_version": "runtime_forward_layer1_attn_norm_before_qkv_after_layer0_status/v1",
        "mode": "layer1-attn-norm-before-qkv-after-layer0-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 1,
            "token_index": 73,
            "hidden_size": HIDDEN_SIZE,
        },
        "source_artifact_paths": {
            "layer0_after_mlp_residual_cleared": str(args.source_layer0_residual),
            "official_layer0_after_mlp_residual_source_input": str(args.official_layer0_after_mlp_residual),
            "official_layer1_attn_norm_reference": str(args.official_layer1_attn_norm),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_layer1_attn_norm),
        "layer0_output_input_guard_metrics": input_guard,
        "layer1_attention_norm_parameter_metadata": {
            "module_path": "model.block[1].attn.norm",
            "local_module_type": type(norm).__name__,
            "official_module_type": official_layer1.get("norm_module", {}).get("type"),
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype).replace("torch.", ""),
            "storage_source": "Checkpoint block.1.attn.norm.scale copied into RMSNorm FP32 scale parameter",
            "epsilon": eps,
            "bias_exists": False,
            "mean_subtraction": False,
            "variance_rule": "mean(x^2) over hidden dimension",
            "parameter_mismatch": bool(parameter_mismatch),
            "official_norm_parameters": official_parameters,
            "weight_summary": vector_summary(weight),
        },
        "layer1_attention_norm_weight_comparison_metrics": weight_metric,
        "local_runtime_layer1_norm_output_metadata": tensor_meta(
            local_runtime_f32,
            "flat hidden dimension vector [hidden_size] for final token before layer1 QKV",
            "f32-expanded BF16 values",
        ),
        "official_tensor_metadata": {
            "shape": official_layer1.get("shape"),
            "tensor_dtype": official_layer1.get("tensor_dtype"),
            "serialization_dtype": official_layer1.get("serialization_dtype"),
            "layout_interpretation": official_layer1.get("layout_interpretation"),
            "norm_computation_semantics": official_layer1.get("norm_computation_semantics"),
        },
        "local_runtime_layer1_norm_vs_official_metrics": local_vs_official,
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
