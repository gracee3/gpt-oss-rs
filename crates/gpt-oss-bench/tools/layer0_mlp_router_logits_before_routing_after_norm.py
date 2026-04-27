#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token MLP router logits before routing status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-mlp-norm", type=Path, required=True)
    parser.add_argument("--official-router-logits", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_vector(lhs, rhs, index_name="index"):
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


def compare_matrix(lhs, rhs):
    lhs = lhs.to(torch.float32)
    rhs = rhs.to(torch.float32)
    diff = (lhs - rhs).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        first_row, first_lane = [int(v) for v in mismatch.nonzero(as_tuple=False)[0].tolist()]
        worst_flat = int(diff.reshape(-1).argmax().item())
        worst_row = worst_flat // lhs.shape[1]
        worst_lane = worst_flat % lhs.shape[1]
        first = {
            "row": first_row,
            "hidden_lane": first_lane,
            "lhs_value": float(lhs[first_row, first_lane].item()),
            "rhs_value": float(rhs[first_row, first_lane].item()),
            "abs_diff": float(diff[first_row, first_lane].item()),
        }
        worst = {
            "row": worst_row,
            "hidden_lane": worst_lane,
            "lhs_value": float(lhs[worst_row, worst_lane].item()),
            "rhs_value": float(rhs[worst_row, worst_lane].item()),
            "abs_diff": float(diff[worst_row, worst_lane].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_element_count": int(mismatch.sum().item()),
        "first_differing_row_lane": first,
        "worst_differing_row_lane": worst,
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


def manual_ltr(input_bf16, weight_bf16, bias_bf16):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16.to(torch.float32)
    out = []
    for expert in range(w.shape[0]):
        acc = torch.tensor(0.0, dtype=torch.float32)
        row = w[expert]
        for lane in range(row.numel()):
            acc = acc + x[lane] * row[lane]
        if bias_bf16 is not None:
            acc = acc + bias_bf16[expert].to(torch.float32)
        out.append(acc)
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def manual_pairwise(input_bf16, weight_bf16, bias_bf16):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16.to(torch.float32)
    out = []
    for expert in range(w.shape[0]):
        terms = x * w[expert]
        while terms.numel() > 1:
            if terms.numel() % 2 == 1:
                tail = terms[-1:]
                terms = terms[:-1]
            else:
                tail = None
            terms = terms.reshape(-1, 2).sum(dim=1)
            if tail is not None:
                terms = torch.cat([terms, tail])
        acc = terms[0]
        if bias_bf16 is not None:
            acc = acc + bias_bf16[expert].to(torch.float32)
        out.append(acc)
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def variant_entry(name, tensor, official, local_runtime):
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official, "expert_index"),
        "metrics_vs_local_runtime": compare_vector(tensor, local_runtime, "expert_index"),
        "matches_official": bool(torch.equal(tensor.reshape(-1).to(torch.float32), official.reshape(-1).to(torch.float32))),
        "matches_local_runtime": bool(torch.equal(tensor.reshape(-1).to(torch.float32), local_runtime.reshape(-1).to(torch.float32))),
    }


def main():
    args = parse_args()
    source_mlp_norm = load_json(args.source_mlp_norm)
    official_router_artifact = load_json(args.official_router_logits)
    local_residual_input_artifact = load_json(args.local_residual_input)

    if source_mlp_norm.get("classification") != "mlp_norm_before_projections_cleared_after_attention_residual":
        raise ValueError("source MLP norm artifact is not cleared")
    if official_router_artifact.get("classification") != "official_layer0_final_token_mlp_router_logits_before_routing_captured":
        raise ValueError("official router logits artifact is not the expected PPP capture")
    if official_router_artifact.get("boundary") != "layer0_final_token_mlp_router_logits_before_routing":
        raise ValueError("official router logits artifact boundary is not usable")

    device = torch.device(args.device)
    mlp_norm_ref_path = Path(source_mlp_norm["official_ppp_reference_path"])
    mlp_norm_ref = load_json(mlp_norm_ref_path)
    input_norm = torch.tensor(mlp_norm_ref["values"], dtype=torch.float32, device=device).reshape(2880)
    input_bf16 = input_norm.to(torch.bfloat16).contiguous()
    official_logits = torch.tensor(
        official_router_artifact["values"], dtype=torch.float32, device=device
    ).reshape(32)

    input_guard = {
        "source_boundary_used": "layer0_final_token_mlp_norm_output_before_mlp_projections",
        "source_artifact_path": str(mlp_norm_ref_path),
        "metrics": source_mlp_norm["local_runtime_mlp_norm_vs_official_metrics"],
        "source_input_digest_matches_official_router_input": (
            mlp_norm_ref.get("finite_value_summary", {}).get("sha256_f32_le")
            == official_router_artifact.get("router_input_boundary", {}).get("sha256_f32_le")
        ),
        "mlp_norm_digest": mlp_norm_ref.get("finite_value_summary", {}).get("sha256_f32_le"),
        "official_router_input_digest": official_router_artifact.get("router_input_boundary", {}).get("sha256_f32_le"),
    }

    mlp = load_layer0_mlp(args.model_root, device)
    gate = mlp.gate
    weight = gate.weight.detach().to(torch.bfloat16).contiguous()
    bias = gate.bias.detach().to(torch.bfloat16).contiguous() if gate.bias is not None else None
    official_weight_meta = official_router_artifact.get("router_weight_metadata", {})
    official_bias_meta = official_router_artifact.get("router_bias_metadata", {})

    with torch.inference_mode():
        local_runtime = gate(input_bf16.reshape(1, 2880)).reshape(32).contiguous()
        pytorch_f_linear = torch.nn.functional.linear(
            input_bf16.reshape(1, 2880), weight, bias
        ).reshape(32).contiguous()
        no_bias = torch.nn.functional.linear(
            input_bf16.reshape(1, 2880), weight, None
        ).reshape(32).contiguous()

    local_runtime_f32 = local_runtime.to(torch.float32)
    f_linear_f32 = pytorch_f_linear.to(torch.float32)
    no_bias_f32 = no_bias.to(torch.float32)
    local_vs_official = compare_vector(local_runtime_f32, official_logits, "expert_index")
    f_linear_vs_official = compare_vector(f_linear_f32, official_logits, "expert_index")
    local_vs_f_linear = compare_vector(local_runtime_f32, f_linear_f32, "expert_index")
    no_bias_vs_official = compare_vector(no_bias_f32, official_logits, "expert_index")
    no_bias_vs_local = compare_vector(no_bias_f32, local_runtime_f32, "expert_index")

    weight_digest = sha256_f32_le(weight)
    bias_digest = sha256_f32_le(bias) if bias is not None else None
    weight_metadata_mismatch = (
        list(weight.shape) != official_weight_meta.get("shape")
        or weight_digest != official_weight_meta.get("sha256_f32_le")
    )
    bias_metadata_mismatch = (
        (bias is None) != bool(not official_bias_meta.get("present", False))
        or (bias is not None and list(bias.shape) != official_bias_meta.get("shape"))
        or (bias is not None and bias_digest != official_bias_meta.get("sha256_f32_le"))
    )

    policy_table = []
    focused_trace = None
    if not local_vs_official["matched"]:
        ltr = manual_ltr(input_bf16, weight, bias)
        ltr_no_bias = manual_ltr(input_bf16, weight, None)
        pairwise = manual_pairwise(input_bf16, weight, bias)
        policy_table = [
            variant_entry(
                "bf16_input_bf16_weight_bf16_bias_torch_functional_linear",
                f_linear_f32,
                official_logits,
                local_runtime_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_no_bias_torch_functional_linear",
                no_bias_f32,
                official_logits,
                local_runtime_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_bf16_bias_left_to_right_f32_accum_bf16_output",
                ltr,
                official_logits,
                local_runtime_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_no_bias_left_to_right_f32_accum_bf16_output",
                ltr_no_bias,
                official_logits,
                local_runtime_f32,
            ),
            variant_entry(
                "bf16_input_bf16_weight_bf16_bias_pairwise_f32_accum_bf16_output",
                pairwise,
                official_logits,
                local_runtime_f32,
            ),
        ]
        worst = local_vs_official["worst_differing_expert_index"]
        if worst is not None:
            expert = int(worst["expert_index"])
            focused_trace = {
                "expert_index": expert,
                "input_digest": sha256_f32_le(input_bf16),
                "router_weight_row_digest": sha256_f32_le(weight[expert]),
                "bias_value": float(bias[expert].to(torch.float32).item()) if bias is not None else None,
                "local_runtime_logit": float(local_runtime_f32[expert].item()),
                "official_ppp_logit": float(official_logits[expert].item()),
                "official_semantics_replay_logit": float(f_linear_f32[expert].item()),
                "no_bias_replay_logit": float(no_bias_f32[expert].item()),
                "local_minus_official": float(local_runtime_f32[expert].item() - official_logits[expert].item()),
                "local_bf16_bits": bf16_bits(float(local_runtime_f32[expert].item())),
                "official_bf16_bits": bf16_bits(float(official_logits[expert].item())),
                "mismatch_scale": "bias-sized" if no_bias_vs_local["matched"] and f_linear_vs_official["matched"] else "accumulation-policy-sized or capture/layout/readout-sized",
                "first_divergent_stage": "undetermined; bounded policy discriminator recorded",
            }

    if not input_guard["metrics"]["matched"]:
        classification = "router_logits_blocked_by_mlp_norm_input_regression"
        earliest = "layer0_final_token_mlp_norm_output_before_mlp_projections"
        next_step = "re-establish MLP norm output before router logits"
    elif weight_metadata_mismatch or bias_metadata_mismatch:
        classification = "router_weight_or_bias_mismatch"
        earliest = "model.block[0].mlp.gate.parameters"
        next_step = "inspect block[0].mlp.gate parameter loading only"
    elif list(local_runtime_f32.shape) != list(official_logits.shape):
        classification = "router_logits_shape_or_layout_mismatch"
        earliest = "layer0_final_token_mlp_router_logits_before_routing"
        next_step = "align router logits shape/layout to official [num_experts]"
    elif local_vs_official["matched"]:
        classification = "router_logits_before_routing_cleared_after_mlp_norm"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_mlp_topk_expert_indices_and_routing_weights"
    elif no_bias_vs_local["matched"] and f_linear_vs_official["matched"]:
        classification = "router_bias_application_mismatch"
        earliest = "model.block[0].mlp.gate.bias application"
        next_step = "implement/prove scoped router bias application fix before top-k routing"
    elif f_linear_vs_official["matched"]:
        classification = "router_runtime_arithmetic_or_capture_mismatch"
        earliest = "layer0_final_token_mlp_router_logits_before_routing"
        next_step = "localize router linear arithmetic/capture only before top-k routing"
    elif local_vs_f_linear["matched"] and not f_linear_vs_official["matched"]:
        classification = "router_replay_policy_not_authoritative"
        earliest = "official-semantics router replay"
        next_step = "refine router linear official replay policy before runtime attribution"
    else:
        classification = "router_runtime_arithmetic_or_capture_mismatch"
        earliest = "layer0_final_token_mlp_router_logits_before_routing"
        next_step = "localize router linear arithmetic/capture only before top-k routing"

    output = {
        "schema_version": "runtime_forward_layer0_mlp_router_logits_before_routing_after_norm_status/v1",
        "mode": "mlp-router-logits-before-routing-after-norm-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_index": 73,
            "num_experts": 32,
            "hidden_size": 2880,
        },
        "source_artifact_paths": {
            "mlp_norm_cleared": str(args.source_mlp_norm),
            "official_mlp_norm_input": str(mlp_norm_ref_path),
            "official_router_logits_reference": str(args.official_router_logits),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_router_logits),
        "mlp_norm_input_guard_metrics": input_guard,
        "router_weight_metadata": {
            "module_path": "model.block[0].mlp.gate",
            "shape": list(weight.shape),
            "dtype": str(weight.dtype).replace("torch.", ""),
            "layout_orientation": "[num_experts, hidden_size]; Linear computes input @ weight.T + bias",
            "sha256_f32_le": weight_digest,
            "official_metadata": official_weight_meta,
            "metadata_digest_matches_official": weight_digest == official_weight_meta.get("sha256_f32_le"),
        },
        "router_bias_metadata": {
            "present": bias is not None,
            "shape": list(bias.shape) if bias is not None else None,
            "dtype": str(bias.dtype).replace("torch.", "") if bias is not None else None,
            "all_zero": bool(torch.all(bias.to(torch.float32) == 0).item()) if bias is not None else None,
            "nonzero_count": int(torch.count_nonzero(bias.to(torch.float32)).item()) if bias is not None else 0,
            "sha256_f32_le": bias_digest,
            "official_metadata": official_bias_meta,
            "metadata_digest_matches_official": bias_digest == official_bias_meta.get("sha256_f32_le"),
            "local_runtime_applies_bias": gate.bias is not None,
        },
        "router_weight_comparison_metrics": {
            "availability_status": "official weight is accessible through the same checkpoint tensor block.0.mlp.gate.weight used for the local module",
            "metrics": compare_matrix(weight, weight),
        },
        "router_bias_comparison_metrics": {
            "availability_status": "official bias is accessible through the same checkpoint tensor block.0.mlp.gate.bias used for the local module",
            "metrics": compare_vector(bias, bias, "expert_index") if bias is not None else None,
        },
        "local_runtime_router_logits_metadata": tensor_meta(
            local_runtime_f32,
            "flat expert-logit vector [num_experts] before top-k/routing",
            "f32-expanded BF16 values",
        ),
        "official_tensor_metadata": {
            "shape": official_router_artifact.get("shape"),
            "tensor_dtype": official_router_artifact.get("tensor_dtype"),
            "serialization_dtype": official_router_artifact.get("serialization_dtype"),
            "layout_interpretation": official_router_artifact.get("layout_interpretation"),
            "router_computation_semantics": official_router_artifact.get("router_computation_semantics"),
            "router_routing_metadata": official_router_artifact.get("router_routing_metadata"),
        },
        "local_runtime_router_logits_vs_official_metrics": local_vs_official,
        "official_semantics_replay_metrics": {
            "policy": "torch.nn.functional.linear with BF16 input, BF16 weight, BF16 bias, BF16 output",
            "f_linear_vs_official": f_linear_vs_official,
            "local_runtime_vs_f_linear": local_vs_f_linear,
            "no_bias_replay_vs_official": no_bias_vs_official,
            "no_bias_replay_vs_local_runtime": no_bias_vs_local,
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
