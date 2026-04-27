#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token attention residual add before MLP status."
    )
    parser.add_argument("--source-o-proj", type=Path, required=True)
    parser.add_argument("--official-residual-add", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bf16_bits(value):
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


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


def extract_case_final_vector(artifact, case_id):
    for case in artifact.get("cases", []):
        if case.get("id") == case_id:
            return case["final_token_hidden_f32"]
    raise ValueError(f"case {case_id} not found")


def policy_variant(name, tensor, official):
    return {
        "name": name,
        "metrics_vs_official": compare_vector(tensor, official),
        "matches_official": bool(torch.equal(tensor.reshape(-1).to(torch.float32), official.reshape(-1).to(torch.float32))),
    }


def main():
    args = parse_args()
    source_o_proj = load_json(args.source_o_proj)
    official_residual_add_artifact = load_json(args.official_residual_add)
    rmsnorm = load_json(args.rmsnorm_replay)
    local_residual_artifact = load_json(args.local_residual_input)

    if source_o_proj.get("classification") != "layer0_attention_o_proj_before_residual_cleared":
        raise ValueError("source o_proj artifact is not cleared")
    if official_residual_add_artifact.get("classification") != "official_layer0_final_token_attention_residual_add_before_mlp_captured":
        raise ValueError("official residual-add artifact is not the expected PPP capture")

    case_id = "developer-message-user-smoke"
    official_residual_input = torch.tensor(
        rmsnorm["official_input_f32"], dtype=torch.float32
    ).reshape(74, 2880)[73]
    local_residual_input = torch.tensor(
        extract_case_final_vector(local_residual_artifact, case_id), dtype=torch.float32
    ).reshape(2880)
    official_o_proj_path = Path(source_o_proj["source_ppp_boundary_artifact_path"])
    official_o_proj_artifact = load_json(official_o_proj_path)
    o_proj = torch.tensor(official_o_proj_artifact["values"], dtype=torch.float32).reshape(2880)
    official_residual_add = torch.tensor(
        official_residual_add_artifact["values"], dtype=torch.float32
    ).reshape(2880)

    o_proj_guard = source_o_proj["o_proj_output_vs_official_metrics"]
    residual_input_guard = compare_vector(local_residual_input.to(torch.bfloat16), official_residual_input.to(torch.bfloat16))

    residual_bf16 = official_residual_input.to(torch.bfloat16)
    o_proj_bf16 = o_proj.to(torch.bfloat16)
    bf16_add = (residual_bf16 + o_proj_bf16).to(torch.bfloat16).to(torch.float32)
    f32_add_bf16_output = (residual_bf16.to(torch.float32) + o_proj_bf16.to(torch.float32)).to(torch.bfloat16).to(torch.float32)
    f32_expanded_add_bf16_output = (official_residual_input + o_proj).to(torch.bfloat16).to(torch.float32)
    local_runtime_like = (local_residual_input.to(torch.bfloat16) + o_proj_bf16).to(torch.bfloat16).to(torch.float32)

    residual_add_metric = compare_vector(bf16_add, official_residual_add)
    discriminator = []
    focused_trace = None
    dtype_finding = "not needed; BF16 residual add output matched official"
    if not residual_add_metric["matched"]:
        discriminator = [
            policy_variant("bf16_input_bf16_o_proj_bf16_add_output", bf16_add, official_residual_add),
            policy_variant("bf16_input_bf16_o_proj_f32_add_then_bf16_output", f32_add_bf16_output, official_residual_add),
            policy_variant("f32_expanded_input_f32_expanded_o_proj_f32_add_then_bf16_output", f32_expanded_add_bf16_output, official_residual_add),
            policy_variant("local_residual_input_bf16_plus_o_proj_bf16", local_runtime_like, official_residual_add),
        ]
        best = next((entry for entry in discriminator if entry["matches_official"]), None)
        dtype_finding = (
            f"{best['name']} matches official"
            if best is not None
            else "no bounded residual-add dtype variant matched official"
        )
        worst = residual_add_metric["worst_differing_hidden_lane"]
        if worst is not None:
            lane = int(worst["hidden_lane"])
            focused_trace = {
                "hidden_lane": lane,
                "residual_input_local_value": float(local_residual_input[lane].item()),
                "residual_input_official_equivalent_value": float(official_residual_input[lane].item()),
                "attention_o_proj_candidate_value": float(o_proj[lane].item()),
                "attention_o_proj_official_value": float(o_proj[lane].item()),
                "local_reconstructed_add_value": float(bf16_add[lane].item()),
                "official_residual_add_value": float(official_residual_add[lane].item()),
                "residual_input_bf16_bits": bf16_bits(float(residual_bf16[lane].to(torch.float32).item())),
                "o_proj_bf16_bits": bf16_bits(float(o_proj_bf16[lane].to(torch.float32).item())),
                "reconstructed_add_bf16_bits": bf16_bits(float(bf16_add[lane].item())),
                "official_residual_add_bf16_bits": bf16_bits(float(official_residual_add[lane].item())),
                "local_minus_official": float(bf16_add[lane].item() - official_residual_add[lane].item()),
            }

    if not o_proj_guard.get("matched"):
        classification = "attention_residual_add_blocked_by_o_proj_guard_regression"
        earliest = "layer0_final_token_attention_output_after_o_proj_before_residual"
        next_step = "re-establish attention o_proj before residual guard"
    elif not residual_input_guard["matched"]:
        classification = "attention_residual_add_blocked_by_residual_input_mismatch"
        earliest = "layer0_residual_input"
        next_step = "inspect layer0 residual input provenance only"
    elif not residual_add_metric["matched"]:
        classification = "attention_residual_add_dtype_policy_mismatch"
        earliest = "layer0_final_token_hidden_state_after_attention_residual_add_before_mlp"
        next_step = "prove scoped BF16 residual-add dtype policy before MLP"
    else:
        classification = "attention_residual_add_before_mlp_cleared_after_o_proj_candidates"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_mlp_norm_output_before_mlp_projections"

    output = {
        "schema_version": "runtime_forward_layer0_attention_residual_add_before_mlp_after_o_proj_candidates_status/v1",
        "mode": "attention-residual-add-before-mlp-after-o-proj-candidates-status",
        "exact_case": {
            "case_id": case_id,
            "layer_index": 0,
            "final_token_index": 73,
            "hidden_size": 2880,
        },
        "source_artifact_paths": {
            "attention_o_proj_cleared": str(args.source_o_proj),
            "official_residual_add_reference": str(args.official_residual_add),
            "official_equivalent_residual_input_from_rmsnorm_replay": str(args.rmsnorm_replay),
            "local_residual_input": str(args.local_residual_input),
            "official_attention_o_proj_output": str(official_o_proj_path),
        },
        "q_k_v_candidates_are_bench_proof_only": True,
        "attention_o_proj_guard_metrics": o_proj_guard,
        "residual_input_guard": {
            "availability_status": "official-equivalent residual input available from layer0 attention RMSNorm replay official_input_f32",
            "equivalence_statement": "layer0 attention RMSNorm input is the layer0 attention block residual input x used by AttentionBlock.forward before attn.norm",
            "source_boundary_used": "layer0_attn_norm_input / layer0_residual_input",
            "metrics": residual_input_guard,
        },
        "official_residual_add_reference_path": str(args.official_residual_add),
        "official_tensor_metadata": {
            "shape": official_residual_add_artifact.get("shape"),
            "tensor_dtype": official_residual_add_artifact.get("tensor_dtype"),
            "serialization_dtype": official_residual_add_artifact.get("serialization_dtype"),
            "layout_interpretation": official_residual_add_artifact.get("layout_interpretation"),
            "residual_add_semantics": official_residual_add_artifact.get("residual_add_semantics"),
        },
        "local_reconstructed_residual_add_metadata": {
            **tensor_meta(bf16_add, "[hidden_size]", "f32-expanded BF16 values"),
            "construction": "BF16 residual input + BF16 attention o_proj output, BF16 output round-trip",
            "mlp_norm_included": False,
            "mlp_included": False,
        },
        "residual_add_comparison_metrics": residual_add_metric,
        "dtype_add_policy_finding": dtype_finding,
        "dtype_add_policy_discriminator_table": discriminator,
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
