#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


BOUNDARY_ORDER = [
    "layer1_final_token_mlp_norm_output_before_mlp_projections",
    "layer1_final_token_mlp_router_logits_before_routing",
    "layer1_final_token_mlp_topk_expert_indices_and_routing_weights",
    "layer1_final_token_selected_expert_outputs_before_routing_weighted_sum",
    "layer1_final_token_mlp_output_after_routing_weighted_sum_before_residual",
    "layer1_final_token_hidden_state_after_mlp_residual_add",
]

CLASSIFICATION_BY_BOUNDARY = {
    "layer1_final_token_mlp_norm_output_before_mlp_projections": "layer1_mlp_norm_mismatch_before_projections",
    "layer1_final_token_mlp_router_logits_before_routing": "layer1_router_logits_mismatch_after_mlp_norm",
    "layer1_final_token_mlp_topk_expert_indices_and_routing_weights": "layer1_topk_routing_mismatch_after_router_logits",
    "layer1_final_token_selected_expert_outputs_before_routing_weighted_sum": "layer1_selected_expert_outputs_mismatch_before_weighted_sum",
    "layer1_final_token_mlp_output_after_routing_weighted_sum_before_residual": "layer1_weighted_expert_sum_mismatch_after_selected_outputs_clear",
    "layer1_final_token_hidden_state_after_mlp_residual_add": "layer1_mlp_residual_add_mismatch_after_weighted_sum_clear",
}

NEXT_BY_CLASS = {
    "layer1_mlp_norm_mismatch_before_projections": "inspect layer1 MLP norm only",
    "layer1_router_logits_mismatch_after_mlp_norm": "inspect layer1 router linear weight/bias/arithmetic only",
    "layer1_topk_routing_mismatch_after_router_logits": "inspect layer1 top-k/routing only",
    "layer1_selected_expert_outputs_mismatch_before_weighted_sum": "localize first mismatching layer1 selected expert internal boundary",
    "layer1_weighted_expert_sum_mismatch_after_selected_outputs_clear": "inspect layer1 weighted expert sum dtype/accumulation only",
    "layer1_mlp_residual_add_mismatch_after_weighted_sum_clear": "inspect layer1 MLP residual add dtype policy only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare local layer1 MLP tensors against the official ordered PPP bundle."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-layer1-attention-bundle", type=Path, required=True)
    parser.add_argument("--official-bundle", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_from_values(boundary, device):
    values = boundary.get("values")
    if not isinstance(values, list):
        raise ValueError(f"official boundary {boundary.get('boundary')} has no values")
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(boundary["shape"])


def sha256_f32_le(tensor):
    flat = tensor.reshape(-1).detach().cpu().to(torch.float32)
    hasher = hashlib.sha256()
    for value in flat.tolist():
        hasher.update(struct.pack("<f", float(value)))
    return hasher.hexdigest()


def finite_compare(lhs, rhs):
    lhs_f = lhs.reshape(-1).to(torch.float32)
    rhs_f = rhs.reshape(-1).to(torch.float32)
    both_nan = torch.isnan(lhs_f) & torch.isnan(rhs_f)
    both_pos_inf = torch.isposinf(lhs_f) & torch.isposinf(rhs_f)
    both_neg_inf = torch.isneginf(lhs_f) & torch.isneginf(rhs_f)
    equal_nonfinite = both_nan | both_pos_inf | both_neg_inf
    finite = torch.isfinite(lhs_f) & torch.isfinite(rhs_f)
    diff = torch.zeros_like(lhs_f, dtype=torch.float32)
    diff[finite] = (lhs_f[finite] - rhs_f[finite]).abs()
    unequal = ~(equal_nonfinite | (finite & (diff == 0)))
    first = None
    worst = None
    if bool(unequal.any().item()):
        first_idx = int(unequal.nonzero(as_tuple=False)[0].item())
        finite_diff = diff.clone()
        finite_diff[~finite] = torch.where(
            unequal[~finite],
            torch.tensor(float("inf"), device=lhs_f.device),
            torch.tensor(0.0, device=lhs_f.device),
        )
        worst_idx = int(finite_diff.argmax().item())
        first = flat_trace(first_idx, lhs_f, rhs_f, diff)
        worst = flat_trace(worst_idx, lhs_f, rhs_f, diff)
    finite_diff_values = diff[finite]
    return {
        "max_abs_diff": float(finite_diff_values.max().item()) if finite_diff_values.numel() else 0.0,
        "mean_abs_diff": float(finite_diff_values.mean().item()) if finite_diff_values.numel() else 0.0,
        "matched": bool(not unequal.any().item()),
        "mismatching_value_count": int(unequal.sum().item()),
        "first_differing_flat_index": first,
        "worst_differing_flat_index": worst,
    }


def flat_trace(idx, lhs, rhs, diff):
    return {
        "flat_index": int(idx),
        "local_value": float(lhs[idx].item()),
        "official_value": float(rhs[idx].item()),
        "abs_diff": float(diff[idx].item()) if torch.isfinite(diff[idx]) else float("inf"),
    }


def unravel(flat_index, shape):
    out = []
    rem = int(flat_index)
    for dim in reversed(shape):
        out.append(rem % int(dim))
        rem //= int(dim)
    return list(reversed(out))


def logical_index(boundary, flat_index):
    if flat_index is None:
        return None
    name = boundary["boundary"]
    idx = unravel(flat_index, boundary["shape"])
    if name == "layer1_final_token_mlp_router_logits_before_routing":
        return {"expert_index": idx[0]}
    if name == "layer1_final_token_mlp_topk_expert_indices_and_routing_weights":
        return {"selected_rank": idx[0]}
    if name == "layer1_final_token_selected_expert_outputs_before_routing_weighted_sum":
        return {"selected_rank": idx[0], "hidden_lane": idx[1]}
    return {"hidden_lane": idx[0]}


def issue_class(boundary_name):
    if "mlp_norm" in boundary_name:
        return "MLP norm"
    if "router_logits" in boundary_name:
        return "router logits"
    if "topk" in boundary_name:
        return "top-k/routing"
    if "selected_expert_outputs" in boundary_name:
        return "selected expert output"
    if "weighted_sum" in boundary_name:
        return "weighted expert sum"
    if "residual_add" in boundary_name:
        return "residual add"
    return "capture/readout"


def tensor_meta(tensor, boundary):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": "f32-expanded BF16 values",
        "layout_interpretation": boundary.get("layout_interpretation"),
    }


def load_local_model(model_root: Path, device):
    torch_mod, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    global torch
    torch = torch_mod
    from gpt_oss.torch.model import MLPBlock, TransformerBlock

    class Layer1MlpReplay(torch.nn.Module):
        def __init__(self, config, device):
            super().__init__()
            self.embedding = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
            )
            self.block0 = TransformerBlock(config=config, layer_idx=0, device=device)
            self.attn1 = AttentionBlock(config=config, layer_idx=1, device=device)
            self.mlp1 = MLPBlock(config=config, device=device)

    config = base.load_restricted_config(model_root, ModelConfig)
    model = Layer1MlpReplay(config=config, device=device)
    model.eval()
    checkpoint = Checkpoint(str(base.resolve_oracle_checkpoint_dir(model_root)), device)
    for name, param in dict(model.named_parameters()).items():
        if name == "embedding.weight":
            param.data.copy_(checkpoint.get("embedding.weight"))
        elif name.startswith("block0."):
            param.data.copy_(checkpoint.get(f"block.0.{name.removeprefix('block0.')}"))
        elif name.startswith("attn1."):
            param.data.copy_(checkpoint.get(f"block.1.attn.{name.removeprefix('attn1.')}"))
        elif name.startswith("mlp1."):
            param.data.copy_(checkpoint.get(f"block.1.mlp.{name.removeprefix('mlp1.')}"))
        else:
            raise ValueError(f"unexpected parameter name {name}")
    return model


def swiglu_local(x, limit):
    gate = x[..., ::2].clamp(min=None, max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(1.702 * gate) * (up + 1)


def build_local_boundaries(model, input_token_ids):
    device = model.embedding.weight.device
    input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        x0 = model.embedding(input_ids)
        layer0_out = model.block0(x0)
        layer1_mlp_input = model.attn1(layer0_out)
        final_input = layer1_mlp_input[-1:].contiguous()
        mlp = model.mlp1
        normed = mlp.norm(final_input)
        router_logits = mlp.gate(normed)
        topk = torch.topk(router_logits, k=mlp.experts_per_token, dim=-1, sorted=True)
        routing_weights = torch.nn.functional.softmax(topk.values, dim=1)
        expert_indices = topk.indices
        mlp1_weight = mlp.mlp1_weight[expert_indices, ...]
        mlp1_bias = mlp.mlp1_bias[expert_indices, ...]
        expert = torch.einsum("beck,bk->bec", mlp1_weight, normed) + mlp1_bias
        expert = swiglu_local(expert, mlp.swiglu_limit)
        mlp2_weight = mlp.mlp2_weight[expert_indices, ...]
        mlp2_bias = mlp.mlp2_bias[expert_indices, ...]
        selected_outputs = torch.einsum("beck,bek->bec", mlp2_weight, expert)
        if mlp.world_size > 1:
            raise ValueError("world_size > 1 is not supported in this diagnostic")
        selected_outputs = (selected_outputs + mlp2_bias).to(torch.bfloat16)
        weighted_sum = torch.einsum("bec,be->bc", selected_outputs, routing_weights).to(torch.bfloat16)
        residual = (final_input + weighted_sum).to(torch.bfloat16)

    local = {
        "layer1_final_token_mlp_norm_output_before_mlp_projections": normed.reshape(2880).to(torch.float32),
        "layer1_final_token_mlp_router_logits_before_routing": router_logits.reshape(32).to(torch.float32),
        "layer1_final_token_mlp_topk_expert_indices_and_routing_weights": routing_weights.reshape(4).to(torch.float32),
        "layer1_final_token_selected_expert_outputs_before_routing_weighted_sum": selected_outputs.reshape(4, 2880).to(torch.float32),
        "layer1_final_token_mlp_output_after_routing_weighted_sum_before_residual": weighted_sum.reshape(2880).to(torch.float32),
        "layer1_final_token_hidden_state_after_mlp_residual_add": residual.reshape(2880).to(torch.float32),
    }
    metadata = {
        "layer1_attention_residual_mlp_input": layer1_mlp_input[-1].to(torch.float32),
        "selected_expert_indices": [int(v) for v in expert_indices.reshape(-1).tolist()],
        "selected_expert_logits": [float(v) for v in topk.values.reshape(-1).to(torch.float32).tolist()],
        "routing_weights": [float(v) for v in routing_weights.reshape(-1).to(torch.float32).tolist()],
        "routing_weight_sum": float(routing_weights.reshape(-1).to(torch.float32).sum().item()),
        "top_k": int(mlp.experts_per_token),
        "number_of_experts": int(mlp.num_experts),
        "outputs_are_unweighted": True,
        "rank_order": "torch.topk(..., sorted=True), descending selected logit",
        "weighted_sum_formula": "torch.einsum('bec,be->bc', selected_outputs, routing_weights)",
    }
    return local, metadata, model


def boundary_summary(name, boundary, local_tensor, official_tensor, local_metadata):
    metric = finite_compare(local_tensor, official_tensor)
    first = metric["first_differing_flat_index"]
    worst = metric["worst_differing_flat_index"]
    summary = {
        "boundary": name,
        "local_tensor_available": local_tensor is not None,
        "official_tensor_available": official_tensor is not None,
        "shape_match": list(local_tensor.shape) == list(boundary["shape"]),
        "local_tensor_metadata": tensor_meta(local_tensor, boundary),
        "official_tensor_metadata": {
            "shape": boundary.get("shape"),
            "tensor_dtype": boundary.get("tensor_dtype"),
            "serialization_dtype": boundary.get("serialization_format"),
            "layout_interpretation": boundary.get("layout_interpretation"),
        },
        "max_abs_diff": metric["max_abs_diff"],
        "mean_abs_diff": metric["mean_abs_diff"],
        "matched": metric["matched"],
        "mismatching_value_count": metric["mismatching_value_count"],
        "first_differing_index": first | {"logical_index": logical_index(boundary, first["flat_index"])} if first else None,
        "worst_differing_index": worst | {"logical_index": logical_index(boundary, worst["flat_index"])} if worst else None,
    }
    if name == "layer1_final_token_mlp_norm_output_before_mlp_projections":
        summary["parameter_metadata"] = {
            "norm_type": boundary.get("norm_type"),
            "official_norm_parameters": boundary.get("norm_parameters"),
        }
        summary["replay_metric"] = metric
    elif name == "layer1_final_token_mlp_router_logits_before_routing":
        summary["router_weight_metadata"] = boundary.get("router_weight_metadata")
        summary["router_bias_metadata"] = boundary.get("router_bias_metadata")
        summary["local_logits_vs_official_metrics"] = metric
    elif name == "layer1_final_token_mlp_topk_expert_indices_and_routing_weights":
        official_indices = [int(v) for v in boundary["selected_expert_indices"]["values"]]
        local_indices = local_metadata["selected_expert_indices"]
        official_logits = torch.tensor(boundary["selected_expert_logits"]["values"], dtype=torch.float32, device=local_tensor.device)
        local_logits = torch.tensor(local_metadata["selected_expert_logits"], dtype=torch.float32, device=local_tensor.device)
        summary["selected_expert_indices_comparison"] = {
            "local_indices": local_indices,
            "official_indices": official_indices,
            "ordered_match": local_indices == official_indices,
            "set_match_ignoring_order": sorted(local_indices) == sorted(official_indices),
            "first_differing_rank": next((i for i, (a, b) in enumerate(zip(local_indices, official_indices)) if a != b), None),
        }
        summary["selected_expert_logits_metric"] = finite_compare(local_logits, official_logits)
        summary["routing_weights_metric"] = metric
        summary["routing_weight_sum_comparison"] = {
            "local": local_metadata["routing_weight_sum"],
            "official": boundary["routing_weights"].get("sum"),
        }
        summary["matched"] = bool(summary["selected_expert_indices_comparison"]["ordered_match"] and metric["matched"])
    elif name == "layer1_final_token_selected_expert_outputs_before_routing_weighted_sum":
        selected = boundary.get("selected_expert_indices")
        summary["selected_expert_order"] = selected
        summary["outputs_are_unweighted"] = bool(boundary.get("outputs_are_unweighted"))
        summary["routing_weights_already_applied"] = bool(boundary.get("routing_weights_already_applied"))
        summary["per_rank_metrics"] = []
        for rank, expert_idx in enumerate(selected):
            rank_metric = finite_compare(local_tensor[rank], official_tensor[rank])
            summary["per_rank_metrics"].append(
                {
                    "rank": int(rank),
                    "expert_index": int(expert_idx),
                    **rank_metric,
                }
            )
    elif name == "layer1_final_token_mlp_output_after_routing_weighted_sum_before_residual":
        summary["selected_outputs_guard"] = "uses same selected outputs compared in prior ordered boundary"
        summary["routing_weights_guard"] = "uses same routing weights compared in top-k ordered boundary"
        summary["dtype_accumulation_policy_finding"] = "torch/einsum-style local replay"
    elif name == "layer1_final_token_hidden_state_after_mlp_residual_add":
        summary["post_attention_residual_input_guard"] = "layer1 attention residual / MLP input guard"
        summary["weighted_sum_guard"] = "uses same weighted expert sum compared in prior ordered boundary"
    return summary


def focused_trace(boundary_summary):
    worst = boundary_summary.get("worst_differing_index")
    if not worst:
        return None
    diff = float(worst["abs_diff"])
    return {
        "boundary_name": boundary_summary["boundary"],
        "logical_index": worst["logical_index"],
        "local_value": worst["local_value"],
        "official_value": worst["official_value"],
        "abs_diff": diff,
        "one_bf16_ulp_or_larger": bool(diff > 0.0),
        "likely_issue_class": issue_class(boundary_summary["boundary"]),
    }


def main():
    args = parse_args()
    source_attention = load_json(args.source_layer1_attention_bundle)
    official_bundle = load_json(args.official_bundle)
    local_residual_input_artifact = load_json(args.local_residual_input)
    if source_attention.get("classification") != "layer1_attention_ordered_bundle_cleared":
        raise ValueError("source layer1 attention ordered bundle artifact is not cleared")
    if official_bundle.get("classification") != "official_layer1_mlp_ordered_boundary_bundle_captured":
        raise ValueError("official MLP ordered bundle is not the expected PPP capture")
    if official_bundle.get("missing_boundaries"):
        raise ValueError("official MLP ordered bundle has missing boundaries")

    device = torch.device(args.device)
    model = load_local_model(args.model_root, device)
    local_by_name, local_metadata, _model = build_local_boundaries(model, official_bundle["input_token_ids"])
    official_by_name = {b["boundary"]: b for b in official_bundle["boundaries"]}

    official_mlp_input = official_bundle["source_layer1_mlp_input"]
    local_mlp_input = local_metadata["layer1_attention_residual_mlp_input"]
    official_mlp_input_boundary = next(
        b
        for b in source_attention["per_boundary_comparison_table"]
        if b["boundary"] == "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp"
    )
    input_guard_metric = {
        "max_abs_diff": official_mlp_input_boundary["max_abs_diff"],
        "mean_abs_diff": official_mlp_input_boundary["mean_abs_diff"],
        "matched": official_mlp_input_boundary["matched"],
        "mismatching_value_count": official_mlp_input_boundary["mismatching_value_count"],
        "source_boundary_used": "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp",
        "source_digest_match": sha256_f32_le(local_mlp_input) == official_mlp_input.get("sha256_f32_le"),
        "local_sha256_f32_le": sha256_f32_le(local_mlp_input),
        "official_sha256_f32_le": official_mlp_input.get("sha256_f32_le"),
    }
    guard_pass = bool(input_guard_metric["matched"]) and bool(input_guard_metric["source_digest_match"])

    table = []
    earliest = None
    for name in BOUNDARY_ORDER:
        official = official_by_name.get(name)
        local = local_by_name.get(name)
        if official is None or local is None:
            summary = {
                "boundary": name,
                "local_tensor_available": local is not None,
                "official_tensor_available": official is not None,
                "matched": False,
                "shape_match": False,
                "max_abs_diff": None,
                "mean_abs_diff": None,
            }
        else:
            official_tensor = tensor_from_values(official, device)
            summary = boundary_summary(name, official, local, official_tensor, local_metadata)
        table.append(summary)
        if earliest is None and not summary.get("matched", False):
            earliest = name

    if not guard_pass:
        classification = "layer1_mlp_bundle_blocked_by_attention_residual_guard_regression"
        earliest_seam = "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp"
        next_step = "revalidate layer1 attention residual / MLP input guard only"
        trace = None
    elif earliest is None:
        classification = "layer1_mlp_ordered_bundle_cleared"
        earliest_seam = "none"
        next_step = "ask PPP for a coarse layer ladder bundle from layer2 through final layer, or ask PPP for layer2 attention ordered bundle"
        trace = None
    else:
        classification = CLASSIFICATION_BY_BOUNDARY[earliest]
        earliest_seam = earliest
        next_step = NEXT_BY_CLASS[classification]
        trace = focused_trace(next(item for item in table if item["boundary"] == earliest))

    selected_summary = {
        "local_selected_expert_indices": local_metadata["selected_expert_indices"],
        "official_selected_expert_indices": official_bundle.get("selected_expert_indices"),
        "local_routing_weights": local_metadata["routing_weights"],
        "official_routing_weights": official_bundle.get("routing_weights"),
        "local_routing_weight_sum": local_metadata["routing_weight_sum"],
        "rank_order": local_metadata["rank_order"],
    }
    output = {
        "schema_version": "runtime_forward_layer1_mlp_ordered_bundle_compare_status/v1",
        "mode": "layer1-mlp-ordered-bundle-compare-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 1,
            "token_index": 73,
        },
        "source_artifact_paths": {
            "layer1_attention_bundle_compare": str(args.source_layer1_attention_bundle),
            "official_layer1_mlp_ordered_bundle": str(args.official_bundle),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_bundle_path": str(args.official_bundle),
        "layer1_attention_residual_mlp_input_guard_metrics": input_guard_metric,
        "selected_experts_routing_summary": selected_summary,
        "per_boundary_comparison_table": table,
        "per_rank_selected_expert_output_metrics": next(
            (
                item.get("per_rank_metrics", [])
                for item in table
                if item["boundary"] == "layer1_final_token_selected_expert_outputs_before_routing_weighted_sum"
            ),
            [],
        ),
        "earliest_remaining_mismatching_seam": earliest_seam,
        "focused_earliest_mismatch_trace": trace,
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
