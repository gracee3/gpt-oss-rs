#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


BOUNDARY_ORDER = [
    "layer1_final_token_q_projection_output_before_rope",
    "layer1_final_token_k_projection_output_before_rope",
    "layer1_final_token_v_projection_output_before_attention",
    "layer1_final_token_q_post_rope_before_attention",
    "layer1_grouped_k_post_rope_before_attention",
    "layer1_final_token_raw_scaled_qk_logits_pre_mask",
    "layer1_final_token_masked_scaled_qk_logits_pre_softmax",
    "layer1_final_token_attention_probs_post_softmax",
    "layer1_final_token_attention_weighted_value_sum_before_output_projection",
    "layer1_final_token_attention_output_after_o_proj_before_residual",
    "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp",
]

CLASSIFICATION_BY_BOUNDARY = {
    "layer1_final_token_q_projection_output_before_rope": "layer1_q_projection_mismatch_before_rope",
    "layer1_final_token_k_projection_output_before_rope": "layer1_k_projection_mismatch_before_rope",
    "layer1_final_token_v_projection_output_before_attention": "layer1_v_projection_mismatch_before_attention",
    "layer1_final_token_q_post_rope_before_attention": "layer1_rope_path_mismatch_after_qk_projection_clear",
    "layer1_grouped_k_post_rope_before_attention": "layer1_rope_path_mismatch_after_qk_projection_clear",
    "layer1_final_token_raw_scaled_qk_logits_pre_mask": "layer1_qk_score_arithmetic_mismatch_after_qk_provenance_clear",
    "layer1_final_token_masked_scaled_qk_logits_pre_softmax": "layer1_attention_mask_or_sink_mismatch_after_raw_qk_clear",
    "layer1_final_token_attention_probs_post_softmax": "layer1_attention_softmax_policy_mismatch_after_logits_clear",
    "layer1_final_token_attention_weighted_value_sum_before_output_projection": "layer1_attention_weighted_v_sum_mismatch_after_probs_clear",
    "layer1_final_token_attention_output_after_o_proj_before_residual": "layer1_attention_o_proj_mismatch_after_weighted_v_clear",
    "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp": "layer1_attention_residual_add_mismatch_after_o_proj_clear",
}

NEXT_BY_CLASS = {
    "layer1_q_projection_mismatch_before_rope": "build/prove scoped layer1 oneDNN Q/K/V projection candidates analogous to layer0, then rerun this bundle compare",
    "layer1_k_projection_mismatch_before_rope": "build/prove scoped layer1 oneDNN Q/K/V projection candidates analogous to layer0, then rerun this bundle compare",
    "layer1_v_projection_mismatch_before_attention": "build/prove scoped layer1 oneDNN Q/K/V projection candidates analogous to layer0, then rerun this bundle compare",
    "layer1_rope_path_mismatch_after_qk_projection_clear": "inspect layer1 Q/K RoPE path only",
    "layer1_qk_score_arithmetic_mismatch_after_qk_provenance_clear": "inspect layer1 QK score arithmetic only",
    "layer1_attention_mask_or_sink_mismatch_after_raw_qk_clear": "inspect layer1 final-token mask/sink construction only",
    "layer1_attention_softmax_policy_mismatch_after_logits_clear": "inspect layer1 attention softmax dtype/output policy only",
    "layer1_attention_weighted_v_sum_mismatch_after_probs_clear": "inspect layer1 V provenance/GQA/weighted-sum only",
    "layer1_attention_o_proj_mismatch_after_weighted_v_clear": "inspect layer1 attention o_proj weight/bias/arithmetic only",
    "layer1_attention_residual_add_mismatch_after_o_proj_clear": "inspect layer1 attention residual add dtype policy only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare local layer1 attention tensors against the official ordered PPP bundle."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-layer1-norm", type=Path, required=True)
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
        finite_diff[~finite] = torch.where(unequal[~finite], torch.tensor(float("inf")), torch.tensor(0.0))
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
    shape = boundary["shape"]
    idx = unravel(flat_index, shape)
    if name in {
        "layer1_final_token_q_projection_output_before_rope",
        "layer1_final_token_q_post_rope_before_attention",
    }:
        feature = idx[0]
        q_head = feature // 64
        return {
            "feature": feature,
            "q_head": q_head,
            "kv_head": q_head // 8,
            "heads_per_kv_index": q_head % 8,
            "lane": feature % 64,
        }
    if name in {
        "layer1_final_token_k_projection_output_before_rope",
        "layer1_final_token_v_projection_output_before_attention",
    }:
        feature = idx[0]
        return {"feature": feature, "kv_head": feature // 64, "lane": feature % 64}
    if name == "layer1_grouped_k_post_rope_before_attention":
        return {"token": idx[0], "kv_head": idx[1], "lane": idx[2]}
    if name in {
        "layer1_final_token_raw_scaled_qk_logits_pre_mask",
        "layer1_final_token_masked_scaled_qk_logits_pre_softmax",
        "layer1_final_token_attention_probs_post_softmax",
    }:
        return {"q_head": idx[0], "key_position_or_sink": idx[1]}
    if name == "layer1_final_token_attention_weighted_value_sum_before_output_projection":
        feature = idx[0]
        return {"feature": feature, "q_head": feature // 64, "lane": feature % 64}
    return {"hidden_lane": idx[0]}


def tensor_meta(tensor, boundary):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": "f32-expanded BF16 values",
        "layout_interpretation": boundary.get("layout_interpretation"),
    }


def row_sum_summary(probs):
    sums = probs.to(torch.float32).sum(dim=-1)
    return {
        "min": float(sums.min().item()),
        "max": float(sums.max().item()),
        "mean": float(sums.mean().item()),
        "max_abs_row_sum_minus_1": float((sums - 1.0).abs().max().item()),
    }


def compare_slice(local, official, slicer):
    return finite_compare(local[slicer], official[slicer])


def load_local_model(model_root: Path, device):
    torch_mod, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    global torch
    torch = torch_mod
    from gpt_oss.torch.model import TransformerBlock

    class Layer1AttentionReplay(torch.nn.Module):
        def __init__(self, config, device):
            super().__init__()
            self.embedding = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
            )
            self.block0 = TransformerBlock(config=config, layer_idx=0, device=device)
            self.attn1 = AttentionBlock(config=config, layer_idx=1, device=device)

    config = base.load_restricted_config(model_root, ModelConfig)
    model = Layer1AttentionReplay(config=config, device=device)
    model.eval()
    checkpoint = Checkpoint(str(base.resolve_oracle_checkpoint_dir(model_root)), device)
    params = dict(model.named_parameters())
    for name, param in params.items():
        if name == "embedding.weight":
            param.data.copy_(checkpoint.get("embedding.weight"))
        elif name.startswith("block0."):
            param.data.copy_(checkpoint.get(f"block.0.{name.removeprefix('block0.')}"))
        elif name.startswith("attn1."):
            param.data.copy_(checkpoint.get(f"block.1.attn.{name.removeprefix('attn1.')}"))
        else:
            raise ValueError(f"unexpected parameter name {name}")
    return model


def build_local_boundaries(model, input_token_ids):
    device = model.embedding.weight.device
    input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        x0 = model.embedding(input_ids)
        layer0_out = model.block0(x0)
        attn = model.attn1
        normed = attn.norm(layer0_out)
        qkv = attn.qkv(normed)
        q_dim = int(attn.num_attention_heads * attn.head_dim)
        kv_dim = int(attn.num_key_value_heads * attn.head_dim)
        q = qkv[:, :q_dim].contiguous()
        k = qkv[:, q_dim : q_dim + kv_dim].contiguous()
        v = qkv[:, q_dim + kv_dim : q_dim + 2 * kv_dim].contiguous()
        q_heads = q.view(-1, attn.num_key_value_heads, attn.num_attention_heads // attn.num_key_value_heads, attn.head_dim)
        k_heads = k.view(-1, attn.num_key_value_heads, attn.head_dim)
        v_heads = v.view(-1, attn.num_key_value_heads, attn.head_dim)
        q_rope, k_rope = attn.rope(q_heads, k_heads)
        n_tokens = q_rope.shape[0]
        q_mult = attn.num_attention_heads // attn.num_key_value_heads
        k_expanded = k_rope[:, :, None, :].expand(-1, -1, q_mult, -1)
        v_expanded = v_heads[:, :, None, :].expand(-1, -1, q_mult, -1)
        sinks = attn.sinks.reshape(attn.num_key_value_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
        mask = torch.triu(q_rope.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
        if int(attn.sliding_window) > 0:
            mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-int(attn.sliding_window))
        qk = torch.einsum("qhmd,khmd->hmqk", q_rope, k_expanded)
        qk *= attn.sm_scale
        qk_pre_mask = qk[:, :, -1, :].reshape(64, n_tokens).contiguous()
        qk_masked_real = qk + mask[None, None, :, :]
        qk_with_sink = torch.cat([qk_masked_real, sinks], dim=-1)
        final_masked = qk_with_sink[:, :, -1, :].reshape(64, n_tokens + 1).contiguous()
        probs = torch.softmax(qk_with_sink, dim=-1)
        final_probs = probs[:, :, -1, :].reshape(64, n_tokens + 1).contiguous()
        weighted_v = torch.einsum("hmk,khmd->hmd", probs[:, :, -1, :-1], v_expanded).reshape(4096).contiguous()
        o_proj = attn.out(weighted_v.reshape(1, -1)).reshape(2880).contiguous()
        residual = (layer0_out[-1] + o_proj).reshape(2880).contiguous()
    return {
        "layer1_final_token_q_projection_output_before_rope": q[-1].to(torch.float32),
        "layer1_final_token_k_projection_output_before_rope": k[-1].to(torch.float32),
        "layer1_final_token_v_projection_output_before_attention": v[-1].to(torch.float32),
        "layer1_final_token_q_post_rope_before_attention": q_rope[-1].reshape(4096).to(torch.float32),
        "layer1_grouped_k_post_rope_before_attention": k_rope.to(torch.float32),
        "layer1_final_token_raw_scaled_qk_logits_pre_mask": qk_pre_mask.to(torch.float32),
        "layer1_final_token_masked_scaled_qk_logits_pre_softmax": final_masked.to(torch.float32),
        "layer1_final_token_attention_probs_post_softmax": final_probs.to(torch.float32),
        "layer1_final_token_attention_weighted_value_sum_before_output_projection": weighted_v.to(torch.float32),
        "layer1_final_token_attention_output_after_o_proj_before_residual": o_proj.to(torch.float32),
        "layer1_final_token_hidden_state_after_attention_residual_add_before_mlp": residual.to(torch.float32),
    }, {
        "token_count": int(n_tokens),
        "query_heads": int(attn.num_attention_heads),
        "kv_heads": int(attn.num_key_value_heads),
        "heads_per_kv": int(q_mult),
        "head_dim": int(attn.head_dim),
        "scale": float(attn.sm_scale),
        "sliding_window": int(attn.sliding_window),
        "sink_position": int(n_tokens),
        "sink_probability_dropped_before_v_sum": True,
        "gqa_mapping": "kv_head = q_head // heads_per_kv",
    }


def boundary_summary(name, boundary, local_tensor, official_tensor):
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
    if name in {
        "layer1_final_token_masked_scaled_qk_logits_pre_softmax",
        "layer1_final_token_attention_probs_post_softmax",
    }:
        real = (slice(None), slice(0, 74))
        sink = (slice(None), slice(74, 75))
        summary["real_key_metric"] = compare_slice(local_tensor, official_tensor, real)
        summary["sink_column_metric"] = compare_slice(local_tensor, official_tensor, sink)
        summary["all_position_metric"] = metric
        if name == "layer1_final_token_attention_probs_post_softmax":
            summary["row_sum_summary_after_bf16_serialization"] = {
                "local": row_sum_summary(local_tensor),
                "official": boundary.get("probability_row_sum_summary_after_bf16_serialization"),
            }
    if name == "layer1_final_token_attention_weighted_value_sum_before_output_projection":
        summary["sink_probability_dropped_before_v_sum"] = True
        summary["gqa_mapping_summary"] = "kv_head = q_head // heads_per_kv; heads_per_kv = 8"
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


def issue_class(boundary_name):
    if "q_projection" in boundary_name or "k_projection" in boundary_name or "v_projection" in boundary_name:
        return "Q/K/V projection policy"
    if "post_rope" in boundary_name:
        return "RoPE"
    if "masked_scaled" in boundary_name:
        return "mask/sink layout"
    if "attention_probs" in boundary_name:
        return "softmax dtype/output"
    if "weighted_value" in boundary_name:
        return "weighted V / GQA"
    if "o_proj" in boundary_name:
        return "o_proj"
    if "residual_add" in boundary_name:
        return "residual add"
    return "capture/readout"


def main():
    args = parse_args()
    source_norm = load_json(args.source_layer1_norm)
    official_bundle = load_json(args.official_bundle)
    local_residual_input_artifact = load_json(args.local_residual_input)
    if source_norm.get("classification") != "layer1_attn_norm_before_qkv_cleared_after_layer0":
        raise ValueError("source layer1 attention norm artifact is not cleared")
    if official_bundle.get("classification") != "official_layer1_attention_ordered_boundary_bundle_captured":
        raise ValueError("official ordered bundle is not the expected PPP capture")
    if official_bundle.get("missing_boundaries"):
        raise ValueError("official ordered bundle has missing boundaries")

    device = torch.device(args.device)
    model = load_local_model(args.model_root, device)
    local_by_name, local_attention_metadata = build_local_boundaries(model, official_bundle["input_token_ids"])
    official_by_name = {b["boundary"]: b for b in official_bundle["boundaries"]}

    norm_metric = source_norm["local_runtime_layer1_norm_vs_official_metrics"]
    input_guard = source_norm["layer0_output_input_guard_metrics"]
    guards_pass = bool(input_guard["metrics"]["matched"]) and bool(
        input_guard["source_input_digest_matches_official_layer1_source"]
    ) and bool(norm_metric["matched"])

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
            summary = boundary_summary(name, official, local, official_tensor)
        table.append(summary)
        if earliest is None and not summary.get("matched", False):
            earliest = name

    if not guards_pass:
        classification = "layer1_attention_bundle_blocked_by_input_or_norm_guard_regression"
        earliest_seam = "layer1 input or attention norm guard"
        next_step = "revalidate layer1 input and attention norm guard only"
        trace = None
    elif earliest is None:
        classification = "layer1_attention_ordered_bundle_cleared"
        earliest_seam = "none"
        next_step = "ask PPP for a bounded layer1 MLP ordered boundary bundle"
        trace = None
    else:
        classification = CLASSIFICATION_BY_BOUNDARY[earliest]
        earliest_seam = earliest
        next_step = NEXT_BY_CLASS[classification]
        trace = focused_trace(next(item for item in table if item["boundary"] == earliest))

    output = {
        "schema_version": "runtime_forward_layer1_attention_ordered_bundle_compare_status/v1",
        "mode": "layer1-attention-ordered-bundle-compare-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 1,
            "token_index": 73,
        },
        "source_artifact_paths": {
            "layer1_attention_norm_cleared": str(args.source_layer1_norm),
            "official_layer1_attention_ordered_bundle": str(args.official_bundle),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_bundle_path": str(args.official_bundle),
        "layer1_input_guard_metrics": input_guard,
        "layer1_attention_norm_guard_metrics": norm_metric,
        "local_attention_metadata": local_attention_metadata,
        "official_bundle_metadata": official_bundle.get("layer1_attention_metadata"),
        "per_boundary_comparison_table": table,
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
