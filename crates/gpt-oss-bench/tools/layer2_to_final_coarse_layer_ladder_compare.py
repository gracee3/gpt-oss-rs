#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import struct
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


BOUNDARY_SUFFIX_ORDER = [
    "final_token_layer_input_before_attention_norm",
    "final_token_attention_norm_output_before_qkv",
    "final_token_hidden_state_after_attention_residual_add_before_mlp",
    "final_token_mlp_norm_output_before_mlp_projections",
    "final_token_hidden_state_after_mlp_residual_add",
]

CLASSIFICATION_BY_SUFFIX = {
    "final_token_layer_input_before_attention_norm": "coarse_ladder_layer_input_mismatch",
    "final_token_attention_norm_output_before_qkv": "coarse_ladder_attention_norm_mismatch",
    "final_token_hidden_state_after_attention_residual_add_before_mlp": "coarse_ladder_attention_subpath_or_residual_mismatch",
    "final_token_mlp_norm_output_before_mlp_projections": "coarse_ladder_mlp_norm_mismatch",
    "final_token_hidden_state_after_mlp_residual_add": "coarse_ladder_mlp_subpath_or_residual_mismatch",
}

NEXT_BY_CLASS = {
    "coarse_ladder_layer_input_mismatch": "inspect previous layer output / current layer input handoff only",
    "coarse_ladder_attention_norm_mismatch": "request or compare detailed attention ordered bundle for that layer only",
    "coarse_ladder_attention_subpath_or_residual_mismatch": "request detailed attention ordered bundle for that layer only",
    "coarse_ladder_mlp_norm_mismatch": "request or compare detailed MLP norm boundary for that layer only",
    "coarse_ladder_mlp_subpath_or_residual_mismatch": "request detailed MLP ordered bundle for that layer only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare local layer2..final coarse ladder tensors against the official PPP bundle."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-layer1-mlp-bundle", type=Path, required=True)
    parser.add_argument("--official-ladder", type=Path, required=True)
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


def tensor_from_values(boundary, device):
    values = boundary.get("values")
    if not isinstance(values, list):
        raise ValueError(f"official boundary {boundary.get('boundary')} has no values")
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(boundary["shape"])


def compare_vector(lhs, rhs):
    lhs_f = lhs.reshape(-1).to(torch.float32)
    rhs_f = rhs.reshape(-1).to(torch.float32)
    diff = (lhs_f - rhs_f).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        first_lane = int(mismatch.nonzero(as_tuple=False)[0].item())
        worst_lane = int(diff.argmax().item())
        first = {
            "hidden_lane": first_lane,
            "local_value": float(lhs_f[first_lane].item()),
            "official_value": float(rhs_f[first_lane].item()),
            "abs_diff": float(diff[first_lane].item()),
        }
        worst = {
            "hidden_lane": worst_lane,
            "local_value": float(lhs_f[worst_lane].item()),
            "official_value": float(rhs_f[worst_lane].item()),
            "abs_diff": float(diff[worst_lane].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs_f, rhs_f)),
        "mismatching_value_count": int(mismatch.sum().item()),
        "first_differing_hidden_lane": first,
        "worst_differing_hidden_lane": worst,
    }


def issue_class(boundary_name):
    if boundary_name.endswith("layer_input_before_attention_norm"):
        return "layer input propagation"
    if boundary_name.endswith("attention_norm_output_before_qkv"):
        return "attention norm"
    if boundary_name.endswith("hidden_state_after_attention_residual_add_before_mlp"):
        return "attention subpath / attention residual add"
    if boundary_name.endswith("mlp_norm_output_before_mlp_projections"):
        return "MLP norm"
    if boundary_name.endswith("hidden_state_after_mlp_residual_add"):
        return "MLP/MoE subpath / MLP residual add"
    return "capture/readout"


def load_checkpoint_model_parts(model_root: Path, device):
    torch_mod, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    global torch
    torch = torch_mod
    from gpt_oss.torch.model import MLPBlock, TransformerBlock

    config = base.load_restricted_config(model_root, ModelConfig)
    checkpoint = Checkpoint(str(base.resolve_oracle_checkpoint_dir(model_root)), device)
    return config, checkpoint, AttentionBlock, MLPBlock, TransformerBlock


def copy_named_params(module, checkpoint, prefix):
    for name, param in dict(module.named_parameters()).items():
        param.data.copy_(checkpoint.get(f"{prefix}.{name}"))


def build_layer2_input(model_root: Path, input_token_ids, device):
    config, checkpoint, AttentionBlock, MLPBlock, TransformerBlock = load_checkpoint_model_parts(model_root, device)
    embedding = torch.nn.Embedding(
        config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
    )
    block0 = TransformerBlock(config=config, layer_idx=0, device=device)
    attn1 = AttentionBlock(config=config, layer_idx=1, device=device)
    mlp1 = MLPBlock(config=config, device=device)
    embedding.weight.data.copy_(checkpoint.get("embedding.weight"))
    copy_named_params(block0, checkpoint, "block.0")
    copy_named_params(attn1, checkpoint, "block.1.attn")
    copy_named_params(mlp1, checkpoint, "block.1.mlp")
    embedding.eval()
    block0.eval()
    attn1.eval()
    mlp1.eval()
    with torch.inference_mode():
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=device)
        x = embedding(input_ids)
        x = block0(x)
        x = attn1(x)
        x = mlp1(x)
    del embedding, block0, attn1, mlp1
    gc.collect()
    return x.detach().contiguous(), config, checkpoint, TransformerBlock


def compute_layer_boundaries(x, layer_idx, config, checkpoint, TransformerBlock, device):
    block = TransformerBlock(config=config, layer_idx=layer_idx, device=device)
    copy_named_params(block, checkpoint, f"block.{layer_idx}")
    block.eval()
    with torch.inference_mode():
        layer_input = x
        attn_norm = block.attn.norm(layer_input)
        after_attn = block.attn(layer_input)
        mlp_norm = block.mlp.norm(after_attn)
        after_mlp = block.mlp(after_attn)
    local = {
        f"layer{layer_idx}_final_token_layer_input_before_attention_norm": layer_input[-1].to(torch.float32).contiguous(),
        f"layer{layer_idx}_final_token_attention_norm_output_before_qkv": attn_norm[-1].to(torch.float32).contiguous(),
        f"layer{layer_idx}_final_token_hidden_state_after_attention_residual_add_before_mlp": after_attn[-1].to(torch.float32).contiguous(),
        f"layer{layer_idx}_final_token_mlp_norm_output_before_mlp_projections": mlp_norm[-1].to(torch.float32).contiguous(),
        f"layer{layer_idx}_final_token_hidden_state_after_mlp_residual_add": after_mlp[-1].to(torch.float32).contiguous(),
    }
    next_x = after_mlp.detach().contiguous()
    del block, attn_norm, after_attn, mlp_norm, after_mlp
    gc.collect()
    return local, next_x


def official_by_layer(official_ladder):
    by_layer = {}
    for layer in official_ladder["layers"]:
        by_layer[int(layer["layer_index"])] = {
            b["boundary"]: b for b in layer["boundaries"]
        }
    return by_layer


def boundary_summary(layer_idx, boundary_name, local_tensor, official_boundary, official_tensor):
    metric = compare_vector(local_tensor, official_tensor)
    local_digest = sha256_f32_le(local_tensor)
    official_digest = official_boundary.get("finite_value_summary", {}).get("sha256_f32_le")
    return {
        "layer_index": int(layer_idx),
        "boundary": boundary_name,
        "local_tensor_available": local_tensor is not None,
        "official_tensor_available": official_boundary is not None,
        "shape_match": list(local_tensor.shape) == list(official_boundary["shape"]),
        "local_tensor_metadata": {
            "shape": list(local_tensor.shape),
            "dtype": str(local_tensor.dtype).replace("torch.", ""),
            "serialization_dtype": "f32-expanded BF16 values",
        },
        "official_tensor_metadata": {
            "shape": official_boundary.get("shape"),
            "tensor_dtype": official_boundary.get("tensor_dtype"),
            "serialization_dtype": official_boundary.get("serialization_format"),
        },
        "max_abs_diff": metric["max_abs_diff"],
        "mean_abs_diff": metric["mean_abs_diff"],
        "matched": metric["matched"],
        "mismatching_value_count": metric["mismatching_value_count"],
        "local_sha256_f32_le": local_digest,
        "official_sha256_f32_le": official_digest,
        "digest_matched": local_digest == official_digest,
        "first_differing_hidden_lane": metric["first_differing_hidden_lane"],
        "worst_differing_hidden_lane": metric["worst_differing_hidden_lane"],
    }


def layer_summary(layer_idx, entries):
    by_suffix = {}
    for entry in entries:
        suffix = entry["boundary"].removeprefix(f"layer{layer_idx}_")
        by_suffix[suffix] = entry
    first_mismatch = next((entry["boundary"] for entry in entries if not entry["matched"]), None)
    worst = max((entry["max_abs_diff"] for entry in entries), default=0.0)
    return {
        "layer_index": int(layer_idx),
        "input_matched": by_suffix["final_token_layer_input_before_attention_norm"]["matched"],
        "attention_norm_matched": by_suffix["final_token_attention_norm_output_before_qkv"]["matched"],
        "after_attention_residual_matched": by_suffix[
            "final_token_hidden_state_after_attention_residual_add_before_mlp"
        ]["matched"],
        "mlp_norm_matched": by_suffix["final_token_mlp_norm_output_before_mlp_projections"]["matched"],
        "after_mlp_residual_matched": by_suffix["final_token_hidden_state_after_mlp_residual_add"]["matched"],
        "worst_max_abs_diff_in_layer": float(worst),
        "first_mismatching_boundary": first_mismatch,
    }


def focused_trace(entry):
    worst = entry.get("worst_differing_hidden_lane")
    if not worst:
        return None
    diff = float(worst["abs_diff"])
    return {
        "layer_index": entry["layer_index"],
        "boundary_name": entry["boundary"],
        "hidden_lane": worst["hidden_lane"],
        "local_value": worst["local_value"],
        "official_value": worst["official_value"],
        "abs_diff": diff,
        "one_bf16_ulp_or_larger": bool(diff > 0.0),
        "likely_issue_class": issue_class(entry["boundary"]),
    }


def main():
    args = parse_args()
    source_layer1 = load_json(args.source_layer1_mlp_bundle)
    official_ladder = load_json(args.official_ladder)
    local_residual_input_artifact = load_json(args.local_residual_input)
    if source_layer1.get("classification") != "layer1_mlp_ordered_bundle_cleared":
        raise ValueError("source layer1 MLP bundle artifact is not cleared")
    if official_ladder.get("classification") != "official_layer2_to_final_coarse_layer_ladder_bundle_captured":
        raise ValueError("official coarse ladder is not the expected PPP capture")
    if official_ladder.get("missing_boundaries"):
        raise ValueError("official coarse ladder has missing boundaries")

    device = torch.device(args.device)
    by_layer = official_by_layer(official_ladder)
    layer_start, layer_end = [int(v) for v in official_ladder["captured_layer_range"]]
    x, config, checkpoint, TransformerBlock = build_layer2_input(
        args.model_root, official_ladder["input_token_ids"], device
    )

    official_layer2_input = by_layer[layer_start][
        f"layer{layer_start}_final_token_layer_input_before_attention_norm"
    ]
    layer2_input_tensor = tensor_from_values(official_layer2_input, device)
    preflight_metric = compare_vector(x[-1].to(torch.float32), layer2_input_tensor)
    preflight_local_digest = sha256_f32_le(x[-1].to(torch.float32))
    preflight_official_digest = official_layer2_input["finite_value_summary"]["sha256_f32_le"]
    preflight_guard = {
        "source_boundary_used": "layer1_final_token_hidden_state_after_mlp_residual_add -> layer2_final_token_layer_input_before_attention_norm",
        "max_abs_diff": preflight_metric["max_abs_diff"],
        "mean_abs_diff": preflight_metric["mean_abs_diff"],
        "matched": preflight_metric["matched"],
        "mismatching_value_count": preflight_metric["mismatching_value_count"],
        "local_sha256_f32_le": preflight_local_digest,
        "official_sha256_f32_le": preflight_official_digest,
        "digest_matched": preflight_local_digest == preflight_official_digest,
    }

    table = []
    per_layer = []
    earliest_entry = None
    current = x
    for layer_idx in range(layer_start, layer_end + 1):
        local_boundaries, current = compute_layer_boundaries(
            current, layer_idx, config, checkpoint, TransformerBlock, device
        )
        layer_entries = []
        for suffix in BOUNDARY_SUFFIX_ORDER:
            name = f"layer{layer_idx}_{suffix}"
            official = by_layer[layer_idx].get(name)
            local = local_boundaries.get(name)
            if official is None or local is None:
                entry = {
                    "layer_index": int(layer_idx),
                    "boundary": name,
                    "local_tensor_available": local is not None,
                    "official_tensor_available": official is not None,
                    "shape_match": False,
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "matched": False,
                }
            else:
                official_tensor = tensor_from_values(official, device)
                entry = boundary_summary(layer_idx, name, local, official, official_tensor)
            table.append(entry)
            layer_entries.append(entry)
            if earliest_entry is None and not entry.get("matched", False):
                earliest_entry = entry
        per_layer.append(layer_summary(layer_idx, layer_entries))

    terminal_entry = next(
        entry
        for entry in table
        if entry["boundary"] == f"layer{layer_end}_final_token_hidden_state_after_mlp_residual_add"
    )
    if not preflight_guard["matched"] or not preflight_guard["digest_matched"]:
        classification = "layer2_to_final_ladder_blocked_by_layer1_output_guard_regression"
        earliest_layer = layer_start
        earliest_seam = f"layer{layer_start}_final_token_layer_input_before_attention_norm"
        next_step = "revalidate layer1 output / layer2 input guard only"
        trace = focused_trace(terminal_entry) if not preflight_guard["matched"] else None
    elif earliest_entry is None:
        classification = "layer2_to_final_coarse_layer_ladder_cleared"
        earliest_layer = None
        earliest_seam = "none"
        next_step = "ask PPP for final norm / logits / LM head boundary bundle"
        trace = None
    else:
        suffix = earliest_entry["boundary"].removeprefix(f"layer{earliest_entry['layer_index']}_")
        classification = CLASSIFICATION_BY_SUFFIX[suffix]
        earliest_layer = earliest_entry["layer_index"]
        earliest_seam = earliest_entry["boundary"]
        next_step = NEXT_BY_CLASS[classification]
        trace = focused_trace(earliest_entry)

    output = {
        "schema_version": "runtime_forward_layer2_to_final_coarse_layer_ladder_compare_status/v1",
        "mode": "layer2-to-final-coarse-layer-ladder-compare-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "token_index": 73,
        },
        "source_artifact_paths": {
            "layer1_mlp_bundle_compare": str(args.source_layer1_mlp_bundle),
            "official_layer2_to_final_coarse_ladder": str(args.official_ladder),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_ladder_path": str(args.official_ladder),
        "layer_count": int(official_ladder["layer_count"]),
        "compared_layer_range": [layer_start, layer_end],
        "preflight_layer1_output_layer2_input_guard_metrics": preflight_guard,
        "per_boundary_comparison_table": table,
        "per_layer_summary_table": per_layer,
        "terminal_layer23_final_output_summary": {
            "boundary": terminal_entry["boundary"],
            "max_abs_diff": terminal_entry["max_abs_diff"],
            "mean_abs_diff": terminal_entry["mean_abs_diff"],
            "matched": terminal_entry["matched"],
            "local_sha256_f32_le": terminal_entry["local_sha256_f32_le"],
            "official_sha256_f32_le": terminal_entry["official_sha256_f32_le"],
            "digest_matched": terminal_entry["digest_matched"],
        },
        "earliest_mismatching_layer": earliest_layer,
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
