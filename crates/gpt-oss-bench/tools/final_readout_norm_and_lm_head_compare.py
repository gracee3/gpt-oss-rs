#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

import layer2_to_final_coarse_layer_ladder_compare as ladder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare local final readout norm and LM head logits against the official PPP bundle."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-ladder", type=Path, required=True)
    parser.add_argument("--official-readout", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def boundary_by_name(bundle):
    return {boundary["boundary"]: boundary for boundary in bundle["boundaries"]}


def norm_replay(input_bf16, scale, eps):
    x = input_bf16.to(torch.float32)
    t = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + float(eps))
    return (t * scale.to(torch.float32)).to(torch.bfloat16).to(torch.float32)


def load_final_modules(config, checkpoint, device):
    torch_mod, _AttentionBlock, _ModelConfig, _Checkpoint = ladder.base.import_gpt_oss()
    global torch
    torch = torch_mod
    from gpt_oss.torch.model import RMSNorm

    final_norm = RMSNorm(config.hidden_size, device=device)
    unembedding = torch.nn.Linear(
        config.hidden_size,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    )
    for name, param in dict(final_norm.named_parameters()).items():
        param.data.copy_(checkpoint.get(f"norm.{name}"))
    unembedding.weight.data.copy_(checkpoint.get("unembedding.weight"))
    final_norm.eval()
    unembedding.eval()
    return final_norm, unembedding


def top20_summary(logits):
    values, indices = torch.topk(logits.reshape(-1).to(torch.float32), k=20, dim=0, sorted=True)
    return {
        "token_ids": [int(v) for v in indices.tolist()],
        "values": [float(v) for v in values.tolist()],
    }


def top20_compare(local_logits, official_top):
    official_ids = [int(v) for v in official_top["token_ids"]]
    official_values = torch.tensor(official_top["logit_values"], dtype=torch.float32, device=local_logits.device)
    local_values_at_official = local_logits[official_ids].reshape(-1).to(torch.float32)
    metric = ladder.compare_vector(local_values_at_official, official_values)
    local_top = top20_summary(local_logits)
    return {
        "official_top_token_ids": official_ids,
        "official_top_values": [float(v) for v in official_top["logit_values"]],
        "local_top_token_ids": local_top["token_ids"],
        "local_top_values": local_top["values"],
        "ordered_top20_token_ids_match": local_top["token_ids"] == official_ids,
        "top20_set_match": sorted(local_top["token_ids"]) == sorted(official_ids),
        "official_top20_token_value_metric": metric,
    }


def focused_norm_trace(metric, final_hidden, final_norm_output, official_norm, replay, scale):
    worst = metric.get("worst_differing_hidden_lane")
    if not worst:
        return None
    lane = int(worst["hidden_lane"])
    return {
        "hidden_lane": lane,
        "final_block_output_value": float(final_hidden[lane].item()),
        "norm_weight_value": float(scale[lane].item()),
        "local_final_norm_value": float(final_norm_output[lane].item()),
        "official_final_norm_value": float(official_norm[lane].item()),
        "replay_value": float(replay[lane].item()),
        "first_divergent_stage": "final norm dtype/policy",
    }


def focused_logits_trace(metric, local_logits, official_logits, official_top_ids):
    worst = metric.get("worst_differing_hidden_lane")
    if not worst:
        return None
    token_id = int(worst["hidden_lane"])
    diff = float(worst["abs_diff"])
    return {
        "token_id": token_id,
        "local_logit": float(local_logits[token_id].item()),
        "official_logit": float(official_logits[token_id].item()),
        "abs_diff": diff,
        "token_is_in_official_top20": token_id in set(official_top_ids),
        "one_bf16_ulp_or_larger": bool(diff > 0.0),
        "likely_issue_class": "LM head weight/layout or arithmetic/accumulation",
    }


def main():
    args = parse_args()
    source_ladder = load_json(args.source_ladder)
    official_bundle = load_json(args.official_readout)
    local_residual_input_artifact = load_json(args.local_residual_input)
    if source_ladder.get("classification") != "layer2_to_final_coarse_layer_ladder_cleared":
        raise ValueError("source coarse ladder artifact is not cleared")
    if official_bundle.get("classification") != "official_final_token_readout_norm_and_lm_head_bundle_captured":
        raise ValueError("official final readout bundle is not the expected PPP capture")
    if official_bundle.get("missing_boundaries"):
        raise ValueError("official final readout bundle has missing boundaries")

    device = torch.device(args.device)
    by_name = boundary_by_name(official_bundle)
    final_hidden_official = ladder.tensor_from_values(
        by_name["final_token_hidden_state_after_final_transformer_block"], device
    )
    final_norm_official = ladder.tensor_from_values(
        by_name["final_token_final_norm_output_before_lm_head"], device
    )
    logits_official = ladder.tensor_from_values(by_name["final_token_lm_head_logits"], device)

    x, config, checkpoint, TransformerBlock = ladder.build_layer2_input(
        args.model_root, official_bundle["input_token_ids"], device
    )
    current = x
    for layer_idx in range(2, 24):
        _local_boundaries, current = ladder.compute_layer_boundaries(
            current, layer_idx, config, checkpoint, TransformerBlock, device
        )
    final_hidden = current[-1].to(torch.float32).contiguous()

    final_norm, unembedding = load_final_modules(config, checkpoint, device)
    with torch.inference_mode():
        final_norm_output_bf16 = final_norm(current[-1:])
        final_norm_output = final_norm_output_bf16.reshape(2880).to(torch.float32).contiguous()
        logits = unembedding(final_norm_output_bf16).reshape(-1).to(torch.float32).contiguous()
        norm_replayed = norm_replay(current[-1:], final_norm.scale, final_norm.eps).reshape(2880)

    final_hidden_metric = ladder.compare_vector(final_hidden, final_hidden_official)
    final_norm_metric = ladder.compare_vector(final_norm_output, final_norm_official)
    norm_replay_metric = ladder.compare_vector(norm_replayed, final_norm_official)
    logits_metric = ladder.compare_vector(logits, logits_official)
    top20 = top20_compare(logits, official_bundle["top_logits_summary"])

    final_hidden_digest = ladder.sha256_f32_le(final_hidden)
    final_norm_digest = ladder.sha256_f32_le(final_norm_output)
    logits_digest = ladder.sha256_f32_le(logits)
    official_final_hidden_digest = by_name["final_token_hidden_state_after_final_transformer_block"]["finite_value_summary"]["sha256_f32_le"]
    official_final_norm_digest = by_name["final_token_final_norm_output_before_lm_head"]["finite_value_summary"]["sha256_f32_le"]
    official_logits_digest = by_name["final_token_lm_head_logits"]["finite_value_summary"]["sha256_f32_le"]

    if not final_hidden_metric["matched"] or final_hidden_digest != official_final_hidden_digest:
        classification = "final_readout_blocked_by_final_transformer_output_regression"
        earliest = "final_token_hidden_state_after_final_transformer_block"
        next_step = "revalidate final transformer block output guard only"
        trace = None
    elif not final_norm_metric["matched"]:
        classification = "final_norm_mismatch_after_transformer_stack_clear"
        earliest = "final_token_final_norm_output_before_lm_head"
        next_step = "inspect final RMSNorm dtype/kernel policy only"
        trace = focused_norm_trace(
            final_norm_metric, final_hidden, final_norm_output, final_norm_official, norm_replayed, final_norm.scale
        )
    elif list(logits.shape) != list(logits_official.shape):
        classification = "lm_head_logits_shape_or_layout_mismatch"
        earliest = "final_token_lm_head_logits"
        next_step = "inspect LM head weight/layout/arithmetic only"
        trace = None
    elif not logits_metric["matched"]:
        classification = "lm_head_logits_mismatch_after_final_norm_clear"
        earliest = "final_token_lm_head_logits"
        next_step = "inspect LM head weight/layout/arithmetic only"
        trace = focused_logits_trace(
            logits_metric, logits, logits_official, official_bundle["top_logits_summary"]["token_ids"]
        )
    else:
        classification = "final_readout_norm_and_lm_head_logits_cleared"
        earliest = "none"
        next_step = "record full final-token oracle parity milestone and decide whether to commit/push before promotion or 4097-boundary work"
        trace = None

    output = {
        "schema_version": "runtime_forward_final_readout_norm_and_lm_head_compare_status/v1",
        "mode": "final-readout-norm-and-lm-head-compare-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "token_index": 73,
        },
        "source_artifact_paths": {
            "coarse_layer_ladder": str(args.source_ladder),
            "official_final_readout_bundle": str(args.official_readout),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_bundle_path": str(args.official_readout),
        "final_block_output_guard_metrics": {
            **final_hidden_metric,
            "local_sha256_f32_le": final_hidden_digest,
            "official_sha256_f32_le": official_final_hidden_digest,
            "digest_matched": final_hidden_digest == official_final_hidden_digest,
        },
        "final_norm_metadata": {
            "module_path": "model.norm",
            "norm_type": "RMSNorm",
            "weight_shape": list(final_norm.scale.shape),
            "weight_dtype": str(final_norm.scale.dtype).replace("torch.", ""),
            "epsilon": float(final_norm.eps),
            "bias_presence": False,
            "input_dtype": "torch.bfloat16",
            "accumulation_dtype": "torch.float32",
            "output_dtype_before_serialization": "torch.bfloat16",
            "bf16_output_cast_rounding_point": "after RMSNorm scale multiply",
        },
        "final_norm_comparison_metrics": {
            **final_norm_metric,
            "local_sha256_f32_le": final_norm_digest,
            "official_sha256_f32_le": official_final_norm_digest,
            "digest_matched": final_norm_digest == official_final_norm_digest,
            "official_semantics_replay_vs_official": norm_replay_metric,
        },
        "lm_head_metadata": {
            "module_path": "model.unembedding",
            "weight_shape": list(unembedding.weight.shape),
            "weight_dtype": str(unembedding.weight.dtype).replace("torch.", ""),
            "orientation_layout": "[vocab_size, hidden_size]; Linear computes input @ weight.T",
            "weights_tied_with_embedding": False,
            "bias_presence": False,
            "input_dtype": "torch.bfloat16",
            "weight_dtype_for_compute": "torch.bfloat16",
            "accumulation_dtype": "unknown",
            "output_dtype_before_serialization": "torch.bfloat16",
            "vocab_size": int(config.vocab_size),
        },
        "logits_comparison_metrics": {
            **logits_metric,
            "shape": list(logits.shape),
            "vocab_size": int(logits.numel()),
            "dtype": "float32",
            "serialization_dtype": "f32-expanded BF16 values",
            "local_sha256_f32_le": logits_digest,
            "official_sha256_f32_le": official_logits_digest,
            "digest_matched": logits_digest == official_logits_digest,
        },
        "top20_logits_comparison": top20,
        "earliest_remaining_mismatching_seam": earliest,
        "focused_mismatch_trace": trace,
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
