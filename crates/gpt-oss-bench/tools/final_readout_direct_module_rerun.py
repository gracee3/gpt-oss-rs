#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

import final_readout_norm_and_lm_head_compare as readout
import layer2_to_final_coarse_layer_ladder_compare as ladder


PRIOR_STALE_LOGITS_DIGEST = "5a7d47edfab63d59c17825b8d7b7668cc7a15ad2d107f902ca2caa05488ecd44"
DIRECT_MODULE_LOGITS_DIGEST = "67f31845dd24db26cc91954607cfae8ae7ff7b9c8954cb9d3b1610ca9c635209"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerun final readout comparison against regenerated direct-module PPP logits."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prior-final-readout", type=Path, required=True)
    parser.add_argument("--source-ladder", type=Path, required=True)
    parser.add_argument("--official-direct-readout", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def boundary_by_name(bundle):
    return {boundary["boundary"]: boundary for boundary in bundle["boundaries"]}


def top20_compare_direct(local_logits, official_top):
    official_ids = [int(v) for v in official_top["ordered_token_ids"]]
    official_values = torch.tensor(official_top["values"], dtype=torch.float32, device=local_logits.device)
    local_values = local_logits[official_ids].reshape(-1).to(torch.float32)
    metric = ladder.compare_vector(local_values, official_values)
    local_top = readout.top20_summary(local_logits)
    return {
        "official_top_token_ids": official_ids,
        "official_top_values": [float(v) for v in official_top["values"]],
        "local_top_token_ids": local_top["token_ids"],
        "local_top_values": local_top["values"],
        "ordered_top20_token_ids_match": local_top["token_ids"] == official_ids,
        "top20_set_match": sorted(local_top["token_ids"]) == sorted(official_ids),
        "official_top20_token_value_metric": metric,
    }


def main():
    args = parse_args()
    prior_final_readout = load_json(args.prior_final_readout)
    source_ladder = load_json(args.source_ladder)
    official_bundle = load_json(args.official_direct_readout)
    local_residual_input_artifact = load_json(args.local_residual_input)
    if prior_final_readout.get("mode") != "final-readout-norm-and-lm-head-compare-status":
        raise ValueError("prior final readout artifact is not the expected runtime diagnostic")
    if source_ladder.get("classification") != "layer2_to_final_coarse_layer_ladder_cleared":
        raise ValueError("source coarse ladder artifact is not cleared")
    if official_bundle.get("classification") != "official_final_readout_direct_module_logits_regenerated":
        raise ValueError("official direct-module final readout bundle is not the expected regenerated PPP capture")
    if official_bundle.get("missing_boundaries"):
        raise ValueError("official direct-module final readout bundle has missing boundaries")

    device = torch.device(args.device)
    by_name = boundary_by_name(official_bundle)
    final_hidden_official = ladder.tensor_from_values(
        by_name["final_token_hidden_state_after_final_transformer_block"], device
    )
    final_norm_official = ladder.tensor_from_values(
        by_name["final_token_final_norm_output_before_lm_head"], device
    )
    logits_official = ladder.tensor_from_values(by_name["final_token_lm_head_logits_direct_module"], device)

    x, config, checkpoint, TransformerBlock = ladder.build_layer2_input(
        args.model_root, official_bundle["input_token_ids"], device
    )
    current = x
    for layer_idx in range(2, 24):
        _local_boundaries, current = ladder.compute_layer_boundaries(
            current, layer_idx, config, checkpoint, TransformerBlock, device
        )
    final_hidden = current[-1].to(torch.float32).contiguous()

    final_norm, unembedding = readout.load_final_modules(config, checkpoint, device)
    with torch.inference_mode():
        final_norm_output_bf16 = final_norm(current[-1:])
        final_norm_output = final_norm_output_bf16.reshape(2880).to(torch.float32).contiguous()
        logits = unembedding(final_norm_output_bf16).reshape(-1).to(torch.float32).contiguous()

    final_hidden_metric = ladder.compare_vector(final_hidden, final_hidden_official)
    final_norm_metric = ladder.compare_vector(final_norm_output, final_norm_official)
    logits_metric = ladder.compare_vector(logits, logits_official)
    top20 = top20_compare_direct(logits, official_bundle["top20_logits_summary"])

    final_hidden_digest = ladder.sha256_f32_le(final_hidden)
    final_norm_digest = ladder.sha256_f32_le(final_norm_output)
    logits_digest = ladder.sha256_f32_le(logits)
    official_final_hidden_digest = by_name["final_token_hidden_state_after_final_transformer_block"][
        "finite_value_summary"
    ]["sha256_f32_le"]
    official_final_norm_digest = by_name["final_token_final_norm_output_before_lm_head"][
        "finite_value_summary"
    ]["sha256_f32_le"]
    official_logits_digest = by_name["final_token_lm_head_logits_direct_module"]["finite_value_summary"][
        "sha256_f32_le"
    ]

    if not final_hidden_metric["matched"] or final_hidden_digest != official_final_hidden_digest:
        classification = "final_readout_direct_rerun_blocked_by_final_transformer_output_regression"
        earliest = "final_token_hidden_state_after_final_transformer_block"
        next_step = "revalidate final transformer block output guard only"
    elif not final_norm_metric["matched"] or final_norm_digest != official_final_norm_digest:
        classification = "final_readout_direct_rerun_blocked_by_final_norm_regression"
        earliest = "final_token_final_norm_output_before_lm_head"
        next_step = "revalidate final norm output guard only"
    elif not logits_metric["matched"] or logits_digest != official_logits_digest:
        classification = "final_readout_direct_module_logits_still_mismatch"
        earliest = "final_token_lm_head_logits"
        next_step = "localize LM-head logits capture/readout against regenerated direct-module PPP only"
    else:
        classification = "final_readout_direct_module_logits_cleared"
        earliest = "none"
        next_step = (
            "record full final-token oracle parity milestone and commit/push before promotion or 4097-boundary work"
        )

    output = {
        "schema_version": "runtime_forward_final_readout_direct_module_rerun_status/v1",
        "mode": "final-readout-direct-module-rerun-status",
        "exact_case": {"case_id": "developer-message-user-smoke", "token_index": 73},
        "source_artifact_paths": {
            "prior_runtime_final_readout": str(args.prior_final_readout),
            "coarse_layer_ladder": str(args.source_ladder),
            "regenerated_direct_module_ppp": str(args.official_direct_readout),
            "local_residual_input": str(args.local_residual_input),
        },
        "regenerated_ppp_reference_path": str(args.official_direct_readout),
        "final_block_output_guard_metrics": {
            **final_hidden_metric,
            "local_sha256_f32_le": final_hidden_digest,
            "official_sha256_f32_le": official_final_hidden_digest,
            "digest_matched": final_hidden_digest == official_final_hidden_digest,
        },
        "final_norm_guard_metrics": {
            **final_norm_metric,
            "local_sha256_f32_le": final_norm_digest,
            "official_sha256_f32_le": official_final_norm_digest,
            "digest_matched": final_norm_digest == official_final_norm_digest,
        },
        "lm_head_logits_metrics": {
            **logits_metric,
            "shape": list(logits.shape),
            "vocab_size": int(logits.numel()),
            "dtype": "float32",
            "serialization_dtype": "f32-expanded BF16 values",
            "local_sha256_f32_le": logits_digest,
            "official_direct_module_sha256_f32_le": official_logits_digest,
            "digest_matched": logits_digest == official_logits_digest,
        },
        "top20_comparison_summary": top20,
        "stale_artifact_note": {
            "prior_ppp_logits_digest": PRIOR_STALE_LOGITS_DIGEST,
            "regenerated_direct_module_logits_digest": DIRECT_MODULE_LOGITS_DIGEST,
            "previous_37_logit_mismatch_explanation": (
                "The prior PPP logits artifact was stale/mismatched on 37 tail logits; "
                "runtime-forward local logits match the regenerated direct-module PPP logits."
            ),
        },
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
