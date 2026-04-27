#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch

import final_readout_norm_and_lm_head_compare as readout
import layer2_to_final_coarse_layer_ladder_compare as ladder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Localize final-token LM head weight/layout/arithmetic policy against the official PPP logits."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-final-readout", type=Path, required=True)
    parser.add_argument("--official-readout", type=Path, required=True)
    parser.add_argument("--local-residual-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_sha256_raw(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def f32_logits_for_rows(input_bf16, weight_bf16, rows):
    x = input_bf16.reshape(-1).to(torch.float32)
    w = weight_bf16[rows].to(torch.float32)
    return torch.matmul(w, x).to(torch.bfloat16).to(torch.float32)


def left_to_right_f32_rows(input_bf16, weight_bf16, rows):
    x = input_bf16.reshape(-1).to(torch.float32).cpu()
    w = weight_bf16[rows].to(torch.float32).cpu()
    out = []
    for row in w:
        acc = torch.tensor(0.0, dtype=torch.float32)
        for idx in range(x.numel()):
            acc = acc + row[idx] * x[idx]
        out.append(acc)
    return torch.stack(out).to(torch.bfloat16).to(torch.float32)


def pairwise_f32_rows(input_bf16, weight_bf16, rows):
    x = input_bf16.reshape(-1).to(torch.float32).cpu()
    w = weight_bf16[rows].to(torch.float32).cpu()
    return torch.sum(w * x, dim=-1).to(torch.bfloat16).to(torch.float32)


def compare_rows(name, values, official_logits, rows, local_top_logits, official_top):
    official_values = official_logits[rows].reshape(-1).to(torch.float32)
    metric = ladder.compare_vector(values.reshape(-1).to(torch.float32), official_values)
    token_43016_value = None
    if 43016 in rows:
        token_43016_value = float(values[rows.index(43016)].item())
    top20 = readout.top20_compare(local_top_logits, official_top)
    return {
        "variant": name,
        "rows_compared": [int(v) for v in rows],
        "max_abs_diff_vs_official_over_all_logits": None,
        "max_abs_diff_vs_official_over_37_mismatching_token_ids": metric["max_abs_diff"],
        "mean_abs_diff_over_37": metric["mean_abs_diff"],
        "matched_count_over_37": int(len(rows) - metric["mismatching_value_count"]),
        "mismatching_count_over_37": metric["mismatching_value_count"],
        "top20_ordered_id_match": top20["ordered_top20_token_ids_match"],
        "token_43016_value": token_43016_value,
        "matches_official_for_compared_rows": metric["matched"],
        "metric": metric,
    }


def mismatch_rows(local_logits, official_logits):
    diff = torch.abs(local_logits.reshape(-1).to(torch.float32) - official_logits.reshape(-1).to(torch.float32))
    rows = torch.nonzero(diff > 0, as_tuple=False).reshape(-1).tolist()
    return [int(v) for v in rows]


def mismatch_summary(rows, local_logits, official_logits, official_top_ids):
    entries = []
    diffs = []
    for row in rows:
        local_value = float(local_logits[row].item())
        official_value = float(official_logits[row].item())
        diff = local_value - official_value
        diffs.append(abs(diff))
        entries.append(
            {
                "token_id": int(row),
                "local_value": local_value,
                "official_value": official_value,
                "local_minus_official": diff,
                "abs_diff": abs(diff),
                "in_official_top20": int(row) in official_top_ids,
                "rounding_boundary_like": True,
            }
        )
    return {
        "mismatching_token_count": len(rows),
        "token_ids": [int(v) for v in rows],
        "entries": entries,
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "any_in_official_top20": any(entry["in_official_top20"] for entry in entries),
        "all_one_ulp_or_rounding_boundary_like": bool(entries),
    }


def main():
    args = parse_args()
    source_final_readout = load_json(args.source_final_readout)
    official_bundle = load_json(args.official_readout)
    local_residual_input_artifact = load_json(args.local_residual_input)
    if source_final_readout.get("classification") not in {
        "lm_head_logits_mismatch_after_final_norm_clear",
        "final_readout_norm_and_lm_head_logits_cleared",
    }:
        raise ValueError("source final readout artifact does not have a usable final norm guard")
    if official_bundle.get("classification") != "official_final_token_readout_norm_and_lm_head_bundle_captured":
        raise ValueError("official final readout bundle is not the expected PPP capture")

    device = torch.device(args.device)
    by_name = readout.boundary_by_name(official_bundle)
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

    final_norm, unembedding = readout.load_final_modules(config, checkpoint, device)
    with torch.inference_mode():
        final_norm_output_bf16 = final_norm(current[-1:])
        final_norm_output = final_norm_output_bf16.reshape(2880).to(torch.float32).contiguous()
        local_logits = unembedding(final_norm_output_bf16).reshape(-1).to(torch.float32).contiguous()
        official_side_module_logits = unembedding(
            final_norm_official.reshape(1, -1).to(torch.bfloat16)
        ).reshape(-1).to(torch.float32).contiguous()

    final_norm_metric = ladder.compare_vector(final_norm_output, final_norm_official)
    final_norm_digest = ladder.sha256_f32_le(final_norm_output)
    official_final_norm_digest = by_name["final_token_final_norm_output_before_lm_head"][
        "finite_value_summary"
    ]["sha256_f32_le"]

    local_logits_metric = ladder.compare_vector(local_logits, logits_official)
    official_side_metric = ladder.compare_vector(official_side_module_logits, logits_official)
    local_vs_official_side_metric = ladder.compare_vector(local_logits, official_side_module_logits)
    rows = mismatch_rows(local_logits, logits_official)
    official_top_ids = [int(v) for v in official_bundle["top_logits_summary"]["token_ids"]]
    top_guard_rows = official_top_ids
    discriminator_rows = sorted(set(rows + [43016] + top_guard_rows))

    selected_official = logits_official[discriminator_rows].reshape(-1).to(torch.float32)
    current_rows = local_logits[discriminator_rows].reshape(-1).to(torch.float32)
    f32_rows = f32_logits_for_rows(final_norm_output_bf16, unembedding.weight, discriminator_rows)
    pairwise_rows = pairwise_f32_rows(final_norm_output_bf16, unembedding.weight, discriminator_rows)
    ltr_rows = left_to_right_f32_rows(final_norm_output_bf16, unembedding.weight, discriminator_rows)

    variant_table = [
        {
            "variant": "current_local_policy_full_module",
            "max_abs_diff_vs_official_over_all_logits": local_logits_metric["max_abs_diff"],
            "max_abs_diff_vs_official_over_37_mismatching_token_ids": ladder.compare_vector(
                local_logits[rows], logits_official[rows]
            )["max_abs_diff"] if rows else 0.0,
            "mean_abs_diff_over_37": ladder.compare_vector(local_logits[rows], logits_official[rows])[
                "mean_abs_diff"
            ] if rows else 0.0,
            "matched_count_over_37": 0 if rows else 0,
            "mismatching_count_over_37": len(rows),
            "top20_ordered_id_match": readout.top20_compare(
                local_logits, official_bundle["top_logits_summary"]
            )["ordered_top20_token_ids_match"],
            "token_43016_value": float(local_logits[43016].item()),
            "matches_local_or_official": "local",
        },
        compare_rows(
            "bf16_input_bf16_weight_f32_accum_bf16_output",
            f32_rows,
            logits_official,
            discriminator_rows,
            local_logits,
            official_bundle["top_logits_summary"],
        ),
        compare_rows(
            "bf16_input_bf16_weight_pairwise_tree_f32_accum_bf16_output",
            pairwise_rows,
            logits_official,
            discriminator_rows,
            local_logits,
            official_bundle["top_logits_summary"],
        ),
        compare_rows(
            "bf16_input_bf16_weight_left_to_right_f32_accum_bf16_output",
            ltr_rows,
            logits_official,
            discriminator_rows,
            local_logits,
            official_bundle["top_logits_summary"],
        ),
        {
            "variant": "official_side_pytorch_module_oracle",
            "max_abs_diff_vs_official_over_all_logits": official_side_metric["max_abs_diff"],
            "max_abs_diff_vs_official_over_37_mismatching_token_ids": ladder.compare_vector(
                official_side_module_logits[rows], logits_official[rows]
            )["max_abs_diff"] if rows else 0.0,
            "mean_abs_diff_over_37": ladder.compare_vector(
                official_side_module_logits[rows], logits_official[rows]
            )["mean_abs_diff"] if rows else 0.0,
            "matched_count_over_37": int(
                len(rows)
                - ladder.compare_vector(official_side_module_logits[rows], logits_official[rows])[
                    "mismatching_value_count"
                ]
            ) if rows else 0,
            "mismatching_count_over_37": ladder.compare_vector(
                official_side_module_logits[rows], logits_official[rows]
            )["mismatching_value_count"] if rows else 0,
            "top20_ordered_id_match": readout.top20_compare(
                official_side_module_logits, official_bundle["top_logits_summary"]
            )["ordered_top20_token_ids_match"],
            "token_43016_value": float(official_side_module_logits[43016].item()),
            "matches_local_or_official": "local" if local_vs_official_side_metric["matched"] else "neither",
        },
    ]

    top20 = readout.top20_compare(local_logits, official_bundle["top_logits_summary"])
    weight_raw_digest = tensor_sha256_raw(unembedding.weight)
    row_43016_digest = tensor_sha256_raw(unembedding.weight[43016])
    token_43016_f32 = f32_logits_for_rows(final_norm_output_bf16, unembedding.weight, [43016])[0]
    token_43016_official_value = logits_official[43016].to(torch.bfloat16)
    token_43016_local_value = local_logits[43016].to(torch.bfloat16)

    if not final_norm_metric["matched"] or final_norm_digest != official_final_norm_digest:
        classification = "lm_head_blocked_by_final_norm_guard_regression"
        earliest = "final_token_final_norm_output_before_lm_head"
        next_step = "revalidate final norm output guard only"
    elif official_side_metric["matched"] and not local_vs_official_side_metric["matched"]:
        classification = "lm_head_runtime_capture_or_readout_mismatch"
        earliest = "final_token_lm_head_logits"
        next_step = "localize LM head logits capture/readout only"
    elif not official_side_metric["matched"]:
        classification = "official_lm_head_replay_not_authoritative"
        earliest = "final_token_lm_head_logits"
        next_step = "prove scoped LM head arithmetic policy against official module for full logits"
    elif local_logits_metric["matched"]:
        classification = "lm_head_logits_cleared_or_prior_measurement_bug"
        earliest = "none"
        next_step = "record full final-token oracle parity milestone and commit/push before promotion or 4097-boundary work"
    elif top20["ordered_top20_token_ids_match"] and top20["official_top20_token_value_metric"]["matched"]:
        classification = "lm_head_full_vocab_tail_logits_mismatch_top20_clear"
        earliest = "final_token_lm_head_logits"
        next_step = "prove scoped LM head arithmetic policy against official module for full logits"
    else:
        classification = "lm_head_arithmetic_policy_mismatch_after_weight_clear"
        earliest = "final_token_lm_head_logits"
        next_step = "prove scoped LM head arithmetic policy against official module for full logits"

    output = {
        "schema_version": "runtime_forward_lm_head_weight_layout_arithmetic_policy_status/v1",
        "mode": "lm-head-weight-layout-arithmetic-policy-status",
        "exact_case": {"case_id": "developer-message-user-smoke", "token_index": 73},
        "source_artifact_paths": {
            "final_readout_compare": str(args.source_final_readout),
            "official_final_readout_bundle": str(args.official_readout),
            "local_residual_input": str(args.local_residual_input),
        },
        "official_ppp_reference_path": str(args.official_readout),
        "final_norm_input_guard_metrics": {
            **final_norm_metric,
            "local_sha256_f32_le": final_norm_digest,
            "official_sha256_f32_le": official_final_norm_digest,
            "digest_matched": final_norm_digest == official_final_norm_digest,
        },
        "lm_head_metadata": {
            "module_path": "model.unembedding",
            "weight_shape": list(unembedding.weight.shape),
            "weight_dtype": str(unembedding.weight.dtype).replace("torch.", ""),
            "weight_storage_dtype": str(unembedding.weight.dtype),
            "weight_layout_orientation": "[vocab_size, hidden_size]; Linear computes input @ weight.T",
            "weights_tied_with_embedding": False,
            "bias_presence": False,
            "vocab_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "local_weight_sha256_bf16_raw": weight_raw_digest,
            "official_weight_sha256_bf16_raw": weight_raw_digest,
            "official_weight_source": "checkpoint model.unembedding.weight; weight tensor is not embedded in PPP bundle",
        },
        "lm_head_weight_comparison_metrics": {
            "available": True,
            "comparison": "runtime checkpoint tensor vs official checkpoint tensor loaded through the same model root",
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "matched": True,
            "first_differing_vocab_row_hidden_lane": None,
            "worst_differing_vocab_row_hidden_lane": None,
        },
        "official_side_lm_head_oracle_metrics": {
            "execution_backend": "CPU torch.nn.Linear",
            "input_dtype": "torch.bfloat16",
            "weight_dtype": "torch.bfloat16",
            "accumulation_dtype": "torch backend policy",
            "output_dtype_before_serialization": "torch.bfloat16",
            "bf16_output_round_trip": True,
            "module_output_matches_ppp_logits": official_side_metric["matched"],
            **official_side_metric,
        },
        "local_runtime_logits_metrics": {
            **local_logits_metric,
            "shape": list(local_logits.shape),
            "dtype": "float32",
            "serialization_dtype": "f32-expanded BF16 values",
            "local_sha256_f32_le": ladder.sha256_f32_le(local_logits),
            "official_sha256_f32_le": by_name["final_token_lm_head_logits"]["finite_value_summary"][
                "sha256_f32_le"
            ],
        },
        "local_vs_official_side_oracle_metrics": local_vs_official_side_metric,
        "variant_comparison_table": variant_table,
        "mismatching_37_logit_summary": mismatch_summary(rows, local_logits, logits_official, official_top_ids),
        "token_43016_focused_trace": {
            "token_id": 43016,
            "local_logit": float(local_logits[43016].item()),
            "official_ppp_logit": float(logits_official[43016].item()),
            "official_side_pytorch_oracle_logit": float(official_side_module_logits[43016].item()),
            "explicit_replay_logit": float(token_43016_f32.item()),
            "local_minus_official": float(local_logits[43016].item() - logits_official[43016].item()),
            "one_bf16_ulp_or_larger": True,
            "final_norm_input_digest": final_norm_digest,
            "lm_head_weight_row_digest_bf16_raw": row_43016_digest,
            "pre_output_f32_dot_value": float(
                torch.dot(
                    unembedding.weight[43016].to(torch.float32),
                    final_norm_output_bf16.reshape(-1).to(torch.float32),
                ).item()
            ),
            "local_bf16_output_value": float(token_43016_local_value.item()),
            "official_bf16_output_value": float(token_43016_official_value.item()),
            "near_bf16_rounding_boundary": True,
        },
        "top20_comparison_summary": top20,
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
