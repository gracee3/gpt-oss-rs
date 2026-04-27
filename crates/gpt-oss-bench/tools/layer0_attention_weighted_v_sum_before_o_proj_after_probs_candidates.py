#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base
import layer0_q_pre_post_rope_runtime_localization as qloc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token weighted V sum before o_proj after exact Q/K/probability candidates."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--attention-probs-proof", type=Path, required=True)
    parser.add_argument("--q-candidate", type=Path, required=True)
    parser.add_argument("--k-candidate", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--local-v-capture", type=Path, required=True)
    parser.add_argument("--official-weighted-v", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(lhs, rhs, torch, kind="flat_feature"):
    return qloc.compare_tensor(lhs, rhs, torch, kind)


def tensor_meta(tensor, layout, serialization_dtype=None):
    meta = qloc.tensor_meta(tensor, layout, serialization_dtype)
    meta["device"] = str(tensor.device)
    return meta


def digest_tensor(tensor, torch):
    return qloc.digest_tensor(tensor, torch)


def bf16_bits(value):
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


def compare_head_lane(lhs, rhs, torch):
    diff = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        coords = mismatch.nonzero(as_tuple=False)
        h0 = int(coords[0, 0].item())
        l0 = int(coords[0, 1].item())
        flat = int(diff.reshape(-1).argmax().item())
        head_dim = int(lhs.shape[1])
        hw = flat // head_dim
        lw = flat % head_dim
        first = {
            "flattened_feature": h0 * head_dim + l0,
            "q_head": h0,
            "lane": l0,
            "lhs_value": float(lhs[h0, l0].to(torch.float32).item()),
            "rhs_value": float(rhs[h0, l0].to(torch.float32).item()),
            "abs_diff": float(diff[h0, l0].item()),
        }
        worst = {
            "flattened_feature": hw * head_dim + lw,
            "q_head": hw,
            "lane": lw,
            "lhs_value": float(lhs[hw, lw].to(torch.float32).item()),
            "rhs_value": float(rhs[hw, lw].to(torch.float32).item()),
            "abs_diff": float(diff[hw, lw].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs.to(torch.float32), rhs.to(torch.float32))),
        "mismatching_head_count": int(mismatch.any(dim=1).sum().item()) if diff.ndim == 2 else None,
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_flattened_feature_q_head_lane": first,
        "worst_differing_flattened_feature_q_head_lane": worst,
    }


def per_head_summary(lhs, rhs, torch):
    diff = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs()
    rows = []
    for head in range(diff.shape[0]):
        rows.append(
            {
                "q_head": head,
                "max_abs_diff": float(diff[head].max().item()),
                "mean_abs_diff": float(diff[head].mean().item()),
                "matched": bool(torch.equal(lhs[head].to(torch.float32), rhs[head].to(torch.float32))),
            }
        )
    worst = max(rows, key=lambda item: item["max_abs_diff"]) if rows else None
    return {"rows": rows, "worst_head": worst}


def metric_for_variant(name, values, official, torch, interpretation):
    return {
        "name": name,
        "interpretation": interpretation,
        "metrics": compare_head_lane(values.reshape(64, 64), official.reshape(64, 64), torch),
    }


def weighted_sum(probs_64x75, v_74x8x64, torch, policy):
    probs_bf16 = probs_64x75.to(torch.bfloat16)
    v_bf16 = v_74x8x64.to(torch.bfloat16)
    rows = []
    for q_head in range(64):
        kv_head = q_head // 8
        if policy == "torch_bf16_einsum_real_keys":
            row = torch.einsum("t,td->d", probs_bf16[q_head, :74], v_bf16[:, kv_head, :])
            rows.append(row.to(torch.bfloat16))
        elif policy == "f32_accum_bf16_output":
            row = (
                probs_bf16[q_head, :74].to(torch.float32)[:, None]
                * v_bf16[:, kv_head, :].to(torch.float32)
            ).sum(dim=0)
            rows.append(row.to(torch.bfloat16))
        elif policy == "include_sink_as_zero_v_guard":
            zero = torch.zeros((1, 64), dtype=torch.bfloat16, device=probs_64x75.device)
            v_with_sink = torch.cat([v_bf16[:, kv_head, :], zero], dim=0)
            row = torch.einsum("t,td->d", probs_bf16[q_head, :], v_with_sink)
            rows.append(row.to(torch.bfloat16))
        else:
            raise ValueError(f"unknown weighted sum policy {policy}")
    return torch.stack(rows, dim=0).reshape(4096).to(torch.float32)


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)

    probs_proof = load_json(args.attention_probs_proof)
    q_candidate = load_json(args.q_candidate)
    k_candidate = load_json(args.k_candidate)
    local_v_capture = load_json(args.local_v_capture)
    official_weighted = load_json(args.official_weighted_v)
    if probs_proof.get("classification") != "attention_probs_cleared_after_post_mask_candidates":
        raise ValueError("attention probability proof is not in the expected cleared state")
    if official_weighted.get("classification") != "official_layer0_final_token_attention_weighted_value_sum_pre_o_proj_captured":
        raise ValueError("official weighted V artifact is not the expected PPP capture")

    q_guard = probs_proof["q_post_rope_guard_metrics"]
    k_guard = probs_proof["k_post_rope_guard_metrics"]
    real_prob_guard = probs_proof["real_key_probability_metrics"]
    sink_prob_guard = probs_proof["sink_probability_metrics"]
    all_prob_guard = probs_proof["all_position_probability_metrics"]
    guards_pass = (
        q_guard.get("matched")
        and k_guard.get("matched")
        and real_prob_guard.get("matched")
        and sink_prob_guard.get("matched")
        and all_prob_guard.get("matched")
    )

    rmsnorm = load_json(args.rmsnorm_replay)
    norm_output = torch.tensor(
        rmsnorm["policy_outputs_f32"]["manual_bf16_input_bf16_weight_f32_reduction_bf16_output"],
        dtype=torch.float32,
        device=device,
    ).reshape(74, 2880)
    norm_bf16 = norm_output.to(torch.bfloat16).contiguous()

    model = base.load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    attn = model.attn
    q_dim = int(attn.num_attention_heads * attn.head_dim)
    kv_dim = int(attn.num_key_value_heads * attn.head_dim)
    v_start = q_dim + kv_dim
    v_end = v_start + kv_dim
    v_weight = attn.qkv.weight[v_start:v_end, :].detach().to(torch.float32).contiguous()
    v_bias = None
    if getattr(attn.qkv, "bias", None) is not None:
        v_bias = attn.qkv.bias[v_start:v_end].detach().to(torch.float32).contiguous()
    v_weight_bf16 = v_weight.to(torch.bfloat16).contiguous()
    v_bias_bf16 = v_bias.to(torch.bfloat16).contiguous() if v_bias is not None else None

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    original_threads = int(torch.get_num_threads())
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        official_module_v = attn.qkv(norm_bf16)[:, v_start:v_end].contiguous()
        onednn_v = torch.nn.functional.linear(norm_bf16, v_weight_bf16, v_bias_bf16).contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn

    local_v = torch.tensor(
        local_v_capture["local_v_f32"], dtype=torch.float32, device=device
    ).reshape(74, 8, 64)
    official_v = official_module_v.reshape(74, 8, 64).to(torch.float32)
    onednn_v_grouped = onednn_v.reshape(74, 8, 64).to(torch.float32)

    official_probs = torch.tensor(
        load_json(Path(probs_proof["official_post_softmax_reference_path"]))["values"],
        dtype=torch.float32,
        device=device,
    ).reshape(64, 75)
    official_weighted_values = torch.tensor(
        official_weighted["values"], dtype=torch.float32, device=device
    ).reshape(4096)

    local_weighted = weighted_sum(official_probs, local_v, torch, "torch_bf16_einsum_real_keys")
    official_v_weighted = weighted_sum(official_probs, official_v, torch, "torch_bf16_einsum_real_keys")
    onednn_v_weighted = weighted_sum(official_probs, onednn_v_grouped, torch, "torch_bf16_einsum_real_keys")
    local_weighted_f32_accum = weighted_sum(official_probs, local_v, torch, "f32_accum_bf16_output")
    local_weighted_sink_zero = weighted_sum(official_probs, local_v, torch, "include_sink_as_zero_v_guard")

    v_provenance_metrics = {
        "local_runtime_v_vs_official_module_v": compare(local_v, official_v, torch, "token_kv_head_lane"),
        "local_runtime_v_vs_onednn_v_oracle": compare(local_v, onednn_v_grouped, torch, "token_kv_head_lane"),
        "onednn_v_oracle_vs_official_module_v": compare(onednn_v_grouped, official_v, torch, "token_kv_head_lane"),
    }
    weighted_metrics = compare_head_lane(local_weighted.reshape(64, 64), official_weighted_values.reshape(64, 64), torch)
    onednn_weighted_metrics = compare_head_lane(onednn_v_weighted.reshape(64, 64), official_weighted_values.reshape(64, 64), torch)

    causality_variants = []
    if not weighted_metrics["matched"]:
        causality_variants = [
            metric_for_variant(
                "local_probabilities_local_v_official_sink_drop_gqa",
                local_weighted,
                official_weighted_values,
                torch,
                "exact probability tensor, local runtime V, drop sink probability before V sum, kv_head=q_head//8",
            ),
            metric_for_variant(
                "official_probabilities_local_v_official_sink_drop_gqa",
                local_weighted,
                official_weighted_values,
                torch,
                "official probabilities equal local probabilities from the previous cleared seam; local V",
            ),
            metric_for_variant(
                "local_probabilities_onednn_v_oracle",
                onednn_v_weighted,
                official_weighted_values,
                torch,
                "exact probabilities with oneDNN/official V oracle",
            ),
            metric_for_variant(
                "official_probabilities_official_module_v",
                official_v_weighted,
                official_weighted_values,
                torch,
                "official probabilities with official module V",
            ),
            metric_for_variant(
                "local_probabilities_local_v_include_sink_as_zero_v_guard",
                local_weighted_sink_zero,
                official_weighted_values,
                torch,
                "same as sink-drop except an explicit zero sink V is appended as a guard",
            ),
            metric_for_variant(
                "local_probabilities_local_v_f32_accum_bf16_output",
                local_weighted_f32_accum,
                official_weighted_values,
                torch,
                "f32 accumulation with BF16 output round-trip",
            ),
        ]

    worst = weighted_metrics["worst_differing_flattened_feature_q_head_lane"]
    focused_trace = None
    if worst is not None:
        q_head = int(worst["q_head"])
        lane = int(worst["lane"])
        kv_head = q_head // 8
        probs = official_probs[q_head, :74]
        top = torch.topk(probs.to(torch.float32), k=min(5, probs.numel()))
        top_positions = [int(i.item()) for i in top.indices]
        focused_trace = {
            "flattened_feature": int(worst["flattened_feature"]),
            "q_head": q_head,
            "lane": lane,
            "mapped_kv_head": kv_head,
            "official_weighted_v_value": float(official_weighted_values.reshape(64, 64)[q_head, lane].item()),
            "local_weighted_v_value": float(local_weighted.reshape(64, 64)[q_head, lane].item()),
            "local_minus_official": float(local_weighted.reshape(64, 64)[q_head, lane].item() - official_weighted_values.reshape(64, 64)[q_head, lane].item()),
            "probability_row_digest": digest_tensor(official_probs[q_head], torch),
            "v_vector_digest_local_mapped_kv_head": digest_tensor(local_v[:, kv_head, lane], torch),
            "v_vector_digest_onednn_mapped_kv_head": digest_tensor(onednn_v_grouped[:, kv_head, lane], torch),
            "sink_probability_was_dropped": True,
            "top_contributing_key_positions_by_probability": [
                {
                    "key_position": pos,
                    "probability": float(official_probs[q_head, pos].item()),
                    "local_v_value": float(local_v[pos, kv_head, lane].item()),
                    "onednn_v_value": float(onednn_v_grouped[pos, kv_head, lane].item()),
                    "official_module_v_value": float(official_v[pos, kv_head, lane].item()),
                }
                for pos in top_positions
            ],
            "mismatch_already_exists_in_v_provenance": not v_provenance_metrics["local_runtime_v_vs_onednn_v_oracle"]["matched"],
            "mismatch_appears_only_after_weighted_sum": v_provenance_metrics["local_runtime_v_vs_onednn_v_oracle"]["matched"] and not weighted_metrics["matched"],
        }

    v_provenance_matches = v_provenance_metrics["local_runtime_v_vs_onednn_v_oracle"]["matched"]
    if not guards_pass:
        classification = "attention_weighted_v_input_guard_regressed"
        earliest = "attention_weighted_v_input_guard"
        next_step = "restore exact Q/K/probability guards before checking weighted V sum"
    elif list(official_weighted_values.shape) != [4096]:
        classification = "attention_weighted_v_sum_shape_or_layout_mismatch"
        earliest = "layer0_final_token_attention_weighted_value_sum_before_output_projection"
        next_step = "align weighted V sum output layout to flattened head-major [q_head, lane]"
    elif not v_provenance_matches:
        classification = "attention_weighted_v_sum_blocked_by_v_provenance_mismatch"
        earliest = "layer0_v_projection_before_attention_weighted_sum"
        next_step = "inspect layer0 V projection weight/bias/arithmetic policy only"
    elif not weighted_metrics["matched"] and onednn_weighted_metrics["matched"]:
        classification = "attention_weighted_v_sum_dtype_policy_mismatch"
        earliest = "layer0_final_token_attention_weighted_value_sum_before_output_projection"
        next_step = "prove scoped weighted-sum dtype policy before o_proj"
    elif weighted_metrics["matched"]:
        classification = "attention_weighted_v_sum_pre_o_proj_cleared_after_probs_candidates"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_attention_output_after_o_proj_before_residual"
    else:
        classification = "attention_weighted_v_sum_dtype_policy_mismatch"
        earliest = "layer0_final_token_attention_weighted_value_sum_before_output_projection"
        next_step = "prove scoped weighted-sum dtype policy before o_proj"

    output = {
        "schema_version": "runtime_forward_layer0_attention_weighted_v_sum_before_o_proj_after_probs_candidates_status/v1",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "query_position": "final_token",
            "query_token_index": 73,
        },
        "mode": "attention-weighted-v-sum-before-o-proj-after-probs-candidates-status",
        "source_artifact_paths": {
            "q_candidate_proof": str(args.q_candidate),
            "k_candidate_proof": str(args.k_candidate),
            "attention_probability_proof": str(args.attention_probs_proof),
            "official_weighted_v_sum_reference": str(args.official_weighted_v),
            "local_v_capture": str(args.local_v_capture),
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
        },
        "qk_candidates_are_bench_proof_only": True,
        "candidate_scope_confirmation": {
            "q_candidate_change_scope": q_candidate.get("change_scope"),
            "q_candidate_runtime_affecting": q_candidate.get("runtime_affecting"),
            "k_candidate_change_scope": k_candidate.get("change_scope"),
            "k_candidate_runtime_affecting": k_candidate.get("runtime_affecting"),
            "not_promoted_to_runtime": True,
        },
        "q_k_probability_guard_metrics": {
            "q_post_rope_guard_metrics": q_guard,
            "k_post_rope_guard_metrics": k_guard,
            "post_softmax_probability_metrics": {
                "real_key_probabilities": real_prob_guard,
                "sink_probabilities": sink_prob_guard,
                "all_positions": all_prob_guard,
                "row_sum_summary_after_bf16_serialization": probs_proof.get("local_row_sum_summary"),
            },
        },
        "official_weighted_v_sum_reference_path": str(args.official_weighted_v),
        "official_tensor_metadata": {
            "shape": official_weighted.get("shape"),
            "per_head_equivalent_shape": official_weighted.get("per_head_shape"),
            "dtype_before_serialization": official_weighted.get("output_dtype_before_serialization"),
            "serialization": official_weighted.get("serialization_dtype"),
            "layout": official_weighted.get("layout_interpretation"),
            "sink_semantics": official_weighted.get("sink_participation_semantics"),
            "weighted_sum_computation_dtype": official_weighted.get("weighted_sum_computation_dtype"),
        },
        "local_weighted_v_tensor_metadata": {
            "shape": [4096],
            "per_head_equivalent_shape": [64, 64],
            "dtype": "float32",
            "serialization_dtype": "json_f32_values",
            "layout": "flattened head-major [q_head, lane]",
            "sink_probability_dropped_before_v_sum": True,
            "gqa_mapping": "kv_head = q_head // 8",
        },
        "v_provenance_metadata": {
            "local_v": {
                "shape": [74, 8, 64],
                "dtype": "float32 debug readout",
                "serialization_dtype": "json_f32_values",
                "layout": "[token, kv_head, head_dim]",
                "token_count": 74,
                "kv_head_count": 8,
                "head_dim": 64,
                "sink_v_present": False,
                "source": local_v_capture.get("local_v_source"),
            },
            "official_v": official_weighted.get("v_tensor_metadata"),
            "v_weight": {
                "slice_range_rows": [v_start, v_end],
                "shape": list(v_weight.shape),
                "dtype": "torch.bfloat16 after cast",
                "layout": "row-major [v_output_feature, hidden]",
                "digest": digest_tensor(v_weight_bf16, torch),
            },
            "v_bias": {
                "present": v_bias is not None,
                "shape": list(v_bias_bf16.shape) if v_bias_bf16 is not None else [],
                "dtype": "torch.bfloat16" if v_bias_bf16 is not None else "none",
                "max_abs": float(v_bias_bf16.to(torch.float32).abs().max().item()) if v_bias_bf16 is not None else 0.0,
                "mean_abs": float(v_bias_bf16.to(torch.float32).abs().mean().item()) if v_bias_bf16 is not None else 0.0,
                "checksum": digest_tensor(v_bias_bf16, torch) if v_bias_bf16 is not None else None,
                "all_zero": bool(torch.all(v_bias_bf16.to(torch.float32) == 0).item()) if v_bias_bf16 is not None else True,
            },
            "onednn_v_oracle": {
                "torch_version": torch.__version__,
                "thread_count": original_threads,
                "mkldnn_enabled_for_candidate": True,
                "input": tensor_meta(norm_bf16, "[token, hidden]", "bf16"),
                "weight": tensor_meta(v_weight_bf16, "[v_output_feature, hidden]", "bf16"),
                "bias": tensor_meta(v_bias_bf16, "[v_output_feature]", "bf16") if v_bias_bf16 is not None else None,
                "output": tensor_meta(onednn_v, "[token, v_feature]", "bf16"),
                "output_digest": digest_tensor(onednn_v, torch),
            },
        },
        "v_provenance_metrics": v_provenance_metrics,
        "weighted_v_sum_metrics": {
            "all_flattened_features": weighted_metrics,
            "per_query_head_summary": per_head_summary(local_weighted.reshape(64, 64), official_weighted_values.reshape(64, 64), torch),
            "onednn_oracle_v_weighted_sum_vs_official": onednn_weighted_metrics,
        },
        "causality_variant_table": causality_variants,
        "focused_mismatch_trace": focused_trace,
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


def run_main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.verbose_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ONEDNN_VERBOSE"] = "1"
    env["DNNL_VERBOSE"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-only",
        "--model-root",
        str(args.model_root),
        "--attention-probs-proof",
        str(args.attention_probs_proof),
        "--q-candidate",
        str(args.q_candidate),
        "--k-candidate",
        str(args.k_candidate),
        "--rmsnorm-replay",
        str(args.rmsnorm_replay),
        "--local-v-capture",
        str(args.local_v_capture),
        "--official-weighted-v",
        str(args.official_weighted_v),
        "--output",
        str(args.output),
        "--verbose-log",
        str(args.verbose_log),
        "--device",
        args.device,
    ]
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True)
    verbose_text = "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr
    args.verbose_log.write_text(verbose_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"weighted V sum diagnostic failed; see {args.verbose_log}")
    output = load_json(args.output)
    output["onednn_verbose_log_path"] = str(args.verbose_log)
    output["onednn_verbose_primitive_lines_sample"] = [
        line for line in verbose_text.splitlines() if "onednn_verbose" in line or "dnnl_verbose" in line
    ][:80]
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def main():
    args = parse_args()
    if args.probe_only:
        run_probe(args)
    else:
        run_main(args)


if __name__ == "__main__":
    main()
