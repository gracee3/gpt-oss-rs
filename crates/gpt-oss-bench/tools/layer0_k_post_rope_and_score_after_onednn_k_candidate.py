#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_projection_onednn_oracle_helper_proof as proof
import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


LOCAL_SCORE_INPUT_BUNDLE = Path(
    ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/"
    "developer-message.runner-layer0-attention-score-input-bundle.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K post-RoPE and score proof after scoped oneDNN K candidate."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-weight-arithmetic", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--grouped-post-rope-official", type=Path, required=True)
    parser.add_argument("--q-post-rope-official", type=Path, required=True)
    parser.add_argument("--score-official", type=Path, required=True)
    parser.add_argument("--q-provenance-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_meta(tensor, layout: str):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "layout": layout,
    }


def bf16_bits_from_float(value: float) -> str:
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


def compare_tensor(lhs, rhs, torch, location_kind: str):
    lhs_f32 = lhs.to(torch.float32)
    rhs_f32 = rhs.to(torch.float32)
    diff = (lhs_f32 - rhs_f32).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        coords = mismatch.nonzero(as_tuple=False)
        first_coords = [int(v.item()) for v in coords[0]]
        worst_flat = int(diff.reshape(-1).argmax().item())
        worst_coords = []
        remainder = worst_flat
        for size in reversed(list(diff.shape)):
            worst_coords.append(remainder % int(size))
            remainder //= int(size)
        worst_coords = list(reversed(worst_coords))

        def location(coords_list):
            lhs_value = float(lhs_f32[tuple(coords_list)].item())
            rhs_value = float(rhs_f32[tuple(coords_list)].item())
            entry = {
                "location_kind": location_kind,
                "indices": coords_list,
                "lhs_value": lhs_value,
                "rhs_value": rhs_value,
                "abs_diff": abs(lhs_value - rhs_value),
            }
            if location_kind == "token_head_lane" and len(coords_list) == 3:
                token, head, lane = coords_list
                entry.update(
                    {
                        "token_index": token,
                        "kv_head_index": head,
                        "head_dim_lane": lane,
                    }
                )
            elif location_kind == "score_head_key" and len(coords_list) == 2:
                score_head, key_pos = coords_list
                entry.update({"score_head_index": score_head, "key_position": key_pos})
            elif location_kind == "token_feature" and len(coords_list) == 2:
                token, feature = coords_list
                entry.update(
                    {
                        "token_index": token,
                        "k_feature_index": feature,
                        "kv_head_index": feature // 64,
                        "head_dim_lane": feature % 64,
                    }
                )
            return entry

        first = location(first_coords)
        worst = location(worst_coords)
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs_f32, rhs_f32)),
        "mismatching_element_count": int(mismatch.sum().item()),
        "first_differing_location": first,
        "worst_differing_location": worst,
    }


def build_onednn_candidate(args, torch, device):
    scoped_source = load_json(args.source)
    if scoped_source["exact_case"]["case_id"] != "developer-message-user-smoke":
        raise ValueError("source artifact is not the exact smoke case")
    if scoped_source["classification"] != "scoped_onednn_oracle_k_projection_helper_fix_proven":
        raise ValueError("source artifact is not the expected scoped candidate proof")

    primitive_source = Path(scoped_source["source_artifact_paths"]["onednn_primitive_reproduction"])
    grouped_source = Path(scoped_source["source_artifact_paths"]["grouped_source_breakdown"])
    primitive_args = argparse.Namespace(
        source=primitive_source,
        official_weight_arithmetic=args.official_weight_arithmetic,
        rmsnorm_replay=args.rmsnorm_replay,
        grouped_source_breakdown=grouped_source,
    )
    (
        _primitive,
        _backend_source,
        _rmsnorm,
        _extraction,
        norm,
        k_weight,
        official,
        _official_weight_status,
        _official_projection_status,
        _grouped_status,
        _official_projection_status_path,
    ) = proof.load_inputs(primitive_args, torch)

    original_threads = torch.get_num_threads()
    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = True
    norm_bf16 = norm.to(device=device, dtype=torch.bfloat16).contiguous()
    weight_bf16 = k_weight.to(device=device, dtype=torch.bfloat16).contiguous()
    with torch.inference_mode():
        candidate = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn
    return scoped_source, candidate, official.to(torch.float32), {
        "torch_version": torch.__version__,
        "thread_count": int(original_threads),
        "mkldnn_enabled_for_candidate": True,
        "input": tensor_meta(norm_bf16, "[token, hidden]"),
        "weight": tensor_meta(weight_bf16, "[k_feature, hidden]; torch F.linear uses weight.T"),
        "output": tensor_meta(candidate, "[token, k_feature]"),
    }


def apply_model_rope(args, candidate_grouped, torch, device):
    AttentionBlock = ModelConfig = Checkpoint = None
    torch_import = base.import_gpt_oss()
    AttentionBlock, ModelConfig, Checkpoint = torch_import[1], torch_import[2], torch_import[3]
    model = base.load_layer0_model(
        args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint
    )
    attn = model.attn
    heads_per_kv = attn.num_attention_heads // attn.num_key_value_heads
    q_dummy = torch.zeros(
        (
            candidate_grouped.shape[0],
            attn.num_key_value_heads,
            heads_per_kv,
            attn.head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.inference_mode():
        _q_rope, k_rope = attn.rope(q_dummy, candidate_grouped.to(torch.bfloat16).to(device))
    return k_rope.contiguous()


def build_score(q_post_rope, k_post_rope_grouped, torch):
    token_count = int(k_post_rope_grouped.shape[0])
    num_kv_heads = int(k_post_rope_grouped.shape[1])
    head_dim = int(k_post_rope_grouped.shape[2])
    heads_per_kv = int(q_post_rope.shape[2])
    final_token = token_count - 1
    q_final = q_post_rope.reshape(token_count, num_kv_heads, heads_per_kv, head_dim)[
        final_token
    ].to(torch.float32)
    k_expanded = (
        k_post_rope_grouped[:, :, None, :]
        .expand(token_count, num_kv_heads, heads_per_kv, head_dim)
        .to(torch.float32)
    )
    score = torch.einsum("hmd,khmd->hmk", q_final, k_expanded)
    score = score * (1.0 / (head_dim**0.5))
    return score.to(torch.bfloat16).to(torch.float32).reshape(num_kv_heads * heads_per_kv, token_count)


def q_provenance_status(args, torch):
    if not args.q_provenance_artifact.exists():
        return {
            "q_provenance_guard_available": False,
            "reason": f"artifact missing: {args.q_provenance_artifact}",
        }, None
    artifact = load_json(args.q_provenance_artifact)
    metrics = artifact.get("local_bundle_q_vs_official_bundle_q")
    return {
        "q_provenance_guard_available": True,
        "source_artifact_path": str(args.q_provenance_artifact),
        "source_classification": artifact.get("classification"),
        "local_bundle_q_vs_official_bundle_q": metrics,
        "closest_q_upstream_seam": artifact.get("closest_q_upstream_seam"),
        "earliest_q_upstream_seam_already_mismatched": artifact.get(
            "earliest_q_upstream_seam_already_mismatched"
        ),
        "q_provenance_stale_or_mismatched": bool(metrics and not metrics.get("matched", False)),
    }, artifact


def load_local_q_if_available(torch):
    if not LOCAL_SCORE_INPUT_BUNDLE.exists():
        return None, {
            "available": False,
            "reason": f"local score input bundle missing: {LOCAL_SCORE_INPUT_BUNDLE}",
        }
    bundle = load_json(LOCAL_SCORE_INPUT_BUNDLE)
    q = bundle.get("q_final_token_post_rope_grouped", {})
    values = q.get("values")
    shape = q.get("shape")
    if not values or shape != [8, 8, 64]:
        return None, {
            "available": False,
            "reason": "local score input bundle Q tensor missing or unexpected shape",
            "source_artifact_path": str(LOCAL_SCORE_INPUT_BUNDLE),
        }
    return torch.tensor(values, dtype=torch.float32).reshape(1, 8, 8, 64), {
        "available": True,
        "source_artifact_path": str(LOCAL_SCORE_INPUT_BUNDLE),
        "shape": shape,
        "layout": q.get("layout"),
        "value_dtype": q.get("value_dtype"),
    }


def run_probe(args):
    torch, _AttentionBlock, _ModelConfig, _Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    scoped_source, candidate_projection, official_pre, candidate_meta = build_onednn_candidate(
        args, torch, device
    )
    official_post_artifact = load_json(args.grouped_post_rope_official)
    official_q_artifact = load_json(args.q_post_rope_official)
    official_score_artifact = load_json(args.score_official)
    if (
        official_post_artifact["case_id"] != "developer-message-user-smoke"
        or official_q_artifact["case_id"] != "developer-message-user-smoke"
        or official_score_artifact["case_id"] != "developer-message-user-smoke"
    ):
        raise ValueError("one or more official artifacts are not the exact smoke case")

    candidate_grouped = candidate_projection.to(torch.float32).reshape(74, 8, 64)
    official_pre_grouped = official_pre.reshape(74, 8, 64)
    official_post = torch.tensor(
        official_post_artifact["values"], dtype=torch.float32, device=device
    ).reshape(74, 8, 64)
    official_q = torch.tensor(
        official_q_artifact["values"], dtype=torch.float32, device=device
    ).reshape(74, 8, 8, 64)
    official_score = torch.tensor(
        official_score_artifact["values"], dtype=torch.float32, device=device
    ).reshape(64, 74)

    candidate_post = apply_model_rope(args, candidate_grouped, torch, device).to(torch.float32)
    official_q_candidate_k_score = build_score(official_q, candidate_post, torch)
    local_q, local_q_meta = load_local_q_if_available(torch)
    if local_q is not None:
        local_q = local_q.to(device)
        local_q_all_token_shape = torch.zeros_like(official_q)
        local_q_all_token_shape[-1] = local_q.reshape(8, 8, 64)
        score_for_main_comparison = build_score(local_q_all_token_shape, candidate_post, torch)
        score_source = "existing local final-token Q score-input bundle with scoped oneDNN K candidate"
    else:
        score_for_main_comparison = official_q_candidate_k_score
        score_source = "official final-token Q with scoped oneDNN K candidate because local Q bundle was unavailable"

    pre_guard = compare_tensor(candidate_grouped, official_pre_grouped, torch, "token_head_lane")
    post_metrics = compare_tensor(candidate_post, official_post, torch, "token_head_lane")
    score_metrics = compare_tensor(score_for_main_comparison, official_score, torch, "score_head_key")
    k_isolated_score_metrics = compare_tensor(
        official_q_candidate_k_score, official_score, torch, "score_head_key"
    )
    q_guard, _q_artifact = q_provenance_status(args, torch)
    q_gap = bool(q_guard.get("q_provenance_stale_or_mismatched", False))

    if not pre_guard["matched"]:
        classification = "onednn_k_candidate_pre_rope_guard_regressed"
        earliest = "grouped_pre_rope_k_guard"
        next_step = "repair the scoped oneDNN K candidate guard before rerunning K RoPE"
    elif not post_metrics["matched"]:
        classification = "k_rope_path_mismatch_after_onednn_k_candidate"
        earliest = "grouped_post_rope_k"
        next_step = "inspect K RoPE arithmetic/path using candidate pre-RoPE K"
    elif q_gap and not score_metrics["matched"]:
        classification = "score_consumer_blocked_by_q_provenance_gap"
        earliest = "q_provenance_guard"
        next_step = "inspect Q provenance or QK score arithmetic, whichever is earliest from available metrics"
    elif not score_metrics["matched"]:
        classification = "first_attention_score_consumer_still_mismatches_after_k_candidate"
        earliest = "layer0_final_token_raw_scaled_qk_logits_pre_mask"
        next_step = "inspect Q provenance or QK score arithmetic, whichever is earliest from available metrics"
    else:
        classification = "onednn_k_candidate_clears_k_rope_and_first_score_consumer"
        earliest = None
        next_step = "decide whether to promote a scoped projection-policy runtime candidate or continue one seam downstream"

    output = {
        "schema_version": "runtime_forward_layer0_k_post_rope_and_score_after_onednn_k_candidate_status/v1",
        "mode": "k-post-rope-and-score-after-onednn-k-candidate-status",
        "exact_case": scoped_source["exact_case"],
        "source_artifact_paths": {
            "onednn_oracle_scoped_helper_fix": str(args.source),
            "onednn_oracle_helper_proof": scoped_source["source_artifact_paths"].get(
                "onednn_oracle_helper_proof"
            ),
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
            "official_grouped_post_rope_k": str(args.grouped_post_rope_official),
            "official_q_post_rope_before_attention": str(args.q_post_rope_official),
            "official_final_token_raw_scaled_qk_logits_pre_mask": str(args.score_official),
            "q_provenance_artifact": str(args.q_provenance_artifact),
        },
        "candidate_is_bench_proof_only": True,
        "active_k_candidate_strategy": scoped_source.get(
            "candidate_helper_projection_strategy",
            "bench-only full-shape oneDNN/MKLDNN BF16 torch.nn.functional.linear oracle output reshaped to grouped [token, kv_head, lane]",
        ),
        "onednn_candidate_construction_metadata": candidate_meta,
        "pre_rope_k_guard_metrics": {
            **pre_guard,
            "local_tensor": tensor_meta(candidate_grouped, "[token, kv_head, lane]"),
            "official_tensor": tensor_meta(official_pre_grouped, "[token, kv_head, lane]"),
        },
        "grouped_post_rope_k_metrics": {
            **post_metrics,
            "candidate_tensor": tensor_meta(candidate_post, "[token, kv_head, lane]"),
            "official_tensor": tensor_meta(official_post, "[token, kv_head, lane]"),
            "application_path": "model.attn.rope applied to scoped oneDNN candidate K heads in this bench/proof diagnostic",
        },
        "first_raw_scaled_qk_score_metrics": {
            **score_metrics,
            "score_source": score_source,
            "official_tensor": tensor_meta(official_score, "[score_head, key_position]"),
            "candidate_tensor": tensor_meta(score_for_main_comparison, "[score_head, key_position]"),
        },
        "k_isolated_score_with_official_q_metrics": {
            **k_isolated_score_metrics,
            "score_source": "official final-token Q post-RoPE with scoped oneDNN K candidate",
        },
        "q_provenance_guard_status": {
            **q_guard,
            "local_q_input_for_main_score": local_q_meta,
        },
        "first_worst_mismatch_locations": {
            "pre_rope_k_guard": {
                "first": pre_guard["first_differing_location"],
                "worst": pre_guard["worst_differing_location"],
            },
            "grouped_post_rope_k": {
                "first": post_metrics["first_differing_location"],
                "worst": post_metrics["worst_differing_location"],
            },
            "layer0_final_token_raw_scaled_qk_logits_pre_mask": {
                "first": score_metrics["first_differing_location"],
                "worst": score_metrics["worst_differing_location"],
            },
        },
        "earliest_remaining_mismatching_seam": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "runtime_forward_k_post_rope_and_score_after_onednn_k_candidate_status_now": (
            "runtime-forward K post-RoPE and score after scoped oneDNN K candidate status now: "
            f"pre_rope matched={pre_guard['matched']} max_abs={pre_guard['max_abs_diff']} "
            f"mean_abs={pre_guard['mean_abs_diff']}; post_rope matched={post_metrics['matched']} "
            f"max_abs={post_metrics['max_abs_diff']} mean_abs={post_metrics['mean_abs_diff']}; "
            f"score matched={score_metrics['matched']} max_abs={score_metrics['max_abs_diff']} "
            f"mean_abs={score_metrics['mean_abs_diff']}; q_guard_available={q_guard.get('q_provenance_guard_available')} "
            f"q_gap={q_gap}; classification={classification}; next bounded step: {next_step}."
        ),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


def run_main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.verbose_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ONEDNN_VERBOSE"] = "1"
    env["DNNL_VERBOSE"] = "1"
    probe_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-only",
        "--model-root",
        str(args.model_root),
        "--source",
        str(args.source),
        "--official-weight-arithmetic",
        str(args.official_weight_arithmetic),
        "--rmsnorm-replay",
        str(args.rmsnorm_replay),
        "--grouped-post-rope-official",
        str(args.grouped_post_rope_official),
        "--q-post-rope-official",
        str(args.q_post_rope_official),
        "--score-official",
        str(args.score_official),
        "--q-provenance-artifact",
        str(args.q_provenance_artifact),
        "--output",
        str(args.output),
        "--verbose-log",
        str(args.verbose_log),
        "--device",
        args.device,
    ]
    completed = subprocess.run(probe_cmd, env=env, text=True, capture_output=True)
    verbose_text = (
        "COMMAND: "
        + " ".join(probe_cmd)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr
    )
    args.verbose_log.write_text(verbose_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"K post-RoPE and score after oneDNN K candidate failed; see {args.verbose_log}"
        )
    output = json.loads(args.output.read_text(encoding="utf-8"))
    output["onednn_verbose_log_path"] = str(args.verbose_log)
    output["onednn_verbose_primitive_lines_sample"] = [
        line
        for line in verbose_text.splitlines()
        if "onednn_verbose" in line or "dnnl_verbose" in line
    ][:60]
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
