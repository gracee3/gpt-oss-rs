#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_post_rope_and_score_after_onednn_k_candidate as kdiag
import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


LOCAL_SCORE_INPUT_BUNDLE = Path(
    ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/"
    "developer-message.runner-layer0-attention-score-input-bundle.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Q provenance source breakdown after scoped oneDNN K candidate."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-k-post-score", type=Path, required=True)
    parser.add_argument("--source-k-candidate", type=Path, required=True)
    parser.add_argument("--source-rmsnorm-replay-proof", type=Path, required=True)
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


def tensor_meta(tensor, layout: str, serialization_dtype=None):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
    }


def bf16_bits(value: float) -> str:
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
            if location_kind == "token_feature" and len(coords_list) == 2:
                token, feature = coords_list
                entry.update(
                    {
                        "token_index": token,
                        "q_feature_index": feature,
                        "q_head_index": feature // 64,
                        "head_dim_lane": feature % 64,
                    }
                )
            elif location_kind == "q_head_lane" and len(coords_list) == 3:
                q_head, heads_per_kv, lane = coords_list
                entry.update(
                    {
                        "q_head_index": q_head * 8 + heads_per_kv,
                        "kv_head_index": q_head,
                        "heads_per_kv_index": heads_per_kv,
                        "head_dim_lane": lane,
                    }
                )
            elif location_kind == "score_head_key" and len(coords_list) == 2:
                score_head, key_pos = coords_list
                entry.update({"score_head_index": score_head, "key_position": key_pos})
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


def digest_tensor(tensor, torch):
    import hashlib
    import struct

    data = tensor.detach().to(torch.float32).cpu().reshape(-1)
    h = hashlib.sha256()
    for value in data[: min(4096, data.numel())]:
        h.update(struct.pack("<f", float(value.item())))
    h.update(str(tuple(tensor.shape)).encode())
    h.update(str(tensor.dtype).encode())
    return "sha256-prefix4096:" + h.hexdigest()


def build_exact_case(args, torch, device):
    k_post_score = load_json(args.source_k_post_score)
    if k_post_score["exact_case"]["case_id"] != "developer-message-user-smoke":
        raise ValueError("K post-score source is not the exact smoke case")
    if k_post_score["classification"] != "score_consumer_blocked_by_q_provenance_gap":
        raise ValueError("K post-score source is not the expected Q gap state")

    k_candidate_args = argparse.Namespace(
        model_root=args.model_root,
        source=args.source_k_candidate,
        official_weight_arithmetic=args.official_weight_arithmetic,
        rmsnorm_replay=args.rmsnorm_replay,
        device=args.device,
    )
    scoped_source, candidate_k_projection, official_k_pre, candidate_k_meta = (
        kdiag.build_onednn_candidate(k_candidate_args, torch, device)
    )
    candidate_k_grouped = candidate_k_projection.to(torch.float32).reshape(74, 8, 64)
    official_k_grouped = official_k_pre.reshape(74, 8, 64)
    candidate_k_post = kdiag.apply_model_rope(
        k_candidate_args, candidate_k_grouped, torch, device
    ).to(torch.float32)

    rmsnorm = load_json(args.rmsnorm_replay)
    norm_output = torch.tensor(
        rmsnorm["policy_outputs_f32"][
            "manual_bf16_input_bf16_weight_f32_reduction_bf16_output"
        ],
        dtype=torch.float32,
        device=device,
    ).reshape(74, 2880)
    official_norm_output = torch.tensor(
        rmsnorm["official_module_output_f32"], dtype=torch.float32, device=device
    ).reshape(74, 2880)

    torch_import = base.import_gpt_oss()
    AttentionBlock, ModelConfig, Checkpoint = torch_import[1], torch_import[2], torch_import[3]
    model = base.load_layer0_model(
        args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint
    )
    return (
        k_post_score,
        scoped_source,
        candidate_k_meta,
        candidate_k_grouped,
        official_k_grouped,
        candidate_k_post,
        norm_output,
        official_norm_output,
        model,
    )


def load_local_q(torch, device):
    if not LOCAL_SCORE_INPUT_BUNDLE.exists():
        return None, {
            "available": False,
            "reason": f"missing local score input bundle: {LOCAL_SCORE_INPUT_BUNDLE}",
        }
    artifact = load_json(LOCAL_SCORE_INPUT_BUNDLE)
    q = artifact.get("q_final_token_post_rope_grouped", {})
    values = q.get("values")
    shape = q.get("shape")
    if values is None or shape != [8, 8, 64]:
        return None, {
            "available": False,
            "reason": "q_final_token_post_rope_grouped is missing or not [8, 8, 64]",
            "source_artifact_path": str(LOCAL_SCORE_INPUT_BUNDLE),
        }
    tensor = torch.tensor(values, dtype=torch.float32, device=device).reshape(8, 8, 64)
    return tensor, {
        "available": True,
        "source_artifact_path": str(LOCAL_SCORE_INPUT_BUNDLE),
        "shape": shape,
        "layout": q.get("layout"),
        "value_dtype": q.get("value_dtype"),
    }


def build_score_from_final_q(final_q_grouped, k_post_rope_grouped, torch):
    token_count = int(k_post_rope_grouped.shape[0])
    num_kv_heads = int(k_post_rope_grouped.shape[1])
    heads_per_kv = int(final_q_grouped.shape[1])
    head_dim = int(k_post_rope_grouped.shape[2])
    q_final = final_q_grouped.to(torch.float32)
    k_expanded = (
        k_post_rope_grouped[:, :, None, :]
        .expand(token_count, num_kv_heads, heads_per_kv, head_dim)
        .to(torch.float32)
    )
    score = torch.einsum("hmd,khmd->hmk", q_final, k_expanded)
    score = score * (1.0 / (head_dim**0.5))
    return score.to(torch.bfloat16).to(torch.float32).reshape(num_kv_heads * heads_per_kv, token_count)


def run_probe(args):
    torch, _AttentionBlock, _ModelConfig, _Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    (
        k_post_score,
        scoped_source,
        candidate_k_meta,
        candidate_k_grouped,
        official_k_grouped,
        candidate_k_post,
        norm_output,
        official_norm_output,
        model,
    ) = build_exact_case(args, torch, device)

    official_q_post_artifact = load_json(args.q_post_rope_official)
    official_score_artifact = load_json(args.score_official)
    if (
        official_q_post_artifact["case_id"] != "developer-message-user-smoke"
        or official_score_artifact["case_id"] != "developer-message-user-smoke"
    ):
        raise ValueError("official Q or score artifact is not the exact smoke case")

    official_q_post_flat = torch.tensor(
        official_q_post_artifact["values"], dtype=torch.float32, device=device
    ).reshape(74, 4096)
    official_q_post = official_q_post_flat.reshape(74, 8, 8, 64)
    official_final_q = official_q_post[-1]
    official_score = torch.tensor(
        official_score_artifact["values"], dtype=torch.float32, device=device
    ).reshape(64, 74)

    attn = model.attn
    q_dim = int(attn.num_attention_heads * attn.head_dim)
    kv_dim = int(attn.num_key_value_heads * attn.head_dim)
    q_weight = attn.qkv.weight[:q_dim, :].detach().to(torch.float32).contiguous()
    q_bias = None
    if getattr(attn.qkv, "bias", None) is not None:
        q_bias = attn.qkv.bias[:q_dim].detach().to(torch.float32).contiguous()
    norm_bf16 = norm_output.to(torch.bfloat16).contiguous()
    q_weight_bf16 = q_weight.to(torch.bfloat16).contiguous()
    q_bias_bf16 = q_bias.to(torch.bfloat16).contiguous() if q_bias is not None else None

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        qkv = attn.qkv(norm_bf16)
        official_module_q_pre = qkv[:, :q_dim].contiguous()
        manual_q_pre = torch.nn.functional.linear(norm_bf16, q_weight_bf16, q_bias_bf16).contiguous()
        q_pre_grouped = manual_q_pre.reshape(74, 8, 8, 64)
        dummy_k = torch.zeros((74, 8, 64), dtype=torch.bfloat16, device=device)
        q_rope_from_manual, _ = attn.rope(q_pre_grouped, dummy_k)
        q_rope_from_module, _ = attn.rope(official_module_q_pre.reshape(74, 8, 8, 64), dummy_k)
    torch.backends.mkldnn.enabled = original_mkldnn

    local_final_q, local_q_meta = load_local_q(torch, device)
    if local_final_q is not None:
        local_q_post_metrics = compare_tensor(local_final_q, official_final_q, torch, "q_head_lane")
        current_local_q_score = build_score_from_final_q(local_final_q, candidate_k_post, torch)
    else:
        local_q_post_metrics = {
            "available": False,
            "reason": local_q_meta["reason"],
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "matched": False,
            "first_differing_location": None,
            "worst_differing_location": None,
        }
        current_local_q_score = None

    official_q_candidate_k_score = build_score_from_final_q(
        official_final_q, candidate_k_post, torch
    )
    score_current_metrics = (
        compare_tensor(current_local_q_score, official_score, torch, "score_head_key")
        if current_local_q_score is not None
        else {
            "available": False,
            "reason": "local final-token Q was unavailable",
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "matched": False,
            "first_differing_location": None,
            "worst_differing_location": None,
        }
    )
    score_official_q_metrics = compare_tensor(
        official_q_candidate_k_score, official_score, torch, "score_head_key"
    )

    rmsnorm_guard = {
        "authoritative_rmsnorm_replay_vs_official_layer0_attn_norm_output": compare_tensor(
            norm_output, official_norm_output, torch, "token_feature"
        ),
        "fixed_live_rmsnorm_vs_authoritative_replay": load_json(
            args.source_k_post_score
        ).get("q_provenance_guard_status", {}).get(
            "upstream_attn_norm_provenance_metrics",
            {
                "available": False,
                "reason": "not present in source K post-score artifact; informational guard is unavailable",
            },
        ),
    }
    k_guard = {
        "candidate_pre_rope_k_vs_official": compare_tensor(
            candidate_k_grouped, official_k_grouped, torch, "token_head_lane"
        ),
        "candidate_post_rope_k_vs_official": k_post_score["grouped_post_rope_k_metrics"],
    }
    q_projection_metrics = {
        "official_module_q_projection_vs_manual_q_projection": {
            **compare_tensor(official_module_q_pre, manual_q_pre, torch, "token_feature"),
            "lhs": "official PyTorch attn.qkv Q slice",
            "rhs": "torch.nn.functional.linear(authoritative_norm_bf16, q_weight_bf16, q_bias)",
            "local_runtime_helper_q_projection_available": False,
            "local_unavailable_reason": (
                "no exact all-token runtime/helper Q pre-RoPE projection artifact is available in this bounded mode"
            ),
        }
    }
    q_layout_metrics = {
        "official_q_projection_flat_vs_grouped_roundtrip": compare_tensor(
            manual_q_pre, q_pre_grouped.reshape(74, 4096), torch, "token_feature"
        ),
        "local_runtime_q_layout_available": False,
        "local_unavailable_reason": (
            "available local Q artifact is final-token post-RoPE grouped only, not pre-RoPE layout"
        ),
    }
    q_rope_metrics = {
        "official_manual_q_rope_vs_official_q_post_rope": compare_tensor(
            q_rope_from_manual.to(torch.float32), official_q_post, torch, "token_feature"
        ),
        "official_module_q_rope_vs_official_q_post_rope": compare_tensor(
            q_rope_from_module.to(torch.float32), official_q_post, torch, "token_feature"
        ),
        "local_final_token_q_post_rope_vs_official": {
            **local_q_post_metrics,
            "local_tensor": local_q_meta,
            "official_tensor": tensor_meta(
                official_final_q, "[kv_head, heads_per_kv, head_dim]", "official f32 JSON values"
            ),
        },
    }

    q_guard = k_post_score["q_provenance_guard_status"]
    q_weight_meta = {
        "runtime_fused_attn_qkv_weight_shape": list(attn.qkv.weight.shape),
        "runtime_fused_attn_qkv_weight_dtype": str(attn.qkv.weight.dtype).replace("torch.", ""),
        "runtime_fused_attn_qkv_weight_stride": list(attn.qkv.weight.stride()),
        "q_slice_offsets_features": [0, q_dim],
        "q_slice_shape": list(q_weight.shape),
        "q_slice_orientation": "row-major [q_output_feature, hidden]; torch F.linear uses weight.T",
        "q_slice_stride": list(q_weight.stride()),
        "q_slice_digest": digest_tensor(q_weight, torch),
        "official_q_slice_metadata_available": True,
        "official_q_slice_source": "same loaded PyTorch layer0 attn.qkv weight used for official Q projection",
    }
    q_bias_meta = {
        "present": q_bias is not None,
        "shape": list(q_bias.shape) if q_bias is not None else None,
        "dtype": str(q_bias.dtype).replace("torch.", "") if q_bias is not None else None,
        "max_abs": float(q_bias.abs().max().item()) if q_bias is not None else None,
        "mean_abs": float(q_bias.abs().mean().item()) if q_bias is not None else None,
        "all_zero": bool(torch.equal(q_bias, torch.zeros_like(q_bias))) if q_bias is not None else True,
        "policy_finding": (
            "Q bias is present and applied in both official module/manual paths"
            if q_bias is not None
            else "No Q bias tensor is present on the loaded official layer0 attn.qkv module"
        ),
    }

    focused_trace = {}
    if local_final_q is not None and not local_q_post_metrics["matched"]:
        loc = local_q_post_metrics["worst_differing_location"]
        kv_head = int(loc["kv_head_index"])
        hpk = int(loc["heads_per_kv_index"])
        lane = int(loc["head_dim_lane"])
        q_feature = (kv_head * 8 + hpk) * 64 + lane
        local_post = float(local_final_q[kv_head, hpk, lane].item())
        official_post = float(official_final_q[kv_head, hpk, lane].item())
        score_loc = score_current_metrics.get("worst_differing_location")
        focused_trace = {
            "token_index": 73,
            "q_feature_index": q_feature,
            "kv_head_index": kv_head,
            "heads_per_kv_index": hpk,
            "head_dim_lane": lane,
            "local_q_projection_value": None,
            "official_q_projection_value": float(manual_q_pre[-1, q_feature].to(torch.float32).item()),
            "local_q_post_rope_value": local_post,
            "official_q_post_rope_value": official_post,
            "local_minus_official": local_post - official_post,
            "abs_diff": abs(local_post - official_post),
            "one_bf16_ulp_or_larger": abs(local_post - official_post) >= 0.0078125,
            "same_token_lane_contributes_to_worst_raw_score_mismatch": bool(
                score_loc
                and score_loc.get("score_head_index") == kv_head * 8 + hpk
            ),
            "local_q_projection_note": "local runtime Q pre-RoPE projection value is not available in existing exact artifacts",
        }

    if not rmsnorm_guard[
        "authoritative_rmsnorm_replay_vs_official_layer0_attn_norm_output"
    ]["matched"]:
        classification = "q_path_blocked_by_rmsnorm_provenance_regression"
        earliest = "layer0_attn_norm_output"
        next_step = "restore authoritative RMSNorm provenance before continuing Q source breakdown"
    elif not q_projection_metrics[
        "official_module_q_projection_vs_manual_q_projection"
    ]["matched"]:
        classification = "q_projection_arithmetic_or_weight_policy_mismatch"
        earliest = "layer0_q_projection_before_rope"
        next_step = "inspect Q projection arithmetic/weight policy with authoritative RMSNorm input"
    elif not q_rope_metrics["official_manual_q_rope_vs_official_q_post_rope"]["matched"]:
        classification = "q_rope_path_mismatch"
        earliest = "layer0_q_post_rope_before_attention"
        next_step = "inspect Q RoPE arithmetic/path with authoritative Q projection input"
    elif (
        local_final_q is not None
        and not local_q_post_metrics["matched"]
        and score_official_q_metrics["matched"]
    ):
        classification = "q_provenance_mismatch_confirmed_as_score_blocker"
        earliest = "layer0_q_post_rope_before_attention_final_token_grouped"
        next_step = "localize local runtime Q before/post RoPE against the official Q path before inspecting QK score arithmetic"
    elif score_current_metrics.get("matched") is False:
        classification = "qk_score_arithmetic_mismatch_after_qk_provenance_clear"
        earliest = "layer0_final_token_raw_scaled_qk_logits_pre_mask"
        next_step = "inspect QK score arithmetic with candidate K after Q provenance is cleared"
    else:
        classification = "q_provenance_and_first_score_consumer_cleared_or_prior_measurement_bug"
        earliest = None
        next_step = "continue one seam downstream or decide whether to promote the scoped projection-policy candidate"

    output = {
        "schema_version": "runtime_forward_layer0_q_provenance_after_k_candidate_source_breakdown_status/v1",
        "mode": "q-provenance-after-k-candidate-source-breakdown-status",
        "exact_case": k_post_score["exact_case"],
        "source_artifact_paths": {
            "k_post_rope_and_score_after_onednn_k_candidate": str(args.source_k_post_score),
            "onednn_k_candidate_proof": str(args.source_k_candidate),
            "authoritative_rmsnorm_replay_proof": str(args.source_rmsnorm_replay_proof),
            "authoritative_rmsnorm_replay_values": str(args.rmsnorm_replay),
            "official_q_post_rope_before_attention": str(args.q_post_rope_official),
            "official_score": str(args.score_official),
            "q_provenance_guard": str(args.q_provenance_artifact),
        },
        "active_k_candidate_strategy": k_post_score["active_k_candidate_strategy"],
        "candidate_is_bench_proof_only": True,
        "rmsnorm_provenance_metrics": rmsnorm_guard,
        "k_guard_metrics": k_guard,
        "q_weight_slice_bias_metadata": {
            "q_weight_slice": q_weight_meta,
            "q_bias": q_bias_meta,
            "finding": (
                "Q weight slice comes from the official loaded layer0 attn.qkv module; no separate runtime-vs-official Q weight mismatch is testable from existing artifacts in this bounded mode. Bias is non-causal."
            ),
        },
        "q_projection_metrics": q_projection_metrics,
        "q_layout_metrics": q_layout_metrics,
        "q_rope_metrics": q_rope_metrics,
        "score_confirmation_metrics": {
            "current_local_q_plus_candidate_k_vs_official": score_current_metrics,
            "official_q_plus_candidate_k_vs_official": score_official_q_metrics,
        },
        "existing_q_guard": q_guard,
        "focused_mismatch_trace": focused_trace,
        "earliest_remaining_mismatching_q_operation": earliest,
        "classification": classification,
        "next_bounded_step": next_step,
        "runtime_forward_q_provenance_after_k_candidate_source_breakdown_status_now": (
            "runtime-forward Q provenance after K candidate source breakdown now: "
            f"rmsnorm_guard_matched={rmsnorm_guard['authoritative_rmsnorm_replay_vs_official_layer0_attn_norm_output']['matched']}; "
            f"k_pre_matched={k_guard['candidate_pre_rope_k_vs_official']['matched']} "
            f"k_post_matched={k_guard['candidate_post_rope_k_vs_official']['matched']}; "
            f"q_post_local_matched={local_q_post_metrics.get('matched')} "
            f"q_post_max_abs={local_q_post_metrics.get('max_abs_diff')} "
            f"score_local_q_matched={score_current_metrics.get('matched')} "
            f"score_official_q_matched={score_official_q_metrics['matched']}; "
            f"classification={classification}; next bounded step: {next_step}."
        ),
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
        "--source-k-post-score",
        str(args.source_k_post_score),
        "--source-k-candidate",
        str(args.source_k_candidate),
        "--source-rmsnorm-replay-proof",
        str(args.source_rmsnorm_replay_proof),
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
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True)
    verbose_text = (
        "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr
    )
    args.verbose_log.write_text(verbose_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Q provenance breakdown failed; see {args.verbose_log}")
    output = load_json(args.output)
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
