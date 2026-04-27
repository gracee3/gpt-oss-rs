#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_post_rope_and_score_after_onednn_k_candidate as kdiag
import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


def parse_args():
    parser = argparse.ArgumentParser(description="Localize runtime Q before/after RoPE.")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-q-provenance", type=Path, required=True)
    parser.add_argument("--source-k-post-score", type=Path, required=True)
    parser.add_argument("--source-k-candidate", type=Path, required=True)
    parser.add_argument("--source-rmsnorm-replay-proof", type=Path, required=True)
    parser.add_argument("--official-weight-arithmetic", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--local-q-capture", type=Path, required=True)
    parser.add_argument("--q-post-rope-official", type=Path, required=True)
    parser.add_argument("--score-official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_meta(tensor, layout, serialization_dtype=None):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": serialization_dtype,
        "layout": layout,
    }


def compare_tensor(lhs, rhs, torch, location_kind):
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
        rem = worst_flat
        for size in reversed(list(diff.shape)):
            worst_coords.append(rem % int(size))
            rem //= int(size)
        worst_coords = list(reversed(worst_coords))

        def loc(coords_list):
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
            elif location_kind == "token_kv_hpk_lane" and len(coords_list) == 4:
                token, kv_head, hpk, lane = coords_list
                entry.update(
                    {
                        "token_index": token,
                        "kv_head_index": kv_head,
                        "heads_per_kv_index": hpk,
                        "q_head_index": kv_head * 8 + hpk,
                        "head_dim_lane": lane,
                        "q_feature_index": (kv_head * 8 + hpk) * 64 + lane,
                    }
                )
            elif location_kind == "score_head_key" and len(coords_list) == 2:
                score_head, key_pos = coords_list
                entry.update({"score_head_index": score_head, "key_position": key_pos})
            return entry

        first = loc(first_coords)
        worst = loc(worst_coords)
    token_count = 0
    if mismatch.ndim >= 2:
        token_count = int(mismatch.reshape(mismatch.shape[0], -1).any(dim=1).sum().item())
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs_f32, rhs_f32)),
        "mismatching_token_count": token_count,
        "mismatching_lane_count": int(mismatch.sum().item()),
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


def build_score(final_q_grouped, k_post_rope_grouped, torch):
    token_count = int(k_post_rope_grouped.shape[0])
    num_kv_heads = int(k_post_rope_grouped.shape[1])
    heads_per_kv = int(final_q_grouped.shape[1])
    head_dim = int(k_post_rope_grouped.shape[2])
    k_expanded = (
        k_post_rope_grouped[:, :, None, :]
        .expand(token_count, num_kv_heads, heads_per_kv, head_dim)
        .to(torch.float32)
    )
    score = torch.einsum("hmd,khmd->hmk", final_q_grouped.to(torch.float32), k_expanded)
    score = score * (1.0 / (head_dim**0.5))
    return score.to(torch.bfloat16).to(torch.float32).reshape(num_kv_heads * heads_per_kv, token_count)


def build_k_candidate(args, torch, device):
    k_args = argparse.Namespace(
        model_root=args.model_root,
        source=args.source_k_candidate,
        official_weight_arithmetic=args.official_weight_arithmetic,
        rmsnorm_replay=args.rmsnorm_replay,
        device=args.device,
    )
    scoped_source, candidate_k_projection, official_k_pre, _candidate_meta = (
        kdiag.build_onednn_candidate(k_args, torch, device)
    )
    candidate_k_grouped = candidate_k_projection.to(torch.float32).reshape(74, 8, 64)
    official_k_grouped = official_k_pre.reshape(74, 8, 64)
    candidate_k_post = kdiag.apply_model_rope(k_args, candidate_k_grouped, torch, device).to(torch.float32)
    return scoped_source, candidate_k_grouped, official_k_grouped, candidate_k_post


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    source_q = load_json(args.source_q_provenance)
    if source_q["classification"] != "q_provenance_mismatch_confirmed_as_score_blocker":
        raise ValueError("source Q provenance artifact is not in the expected score-blocker state")
    source_k_post_score = load_json(args.source_k_post_score)
    local_capture = load_json(args.local_q_capture)
    if local_capture["case_id"] != "developer-message-user-smoke":
        raise ValueError("local Q capture is not the exact smoke case")

    rmsnorm = load_json(args.rmsnorm_replay)
    norm_output = torch.tensor(
        rmsnorm["policy_outputs_f32"]["manual_bf16_input_bf16_weight_f32_reduction_bf16_output"],
        dtype=torch.float32,
        device=device,
    ).reshape(74, 2880)
    official_norm = torch.tensor(
        rmsnorm["official_module_output_f32"], dtype=torch.float32, device=device
    ).reshape(74, 2880)

    model = base.load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    attn = model.attn
    q_dim = int(attn.num_attention_heads * attn.head_dim)
    q_weight = attn.qkv.weight[:q_dim, :].detach().to(torch.float32).contiguous()
    q_bias = None
    if getattr(attn.qkv, "bias", None) is not None:
        q_bias = attn.qkv.bias[:q_dim].detach().to(torch.float32).contiguous()
    q_bias_bf16 = q_bias.to(torch.bfloat16).contiguous() if q_bias is not None else None
    norm_bf16 = norm_output.to(torch.bfloat16).contiguous()
    q_weight_bf16 = q_weight.to(torch.bfloat16).contiguous()

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        module_q_pre = attn.qkv(norm_bf16)[:, :q_dim].contiguous()
        manual_q_pre_with_bias = torch.nn.functional.linear(
            norm_bf16, q_weight_bf16, q_bias_bf16
        ).contiguous()
        manual_q_pre_no_bias = torch.nn.functional.linear(norm_bf16, q_weight_bf16, None).contiguous()
        q_pre_grouped = manual_q_pre_with_bias.reshape(74, 8, 8, 64)
        dummy_k = torch.zeros((74, 8, 64), dtype=torch.bfloat16, device=device)
        q_post_from_official_pre, _ = attn.rope(q_pre_grouped, dummy_k)
    torch.backends.mkldnn.enabled = original_mkldnn

    local_q_pre = torch.tensor(
        local_capture["local_q_pre_rope_f32"], dtype=torch.float32, device=device
    ).reshape(74, q_dim)
    local_q_post = torch.tensor(
        local_capture["local_q_post_rope_f32"], dtype=torch.float32, device=device
    ).reshape(74, q_dim)
    local_q_pre_grouped = local_q_pre.reshape(74, 8, 8, 64)
    local_q_post_grouped = local_q_post.reshape(74, 8, 8, 64)

    official_q_post_artifact = load_json(args.q_post_rope_official)
    official_q_post = torch.tensor(
        official_q_post_artifact["values"], dtype=torch.float32, device=device
    ).reshape(74, 8, 8, 64)
    official_score = torch.tensor(
        load_json(args.score_official)["values"], dtype=torch.float32, device=device
    ).reshape(64, 74)

    _scoped_source, candidate_k_grouped, official_k_grouped, candidate_k_post = build_k_candidate(
        args, torch, device
    )
    score_local_q = build_score(local_q_post_grouped[-1], candidate_k_post, torch)
    score_official_q = build_score(official_q_post[-1], candidate_k_post, torch)
    score_official_pre_local_rope = build_score(q_post_from_official_pre[-1].to(torch.float32), candidate_k_post, torch)

    rmsnorm_guard = {
        "authoritative_rmsnorm_replay_vs_official_layer0_attn_norm_output": compare_tensor(
            norm_output, official_norm, torch, "token_feature"
        ),
        "fixed_live_rmsnorm_vs_authoritative_replay": {
            "available": False,
            "reason": "not captured by this mode; prior RMSNorm causality artifacts remain informational",
        },
    }
    k_guard = {
        "candidate_pre_rope_k_vs_official": compare_tensor(
            candidate_k_grouped, official_k_grouped, torch, "token_kv_hpk_lane"
        ),
        "candidate_post_rope_k_vs_official": source_k_post_score["grouped_post_rope_k_metrics"],
    }
    official_q_guard = {
        "official_module_q_projection_vs_manual_f_linear": compare_tensor(
            module_q_pre, manual_q_pre_with_bias, torch, "token_feature"
        ),
        "official_manual_q_rope_vs_official_post_rope_q": compare_tensor(
            q_post_from_official_pre.to(torch.float32), official_q_post, torch, "token_kv_hpk_lane"
        ),
    }
    q_pre_metrics = {
        "local_runtime_q_pre_rope_vs_official_q_projection": {
            **compare_tensor(local_q_pre, manual_q_pre_with_bias, torch, "token_feature"),
            "local_tensor": tensor_meta(local_q_pre, "[token, q_feature]", "runtime f32 debug readout"),
            "official_tensor": tensor_meta(manual_q_pre_with_bias, "[token, q_feature]", "oneDNN/module BF16 output expanded to f32"),
        },
        "explicit_rust_cpu_q_replay_vs_official_q_projection": {
            "available": False,
            "reason": "no exact all-token Rust CPU Q replay with bias exists as an artifact; this mode captures live runtime pre-RoPE Q directly",
        },
        "local_runtime_q_pre_rope_vs_explicit_rust_cpu_replay": {
            "available": False,
            "reason": "explicit all-token Rust CPU Q replay with bias unavailable",
        },
        "official_onednn_manual_q_projection_vs_official_module_q_projection": compare_tensor(
            manual_q_pre_with_bias, module_q_pre, torch, "token_feature"
        ),
    }
    q_bias_comparisons = {
        "local_q_pre_rope_vs_official_projection_without_bias": compare_tensor(
            local_q_pre, manual_q_pre_no_bias, torch, "token_feature"
        ),
        "local_q_pre_rope_vs_official_projection_with_bias": compare_tensor(
            local_q_pre, manual_q_pre_with_bias, torch, "token_feature"
        ),
    }
    q_layout_metrics = {
        "local_flattened_pre_rope_q_vs_official_flattened_q_projection": compare_tensor(
            local_q_pre, manual_q_pre_with_bias, torch, "token_feature"
        ),
        "local_grouped_pre_rope_q_vs_official_grouped_q_projection": compare_tensor(
            local_q_pre_grouped, q_pre_grouped, torch, "token_kv_hpk_lane"
        ),
    }
    q_rope_metrics = {
        "local_q_post_rope_vs_official_q_post_rope_all_tokens": compare_tensor(
            local_q_post_grouped, official_q_post, torch, "token_kv_hpk_lane"
        ),
        "local_final_token_q_post_rope_vs_official": compare_tensor(
            local_q_post_grouped[-1:], official_q_post[-1:], torch, "token_kv_hpk_lane"
        ),
        "official_q_pre_rope_local_runtime_rope_vs_official_q_post_rope": compare_tensor(
            q_post_from_official_pre.to(torch.float32), official_q_post, torch, "token_kv_hpk_lane"
        ),
    }
    score_metrics = {
        "local_q_post_rope_plus_candidate_k_vs_official": compare_tensor(
            score_local_q, official_score, torch, "score_head_key"
        ),
        "official_q_post_rope_plus_candidate_k_vs_official": compare_tensor(
            score_official_q, official_score, torch, "score_head_key"
        ),
        "official_q_pre_rope_local_rope_plus_candidate_k_vs_official": compare_tensor(
            score_official_pre_local_rope, official_score, torch, "score_head_key"
        ),
    }
    q_weight_meta = {
        "runtime_q_slice_range": [0, q_dim],
        "official_q_slice_range": [0, q_dim],
        "q_slice_shape": list(q_weight.shape),
        "q_slice_orientation": "row-major [q_output_feature, hidden]",
        "q_slice_digest": digest_tensor(q_weight, torch),
        "slice_finding": "runtime and official Q slice ranges are the canonical [0, 4096]; direct runtime weight digest comparison is not separately available in this mode",
    }
    q_bias_meta = {
        "runtime_q_bias_presence": "not directly surfaced by runtime trace",
        "official_q_bias_present": q_bias is not None,
        "shape": list(q_bias.shape) if q_bias is not None else None,
        "dtype": str(q_bias.dtype).replace("torch.", "") if q_bias is not None else None,
        "max_abs": float(q_bias.abs().max().item()) if q_bias is not None else None,
        "mean_abs": float(q_bias.abs().mean().item()) if q_bias is not None else None,
        "checksum": digest_tensor(q_bias, torch) if q_bias is not None else None,
        "official_bias_all_zero": bool(torch.equal(q_bias, torch.zeros_like(q_bias))) if q_bias is not None else True,
        "application_finding": (
            "local pre-RoPE Q matches the no-bias official projection"
            if q_bias_comparisons["local_q_pre_rope_vs_official_projection_without_bias"]["matched"]
            else "local pre-RoPE Q does not match the no-bias official projection"
        ),
    }

    worst = q_rope_metrics["local_final_token_q_post_rope_vs_official"]["worst_differing_location"]
    focused = {}
    if worst:
        token = 73
        kv = int(worst["kv_head_index"])
        hpk = int(worst["heads_per_kv_index"])
        lane = int(worst["head_dim_lane"])
        feature = int(worst["q_feature_index"])
        local_pre = float(local_q_pre[token, feature].item())
        official_pre = float(manual_q_pre_with_bias[token, feature].to(torch.float32).item())
        local_post = float(local_q_post_grouped[token, kv, hpk, lane].item())
        official_post = float(official_q_post[token, kv, hpk, lane].item())
        pre_abs = abs(local_pre - official_pre)
        post_abs = abs(local_post - official_post)
        focused = {
            "token_index": token,
            "q_head_index": kv * 8 + hpk,
            "kv_head_index": kv,
            "heads_per_kv_index": hpk,
            "head_dim_lane": lane,
            "flattened_q_feature": feature,
            "local_pre_rope_q_value": local_pre,
            "official_pre_rope_q_value": official_pre,
            "local_post_rope_q_value": local_post,
            "official_post_rope_q_value": official_post,
            "local_minus_official_pre_rope": local_pre - official_pre,
            "local_minus_official_post_rope": local_post - official_post,
            "pre_rope_abs_diff": pre_abs,
            "post_rope_abs_diff": post_abs,
            "mismatch_already_exists_pre_rope": pre_abs > 0,
            "rope_effect": "amplifies" if post_abs > pre_abs else "carries_or_reduces",
        }

    if not rmsnorm_guard["authoritative_rmsnorm_replay_vs_official_layer0_attn_norm_output"]["matched"]:
        classification = "q_path_blocked_by_rmsnorm_provenance_regression"
        earliest = "layer0_attn_norm_output"
        next_step = "restore RMSNorm provenance before continuing Q localization"
    elif q_bias_comparisons["local_q_pre_rope_vs_official_projection_without_bias"]["matched"] and not q_bias_comparisons["local_q_pre_rope_vs_official_projection_with_bias"]["matched"]:
        classification = "q_bias_application_mismatch"
        earliest = "layer0_q_projection_bias_application_before_rope"
        next_step = "prove/fix scoped Q bias application before rechecking Q RoPE and score"
    elif not q_pre_metrics["local_runtime_q_pre_rope_vs_official_q_projection"]["matched"]:
        classification = "q_projection_arithmetic_policy_mismatch_before_rope"
        earliest = "layer0_q_projection_before_rope"
        next_step = "inspect Q projection arithmetic policy with bias/weights held fixed"
    elif not q_layout_metrics["local_grouped_pre_rope_q_vs_official_grouped_q_projection"]["matched"]:
        classification = "q_layout_or_view_mismatch_before_rope"
        earliest = "layer0_q_grouped_head_layout_before_rope"
        next_step = "inspect Q grouped/head layout before RoPE"
    elif not q_rope_metrics["local_q_post_rope_vs_official_q_post_rope_all_tokens"]["matched"]:
        classification = "q_rope_path_mismatch_after_projection_clear"
        earliest = "layer0_q_post_rope_before_attention"
        next_step = "inspect Q RoPE arithmetic/path with pre-RoPE projection cleared"
    elif not score_metrics["local_q_post_rope_plus_candidate_k_vs_official"]["matched"]:
        classification = "qk_score_arithmetic_mismatch_after_qk_provenance_clear"
        earliest = "layer0_final_token_raw_scaled_qk_logits_pre_mask"
        next_step = "inspect QK score arithmetic now that Q and K provenance are clear"
    else:
        classification = "q_pre_post_rope_and_score_cleared_or_prior_measurement_bug"
        earliest = None
        next_step = "continue one seam downstream or reassess stale Q artifact provenance"

    output = {
        "schema_version": "runtime_forward_layer0_q_pre_post_rope_runtime_localization_status/v1",
        "mode": "q-pre-post-rope-runtime-localization-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "token_count": 74,
            "hidden_size": 2880,
            "q_dim": q_dim,
        },
        "source_artifact_paths": {
            "q_provenance_status": str(args.source_q_provenance),
            "k_post_rope_and_score_status": str(args.source_k_post_score),
            "scoped_onednn_k_candidate": str(args.source_k_candidate),
            "authoritative_rmsnorm_replay_proof": str(args.source_rmsnorm_replay_proof),
            "local_q_runtime_capture": str(args.local_q_capture),
            "official_q_post_rope": str(args.q_post_rope_official),
            "official_score": str(args.score_official),
        },
        "rmsnorm_guard_metrics": rmsnorm_guard,
        "k_guard_metrics": k_guard,
        "official_q_guard_metrics": official_q_guard,
        "q_weight_slice_metadata": q_weight_meta,
        "q_bias_metadata_and_application_finding": q_bias_meta,
        "local_q_pre_rope_capture_metrics": q_pre_metrics,
        "explicit_rust_cpu_q_replay_metrics": {
            "available": False,
            "reason": "no exact all-token Rust CPU Q replay with bias exists; live runtime pre-RoPE Q capture is used as the local source",
        },
        "official_onednn_manual_q_oracle_metrics": {
            "manual_with_bias_vs_module": compare_tensor(manual_q_pre_with_bias, module_q_pre, torch, "token_feature"),
            "manual_without_bias_vs_module": compare_tensor(manual_q_pre_no_bias, module_q_pre, torch, "token_feature"),
        },
        "q_bias_application_comparisons": q_bias_comparisons,
        "q_layout_metrics": q_layout_metrics,
        "q_rope_metrics": q_rope_metrics,
        "score_confirmation_metrics": score_metrics,
        "focused_mismatch_trace": focused,
        "earliest_remaining_mismatching_q_operation": earliest,
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
        "--source-q-provenance",
        str(args.source_q_provenance),
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
        "--local-q-capture",
        str(args.local_q_capture),
        "--q-post-rope-official",
        str(args.q_post_rope_official),
        "--score-official",
        str(args.score_official),
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
        raise RuntimeError(f"Q pre/post RoPE localization failed; see {args.verbose_log}")
    output = load_json(args.output)
    output["onednn_verbose_log_path"] = str(args.verbose_log)
    output["onednn_verbose_primitive_lines_sample"] = [
        line for line in verbose_text.splitlines() if "onednn_verbose" in line or "dnnl_verbose" in line
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
