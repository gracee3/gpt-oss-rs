#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_post_rope_and_score_after_onednn_k_candidate as kdiag
import layer0_k_projection_pytorch_bf16_linear_backend_policy as base
import layer0_q_pre_post_rope_runtime_localization as qloc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bench-only full-shape oneDNN Q projection candidate proof."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-q-policy", type=Path, required=True)
    parser.add_argument("--source-q-localization", type=Path, required=True)
    parser.add_argument("--source-k-candidate", type=Path, required=True)
    parser.add_argument("--source-k-post-score", type=Path, required=True)
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


def compare(lhs, rhs, torch, kind):
    return qloc.compare_tensor(lhs, rhs, torch, kind)


def tensor_meta(tensor, layout, serialization_dtype=None):
    meta = qloc.tensor_meta(tensor, layout, serialization_dtype)
    meta["device"] = str(tensor.device)
    return meta


def digest(tensor, torch):
    return qloc.digest_tensor(tensor, torch)


def bf16_bits(value):
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    q_policy = load_json(args.source_q_policy)
    q_localization = load_json(args.source_q_localization)
    if q_policy.get("classification") != "q_projection_arithmetic_policy_mismatch_after_weight_bias_clear":
        raise ValueError("source Q policy artifact is not in the expected weight/bias-cleared state")
    if q_localization.get("classification") != "q_projection_arithmetic_policy_mismatch_before_rope":
        raise ValueError("source Q localization artifact is not in the expected pre-RoPE mismatch state")

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
    q_weight = attn.qkv.weight[:q_dim, :].detach().to(torch.float32).contiguous()
    q_bias = attn.qkv.bias[:q_dim].detach().to(torch.float32).contiguous()
    q_weight_bf16 = q_weight.to(torch.bfloat16).contiguous()
    q_bias_bf16 = q_bias.to(torch.bfloat16).contiguous()

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    original_threads = int(torch.get_num_threads())
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        official_module_q = attn.qkv(norm_bf16)[:, :q_dim].contiguous()
        candidate_q = torch.nn.functional.linear(norm_bf16, q_weight_bf16, q_bias_bf16).contiguous()
        candidate_grouped = candidate_q.reshape(74, 8, 8, 64)
        dummy_k = torch.zeros((74, 8, 64), dtype=torch.bfloat16, device=device)
        candidate_q_post, _ = attn.rope(candidate_grouped, dummy_k)
    torch.backends.mkldnn.enabled = original_mkldnn

    local_capture = load_json(args.local_q_capture)
    local_q_pre = torch.tensor(
        local_capture["local_q_pre_rope_f32"], dtype=torch.float32, device=device
    ).reshape(74, q_dim)
    local_q_post = torch.tensor(
        local_capture["local_q_post_rope_f32"], dtype=torch.float32, device=device
    ).reshape(74, 8, 8, 64)
    official_q_post = torch.tensor(
        load_json(args.q_post_rope_official)["values"], dtype=torch.float32, device=device
    ).reshape(74, 8, 8, 64)

    k_args = argparse.Namespace(
        model_root=args.model_root,
        source=args.source_k_candidate,
        official_weight_arithmetic=args.official_weight_arithmetic,
        rmsnorm_replay=args.rmsnorm_replay,
        device=args.device,
    )
    _scoped_source, candidate_k_projection, _official_k_pre, _candidate_meta = kdiag.build_onednn_candidate(
        k_args, torch, device
    )
    candidate_k_grouped = candidate_k_projection.to(torch.float32).reshape(74, 8, 64)
    candidate_k_post = kdiag.apply_model_rope(k_args, candidate_k_grouped, torch, device).to(torch.float32)
    official_score = torch.tensor(
        load_json(args.score_official)["values"], dtype=torch.float32, device=device
    ).reshape(64, 74)
    score_candidate = qloc.build_score(candidate_q_post[-1].to(torch.float32), candidate_k_post, torch)
    score_local = qloc.build_score(local_q_post[-1], candidate_k_post, torch)
    score_official_q = qloc.build_score(official_q_post[-1], candidate_k_post, torch)

    candidate_vs_official = compare(candidate_q, official_module_q, torch, "token_feature")
    local_vs_candidate = compare(local_q_pre, candidate_q, torch, "token_feature")
    local_vs_official = compare(local_q_pre, official_module_q, torch, "token_feature")
    candidate_grouped_vs_official = compare(
        candidate_grouped, official_module_q.reshape(74, 8, 8, 64), torch, "token_kv_hpk_lane"
    )
    candidate_post_vs_official = compare(candidate_q_post.to(torch.float32), official_q_post, torch, "token_kv_hpk_lane")

    token = 6
    feature = 3755
    q_head = feature // 64
    lane = feature % 64
    kv_head = q_head // 8
    hpk = q_head % 8
    trace = {
        "token_index": token,
        "q_feature_index": feature,
        "q_head_index": q_head,
        "kv_head_index": kv_head,
        "heads_per_kv_index": hpk,
        "head_dim_lane": lane,
        "official_module_q_pre_rope_value": float(official_module_q[token, feature].to(torch.float32).item()),
        "onednn_candidate_q_pre_rope_value": float(candidate_q[token, feature].to(torch.float32).item()),
        "local_runtime_helper_q_pre_rope_value": float(local_q_pre[token, feature].item()),
        "rust_cpu_replay_q_pre_rope_value": q_policy.get("focused_mismatch_trace", {}).get("rust_bias_added_replay_value"),
        "q_bias_value": float(q_bias_bf16[feature].to(torch.float32).item()),
        "candidate_q_post_rope_value": float(candidate_q_post[token, kv_head, hpk, lane].to(torch.float32).item()),
        "official_q_post_rope_value": float(official_q_post[token, kv_head, hpk, lane].item()),
        "local_q_post_rope_value": float(local_q_post[token, kv_head, hpk, lane].item()),
    }
    trace["candidate_minus_official_pre_rope"] = (
        trace["onednn_candidate_q_pre_rope_value"] - trace["official_module_q_pre_rope_value"]
    )
    trace["local_minus_official_pre_rope"] = (
        trace["local_runtime_helper_q_pre_rope_value"] - trace["official_module_q_pre_rope_value"]
    )
    trace["candidate_minus_official_post_rope"] = (
        trace["candidate_q_post_rope_value"] - trace["official_q_post_rope_value"]
    )
    trace["local_minus_official_post_rope"] = (
        trace["local_q_post_rope_value"] - trace["official_q_post_rope_value"]
    )
    trace["candidate_clears_the_mismatch"] = (
        trace["candidate_minus_official_pre_rope"] == 0.0
        and trace["candidate_minus_official_post_rope"] == 0.0
    )
    trace["official_bf16_bits"] = bf16_bits(trace["official_module_q_pre_rope_value"])
    trace["candidate_bf16_bits"] = bf16_bits(trace["onednn_candidate_q_pre_rope_value"])
    trace["local_bf16_bits"] = bf16_bits(trace["local_runtime_helper_q_pre_rope_value"])

    score_candidate_metrics = compare(score_candidate, official_score, torch, "score_head_key")
    score_local_metrics = compare(score_local, official_score, torch, "score_head_key")
    score_official_q_metrics = compare(score_official_q, official_score, torch, "score_head_key")

    if not candidate_vs_official["matched"]:
        classification = "onednn_q_projection_candidate_not_authoritative"
        earliest = "layer0_q_projection_candidate"
        next_step = "repair the oneDNN Q candidate construction before using it downstream"
    elif not candidate_post_vs_official["matched"]:
        classification = "q_rope_path_mismatch_after_onednn_q_candidate"
        earliest = "layer0_q_post_rope_before_attention"
        next_step = "inspect Q RoPE path using oneDNN Q candidate pre-RoPE input"
    elif not score_candidate_metrics["matched"]:
        classification = "qk_score_arithmetic_mismatch_after_qk_candidates_clear"
        earliest = "layer0_final_token_raw_scaled_qk_logits_pre_mask"
        next_step = "inspect QK score arithmetic with Q/K candidate provenance cleared"
    else:
        classification = "onednn_q_candidate_clears_q_rope_and_first_score_consumer"
        earliest = None
        next_step = "decide whether to promote scoped Q/K projection-policy runtime candidates or continue one seam downstream"

    if (
        classification == "onednn_q_candidate_clears_q_rope_and_first_score_consumer"
        and not local_vs_candidate["matched"]
    ):
        classification = "onednn_q_candidate_confirms_runtime_q_projection_policy_delta"

    output = {
        "schema_version": "runtime_forward_layer0_q_projection_onednn_oracle_scoped_candidate_status/v1",
        "mode": "q-projection-onednn-oracle-scoped-candidate-status",
        "exact_case": {"case_id": "developer-message-user-smoke", "token_count": 74, "hidden_size": 2880, "q_dim": q_dim},
        "source_artifact_paths": {
            "q_weight_bias_arithmetic_policy": str(args.source_q_policy),
            "q_pre_post_rope_runtime_localization": str(args.source_q_localization),
            "scoped_k_onednn_candidate_proof": str(args.source_k_candidate),
            "k_post_rope_and_score_status": str(args.source_k_post_score),
            "authoritative_rmsnorm_replay_proof": str(args.source_rmsnorm_replay_proof),
            "local_q_runtime_capture": str(args.local_q_capture),
            "official_q_post_rope": str(args.q_post_rope_official),
            "official_score": str(args.score_official),
        },
        "files_changed": [
            "crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs",
            "crates/gpt-oss-bench/tools/layer0_q_projection_onednn_oracle_scoped_candidate.py",
        ],
        "change_scope": "bench/proof-only",
        "runtime_affecting": False,
        "candidate_q_projection_strategy": "bench-only full-shape oneDNN/MKLDNN BF16 torch.nn.functional.linear(authoritative_norm_bf16, Q_weight_bf16, Q_bias_bf16)",
        "scope_statement": {
            "affected_paths": [
                "this diagnostic/proof mode",
                "bench-only oneDNN Q projection candidate for the exact layer0 smoke case",
            ],
            "not_affected_paths": [
                "default runtime Q helper",
                "default runtime K helper",
                "QKV projection runtime behavior",
                "masked attention",
                "softmax",
                "V path",
                "logits",
                "MoE",
                "Harmony",
                "cache",
                "later layers",
            ],
        },
        "onednn_q_candidate_construction_metadata": {
            "torch_version": torch.__version__,
            "thread_count": original_threads,
            "mkldnn_enabled_for_candidate": True,
            "input": tensor_meta(norm_bf16, "[token, hidden]", "bf16"),
            "weight": tensor_meta(q_weight_bf16, "[q_output_feature, hidden]", "bf16"),
            "bias": tensor_meta(q_bias_bf16, "[q_output_feature]", "bf16"),
            "output": tensor_meta(candidate_q, "[token, q_feature]", "bf16"),
            "output_digest": digest(candidate_q, torch),
            "bench_proof_only": True,
        },
        "q_candidate_vs_official_metrics": {
            "candidate_q_pre_rope_vs_official_module_q_projection": candidate_vs_official,
            "candidate_grouped_pre_rope_q_vs_official_grouped_q_projection": candidate_grouped_vs_official,
        },
        "legacy_local_rust_q_metrics": {
            "local_runtime_helper_q_pre_rope_capture_vs_candidate": local_vs_candidate,
            "local_runtime_helper_q_pre_rope_capture_vs_official": local_vs_official,
            "rust_bench_cpu_replay_with_q_weight_and_q_bias": {
                "available": False,
                "reason": "no exact all-token Rust CPU Q replay with bias is available; prior focused scalar trace is carried in known_mismatch_trace",
            },
            "rust_bench_cpu_replay_with_q_weight_no_bias": {
                "available": False,
                "reason": "bias guard is represented by official oneDNN no-bias comparison in the source Q policy artifact",
            },
        },
        "grouped_q_pre_rope_metrics": {
            "candidate_grouped_pre_rope_q_vs_official": candidate_grouped_vs_official,
            "official_grouped_pre_rope_available": True,
            "layout": "[token, kv_head, heads_per_kv, lane] equivalent to [token, q_head, lane]",
        },
        "q_post_rope_metrics": {
            "candidate_q_post_rope_vs_official": candidate_post_vs_official,
            "local_runtime_q_post_rope_vs_official": compare(local_q_post, official_q_post, torch, "token_kv_hpk_lane"),
        },
        "score_confirmation_metrics": {
            "candidate_q_plus_candidate_k_vs_official_raw_scaled_qk_logits_pre_mask": score_candidate_metrics,
            "local_q_plus_candidate_k_vs_official_raw_scaled_qk_logits_pre_mask": score_local_metrics,
            "official_q_plus_candidate_k_vs_official_raw_scaled_qk_logits_pre_mask": score_official_q_metrics,
        },
        "known_mismatch_trace": trace,
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
        "--source-q-policy",
        str(args.source_q_policy),
        "--source-q-localization",
        str(args.source_q_localization),
        "--source-k-candidate",
        str(args.source_k_candidate),
        "--source-k-post-score",
        str(args.source_k_post_score),
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
        raise RuntimeError(f"Q oneDNN scoped candidate proof failed; see {args.verbose_log}")
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
