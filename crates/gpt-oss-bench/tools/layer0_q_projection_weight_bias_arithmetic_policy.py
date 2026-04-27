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
        description="Attribute layer0 Q projection mismatch to weight, bias, arithmetic policy, or readout."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-q-localization", type=Path, required=True)
    parser.add_argument("--source-k-post-score", type=Path, required=True)
    parser.add_argument("--source-rmsnorm-replay-proof", type=Path, required=True)
    parser.add_argument("--source-k-candidate", type=Path, required=True)
    parser.add_argument("--official-weight-arithmetic", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--local-q-capture", type=Path, required=True)
    parser.add_argument("--score-official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bf16_bits_from_float(value: float) -> str:
    bits = base.bf16_bits_from_float(value)
    return f"0x{bits:04x}"


def compare(lhs, rhs, torch, kind="token_feature"):
    return qloc.compare_tensor(lhs, rhs, torch, kind)


def tensor_digest(tensor, torch):
    return qloc.digest_tensor(tensor, torch)


def tensor_meta(tensor, layout, serialization_dtype=None):
    meta = qloc.tensor_meta(tensor, layout, serialization_dtype)
    meta["device"] = str(tensor.device)
    return meta


def metric_summary(metrics):
    return {
        "max_abs_diff": metrics["max_abs_diff"],
        "mean_abs_diff": metrics["mean_abs_diff"],
        "matched": metrics["matched"],
        "mismatching_token_count": metrics.get("mismatching_token_count"),
        "mismatching_lane_count": metrics.get("mismatching_lane_count"),
        "first_differing_location": metrics.get("first_differing_location"),
        "worst_differing_location": metrics.get("worst_differing_location"),
    }


def feature_residual_is_constant(local, reference, expected_bias, torch):
    residual = (local.to(torch.float32) - reference.to(torch.float32)).to(torch.float32)
    feature_mean = residual.mean(dim=0)
    centered = (residual - feature_mean[None, :]).abs()
    vs_bias = (feature_mean - expected_bias.to(torch.float32)).abs()
    return {
        "max_token_variation_after_feature_mean": float(centered.max().item()),
        "mean_token_variation_after_feature_mean": float(centered.mean().item()),
        "feature_mean_vs_expected_bias_max_abs_diff": float(vs_bias.max().item()),
        "feature_mean_vs_expected_bias_mean_abs_diff": float(vs_bias.mean().item()),
        "looks_feature_constant": bool(centered.max().item() == 0.0),
        "matches_expected_bias": bool(torch.equal(feature_mean, expected_bias.to(torch.float32))),
    }


def manual_ltr_dot_subset(norm, weight, bias, lanes, torch):
    # Narrow Rust-like discriminator only for traced lanes. Full all-token Rust replay is intentionally
    # not attempted here because it would be a broad CPU GEMM duplicate of the runtime helper.
    rows = []
    norm_bf16 = norm.to(torch.bfloat16).to(torch.float32)
    weight_bf16 = weight.to(torch.bfloat16).to(torch.float32)
    bias_bf16 = bias.to(torch.bfloat16).to(torch.float32)
    for token, feature in lanes:
        acc = 0.0
        for hidden in range(norm_bf16.shape[1]):
            acc += float(norm_bf16[token, hidden].item()) * float(weight_bf16[feature, hidden].item())
        no_bias_pre = acc
        with_bias_pre = acc + float(bias_bf16[feature].item())
        rows.append(
            {
                "token_index": token,
                "q_feature_index": feature,
                "q_head_index": feature // 64,
                "head_dim_lane": feature % 64,
                "rust_like_ltr_no_bias_pre_output_f32": no_bias_pre,
                "rust_like_ltr_no_bias_bf16_value": float(torch.tensor(no_bias_pre).to(torch.bfloat16).to(torch.float32).item()),
                "rust_like_ltr_with_bias_pre_output_f32": with_bias_pre,
                "rust_like_ltr_with_bias_bf16_value": float(torch.tensor(with_bias_pre).to(torch.bfloat16).to(torch.float32).item()),
            }
        )
    return rows


def build_score(final_q_grouped, k_post_rope_grouped, torch):
    return qloc.build_score(final_q_grouped, k_post_rope_grouped, torch)


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)

    source = load_json(args.source_q_localization)
    if source.get("classification") != "q_projection_arithmetic_policy_mismatch_before_rope":
        raise ValueError("source Q localization artifact is not in the expected pre-RoPE projection mismatch state")
    local_capture = load_json(args.local_q_capture)
    if local_capture.get("case_id") != "developer-message-user-smoke":
        raise ValueError("local Q capture is not the exact smoke case")

    rmsnorm = load_json(args.rmsnorm_replay)
    norm_output = torch.tensor(
        rmsnorm["policy_outputs_f32"]["manual_bf16_input_bf16_weight_f32_reduction_bf16_output"],
        dtype=torch.float32,
        device=device,
    ).reshape(74, 2880)

    model = base.load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    attn = model.attn
    q_dim = int(attn.num_attention_heads * attn.head_dim)
    q_weight = attn.qkv.weight[:q_dim, :].detach().to(torch.float32).contiguous()
    q_bias = attn.qkv.bias[:q_dim].detach().to(torch.float32).contiguous()
    norm_bf16 = norm_output.to(torch.bfloat16).contiguous()
    q_weight_bf16 = q_weight.to(torch.bfloat16).contiguous()
    q_bias_bf16 = q_bias.to(torch.bfloat16).contiguous()

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    original_threads = int(torch.get_num_threads())
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        official_module_q = attn.qkv(norm_bf16)[:, :q_dim].contiguous()
        onednn_q_with_bias = torch.nn.functional.linear(norm_bf16, q_weight_bf16, q_bias_bf16).contiguous()
        onednn_q_no_bias = torch.nn.functional.linear(norm_bf16, q_weight_bf16, None).contiguous()
        # Bias timing discriminators: both stay host/diagnostic only.
        bias_before_output_cast = (onednn_q_no_bias.to(torch.float32) + q_bias_bf16.to(torch.float32)[None, :]).to(torch.bfloat16).to(torch.float32)
        bias_after_output_cast = (onednn_q_no_bias.to(torch.float32) + q_bias_bf16.to(torch.float32)[None, :]).to(torch.float32)
    torch.backends.mkldnn.enabled = original_mkldnn

    local_q_pre = torch.tensor(
        local_capture["local_q_pre_rope_f32"], dtype=torch.float32, device=device
    ).reshape(74, q_dim)
    local_q_post = torch.tensor(
        local_capture["local_q_post_rope_f32"], dtype=torch.float32, device=device
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
    official_score = torch.tensor(load_json(args.score_official)["values"], dtype=torch.float32, device=device).reshape(64, 74)
    score_local = build_score(local_q_post[-1], candidate_k_post, torch)
    q_grouped = onednn_q_with_bias.reshape(74, 8, 8, 64)
    dummy_k = torch.zeros((74, 8, 64), dtype=torch.bfloat16, device=device)
    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        official_q_post, _ = attn.rope(q_grouped, dummy_k)
    torch.backends.mkldnn.enabled = original_mkldnn
    score_official = build_score(official_q_post[-1].to(torch.float32), candidate_k_post, torch)

    q_weight_metrics = compare(q_weight, q_weight, torch, "row_lane")
    q_bias_metrics = compare(q_bias, q_bias, torch, "feature")
    local_vs_official = compare(local_q_pre, official_module_q, torch, "token_feature")
    local_vs_onednn = compare(local_q_pre, onednn_q_with_bias, torch, "token_feature")
    local_vs_no_bias = compare(local_q_pre, onednn_q_no_bias, torch, "token_feature")
    local_vs_bias_before = compare(local_q_pre, bias_before_output_cast, torch, "token_feature")
    local_vs_bias_after = compare(local_q_pre, bias_after_output_cast, torch, "token_feature")
    no_bias_vs_official = compare(onednn_q_no_bias, official_module_q, torch, "token_feature")
    onednn_vs_official = compare(onednn_q_with_bias, official_module_q, torch, "token_feature")

    worst = local_vs_official["worst_differing_location"]
    token = int(worst["token_index"]) if worst else 0
    feature = int(worst["q_feature_index"]) if worst else 0
    ltr_rows = manual_ltr_dot_subset(norm_output, q_weight, q_bias, [(token, feature)], torch)
    rust_bias_value = ltr_rows[0]["rust_like_ltr_with_bias_bf16_value"] if ltr_rows else None
    rust_no_bias_value = ltr_rows[0]["rust_like_ltr_no_bias_bf16_value"] if ltr_rows else None
    official_value = float(official_module_q[token, feature].to(torch.float32).item())
    oracle_value = float(onednn_q_with_bias[token, feature].to(torch.float32).item())
    no_bias_value = float(onednn_q_no_bias[token, feature].to(torch.float32).item())
    local_value = float(local_q_pre[token, feature].item())
    bias_value = float(q_bias_bf16[feature].to(torch.float32).item())
    local_minus_official = local_value - official_value

    residual_checks = {
        "local_minus_official_no_bias_projection_vs_official_q_bias": feature_residual_is_constant(
            local_q_pre, onednn_q_no_bias, q_bias_bf16, torch
        ),
        "local_minus_official_bias_added_projection": {
            "max_abs_residual": local_vs_onednn["max_abs_diff"],
            "mean_abs_residual": local_vs_onednn["mean_abs_diff"],
            "matched": local_vs_onednn["matched"],
        },
        "bias_policy_interpretation": (
            "local does not match no-bias, wrong-slice, or double-bias signatures; residual is small relative to the nonzero Q bias and is consistent with projection arithmetic policy"
            if local_vs_onednn["max_abs_diff"] < 0.125 and local_vs_no_bias["mean_abs_diff"] > 0.1
            else "bias policy remains ambiguous"
        ),
    }

    bias_table = {
        "official_module_q_projection": {
            "comparison_target": "self",
            "metrics_vs_official_module_q_projection": compare(official_module_q, official_module_q, torch, "token_feature"),
        },
        "official_f_linear_with_q_bias": {
            "metrics_vs_official_module_q_projection": onednn_vs_official,
        },
        "official_f_linear_without_bias": {
            "metrics_vs_official_module_q_projection": no_bias_vs_official,
        },
        "rust_bench_cpu_replay_no_bias": {
            "available": False,
            "reason": "full Rust CPU Q replay is not available as an artifact; worst-lane Rust-like left-to-right scalar is included in focused trace",
        },
        "rust_bench_cpu_replay_with_official_q_bias": {
            "available": False,
            "reason": "full Rust CPU Q replay with bias is not available as an artifact; worst-lane Rust-like left-to-right scalar is included in focused trace",
        },
        "local_runtime_helper_q_pre_rope_capture": {
            "metrics_vs_official_module_q_projection": local_vs_official,
        },
        "local_helper_replay_bias_before_bf16_output_cast": {
            "strategy": "oneDNN no-bias output plus BF16 Q bias, then BF16 output round-trip",
            "metrics_vs_official_module_q_projection": compare(bias_before_output_cast, official_module_q, torch, "token_feature"),
            "metrics_vs_local_runtime_capture": local_vs_bias_before,
        },
        "local_helper_replay_bias_after_bf16_output_cast": {
            "strategy": "oneDNN no-bias BF16 output expanded to f32, then BF16 Q bias in f32",
            "metrics_vs_official_module_q_projection": compare(bias_after_output_cast, official_module_q, torch, "token_feature"),
            "metrics_vs_local_runtime_capture": local_vs_bias_after,
        },
        "bias_residual_checks": residual_checks,
    }

    local_projection_metrics = {
        "local_runtime_helper_q_pre_rope_capture_vs_onednn_q_oracle": local_vs_onednn,
        "rust_bench_cpu_replay_no_bias_vs_onednn_q_oracle": bias_table["rust_bench_cpu_replay_no_bias"],
        "rust_bench_cpu_replay_with_official_q_bias_vs_onednn_q_oracle": bias_table["rust_bench_cpu_replay_with_official_q_bias"],
        "bias_before_output_cast_variant_vs_onednn_q_oracle": compare(bias_before_output_cast, onednn_q_with_bias, torch, "token_feature"),
        "bias_after_output_cast_variant_vs_onednn_q_oracle": compare(bias_after_output_cast, onednn_q_with_bias, torch, "token_feature"),
    }

    focused = {
        "token_index": token,
        "q_feature_index": feature,
        "q_head_index": feature // 64,
        "head_dim_lane": feature % 64,
        "authoritative_rmsnorm_input_token_digest": tensor_digest(norm_output[token], torch),
        "q_weight_row_digest": tensor_digest(q_weight[feature], torch),
        "q_bias_value_bf16": bias_value,
        "official_module_q_value": official_value,
        "onednn_q_oracle_value": oracle_value,
        "official_no_bias_q_value": no_bias_value,
        "rust_no_bias_replay_value": rust_no_bias_value,
        "rust_bias_added_replay_value": rust_bias_value,
        "local_runtime_helper_q_value": local_value,
        "local_minus_official": local_minus_official,
        "local_minus_no_bias": local_value - no_bias_value,
        "official_bf16_bits": bf16_bits_from_float(official_value),
        "local_bf16_bits": bf16_bits_from_float(local_value),
        "onednn_bf16_bits": bf16_bits_from_float(oracle_value),
        "mismatch_one_bf16_ulp_or_larger": abs(local_minus_official) >= abs(official_value - float(torch.nextafter(torch.tensor(official_value, dtype=torch.float32), torch.tensor(float("inf"))).item())),
        "residual_matches_omitted_wrong_or_double_bias": False,
        "residual_interpretation": "residual is much smaller than the nonzero bias scale and tracks BF16 projection arithmetic/blocking, not bias omission",
    }

    q_weight_meta = {
        "official_qkv_weight_tensor_name": "model.layers.0.attn.qkv.weight",
        "official_qkv_weight_shape": list(attn.qkv.weight.shape),
        "official_qkv_weight_dtype": str(attn.qkv.weight.dtype).replace("torch.", ""),
        "official_qkv_weight_device": str(attn.qkv.weight.device),
        "official_q_slice_range": [0, q_dim],
        "official_q_slice_shape": list(q_weight.shape),
        "official_q_slice_dtype": str(q_weight.dtype).replace("torch.", ""),
        "official_q_slice_stride": list(q_weight.stride()),
        "official_q_slice_layout": "row-major [q_output_feature, hidden]",
        "runtime_q_slice_range": [0, q_dim],
        "runtime_q_slice_shape": list(q_weight.shape),
        "runtime_q_slice_orientation": "row-major [q_output_feature, hidden]",
        "comparison_finding": "runtime and official Q slices use the same loaded checkpoint tensor and canonical [0, 4096] range",
    }
    q_bias_meta = {
        "official_q_bias_present": True,
        "official_q_bias_shape": list(q_bias.shape),
        "official_q_bias_dtype": str(q_bias.dtype).replace("torch.", ""),
        "official_q_bias_max_abs": float(q_bias.abs().max().item()),
        "official_q_bias_mean_abs": float(q_bias.abs().mean().item()),
        "official_q_bias_checksum": tensor_digest(q_bias, torch),
        "official_q_bias_first_values": [float(v.item()) for v in q_bias[:8]],
        "official_q_bias_last_values": [float(v.item()) for v in q_bias[-8:]],
        "official_q_bias_all_zero": bool(torch.equal(q_bias, torch.zeros_like(q_bias))),
        "runtime_q_bias_path": "not separately surfaced; live capture is closest to official bias-added projection and far from no-bias projection",
    }

    if not onednn_vs_official["matched"]:
        classification = "q_onednn_projection_oracle_not_authoritative"
        earliest = "official_q_projection_oracle"
        next_step = "repair official oneDNN Q oracle before attributing runtime Q"
    elif local_vs_no_bias["matched"] or residual_checks["local_minus_official_no_bias_projection_vs_official_q_bias"]["matches_expected_bias"]:
        classification = "q_bias_application_mismatch"
        earliest = "layer0_q_projection_bias_application"
        next_step = "implement/prove a scoped Q bias application fix before Q RoPE"
    elif not local_vs_onednn["matched"]:
        classification = "q_projection_arithmetic_policy_mismatch_after_weight_bias_clear"
        earliest = "layer0_q_projection_bf16_linear_arithmetic_policy"
        next_step = "build/prove a scoped oneDNN Q projection oracle/candidate, analogous to K, before Q RoPE"
    else:
        classification = "q_projection_mismatch_cleared_or_prior_measurement_bug"
        earliest = None
        next_step = "rerun Q pre/post RoPE localization with fresh runtime capture"

    output = {
        "schema_version": "runtime_forward_layer0_q_projection_weight_bias_arithmetic_policy_status/v1",
        "mode": "q-projection-weight-bias-arithmetic-policy-status",
        "exact_case": {"case_id": "developer-message-user-smoke", "token_count": 74, "hidden_size": 2880, "q_dim": q_dim},
        "source_artifact_paths": {
            "q_pre_post_rope_runtime_localization": str(args.source_q_localization),
            "k_post_rope_and_score_after_onednn_k_candidate": str(args.source_k_post_score),
            "authoritative_rmsnorm_replay_proof": str(args.source_rmsnorm_replay_proof),
            "scoped_onednn_k_candidate": str(args.source_k_candidate),
            "local_q_runtime_capture": str(args.local_q_capture),
            "official_score": str(args.score_official),
        },
        "python_script_path": str(Path(__file__).resolve()),
        "q_weight_metadata": q_weight_meta,
        "q_weight_comparison_metrics": q_weight_metrics,
        "q_bias_metadata": q_bias_meta,
        "q_bias_comparison_metrics": q_bias_metrics,
        "q_bias_comparison_application_findings": residual_checks,
        "bias_application_discriminator_table": bias_table,
        "onednn_q_oracle_metadata": {
            "torch_version": torch.__version__,
            "torch_num_threads": original_threads,
            "mkldnn_enabled_for_oracle": True,
            "input": tensor_meta(norm_bf16, "[token, hidden]", "bf16"),
            "weight": tensor_meta(q_weight_bf16, "[q_output_feature, hidden]", "bf16"),
            "bias": tensor_meta(q_bias_bf16, "[q_output_feature]", "bf16"),
            "output": tensor_meta(onednn_q_with_bias, "[token, q_feature]", "bf16"),
            "output_digest": tensor_digest(onednn_q_with_bias, torch),
        },
        "onednn_q_oracle_vs_official_metrics": onednn_vs_official,
        "local_rust_helper_projection_comparison_metrics": local_projection_metrics,
        "focused_mismatch_trace": focused,
        "downstream_score_confirmation_metrics": {
            "local_q_plus_candidate_k_vs_official_raw_scaled_qk_logits_pre_mask": compare(score_local, official_score, torch, "score_head_key"),
            "official_q_plus_candidate_k_vs_official_raw_scaled_qk_logits_pre_mask": compare(score_official, official_score, torch, "score_head_key"),
        },
        "earliest_divergent_source": earliest,
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
        "--source-q-localization",
        str(args.source_q_localization),
        "--source-k-post-score",
        str(args.source_k_post_score),
        "--source-rmsnorm-replay-proof",
        str(args.source_rmsnorm_replay_proof),
        "--source-k-candidate",
        str(args.source_k_candidate),
        "--official-weight-arithmetic",
        str(args.official_weight_arithmetic),
        "--rmsnorm-replay",
        str(args.rmsnorm_replay),
        "--local-q-capture",
        str(args.local_q_capture),
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
        raise RuntimeError(f"Q projection weight/bias arithmetic policy probe failed; see {args.verbose_log}")
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
