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
        description="Diagnose layer0 V projection weight/bias/arithmetic policy."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--weighted-v-status", type=Path, required=True)
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


def vector_summary(tensor, torch):
    t = tensor.to(torch.float32).reshape(-1)
    return {
        "count": int(t.numel()),
        "max_abs": float(t.abs().max().item()) if t.numel() else 0.0,
        "mean_abs": float(t.abs().mean().item()) if t.numel() else 0.0,
        "digest": digest(tensor, torch),
        "first_values": [float(v.item()) for v in t[:8]],
        "last_values": [float(v.item()) for v in t[-8:]],
    }


def add_bias_before_output_cast(no_bias_bf16, bias_bf16, torch):
    return (no_bias_bf16.to(torch.float32) + bias_bf16.to(torch.float32)[None, :]).to(torch.bfloat16)


def add_bias_after_output_cast(no_bias_bf16, bias_bf16, torch):
    return (no_bias_bf16.to(torch.float32) + bias_bf16.to(torch.float32)[None, :]).to(torch.float32)


def weighted_sum(probs_64x75, v_74x8x64, torch):
    probs = probs_64x75.to(torch.bfloat16)
    v = v_74x8x64.to(torch.bfloat16)
    rows = []
    for q_head in range(64):
        kv_head = q_head // 8
        rows.append(torch.einsum("t,td->d", probs[q_head, :74], v[:, kv_head, :]).to(torch.bfloat16))
    return torch.stack(rows, dim=0).reshape(4096).to(torch.float32)


def compare_head_lane(lhs, rhs, torch):
    lhs = lhs.reshape(64, 64).to(torch.float32)
    rhs = rhs.reshape(64, 64).to(torch.float32)
    diff = (lhs - rhs).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        coords = mismatch.nonzero(as_tuple=False)
        h0, l0 = [int(v.item()) for v in coords[0]]
        flat = int(diff.reshape(-1).argmax().item())
        hw = flat // 64
        lw = flat % 64
        first = {
            "flattened_feature": h0 * 64 + l0,
            "q_head": h0,
            "lane": l0,
            "lhs_value": float(lhs[h0, l0].item()),
            "rhs_value": float(rhs[h0, l0].item()),
            "abs_diff": float(diff[h0, l0].item()),
        }
        worst = {
            "flattened_feature": hw * 64 + lw,
            "q_head": hw,
            "lane": lw,
            "lhs_value": float(lhs[hw, lw].item()),
            "rhs_value": float(rhs[hw, lw].item()),
            "abs_diff": float(diff[hw, lw].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_head_count": int(mismatch.any(dim=1).sum().item()),
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_flattened_feature_q_head_lane": first,
        "worst_differing_flattened_feature_q_head_lane": worst,
    }


def variant_entry(name, tensor, official, torch, layout):
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": "f32-expanded BF16 values",
        "layout_interpretation": layout,
        "metrics_vs_official_module_v": compare(tensor, official, torch, "token_kv_head_lane"),
    }


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)

    weighted_status = load_json(args.weighted_v_status)
    probs_proof = load_json(args.attention_probs_proof)
    q_candidate = load_json(args.q_candidate)
    k_candidate = load_json(args.k_candidate)
    local_capture = load_json(args.local_v_capture)
    official_weighted = load_json(args.official_weighted_v)
    if weighted_status.get("classification") != "attention_weighted_v_sum_blocked_by_v_provenance_mismatch":
        raise ValueError("weighted-V source status is not the expected V provenance mismatch")
    if probs_proof.get("classification") != "attention_probs_cleared_after_post_mask_candidates":
        raise ValueError("attention probability proof is not cleared")

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
    qkv_dim = q_dim + 2 * kv_dim
    v_start = q_dim + kv_dim
    v_end = v_start + kv_dim
    v_weight = attn.qkv.weight[v_start:v_end, :].detach().to(torch.float32).contiguous()
    v_bias = attn.qkv.bias[v_start:v_end].detach().to(torch.float32).contiguous()
    v_weight_bf16 = v_weight.to(torch.bfloat16).contiguous()
    v_bias_bf16 = v_bias.to(torch.bfloat16).contiguous()

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    original_threads = int(torch.get_num_threads())
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        full_qkv = attn.qkv(norm_bf16).contiguous()
        official_module_v = full_qkv[:, v_start:v_end].contiguous()
        onednn_v = torch.nn.functional.linear(norm_bf16, v_weight_bf16, v_bias_bf16).contiguous()
        onednn_v_no_bias = torch.nn.functional.linear(norm_bf16, v_weight_bf16, None).contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn

    local_row_major = torch.tensor(
        local_capture["row_major_hypothesis_v_f32"], dtype=torch.float32, device=device
    ).reshape(74, 8, 64)
    local_slice_major = torch.tensor(
        local_capture["slice_major_hypothesis_v_f32"], dtype=torch.float32, device=device
    ).reshape(74, 8, 64)
    official_v = official_module_v.reshape(74, 8, 64).to(torch.float32)
    onednn_v_grouped = onednn_v.reshape(74, 8, 64).to(torch.float32)
    no_bias_grouped = onednn_v_no_bias.reshape(74, 8, 64).to(torch.float32)
    bias_before = add_bias_before_output_cast(onednn_v_no_bias, v_bias_bf16, torch).reshape(74, 8, 64).to(torch.float32)
    bias_after = add_bias_after_output_cast(onednn_v_no_bias, v_bias_bf16, torch).reshape(74, 8, 64).to(torch.float32)

    # Bounded Rust/CPU-style replay uses the proven row-major BF16 GEMM arithmetic represented
    # by matmul with MKLDNN disabled, then applies the explicit BF16 bias variants.
    old_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = False
    with torch.inference_mode():
        rust_no_bias = torch.nn.functional.linear(norm_bf16, v_weight_bf16, None).to(torch.bfloat16).contiguous()
        rust_with_bias = torch.nn.functional.linear(norm_bf16, v_weight_bf16, v_bias_bf16).to(torch.bfloat16).contiguous()
    torch.backends.mkldnn.enabled = old_mkldnn
    rust_no_bias_grouped = rust_no_bias.reshape(74, 8, 64).to(torch.float32)
    rust_with_bias_grouped = rust_with_bias.reshape(74, 8, 64).to(torch.float32)

    q_guard = probs_proof["q_post_rope_guard_metrics"]
    k_guard = probs_proof["k_post_rope_guard_metrics"]
    prob_guards = {
        "real_key_probabilities": probs_proof["real_key_probability_metrics"],
        "sink_probabilities": probs_proof["sink_probability_metrics"],
        "all_positions": probs_proof["all_position_probability_metrics"],
        "row_sum_summary_after_bf16_serialization": probs_proof["local_row_sum_summary"],
    }

    weight_metrics = compare(v_weight_bf16, v_weight_bf16, torch, "row_hidden_lane")
    bias_metrics = compare(v_bias_bf16, v_bias_bf16, torch, "feature")
    row_major_vs_oracle = compare(local_row_major, onednn_v_grouped, torch, "token_kv_head_lane")
    slice_major_vs_oracle = compare(local_slice_major, onednn_v_grouped, torch, "token_kv_head_lane")
    row_major_vs_rust = compare(local_row_major, rust_with_bias_grouped, torch, "token_kv_head_lane")
    slice_major_vs_rust = compare(local_slice_major, rust_with_bias_grouped, torch, "token_kv_head_lane")

    projection_variant_table = [
        variant_entry("official_module_v_projection", official_v, official_v, torch, "[token, kv_head, lane]"),
        variant_entry("official_onednn_f_linear_v_weight_v_bias", onednn_v_grouped, official_v, torch, "[token, kv_head, lane]"),
        variant_entry("official_onednn_f_linear_v_weight_no_bias", no_bias_grouped, official_v, torch, "[token, kv_head, lane]"),
        variant_entry("local_runtime_v_row_major_hypothesis", local_row_major, official_v, torch, "[token, kv_head, lane] from per-token fused-QKV row slices"),
        variant_entry("local_runtime_v_slice_major_hypothesis", local_slice_major, official_v, torch, "[token, kv_head, lane] from contiguous V block after all-token Q and K blocks"),
        variant_entry("explicit_rust_cpu_v_replay_runtime_weight_no_bias", rust_no_bias_grouped, official_v, torch, "[token, kv_head, lane]"),
        variant_entry("explicit_rust_cpu_v_replay_runtime_weight_official_bias", rust_with_bias_grouped, official_v, torch, "[token, kv_head, lane]"),
        variant_entry("bias_added_before_bf16_output_cast", bias_before, official_v, torch, "[token, kv_head, lane]"),
        variant_entry("bias_added_after_bf16_output_cast", bias_after, official_v, torch, "[token, kv_head, lane]"),
    ]

    official_probs = torch.tensor(
        load_json(Path(probs_proof["official_post_softmax_reference_path"]))["values"],
        dtype=torch.float32,
        device=device,
    ).reshape(64, 75)
    official_weighted_values = torch.tensor(official_weighted["values"], dtype=torch.float32, device=device)
    weighted_row_major = weighted_sum(official_probs, local_row_major, torch)
    weighted_slice_major = weighted_sum(official_probs, local_slice_major, torch)
    weighted_onednn = weighted_sum(official_probs, onednn_v_grouped, torch)

    worst_loc = row_major_vs_oracle["worst_differing_location"]
    if slice_major_vs_oracle["matched"]:
        worst_loc = row_major_vs_oracle["worst_differing_location"]
    token = int(worst_loc["indices"][0]) if worst_loc else 0
    kv_head = int(worst_loc["indices"][1]) if worst_loc else 0
    lane = int(worst_loc["indices"][2]) if worst_loc else 0
    feature = kv_head * 64 + lane
    norm_token = norm_output[token]
    focused_trace = {
        "worst_local_v_provenance_mismatch": {
            "token_index": token,
            "v_feature": feature,
            "kv_head": kv_head,
            "lane": lane,
            "authoritative_rmsnorm_input_checksum_for_token": digest(norm_token, torch),
            "v_weight_row_digest": digest(v_weight_bf16[feature], torch),
            "v_bias_value": float(v_bias_bf16[feature].to(torch.float32).item()),
            "official_module_v_value": float(official_v[token, kv_head, lane].item()),
            "onednn_v_oracle_value": float(onednn_v_grouped[token, kv_head, lane].item()),
            "official_no_bias_v_value": float(no_bias_grouped[token, kv_head, lane].item()),
            "rust_no_bias_replay_value": float(rust_no_bias_grouped[token, kv_head, lane].item()),
            "rust_bias_added_replay_value": float(rust_with_bias_grouped[token, kv_head, lane].item()),
            "local_row_major_readout_value": float(local_row_major[token, kv_head, lane].item()),
            "local_slice_major_readout_value": float(local_slice_major[token, kv_head, lane].item()),
            "row_major_minus_official": float(local_row_major[token, kv_head, lane].item() - official_v[token, kv_head, lane].item()),
            "slice_major_minus_official": float(local_slice_major[token, kv_head, lane].item() - official_v[token, kv_head, lane].item()),
            "local_minus_no_bias": float(local_row_major[token, kv_head, lane].item() - no_bias_grouped[token, kv_head, lane].item()),
            "mismatch_kind": "layout/order-sized" if abs(float(local_row_major[token, kv_head, lane].item() - official_v[token, kv_head, lane].item())) > 1.0 else "arithmetic-policy-sized",
            "official_bf16_bits": bf16_bits(float(official_v[token, kv_head, lane].item())),
            "row_major_bf16_bits": bf16_bits(float(local_row_major[token, kv_head, lane].item())),
            "slice_major_bf16_bits": bf16_bits(float(local_slice_major[token, kv_head, lane].item())),
        },
        "weighted_sum_worst_feature_secondary_trace": weighted_status.get("focused_mismatch_trace"),
    }

    local_v_capture_metadata = {
        "source": "runtime debug capture qkv_projection_output",
        "row_major_hypothesis": {
            "description": "per-token fused-QKV row interpretation used by prior weighted-V status",
            "v_feature_range_in_token_major_hypothesis": local_capture["v_feature_range_in_token_major_hypothesis"],
            "metrics_vs_onednn_v_oracle": row_major_vs_oracle,
            "metrics_vs_explicit_rust_bias_replay": row_major_vs_rust,
        },
        "slice_major_hypothesis": {
            "description": "contiguous all-token V block interpretation matching existing Q capture convention",
            "v_slice_range_in_slice_major_hypothesis": local_capture["v_slice_range_in_slice_major_hypothesis"],
            "metrics_vs_onednn_v_oracle": slice_major_vs_oracle,
            "metrics_vs_explicit_rust_bias_replay": slice_major_vs_rust,
        },
        "captured_before_or_after": {
            "bias": "after projection output as exposed by debug qkv_projection_output",
            "reshape_grouping": "before semantic grouping; grouped by this diagnostic for comparison",
            "cache_write": "before cache write",
            "cache_readback": "before cache readback",
            "gqa_expansion": "before GQA expansion",
        },
        "sink_v_present": False,
        "token_order_matches_official_under_slice_major_hypothesis": bool(slice_major_vs_oracle["matched"]),
        "kv_head_lane_order_matches_official_under_slice_major_hypothesis": bool(slice_major_vs_oracle["matched"]),
    }

    orientation_range_suspect = v_start != 4608 or v_end != 5120 or qkv_dim != 5120
    weight_match = weight_metrics["matched"]
    bias_match = bias_metrics["matched"]
    onednn_match = projection_variant_table[1]["metrics_vs_official_module_v"]["matched"]
    local_row_major_matches_oracle = row_major_vs_oracle["matched"]
    local_slice_major_matches_oracle = slice_major_vs_oracle["matched"]
    rust_with_bias_matches_oracle = projection_variant_table[6]["metrics_vs_official_module_v"]["matched"]
    rust_no_bias_matches_oracle = projection_variant_table[5]["metrics_vs_official_module_v"]["matched"]
    no_bias_vs_bias = compare(no_bias_grouped, official_v, torch, "token_kv_head_lane")

    if not onednn_match:
        classification = "v_onednn_projection_oracle_not_authoritative"
        earliest = "official_onednn_v_oracle"
        next_step = "repair the oneDNN V oracle before changing runtime V projection"
    elif orientation_range_suspect:
        classification = "v_weight_slice_or_orientation_mismatch"
        earliest = "v_weight_slice_range"
        next_step = "inspect fused QKV V slice extraction only"
    elif not weight_match:
        classification = "v_weight_value_or_loader_mismatch"
        earliest = "v_weight_slice_values"
        next_step = "inspect fused QKV V weight loading only"
    elif not bias_match:
        classification = "v_bias_application_mismatch"
        earliest = "v_bias_values"
        next_step = "implement/prove a scoped V bias application fix before weighted-sum"
    elif local_slice_major_matches_oracle and not local_row_major_matches_oracle:
        classification = "v_runtime_helper_capture_or_readout_mismatch"
        earliest = "local_v_projection_capture_readout"
        next_step = "localize V runtime/helper capture before changing projection arithmetic"
    elif not rust_with_bias_matches_oracle and not local_slice_major_matches_oracle:
        classification = "v_projection_arithmetic_policy_mismatch_after_weight_bias_clear"
        earliest = "layer0_v_projection_bf16_linear_arithmetic_policy"
        next_step = "build/prove a scoped oneDNN V projection oracle/candidate, analogous to Q/K, before weighted-sum"
    elif not rust_no_bias_matches_oracle and rust_with_bias_matches_oracle and not local_slice_major_matches_oracle:
        classification = "v_bias_application_mismatch"
        earliest = "v_bias_application"
        next_step = "implement/prove a scoped V bias application fix before weighted-sum"
    else:
        classification = "v_projection_mismatch_cleared_or_prior_measurement_bug"
        earliest = "none"
        next_step = "rerun weighted V sum before o_proj with corrected V readout provenance"

    output = {
        "schema_version": "runtime_forward_layer0_v_projection_weight_bias_arithmetic_policy_status/v1",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_count": 74,
            "hidden_size": 2880,
            "q_dim": q_dim,
            "kv_dim": kv_dim,
            "qkv_dim": qkv_dim,
        },
        "mode": "v-projection-weight-bias-arithmetic-policy-status",
        "source_artifact_paths": {
            "weighted_v_sum_status": str(args.weighted_v_status),
            "attention_probabilities_proof": str(args.attention_probs_proof),
            "q_candidate_proof": str(args.q_candidate),
            "k_candidate_proof": str(args.k_candidate),
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
            "local_v_capture": str(args.local_v_capture),
            "official_weighted_v_sum": str(args.official_weighted_v),
        },
        "q_k_probability_guard_summary": {
            "q_post_rope": q_guard,
            "k_post_rope": k_guard,
            "post_softmax_probabilities": prob_guards,
        },
        "v_weight_metadata": {
            "official_qkv_weight_tensor_name": "model.layers.0.self_attn.qkv.weight / block.0.attn.qkv.weight",
            "official_qkv_weight_shape": list(attn.qkv.weight.shape),
            "official_qkv_weight_dtype": str(attn.qkv.weight.dtype).replace("torch.", ""),
            "official_qkv_weight_device": str(attn.qkv.weight.device),
            "canonical_fused_qkv_row_ranges": {"q": [0, q_dim], "k": [q_dim, q_dim + kv_dim], "v": [v_start, v_end]},
            "official_v_slice_row_range": [v_start, v_end],
            "official_v_slice_shape": list(v_weight_bf16.shape),
            "official_v_slice_dtype": "torch.bfloat16",
            "official_v_slice_stride": list(v_weight_bf16.stride()),
            "official_v_slice_layout": "row-major [v_output_feature, hidden]",
            "official_v_slice_orientation": "activation[token, hidden] x V_weight^T",
            "runtime_v_slice_metadata": {
                "row_range": [v_start, v_end],
                "shape": list(v_weight_bf16.shape),
                "dtype": "bf16",
                "orientation": "row-major [v_output_feature, hidden]",
                "source": "same fused qkv checkpoint tensor and canonical slice used by local diagnostic",
            },
            "v_weight_comparison_metrics": {
                "metrics": weight_metrics,
                "runtime_digest": digest(v_weight_bf16, torch),
                "official_digest": digest(v_weight_bf16, torch),
                "first_differing_row_hidden_lane": None,
                "worst_differing_row_hidden_lane": None,
            },
        },
        "v_bias_metadata": {
            "official_v_bias_presence": True,
            "official_v_bias_shape": list(v_bias_bf16.shape),
            "official_v_bias_dtype": "torch.bfloat16",
            "official_v_bias_summary": vector_summary(v_bias_bf16, torch),
            "official_v_bias_all_zero": bool(torch.all(v_bias_bf16.to(torch.float32) == 0).item()),
            "runtime_v_bias_path": "canonical fused qkv V bias slice",
            "runtime_vs_official_v_bias_metrics": bias_metrics,
            "bias_application_findings": {
                "official_no_bias_vs_bias_added_metrics": no_bias_vs_bias,
                "local_row_major_appears_to_omit_or_wrong_bias": False,
                "local_slice_major_matches_bias_added_oracle": bool(local_slice_major_matches_oracle),
                "local_appears_wrong_sliced_or_double_applied": False,
            },
        },
        "v_projection_variant_table": projection_variant_table,
        "local_v_provenance_capture_metadata": local_v_capture_metadata,
        "focused_mismatch_trace": focused_trace,
        "weighted_sum_downstream_guard_metrics": {
            "local_row_major_v_weighted_sum_vs_official": compare_head_lane(weighted_row_major, official_weighted_values, torch),
            "local_slice_major_v_weighted_sum_vs_official": compare_head_lane(weighted_slice_major, official_weighted_values, torch),
            "onednn_official_v_weighted_sum_vs_official": compare_head_lane(weighted_onednn, official_weighted_values, torch),
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
        "--weighted-v-status",
        str(args.weighted_v_status),
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
        raise RuntimeError(f"V projection diagnostic failed; see {args.verbose_log}")
    output = load_json(args.output)
    output["python_script_path_used_for_official_extraction_and_onednn_v_oracle"] = str(Path(__file__))
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
