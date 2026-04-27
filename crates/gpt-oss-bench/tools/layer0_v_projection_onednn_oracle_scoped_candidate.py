#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base
import layer0_q_pre_post_rope_runtime_localization as qloc
import layer0_v_projection_weight_bias_arithmetic_policy as vpol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bench-only full-shape oneDNN V projection candidate proof."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-v-policy", type=Path, required=True)
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


def digest(tensor, torch):
    return qloc.digest_tensor(tensor, torch)


def bf16_bits(value):
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


def variant_entry(name, tensor, reference, torch, layout, metric_name):
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "serialization_dtype": "f32-expanded BF16 values",
        "layout_interpretation": layout,
        metric_name: compare(tensor, reference, torch, "token_kv_head_lane"),
    }


def weighted_entry(name, tensor, official, torch):
    return {
        "name": name,
        "metrics_vs_official_weighted_v_sum": vpol.compare_head_lane(tensor, official, torch),
    }


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)

    source_v_policy = load_json(args.source_v_policy)
    weighted_status = load_json(args.weighted_v_status)
    probs_proof = load_json(args.attention_probs_proof)
    q_candidate = load_json(args.q_candidate)
    k_candidate = load_json(args.k_candidate)
    local_capture = load_json(args.local_v_capture)
    official_weighted = load_json(args.official_weighted_v)
    if source_v_policy.get("classification") != "v_projection_arithmetic_policy_mismatch_after_weight_bias_clear":
        raise ValueError("source V policy artifact is not the expected weight/bias-cleared state")
    if weighted_status.get("classification") != "attention_weighted_v_sum_blocked_by_v_provenance_mismatch":
        raise ValueError("source weighted-V artifact is not the expected V provenance mismatch")
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
    v_weight_bf16 = attn.qkv.weight[v_start:v_end, :].detach().to(torch.bfloat16).contiguous()
    v_bias_bf16 = attn.qkv.bias[v_start:v_end].detach().to(torch.bfloat16).contiguous()

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    original_threads = int(torch.get_num_threads())
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        full_qkv = attn.qkv(norm_bf16).contiguous()
        official_module_v = full_qkv[:, v_start:v_end].contiguous()
        candidate_v = torch.nn.functional.linear(norm_bf16, v_weight_bf16, v_bias_bf16).contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn

    old_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = False
    with torch.inference_mode():
        rust_no_bias = torch.nn.functional.linear(norm_bf16, v_weight_bf16, None).to(torch.bfloat16).contiguous()
        rust_with_bias = torch.nn.functional.linear(norm_bf16, v_weight_bf16, v_bias_bf16).to(torch.bfloat16).contiguous()
    torch.backends.mkldnn.enabled = old_mkldnn

    candidate_grouped = candidate_v.reshape(74, 8, 64).to(torch.float32)
    official_grouped = official_module_v.reshape(74, 8, 64).to(torch.float32)
    local_row_major = torch.tensor(
        local_capture["row_major_hypothesis_v_f32"], dtype=torch.float32, device=device
    ).reshape(74, 8, 64)
    local_slice_major = torch.tensor(
        local_capture["slice_major_hypothesis_v_f32"], dtype=torch.float32, device=device
    ).reshape(74, 8, 64)
    rust_no_bias_grouped = rust_no_bias.reshape(74, 8, 64).to(torch.float32)
    rust_with_bias_grouped = rust_with_bias.reshape(74, 8, 64).to(torch.float32)

    official_probs = torch.tensor(
        load_json(Path(probs_proof["official_post_softmax_reference_path"]))["values"],
        dtype=torch.float32,
        device=device,
    ).reshape(64, 75)
    official_weighted_values = torch.tensor(official_weighted["values"], dtype=torch.float32, device=device)
    weighted_candidate = vpol.weighted_sum(official_probs, candidate_grouped, torch)
    weighted_official_v = vpol.weighted_sum(official_probs, official_grouped, torch)
    weighted_slice_major = vpol.weighted_sum(official_probs, local_slice_major, torch)
    weighted_row_major = vpol.weighted_sum(official_probs, local_row_major, torch)

    candidate_vs_official = compare(candidate_grouped, official_grouped, torch, "token_kv_head_lane")
    local_slice_vs_candidate = compare(local_slice_major, candidate_grouped, torch, "token_kv_head_lane")
    local_row_vs_candidate = compare(local_row_major, candidate_grouped, torch, "token_kv_head_lane")
    rust_bias_vs_candidate = compare(rust_with_bias_grouped, candidate_grouped, torch, "token_kv_head_lane")
    rust_no_bias_vs_candidate = compare(rust_no_bias_grouped, candidate_grouped, torch, "token_kv_head_lane")
    candidate_weighted_vs_official = vpol.compare_head_lane(weighted_candidate, official_weighted_values, torch)

    token = 59
    kv_head = 5
    lane = 11
    provenance_trace = {
        "token_index": token,
        "kv_head": kv_head,
        "lane": lane,
        "official_onednn_v_value": float(official_grouped[token, kv_head, lane].item()),
        "candidate_v_value": float(candidate_grouped[token, kv_head, lane].item()),
        "local_row_major_readout_value": float(local_row_major[token, kv_head, lane].item()),
        "local_slice_major_readout_value": float(local_slice_major[token, kv_head, lane].item()),
    }
    provenance_trace["candidate_minus_official"] = (
        provenance_trace["candidate_v_value"] - provenance_trace["official_onednn_v_value"]
    )
    provenance_trace["slice_major_minus_official"] = (
        provenance_trace["local_slice_major_readout_value"] - provenance_trace["official_onednn_v_value"]
    )
    provenance_trace["row_major_minus_official"] = (
        provenance_trace["local_row_major_readout_value"] - provenance_trace["official_onednn_v_value"]
    )
    provenance_trace["official_bf16_bits"] = bf16_bits(provenance_trace["official_onednn_v_value"])
    provenance_trace["candidate_bf16_bits"] = bf16_bits(provenance_trace["candidate_v_value"])
    provenance_trace["row_major_bf16_bits"] = bf16_bits(provenance_trace["local_row_major_readout_value"])
    provenance_trace["slice_major_bf16_bits"] = bf16_bits(provenance_trace["local_slice_major_readout_value"])

    flat = 11
    q_head = 0
    weighted_trace = {
        "flattened_feature": flat,
        "q_head": q_head,
        "lane": flat % 64,
        "mapped_kv_head": q_head // 8,
        "official_weighted_v_value": float(official_weighted_values[flat].item()),
        "candidate_v_weighted_sum_value": float(weighted_candidate[flat].item()),
        "local_slice_major_v_weighted_sum_value": float(weighted_slice_major[flat].item()),
        "local_row_major_v_weighted_sum_value": float(weighted_row_major[flat].item()),
    }
    weighted_trace["candidate_minus_official"] = (
        weighted_trace["candidate_v_weighted_sum_value"] - weighted_trace["official_weighted_v_value"]
    )
    weighted_trace["slice_major_minus_official"] = (
        weighted_trace["local_slice_major_v_weighted_sum_value"] - weighted_trace["official_weighted_v_value"]
    )
    weighted_trace["row_major_minus_official"] = (
        weighted_trace["local_row_major_v_weighted_sum_value"] - weighted_trace["official_weighted_v_value"]
    )
    weighted_trace["candidate_clears_the_weighted_sum_mismatch"] = (
        weighted_trace["candidate_minus_official"] == 0.0
    )

    weighted_metrics = {
        "candidate_v_weighted_sum_vs_official": vpol.compare_head_lane(weighted_candidate, official_weighted_values, torch),
        "official_onednn_v_weighted_sum_vs_official": vpol.compare_head_lane(weighted_official_v, official_weighted_values, torch),
        "local_slice_major_v_weighted_sum_vs_official": vpol.compare_head_lane(weighted_slice_major, official_weighted_values, torch),
        "local_row_major_v_weighted_sum_vs_official": vpol.compare_head_lane(weighted_row_major, official_weighted_values, torch),
    }

    if not candidate_vs_official["matched"]:
        classification = "onednn_v_projection_candidate_not_authoritative"
        earliest = "layer0_v_projection_candidate"
        next_step = "inspect oneDNN V oracle construction before weighted-sum"
    elif not candidate_weighted_vs_official["matched"]:
        classification = "weighted_v_sum_mismatch_after_onednn_v_candidate"
        earliest = "layer0_final_token_attention_weighted_value_sum_before_output_projection"
        next_step = "inspect weighted V sum dtype/GQA/sink-drop policy with V provenance cleared"
    elif not local_slice_vs_candidate["matched"] or not local_row_vs_candidate["matched"]:
        classification = "onednn_v_candidate_confirms_runtime_v_projection_policy_delta"
        earliest = "layer0_v_projection_bf16_linear_arithmetic_policy"
        next_step = "ask PPP for exactly layer0_final_token_attention_output_after_o_proj_before_residual"
    else:
        classification = "onednn_v_candidate_clears_weighted_v_sum_pre_o_proj"
        earliest = "none"
        next_step = "ask PPP for exactly layer0_final_token_attention_output_after_o_proj_before_residual"

    output = {
        "schema_version": "runtime_forward_layer0_v_projection_onednn_oracle_scoped_candidate_status/v1",
        "mode": "v-projection-onednn-oracle-scoped-candidate-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "token_count": 74,
            "hidden_size": 2880,
            "q_dim": q_dim,
            "kv_dim": kv_dim,
            "qkv_dim": qkv_dim,
        },
        "source_artifact_paths": {
            "v_weight_bias_arithmetic_policy": str(args.source_v_policy),
            "weighted_v_sum_status": str(args.weighted_v_status),
            "attention_probabilities_proof": str(args.attention_probs_proof),
            "q_candidate_proof": str(args.q_candidate),
            "k_candidate_proof": str(args.k_candidate),
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
            "local_v_capture": str(args.local_v_capture),
            "official_weighted_v_sum": str(args.official_weighted_v),
        },
        "files_changed": [
            "crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs",
            "crates/gpt-oss-bench/tools/layer0_v_projection_onednn_oracle_scoped_candidate.py",
        ],
        "change_scope": "bench/proof-only",
        "runtime_affecting": False,
        "candidate_v_projection_strategy": "bench-only full-shape oneDNN/MKLDNN BF16 torch.nn.functional.linear(authoritative_norm_bf16, V_weight_bf16, V_bias_bf16)",
        "scope_statement": {
            "affected_paths": [
                "this diagnostic/proof mode",
                "bench-only oneDNN V projection candidate for the exact layer0 smoke case",
                "weighted V sum confirmation using already-cleared attention probabilities",
            ],
            "not_affected_paths": [
                "default runtime V helper",
                "default runtime Q/K helpers",
                "QKV projection runtime behavior",
                "weighted-sum runtime behavior",
                "o_proj",
                "residual",
                "MLP",
                "logits",
                "MoE",
                "Harmony",
                "cache",
                "later layers",
            ],
        },
        "oneDNN_v_candidate_construction_metadata": {
            "torch_version": torch.__version__,
            "thread_count": original_threads,
            "mkldnn_enabled_for_candidate": True,
            "input": vpol.tensor_meta(norm_bf16, "[token, hidden]", "bf16"),
            "weight": vpol.tensor_meta(v_weight_bf16, "[v_output_feature, hidden]", "bf16"),
            "bias": vpol.tensor_meta(v_bias_bf16, "[v_output_feature]", "bf16"),
            "output": vpol.tensor_meta(candidate_v, "[token, v_output_feature]", "bf16"),
            "output_digest": digest(candidate_v, torch),
            "candidate_is_bench_proof_only": True,
            "cloned_contiguous_tensors_used": True,
            "v_slice_row_range": [v_start, v_end],
        },
        "v_candidate_vs_official_metrics": candidate_vs_official,
        "legacy_local_rust_v_metrics": {
            "local_row_major_v_vs_candidate": local_row_vs_candidate,
            "local_slice_major_v_vs_candidate": local_slice_vs_candidate,
            "explicit_rust_cpu_v_replay_with_bias_vs_candidate": rust_bias_vs_candidate,
            "explicit_rust_cpu_v_replay_without_bias_vs_candidate": rust_no_bias_vs_candidate,
        },
        "v_projection_variant_table": [
            variant_entry(
                "official_module_v_projection",
                official_grouped,
                official_grouped,
                torch,
                "[token, kv_head, lane]",
                "metrics_vs_candidate_v",
            ),
            variant_entry(
                "onednn_v_projection_candidate",
                candidate_grouped,
                official_grouped,
                torch,
                "[token, kv_head, lane]",
                "metrics_vs_official_module_v",
            ),
            variant_entry(
                "local_runtime_v_slice_major_hypothesis",
                local_slice_major,
                candidate_grouped,
                torch,
                "[token, kv_head, lane] from contiguous V block after all-token Q and K blocks",
                "metrics_vs_candidate_v",
            ),
            variant_entry(
                "local_runtime_v_row_major_hypothesis",
                local_row_major,
                candidate_grouped,
                torch,
                "[token, kv_head, lane] from per-token fused-QKV row slices",
                "metrics_vs_candidate_v",
            ),
            variant_entry(
                "explicit_rust_cpu_v_replay_runtime_weight_official_bias",
                rust_with_bias_grouped,
                candidate_grouped,
                torch,
                "[token, kv_head, lane]",
                "metrics_vs_candidate_v",
            ),
            variant_entry(
                "explicit_rust_cpu_v_replay_runtime_weight_no_bias",
                rust_no_bias_grouped,
                candidate_grouped,
                torch,
                "[token, kv_head, lane]",
                "metrics_vs_candidate_v",
            ),
        ],
        "grouped_v_metrics": {
            "candidate_grouped_v_vs_official_grouped_v": candidate_vs_official,
            "local_slice_major_grouped_v_vs_candidate_grouped_v": local_slice_vs_candidate,
            "local_row_major_grouped_v_vs_candidate_grouped_v": local_row_vs_candidate,
        },
        "weighted_v_sum_before_o_proj_metrics": weighted_metrics,
        "focused_traces": {
            "prior_worst_row_major_provenance_lane": provenance_trace,
            "prior_worst_weighted_sum_lane": weighted_trace,
        },
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
        "--source-v-policy",
        str(args.source_v_policy),
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
        raise RuntimeError(f"V oneDNN scoped candidate proof failed; see {args.verbose_log}")
    output = load_json(args.output)
    output["python_script_path_used_for_onednn_v_candidate"] = str(Path(__file__))
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
