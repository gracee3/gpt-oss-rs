#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


KNOWN_LANES = base.KNOWN_LANES
WORST_TOKEN = base.WORST_TOKEN
WORST_FEATURE = base.WORST_FEATURE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full-shape oneDNN K projection oracle and compare helper paths."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-weight-arithmetic", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--grouped-source-breakdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_inputs(args, torch):
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source["exact_case"]["case_id"] != "developer-message-user-smoke":
        raise ValueError("source artifact is not the exact smoke case")
    if source["classification"] != "onednn_bf16_linear_thread_partition_required":
        raise ValueError("source artifact is not the expected oneDNN classification")
    discriminator_path = Path(source["source_artifact_path"])
    backend_source = json.loads(discriminator_path.read_text(encoding="utf-8"))
    base_args = argparse.Namespace(
        source=Path(backend_source["source_artifact_path"]),
        rmsnorm_replay=args.rmsnorm_replay,
        official_extraction=args.official_weight_arithmetic,
    )
    _, rmsnorm, extraction, norm, k_weight, official = base.load_inputs(base_args, torch)
    official_weight_status = json.loads(args.official_weight_arithmetic.read_text(encoding="utf-8"))
    official_projection_status_path = Path(
        ".live/runtime-forward-layer0-k-consumption-20260423/"
        "developer-message.runner-layer0-k-projection-official-weight-arithmetic-status.json"
    )
    official_projection_status = json.loads(official_projection_status_path.read_text(encoding="utf-8"))
    grouped_status = json.loads(args.grouped_source_breakdown.read_text(encoding="utf-8"))
    return (
        source,
        backend_source,
        rmsnorm,
        extraction,
        norm,
        k_weight,
        official,
        official_weight_status,
        official_projection_status,
        grouped_status,
        official_projection_status_path,
    )


def tensor_meta(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


def tensor_digest(values):
    # Small stable checksum for provenance, not cryptographic.
    import hashlib
    import struct

    h = hashlib.sha256()
    flat = values.reshape(-1).to(dtype=values.dtype)
    for value in flat[: min(4096, flat.numel())].to("cpu").to(dtype=values.dtype).reshape(-1):
        h.update(struct.pack("<f", float(value.to("cpu").to(dtype=values.dtype).item())))
    h.update(str(tuple(values.shape)).encode())
    h.update(str(values.dtype).encode())
    return "sha256-prefix4096:" + h.hexdigest()


def metrics_from_status_entry(entry):
    metrics = dict(entry["metrics"])
    if "differing_tokens" in metrics and "mismatching_token_count" not in metrics:
        metrics["mismatching_token_count"] = metrics["differing_tokens"]
    if "differing_lanes" in metrics and "mismatching_lane_count" not in metrics:
        metrics["mismatching_lane_count"] = metrics["differing_lanes"]
    return metrics


def compare_full(lhs, rhs, torch):
    diff = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs()
    mismatch = diff > 0
    token_any = mismatch.any(dim=1)
    lane_count = int(mismatch.sum().item())
    token_count = int(token_any.sum().item())
    first = None
    worst = None
    if lane_count:
        coords = mismatch.nonzero(as_tuple=False)
        first_token = int(coords[0, 0].item())
        first_feature = int(coords[0, 1].item())
        flat = diff.reshape(-1)
        worst_flat = int(flat.argmax().item())
        worst_token = worst_flat // int(diff.shape[1])
        worst_feature = worst_flat % int(diff.shape[1])
        first = {
            "token_index": first_token,
            "k_feature_index": first_feature,
            "lhs_value": float(lhs[first_token, first_feature].to(torch.float32).item()),
            "rhs_value": float(rhs[first_token, first_feature].to(torch.float32).item()),
            "abs_diff": float(diff[first_token, first_feature].item()),
        }
        worst = {
            "token_index": worst_token,
            "k_feature_index": worst_feature,
            "lhs_value": float(lhs[worst_token, worst_feature].to(torch.float32).item()),
            "rhs_value": float(rhs[worst_token, worst_feature].to(torch.float32).item()),
            "abs_diff": float(diff[worst_token, worst_feature].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs.to(torch.float32), rhs.to(torch.float32))),
        "mismatching_token_count": token_count,
        "mismatching_lane_count": lane_count,
        "first_differing_token_feature": first,
        "worst_differing_token_feature": worst,
    }


def known_values_from_tensor(tensor, torch):
    rows = []
    for token, feature in KNOWN_LANES:
        value = float(tensor[token, feature].to(torch.float32).item())
        rows.append(
            {
                "token_index": token,
                "k_feature_index": feature,
                "kv_head_index": feature // 64,
                "head_dim_lane": feature % 64,
                "value": value,
                "bf16_bits": f"0x{base.bf16_bits_from_float(value):04x}",
            }
        )
    return rows


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    (
        source,
        backend_source,
        _,
        _,
        norm,
        k_weight,
        official,
        _official_weight_status,
        official_projection_status,
        grouped_status,
        official_projection_status_path,
    ) = load_inputs(args, torch)

    model = base.load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    q_dim = model.attn.num_attention_heads * model.attn.head_dim
    kv_dim = model.attn.num_key_value_heads * model.attn.head_dim
    k_start = q_dim
    k_end = q_dim + kv_dim

    original_threads = torch.get_num_threads()
    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = True
    norm_bf16 = norm.to(device=device, dtype=torch.bfloat16).contiguous()
    weight_bf16 = k_weight.to(device=device, dtype=torch.bfloat16).contiguous()
    official_f32 = official.to(torch.float32)
    with torch.inference_mode():
        oracle = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous()
        official_module = model.attn.qkv(norm_bf16)[:, k_start:k_end].contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn

    oracle_vs_official = compare_full(oracle, official_f32, torch)
    oracle_vs_module = compare_full(oracle, official_module.to(torch.float32), torch)
    arithmetic = official_projection_status["projection_arithmetic_comparison_table"]
    grouped_metrics = grouped_status["per_stage_source_breakdown_metrics"]["grouped_pre_rope_k_layout_view"]
    helper_vs_oracle = {
        "rust_cpu_replay_vs_onednn_oracle": {
            "metrics": metrics_from_status_entry(
                arithmetic["rust_cpu_replay_authoritative_input_runtime_k_weight_vs_official_module"]
            ),
            "known_six_lane_values": [
                {
                    "token_index": row["token_index"],
                    "k_feature_index": row["k_feature_index"],
                    "value": row["rust_cpu_replay_value"],
                    "oracle_value": row["official_module_value"],
                    "abs_diff": abs(row["rust_cpu_replay_value"] - row["official_module_value"]),
                }
                for row in backend_source["known_6_lane_comparison_summary"]
            ],
        },
        "pedantic_cuda_helper_vs_onednn_oracle": {
            "metrics": metrics_from_status_entry(
                arithmetic["rust_pedantic_helper_authoritative_input_runtime_k_weight_vs_official_module"]
            ),
            "known_six_lane_values": [
                {
                    "token_index": row["token_index"],
                    "k_feature_index": row["k_feature_index"],
                    "value": row["pedantic_helper_value"],
                    "oracle_value": row["official_module_value"],
                    "abs_diff": abs(row["pedantic_helper_value"] - row["official_module_value"]),
                }
                for row in backend_source["known_6_lane_comparison_summary"]
            ],
        },
        "grouped_pre_rope_local_flattened_vs_onednn_oracle": {
            "metrics": {
                **grouped_metrics["metrics"],
                "mismatching_token_count": arithmetic[
                    "rust_pedantic_helper_authoritative_input_runtime_k_weight_vs_official_module"
                ]["metrics"]["differing_tokens"],
                "mismatching_lane_count": arithmetic[
                    "rust_pedantic_helper_authoritative_input_runtime_k_weight_vs_official_module"
                ]["metrics"]["differing_lanes"],
            },
            "first_differing_token_feature": grouped_metrics["first_differing_token_head_lane"],
            "worst_differing_token_feature": grouped_metrics["worst_differing_token_head_lane"],
        },
        "official_grouped_pre_rope_reference_vs_onednn_oracle_sanity": {
            "metrics": oracle_vs_official,
            "interpretation": "official grouped/pre-RoPE K shares the same [token, k_feature] values as the official module K slice",
        },
    }

    rows_by_trace = {
        int(row["token_index"]): row
        for row in official_projection_status["known_mismatch_token_trace"]
    }
    six_lane_table = []
    for row in backend_source["known_6_lane_comparison_summary"]:
        token = int(row["token_index"])
        feature = int(row["k_feature_index"])
        trace = rows_by_trace.get(token, {})
        oracle_value = float(oracle[token, feature].to(torch.float32).item())
        official_value = float(row["official_module_value"])
        rust_value = float(row["rust_cpu_replay_value"])
        helper_value = float(row["pedantic_helper_value"])
        local_grouped = float(trace.get("local_grouped_k_value", helper_value))
        lower_bits = min(base.bf16_bits_from_float(oracle_value), base.bf16_bits_from_float(helper_value))
        upper_bits = max(base.bf16_bits_from_float(oracle_value), base.bf16_bits_from_float(helper_value))
        midpoint = (base.bf16_value_from_bits(lower_bits) + base.bf16_value_from_bits(upper_bits)) / 2.0
        six_lane_table.append(
            {
                "token_index": token,
                "k_feature_index": feature,
                "kv_head_index": int(row["kv_head_index"]),
                "head_dim_lane": int(row["head_dim_lane"]),
                "onednn_oracle_value": oracle_value,
                "onednn_oracle_bf16_bits": f"0x{base.bf16_bits_from_float(oracle_value):04x}",
                "official_module_value": official_value,
                "official_module_bf16_bits": row["official_bf16_bits"],
                "rust_cpu_replay_value": rust_value,
                "rust_cpu_replay_bf16_bits": row["rust_bf16_bits"],
                "pedantic_helper_value": helper_value,
                "pedantic_helper_bf16_bits": row["rust_bf16_bits"],
                "local_grouped_value": local_grouped,
                "local_grouped_bf16_bits": f"0x{base.bf16_bits_from_float(local_grouped):04x}",
                "oracle_minus_rust_helper": oracle_value - helper_value,
                "abs_oracle_minus_rust_helper": abs(oracle_value - helper_value),
                "one_bf16_ulp_or_less": bool(row["one_bf16_ulp_or_less_at_this_magnitude"]),
                "bf16_midpoint_between_oracle_and_helper_bins": midpoint,
                "helper_is_near_rounding_midpoint": bool(row["one_bf16_ulp_or_less_at_this_magnitude"]),
            }
        )

    rust_metrics = helper_vs_oracle["rust_cpu_replay_vs_onednn_oracle"]["metrics"]
    helper_metrics = helper_vs_oracle["pedantic_cuda_helper_vs_onednn_oracle"]["metrics"]
    oracle_authoritative = bool(oracle_vs_official["matched"] and oracle_vs_module["matched"])
    rust_helper_identical = bool(
        arithmetic["rust_pedantic_helper_vs_rust_cpu_replay_runtime_k_weight"]["metrics"]["matched"]
    )
    if not oracle_authoritative:
        classification = "onednn_projection_oracle_not_authoritative"
        earliest = "onednn_oracle_construction"
        next_step = "fix oracle construction before using it as a helper target"
    elif rust_metrics["matched"] and helper_metrics["matched"]:
        classification = "k_projection_delta_cleared_or_prior_measurement_bug"
        earliest = "none"
        next_step = "rerun grouped_pre_rope_k current path status"
    elif rust_metrics["matched"] is False and helper_metrics["matched"] and not rust_helper_identical:
        classification = "rust_cpu_replay_projection_policy_stale"
        earliest = "rust_cpu_replay"
        next_step = "update only the Rust replay oracle before touching CUDA helper behavior"
    elif not rust_helper_identical:
        classification = "cuda_helper_projection_delta_after_onednn_oracle"
        earliest = "cuda_helper_projection"
        next_step = "separate helper integration from Rust replay before selecting a projection fix"
    else:
        classification = "onednn_projection_oracle_confirms_helper_arithmetic_delta"
        earliest = "BF16 projection accumulation/blocking policy in Rust/CUDA helper"
        next_step = "keep runtime unchanged; select one deliberately scoped helper proof/fix against the full-shape oneDNN oracle"

    output = {
        "schema_version": "runtime_forward_layer0_k_projection_onednn_oracle_helper_proof_status/v1",
        "exact_case": source["exact_case"],
        "mode": "k-projection-onednn-oracle-helper-proof-status",
        "source_artifact_paths": {
            "onednn_primitive_reproduction": str(args.source),
            "official_weight_arithmetic_status": str(official_projection_status_path),
            "official_weight_arithmetic_capture": str(args.official_weight_arithmetic),
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
            "grouped_source_breakdown": str(args.grouped_source_breakdown),
        },
        "python_script_path_used_for_onednn_oracle_construction": str(Path(__file__).resolve()),
        "onednn_verbose_log_path": str(args.verbose_log),
        "oracle_construction_metadata": {
            "torch_version": torch.__version__,
            "thread_count": int(original_threads),
            "mkldnn_enabled_for_oracle": True,
            "input": tensor_meta(norm_bf16),
            "weight": tensor_meta(weight_bf16),
            "output": tensor_meta(oracle),
            "output_digest": tensor_digest(oracle.to(torch.float32)),
        },
        "oracle_vs_official_metrics": {
            "onednn_oracle_vs_official_module_or_captured_k": oracle_vs_official,
            "onednn_oracle_vs_official_module_recomputed_k": oracle_vs_module,
            "known_six_lane_match_count": sum(
                1
                for row in six_lane_table
                if row["onednn_oracle_value"] == row["official_module_value"]
            ),
            "value_token65_feature293": float(oracle[WORST_TOKEN, WORST_FEATURE].to(torch.float32).item()),
        },
        "helper_replay_vs_oracle_metrics": helper_vs_oracle,
        "known_six_lane_table": six_lane_table,
        "earliest_divergent_source": earliest,
        "decision_framing": {
            "cuda_helper_projection_arithmetic_policy_mismatch_against_onednn_oracle": classification
            == "onednn_projection_oracle_confirms_helper_arithmetic_delta",
            "cpu_replay_only_issue": classification == "rust_cpu_replay_projection_policy_stale",
            "grouped_readout_issue": False,
            "oracle_construction_issue": classification == "onednn_projection_oracle_not_authoritative",
        },
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
        "--grouped-source-breakdown",
        str(args.grouped_source_breakdown),
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
        raise RuntimeError(f"oneDNN oracle helper proof failed; see {args.verbose_log}")
    output = json.loads(args.output.read_text(encoding="utf-8"))
    primitive_lines = [
        line
        for line in verbose_text.splitlines()
        if "onednn_verbose" in line or "dnnl_verbose" in line
    ]
    output["onednn_verbose_primitive_lines_sample"] = primitive_lines[:60]
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
