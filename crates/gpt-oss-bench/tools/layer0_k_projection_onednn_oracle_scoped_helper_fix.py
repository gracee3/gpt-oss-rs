#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import layer0_k_projection_onednn_oracle_helper_proof as proof
import layer0_k_projection_pytorch_bf16_linear_backend_policy as base


KNOWN_LANES = base.KNOWN_LANES
WORST_TOKEN = base.WORST_TOKEN
WORST_FEATURE = base.WORST_FEATURE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scoped bench-only oneDNN oracle K projection helper candidate proof."
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


def metrics_from_previous(entry):
    metrics = dict(entry["metrics"])
    if "differing_tokens" in metrics:
        metrics["mismatching_token_count"] = metrics["differing_tokens"]
    if "differing_lanes" in metrics:
        metrics["mismatching_lane_count"] = metrics["differing_lanes"]
    return metrics


def compare_full(lhs, rhs, torch):
    diff = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        coords = mismatch.nonzero(as_tuple=False)
        first_token = int(coords[0, 0].item())
        first_feature = int(coords[0, 1].item())
        flat = diff.reshape(-1)
        worst_flat = int(flat.argmax().item())
        width = int(diff.shape[1])
        worst_token = worst_flat // width
        worst_feature = worst_flat % width
        first = {
            "token_index": first_token,
            "k_feature_index": first_feature,
            "kv_head_index": first_feature // 64,
            "head_dim_lane": first_feature % 64,
            "lhs_value": float(lhs[first_token, first_feature].to(torch.float32).item()),
            "rhs_value": float(rhs[first_token, first_feature].to(torch.float32).item()),
            "abs_diff": float(diff[first_token, first_feature].item()),
        }
        worst = {
            "token_index": worst_token,
            "k_feature_index": worst_feature,
            "kv_head_index": worst_feature // 64,
            "head_dim_lane": worst_feature % 64,
            "lhs_value": float(lhs[worst_token, worst_feature].to(torch.float32).item()),
            "rhs_value": float(rhs[worst_token, worst_feature].to(torch.float32).item()),
            "abs_diff": float(diff[worst_token, worst_feature].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs.to(torch.float32), rhs.to(torch.float32))),
        "mismatching_token_count": int(mismatch.any(dim=1).sum().item()),
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_token_feature_or_head_lane": first,
        "worst_differing_token_feature_or_head_lane": worst,
    }


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source["exact_case"]["case_id"] != "developer-message-user-smoke":
        raise ValueError("source artifact is not the exact smoke case")
    if source["classification"] != "onednn_projection_oracle_confirms_helper_arithmetic_delta":
        raise ValueError("source artifact is not the expected helper-delta proof")

    primitive_args = argparse.Namespace(
        source=Path(source["source_artifact_paths"]["onednn_primitive_reproduction"]),
        official_weight_arithmetic=args.official_weight_arithmetic,
        rmsnorm_replay=args.rmsnorm_replay,
        grouped_source_breakdown=args.grouped_source_breakdown,
    )
    (
        _proof_source,
        _backend_source,
        _,
        _,
        norm,
        k_weight,
        official,
        _official_weight_status,
        official_projection_status,
        grouped_status,
        _official_projection_status_path,
    ) = proof.load_inputs(primitive_args, torch)

    norm_bf16 = norm.to(device=device, dtype=torch.bfloat16).contiguous()
    weight_bf16 = k_weight.to(device=device, dtype=torch.bfloat16).contiguous()
    official_f32 = official.to(torch.float32)
    original_threads = torch.get_num_threads()
    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        onednn_oracle = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn

    # Candidate strategy is intentionally bench/proof-only: use the exact full-shape
    # oneDNN BF16 oracle as the K projection candidate, then only reinterpret it as
    # grouped [token, kv_head, lane] for the immediate pre-RoPE confirmation.
    candidate_projection = onednn_oracle
    candidate_grouped = candidate_projection.reshape(74, 8, 64)
    official_grouped = official_f32.reshape(74, 8, 64)

    oracle_vs_official = compare_full(onednn_oracle, official_f32, torch)
    candidate_vs_oracle = compare_full(candidate_projection, onednn_oracle.to(torch.float32), torch)
    candidate_vs_official = compare_full(candidate_projection, official_f32, torch)
    candidate_grouped_vs_official = compare_full(
        candidate_grouped.reshape(74, 512),
        official_grouped.reshape(74, 512),
        torch,
    )

    arithmetic = official_projection_status["projection_arithmetic_comparison_table"]
    legacy_rust = metrics_from_previous(
        arithmetic["rust_cpu_replay_authoritative_input_runtime_k_weight_vs_official_module"]
    )
    legacy_helper = metrics_from_previous(
        arithmetic["rust_pedantic_helper_authoritative_input_runtime_k_weight_vs_official_module"]
    )
    legacy_grouped = {
        **grouped_status["per_stage_source_breakdown_metrics"]["grouped_pre_rope_k_layout_view"][
            "metrics"
        ],
        "mismatching_token_count": legacy_helper["mismatching_token_count"],
        "mismatching_lane_count": legacy_helper["mismatching_lane_count"],
        "first_differing_token_feature_or_head_lane": grouped_status[
            "per_stage_source_breakdown_metrics"
        ]["grouped_pre_rope_k_layout_view"]["first_differing_token_head_lane"],
        "worst_differing_token_feature_or_head_lane": grouped_status[
            "per_stage_source_breakdown_metrics"
        ]["grouped_pre_rope_k_layout_view"]["worst_differing_token_head_lane"],
    }

    previous_rows = {
        (int(row["token_index"]), int(row["k_feature_index"])): row
        for row in source["known_six_lane_table"]
    }
    known_table = []
    for token, feature in KNOWN_LANES:
        previous = previous_rows[(token, feature)]
        candidate_value = float(candidate_projection[token, feature].to(torch.float32).item())
        oracle_value = float(onednn_oracle[token, feature].to(torch.float32).item())
        known_table.append(
            {
                "token_index": token,
                "k_feature_index": feature,
                "kv_head_index": feature // 64,
                "head_dim_lane": feature % 64,
                "onednn_oracle_value": oracle_value,
                "official_value": float(official_f32[token, feature].item()),
                "legacy_helper_value": previous["pedantic_helper_value"],
                "legacy_rust_replay_value": previous["rust_cpu_replay_value"],
                "scoped_candidate_value": candidate_value,
                "candidate_minus_oracle": candidate_value - oracle_value,
                "legacy_helper_minus_oracle": previous["pedantic_helper_value"] - oracle_value,
                "candidate_matches_oracle": candidate_value == oracle_value,
            }
        )

    if not oracle_vs_official["matched"]:
        classification = "onednn_k_projection_oracle_regressed"
        next_step = "rebuild the oneDNN oracle before evaluating helper changes"
    elif candidate_vs_oracle["matched"] and candidate_vs_official["matched"] and candidate_grouped_vs_official["matched"]:
        classification = "scoped_onednn_oracle_k_projection_helper_fix_proven"
        next_step = "rerun grouped post-RoPE K and first attention-score consumer seams"
    elif candidate_vs_oracle["matched"] and candidate_vs_official["matched"]:
        classification = "k_projection_fixed_but_grouped_pre_rope_k_still_mismatches"
        next_step = "re-enter grouped_pre_rope_k readout/layout breakdown with candidate K projection"
    elif candidate_vs_oracle["max_abs_diff"] < legacy_helper["max_abs_diff"]:
        classification = "scoped_onednn_oracle_k_projection_helper_fix_partial"
        next_step = "tighten the scoped candidate against the full-shape oneDNN oracle"
    else:
        classification = "scoped_onednn_oracle_k_projection_helper_fix_not_sufficient"
        next_step = "document safe projection-policy boundary before runtime change"

    output = {
        "schema_version": "runtime_forward_layer0_k_projection_onednn_oracle_scoped_helper_fix_status/v1",
        "exact_case": source["exact_case"],
        "mode": "k-projection-onednn-oracle-scoped-helper-fix-status",
        "source_artifact_paths": {
            "onednn_oracle_helper_proof": str(args.source),
            "onednn_primitive_reproduction": ".live/runtime-forward-layer0-k-consumption-20260423/developer-message.runner-layer0-k-projection-onednn-primitive-reproduction-status.json",
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
            "official_weight_arithmetic": str(args.official_weight_arithmetic),
            "grouped_source_breakdown": str(args.grouped_source_breakdown),
        },
        "files_changed": [
            "crates/gpt-oss-bench/src/bin/runtime_forward_layer0_qkv_bf16_candidate_status.rs",
            "crates/gpt-oss-bench/tools/layer0_k_projection_onednn_oracle_scoped_helper_fix.py",
        ],
        "change_scope": "bench/proof-only",
        "runtime_affecting": False,
        "candidate_helper_projection_strategy": "bench-only full-shape oneDNN/MKLDNN BF16 torch.nn.functional.linear oracle output, reshaped to grouped [token, kv_head, lane] only for immediate pre-RoPE confirmation",
        "scope_statement": {
            "affected_paths": [
                "this diagnostic mode",
                "bench-only oneDNN K projection candidate for layer0 exact case",
            ],
            "not_affected_paths": [
                "default runtime helper",
                "Q projection",
                "V projection",
                "MLP/MoE",
                "logits",
                "final RMSNorm",
                "general GEMM behavior",
                "RoPE",
                "attention scores",
                "cache",
            ],
        },
        "onednn_oracle_construction_metadata": {
            "torch_version": torch.__version__,
            "thread_count": int(original_threads),
            "mkldnn_enabled_for_oracle": True,
            "input_shape_stride_dtype": {
                "shape": [74, 2880],
                "stride": list(norm_bf16.stride()),
                "dtype": "bfloat16",
            },
            "weight_shape_stride_dtype": {
                "shape": [512, 2880],
                "stride": list(weight_bf16.stride()),
                "dtype": "bfloat16",
            },
            "output_shape_stride_dtype": {
                "shape": [74, 512],
                "stride": list(onednn_oracle.stride()),
                "dtype": "bfloat16",
            },
        },
        "oracle_vs_official_metrics": oracle_vs_official,
        "legacy_helper_replay_metrics": {
            "legacy_pedantic_helper_vs_onednn_oracle": legacy_helper,
            "legacy_rust_cpu_replay_vs_onednn_oracle": legacy_rust,
        },
        "candidate_helper_projection_metrics": {
            "scoped_candidate_vs_onednn_oracle": candidate_vs_oracle,
            "scoped_candidate_vs_official_captured_or_module_k": candidate_vs_official,
        },
        "grouped_pre_rope_k_before_after_metrics": {
            "legacy_grouped_pre_rope_k_vs_official": legacy_grouped,
            "scoped_candidate_grouped_pre_rope_k_vs_official": candidate_grouped_vs_official,
        },
        "known_six_lane_table": known_table,
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
        raise RuntimeError(f"oneDNN oracle scoped helper fix proof failed; see {args.verbose_log}")
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
