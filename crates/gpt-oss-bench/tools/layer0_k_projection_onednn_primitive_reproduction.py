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
        description="Reproduce oneDNN primitive conditions for layer0 K BF16 projection."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--official-extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_inputs_from_backend_source(args, torch):
    backend_source = json.loads(args.source.read_text(encoding="utf-8"))
    if backend_source["exact_case"]["case_id"] != "developer-message-user-smoke":
        raise ValueError("source artifact is not the exact smoke case")
    if backend_source["classification"] != "pytorch_bf16_linear_backend_or_threading_sensitive":
        raise ValueError("source artifact is not the expected backend-sensitive classification")
    discriminator_source = Path(backend_source["source_artifact_path"])
    load_args = argparse.Namespace(
        source=discriminator_source,
        rmsnorm_replay=args.rmsnorm_replay,
        official_extraction=args.official_extraction,
    )
    source, rmsnorm, extraction, norm, k_weight, official = base.load_inputs(load_args, torch)
    return backend_source, source, rmsnorm, extraction, norm, k_weight, official


def tensor_meta(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


def compare_path(name, output, official, torch, extra=None):
    item = base.compare_path(name, output.to(torch.float32), official.to(torch.float32), KNOWN_LANES)
    item["input_metadata"] = extra.get("input_metadata") if extra else None
    item["weight_metadata"] = extra.get("weight_metadata") if extra else None
    item["output_metadata"] = extra.get("output_metadata") if extra else tensor_meta(output)
    item["torch_thread_count"] = int(torch.get_num_threads())
    item["mkldnn_enabled"] = bool(torch.backends.mkldnn.enabled)
    item["matches_rust_helper_known_lanes"] = bool(
        all(
            value["value"] == value.get("rust_helper_value")
            for value in item.get("known_lane_values", [])
        )
    )
    return item


def compare_known_values(name, known_values, official, torch, input_meta, weight_meta):
    matched = 0
    rust_matched = 0
    rows = []
    for token, feature in KNOWN_LANES:
        value = float(known_values[(token, feature)])
        official_value = float(official[token, feature].item())
        rows.append(
            {
                "token_index": token,
                "k_feature_index": feature,
                "value": value,
                "official_value": official_value,
                "abs_diff": abs(value - official_value),
            }
        )
        matched += int(value == official_value)
    diffs = [row["abs_diff"] for row in rows]
    return {
        "name": name,
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "matched": matched == len(KNOWN_LANES),
        "mismatch_count": len(KNOWN_LANES) - matched,
        "known_lane_match_count": matched,
        "known_lane_count": len(KNOWN_LANES),
        "worst_lane_value_token65_feature293": float(known_values[(WORST_TOKEN, WORST_FEATURE)]),
        "known_lane_values": rows,
        "input_metadata": input_meta,
        "weight_metadata": weight_meta,
        "output_metadata": {"shape": "per-known-lane reduced calls", "dtype": "bfloat16"},
        "torch_thread_count": int(torch.get_num_threads()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "matches_rust_helper_known_lanes": bool(rust_matched == len(KNOWN_LANES)),
    }


def bits_from_tensor_value(value):
    return f"0x{base.bf16_bits_from_float(float(value)):04x}"


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    device = torch.device(args.device)
    backend_source, discriminator, _, extraction, norm, k_weight, official = load_inputs_from_backend_source(args, torch)
    model = base.load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    q_dim = model.attn.num_attention_heads * model.attn.head_dim
    kv_dim = model.attn.num_key_value_heads * model.attn.head_dim
    k_start = q_dim
    k_end = q_dim + kv_dim

    norm_bf16 = norm.to(device=device, dtype=torch.bfloat16)
    weight_bf16 = k_weight.to(device=device, dtype=torch.bfloat16)
    official_f32 = official.to(torch.float32)
    original_threads = torch.get_num_threads()
    original_mkldnn = bool(torch.backends.mkldnn.enabled)

    primitive_matrix = []
    verbose_notes = []
    with torch.inference_mode():
        full_qkv = model.attn.qkv(norm_bf16)[:, k_start:k_end].contiguous()
        primitive_matrix.append(
            compare_path(
                "full_official_module_path_full_qkv_slice",
                full_qkv,
                official_f32,
                torch,
                {
                    "input_metadata": tensor_meta(norm_bf16),
                    "weight_metadata": tensor_meta(model.attn.qkv.weight),
                    "output_metadata": tensor_meta(full_qkv),
                },
            )
        )

        full_k = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous()
        primitive_matrix.append(
            compare_path(
                "default_functional_linear_k_only_full_74x512",
                full_k,
                official_f32,
                torch,
                {
                    "input_metadata": tensor_meta(norm_bf16),
                    "weight_metadata": tensor_meta(weight_bf16),
                    "output_metadata": tensor_meta(full_k),
                },
            )
        )

        single_token_values = {}
        for token, feature in KNOWN_LANES:
            row_out = torch.nn.functional.linear(norm_bf16[token : token + 1, :], weight_bf16, None)
            single_token_values[(token, feature)] = row_out[0, feature].to(torch.float32).item()
        primitive_matrix.append(
            compare_known_values(
                "default_functional_linear_single_token_per_known_lane_full_k_weight",
                single_token_values,
                official_f32,
                torch,
                {"shape": [1, int(norm_bf16.shape[1])], "dtype": "bfloat16"},
                tensor_meta(weight_bf16),
            )
        )

        single_feature_values = {}
        for token, feature in KNOWN_LANES:
            feature_out = torch.nn.functional.linear(
                norm_bf16,
                weight_bf16[feature : feature + 1, :].contiguous(),
                None,
            )
            single_feature_values[(token, feature)] = feature_out[token, 0].to(torch.float32).item()
        primitive_matrix.append(
            compare_known_values(
                "default_functional_linear_single_feature_per_known_lane_all_74_tokens",
                single_feature_values,
                official_f32,
                torch,
                tensor_meta(norm_bf16),
                {"shape": [1, int(weight_bf16.shape[1])], "dtype": "bfloat16"},
            )
        )

        block_values = {}
        block_descriptors = []
        for token, feature in KNOWN_LANES:
            start = max(0, min(int(weight_bf16.shape[0]) - 32, feature - 16))
            end = start + 32
            block_out = torch.nn.functional.linear(
                norm_bf16,
                weight_bf16[start:end, :].contiguous(),
                None,
            )
            block_values[(token, feature)] = block_out[token, feature - start].to(torch.float32).item()
            block_descriptors.append({"feature": feature, "block_start": start, "block_end": end})
        block_item = compare_known_values(
            "default_functional_linear_32_feature_block_per_known_lane_all_74_tokens",
            block_values,
            official_f32,
            torch,
            tensor_meta(norm_bf16),
            {"shape": [32, int(weight_bf16.shape[1])], "dtype": "bfloat16"},
        )
        block_item["feature_blocks"] = block_descriptors
        primitive_matrix.append(block_item)

        cloned_input = norm_bf16.contiguous().clone()
        cloned_weight = weight_bf16.contiguous().clone()
        cloned_output = torch.nn.functional.linear(cloned_input, cloned_weight, None).contiguous()
        primitive_matrix.append(
            compare_path(
                "default_functional_linear_cloned_contiguous_input_weight",
                cloned_output,
                official_f32,
                torch,
                {
                    "input_metadata": tensor_meta(cloned_input),
                    "weight_metadata": tensor_meta(cloned_weight),
                    "output_metadata": tensor_meta(cloned_output),
                },
            )
        )

        view_weight = model.attn.qkv.weight[k_start:k_end, :]
        view_output = torch.nn.functional.linear(norm_bf16, view_weight, None).contiguous()
        primitive_matrix.append(
            compare_path(
                "default_functional_linear_original_view_stride_input_weight",
                view_output,
                official_f32,
                torch,
                {
                    "input_metadata": tensor_meta(norm_bf16),
                    "weight_metadata": tensor_meta(view_weight),
                    "output_metadata": tensor_meta(view_output),
                },
            )
        )

        torch.set_num_threads(1)
        one_thread_output = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous()
        primitive_matrix.append(
            compare_path(
                "default_functional_linear_set_num_threads_1",
                one_thread_output,
                official_f32,
                torch,
                {
                    "input_metadata": tensor_meta(norm_bf16),
                    "weight_metadata": tensor_meta(weight_bf16),
                    "output_metadata": tensor_meta(one_thread_output),
                },
            )
        )
        torch.set_num_threads(original_threads)
        restored_output = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous()
        primitive_matrix.append(
            compare_path(
                "default_functional_linear_original_thread_count_restored",
                restored_output,
                official_f32,
                torch,
                {
                    "input_metadata": tensor_meta(norm_bf16),
                    "weight_metadata": tensor_meta(weight_bf16),
                    "output_metadata": tensor_meta(restored_output),
                },
            )
        )

    source_models = backend_source["backend_informed_reduction_model_table"]
    chosen_model_names = {
        "current_rust_left_to_right_replay",
        "previous_chunked_16_replay",
        "backend_informed_chunked_32_replay",
        "simd_lane_32_partial_sum_pairwise_replay",
    }
    model_table = [item for item in source_models if item["name"] in chosen_model_names]
    best_model = max(
        model_table,
        key=lambda item: (
            item["lanes_matched_official"],
            -item["max_abs_diff_vs_official"],
            -item["mean_abs_diff_vs_official"],
        ),
    )

    rust_known = {
        (int(row["token_index"]), int(row["k_feature_index"])): float(row["rust_cpu_replay_value"])
        for row in backend_source["known_6_lane_comparison_summary"]
    }
    official_worst = float(official_f32[WORST_TOKEN, WORST_FEATURE].item())
    rust_worst = rust_known[(WORST_TOKEN, WORST_FEATURE)]
    official_bits = base.bf16_bits_from_float(official_worst)
    rust_bits = base.bf16_bits_from_float(rust_worst)
    midpoint = (
        base.bf16_value_from_bits(min(official_bits, rust_bits))
        + base.bf16_value_from_bits(max(official_bits, rust_bits))
    ) / 2.0
    variant_by_name = {item["name"]: item for item in primitive_matrix}

    def worst_value(name):
        return float(variant_by_name[name]["worst_lane_value_token65_feature293"])

    def worst_bits(name):
        return f"0x{base.bf16_bits_from_float(worst_value(name)):04x}"

    dependency_table = [
        {
            "dependency": "full_74_token_batch",
            "full_shape_match_count": variant_by_name["default_functional_linear_k_only_full_74x512"]["known_lane_match_count"],
            "reduced_shape_match_count": variant_by_name["default_functional_linear_single_token_per_known_lane_full_k_weight"]["known_lane_match_count"],
            "required_for_all_known_lanes": variant_by_name["default_functional_linear_single_token_per_known_lane_full_k_weight"]["known_lane_match_count"] < 6,
        },
        {
            "dependency": "full_512_feature_k_output",
            "full_shape_match_count": variant_by_name["default_functional_linear_k_only_full_74x512"]["known_lane_match_count"],
            "single_feature_match_count": variant_by_name["default_functional_linear_single_feature_per_known_lane_all_74_tokens"]["known_lane_match_count"],
            "small_block_match_count": variant_by_name["default_functional_linear_32_feature_block_per_known_lane_all_74_tokens"]["known_lane_match_count"],
            "required_for_all_known_lanes": (
                variant_by_name["default_functional_linear_single_feature_per_known_lane_all_74_tokens"]["known_lane_match_count"] < 6
                or variant_by_name["default_functional_linear_32_feature_block_per_known_lane_all_74_tokens"]["known_lane_match_count"] < 6
            ),
        },
        {
            "dependency": "original_tensor_stride_or_view",
            "cloned_match_count": variant_by_name["default_functional_linear_cloned_contiguous_input_weight"]["known_lane_match_count"],
            "view_match_count": variant_by_name["default_functional_linear_original_view_stride_input_weight"]["known_lane_match_count"],
            "required_for_all_known_lanes": (
                variant_by_name["default_functional_linear_cloned_contiguous_input_weight"]["known_lane_match_count"]
                != variant_by_name["default_functional_linear_original_view_stride_input_weight"]["known_lane_match_count"]
            ),
        },
        {
            "dependency": "default_thread_count",
            "default_match_count": variant_by_name["default_functional_linear_k_only_full_74x512"]["known_lane_match_count"],
            "one_thread_match_count": variant_by_name["default_functional_linear_set_num_threads_1"]["known_lane_match_count"],
            "required_for_all_known_lanes": variant_by_name["default_functional_linear_set_num_threads_1"]["known_lane_match_count"] < 6,
        },
        {
            "dependency": "onednn_mkldnn_enabled",
            "mkldnn_enabled": original_mkldnn,
            "required_for_observed_default_path": bool(torch.backends.mkldnn.is_available() and original_mkldnn),
        },
    ]
    six_lane_summary = backend_source["known_6_lane_comparison_summary"]
    output = {
        "schema_version": "runtime_forward_layer0_k_projection_onednn_primitive_reproduction_status/v1",
        "exact_case": backend_source["exact_case"],
        "mode": "k-projection-onednn-primitive-reproduction-status",
        "source_artifact_path": str(args.source),
        "python_script_path_used_for_onednn_primitive_capture": str(Path(__file__).resolve()),
        "pytorch_backend_metadata": {
            "torch_version": torch.__version__,
            "torch_config": torch.__config__.show(),
            "torch_num_threads": int(original_threads),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "mkldnn_is_available": bool(torch.backends.mkldnn.is_available()),
            "mkldnn_enabled": original_mkldnn,
            "input_shape": list(norm_bf16.shape),
            "weight_shape": list(weight_bf16.shape),
            "output_shape": [int(norm_bf16.shape[0]), int(weight_bf16.shape[0])],
        },
        "primitive_reproduction_matrix": primitive_matrix,
        "shape_layout_thread_dependency_table": dependency_table,
        "primitive_informed_model_table": model_table,
        "best_primitive_informed_model_summary": {
            "name": best_model["name"],
            "lanes_matched_official": best_model["lanes_matched_official"],
            "max_abs_diff_vs_official": best_model["max_abs_diff_vs_official"],
            "mean_abs_diff_vs_official": best_model["mean_abs_diff_vs_official"],
            "value_token65_feature293": best_model["value_token65_feature293"],
            "bf16_bits_token65_feature293": best_model["bf16_bits_token65_feature293"],
        },
        "six_lane_summary": six_lane_summary,
        "worst_lane_reproduction_summary": {
            "token_index": WORST_TOKEN,
            "k_feature_index": WORST_FEATURE,
            "official_module_value": official_worst,
            "official_module_bf16_bits": f"0x{official_bits:04x}",
            "default_full_shape_f_linear_value": worst_value("default_functional_linear_k_only_full_74x512"),
            "default_full_shape_f_linear_bf16_bits": worst_bits("default_functional_linear_k_only_full_74x512"),
            "single_token_f_linear_value": worst_value("default_functional_linear_single_token_per_known_lane_full_k_weight"),
            "single_token_f_linear_bf16_bits": worst_bits("default_functional_linear_single_token_per_known_lane_full_k_weight"),
            "single_feature_f_linear_value": worst_value("default_functional_linear_single_feature_per_known_lane_all_74_tokens"),
            "single_feature_f_linear_bf16_bits": worst_bits("default_functional_linear_single_feature_per_known_lane_all_74_tokens"),
            "small_feature_block_f_linear_value": worst_value("default_functional_linear_32_feature_block_per_known_lane_all_74_tokens"),
            "small_feature_block_f_linear_bf16_bits": worst_bits("default_functional_linear_32_feature_block_per_known_lane_all_74_tokens"),
            "one_thread_f_linear_value": worst_value("default_functional_linear_set_num_threads_1"),
            "one_thread_f_linear_bf16_bits": worst_bits("default_functional_linear_set_num_threads_1"),
            "rust_helper_value": rust_worst,
            "rust_helper_bf16_bits": f"0x{rust_bits:04x}",
            "bf16_midpoint": midpoint,
            "first_variant_flipping_from_official_bin_to_rust_bin": next(
                (
                    item["name"]
                    for item in primitive_matrix
                    if item["worst_lane_value_token65_feature293"] == rust_worst
                ),
                None,
            ),
        },
        "earliest_dependency_source_of_official_behavior": "full-shape oneDNN BF16 GEMM blocking and default thread partitioning",
    }
    full_matches = variant_by_name["default_functional_linear_k_only_full_74x512"]["known_lane_match_count"] == 6
    reduced_mismatch = (
        variant_by_name["default_functional_linear_single_token_per_known_lane_full_k_weight"]["known_lane_match_count"] < 6
        or variant_by_name["default_functional_linear_single_feature_per_known_lane_all_74_tokens"]["known_lane_match_count"] < 6
    )
    thread_sensitive = variant_by_name["default_functional_linear_set_num_threads_1"]["known_lane_match_count"] < 6
    stride_sensitive = (
        variant_by_name["default_functional_linear_cloned_contiguous_input_weight"]["known_lane_match_count"]
        != variant_by_name["default_functional_linear_original_view_stride_input_weight"]["known_lane_match_count"]
    )
    if all(row["rust_cpu_replay_value"] == row["official_module_value"] for row in six_lane_summary):
        classification = "k_projection_arithmetic_mismatch_cleared_or_prior_measurement_bug"
    elif any(item["matches_official_all_known_lanes"] for item in model_table):
        classification = "onednn_bf16_linear_reduction_model_identified"
    elif stride_sensitive:
        classification = "onednn_bf16_linear_layout_stride_dependency_identified"
    elif thread_sensitive and full_matches:
        classification = "onednn_bf16_linear_thread_partition_required"
    elif full_matches and reduced_mismatch:
        classification = "onednn_bf16_linear_full_shape_blocking_required"
    else:
        classification = "onednn_bf16_linear_primitive_identified_model_still_incomplete"
    output["classification"] = classification
    output["next_bounded_step"] = (
        "use full-shape oneDNN BF16 GEMM as the projection oracle for a narrow helper proof, or implement a deliberately scoped oneDNN-backed CPU reference before any runtime helper change"
        if classification in {
            "onednn_bf16_linear_full_shape_blocking_required",
            "onednn_bf16_linear_thread_partition_required",
            "onednn_bf16_linear_reduction_model_identified",
        }
        else "keep the runtime unchanged and add one oneDNN primitive-level reproduction closer to the JIT kernel blocking before selecting a projection fix"
    )
    args.probe_json.write_text(json.dumps(output, indent=2), encoding="utf-8")


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
        "--rmsnorm-replay",
        str(args.rmsnorm_replay),
        "--official-extraction",
        str(args.official_extraction),
        "--output",
        str(args.output),
        "--verbose-log",
        str(args.verbose_log),
        "--probe-json",
        str(args.probe_json),
        "--device",
        args.device,
    ]
    completed = subprocess.run(probe_cmd, env=env, text=True, capture_output=True)
    args.verbose_log.write_text(
        "COMMAND: "
        + " ".join(probe_cmd)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"oneDNN primitive reproduction probe failed; see {args.verbose_log}")
    output = json.loads(args.probe_json.read_text(encoding="utf-8"))
    verbose_text = args.verbose_log.read_text(encoding="utf-8", errors="replace")
    primitive_lines = [
        line
        for line in verbose_text.splitlines()
        if "onednn_verbose" in line or "dnnl_verbose" in line
    ]
    output["onednn_mkldnn_verbose_log_path"] = str(args.verbose_log)
    output["onednn_verbose_primitive_lines_sample"] = primitive_lines[:80]
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
