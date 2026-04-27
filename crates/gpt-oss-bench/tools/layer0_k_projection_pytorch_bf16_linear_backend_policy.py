#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
KNOWN_LANES = [(5, 287), (39, 460), (44, 303), (55, 310), (56, 13), (65, 293)]
WORST_TOKEN = 65
WORST_FEATURE = 293


def find_openai_gpt_oss_root(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "gpt-oss",
        repo_root.parents[1] / "gpt-oss",
    ]
    for candidate in candidates:
        if (candidate / "gpt_oss").is_dir():
            return candidate
    raise FileNotFoundError(f"could not locate sibling gpt-oss checkout from {repo_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch CPU BF16 linear backend policy for layer0 K projection."
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


def import_gpt_oss():
    sys.path.insert(0, str(find_openai_gpt_oss_root(REPO_ROOT)))
    import torch
    from gpt_oss.torch.model import AttentionBlock, ModelConfig
    from gpt_oss.torch.weights import Checkpoint

    return torch, AttentionBlock, ModelConfig, Checkpoint


def resolve_oracle_checkpoint_dir(path: Path) -> Path:
    original_dir = path / "original"
    if original_dir.is_dir():
        return original_dir
    return path


def load_restricted_config(path: Path, ModelConfig):
    config_path = path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        json_config = json.load(handle)
    aliases = {
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "experts_per_token",
    }
    accepted = set(inspect.signature(ModelConfig).parameters)
    filtered = {}
    for key, value in json_config.items():
        mapped = aliases.get(key, key)
        if mapped in accepted:
            filtered[mapped] = value
    config = ModelConfig(**filtered)
    config.sliding_window = 0
    return config


def load_layer0_model(model_root: Path, device, torch, AttentionBlock, ModelConfig, Checkpoint):
    class Layer0Replay(torch.nn.Module):
        def __init__(self, config, device):
            super().__init__()
            self.embedding = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
            )
            self.attn = AttentionBlock(config=config, layer_idx=0, device=device)

    config = load_restricted_config(model_root, ModelConfig)
    model = Layer0Replay(config=config, device=device)
    model.eval()
    checkpoint = Checkpoint(str(resolve_oracle_checkpoint_dir(model_root)), device)
    for name, param in dict(model.named_parameters()).items():
        if name == "embedding.weight":
            param.data.copy_(checkpoint.get("embedding.weight"))
        else:
            param.data.copy_(checkpoint.get(f"block.0.{name}"))
    return model


def load_inputs(args, torch):
    source = json.loads(args.source.read_text(encoding="utf-8"))
    rmsnorm = json.loads(args.rmsnorm_replay.read_text(encoding="utf-8"))
    extraction = json.loads(args.official_extraction.read_text(encoding="utf-8"))
    if source["exact_case"]["case_id"] != "developer-message-user-smoke":
        raise ValueError("source artifact is not the exact smoke case")
    if source["classification"] != "official_k_projection_policy_not_fully_modeled":
        raise ValueError("source artifact is not the expected prior classification")
    token_count = int(source["exact_case"]["token_count"])
    hidden_size = int(source["exact_case"]["hidden_size"])
    kv_dim = int(source["exact_case"]["kv_dim"])
    policy_output = rmsnorm["policy_outputs_f32"][
        "manual_bf16_input_bf16_weight_f32_reduction_bf16_output"
    ]
    norm = torch.tensor(policy_output, dtype=torch.float32).reshape(token_count, hidden_size)
    k_weight = torch.tensor(
        extraction["official_k_weight_f32"], dtype=torch.float32
    ).reshape(kv_dim, hidden_size)
    official_module = torch.tensor(
        extraction["official_projection_outputs"]["official_module_k_output_f32"],
        dtype=torch.float32,
    ).reshape(token_count, kv_dim)
    return source, rmsnorm, extraction, norm, k_weight, official_module


def compare(lhs, rhs):
    import torch

    diff = (lhs.to(dtype=rhs.dtype) - rhs).abs().reshape(-1)
    mismatch = int((diff > 0).sum().item())
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs.to(dtype=rhs.dtype), rhs)),
        "mismatch_count": mismatch,
    }


def bf16_bits_from_float(value: float) -> int:
    [raw] = struct.unpack("<I", struct.pack("<f", float(value)))
    lsb = (raw >> 16) & 1
    rounded = raw + (0x7FFF + lsb)
    return (rounded >> 16) & 0xFFFF


def bf16_value_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def compare_path(name, output, official, known_lanes):
    import torch

    metrics = compare(output, official)
    known_matches = 0
    known_values = []
    for token, feature in known_lanes:
        value = float(output[token, feature].to(torch.float32).item())
        official_value = float(official[token, feature].item())
        if value == official_value:
            known_matches += 1
        known_values.append(
            {
                "token_index": token,
                "k_feature_index": feature,
                "value": value,
                "official_value": official_value,
                "abs_diff": abs(value - official_value),
            }
        )
    return {
        "name": name,
        **metrics,
        "known_lane_match_count": known_matches,
        "known_lane_count": len(known_lanes),
        "worst_lane_value_token65_feature293": float(
            output[WORST_TOKEN, WORST_FEATURE].to(torch.float32).item()
        ),
        "known_lane_values": known_values,
    }


def f32_left(values):
    import numpy as np

    total = np.float32(0.0)
    for value in values:
        total = np.float32(total + np.float32(value))
    return float(total)


def f32_pairwise(values):
    import numpy as np

    work = [np.float32(v) for v in values]
    while len(work) > 1:
        nxt = []
        for index in range(0, len(work), 2):
            if index + 1 < len(work):
                nxt.append(np.float32(work[index] + work[index + 1]))
            else:
                nxt.append(work[index])
        work = nxt
    return float(work[0]) if work else 0.0


def f32_chunked(values, chunk):
    return f32_left([f32_left(values[i : i + chunk]) for i in range(0, len(values), chunk)])


def f32_simd_lane(values, lanes):
    import numpy as np

    partials = [np.float32(0.0) for _ in range(lanes)]
    for index, value in enumerate(values):
        partials[index % lanes] = np.float32(partials[index % lanes] + np.float32(value))
    return f32_pairwise(partials)


def model_outputs_for_known_lanes(name, fn, norm_bf16, k_weight_bf16, official, rust_known):
    import torch

    rows = []
    outputs = []
    pre_outputs = []
    for token, feature in KNOWN_LANES:
        x = norm_bf16[token].to(torch.float32).tolist()
        w = k_weight_bf16[feature].to(torch.float32).tolist()
        products = [float(a * b) for a, b in zip(x, w)]
        pre = fn(products)
        bits = bf16_bits_from_float(pre)
        value = bf16_value_from_bits(bits)
        official_value = float(official[token, feature].item())
        rust_value = float(rust_known[(token, feature)])
        rows.append(
            {
                "token_index": token,
                "k_feature_index": feature,
                "pre_output_f32": pre,
                "bf16_output_bits": f"0x{bits:04x}",
                "bf16_output_value": value,
                "official_value": official_value,
                "rust_helper_value": rust_value,
                "abs_diff_vs_official": abs(value - official_value),
                "matches_official": value == official_value,
                "matches_rust_helper": value == rust_value,
            }
        )
        outputs.append(value)
        pre_outputs.append(pre)
    max_diff = max(row["abs_diff_vs_official"] for row in rows)
    mean_diff = sum(row["abs_diff_vs_official"] for row in rows) / len(rows)
    return {
        "name": name,
        "max_abs_diff_vs_official": max_diff,
        "mean_abs_diff_vs_official": mean_diff,
        "lanes_matched_official": sum(1 for row in rows if row["matches_official"]),
        "matches_official_all_known_lanes": all(row["matches_official"] for row in rows),
        "lanes_matched_rust_helper": sum(1 for row in rows if row["matches_rust_helper"]),
        "matches_rust_helper_all_known_lanes": all(row["matches_rust_helper"] for row in rows),
        "value_token65_feature293": rows[-1]["bf16_output_value"],
        "pre_output_token65_feature293": rows[-1]["pre_output_f32"],
        "bf16_bits_token65_feature293": rows[-1]["bf16_output_bits"],
        "lane_rows": rows,
    }


def run_probe(args):
    torch, AttentionBlock, ModelConfig, Checkpoint = import_gpt_oss()
    device = torch.device(args.device)
    source, _, extraction, norm, _, official = load_inputs(args, torch)
    model = load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    q_dim = model.attn.num_attention_heads * model.attn.head_dim
    kv_dim = model.attn.num_key_value_heads * model.attn.head_dim
    k_start = q_dim
    k_end = q_dim + kv_dim
    norm_bf16 = norm.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        module = model.attn.qkv(norm_bf16)[:, k_start:k_end].contiguous()
        linear = torch.nn.functional.linear(
            norm_bf16,
            model.attn.qkv.weight[k_start:k_end, :].contiguous(),
            model.attn.qkv.bias[k_start:k_end].contiguous(),
        )
    metadata = {
        "torch_version": torch.__version__,
        "torch_config": torch.__config__.show(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_is_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "matmul_tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32_allowed": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "input_shape": list(norm_bf16.shape),
        "weight_shape": list(model.attn.qkv.weight[k_start:k_end, :].shape),
        "output_shape": list(module.shape),
        "input_dtype": str(norm_bf16.dtype).replace("torch.", ""),
        "weight_dtype": str(model.attn.qkv.weight.dtype).replace("torch.", ""),
        "output_dtype": str(module.dtype).replace("torch.", ""),
        "input_stride": list(norm_bf16.stride()),
        "weight_stride": list(model.attn.qkv.weight[k_start:k_end, :].stride()),
        "output_stride": list(module.stride()),
        "module_vs_functional_linear": compare(module.to(torch.float32), linear.to(torch.float32)),
        "source_classification": source["classification"],
        "official_extraction_manual_vs_module": extraction["official_manual_projection_vs_module"]["metrics"],
    }
    args.probe_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
        "COMMAND: " + " ".join(probe_cmd) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"backend probe failed; see {args.verbose_log}")

    torch, AttentionBlock, ModelConfig, Checkpoint = import_gpt_oss()
    device = torch.device(args.device)
    source, rmsnorm, extraction, norm, k_weight, official = load_inputs(args, torch)
    metadata = json.loads(args.probe_json.read_text(encoding="utf-8"))
    model = load_layer0_model(args.model_root, device, torch, AttentionBlock, ModelConfig, Checkpoint)
    q_dim = model.attn.num_attention_heads * model.attn.head_dim
    kv_dim = model.attn.num_key_value_heads * model.attn.head_dim
    k_start = q_dim
    k_end = q_dim + kv_dim
    norm_bf16 = norm.to(device=device, dtype=torch.bfloat16)
    weight_bf16 = k_weight.to(device=device, dtype=torch.bfloat16)
    official_module = official.to(torch.float32)

    original_threads = torch.get_num_threads()
    toggles = []
    with torch.inference_mode():
        default_module = model.attn.qkv(norm_bf16)[:, k_start:k_end].contiguous().to(torch.float32)
        default_linear = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous().to(torch.float32)
        toggles.append(compare_path("default_official_module_k_output", default_module, official_module, KNOWN_LANES))
        toggles.append(compare_path("default_functional_linear_cpu_bf16", default_linear, official_module, KNOWN_LANES))
        for enabled in [True, False]:
            if torch.backends.mkldnn.is_available():
                previous = torch.backends.mkldnn.enabled
                torch.backends.mkldnn.enabled = enabled
                try:
                    output = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous().to(torch.float32)
                    item = compare_path(f"functional_linear_mkldnn_enabled_{enabled}", output, official_module, KNOWN_LANES)
                    item["available"] = True
                except Exception as exc:
                    item = {"name": f"functional_linear_mkldnn_enabled_{enabled}", "available": False, "error": repr(exc)}
                finally:
                    torch.backends.mkldnn.enabled = previous
                toggles.append(item)
        torch.set_num_threads(1)
        one_thread = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous().to(torch.float32)
        toggles.append(compare_path("functional_linear_set_num_threads_1", one_thread, official_module, KNOWN_LANES))
        torch.set_num_threads(original_threads)
        restored = torch.nn.functional.linear(norm_bf16, weight_bf16, None).contiguous().to(torch.float32)
        toggles.append(compare_path("functional_linear_original_thread_count_restored", restored, official_module, KNOWN_LANES))

    source_lanes = source["known_mismatch_lane_table"]
    rust_known = {
        (int(row["token_index"]), int(row["k_feature_index"])): float(row["rust_cpu_replay_value"])
        for row in source_lanes
    }
    models = [
        ("current_rust_left_to_right_replay", f32_left),
        ("previous_pairwise_tree_replay", f32_pairwise),
        ("previous_chunked_16_replay", lambda values: f32_chunked(values, 16)),
        ("previous_chunked_32_replay", lambda values: f32_chunked(values, 32)),
        ("backend_informed_chunked_32_replay", lambda values: f32_chunked(values, 32)),
    ]
    lower_config = metadata.get("torch_config", "").lower()
    if any(key in lower_config for key in ["avx512", "amx", "bf16"]):
        models.append(("simd_lane_16_partial_sum_pairwise_replay", lambda values: f32_simd_lane(values, 16)))
        models.append(("simd_lane_32_partial_sum_pairwise_replay", lambda values: f32_simd_lane(values, 32)))
    model_table = [
        model_outputs_for_known_lanes(name, fn, norm_bf16.cpu(), weight_bf16.cpu(), official_module.cpu(), rust_known)
        for name, fn in models
    ]
    best_model = max(
        model_table,
        key=lambda item: (
            item["lanes_matched_official"],
            -item["max_abs_diff_vs_official"],
            -item["mean_abs_diff_vs_official"],
        ),
    )
    official_bits = bf16_bits_from_float(float(official_module[WORST_TOKEN, WORST_FEATURE].item()))
    rust_value = rust_known[(WORST_TOKEN, WORST_FEATURE)]
    rust_bits = bf16_bits_from_float(rust_value)
    lower_bits = min(official_bits, rust_bits)
    upper_bits = max(official_bits, rust_bits)
    midpoint = (bf16_value_from_bits(lower_bits) + bf16_value_from_bits(upper_bits)) / 2.0
    rust_model = next(item for item in model_table if item["name"] == "current_rust_left_to_right_replay")
    best_pre = float(best_model["pre_output_token65_feature293"])
    rust_pre = float(rust_model["pre_output_token65_feature293"])
    verbose_text = args.verbose_log.read_text(encoding="utf-8", errors="replace")
    verbose_present = "onednn_verbose" in verbose_text or "dnnl_verbose" in verbose_text
    primitive_lines = [
        line for line in verbose_text.splitlines()
        if "onednn_verbose" in line or "dnnl_verbose" in line
    ][:40]

    toggles_toward_rust = [
        item for item in toggles
        if item.get("available", True)
        and item.get("worst_lane_value_token65_feature293") == rust_value
    ]
    any_toggle_changes = any(
        item.get("available", True)
        and item.get("max_abs_diff", 0.0) > 0.0
        for item in toggles
    )
    full_model_match = any(item["matches_official_all_known_lanes"] for item in model_table)
    if any(item.get("name") == "functional_linear_mkldnn_enabled_False" and item.get("known_lane_match_count") == 0 for item in toggles_toward_rust):
        classification = "pytorch_bf16_linear_onednn_backend_policy_identified"
    elif full_model_match:
        classification = "pytorch_bf16_linear_reduction_model_identified"
    elif any_toggle_changes:
        classification = "pytorch_bf16_linear_backend_or_threading_sensitive"
    elif verbose_present or metadata.get("mkldnn_is_available"):
        classification = "pytorch_bf16_linear_backend_identified_but_scalar_model_incomplete"
    else:
        classification = "pytorch_bf16_linear_backend_policy_still_unmodeled"

    if all(row["rust_cpu_replay_value"] == row["official_module_value"] for row in source_lanes):
        classification = "k_projection_arithmetic_mismatch_cleared_or_prior_measurement_bug"

    next_step = (
        "use the identified PyTorch CPU BF16 backend policy to choose one exact projection arithmetic proof/fix candidate"
        if classification in {
            "pytorch_bf16_linear_onednn_backend_policy_identified",
            "pytorch_bf16_linear_reduction_model_identified",
        }
        else "request or build a narrower oneDNN/PyTorch primitive-level reproduction for the remaining known-lane scalar mismatch before changing runtime"
    )
    output = {
        "schema_version": "runtime_forward_layer0_k_projection_pytorch_bf16_linear_backend_policy_status/v1",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "token_count": int(source["exact_case"]["token_count"]),
            "hidden_size": int(source["exact_case"]["hidden_size"]),
            "kv_dim": int(source["exact_case"]["kv_dim"]),
        },
        "mode": "k-projection-pytorch-bf16-linear-backend-policy-status",
        "source_artifact_path": str(args.source),
        "python_script_path_used_for_backend_capture": str(Path(__file__).resolve()),
        "pytorch_backend_metadata": {
            **metadata,
            "onednn_or_dnnl_verbose_output_present": verbose_present,
            "verbose_primitive_lines_sample": primitive_lines,
        },
        "onednn_mkldnn_verbose_log_path": str(args.verbose_log),
        "backend_toggle_comparison_table": toggles,
        "backend_informed_reduction_model_table": model_table,
        "known_6_lane_comparison_summary": source_lanes,
        "best_reduction_model_summary": {
            "name": best_model["name"],
            "lanes_matched_official": best_model["lanes_matched_official"],
            "max_abs_diff_vs_official": best_model["max_abs_diff_vs_official"],
            "mean_abs_diff_vs_official": best_model["mean_abs_diff_vs_official"],
            "value_token65_feature293": best_model["value_token65_feature293"],
        },
        "worst_lane_boundary_analysis": {
            "token_index": WORST_TOKEN,
            "k_feature_index": WORST_FEATURE,
            "official_module_value": float(official_module[WORST_TOKEN, WORST_FEATURE].item()),
            "official_module_bf16_bits": f"0x{official_bits:04x}",
            "default_functional_linear_value": float(default_linear[WORST_TOKEN, WORST_FEATURE].item()),
            "default_functional_linear_bf16_bits": f"0x{bf16_bits_from_float(float(default_linear[WORST_TOKEN, WORST_FEATURE].item())):04x}",
            "rust_helper_value": rust_value,
            "rust_helper_bf16_bits": f"0x{rust_bits:04x}",
            "best_backend_informed_model_name": best_model["name"],
            "best_backend_informed_model_value": best_model["value_token65_feature293"],
            "best_backend_informed_model_bf16_bits": best_model["bf16_bits_token65_feature293"],
            "rust_pre_output_f32_dot": rust_pre,
            "best_model_pre_output_f32_dot": best_pre,
            "bf16_rounding_midpoint_between_official_and_rust_bins": midpoint,
            "rust_pre_output_distance_from_midpoint": rust_pre - midpoint,
            "best_model_pre_output_distance_from_midpoint": best_pre - midpoint,
            "remaining_mismatch_explained_by_accumulation_crossing_bf16_midpoint": (rust_pre - midpoint) * (best_pre - midpoint) <= 0.0,
        },
        "earliest_divergent_source": "backend BF16 linear accumulation/blocking policy",
        "classification": classification,
        "next_bounded_step": next_step,
    }
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
