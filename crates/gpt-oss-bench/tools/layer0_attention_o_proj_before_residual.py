#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

import layer0_k_projection_pytorch_bf16_linear_backend_policy as base
import layer0_q_pre_post_rope_runtime_localization as qloc
import layer0_v_projection_weight_bias_arithmetic_policy as vpol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer0 final-token attention o_proj output before residual status."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-v-candidate", type=Path, required=True)
    parser.add_argument("--attention-probs-proof", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--official-o-proj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-log", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_vector(lhs, rhs):
    lhs = lhs.reshape(-1).to(torch.float32)
    rhs = rhs.reshape(-1).to(torch.float32)
    diff = (lhs - rhs).abs()
    mismatch = diff > 0
    first = None
    worst = None
    if bool(mismatch.any().item()):
        first_idx = int(mismatch.nonzero(as_tuple=False)[0].item())
        worst_idx = int(diff.argmax().item())
        first = {
            "hidden_lane": first_idx,
            "lhs_value": float(lhs[first_idx].item()),
            "rhs_value": float(rhs[first_idx].item()),
            "abs_diff": float(diff[first_idx].item()),
        }
        worst = {
            "hidden_lane": worst_idx,
            "lhs_value": float(lhs[worst_idx].item()),
            "rhs_value": float(rhs[worst_idx].item()),
            "abs_diff": float(diff[worst_idx].item()),
        }
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
        "mismatching_lane_count": int(mismatch.sum().item()),
        "first_differing_hidden_lane": first,
        "worst_differing_hidden_lane": worst,
    }


def digest(tensor):
    return qloc.digest_tensor(tensor, torch)


def tensor_meta(tensor, layout, serialization_dtype=None):
    meta = qloc.tensor_meta(tensor, layout, serialization_dtype)
    meta["device"] = str(tensor.device)
    return meta


def vector_summary(tensor):
    flat = tensor.reshape(-1).to(torch.float32)
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "digest": digest(tensor),
    }


def run_probe(args):
    torch_mod, AttentionBlock, ModelConfig, Checkpoint = base.import_gpt_oss()
    global torch
    torch = torch_mod
    device = torch.device(args.device)

    v_candidate_status = load_json(args.source_v_candidate)
    probs_proof = load_json(args.attention_probs_proof)
    official_o_proj_artifact = load_json(args.official_o_proj)
    if v_candidate_status.get("classification") not in {
        "onednn_v_candidate_confirms_runtime_v_projection_policy_delta",
        "onednn_v_candidate_clears_weighted_v_sum_pre_o_proj",
    }:
        raise ValueError("source V candidate artifact is not a cleared scoped candidate state")
    if not v_candidate_status["weighted_v_sum_before_o_proj_metrics"]["candidate_v_weighted_sum_vs_official"]["matched"]:
        raise ValueError("candidate weighted V sum guard regressed")
    if probs_proof.get("classification") != "attention_probs_cleared_after_post_mask_candidates":
        raise ValueError("attention probability proof is not cleared")
    if official_o_proj_artifact.get("classification") != "official_attention_o_proj_before_residual_boundary_available":
        raise ValueError("official o_proj PPP artifact is not the expected boundary")
    if official_o_proj_artifact.get("boundary") != "layer0_final_token_attention_output_after_o_proj_before_residual":
        raise ValueError("official o_proj PPP boundary name is not usable")

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
    v_start = q_dim + kv_dim
    v_end = v_start + kv_dim
    v_weight_bf16 = attn.qkv.weight[v_start:v_end, :].detach().to(torch.bfloat16).contiguous()
    v_bias_bf16 = attn.qkv.bias[v_start:v_end].detach().to(torch.bfloat16).contiguous()

    original_mkldnn = bool(torch.backends.mkldnn.enabled)
    original_threads = int(torch.get_num_threads())
    torch.backends.mkldnn.enabled = True
    with torch.inference_mode():
        candidate_v = torch.nn.functional.linear(norm_bf16, v_weight_bf16, v_bias_bf16).contiguous()
    torch.backends.mkldnn.enabled = original_mkldnn
    candidate_v_grouped = candidate_v.reshape(74, 8, 64).to(torch.float32)

    official_probs = torch.tensor(
        load_json(Path(probs_proof["official_post_softmax_reference_path"]))["values"],
        dtype=torch.float32,
        device=device,
    ).reshape(64, 75)
    candidate_weighted = vpol.weighted_sum(official_probs, candidate_v_grouped, torch)
    official_weighted_path = Path(v_candidate_status["source_artifact_paths"]["official_weighted_v_sum"])
    official_weighted = torch.tensor(load_json(official_weighted_path)["values"], dtype=torch.float32, device=device)
    weighted_guard = vpol.compare_head_lane(candidate_weighted, official_weighted, torch)

    o_proj_input = candidate_weighted.reshape(1, -1).to(torch.bfloat16).contiguous()
    official_output = torch.tensor(official_o_proj_artifact["values"], dtype=torch.float32, device=device).reshape(2880)
    with torch.inference_mode():
        local_o_proj = attn.out(o_proj_input).reshape(2880).contiguous()
    local_o_proj_f32 = local_o_proj.to(torch.float32)
    local_metrics = compare_vector(local_o_proj_f32, official_output)

    o_proj_weight = attn.out.weight.detach().to(torch.bfloat16).contiguous()
    o_proj_bias = getattr(attn.out, "bias", None)
    if o_proj_bias is not None:
        o_proj_bias_bf16 = o_proj_bias.detach().to(torch.bfloat16).contiguous()
        bias_summary = vector_summary(o_proj_bias_bf16)
        bias_present = True
        bias_all_zero = bool(torch.all(o_proj_bias_bf16.to(torch.float32) == 0).item())
    else:
        o_proj_bias_bf16 = None
        bias_summary = None
        bias_present = False
        bias_all_zero = None

    onednn_candidate_metrics = None
    onednn_candidate_output = None
    if not local_metrics["matched"]:
        torch.backends.mkldnn.enabled = True
        with torch.inference_mode():
            onednn_candidate_output = torch.nn.functional.linear(
                o_proj_input,
                o_proj_weight,
                o_proj_bias_bf16,
            ).reshape(2880).contiguous()
        torch.backends.mkldnn.enabled = original_mkldnn
        onednn_candidate_metrics = compare_vector(onednn_candidate_output.to(torch.float32), official_output)

    ppp_digest = (
        official_o_proj_artifact.get("finite_value_summary", {}).get("sha256_f32_le")
        or official_o_proj_artifact.get("digest")
    )
    weight_bias_expected = (
        list(o_proj_weight.shape) == [2880, 4096]
        and (not bias_present or list(o_proj_bias_bf16.shape) == [2880])
    )
    if not weighted_guard["matched"]:
        classification = "weighted_v_sum_guard_regressed_before_o_proj"
        next_step = "re-establish candidate weighted V sum before o_proj"
    elif not weight_bias_expected:
        classification = "layer0_attention_o_proj_weight_or_bias_mismatch"
        next_step = "inspect layer0 attention o_proj weight/bias metadata only"
    elif local_metrics["matched"]:
        classification = "layer0_attention_o_proj_before_residual_cleared"
        next_step = "continue to residual add only after a bounded PPP/reference boundary exists"
    elif onednn_candidate_metrics is not None and onednn_candidate_metrics["matched"]:
        classification = "layer0_attention_o_proj_onednn_policy_candidate_clears_boundary"
        next_step = "decide whether to prove a scoped o_proj policy candidate before residual add"
    else:
        classification = "layer0_attention_o_proj_arithmetic_policy_mismatch"
        next_step = "inspect layer0 attention o_proj arithmetic policy only"

    output = {
        "schema_version": "runtime_forward_layer0_attention_o_proj_before_residual_status/v1",
        "mode": "layer0-attention-o-proj-before-residual-status",
        "exact_case": {
            "case_id": "developer-message-user-smoke",
            "layer_index": 0,
            "final_token_index": 73,
            "hidden_size": 2880,
        },
        "source_ppp_boundary_artifact_path": str(args.official_o_proj),
        "ppp_boundary_digest": ppp_digest,
        "before_residual_add_confirmed": True,
        "residual_add_included": False,
        "source_artifact_paths": {
            "v_candidate_proof": str(args.source_v_candidate),
            "attention_probabilities_proof": str(args.attention_probs_proof),
            "authoritative_rmsnorm_replay": str(args.rmsnorm_replay),
            "official_weighted_v_sum": str(official_weighted_path),
            "official_o_proj_before_residual": str(args.official_o_proj),
        },
        "q_k_v_candidate_provenance_guard_summary": {
            "q_post_rope": probs_proof["q_post_rope_guard_metrics"],
            "k_post_rope": probs_proof["k_post_rope_guard_metrics"],
            "post_softmax_probabilities": {
                "real_key_probabilities": probs_proof["real_key_probability_metrics"],
                "sink_probabilities": probs_proof["sink_probability_metrics"],
                "all_positions": probs_proof["all_position_probability_metrics"],
            },
            "v_candidate_vs_official": v_candidate_status["v_candidate_vs_official_metrics"],
            "v_candidate_weighted_sum_vs_official": weighted_guard,
            "q_k_v_candidates_are_bench_proof_only": True,
        },
        "weighted_v_sum_guard_metric": weighted_guard,
        "o_proj_input": {
            **tensor_meta(o_proj_input, "[1, q_dim] final-token weighted V sum before o_proj", "bf16"),
            "source": "candidate weighted V sum reconstructed from oneDNN V candidate and cleared probabilities",
        },
        "o_proj_weight_bias_metadata": {
            "weight": tensor_meta(o_proj_weight, "[hidden_size, q_dim]", "bf16"),
            "weight_digest": digest(o_proj_weight),
            "orientation": "activation[1, q_dim] x o_proj_weight^T",
            "bias_present": bias_present,
            "bias": tensor_meta(o_proj_bias_bf16, "[hidden_size]", "bf16") if bias_present else None,
            "bias_summary": bias_summary,
            "bias_all_zero": bias_all_zero,
            "metadata_matches_expected": weight_bias_expected,
            "finding": "o_proj weight/bias metadata matches expected [2880,4096] weight and [2880] bias"
            if weight_bias_expected
            else "o_proj weight/bias metadata differs from expected layer0 attention output projection",
        },
        "final_token_o_proj_output": {
            **tensor_meta(local_o_proj, "[hidden_size]", "bf16"),
            "boundary": "layer0_final_token_attention_output_after_o_proj_before_residual",
            "before_residual_add": True,
        },
        "official_tensor_metadata": {
            "shape": official_o_proj_artifact.get("shape"),
            "tensor_dtype": official_o_proj_artifact.get("tensor_dtype"),
            "serialization_dtype": official_o_proj_artifact.get("serialization_dtype"),
            "layout_interpretation": official_o_proj_artifact.get("layout_interpretation"),
            "confirmed_after_o_proj": official_o_proj_artifact.get("computation_summary", {}).get("o_proj_output_dtype") is not None,
            "confirmed_before_residual": True,
        },
        "o_proj_output_vs_official_metrics": local_metrics,
        "optional_onednn_o_proj_candidate_metrics": onednn_candidate_metrics,
        "first_differing_hidden_lane": local_metrics["first_differing_hidden_lane"],
        "worst_differing_hidden_lane": local_metrics["worst_differing_hidden_lane"],
        "classification": classification,
        "next_bounded_step": next_step,
        "python_script_path": str(Path(__file__)),
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
        "--source-v-candidate",
        str(args.source_v_candidate),
        "--attention-probs-proof",
        str(args.attention_probs_proof),
        "--rmsnorm-replay",
        str(args.rmsnorm_replay),
        "--official-o-proj",
        str(args.official_o_proj),
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
        raise RuntimeError(f"attention o_proj before residual status failed; see {args.verbose_log}")
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
