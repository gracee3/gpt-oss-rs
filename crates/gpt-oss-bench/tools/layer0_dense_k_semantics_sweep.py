#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]

EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS = [
    200006,
    17360,
    200008,
    3575,
    553,
    17554,
    162016,
    11,
    261,
    4410,
    6439,
    2359,
    22203,
    656,
    7788,
    17527,
    558,
    87447,
    100594,
    25,
    220,
    1323,
    19,
    12,
    3218,
    279,
    30377,
    289,
    25,
    14093,
    279,
    2,
    13888,
    18403,
    25,
    8450,
    11,
    49159,
    11,
    1721,
    13,
    21030,
    2804,
    413,
    7360,
    395,
    1753,
    3176,
    13,
    200007,
    200006,
    77944,
    200008,
    2,
    68406,
    279,
    17045,
    59453,
    1151,
    13,
    200007,
    200006,
    1428,
    200008,
    25968,
    483,
    9707,
    1001,
    2195,
    25,
    40617,
    200007,
    200006,
    173781,
]


def find_openai_gpt_oss_root(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "gpt-oss",
        repo_root.parents[1] / "gpt-oss",
    ]
    for candidate in candidates:
        if (candidate / "gpt_oss").is_dir():
            return candidate
    raise FileNotFoundError(f"could not locate sibling gpt-oss checkout from {repo_root}")


OPENAI_GPT_OSS_ROOT = find_openai_gpt_oss_root(REPO_ROOT)
sys.path.insert(0, str(OPENAI_GPT_OSS_ROOT))

from gpt_oss.torch.weights import Checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep layer-0 dense K projection semantics against the official K slice."
    )
    parser.add_argument("--local-residual-input-artifact", type=Path, required=True)
    parser.add_argument("--local-attn-norm-artifact", type=Path, required=True)
    parser.add_argument("--local-qkv-projection-artifact", type=Path, required=True)
    parser.add_argument("--local-k-projection-pre-bias-artifact", type=Path, required=True)
    parser.add_argument("--official-qkv-projection-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_single_case_artifact(path: Path) -> tuple[dict, dict]:
    artifact = load_json(path)
    cases = artifact.get("cases", [])
    if len(cases) != 1:
        raise ValueError(f"{path} must contain exactly one case, found {len(cases)}")
    return artifact, cases[0]


def validate_case(
    artifact: dict,
    case: dict,
    expected_boundary: str,
    expected_case_id: str,
    expected_tokens: list[int],
) -> None:
    boundary = artifact.get("boundary")
    if boundary != expected_boundary:
        raise ValueError(f"expected boundary {expected_boundary!r}, found {boundary!r}")
    if case.get("id") != expected_case_id:
        raise ValueError(f"expected case id {expected_case_id!r}, found {case.get('id')!r}")
    if case.get("input_token_ids") != expected_tokens:
        raise ValueError(f"{artifact.get('boundary')} input token ids do not match the exact smoke case")


def mean_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / max(len(lhs), 1)


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(lhs, rhs))


def compare_vectors(lhs: list[float], rhs: list[float]) -> dict:
    return {
        "max_abs_diff": float(max_abs_diff(lhs, rhs)),
        "mean_abs_diff": float(mean_abs_diff(lhs, rhs)),
        "matched": lhs == rhs,
    }


def is_material_improvement(baseline: dict, candidate: dict, threshold: float = 0.5) -> bool:
    return (
        candidate["mean_abs_diff"] <= baseline["mean_abs_diff"] * threshold
        or candidate["max_abs_diff"] <= baseline["max_abs_diff"] * threshold
    )


def extract_model_root(*artifacts: dict) -> Path:
    model_roots = []
    for artifact in artifacts:
        model = artifact.get("provenance", {}).get("model")
        if not model:
            raise ValueError("missing provenance.model in one of the input artifacts")
        model_roots.append(Path(model))
    first = model_roots[0]
    for model_root in model_roots[1:]:
        if model_root != first:
            raise ValueError(f"input artifacts disagree on model root: {first} vs {model_root}")
    return first


def load_checkpoint(model_root: Path, device: torch.device) -> Checkpoint:
    return Checkpoint(str(model_root), device)


def load_weight_group(checkpoint: Checkpoint, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    weight = checkpoint.get(f"{prefix}.weight")
    bias = checkpoint.get(f"{prefix}.bias")
    return weight, bias


def precision_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    return str(dtype).replace("torch.", "")


def dense_linear(
    input_vector: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    compute_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    rounded_input = input_vector.to(compute_dtype).to(torch.float32)
    rounded_weight = weight.to(compute_dtype).to(torch.float32)
    rounded_bias = bias.to(compute_dtype).to(torch.float32)
    projected = torch.nn.functional.linear(rounded_input, rounded_weight, rounded_bias)
    return projected.to(output_dtype)


def project_k(
    input_vector: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_dim: int,
    kv_dim: int,
    source: str,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if source == "fused_qkv":
        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        fused_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        fused = dense_linear(
            input_vector,
            fused_weight,
            fused_bias,
            compute_dtype=compute_dtype,
            output_dtype=compute_dtype,
        )
        return fused[q_dim : q_dim + kv_dim]
    if source == "standalone_k":
        return dense_linear(
            input_vector,
            k_weight,
            k_bias,
            compute_dtype=compute_dtype,
            output_dtype=compute_dtype,
        )
    raise ValueError(f"unknown K projection source {source!r}")


def variant_rows(
    input_vector: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    q_dim: int,
    kv_dim: int,
    official_k: list[float],
    baseline: dict,
) -> list[dict]:
    variant_specs = [
        ("fused_qkv_f16", "fused_qkv", torch.float16),
        ("standalone_k_f16", "standalone_k", torch.float16),
        ("fused_qkv_bf16", "fused_qkv", torch.bfloat16),
        ("standalone_k_bf16", "standalone_k", torch.bfloat16),
        ("standalone_k_f32_bound", "standalone_k", torch.float32),
    ]
    rows = []
    for variant, source, compute_dtype in variant_specs:
        projected = project_k(
            input_vector=input_vector,
            q_weight=q_weight,
            q_bias=q_bias,
            k_weight=k_weight,
            k_bias=k_bias,
            v_weight=v_weight,
            v_bias=v_bias,
            q_dim=q_dim,
            kv_dim=kv_dim,
            source=source,
            compute_dtype=compute_dtype,
        )
        candidate = projected.float().cpu().tolist()
        metrics = compare_vectors(candidate, official_k)
        rows.append(
            {
                "variant": variant,
                "source": source,
                "compute_precision": precision_label(compute_dtype),
                "reduction_precision": "f32",
                "output_precision": precision_label(compute_dtype),
                "max_abs_diff": metrics["max_abs_diff"],
                "mean_abs_diff": metrics["mean_abs_diff"],
                "improves_mean_by_at_least_half": metrics["mean_abs_diff"]
                <= baseline["mean_abs_diff"] * 0.5,
                "improves_max_by_at_least_half": metrics["max_abs_diff"]
                <= baseline["max_abs_diff"] * 0.5,
                "materially_improves_current_mismatch": is_material_improvement(
                    baseline, metrics
                ),
            }
        )
    rows.sort(key=lambda item: (item["mean_abs_diff"], item["max_abs_diff"], item["variant"]))
    return rows


def main() -> int:
    args = parse_args()

    residual_artifact, residual_case = load_single_case_artifact(args.local_residual_input_artifact)
    norm_artifact, norm_case = load_single_case_artifact(args.local_attn_norm_artifact)
    qkv_artifact, qkv_case = load_single_case_artifact(args.local_qkv_projection_artifact)
    k_pre_bias_artifact, k_pre_bias_case = load_single_case_artifact(
        args.local_k_projection_pre_bias_artifact
    )
    official_qkv_artifact, official_qkv_case = load_single_case_artifact(
        args.official_qkv_projection_artifact
    )

    for artifact, case, boundary in [
        (residual_artifact, residual_case, "layer0_residual_input"),
        (norm_artifact, norm_case, "layer0_attn_norm_output"),
        (qkv_artifact, qkv_case, "layer0_qkv_projection_output"),
        (k_pre_bias_artifact, k_pre_bias_case, "layer0_k_projection_pre_bias_output"),
        (official_qkv_artifact, official_qkv_case, "layer0_qkv_projection_output"),
    ]:
        validate_case(
            artifact,
            case,
            expected_boundary=boundary,
            expected_case_id="developer-message-user-smoke",
            expected_tokens=EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        )

    model_root = extract_model_root(norm_artifact, qkv_artifact, k_pre_bias_artifact)
    checkpoint = load_checkpoint(model_root, torch.device(args.device))

    q_weight, q_bias = load_weight_group(checkpoint, "model.layers.0.self_attn.q_proj")
    k_weight, k_bias = load_weight_group(checkpoint, "model.layers.0.self_attn.k_proj")
    v_weight, v_bias = load_weight_group(checkpoint, "model.layers.0.self_attn.v_proj")

    q_dim = int(qkv_case["q_dim"])
    kv_dim = int(qkv_case["kv_dim"])
    qkv_dim = int(qkv_case["qkv_dim"])

    local_norm_vector = torch.tensor(
        norm_case["final_token_hidden_f32"], dtype=torch.float32, device=torch.device(args.device)
    )
    local_current_k = k_pre_bias_case["final_token_hidden_f32"]
    local_qkv_k = qkv_case["final_token_hidden_f32"][q_dim : q_dim + kv_dim]
    official_k = official_qkv_case["final_token_hidden_f32"][q_dim : q_dim + kv_dim]

    if len(local_qkv_k) != kv_dim or len(official_k) != kv_dim:
        raise ValueError("K slice boundaries do not line up with the official qkv artifact")

    local_qkv_vs_pre_bias = compare_vectors(local_qkv_k, local_current_k)
    baseline_current_k_vs_official = compare_vectors(local_current_k, official_k)
    if local_qkv_vs_pre_bias["max_abs_diff"] != 0.0 or local_qkv_vs_pre_bias["mean_abs_diff"] != 0.0:
        raise ValueError(
            "local qkv K slice does not exactly match the local K pre-bias artifact"
        )

    ranked_variants = variant_rows(
        input_vector=local_norm_vector,
        q_weight=q_weight,
        q_bias=q_bias,
        k_weight=k_weight,
        k_bias=k_bias,
        v_weight=v_weight,
        v_bias=v_bias,
        q_dim=q_dim,
        kv_dim=kv_dim,
        official_k=official_k,
        baseline=baseline_current_k_vs_official,
    )

    best_variant = ranked_variants[0]
    materially_improves_current_mismatch = best_variant["materially_improves_current_mismatch"]

    report = {
        "schema_version": "runtime_forward_layer0_dense_k_semantics_status/v1",
        "provenance": {
            "local_residual_input_artifact_path": str(args.local_residual_input_artifact),
            "local_attn_norm_artifact_path": str(args.local_attn_norm_artifact),
            "local_qkv_projection_artifact_path": str(args.local_qkv_projection_artifact),
            "local_k_projection_pre_bias_artifact_path": str(
                args.local_k_projection_pre_bias_artifact
            ),
            "official_qkv_projection_artifact_path": str(args.official_qkv_projection_artifact),
            "model_root": str(model_root),
            "device": str(torch.device(args.device)),
        },
        "case": {
            "case_id": "developer-message-user-smoke",
            "input_token_ids": EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
            "hidden_size": int(qkv_case["hidden_size"]),
            "q_dim": q_dim,
            "kv_dim": kv_dim,
            "qkv_dim": qkv_dim,
        },
        "baseline_current_k_vs_official": baseline_current_k_vs_official,
        "sanity_checks": {
            "local_qkv_k_slice_vs_local_k_pre_bias": local_qkv_vs_pre_bias,
        },
        "ranked_variants": ranked_variants,
        "best_variant": best_variant,
        "materially_improves_current_mismatch": materially_improves_current_mismatch,
        "conclusion": "bf16_dense_k_semantics_best"
        if materially_improves_current_mismatch
        else "no_material_dense_k_improvement",
        "runtime_forward_layer0_dense_k_status_now": (
            f"Layer-0 K baseline is mean_abs_diff {baseline_current_k_vs_official['mean_abs_diff']:.9f}, "
            f"max_abs_diff {baseline_current_k_vs_official['max_abs_diff']:.6f}; "
            f"best dense-K variant is {best_variant['variant']} with "
            f"mean_abs_diff {best_variant['mean_abs_diff']:.9f}, "
            f"max_abs_diff {best_variant['max_abs_diff']:.6f}."
        ),
        "next_bounded_step": "surgical_dense_projection_semantics_fix",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
