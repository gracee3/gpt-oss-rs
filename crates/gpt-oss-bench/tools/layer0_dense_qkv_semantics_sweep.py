#!/usr/bin/env python3
import argparse
import inspect
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

from gpt_oss.torch.model import AttentionBlock, ModelConfig  # noqa: E402
from gpt_oss.torch.weights import Checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep layer-0 dense QKV semantics and downstream post-attention residual against the official artifacts."
    )
    parser.add_argument(
        "--mode",
        choices=["post-attention-bf16", "live-vs-shadow", "internal-live-vs-shadow"],
        default="post-attention-bf16",
    )
    parser.add_argument(
        "--local-residual-input-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-first-block-20260423/developer-message.runner-layer0-residual-input.json"
        ),
    )
    parser.add_argument(
        "--local-attn-norm-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-layer0-attn-bisect-20260423/developer-message.runner-layer0-attn-norm-output.json"
        ),
    )
    parser.add_argument(
        "--local-qkv-projection-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-layer0-attn-bisect-20260423/developer-message.runner-layer0-qkv-projection-output.json"
        ),
    )
    parser.add_argument(
        "--local-post-attention-residual-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-first-block-20260423/developer-message.runner-layer0-post-attention-residual.json"
        ),
    )
    parser.add_argument(
        "--official-qkv-projection-artifact",
        type=Path,
        default=Path(
            "/tmp/pinned-prompt-parity-official-reference-20260423/developer-message.official-layer0-qkv-projection-output.cpu.json"
        ),
    )
    parser.add_argument(
        "--official-post-attention-residual-artifact",
        type=Path,
        default=Path(
            "/tmp/pinned-prompt-parity-official-reference-20260423/developer-message.official-layer0-post-attention-residual.cpu.json"
        ),
    )
    parser.add_argument(
        "--live-candidate-qkv-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/developer-message.runner-layer0-qkv-projection-output.json"
        ),
    )
    parser.add_argument(
        "--live-candidate-post-attention-residual-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/developer-message.runner-layer0-post-attention-residual.json"
        ),
    )
    parser.add_argument(
        "--live-candidate-internal-trace-artifact",
        type=Path,
        default=Path(
            ".live/runtime-forward-layer0-qkv-bf16-internal-20260423/developer-message.runner-layer0-qkv-bf16-internal-trace.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
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
    *,
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


def qkv_slices(vector: list[float], q_dim: int, kv_dim: int) -> dict[str, list[float]]:
    q_end = q_dim
    k_end = q_end + kv_dim
    return {
        "Q": vector[:q_end],
        "K": vector[q_end:k_end],
        "V": vector[k_end:k_end + kv_dim],
    }


def compare_qkv_slices(lhs: list[float], rhs: list[float], q_dim: int, kv_dim: int) -> dict:
    lhs_slices = qkv_slices(lhs, q_dim=q_dim, kv_dim=kv_dim)
    rhs_slices = qkv_slices(rhs, q_dim=q_dim, kv_dim=kv_dim)
    return {
        name: compare_vectors(lhs_slices[name], rhs_slices[name])
        for name in ("Q", "K", "V")
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


def resolve_oracle_checkpoint_dir(path: Path) -> Path:
    original_dir = path / "original"
    if original_dir.is_dir():
        return original_dir
    return path


def load_restricted_config(path: Path) -> ModelConfig:
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


class Layer0Replay(torch.nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.attn = AttentionBlock(config=config, layer_idx=0, device=device)


def load_layer0_replay_model(
    restricted_config_path: Path, checkpoint_path: Path, device: torch.device
) -> Layer0Replay:
    config = load_restricted_config(restricted_config_path)
    model = Layer0Replay(config=config, device=device)
    model.eval()

    checkpoint = Checkpoint(str(resolve_oracle_checkpoint_dir(checkpoint_path)), device)
    named_parameters = dict(model.named_parameters())
    for name, param in named_parameters.items():
        if name == "embedding.weight":
            param.data.copy_(checkpoint.get("embedding.weight"))
        else:
            param.data.copy_(checkpoint.get(f"block.0.{name}"))

    return model


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


def project_qkv(
    input_vector: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    return dense_linear(
        input_vector,
        qkv_weight,
        qkv_bias,
        compute_dtype=compute_dtype,
        output_dtype=compute_dtype,
    )


def project_qkv_pre_bias(
    input_vector: torch.Tensor,
    qkv_weight: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    rounded_input = input_vector.to(compute_dtype).to(torch.float32)
    rounded_weight = qkv_weight.to(compute_dtype).to(torch.float32)
    projected = torch.nn.functional.linear(rounded_input, rounded_weight, bias=None)
    return projected.to(compute_dtype)


def add_bias_with_rounding(
    projected: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if bias is None:
        return projected.to(output_dtype)
    rounded_bias = bias.to(output_dtype)
    post_bias = projected.to(torch.float32) + rounded_bias.to(torch.float32)
    return post_bias.to(output_dtype)


def build_bf16_shadow_internal_trace(
    input_vector: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None,
) -> dict:
    pre_bias = project_qkv_pre_bias(
        input_vector=input_vector,
        qkv_weight=qkv_weight,
        compute_dtype=torch.bfloat16,
    )
    post_bias = add_bias_with_rounding(pre_bias, qkv_bias, torch.bfloat16)
    packed = post_bias.to(torch.float16)
    return {
        "fused_qkv_pre_bias": pre_bias.float().cpu().tolist(),
        "fused_qkv_post_bias": post_bias.float().cpu().tolist(),
        "packed_qkv_output": packed.float().cpu().tolist(),
    }


def last_token(tensor: torch.Tensor) -> list[float]:
    return tensor[-1].float().cpu().tolist()


def flatten_last_token(tensor: torch.Tensor) -> list[float]:
    return tensor[-1].reshape(-1).float().cpu().tolist()


def replay_layer0_attention_from_qkv(
    attn: AttentionBlock,
    residual_input: torch.Tensor,
    qkv: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    q = qkv[:, : attn.num_attention_heads * attn.head_dim].contiguous()
    k = qkv[
        :,
        attn.num_attention_heads
        * attn.head_dim : (attn.num_attention_heads + attn.num_key_value_heads)
        * attn.head_dim,
    ].contiguous()
    v = qkv[
        :,
        (attn.num_attention_heads + attn.num_key_value_heads)
        * attn.head_dim : (attn.num_attention_heads + 2 * attn.num_key_value_heads)
        * attn.head_dim,
    ].contiguous()

    q_heads = q.view(
        -1,
        attn.num_key_value_heads,
        attn.num_attention_heads // attn.num_key_value_heads,
        attn.head_dim,
    )
    k_heads = k.view(-1, attn.num_key_value_heads, attn.head_dim)
    v_heads = v.view(-1, attn.num_key_value_heads, attn.head_dim)
    q_rope, k_rope = attn.rope(q_heads, k_heads)

    n_tokens = q_rope.shape[0]
    q_mult = attn.num_attention_heads // attn.num_key_value_heads
    K = k_rope[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = v_heads[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = attn.sinks.reshape(attn.num_key_value_heads, q_mult, 1, 1).expand(
        -1, -1, n_tokens, -1
    )
    mask = torch.triu(q_rope.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    qk = torch.einsum("qhmd,khmd->hmqk", q_rope, K)
    qk *= attn.sm_scale
    qk += mask[None, None, :, :]
    qk_with_sink = torch.cat([qk, sinks], dim=-1)
    probs = torch.softmax(qk_with_sink, dim=-1)
    context = torch.einsum("hmqk,khmd->qhmd", probs[..., :-1], V).reshape(n_tokens, -1)
    o_proj = attn.out(context)
    residual_add = residual_input + o_proj

    return (
        {
            "q_proj": flatten_last_token(q),
            "k_proj": flatten_last_token(k),
            "v_proj": flatten_last_token(v),
            "q_rope": flatten_last_token(q_rope),
            "k_rope": flatten_last_token(k_rope),
            "masked_scores": qk_with_sink[:, :, -1, :].reshape(-1).float().cpu().tolist(),
            "attention_probs": probs[:, :, -1, :].reshape(-1).float().cpu().tolist(),
            "attention_context": flatten_last_token(context),
            "o_proj": flatten_last_token(o_proj),
            "residual_add": flatten_last_token(residual_add),
            "post_attn_residual": last_token(residual_add),
        },
        residual_add,
    )


def slice_metrics(candidate: list[float], official: list[float], q_dim: int, kv_dim: int) -> dict:
    q_start = 0
    q_end = q_dim
    k_start = q_end
    k_end = q_end + kv_dim
    v_start = k_end
    v_end = k_end + kv_dim
    slices = {
        "Q": (q_start, q_end),
        "K": (k_start, k_end),
        "V": (v_start, v_end),
    }
    metrics = {}
    for name, (start, end) in slices.items():
        metrics[name] = {
            "start": start,
            "end": end,
            "max_abs_diff": float(max_abs_diff(candidate[start:end], official[start:end])),
            "mean_abs_diff": float(mean_abs_diff(candidate[start:end], official[start:end])),
        }
    return metrics


def rank_slices_by_mean_abs_diff(slices: dict) -> list[dict]:
    return sorted(
        (
            {
                "slice": name,
                "start": metrics["start"],
                "end": metrics["end"],
                "max_abs_diff": metrics["max_abs_diff"],
                "mean_abs_diff": metrics["mean_abs_diff"],
            }
            for name, metrics in slices.items()
        ),
        key=lambda item: (-item["mean_abs_diff"], -item["max_abs_diff"], item["slice"]),
    )


def mismatch_bucket_for_stage(stage_name: str) -> str:
    if stage_name.startswith("raw_"):
        return "bf16_gemm_output_before_bias"
    if stage_name.startswith("post_bias_"):
        return "bias_application_after_bf16_gemm"
    if stage_name == "final_packed_qkv":
        return "bf16_to_f16_cast_or_packed_serialization"
    return "matched_through_final_packed_qkv"


def summarize_shadow(
    *,
    name: str,
    compute_dtype: torch.dtype,
    input_vector: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    official_vector: list[float],
    baseline_vector: list[float],
    q_dim: int,
    kv_dim: int,
) -> dict:
    shadow_vector = project_qkv(
        input_vector=input_vector,
        qkv_weight=qkv_weight,
        qkv_bias=qkv_bias,
        compute_dtype=compute_dtype,
    )
    shadow_values = shadow_vector.float().cpu().tolist()
    overall_vs_official = compare_vectors(shadow_values, official_vector)
    runner_alignment = compare_vectors(shadow_values, baseline_vector)
    slices = slice_metrics(shadow_values, official_vector, q_dim=q_dim, kv_dim=kv_dim)
    return {
        "variant": name,
        "compute_precision": precision_label(compute_dtype),
        "reduction_precision": "f32",
        "output_precision": precision_label(compute_dtype),
        "overall_vs_official": overall_vs_official,
        "runner_alignment": runner_alignment,
        "slices": slices,
        "slice_ranking_by_mean_abs_diff": rank_slices_by_mean_abs_diff(slices),
        "materially_improves_current_mismatch": is_material_improvement(
            compare_vectors(baseline_vector, official_vector), overall_vs_official
        ),
    }


def build_synthetic_post_attention_residual_artifact(
    *,
    input_token_ids: list[int],
    model_root: Path,
    replay_checkpoint_root: Path,
    candidate_residual: torch.Tensor,
    device: torch.device,
) -> dict:
    return {
        "schema_version": "pinned-prompt-intermediate-artifact/v2",
        "suite_id": "developer-message",
        "boundary": "layer0_post_attention_residual",
        "layer_idx": 0,
        "provenance": {
            "model": str(replay_checkpoint_root),
            "capture_source": "host_replay_bf16_dense_qkv",
            "reference_kind": "local_candidate",
            "authority_level": "scaffold",
            "visible_devices": str(device),
            "max_model_len": 128,
            "gpu_memory_utilization": 0.75,
            "prompt_renderer": "harmony_gpt_oss_rs",
            "source_model_root": str(model_root),
        },
        "cases": [
            {
                "id": "developer-message-user-smoke",
                "input_token_ids": input_token_ids,
                "hidden_size": int(candidate_residual.shape[-1]),
                "final_token_hidden_f32": candidate_residual[-1].float().cpu().tolist(),
            }
        ],
    }


def run_internal_live_vs_shadow_mode(args: argparse.Namespace) -> int:
    residual_artifact, residual_case = load_single_case_artifact(args.local_residual_input_artifact)
    norm_artifact, norm_case = load_single_case_artifact(args.local_attn_norm_artifact)
    live_qkv_artifact, live_qkv_case = load_single_case_artifact(args.live_candidate_qkv_artifact)
    live_internal_artifact, live_internal_case = load_single_case_artifact(
        args.live_candidate_internal_trace_artifact
    )

    for artifact, case, boundary in [
        (residual_artifact, residual_case, "layer0_residual_input"),
        (norm_artifact, norm_case, "layer0_attn_norm_output"),
        (live_qkv_artifact, live_qkv_case, "layer0_qkv_projection_output"),
        (live_internal_artifact, live_internal_case, "layer0_qkv_bf16_internal_trace"),
    ]:
        validate_case(
            artifact,
            case,
            expected_boundary=boundary,
            expected_case_id="developer-message-user-smoke",
            expected_tokens=EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        )

    device = torch.device(args.device)
    model_root = extract_model_root(
        residual_artifact,
        norm_artifact,
        live_qkv_artifact,
        live_internal_artifact,
    )
    checkpoint_root = resolve_oracle_checkpoint_dir(model_root)
    checkpoint = load_checkpoint(checkpoint_root, device)
    qkv_weight, qkv_bias = load_weight_group(checkpoint, "block.0.attn.qkv")

    q_dim = int(live_internal_case["q_dim"])
    kv_dim = int(live_internal_case["kv_dim"])
    qkv_dim = int(live_internal_case["qkv_dim"])
    if (q_dim, kv_dim, qkv_dim) != (
        int(live_qkv_case["q_dim"]),
        int(live_qkv_case["kv_dim"]),
        int(live_qkv_case["qkv_dim"]),
    ):
        raise ValueError("live internal trace and live qkv sidecar disagree on QKV dimensions")

    live_pre_bias = live_internal_case["fused_qkv_pre_bias_f32"]
    live_post_bias = live_internal_case["fused_qkv_post_bias_f32"]
    live_packed = live_internal_case["packed_qkv_output_f32"]
    live_qkv_vector = live_qkv_case["final_token_hidden_f32"]

    if len(live_pre_bias) != qkv_dim or len(live_post_bias) != qkv_dim or len(live_packed) != qkv_dim:
        raise ValueError("live internal stage vectors do not match qkv_dim")
    if len(live_qkv_vector) != qkv_dim:
        raise ValueError("live qkv sidecar vector does not match qkv_dim")

    packed_vs_live_qkv = compare_vectors(live_packed, live_qkv_vector)
    local_norm_vector = torch.tensor(
        norm_case["final_token_hidden_f32"], dtype=torch.bfloat16, device=device
    )
    with torch.inference_mode():
        shadow_internal = build_bf16_shadow_internal_trace(
            input_vector=local_norm_vector,
            qkv_weight=qkv_weight,
            qkv_bias=qkv_bias,
        )

    stage_metrics = {
        "raw_q_pre_bias": compare_vectors(
            qkv_slices(shadow_internal["fused_qkv_pre_bias"], q_dim=q_dim, kv_dim=kv_dim)["Q"],
            qkv_slices(live_pre_bias, q_dim=q_dim, kv_dim=kv_dim)["Q"],
        ),
        "raw_k_pre_bias": compare_vectors(
            qkv_slices(shadow_internal["fused_qkv_pre_bias"], q_dim=q_dim, kv_dim=kv_dim)["K"],
            qkv_slices(live_pre_bias, q_dim=q_dim, kv_dim=kv_dim)["K"],
        ),
        "raw_v_pre_bias": compare_vectors(
            qkv_slices(shadow_internal["fused_qkv_pre_bias"], q_dim=q_dim, kv_dim=kv_dim)["V"],
            qkv_slices(live_pre_bias, q_dim=q_dim, kv_dim=kv_dim)["V"],
        ),
        "post_bias_q": compare_vectors(
            qkv_slices(shadow_internal["fused_qkv_post_bias"], q_dim=q_dim, kv_dim=kv_dim)["Q"],
            qkv_slices(live_post_bias, q_dim=q_dim, kv_dim=kv_dim)["Q"],
        ),
        "post_bias_k": compare_vectors(
            qkv_slices(shadow_internal["fused_qkv_post_bias"], q_dim=q_dim, kv_dim=kv_dim)["K"],
            qkv_slices(live_post_bias, q_dim=q_dim, kv_dim=kv_dim)["K"],
        ),
        "post_bias_v": compare_vectors(
            qkv_slices(shadow_internal["fused_qkv_post_bias"], q_dim=q_dim, kv_dim=kv_dim)["V"],
            qkv_slices(live_post_bias, q_dim=q_dim, kv_dim=kv_dim)["V"],
        ),
        "final_packed_qkv": compare_vectors(shadow_internal["packed_qkv_output"], live_packed),
    }
    final_packed_slices = compare_qkv_slices(
        shadow_internal["packed_qkv_output"], live_packed, q_dim=q_dim, kv_dim=kv_dim
    )
    stage_order = [
        "raw_q_pre_bias",
        "raw_k_pre_bias",
        "raw_v_pre_bias",
        "post_bias_q",
        "post_bias_k",
        "post_bias_v",
        "final_packed_qkv",
    ]
    earliest_internal_divergence_stage = "none_within_layer0_qkv_internal_trace"
    for stage_name in stage_order:
        if not stage_metrics[stage_name]["matched"]:
            earliest_internal_divergence_stage = stage_name
            break
    first_mismatch_consistent_with = mismatch_bucket_for_stage(
        earliest_internal_divergence_stage
    )
    if earliest_internal_divergence_stage == "none_within_layer0_qkv_internal_trace":
        summary_now = "shadow and live candidate match through the layer-0 BF16 QKV internal stages."
    else:
        stage = stage_metrics[earliest_internal_divergence_stage]
        summary_now = (
            f"shadow and live candidate first diverge at {earliest_internal_divergence_stage}; "
            f"mean_abs_diff {stage['mean_abs_diff']:.9f}, "
            f"max_abs_diff {stage['max_abs_diff']:.6f}."
        )

    report = {
        "schema_version": "runtime_forward_layer0_qkv_bf16_internal_status/v1",
        "provenance": {
            "local_residual_input_artifact_path": str(args.local_residual_input_artifact),
            "local_attn_norm_artifact_path": str(args.local_attn_norm_artifact),
            "live_candidate_qkv_artifact_path": str(args.live_candidate_qkv_artifact),
            "live_candidate_internal_trace_artifact_path": str(
                args.live_candidate_internal_trace_artifact
            ),
            "model_root": str(model_root),
            "checkpoint_root": str(checkpoint_root),
            "device": str(device),
            "shadow_helper_source": (
                "crates/gpt-oss-bench/tools/layer0_dense_qkv_semantics_sweep.py::run_internal_live_vs_shadow_mode"
            ),
        },
        "case": {
            "case_id": "developer-message-user-smoke",
            "input_token_ids": residual_case["input_token_ids"],
            "hidden_size": int(live_internal_case["hidden_size"]),
            "branch_taken": bool(live_internal_case["branch_taken"]),
            "q_dim": q_dim,
            "kv_dim": kv_dim,
            "qkv_dim": qkv_dim,
        },
        "live_internal_trace_consistent_with_live_qkv_sidecar": packed_vs_live_qkv,
        "stage_metrics": stage_metrics,
        "final_packed_slices": final_packed_slices,
        "earliest_internal_divergence_stage": earliest_internal_divergence_stage,
        "first_mismatch_consistent_with": first_mismatch_consistent_with,
        "decision_rule": {
            "raw_*_pre_bias": "bf16_gemm_output_before_bias",
            "post_bias_*": "bias_application_after_bf16_gemm",
            "final_packed_qkv": "bf16_to_f16_cast_or_packed_serialization",
            "none_within_layer0_qkv_internal_trace": "matched_through_final_packed_qkv",
        },
        "runtime_forward_layer0_qkv_bf16_internal_status_now": summary_now,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


def run_live_vs_shadow_mode(args: argparse.Namespace) -> int:
    residual_artifact, residual_case = load_single_case_artifact(args.local_residual_input_artifact)
    norm_artifact, norm_case = load_single_case_artifact(args.local_attn_norm_artifact)
    live_qkv_artifact, live_qkv_case = load_single_case_artifact(args.live_candidate_qkv_artifact)
    live_post_attention_residual_artifact, live_post_attention_residual_case = (
        load_single_case_artifact(args.live_candidate_post_attention_residual_artifact)
    )

    for artifact, case, boundary in [
        (residual_artifact, residual_case, "layer0_residual_input"),
        (norm_artifact, norm_case, "layer0_attn_norm_output"),
        (live_qkv_artifact, live_qkv_case, "layer0_qkv_projection_output"),
        (
            live_post_attention_residual_artifact,
            live_post_attention_residual_case,
            "layer0_post_attention_residual",
        ),
    ]:
        validate_case(
            artifact,
            case,
            expected_boundary=boundary,
            expected_case_id="developer-message-user-smoke",
            expected_tokens=EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        )

    device = torch.device(args.device)
    model_root = extract_model_root(
        residual_artifact,
        norm_artifact,
        live_qkv_artifact,
        live_post_attention_residual_artifact,
    )
    replay_model = load_layer0_replay_model(model_root, model_root, device)
    checkpoint_root = resolve_oracle_checkpoint_dir(model_root)
    checkpoint = load_checkpoint(checkpoint_root, device)
    qkv_weight, qkv_bias = load_weight_group(checkpoint, "block.0.attn.qkv")

    input_token_ids = residual_case["input_token_ids"]
    input_ids = torch.tensor(input_token_ids, dtype=torch.int64, device=device)
    local_norm_vector = torch.tensor(
        norm_case["final_token_hidden_f32"], dtype=torch.bfloat16, device=device
    )
    live_qkv_vector = live_qkv_case["final_token_hidden_f32"]
    live_post_attention_residual_vector = live_post_attention_residual_case[
        "final_token_hidden_f32"
    ]

    if len(live_qkv_vector) != int(live_qkv_case["qkv_dim"]):
        raise ValueError("live qkv artifact vector_size/qkv_dim mismatch")
    if len(live_post_attention_residual_vector) != int(live_post_attention_residual_case["hidden_size"]):
        raise ValueError("live post-attention residual artifact hidden_size mismatch")

    with torch.inference_mode():
        embedded = replay_model.embedding(input_ids)
        shadow_qkv = project_qkv(
            input_vector=local_norm_vector,
            qkv_weight=qkv_weight,
            qkv_bias=qkv_bias,
            compute_dtype=torch.bfloat16,
        )

    shadow_qkv_vector = shadow_qkv.float().cpu().tolist()
    shadow_vs_live_candidate_qkv = compare_vectors(shadow_qkv_vector, live_qkv_vector)
    shadow_vs_live_candidate_post_attention_residual = None
    earliest_divergence_boundary = "none_within_layer0"
    summary_now = ""

    if not shadow_vs_live_candidate_qkv["matched"]:
        earliest_divergence_boundary = "layer0_qkv_projection_output"
        summary_now = (
            "shadow and live candidate diverge at layer0_qkv_projection_output; "
            f"mean_abs_diff {shadow_vs_live_candidate_qkv['mean_abs_diff']:.9f}, "
            f"max_abs_diff {shadow_vs_live_candidate_qkv['max_abs_diff']:.6f}."
        )
    else:
        with torch.inference_mode():
            _shadow_attention_trace, shadow_post_attention_residual = (
                replay_layer0_attention_from_qkv(replay_model.attn, embedded, shadow_qkv)
            )
        shadow_post_attention_residual_vector = last_token(shadow_post_attention_residual)
        shadow_vs_live_candidate_post_attention_residual = compare_vectors(
            shadow_post_attention_residual_vector, live_post_attention_residual_vector
        )
        if shadow_vs_live_candidate_post_attention_residual["matched"]:
            summary_now = "shadow and live candidate match through layer 0."
        else:
            earliest_divergence_boundary = "layer0_post_attention_residual"
            summary_now = (
                "shadow and live candidate match at layer0_qkv_projection_output but diverge "
                "at layer0_post_attention_residual; "
                f"mean_abs_diff {shadow_vs_live_candidate_post_attention_residual['mean_abs_diff']:.9f}, "
                f"max_abs_diff {shadow_vs_live_candidate_post_attention_residual['max_abs_diff']:.6f}."
            )

    report = {
        "schema_version": "runtime_forward_layer0_qkv_live_vs_shadow_status/v1",
        "provenance": {
            "local_residual_input_artifact_path": str(args.local_residual_input_artifact),
            "local_attn_norm_artifact_path": str(args.local_attn_norm_artifact),
            "live_candidate_qkv_artifact_path": str(args.live_candidate_qkv_artifact),
            "live_candidate_post_attention_residual_artifact_path": str(
                args.live_candidate_post_attention_residual_artifact
            ),
            "model_root": str(model_root),
            "checkpoint_root": str(checkpoint_root),
            "device": str(device),
            "shadow_helper_source": (
                "crates/gpt-oss-bench/tools/layer0_dense_qkv_semantics_sweep.py::run_live_vs_shadow_mode"
            ),
        },
        "case": {
            "case_id": "developer-message-user-smoke",
            "input_token_ids": input_token_ids,
            "hidden_size": int(live_qkv_case["hidden_size"]),
            "q_dim": int(live_qkv_case["q_dim"]),
            "kv_dim": int(live_qkv_case["kv_dim"]),
            "qkv_dim": int(live_qkv_case["qkv_dim"]),
        },
        "shadow_vs_live_candidate_qkv": shadow_vs_live_candidate_qkv,
        "earliest_divergence_boundary": earliest_divergence_boundary,
        "runtime_forward_layer0_qkv_live_vs_shadow_status_now": summary_now,
    }
    if shadow_vs_live_candidate_post_attention_residual is not None:
        report["shadow_vs_live_candidate_post_attention_residual"] = (
            shadow_vs_live_candidate_post_attention_residual
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.output is None:
        if args.mode == "live-vs-shadow":
            args.output = Path(
                ".live/runtime-forward-layer0-qkv-bf16-live-vs-shadow-20260423/developer-message.runner-layer0-qkv-bf16-live-vs-shadow-status.json"
            )
        elif args.mode == "internal-live-vs-shadow":
            args.output = Path(
                ".live/runtime-forward-layer0-qkv-bf16-internal-20260423/developer-message.runner-layer0-qkv-bf16-internal-status.json"
            )
        else:
            args.output = Path(
                ".live/runtime-forward-layer0-qkv-bf16-candidate-20260423/developer-message.runner-layer0-qkv-bf16-candidate-status.json"
            )
    if args.mode == "live-vs-shadow":
        return run_live_vs_shadow_mode(args)
    if args.mode == "internal-live-vs-shadow":
        return run_internal_live_vs_shadow_mode(args)

    residual_artifact, residual_case = load_single_case_artifact(args.local_residual_input_artifact)
    norm_artifact, norm_case = load_single_case_artifact(args.local_attn_norm_artifact)
    qkv_artifact, qkv_case = load_single_case_artifact(args.local_qkv_projection_artifact)
    local_post_attention_residual_artifact, local_post_attention_residual_case = (
        load_single_case_artifact(args.local_post_attention_residual_artifact)
    )
    official_qkv_artifact, official_qkv_case = load_single_case_artifact(
        args.official_qkv_projection_artifact
    )
    official_post_attention_residual_artifact, official_post_attention_residual_case = (
        load_single_case_artifact(args.official_post_attention_residual_artifact)
    )

    for artifact, case, boundary in [
        (residual_artifact, residual_case, "layer0_residual_input"),
        (norm_artifact, norm_case, "layer0_attn_norm_output"),
        (qkv_artifact, qkv_case, "layer0_qkv_projection_output"),
        (
            local_post_attention_residual_artifact,
            local_post_attention_residual_case,
            "layer0_post_attention_residual",
        ),
        (official_qkv_artifact, official_qkv_case, "layer0_qkv_projection_output"),
        (
            official_post_attention_residual_artifact,
            official_post_attention_residual_case,
            "layer0_post_attention_residual",
        ),
    ]:
        validate_case(
            artifact,
            case,
            expected_boundary=boundary,
            expected_case_id="developer-message-user-smoke",
            expected_tokens=EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        )

    device = torch.device(args.device)
    model_root = extract_model_root(
        residual_artifact, norm_artifact, qkv_artifact, local_post_attention_residual_artifact
    )
    replay_model = load_layer0_replay_model(model_root, model_root, device)
    checkpoint_root = resolve_oracle_checkpoint_dir(model_root)
    checkpoint = load_checkpoint(checkpoint_root, device)

    qkv_weight, qkv_bias = load_weight_group(checkpoint, "block.0.attn.qkv")

    q_dim = int(qkv_case["q_dim"])
    kv_dim = int(qkv_case["kv_dim"])
    qkv_dim = int(qkv_case["qkv_dim"])

    input_token_ids = residual_case["input_token_ids"]
    input_ids = torch.tensor(input_token_ids, dtype=torch.int64, device=device)
    local_norm_vector = torch.tensor(
        norm_case["final_token_hidden_f32"], dtype=torch.bfloat16, device=device
    )
    current_runner_vector = qkv_case["final_token_hidden_f32"]
    official_vector = official_qkv_case["final_token_hidden_f32"]
    current_post_attention_residual_vector = local_post_attention_residual_case[
        "final_token_hidden_f32"
    ]
    official_post_attention_residual_vector = official_post_attention_residual_case[
        "final_token_hidden_f32"
    ]

    if len(current_runner_vector) != qkv_dim or len(official_vector) != qkv_dim:
        raise ValueError("QKV slice boundaries do not line up with the official qkv artifact")
    if len(current_post_attention_residual_vector) != len(official_post_attention_residual_vector):
        raise ValueError(
            "post-attention residual slice boundaries do not line up with the official artifact"
        )

    with torch.inference_mode():
        embedded = replay_model.embedding(input_ids)
        normed = replay_model.attn.norm(embedded)
        current_model_qkv = replay_model.attn.qkv(normed)
        local_embedding_vs_residual_input = compare_vectors(
            last_token(embedded), residual_case["final_token_hidden_f32"]
        )
        local_norm_vs_artifact = compare_vectors(
            last_token(normed), norm_case["final_token_hidden_f32"]
        )
        local_model_qkv_vs_artifact = compare_vectors(
            last_token(current_model_qkv), current_runner_vector
        )

    current_runner_post_attention_residual_vs_official = compare_vectors(
        current_post_attention_residual_vector, official_post_attention_residual_vector
    )
    local_style_shadow = summarize_shadow(
        name="local_style_f16",
        compute_dtype=torch.float16,
        input_vector=local_norm_vector,
        qkv_weight=qkv_weight,
        qkv_bias=qkv_bias,
        official_vector=official_vector,
        baseline_vector=current_runner_vector,
        q_dim=q_dim,
        kv_dim=kv_dim,
    )
    bf16_style_shadow = summarize_shadow(
        name="bf16_style_bf16",
        compute_dtype=torch.bfloat16,
        input_vector=local_norm_vector,
        qkv_weight=qkv_weight,
        qkv_bias=qkv_bias,
        official_vector=official_vector,
        baseline_vector=current_runner_vector,
        q_dim=q_dim,
        kv_dim=kv_dim,
    )

    with torch.inference_mode():
        bf16_qkv_candidate = project_qkv(
            input_vector=normed,
            qkv_weight=qkv_weight,
            qkv_bias=qkv_bias,
            compute_dtype=torch.bfloat16,
        )
        _bf16_candidate_attention_trace, bf16_candidate_residual = (
            replay_layer0_attention_from_qkv(
                replay_model.attn,
                embedded,
                bf16_qkv_candidate,
            )
        )

    bf16_candidate_post_attention_residual_vector = last_token(bf16_candidate_residual)
    if len(bf16_candidate_post_attention_residual_vector) != len(
        official_post_attention_residual_vector
    ):
        raise ValueError("bf16 candidate post-attention residual does not match official length")
    bf16_candidate_post_attention_residual_vs_official = compare_vectors(
        bf16_candidate_post_attention_residual_vector, official_post_attention_residual_vector
    )
    materially_improves_current_mismatch = is_material_improvement(
        current_runner_post_attention_residual_vs_official,
        bf16_candidate_post_attention_residual_vs_official,
    )
    bf16_k_mean = bf16_style_shadow["slices"]["K"]["mean_abs_diff"]
    bf16_q_mean = bf16_style_shadow["slices"]["Q"]["mean_abs_diff"]
    bf16_v_mean = bf16_style_shadow["slices"]["V"]["mean_abs_diff"]
    bf16_remaining_mismatch_k_dominant = bf16_k_mean >= bf16_q_mean and bf16_k_mean >= bf16_v_mean

    if materially_improves_current_mismatch:
        conclusion = "bf16_dense_qkv_resolves_downstream_post_attention_residual"
    else:
        conclusion = "bf16_dense_qkv_does_not_resolve_downstream_post_attention_residual"

    synthetic_post_attention_residual_artifact = build_synthetic_post_attention_residual_artifact(
        input_token_ids=input_token_ids,
        model_root=model_root,
        replay_checkpoint_root=checkpoint_root,
        candidate_residual=bf16_candidate_residual,
        device=device,
    )
    synthetic_post_attention_residual_artifact_path = (
        args.output.parent / "developer-message.runner-layer0-post-attention-residual.json"
    )

    report = {
        "schema_version": "runtime_forward_layer0_qkv_post_attention_bf16_status/v1",
        "provenance": {
            "local_residual_input_artifact_path": str(args.local_residual_input_artifact),
            "local_attn_norm_artifact_path": str(args.local_attn_norm_artifact),
            "local_qkv_projection_artifact_path": str(args.local_qkv_projection_artifact),
            "local_post_attention_residual_artifact_path": str(
                args.local_post_attention_residual_artifact
            ),
            "official_qkv_projection_artifact_path": str(args.official_qkv_projection_artifact),
            "official_post_attention_residual_artifact_path": str(
                args.official_post_attention_residual_artifact
            ),
            "model_root": str(model_root),
            "checkpoint_root": str(checkpoint_root),
            "device": str(torch.device(args.device)),
        },
        "case": {
            "case_id": "developer-message-user-smoke",
            "input_token_ids": input_token_ids,
            "hidden_size": int(qkv_case["hidden_size"]),
            "q_dim": q_dim,
            "kv_dim": kv_dim,
            "qkv_dim": qkv_dim,
        },
        "exact_case_sanity_checks": {
            "embedding_last_token_vs_local_residual_input": local_embedding_vs_residual_input,
            "attention_norm_last_token_vs_local_norm_output": local_norm_vs_artifact,
            "qkv_projection_last_token_vs_local_qkv_output": local_model_qkv_vs_artifact,
        },
        "baseline_current_post_attention_residual_vs_official": current_runner_post_attention_residual_vs_official,
        "bf16_candidate_post_attention_residual_vs_official": bf16_candidate_post_attention_residual_vs_official,
        "shadow_summaries": {
            "local_style_f16": local_style_shadow,
            "bf16_style_bf16": bf16_style_shadow,
        },
        "materially_improves_current_mismatch": materially_improves_current_mismatch,
        "bf16_remaining_mismatch_k_dominant": bf16_remaining_mismatch_k_dominant,
        "conclusion": conclusion,
        "runtime_forward_layer0_qkv_post_attention_bf16_status_now": (
            f"Layer-0 post-attention residual baseline is mean_abs_diff "
            f"{current_runner_post_attention_residual_vs_official['mean_abs_diff']:.9f}, "
            f"max_abs_diff {current_runner_post_attention_residual_vs_official['max_abs_diff']:.6f}; "
            f"bf16 dense-qkv candidate is mean_abs_diff "
            f"{bf16_candidate_post_attention_residual_vs_official['mean_abs_diff']:.9f}, "
            f"max_abs_diff {bf16_candidate_post_attention_residual_vs_official['max_abs_diff']:.6f}."
        ),
        "synthetic_post_attention_residual_artifact_path": str(
            synthetic_post_attention_residual_artifact_path
        ),
        "next_bounded_step": "surgical_dense_projection_semantics_fix",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with synthetic_post_attention_residual_artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(synthetic_post_attention_residual_artifact, handle, indent=2)
        handle.write("\n")
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
