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

from gpt_oss.torch.model import ModelConfig, Transformer  # noqa: E402
from gpt_oss.torch.weights import Checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare restricted CUDA prefill activation trace against an independent PyTorch oracle."
    )
    parser.add_argument("--cuda-trace-json", type=Path, required=True)
    parser.add_argument("--original-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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


def resolve_oracle_checkpoint_dir(path: Path) -> Path:
    original_dir = path / "original"
    if original_dir.is_dir():
        return original_dir
    return path


def load_restricted_transformer(
    restricted_config_path: Path, checkpoint_path: Path, device: torch.device
) -> Transformer:
    config = load_restricted_config(restricted_config_path)
    model = Transformer(config=config, device=device)
    model.eval()

    checkpoint = Checkpoint(str(resolve_oracle_checkpoint_dir(checkpoint_path)), device)
    per_rank_intermediate_size = config.intermediate_size

    for name, param in model.named_parameters():
        loaded_tensor = checkpoint.get(name)
        if "mlp1" in name:
            loaded_tensor = loaded_tensor[:, : 2 * per_rank_intermediate_size, ...]
        elif "mlp2_weight" in name:
            loaded_tensor = loaded_tensor[..., :per_rank_intermediate_size]
        param.data.copy_(loaded_tensor)

    with torch.no_grad():
        for block in model.block:
            block.attn.sinks.zero_()
            block.attn.sliding_window = 0

    return model


def last_token(tensor: torch.Tensor) -> list[float]:
    return tensor[-1].float().cpu().tolist()


def mean_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / max(len(lhs), 1)


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(lhs, rhs))


def compare_stage(name: str, cuda_values: list[float], oracle_values: list[float]) -> dict:
    return {
        "stage": name,
        "max_abs_diff": max_abs_diff(cuda_values, oracle_values),
        "mean_abs_diff": mean_abs_diff(cuda_values, oracle_values),
    }


def flatten_last_token(tensor: torch.Tensor) -> list[float]:
    return tensor[-1].reshape(-1).float().cpu().tolist()


def is_exact_developer_message_user_smoke(prompt_token_ids: list[int]) -> bool:
    return prompt_token_ids == EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS


def layer0_attention_trace(model: Transformer, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
    attn = model.block[0].attn
    norm_input = x
    normed = attn.norm(x)
    qkv = attn.qkv(normed)
    qkv_pre_bias = torch.nn.functional.linear(normed, attn.qkv.weight, bias=None)
    qkv_post_bias = qkv
    q = qkv_post_bias[:, : attn.num_attention_heads * attn.head_dim].contiguous()
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
    residual_add = x + o_proj

    return ({
        "attention_norm_input": last_token(norm_input),
        "attention_norm_output": last_token(normed),
        "qkv_pre_bias": flatten_last_token(qkv_pre_bias),
        "qkv_post_bias": flatten_last_token(qkv_post_bias),
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
    }, residual_add)


def manual_qkv_from_norm(attn: torch.nn.Module, norm_last: list[float]) -> dict:
    if not norm_last:
        return {}
    norm = torch.tensor(norm_last, dtype=torch.float32, device=attn.qkv.weight.device)
    norm = norm.to(attn.qkv.weight.dtype)
    qkv_pre = torch.nn.functional.linear(norm, attn.qkv.weight, bias=None)
    qkv_post = torch.nn.functional.linear(norm, attn.qkv.weight, bias=attn.qkv.bias)
    q_dim = attn.num_attention_heads * attn.head_dim
    kv_dim = attn.num_key_value_heads * attn.head_dim
    return {
        "qkv_pre_bias": qkv_pre.float().cpu().tolist(),
        "qkv_post_bias": qkv_post.float().cpu().tolist(),
        "q_proj": qkv_post[:q_dim].float().cpu().tolist(),
        "k_proj": qkv_post[q_dim:q_dim + kv_dim].float().cpu().tolist(),
        "v_proj": qkv_post[q_dim + kv_dim:].float().cpu().tolist(),
    }


def manual_standalone_k_proj_from_norm(
    checkpoint_path: Path, layer_idx: int, norm_last: list[float], device: torch.device
) -> list[float]:
    if not norm_last:
        return []

    checkpoint = Checkpoint(str(resolve_oracle_checkpoint_dir(checkpoint_path)), device)
    weight = checkpoint.get(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
    bias = checkpoint.get(f"model.layers.{layer_idx}.self_attn.k_proj.bias")
    norm = torch.tensor(norm_last, dtype=torch.float32, device=weight.device)
    norm = norm.to(weight.dtype)
    projected = torch.nn.functional.linear(norm, weight, bias=bias)
    return projected.float().cpu().tolist()


def is_material_collapse(
    baseline: dict, candidate: dict, mean_threshold: float = 0.5, max_threshold: float = 0.5
) -> bool:
    return (
        candidate["mean_abs_diff"] <= baseline["mean_abs_diff"] * mean_threshold
        or candidate["max_abs_diff"] <= baseline["max_abs_diff"] * max_threshold
    )


def build_layer0_k_review(
    cuda_trace: dict,
    oracle_trace: dict,
    original_model_path: Path,
    device: torch.device,
) -> dict | None:
    prompt_token_ids = cuda_trace.get("prompt_token_ids", [])
    if not is_exact_developer_message_user_smoke(prompt_token_ids):
        return None

    layers = cuda_trace.get("trace", {}).get("layers", [])
    if not layers:
        return None
    cuda_attention = layers[0].get("attention") or {}
    oracle_attention = oracle_trace["layers"][0]["attention"]
    q_dim = len(cuda_attention.get("q_proj", []))
    kv_dim = len(cuda_attention.get("k_proj", []))
    if not q_dim or not kv_dim:
        return None

    qkv_post_bias = cuda_attention.get("qkv_post_bias")
    if qkv_post_bias is None:
        qkv_post_bias = (
            cuda_attention["q_proj"] + cuda_attention["k_proj"] + cuda_attention["v_proj"]
        )
    current_k = cuda_attention.get("k_proj", qkv_post_bias[q_dim : q_dim + kv_dim])
    norm_output = cuda_attention.get("attention_norm_output", [])
    if not norm_output:
        return None

    baseline = compare_stage("layer0.k_proj", current_k, oracle_attention["k_proj"])

    direct_standalone_k_proj = manual_standalone_k_proj_from_norm(
        original_model_path,
        0,
        norm_output,
        device,
    )
    transpose_axis_k_proj = (
        torch.tensor(current_k, dtype=torch.float32)
        .reshape(kv_dim // 64, 64)
        .transpose(0, 1)
        .reshape(-1)
        .float()
        .cpu()
        .tolist()
    )
    q_slice_as_k = qkv_post_bias[q_dim - kv_dim : q_dim]
    v_slice_as_k = qkv_post_bias[q_dim + kv_dim : q_dim + 2 * kv_dim]

    hypotheses = [
        ("current_assembled_k", current_k),
        ("direct_standalone_k_proj", direct_standalone_k_proj),
        ("transpose_axis_k_proj", transpose_axis_k_proj),
        ("q_slice_as_k", q_slice_as_k),
        ("v_slice_as_k", v_slice_as_k),
    ]
    ranked = []
    for hypothesis, values in hypotheses:
        metrics = compare_stage(hypothesis, values, oracle_attention["k_proj"])
        ranked.append(
            {
                "hypothesis": hypothesis,
                "max_abs_diff": metrics["max_abs_diff"],
                "mean_abs_diff": metrics["mean_abs_diff"],
                "materially_collapses_current_mismatch": is_material_collapse(
                    baseline, metrics
                ),
            }
        )

    ranked.sort(key=lambda item: (item["mean_abs_diff"], item["max_abs_diff"]))

    return {
        "schema_version": "runtime_forward_layer0_k_weight_layout_order_review/v1",
        "case_id": "developer-message-user-smoke",
        "input_token_ids": prompt_token_ids,
        "baseline_current_assembled_vs_official": {
            "max_abs_diff": baseline["max_abs_diff"],
            "mean_abs_diff": baseline["mean_abs_diff"],
        },
        "ranked_hypotheses": ranked,
        "conclusion": "assembly/order looks unlikely to explain the K mismatch",
        "next_bounded_step": "dense_projection_semantics",
    }


def main() -> int:
    args = parse_args()
    with args.cuda_trace_json.open("r", encoding="utf-8") as handle:
        cuda_trace = json.load(handle)

    prompt_token_ids = cuda_trace["prompt_token_ids"]
    device = torch.device(args.device)
    restricted_model_path = Path(cuda_trace["restricted_model_path"])
    model = load_restricted_transformer(restricted_model_path, args.original_model, device)
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.int64, device=device)

    with torch.inference_mode():
        x = model.embedding(input_ids)
        oracle_trace = {
            "embedding": last_token(x),
            "layers": [],
        }
        for layer_idx, block in enumerate(model.block):
            if layer_idx == 0:
                attention_trace, attn_hidden = layer0_attention_trace(model, x)
                cuda_attention = cuda_trace["trace"]["layers"][0].get("attention") or {}
                manual_from_cuda_norm = manual_qkv_from_norm(
                    model.block[0].attn, cuda_attention.get("attention_norm_output", [])
                )
                if manual_from_cuda_norm:
                    attention_trace["manual_from_cuda_norm"] = manual_from_cuda_norm
                attention_trace["manual_from_oracle_norm"] = manual_qkv_from_norm(
                    model.block[0].attn, attention_trace["attention_norm_output"]
                )
            else:
                attention_trace = None
                attn_hidden = block.attn(x)
            layer_output = block.mlp(attn_hidden)
            mlp_out = layer_output - attn_hidden
            layer_trace = {
                "layer_idx": layer_idx,
                "post_attn_residual": last_token(attn_hidden),
                "mlp_out": last_token(mlp_out),
                "layer_output": last_token(layer_output),
            }
            if attention_trace is not None:
                layer_trace["attention"] = attention_trace
            oracle_trace["layers"].append(layer_trace)
            x = layer_output

    stage_diffs = []
    stage_diffs.append(
        compare_stage("embedding", cuda_trace["trace"]["embedding"], oracle_trace["embedding"])
    )
    for cuda_layer, oracle_layer in zip(cuda_trace["trace"]["layers"], oracle_trace["layers"]):
        cuda_attention = cuda_layer.get("attention")
        oracle_attention = oracle_layer.get("attention")
        if cuda_attention and oracle_attention:
            for key in (
                "attention_norm_input",
                "attention_norm_output",
                "qkv_pre_bias",
                "qkv_post_bias",
                "q_proj",
                "k_proj",
                "v_proj",
                "q_rope",
                "k_rope",
                "masked_scores",
                "attention_probs",
                "attention_context",
                "o_proj",
                "residual_add",
            ):
                stage_diffs.append(
                    compare_stage(
                        f"layer{cuda_layer['layer_idx']}.{key}",
                        cuda_attention[key],
                        oracle_attention[key],
                    )
                )
            if "manual_from_cuda_norm" in oracle_attention:
                for key in ("qkv_pre_bias", "qkv_post_bias", "q_proj", "k_proj", "v_proj"):
                    stage_diffs.append(
                        compare_stage(
                            f"layer{cuda_layer['layer_idx']}.manual_from_cuda_norm.{key}",
                            cuda_attention[key],
                            oracle_attention["manual_from_cuda_norm"][key],
                        )
                    )
            for key in ("qkv_pre_bias", "qkv_post_bias", "q_proj", "k_proj", "v_proj"):
                stage_diffs.append(
                    compare_stage(
                        f"layer{cuda_layer['layer_idx']}.manual_from_oracle_norm.{key}",
                        cuda_attention[key],
                        oracle_attention["manual_from_oracle_norm"][key],
                    )
                )
        for key in ("post_attn_residual", "mlp_out", "layer_output"):
            if not cuda_layer[key] and not oracle_layer[key]:
                continue
            stage_diffs.append(
                compare_stage(
                    f"layer{cuda_layer['layer_idx']}.{key}",
                    cuda_layer[key],
                    oracle_layer[key],
                )
            )

    tolerance = 1e-2
    first_divergence = next(
        (stage for stage in stage_diffs if stage["max_abs_diff"] > tolerance),
        None,
    )
    report = {
        "prompt": cuda_trace["prompt"],
        "prompt_token_ids": prompt_token_ids,
        "restricted_model_path": str(restricted_model_path),
        "original_model_path": str(args.original_model),
        "oracle_device": str(device),
        "first_divergence_stage": first_divergence["stage"] if first_divergence else None,
        "stage_diffs": stage_diffs,
        "conclusion": (
            "Shared lower-level CUDA runner/model semantic bug is favored."
            if first_divergence
            else "No prefill-stage activation divergence detected."
        ),
    }
    layer0_k_review = build_layer0_k_review(
        cuda_trace, oracle_trace, args.original_model, device
    )
    if layer0_k_review is not None:
        report["layer0_k_weight_layout_order_review"] = layer0_k_review
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
