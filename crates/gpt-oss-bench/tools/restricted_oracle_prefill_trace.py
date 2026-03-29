#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OPENAI_WORKSPACE = REPO_ROOT.parents[1]
OPENAI_GPT_OSS_ROOT = OPENAI_WORKSPACE / "gpt-oss"
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
    config = ModelConfig(**json_config)
    config.sliding_window = 0
    return config


def load_restricted_transformer(path: Path, device: torch.device) -> Transformer:
    config = load_restricted_config(path)
    model = Transformer(config=config, device=device)
    model.eval()

    checkpoint = Checkpoint(str(path), device)
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


def layer0_attention_trace(model: Transformer, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
    attn = model.block[0].attn
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


def main() -> int:
    args = parse_args()
    with args.cuda_trace_json.open("r", encoding="utf-8") as handle:
        cuda_trace = json.load(handle)

    prompt_token_ids = cuda_trace["prompt_token_ids"]
    device = torch.device(args.device)
    model = load_restricted_transformer(args.original_model, device)
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
        "restricted_model_path": cuda_trace["restricted_model_path"],
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
