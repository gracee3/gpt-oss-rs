#!/usr/bin/env python3
import argparse
import copy
import inspect
import json
import math
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_SCHEMA_VERSION = "restricted_oracle_prefill_trace.v3"


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
    parser.add_argument("--cuda-trace-json", type=Path)
    parser.add_argument("--original-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--listen",
        action="store_true",
        help="opt-in NDJSON stdin listener that reuses one loaded oracle session across multiple compare requests",
    )
    parser.add_argument(
        "--compare-mode",
        choices=("raw", "runtime-emulated"),
        default="raw",
        help="raw compares against the bf16 PyTorch oracle; runtime-emulated gates layer-0 numeric-policy-sensitive surfaces against an fp16-emulated oracle while retaining raw deltas as telemetry.",
    )
    parser.add_argument(
        "--local-replay-layer",
        type=int,
        help="opt-in same-input local replay layer index; uses a traced runtime seed already captured in the CUDA trace",
    )
    parser.add_argument(
        "--local-replay-path",
        choices=("coarse", "attention", "mlp"),
        help="opt-in same-input local replay path; requires --local-replay-layer",
    )
    return parser.parse_args()


RUNTIME_EMULATED_NON_GATING_STAGES = {
    "layer0.raw_k_proj",
    "layer0.biased_k_proj",
    "layer0.k_after_proj",
    "layer0.last_q_for_scores",
    "layer0.k_for_scores",
    "layer0.attention_scores",
    "layer0.attention_probs",
    "layer0.attention_context",
    "layer0.o_proj",
    "layer0.post_attn_residual",
    "layer0.post_attn_norm_output",
    "layer0.mlp_out",
    "layer0.layer_output",
}

MXFP4_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
GPT_OSS_SWIGLU_ALPHA = 1.702
GPT_OSS_SWIGLU_LIMIT = 7.0


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


def flatten_all(tensor: torch.Tensor) -> list[float]:
    return tensor.reshape(-1).float().cpu().tolist()


def reduction_and_label(base_mean: float, base_max: float, emu_mean: float, emu_max: float) -> tuple[float, float, str]:
    mean_reduction = 0.0 if base_mean == 0 else 100.0 * (base_mean - emu_mean) / base_mean
    max_reduction = 0.0 if base_max == 0 else 100.0 * (base_max - emu_max) / base_max
    score = min(mean_reduction, max_reduction)
    if score >= 70.0:
        label = "materially explained by numeric policy"
    elif score >= 40.0:
        label = "partially explained by numeric policy"
    else:
        label = "not explained by numeric policy"
    return mean_reduction, max_reduction, label


def make_local_ledger_entry(name: str, runtime_values: list[float], baseline_values: list[float], emulated_values: list[float]) -> dict:
    base_mean = mean_abs_diff(baseline_values, runtime_values)
    base_max = max_abs_diff(baseline_values, runtime_values)
    emu_mean = mean_abs_diff(emulated_values, runtime_values)
    emu_max = max_abs_diff(emulated_values, runtime_values)
    mean_reduction, max_reduction, label = reduction_and_label(base_mean, base_max, emu_mean, emu_max)
    return {
        "surface": name,
        "baseline_mean_abs_diff": base_mean,
        "baseline_max_abs_diff": base_max,
        "emulated_mean_abs_diff": emu_mean,
        "emulated_max_abs_diff": emu_max,
        "mean_reduction_pct": mean_reduction,
        "max_reduction_pct": max_reduction,
        "classification": label,
    }


def stable_top_k_indices(logits: list[float], k: int) -> list[int]:
    indexed = list(enumerate(logits))
    indexed.sort(key=lambda item: (-float(item[1]), item[0]))
    return [idx for idx, _ in indexed[:k]]


def softmax_weights(vals: list[float]) -> list[float]:
    if not vals:
        return []
    max_val = max(vals)
    exps = [math.exp(float(v - max_val)) for v in vals]
    denom = sum(exps)
    return [e / denom for e in exps]


def decode_mxfp4_expert(blocks: torch.Tensor, scales: torch.Tensor, expert_idx: int, dtype: torch.dtype) -> torch.Tensor:
    blk = blocks[expert_idx].to(torch.int64)
    scl = scales[expert_idx].to(torch.int32) - 127
    lo = blk & 0x0F
    hi = blk >> 4
    lut = MXFP4_VALUES.to(dtype)
    out = torch.empty((*blk.shape[:-1], blk.shape[-1] * 2), dtype=dtype)
    out[..., 0::2] = lut[lo]
    out[..., 1::2] = lut[hi]
    out = torch.ldexp(out, scl.unsqueeze(-1)).reshape(blk.shape[0], -1)
    return out


def apply_gpt_oss_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate = x[0::2].clamp(max=GPT_OSS_SWIGLU_LIMIT)
    up = x[1::2].clamp(min=-GPT_OSS_SWIGLU_LIMIT, max=GPT_OSS_SWIGLU_LIMIT)
    glu = gate * torch.sigmoid(gate * GPT_OSS_SWIGLU_ALPHA)
    return (up + 1.0) * glu


def load_runtime_moe_tensors(checkpoint: Checkpoint, layer_idx: int) -> dict:
    prefix = f"block.{layer_idx}.mlp"
    return {
        "gate_weight": checkpoint._get_tensor(f"{prefix}.gate.weight"),
        "gate_bias": checkpoint._get_tensor(f"{prefix}.gate.bias"),
        "mlp1_blocks": checkpoint._get_tensor(f"{prefix}.mlp1_weight.blocks"),
        "mlp1_scales": checkpoint._get_tensor(f"{prefix}.mlp1_weight.scales"),
        "mlp1_bias": checkpoint._get_tensor(f"{prefix}.mlp1_bias"),
        "mlp2_blocks": checkpoint._get_tensor(f"{prefix}.mlp2_weight.blocks"),
        "mlp2_scales": checkpoint._get_tensor(f"{prefix}.mlp2_weight.scales"),
        "mlp2_bias": checkpoint._get_tensor(f"{prefix}.mlp2_bias"),
    }


def replay_mlp_from_seed(
    runtime_checkpoint: Checkpoint,
    config: ModelConfig,
    layer_idx: int,
    seed_last: torch.Tensor,
    residual_last: torch.Tensor,
    *,
    mode: str,
) -> dict:
    weights = load_runtime_moe_tensors(runtime_checkpoint, layer_idx)
    top_k = min(config.experts_per_token, config.num_experts)

    if mode == "baseline":
        seed_for_router = seed_last.to(torch.bfloat16)
        gate_weight = weights["gate_weight"]
        gate_bias = weights["gate_bias"]
        router_logits = torch.nn.functional.linear(seed_for_router, gate_weight, gate_bias).to(torch.float32)
        topk_indices = stable_top_k_indices(router_logits.tolist(), top_k)
        topk_logits = [router_logits[i].item() for i in topk_indices]
        route_weights = torch.tensor(softmax_weights(topk_logits), dtype=torch.float32)
        expert_sum = torch.zeros_like(seed_last, dtype=torch.float32)
        for rank, expert_idx in enumerate(topk_indices):
            w1 = decode_mxfp4_expert(weights["mlp1_blocks"], weights["mlp1_scales"], expert_idx, torch.bfloat16)
            b1 = weights["mlp1_bias"][expert_idx]
            gate_up = torch.matmul(w1, seed_for_router) + b1
            act = apply_gpt_oss_swiglu(gate_up.to(torch.float32)).to(torch.bfloat16)
            w2 = decode_mxfp4_expert(weights["mlp2_blocks"], weights["mlp2_scales"], expert_idx, torch.bfloat16)
            b2 = weights["mlp2_bias"][expert_idx]
            expert_out = (torch.matmul(w2, act) + b2).to(torch.float32)
            expert_sum += expert_out * route_weights[rank]
        mlp_out = expert_sum.to(torch.float32)
    else:
        seed_for_router = seed_last.to(torch.float32)
        gate_weight = weights["gate_weight"].to(torch.float32)
        gate_bias = weights["gate_bias"].to(torch.float32)
        router_logits = torch.nn.functional.linear(seed_for_router, gate_weight, gate_bias)
        topk_indices = stable_top_k_indices(router_logits.tolist(), top_k)
        topk_logits = [router_logits[i].item() for i in topk_indices]
        route_weights = torch.tensor(softmax_weights(topk_logits), dtype=torch.float32)
        expert_sum = torch.zeros_like(seed_last, dtype=torch.float32)
        for rank, expert_idx in enumerate(topk_indices):
            w1 = decode_mxfp4_expert(weights["mlp1_blocks"], weights["mlp1_scales"], expert_idx, torch.float32)
            b1 = weights["mlp1_bias"][expert_idx].to(torch.float32)
            gate_up = torch.matmul(w1, seed_for_router) + b1
            act = apply_gpt_oss_swiglu(gate_up)
            w2 = decode_mxfp4_expert(weights["mlp2_blocks"], weights["mlp2_scales"], expert_idx, torch.float32)
            b2 = weights["mlp2_bias"][expert_idx].to(torch.float32)
            expert_out = torch.matmul(w2, act) + b2
            expert_sum += expert_out * route_weights[rank]
        mlp_out = expert_sum.to(torch.float16).to(torch.float32)

    layer_output = residual_last.to(torch.float32) + mlp_out
    return {
        "router_logits": router_logits.to(torch.float32).cpu().tolist(),
        "router_topk_indices": [int(idx) for idx in topk_indices],
        "router_topk_weights": route_weights.to(torch.float32).cpu().tolist(),
        "expert_weighted_sum_pre_cast": expert_sum.to(torch.float32).cpu().tolist(),
        "mlp_out": mlp_out.to(torch.float32).cpu().tolist(),
        "layer_output": layer_output.to(torch.float32).cpu().tolist(),
    }


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
    post_attn_norm_output = model.block[0].mlp.norm(residual_add)

    return ({
        "attention_norm_input": last_token(norm_input),
        "attention_norm_output": last_token(normed),
        "qkv_pre_bias": flatten_last_token(qkv_pre_bias),
        "qkv_post_bias": flatten_last_token(qkv_post_bias),
        "q_proj": flatten_last_token(q),
        "k_proj": flatten_last_token(k),
        "v_proj": flatten_last_token(v),
        "raw_k_proj": qkv_pre_bias[
            :,
            attn.num_attention_heads
            * attn.head_dim : (attn.num_attention_heads + attn.num_key_value_heads)
            * attn.head_dim,
        ].contiguous().reshape(-1).float().cpu().tolist(),
        "biased_k_proj": k.reshape(-1).float().cpu().tolist(),
        "k_after_proj": k_heads[:, :, None, :].expand(-1, -1, q_mult, -1).permute(1, 2, 0, 3).reshape(-1).float().cpu().tolist(),
        "q_rope": flatten_last_token(q_rope),
        "k_rope": flatten_last_token(k_rope),
        "last_q_for_scores": q_rope[-1].reshape(-1).float().cpu().tolist(),
        "k_for_scores": K.permute(1, 2, 0, 3).reshape(-1).float().cpu().tolist(),
        "attention_scores": qk_with_sink[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "masked_scores": qk_with_sink[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_probs": probs[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_context": flatten_last_token(context),
        "o_proj": flatten_last_token(o_proj),
        "residual_add": flatten_last_token(residual_add),
        "post_attn_residual": last_token(residual_add),
        "post_attn_norm_output": last_token(post_attn_norm_output),
    }, residual_add)


def build_runtime_rope_tables(
    restricted_config_path: Path, num_tokens: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with (restricted_config_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rope_scaling = config.get("rope_scaling", {}) or {}
    rope_theta = float(config["rope_theta"])
    rope_scaling_type = rope_scaling.get("rope_type")
    rope_scaling_factor = float(rope_scaling.get("factor", 1.0))
    rope_ntk_alpha = float(rope_scaling.get("beta_slow", 1.0))
    rope_ntk_beta = float(rope_scaling.get("beta_fast", 32.0))
    initial_context_length = int(
        config.get(
            "initial_context_length",
            rope_scaling.get("original_max_position_embeddings", 4096),
        )
    )
    rope_scaling_truncate = bool(rope_scaling.get("truncate", False))

    half_dim = head_dim // 2
    inv_freq = torch.empty(half_dim, dtype=torch.float32, device=device)
    concentration = 1.0
    use_yarn = rope_scaling_type == "yarn" and rope_scaling_factor > 1.0
    if use_yarn:
        concentration = 0.1 * math.log(rope_scaling_factor) + 1.0
        d_half = head_dim / 2.0
        base_ln = math.log(rope_theta)
        context_len = max(initial_context_length, 1)
        low = d_half * math.log(context_len / (rope_ntk_beta * 2.0 * math.pi)) / base_ln
        high = d_half * math.log(context_len / (rope_ntk_alpha * 2.0 * math.pi)) / base_ln
        if rope_scaling_truncate:
            low = math.floor(low)
            high = math.ceil(high)
        if abs(high - low) < 1e-12:
            high = low + 1e-3
        for i in range(half_dim):
            freq = rope_theta ** (2.0 * i / head_dim)
            extrapolation = 1.0 / freq
            interpolation = 1.0 / (rope_scaling_factor * freq)
            ramp = min(1.0, max(0.0, (i - low) / (high - low)))
            maskv = 1.0 - ramp
            inv_freq[i] = interpolation * (1.0 - maskv) + extrapolation * maskv
    else:
        for i in range(half_dim):
            inv_freq[i] = 1.0 / (rope_theta ** (2.0 * i / head_dim))

    positions = torch.arange(num_tokens, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin


def apply_runtime_fp16_rope_q(
    q_pre_fp16: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(q_pre_fp16)
    half_dim = q_pre_fp16.shape[-1] // 2
    num_tokens, num_kv_heads = q_pre_fp16.shape[:2]
    for t in range(num_tokens):
        c = cos[t]
        s = sin[t]
        for kv in range(num_kv_heads):
            x0 = q_pre_fp16[t, kv, :, :half_dim].float()
            x1 = q_pre_fp16[t, kv, :, half_dim:].float()
            o0 = x0 * c.unsqueeze(0) - x1 * s.unsqueeze(0)
            o1 = x0 * s.unsqueeze(0) + x1 * c.unsqueeze(0)
            out[t, kv, :, :half_dim] = o0.to(torch.float16)
            out[t, kv, :, half_dim:] = o1.to(torch.float16)
    return out


def apply_runtime_fp16_rope_k(
    k_pre_fp16: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(k_pre_fp16)
    half_dim = k_pre_fp16.shape[-1] // 2
    num_tokens = k_pre_fp16.shape[0]
    for t in range(num_tokens):
        c = cos[t]
        s = sin[t]
        x0 = k_pre_fp16[t, :, :half_dim].float()
        x1 = k_pre_fp16[t, :, half_dim:].float()
        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c
        out[t, :, :half_dim] = o0.to(torch.float16)
        out[t, :, half_dim:] = o1.to(torch.float16)
    return out


def generic_oracle_attention_trace(block: torch.nn.Module, x: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor]:
    attn = block.attn
    normed = attn.norm(x)
    qkv = attn.qkv(normed)
    q_dim = attn.num_attention_heads * attn.head_dim
    kv_dim = attn.num_key_value_heads * attn.head_dim
    q_mult = attn.num_attention_heads // attn.num_key_value_heads

    q = qkv[:, :q_dim].contiguous()
    k = qkv[:, q_dim:q_dim + kv_dim].contiguous()
    v = qkv[:, q_dim + kv_dim:q_dim + 2 * kv_dim].contiguous()
    q_heads = q.view(-1, attn.num_key_value_heads, q_mult, attn.head_dim)
    k_heads = k.view(-1, attn.num_key_value_heads, attn.head_dim)
    v_heads = v.view(-1, attn.num_key_value_heads, attn.head_dim)
    q_rope, k_rope = attn.rope(q_heads, k_heads)

    n_tokens = q_rope.shape[0]
    K = k_rope[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = v_heads[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = attn.sinks.reshape(attn.num_key_value_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(q_rope.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    qk = torch.einsum("qhmd,khmd->hmqk", q_rope, K) * attn.sm_scale
    qk += mask[None, None, :, :]
    qk_with_sink = torch.cat([qk, sinks], dim=-1)
    probs = torch.softmax(qk_with_sink, dim=-1)
    context = torch.einsum("hmqk,khmd->qhmd", probs[..., :-1], V).reshape(n_tokens, -1)
    o_proj = attn.out(context)
    residual_add = x + o_proj
    post_attn_norm_output = block.mlp.norm(residual_add)
    return ({
        "attention_scores": qk_with_sink[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_probs": probs[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_context_pre_cast": context[-1].reshape(-1).float().cpu().tolist(),
        "attention_context": context[-1].reshape(-1).float().cpu().tolist(),
        "o_proj": o_proj[-1].reshape(-1).float().cpu().tolist(),
        "post_attn_residual": residual_add[-1].reshape(-1).float().cpu().tolist(),
        "post_attn_norm_output": post_attn_norm_output[-1].reshape(-1).float().cpu().tolist(),
    }, residual_add, post_attn_norm_output)


def generic_runtime_emulated_attention_trace(
    block: torch.nn.Module, restricted_model_path: Path, x: torch.Tensor
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    attn = block.attn
    num_tokens = x.shape[0]
    q_dim = attn.num_attention_heads * attn.head_dim
    kv_dim = attn.num_key_value_heads * attn.head_dim
    q_mult = attn.num_attention_heads // attn.num_key_value_heads

    normed = attn.norm(x)
    normed_fp16 = normed.to(torch.float16)
    q_weight_fp16 = attn.qkv.weight[:q_dim].to(torch.float16)
    k_weight_fp16 = attn.qkv.weight[q_dim:q_dim + kv_dim].to(torch.float16)
    v_weight_fp16 = attn.qkv.weight[q_dim + kv_dim:q_dim + 2 * kv_dim].to(torch.float16)
    q_bias_fp16 = attn.qkv.bias[:q_dim].to(torch.float16)
    k_bias_fp16 = attn.qkv.bias[q_dim:q_dim + kv_dim].to(torch.float16)
    v_bias_fp16 = attn.qkv.bias[q_dim + kv_dim:q_dim + 2 * kv_dim].to(torch.float16)

    biased_q_proj = torch.nn.functional.linear(normed_fp16, q_weight_fp16, bias=q_bias_fp16).to(torch.float16)
    biased_k_proj = torch.nn.functional.linear(normed_fp16, k_weight_fp16, bias=k_bias_fp16).to(torch.float16)
    biased_v_proj = torch.nn.functional.linear(normed_fp16, v_weight_fp16, bias=v_bias_fp16).to(torch.float16)

    q_pre = biased_q_proj.view(num_tokens, attn.num_key_value_heads, q_mult, attn.head_dim)
    k_pre = biased_k_proj.view(num_tokens, attn.num_key_value_heads, attn.head_dim)
    v_pre = biased_v_proj.view(num_tokens, attn.num_key_value_heads, attn.head_dim)

    cos, sin = build_runtime_rope_tables(restricted_model_path, num_tokens, attn.head_dim, normed.device)
    q_rope = apply_runtime_fp16_rope_q(q_pre, cos, sin)
    k_rope = apply_runtime_fp16_rope_k(k_pre, cos, sin)

    K = k_rope[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = v_pre[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = attn.sinks.reshape(attn.num_key_value_heads, q_mult, 1, 1).expand(-1, -1, num_tokens, -1)
    mask = torch.triu(q_rope.new_full((num_tokens, num_tokens), -float("inf")), diagonal=1)
    qk = torch.einsum("qhmd,khmd->hmqk", q_rope.float(), K.float()) * attn.sm_scale
    qk += mask[None, None, :, :]
    qk_with_sink = torch.cat([qk, sinks], dim=-1)
    probs = torch.softmax(qk_with_sink, dim=-1)
    context_pre_cast = torch.einsum("hmqk,khmd->qhmd", probs[..., :-1], V.float()).reshape(num_tokens, -1)
    context_fp16 = context_pre_cast.to(torch.float16)

    out_weight_fp16 = attn.out.weight.to(torch.float16)
    out_bias_fp16 = attn.out.bias.to(torch.float16)
    o_proj = torch.nn.functional.linear(context_fp16, out_weight_fp16, bias=out_bias_fp16).to(torch.float16)
    residual_add = (x.to(torch.float16) + o_proj).to(torch.float16)
    norm_scale_fp16 = block.mlp.norm.scale.to(torch.float16)
    post_attn_norm_output = (
        residual_add.float()
        * torch.rsqrt(torch.mean(residual_add.float() ** 2, dim=-1, keepdim=True) + block.mlp.norm.eps)
        * norm_scale_fp16.float()
    ).to(torch.float16)

    return ({
        "attention_scores": qk_with_sink[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_probs": probs[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_context_pre_cast": context_pre_cast[-1].reshape(-1).float().cpu().tolist(),
        "attention_context": context_fp16[-1].reshape(-1).float().cpu().tolist(),
        "o_proj": o_proj[-1].reshape(-1).float().cpu().tolist(),
        "post_attn_residual": residual_add[-1].reshape(-1).float().cpu().tolist(),
        "post_attn_norm_output": post_attn_norm_output[-1].reshape(-1).float().cpu().tolist(),
        "v_for_context": V.permute(1, 2, 0, 3).reshape(-1).float().cpu().tolist(),
    }, residual_add, post_attn_norm_output)


def layer0_runtime_emulated_attention_trace(
    model: Transformer, restricted_model_path: Path, x: torch.Tensor
) -> tuple[dict, torch.Tensor]:
    trace, residual_add, _ = generic_runtime_emulated_attention_trace(model.block[0], restricted_model_path, x)
    attn = model.block[0].attn
    block = model.block[0]
    num_tokens = x.shape[0]
    q_dim = attn.num_attention_heads * attn.head_dim
    kv_dim = attn.num_key_value_heads * attn.head_dim
    q_mult = attn.num_attention_heads // attn.num_key_value_heads

    normed = attn.norm(x)
    normed_fp16 = normed.to(torch.float16)
    k_weight_fp16 = attn.qkv.weight[q_dim : q_dim + kv_dim].to(torch.float16)
    k_bias_fp16 = attn.qkv.bias[q_dim : q_dim + kv_dim].to(torch.float16)
    raw_k_proj = torch.nn.functional.linear(normed_fp16, k_weight_fp16, bias=None).to(torch.float16)
    biased_k_proj = torch.nn.functional.linear(normed_fp16, k_weight_fp16, bias=k_bias_fp16).to(torch.float16)
    k_pre = biased_k_proj.view(num_tokens, attn.num_key_value_heads, attn.head_dim)
    cos, sin = build_runtime_rope_tables(restricted_model_path, num_tokens, attn.head_dim, normed.device)
    q_weight_fp16 = attn.qkv.weight[:q_dim].to(torch.float16)
    q_bias_fp16 = attn.qkv.bias[:q_dim].to(torch.float16)
    biased_q_proj = torch.nn.functional.linear(normed_fp16, q_weight_fp16, bias=q_bias_fp16).to(torch.float16)
    q_pre = biased_q_proj.view(num_tokens, attn.num_key_value_heads, q_mult, attn.head_dim)
    q_rope = apply_runtime_fp16_rope_q(q_pre, cos, sin)
    k_rope = apply_runtime_fp16_rope_k(k_pre, cos, sin)
    K = k_rope[:, :, None, :].expand(-1, -1, q_mult, -1)
    trace.update({
        "raw_k_proj": flatten_all(raw_k_proj),
        "biased_k_proj": flatten_all(biased_k_proj),
        "k_after_proj": flatten_all(k_pre[:, :, None, :].expand(-1, -1, q_mult, -1).permute(1, 2, 0, 3)),
        "last_q_for_scores": flatten_all(q_rope[-1]),
        "k_for_scores": flatten_all(K.permute(1, 2, 0, 3)),
    })
    return trace, residual_add


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


def build_stage_diffs(cuda_trace: dict, oracle_trace: dict) -> list[dict]:
    stage_diffs = []
    stage_diffs.append(compare_stage("embedding", cuda_trace["trace"]["embedding"], oracle_trace["embedding"]))
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
                "raw_k_proj",
                "biased_k_proj",
                "k_after_proj",
                "q_rope",
                "k_rope",
                "last_q_for_scores",
                "k_for_scores",
                "attention_scores",
                "masked_scores",
                "attention_probs",
                "attention_context",
                "o_proj",
                "residual_add",
                "post_attn_norm_output",
            ):
                if key not in cuda_attention or key not in oracle_attention:
                    continue
                stage_diffs.append(compare_stage(f"layer{cuda_layer['layer_idx']}.{key}", cuda_attention[key], oracle_attention[key]))
            if "manual_from_cuda_norm" in oracle_attention:
                for key in ("qkv_pre_bias", "qkv_post_bias", "q_proj", "k_proj", "v_proj"):
                    if key not in cuda_attention:
                        continue
                    stage_diffs.append(compare_stage(f"layer{cuda_layer['layer_idx']}.manual_from_cuda_norm.{key}", cuda_attention[key], oracle_attention["manual_from_cuda_norm"][key]))
            for key in ("qkv_pre_bias", "qkv_post_bias", "q_proj", "k_proj", "v_proj"):
                if key not in cuda_attention:
                    continue
                stage_diffs.append(compare_stage(f"layer{cuda_layer['layer_idx']}.manual_from_oracle_norm.{key}", cuda_attention[key], oracle_attention["manual_from_oracle_norm"][key]))
        for key in ("post_attn_residual", "mlp_out", "layer_output"):
            if not cuda_layer[key] and not oracle_layer[key]:
                continue
            stage_diffs.append(compare_stage(f"layer{cuda_layer['layer_idx']}.{key}", cuda_layer[key], oracle_layer[key]))
    return stage_diffs


def require_attention_seed(cuda_layer: dict, layer_idx: int) -> list[float]:
    attention = cuda_layer.get("attention") or {}
    seed = attention.get("attention_norm_input_full") or []
    if not seed:
        raise SystemExit(f"local replay layer {layer_idx} requires traced seed layer{layer_idx}.attention_norm_input_full")
    return seed


def require_post_attn_norm_output(cuda_layer: dict, layer_idx: int) -> list[float]:
    attention = cuda_layer.get("attention") or {}
    seed = attention.get("post_attn_norm_output") or []
    if not seed:
        raise SystemExit(f"local replay layer {layer_idx} requires traced seed layer{layer_idx}.post_attn_norm_output")
    return seed


def build_local_replay_report(
    cuda_trace: dict,
    model: Transformer,
    config: ModelConfig,
    runtime_checkpoint: Checkpoint,
    restricted_model_path: Path,
    layer_idx: int,
    path: str,
) -> dict:
    if layer_idx < 0 or layer_idx >= len(cuda_trace["trace"]["layers"]):
        raise SystemExit(f"local replay layer {layer_idx} is out of range")
    cuda_layer = cuda_trace["trace"]["layers"][layer_idx]
    block = model.block[layer_idx]

    result = {
        "layer": layer_idx,
        "path": path,
        "seed_stage": None,
        "ledger": [],
    }

    if path in {"coarse", "attention"}:
        seed_values = require_attention_seed(cuda_layer, layer_idx)
        x_seed = torch.tensor(seed_values, dtype=torch.float32, device=block.attn.qkv.weight.device).view(-1, config.hidden_size)
        result["seed_stage"] = f"layer{layer_idx}.attention_norm_input_full"
        baseline_attn, baseline_residual, baseline_post_attn_norm = generic_oracle_attention_trace(block, x_seed.to(block.attn.qkv.weight.dtype))
        emu_attn, emu_residual, emu_post_attn_norm = generic_runtime_emulated_attention_trace(block, restricted_model_path, x_seed)

        if path == "attention":
            needed = {
                "attention_context_pre_cast": cuda_layer.get("attention", {}).get("attention_context_pre_cast"),
                "attention_context": cuda_layer.get("attention", {}).get("attention_context"),
                "o_proj": cuda_layer.get("attention", {}).get("o_proj"),
                "post_attn_residual": cuda_layer.get("post_attn_residual"),
            }
            for surface, runtime_values in needed.items():
                if not runtime_values:
                    raise SystemExit(f"local replay attention path requires traced runtime surface layer{layer_idx}.{surface}")
                result["ledger"].append(
                    make_local_ledger_entry(
                        f"layer{layer_idx}.{surface}",
                        runtime_values,
                        baseline_attn[surface] if surface in baseline_attn else baseline_residual[-1].reshape(-1).float().cpu().tolist(),
                        emu_attn[surface] if surface in emu_attn else emu_residual[-1].reshape(-1).float().cpu().tolist(),
                    )
                )
            return result

        runtime_post_attn_residual = cuda_layer.get("post_attn_residual") or []
        runtime_post_attn_norm_output = cuda_layer.get("attention", {}).get("post_attn_norm_output") or []
        runtime_mlp_out = cuda_layer.get("mlp_out") or []
        runtime_layer_output = cuda_layer.get("layer_output") or []
        if not runtime_post_attn_residual:
            raise SystemExit(f"local replay coarse path requires traced runtime surface layer{layer_idx}.post_attn_residual")
        if not runtime_post_attn_norm_output:
            raise SystemExit(f"local replay coarse path requires traced runtime surface layer{layer_idx}.post_attn_norm_output")
        if not runtime_mlp_out or not runtime_layer_output:
            raise SystemExit(f"local replay coarse path requires traced runtime surfaces layer{layer_idx}.mlp_out and layer{layer_idx}.layer_output")

        baseline_mlp = replay_mlp_from_seed(
            runtime_checkpoint,
            config,
            layer_idx,
            baseline_post_attn_norm[-1].float(),
            baseline_residual[-1].float(),
            mode="baseline",
        )
        emu_mlp = replay_mlp_from_seed(
            runtime_checkpoint,
            config,
            layer_idx,
            emu_post_attn_norm[-1].float(),
            emu_residual[-1].float(),
            mode="runtime-emulated",
        )

        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.post_attn_residual", runtime_post_attn_residual, baseline_attn["post_attn_residual"], emu_attn["post_attn_residual"]))
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.post_attn_norm_output", runtime_post_attn_norm_output, baseline_attn["post_attn_norm_output"], emu_attn["post_attn_norm_output"]))
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.mlp_out", runtime_mlp_out, baseline_mlp["mlp_out"], emu_mlp["mlp_out"]))
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.layer_output", runtime_layer_output, baseline_mlp["layer_output"], emu_mlp["layer_output"]))
        result["routing"] = {
            "runtime_selected_experts": (cuda_layer.get("mlp") or {}).get("router_topk_indices"),
            "baseline_selected_experts": baseline_mlp["router_topk_indices"],
            "emulated_selected_experts": emu_mlp["router_topk_indices"],
            "runtime_route_weights": (cuda_layer.get("mlp") or {}).get("router_topk_weights"),
            "baseline_route_weights": baseline_mlp["router_topk_weights"],
            "emulated_route_weights": emu_mlp["router_topk_weights"],
        }
        return result

    if path == "mlp":
        seed_values = require_post_attn_norm_output(cuda_layer, layer_idx)
        runtime_mlp = cuda_layer.get("mlp") or {}
        if not runtime_mlp.get("router_logits") or not runtime_mlp.get("expert_weighted_sum_pre_cast"):
            raise SystemExit(
                f"local replay mlp path requires traced runtime surfaces layer{layer_idx}.router_logits and layer{layer_idx}.expert_weighted_sum_pre_cast"
            )
        runtime_post_attn_residual = cuda_layer.get("post_attn_residual") or []
        runtime_mlp_out = cuda_layer.get("mlp_out") or []
        runtime_layer_output = cuda_layer.get("layer_output") or []
        if not runtime_post_attn_residual or not runtime_mlp_out or not runtime_layer_output:
            raise SystemExit(
                f"local replay mlp path requires traced runtime surfaces layer{layer_idx}.post_attn_residual, mlp_out, and layer_output"
            )
        result["seed_stage"] = f"layer{layer_idx}.post_attn_norm_output"
        seed = torch.tensor(seed_values, dtype=torch.float32, device=block.mlp.gate.weight.device)
        residual_last = torch.tensor(runtime_post_attn_residual, dtype=torch.float32, device=block.mlp.gate.weight.device)
        baseline_mlp = replay_mlp_from_seed(runtime_checkpoint, config, layer_idx, seed, residual_last, mode="baseline")
        emu_mlp = replay_mlp_from_seed(runtime_checkpoint, config, layer_idx, seed, residual_last, mode="runtime-emulated")
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.router_logits", runtime_mlp["router_logits"], baseline_mlp["router_logits"], emu_mlp["router_logits"]))
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.expert_weighted_sum_pre_cast", runtime_mlp["expert_weighted_sum_pre_cast"], baseline_mlp["expert_weighted_sum_pre_cast"], emu_mlp["expert_weighted_sum_pre_cast"]))
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.mlp_out", runtime_mlp_out, baseline_mlp["mlp_out"], emu_mlp["mlp_out"]))
        result["ledger"].append(make_local_ledger_entry(f"layer{layer_idx}.layer_output", runtime_layer_output, baseline_mlp["layer_output"], emu_mlp["layer_output"]))
        result["routing"] = {
            "runtime_selected_experts": runtime_mlp.get("router_topk_indices"),
            "baseline_selected_experts": baseline_mlp["router_topk_indices"],
            "emulated_selected_experts": emu_mlp["router_topk_indices"],
            "runtime_route_weights": runtime_mlp.get("router_topk_weights"),
            "baseline_route_weights": baseline_mlp["router_topk_weights"],
            "emulated_route_weights": emu_mlp["router_topk_weights"],
            "topk_exact_match": runtime_mlp.get("router_topk_indices") == emu_mlp["router_topk_indices"],
            "route_weight_mean_abs_diff": mean_abs_diff(runtime_mlp.get("router_topk_weights") or [], emu_mlp["router_topk_weights"]),
            "route_weight_max_abs_diff": max_abs_diff(runtime_mlp.get("router_topk_weights") or [], emu_mlp["router_topk_weights"]),
        }
        return result

    raise SystemExit(f"unknown local replay path: {path}")


def load_cuda_trace(trace_path: Path) -> dict:
    with trace_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_mode_enabled() -> bool:
    return os.environ.get("GPT_OSS_ORACLE_TEST_MODE") == "1"


def create_oracle_session(device: torch.device) -> dict:
    return {
        "device": device,
        "restricted_model_path": None,
        "original_model_path": None,
        "model": None,
        "config": None,
        "runtime_checkpoint": None,
        "load_count": 0,
    }


def ensure_oracle_session(session: dict, cuda_trace: dict, original_model: Path) -> tuple[Transformer, ModelConfig, Checkpoint, Path, bool]:
    restricted_model_path = Path(cuda_trace["restricted_model_path"])
    reuse_ready = (
        session["model"] is not None
        and session["restricted_model_path"] == restricted_model_path
        and session["original_model_path"] == original_model
    )
    if not reuse_ready:
        session["restricted_model_path"] = restricted_model_path
        session["original_model_path"] = original_model
        session["config"] = load_restricted_config(restricted_model_path)
        session["model"] = load_restricted_transformer(restricted_model_path, original_model, session["device"])
        session["runtime_checkpoint"] = Checkpoint(str(resolve_oracle_checkpoint_dir(original_model)), session["device"])
        session["load_count"] += 1
    return (
        session["model"],
        session["config"],
        session["runtime_checkpoint"],
        restricted_model_path,
        reuse_ready,
    )


def build_compare_report(
    cuda_trace: dict,
    model: Transformer,
    config: ModelConfig,
    runtime_checkpoint: Checkpoint,
    restricted_model_path: Path,
    original_model: Path,
    device: torch.device,
    compare_mode: str,
    local_replay_layer: int | None,
    local_replay_path: str | None,
) -> dict:
    prompt_token_ids = cuda_trace["prompt_token_ids"]
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
                runtime_emulated_attention, runtime_emulated_attn_hidden = layer0_runtime_emulated_attention_trace(model, restricted_model_path, x)
                cuda_attention = cuda_trace["trace"]["layers"][0].get("attention") or {}
                manual_from_cuda_norm = manual_qkv_from_norm(model.block[0].attn, cuda_attention.get("attention_norm_output", []))
                if manual_from_cuda_norm:
                    attention_trace["manual_from_cuda_norm"] = manual_from_cuda_norm
                attention_trace["manual_from_oracle_norm"] = manual_qkv_from_norm(model.block[0].attn, attention_trace["attention_norm_output"])
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

    raw_stage_diffs = build_stage_diffs(cuda_trace, oracle_trace)
    gating_oracle_trace = oracle_trace
    if compare_mode == "runtime-emulated":
        gating_oracle_trace = copy.deepcopy(oracle_trace)
        gating_oracle_trace["layers"][0]["attention"].update(runtime_emulated_attention)
        gating_oracle_trace["layers"][0]["post_attn_residual"] = last_token(runtime_emulated_attn_hidden)
    stage_diffs = build_stage_diffs(cuda_trace, gating_oracle_trace)

    tolerance = 1e-2
    non_gating_stages = RUNTIME_EMULATED_NON_GATING_STAGES if compare_mode == "runtime-emulated" else set()
    first_divergence = next((stage for stage in stage_diffs if stage["stage"] not in non_gating_stages and stage["max_abs_diff"] > tolerance), None)
    raw_first_divergence = next((stage for stage in raw_stage_diffs if stage["max_abs_diff"] > tolerance), None)
    report = {
        "prompt": cuda_trace["prompt"],
        "prompt_token_ids": prompt_token_ids,
        "restricted_model_path": str(restricted_model_path),
        "original_model_path": str(original_model),
        "oracle_device": str(device),
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "compare_mode": compare_mode,
        "non_gating_stages": sorted(non_gating_stages),
        "first_divergence_stage": first_divergence["stage"] if first_divergence else None,
        "raw_first_divergence_stage": raw_first_divergence["stage"] if raw_first_divergence else None,
        "stage_diffs": stage_diffs,
        "raw_stage_diffs": raw_stage_diffs if compare_mode != "raw" else None,
        "conclusion": (
            "Shared lower-level CUDA runner/model semantic bug is favored." if first_divergence else "No prefill-stage activation divergence detected."
        ),
        "local_replay": None,
    }
    if local_replay_layer is not None:
        report["local_replay"] = build_local_replay_report(
            cuda_trace,
            model,
            config,
            runtime_checkpoint,
            restricted_model_path,
            local_replay_layer,
            local_replay_path,
        )
    return report


def write_report(output_path: Path, report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def run_compare_request(
    session: dict,
    trace_path: Path,
    original_model: Path,
    output_path: Path,
    compare_mode: str,
    local_replay_layer: int | None,
    local_replay_path: str | None,
) -> tuple[dict, bool]:
    cuda_trace = load_cuda_trace(trace_path)
    if test_mode_enabled():
        reused_session = session["load_count"] > 0
        if not reused_session:
            session["load_count"] = 1
        report = {
            "prompt": cuda_trace["prompt"],
            "prompt_token_ids": cuda_trace["prompt_token_ids"],
            "restricted_model_path": str(cuda_trace["restricted_model_path"]),
            "original_model_path": str(original_model),
            "oracle_device": str(session["device"]),
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "compare_mode": compare_mode,
            "non_gating_stages": sorted(
                RUNTIME_EMULATED_NON_GATING_STAGES if compare_mode == "runtime-emulated" else []
            ),
            "first_divergence_stage": None,
            "raw_first_divergence_stage": None,
            "stage_diffs": [
                {
                    "stage": "embedding",
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                }
            ],
            "raw_stage_diffs": None,
            "conclusion": "No prefill-stage activation divergence detected.",
            "local_replay": (
                None
                if local_replay_layer is None
                else {
                    "layer": local_replay_layer,
                    "path": local_replay_path,
                    "test_mode": True,
                }
            ),
        }
        write_report(output_path, report)
        return report, reused_session
    model, config, runtime_checkpoint, restricted_model_path, reused_session = ensure_oracle_session(
        session, cuda_trace, original_model
    )
    report = build_compare_report(
        cuda_trace,
        model,
        config,
        runtime_checkpoint,
        restricted_model_path,
        original_model,
        session["device"],
        compare_mode,
        local_replay_layer,
        local_replay_path,
    )
    write_report(output_path, report)
    return report, reused_session


def build_listener_response(
    report: dict,
    trace_path: Path,
    output_path: Path,
    compare_mode: str,
    reused_session: bool,
    session: dict,
) -> dict:
    return {
        "ok": True,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "warm_oracle": True,
        "session_reused": reused_session,
        "session_load_count": session["load_count"],
        "compare_mode": compare_mode,
        "trace_json": str(trace_path),
        "output": str(output_path),
        "first_divergence_stage": report["first_divergence_stage"],
        "raw_first_divergence_stage": report["raw_first_divergence_stage"],
        "local_replay": report["local_replay"] is not None,
        "conclusion": report["conclusion"],
    }


def validate_one_shot_args(args: argparse.Namespace) -> None:
    if args.original_model is None:
        raise SystemExit("--original-model is required")
    if args.cuda_trace_json is None:
        raise SystemExit("--cuda-trace-json is required unless --listen is used")
    if args.output is None:
        raise SystemExit("--output is required unless --listen is used")


def validate_local_replay_args(local_replay_layer: int | None, local_replay_path: str | None) -> None:
    if (local_replay_layer is None) != (local_replay_path is None):
        raise SystemExit("--local-replay-layer and --local-replay-path must be provided together")


def run_listen(args: argparse.Namespace) -> int:
    if args.original_model is None:
        raise SystemExit("--original-model is required with --listen")
    session = create_oracle_session(torch.device(args.device))
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        op = request.get("op", "compare")
        if op == "shutdown":
            print(json.dumps({"ok": True, "shutdown": True}))
            return 0
        if op != "compare":
            raise SystemExit(f"unsupported listen op: {op}")

        local_replay_layer = request.get("local_replay_layer")
        local_replay_path = request.get("local_replay_path")
        validate_local_replay_args(local_replay_layer, local_replay_path)
        trace_path = Path(request["cuda_trace_json"])
        output_path = Path(request["output"])
        compare_mode = request.get("compare_mode", args.compare_mode)
        report, reused_session = run_compare_request(
            session,
            trace_path,
            args.original_model,
            output_path,
            compare_mode,
            local_replay_layer,
            local_replay_path,
        )
        print(json.dumps(build_listener_response(report, trace_path, output_path, compare_mode, reused_session, session)))
        sys.stdout.flush()
    return 0


def main() -> int:
    args = parse_args()
    validate_local_replay_args(args.local_replay_layer, args.local_replay_path)
    if args.listen:
        return run_listen(args)

    validate_one_shot_args(args)
    session = create_oracle_session(torch.device(args.device))
    report, _ = run_compare_request(
        session,
        args.cuda_trace_json,
        args.original_model,
        args.output,
        args.compare_mode,
        args.local_replay_layer,
        args.local_replay_path,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
