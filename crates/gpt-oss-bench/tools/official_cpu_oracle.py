#!/usr/bin/env python3
"""Memory-bounded official PyTorch semantic oracle for pinned CPU prompts.

This uses the attention, rotary, and SwiGLU operators from the pinned OpenAI
reference source while reading the converted SafeTensors checkpoint a layer at
a time. Expert weights are dequantized one selected expert at a time so the
oracle remains usable on the 32 GiB i7; generated artifacts stay outside Git.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


OFFICIAL_SOURCE_REVISION = "7802bf263f902efd4c7d18fcceff3ba72f941e80"
MXFP4_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--trace-layers", type=parse_layers, default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


class TensorStore:
    def __init__(self, root: Path):
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.handles = {
            filename: safe_open(root / filename, framework="pt", device="cpu")
            for filename in sorted(set(self.weight_map.values()))
        }

    def tensor(self, name: str) -> torch.Tensor:
        return self.handles[self.weight_map[name]].get_tensor(name)

    def slice(self, name: str, index: int) -> torch.Tensor:
        return self.handles[self.weight_map[name]].get_slice(name)[index : index + 1][0]


def load_official_operators(source: Path):
    revision = (
        __import__("subprocess")
        .check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    if revision != OFFICIAL_SOURCE_REVISION:
        raise RuntimeError(
            f"official source revision {revision} does not match {OFFICIAL_SOURCE_REVISION}"
        )
    sys.path.insert(0, str(source))
    from gpt_oss.torch import model as official_model

    return official_model


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    value = x.float()
    value = value * torch.rsqrt(torch.mean(value**2, dim=-1, keepdim=True) + epsilon)
    return (value * weight.float()).to(x.dtype)


def dequantize_expert(
    store: TensorStore, blocks_name: str, scales_name: str, expert: int
) -> torch.Tensor:
    blocks = store.slice(blocks_name, expert)
    scales = store.slice(scales_name, expert).to(torch.int32) - 127
    rows, groups, packed = blocks.shape
    output = torch.empty((rows, groups, packed * 2), dtype=torch.bfloat16)
    lut = torch.tensor(MXFP4_VALUES, dtype=torch.bfloat16)
    output[..., 0::2] = lut[(blocks & 0x0F).long()]
    output[..., 1::2] = lut[(blocks >> 4).long()]
    torch.ldexp(output, scales.unsqueeze(-1), out=output)
    return output.reshape(rows, groups * packed * 2)


def top_logits(logits: torch.Tensor, count: int) -> list[dict]:
    values, indices = torch.topk(logits.float(), k=min(count, logits.numel()), sorted=True)
    return [
        {"token_id": int(token_id), "logit": float(logit)}
        for token_id, logit in zip(indices.tolist(), values.tolist())
    ]


def cpu_values(value: torch.Tensor) -> list[float]:
    return value.float().reshape(-1).tolist()


class Oracle:
    def __init__(self, model: Path, official_source: Path, trace_layers: list[int], top_k: int):
        self.store = TensorStore(model)
        self.config = json.loads((model / "config.json").read_text())
        self.official = load_official_operators(official_source)
        self.trace_layers = set(trace_layers)
        self.top_k = top_k
        self.epsilon = float(self.config["rms_norm_eps"])
        rope = self.config["rope_scaling"]
        self.rotary = self.official.RotaryEmbedding(
            self.config["head_dim"],
            self.config["rope_theta"],
            torch.float32,
            initial_context_length=rope["original_max_position_embeddings"],
            scaling_factor=rope["factor"],
            ntk_alpha=rope["beta_slow"],
            ntk_beta=rope["beta_fast"],
            device=torch.device("cpu"),
        )

    @torch.inference_mode()
    def forward(self, token_ids: list[int]) -> tuple[torch.Tensor, dict]:
        hidden = self.store.tensor("model.embed_tokens.weight")[token_ids].clone()
        traces = []
        for layer_index in range(self.config["num_hidden_layers"]):
            hidden, trace = self.forward_layer(layer_index, hidden)
            if trace is not None:
                traces.append(trace)
        normalized = rms_norm(hidden, self.store.tensor("model.norm.weight"), self.epsilon)
        logits = F.linear(normalized[-1], self.store.tensor("lm_head.weight"))
        trace = {
            "prompt_token_ids": token_ids,
            "official_source_revision": OFFICIAL_SOURCE_REVISION,
            "layers": traces,
            "final_norm": cpu_values(normalized[-1]),
            "top_logits": top_logits(logits, self.top_k),
        }
        return logits, trace

    def forward_layer(self, layer_index: int, hidden: torch.Tensor):
        prefix = f"model.layers.{layer_index}"
        attention = f"{prefix}.self_attn"
        normalized = rms_norm(
            hidden, self.store.tensor(f"{prefix}.input_layernorm.weight"), self.epsilon
        )
        q = F.linear(
            normalized,
            self.store.tensor(f"{attention}.q_proj.weight"),
            self.store.tensor(f"{attention}.q_proj.bias"),
        )
        k = F.linear(
            normalized,
            self.store.tensor(f"{attention}.k_proj.weight"),
            self.store.tensor(f"{attention}.k_proj.bias"),
        )
        v = F.linear(
            normalized,
            self.store.tensor(f"{attention}.v_proj.weight"),
            self.store.tensor(f"{attention}.v_proj.bias"),
        )
        token_count = hidden.shape[0]
        head_dim = self.config["head_dim"]
        kv_heads = self.config["num_key_value_heads"]
        attention_heads = self.config["num_attention_heads"]
        q = q.view(token_count, kv_heads, attention_heads // kv_heads, head_dim)
        k = k.view(token_count, kv_heads, head_dim)
        v = v.view(token_count, kv_heads, head_dim)
        q, k = self.rotary(q, k)
        window = (
            self.config["sliding_window"]
            if self.config["layer_types"][layer_index] in ("sliding_attention", "local_attention")
            else 0
        )
        attention_context = self.official.sdpa(
            q,
            k,
            v,
            self.store.tensor(f"{attention}.sinks"),
            1 / math.sqrt(head_dim),
            window,
        )
        attention_projection = F.linear(
            attention_context,
            self.store.tensor(f"{attention}.o_proj.weight"),
            self.store.tensor(f"{attention}.o_proj.bias"),
        )
        after_attention = hidden + attention_projection

        moe_input = rms_norm(
            after_attention,
            self.store.tensor(f"{prefix}.post_attention_layernorm.weight"),
            self.epsilon,
        )
        router_logits = F.linear(
            moe_input,
            self.store.tensor(f"{prefix}.mlp.router.weight"),
            self.store.tensor(f"{prefix}.mlp.router.bias"),
        )
        selected = torch.topk(
            router_logits,
            k=self.config["num_experts_per_tok"],
            dim=-1,
            sorted=True,
        )
        route_weights = F.softmax(selected.values, dim=1)
        expert_outputs = torch.empty(
            (hidden.shape[0], self.config["num_experts_per_tok"], hidden.shape[1]),
            dtype=torch.bfloat16,
        )
        experts_prefix = f"{prefix}.mlp.experts"
        for expert in torch.unique(selected.indices).tolist():
            token_indices, ranks = torch.where(selected.indices == expert)
            gate_weight = dequantize_expert(
                self.store,
                f"{experts_prefix}.gate_up_proj_blocks",
                f"{experts_prefix}.gate_up_proj_scales",
                expert,
            )
            gate_bias = self.store.tensor(f"{experts_prefix}.gate_up_proj_bias")[expert]
            gate_up = F.linear(moe_input[token_indices], gate_weight, gate_bias)
            del gate_weight
            activated = self.official.swiglu(
                gate_up, alpha=float(self.config.get("alpha", 1.702)), limit=self.config["swiglu_limit"]
            )
            down_weight = dequantize_expert(
                self.store,
                f"{experts_prefix}.down_proj_blocks",
                f"{experts_prefix}.down_proj_scales",
                expert,
            )
            down_bias = self.store.tensor(f"{experts_prefix}.down_proj_bias")[expert]
            expert_outputs[token_indices, ranks] = F.linear(activated, down_weight, down_bias)
            del down_weight
        moe_output = torch.einsum("bec,be->bc", expert_outputs, route_weights)
        layer_output = after_attention + moe_output

        trace = None
        if layer_index in self.trace_layers:
            trace = {
                "layer_index": layer_index,
                "input_norm": cpu_values(normalized[-1]),
                "query_after_rope": cpu_values(q[-1]),
                "key_after_rope": cpu_values(k[-1]),
                "value_projection": cpu_values(v[-1]),
                "attention_context": cpu_values(attention_context[-1]),
                "attention_projection": cpu_values(attention_projection[-1]),
                "post_attention_residual": cpu_values(after_attention[-1]),
                "router_logits": cpu_values(router_logits[-1]),
                "selected_experts": selected.indices[-1].tolist(),
                "routing_weights": cpu_values(route_weights[-1]),
                "moe_output": cpu_values(moe_output[-1]),
                "layer_output": cpu_values(layer_output[-1]),
            }
        return layer_output, trace


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    native_capture = json.loads(args.native_capture.read_text())
    prompt_token_ids = native_capture["prompt_token_ids"]
    oracle = Oracle(args.model, args.official_source, args.trace_layers, args.top_k)

    start = time.monotonic()
    tokens = list(prompt_token_ids)
    generated = []
    first_trace = None
    top_logits_by_step = []
    for step in range(args.max_new_tokens):
        logits, trace = oracle.forward(tokens)
        if first_trace is None:
            first_trace = trace
        top_logits_by_step.append(top_logits(logits, args.top_k))
        token_id = int(torch.argmax(logits.float()))
        generated.append(token_id)
        tokens.append(token_id)
        if token_id in (200002, 200012):
            break
    elapsed = time.monotonic() - start

    report = {
        "schema_version": 1,
        "scenario": native_capture["scenario"],
        "model_path": str(args.model),
        "official_source_path": str(args.official_source),
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated,
        "top_logits_by_step": top_logits_by_step,
        "elapsed_seconds": elapsed,
        "trace": first_trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
