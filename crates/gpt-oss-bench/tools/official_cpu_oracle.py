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
import os
import struct
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn.functional as F
from safetensors import safe_open


OFFICIAL_SOURCE_REVISION = "599476783c6f88508dab8577808b5ead5cbee8d2"
OFFICIAL_SOURCE_ARCHIVE_SHA256 = "7306d68ae017f461f2ebb82d04628f8dcba7cc7b431ef28e8786c947510c6f6b"
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
    parser.add_argument(
        "--trace-step",
        type=int,
        default=0,
        help="zero-based generated-token index whose selecting context/logits are captured",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dense-boundary-projection", choices=("k", "v"))
    parser.add_argument("--dense-boundary-output", type=int)
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
    marker = source / ".official-source.json"
    if marker.is_file():
        identity = json.loads(marker.read_text())
        revision = identity.get("revision")
        if identity.get("archive_sha256") != OFFICIAL_SOURCE_ARCHIVE_SHA256:
            raise RuntimeError("official source archive identity is not v0.0.9")
    else:
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


def bf16_bits(value: torch.Tensor) -> list[int]:
    raw = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16)
    return [int(item) for item in raw.reshape(-1).tolist()]


def fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class Oracle:
    def __init__(
        self,
        model: Path,
        official_source: Path,
        trace_layers: list[int],
        top_k: int,
        dense_boundary_projection: str | None,
        dense_boundary_output: int | None,
    ):
        self.store = TensorStore(model)
        self.config = json.loads((model / "config.json").read_text())
        self.official = load_official_operators(official_source)
        self.trace_layers = set(trace_layers)
        self.top_k = top_k
        self.dense_boundary_projection = dense_boundary_projection
        self.dense_boundary_output = dense_boundary_output
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
    def forward(
        self, token_ids: list[int], trace_step: int, capture_trace: bool
    ) -> tuple[torch.Tensor, dict | None]:
        hidden = self.store.tensor("model.embed_tokens.weight")[token_ids].clone()
        traces = []
        for layer_index in range(self.config["num_hidden_layers"]):
            hidden, trace = self.forward_layer(layer_index, hidden, capture_trace)
            if trace is not None:
                traces.append(trace)
        normalized = rms_norm(hidden, self.store.tensor("model.norm.weight"), self.epsilon)
        logits = F.linear(normalized[-1], self.store.tensor("lm_head.weight"))
        if not capture_trace:
            return logits, None
        trace = {
            "prompt_token_ids": list(token_ids),
            "context_token_ids": list(token_ids),
            "trace_step": trace_step,
            "expert_projection": "exact-bf16",
            "official_source_revision": OFFICIAL_SOURCE_REVISION,
            "layers": traces,
            "final_norm": cpu_values(normalized[-1]),
            "top_logits": top_logits(logits, self.top_k),
        }
        return logits, trace

    def forward_layer(
        self, layer_index: int, hidden: torch.Tensor, capture_trace: bool
    ):
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
        dense_boundary = None
        if capture_trace and layer_index == 0:
            dense_boundary = {
                "normalized_input_bf16_bits": bf16_bits(normalized[-1]),
                "key_pre_rope_bf16_bits": bf16_bits(k[-1]),
                "value_pre_rope_bf16_bits": bf16_bits(v[-1]),
            }
            if self.dense_boundary_projection is not None:
                output_index = self.dense_boundary_output
                if output_index is None:
                    raise ValueError("dense boundary output is required with a projection")
                weight_name = f"{attention}.{self.dense_boundary_projection}_proj.weight"
                bias_name = f"{attention}.{self.dense_boundary_projection}_proj.bias"
                weight = self.store.tensor(weight_name)[output_index]
                bias = self.store.tensor(bias_name)[output_index].float()
                prefixes = []
                for prefix_len in range(1, normalized.shape[-1] + 1):
                    repeated = []
                    for _ in range(5):
                        dot = torch.sum(
                            normalized[-1, :prefix_len].float()
                            * weight[:prefix_len].float(),
                            dtype=torch.float32,
                        )
                        post_bias = dot + bias
                        repeated.append(
                            (
                                fp32_bits(float(dot)),
                                fp32_bits(float(post_bias)),
                                bf16_bits(post_bias.to(torch.bfloat16))[0],
                            )
                        )
                    if len(set(repeated)) != 1:
                        raise RuntimeError("dense boundary prefix was not repeat-identical")
                    dot_bits, post_bias_bits, result_bits = repeated[0]
                    prefixes.append(
                        {
                            "prefix_len": prefix_len,
                            "dot_fp32_bits": dot_bits,
                            "post_bias_fp32_bits": post_bias_bits,
                            "result_bf16_bits": result_bits,
                        }
                    )
                observed = k[-1] if self.dense_boundary_projection == "k" else v[-1]
                dense_boundary["isolated_probe"] = {
                    "projection": self.dense_boundary_projection,
                    "output_index": output_index,
                    "normalized_input_bf16_bits": bf16_bits(normalized[-1]),
                    "weight_row_bf16_bits": bf16_bits(weight),
                    "bias_fp32_bits": fp32_bits(float(bias)),
                    "observed_projection_bf16_bits": bf16_bits(observed)[output_index],
                    "repetitions": 5,
                    "repeat_identical": True,
                    "prefixes": prefixes,
                }
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
        traced_experts = {}
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
            down_projection = F.linear(activated, down_weight, down_bias)
            expert_outputs[token_indices, ranks] = down_projection
            del down_weight
            if capture_trace and layer_index in self.trace_layers:
                final_token_matches = torch.where(token_indices == hidden.shape[0] - 1)[0]
                for local_index in final_token_matches.tolist():
                    rank = int(ranks[local_index])
                    route_weight = route_weights[-1, rank]
                    traced_experts[rank] = {
                        "rank": rank,
                        "expert_index": int(expert),
                        "gate_up_projection": cpu_values(gate_up[local_index]),
                        "swiglu": cpu_values(activated[local_index]),
                        "down_projection": cpu_values(down_projection[local_index]),
                        "weighted_output": cpu_values(
                            down_projection[local_index].float() * route_weight.float()
                        ),
                    }
        moe_output = torch.einsum("bec,be->bc", expert_outputs, route_weights)
        layer_output = after_attention + moe_output

        trace = None
        if capture_trace and layer_index in self.trace_layers:
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
                "experts": [traced_experts[rank] for rank in sorted(traced_experts)],
                "moe_output": cpu_values(moe_output[-1]),
                "layer_output": cpu_values(layer_output[-1]),
                "dense_boundary": dense_boundary,
            }
        return layer_output, trace


def write_new_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(args: argparse.Namespace) -> int:
    if args.trace_step < 0 or args.trace_step >= args.max_new_tokens:
        raise ValueError("--trace-step must be in [0, --max-new-tokens)")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    native_capture = json.loads(args.native_capture.read_text())
    prompt_token_ids = native_capture["prompt_token_ids"]
    if (args.dense_boundary_projection is None) != (args.dense_boundary_output is None):
        raise ValueError("dense boundary projection and output must be supplied together")
    oracle = Oracle(
        args.model,
        args.official_source,
        args.trace_layers,
        args.top_k,
        args.dense_boundary_projection,
        args.dense_boundary_output,
    )

    start = time.monotonic()
    tokens = list(prompt_token_ids)
    generated = []
    selected_trace = None
    top_logits_by_step = []
    for step in range(args.max_new_tokens):
        capture_trace = step == args.trace_step
        logits, trace = oracle.forward(tokens, step, capture_trace)
        if trace is not None:
            selected_trace = trace
        top_logits_by_step.append(top_logits(logits, args.top_k))
        token_id = int(torch.argmax(logits.float()))
        generated.append(token_id)
        tokens.append(token_id)
        if token_id in (200002, 200012):
            break
    if selected_trace is None:
        raise RuntimeError("generation stopped before the requested --trace-step")
    elapsed = time.monotonic() - start

    identity_json = os.environ.get("GPT_OSS_ORACLE_IDENTITY_JSON")
    oracle_identity = json.loads(identity_json) if identity_json else None
    report = {
        "schema_version": 1,
        "evidence_status": "insufficient_evidence",
        "scenario": native_capture["scenario"],
        "model_path": str(args.model),
        "official_source_path": str(args.official_source),
        "official_source_revision": OFFICIAL_SOURCE_REVISION,
        "oracle_identity": oracle_identity,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated,
        "expert_projection": "exact-bf16",
        "top_logits_by_step": top_logits_by_step,
        "elapsed_seconds": elapsed,
        "trace": selected_trace,
    }
    write_new_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "evidence_status": "incomplete",
            "worker": "official_cpu_oracle",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            write_new_atomic(args.output, failure)
        except FileExistsError:
            pass
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
