#!/usr/bin/env python3
import argparse
import inspect
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]

EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS = [
    200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410,
    6439, 2359, 22203, 656, 7788, 17527, 558, 87447, 100594, 25,
    220, 1323, 19, 12, 3218, 279, 30377, 289, 25, 14093, 279, 2,
    13888, 18403, 25, 8450, 11, 49159, 11, 1721, 13, 21030, 2804,
    413, 7360, 395, 1753, 3176, 13, 200007, 200006, 77944, 200008,
    2, 68406, 279, 17045, 59453, 1151, 13, 200007, 200006, 1428,
    200008, 25968, 483, 9707, 1001, 2195, 25, 40617, 200007,
    200006, 173781,
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
        description="Replay official PyTorch layer-0 attention RMSNorm policies for the exact smoke case."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-token", type=int)
    parser.add_argument("--trace-lane", type=int)
    return parser.parse_args()


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


def manual_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    input_policy: str,
    weight_policy: str,
    output_policy: str,
) -> torch.Tensor:
    if input_policy == "bf16":
        calc_x = x.to(torch.bfloat16).to(torch.float32)
    elif input_policy == "f32":
        calc_x = x.to(torch.float32)
    else:
        raise ValueError(input_policy)

    if weight_policy == "bf16":
        calc_weight = weight.to(torch.bfloat16).to(torch.float32)
    elif weight_policy == "f32":
        calc_weight = weight.to(torch.float32)
    else:
        raise ValueError(weight_policy)

    mean_square = (calc_x * calc_x).sum(dim=-1, dtype=torch.float32) / calc_x.shape[-1]
    inv_rms = torch.rsqrt(mean_square + eps)
    output = calc_x * inv_rms[:, None] * calc_weight[None, :]
    if output_policy == "bf16":
        return output.to(torch.bfloat16).to(torch.float32).contiguous()
    if output_policy == "f32":
        return output.to(torch.float32).contiguous()
    raise ValueError(output_policy)


def bf16_bits(value: torch.Tensor) -> int:
    return int(value.detach().reshape(()).to(torch.bfloat16).view(torch.uint16).item())


def scalar_trace(
    x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, token: int, lane: int, eps: float
) -> dict:
    calc_x = x.to(torch.bfloat16).to(torch.float32)
    calc_weight = weight.to(torch.bfloat16).to(torch.float32)
    row = calc_x[token]
    sum_of_squares = (row * row).sum(dtype=torch.float32)
    mean_square = sum_of_squares / row.shape[-1]
    mean_square_plus_epsilon = mean_square + eps
    inverse_rms = torch.rsqrt(mean_square_plus_epsilon)
    normalized = row[lane] * inverse_rms
    weighted = normalized * calc_weight[lane]
    output_bf16 = weighted.to(torch.bfloat16).to(torch.float32)
    return {
        "token_index": token,
        "lane_index": lane,
        "input_f32_value": float(x[token, lane].to(torch.float32).item()),
        "input_bf16_bits_after_roundtrip": f"0x{bf16_bits(x[token, lane]):04x}",
        "input_bf16_decoded_value": float(calc_x[token, lane].item()),
        "weight_f32_value": float(weight[lane].to(torch.float32).item()),
        "weight_bf16_bits_after_roundtrip": f"0x{bf16_bits(weight[lane]):04x}",
        "weight_bf16_decoded_value": float(calc_weight[lane].item()),
        "sum_of_squares": float(sum_of_squares.item()),
        "mean_square": float(mean_square.item()),
        "mean_square_plus_epsilon": float(mean_square_plus_epsilon.item()),
        "inverse_rms_scalar": float(inverse_rms.item()),
        "normalized_value_before_weight_multiply": float(normalized.item()),
        "weighted_f32_value_before_output_bf16_cast": float(weighted.item()),
        "output_bf16_bits": f"0x{bf16_bits(weighted):04x}",
        "output_bf16_decoded_value": float(output_bf16.item()),
        "official_manual_pytorch_bf16_bits": f"0x{bf16_bits(output[token, lane]):04x}",
        "official_manual_pytorch_bf16_value": float(output[token, lane].item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_layer0_replay_model(args.model_root, args.model_root, device)
    input_ids = torch.tensor(
        EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        dtype=torch.int64,
        device=device,
    )
    with torch.inference_mode():
        embedded = model.embedding(input_ids).contiguous()
        module_output = model.attn.norm(embedded).contiguous()
        weight = model.attn.norm.scale.detach().contiguous()
        eps = float(model.attn.norm.eps)

        policies = [
            (
                "manual_bf16_input_bf16_weight_f32_reduction_bf16_output",
                "bf16",
                "bf16",
                "bf16",
            ),
            (
                "manual_bf16_input_f32_weight_f32_reduction_bf16_output",
                "bf16",
                "f32",
                "bf16",
            ),
            (
                "manual_f32_input_f32_weight_f32_reduction_bf16_output",
                "f32",
                "f32",
                "bf16",
            ),
            (
                "exact_module_forward_path",
                "module",
                "module",
                "module",
            ),
        ]
        policy_outputs = {}
        policy_details = []
        for name, input_policy, weight_policy, output_policy in policies:
            if name == "exact_module_forward_path":
                output = module_output.to(torch.float32).cpu()
                details = {
                    "policy_name": name,
                    "input_cast_point": "module internal",
                    "weight_cast_point": "module internal",
                    "accumulation_dtype": "module internal",
                    "rsqrt_dtype": "module internal",
                    "multiply_dtype": "module internal",
                    "output_cast_serialization_point": "module output converted to f32 for JSON",
                }
            else:
                output = manual_rmsnorm(
                    embedded, weight, eps, input_policy, weight_policy, output_policy
                ).cpu()
                details = {
                    "policy_name": name,
                    "input_cast_point": input_policy,
                    "weight_cast_point": weight_policy,
                    "accumulation_dtype": "f32",
                    "rsqrt_dtype": "f32",
                    "multiply_dtype": "f32",
                    "output_cast_serialization_point": output_policy,
                }
            policy_outputs[name] = output.reshape(-1).tolist()
            policy_details.append(details)
        scalar_traces = {}
        if args.trace_token is not None and args.trace_lane is not None:
            manual_output = manual_rmsnorm(
                embedded, weight, eps, "bf16", "bf16", "bf16"
            )
            scalar_traces[
                "manual_bf16_input_bf16_weight_f32_reduction_bf16_output"
            ] = scalar_trace(
                embedded,
                weight,
                manual_output,
                args.trace_token,
                args.trace_lane,
                eps,
            )

    embedded_cpu = embedded.to(torch.float32).cpu()
    module_cpu = module_output.to(torch.float32).cpu()
    weight_cpu = weight.to(torch.float32).cpu()
    output = {
        "schema_version": "runtime_forward_layer0_attn_rmsnorm_all_token_bf16_replay_official_side/v1",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer0_attn_norm_output",
        "provenance": {
            "capture_source": "official_torch_manual_policy_replay",
            "script_path": str(Path(__file__).resolve()),
            "model": str(args.model_root),
            "checkpoint_root": str(resolve_oracle_checkpoint_dir(args.model_root)),
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "input_token_ids": EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        "hidden_size": int(module_cpu.shape[1]),
        "token_count": int(module_cpu.shape[0]),
        "epsilon": eps,
        "input_shape": list(embedded_cpu.shape),
        "output_shape": list(module_cpu.shape),
        "weight_shape": list(weight_cpu.shape),
        "official_input_f32": embedded_cpu.reshape(-1).tolist(),
        "official_module_output_f32": module_cpu.reshape(-1).tolist(),
        "official_weight_f32": weight_cpu.reshape(-1).tolist(),
        "policy_details": policy_details,
        "policy_outputs_f32": policy_outputs,
        "scalar_traces": scalar_traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": output["case_id"],
                "boundary": output["boundary"],
                "output": str(args.output),
                "output_shape": output["output_shape"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
