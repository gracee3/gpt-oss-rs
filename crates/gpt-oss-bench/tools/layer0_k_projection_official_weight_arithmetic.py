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
        description="Extract official layer-0 attn.qkv K slice and K projection arithmetic."
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--rmsnorm-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
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


def fnv1a64_f32(values: torch.Tensor) -> str:
    raw = values.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    h = 0xCBF29CE484222325
    for byte in raw:
        h ^= int(byte)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{h:016x}"


def compare_vectors(lhs: torch.Tensor, rhs: torch.Tensor) -> dict:
    lhs = lhs.to(torch.float32).reshape(-1).cpu()
    rhs = rhs.to(torch.float32).reshape(-1).cpu()
    if lhs.numel() != rhs.numel():
        raise ValueError(f"length mismatch {lhs.numel()} vs {rhs.numel()}")
    diff = (lhs - rhs).abs()
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "matched": bool(torch.equal(lhs, rhs)),
    }


def first_worst_token_feature(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[dict | None, dict | None]:
    lhs = lhs.to(torch.float32).contiguous()
    rhs = rhs.to(torch.float32).contiguous()
    diff = (lhs - rhs).abs()
    differing = torch.nonzero(diff > 0, as_tuple=False)
    first = None
    if differing.numel():
        token, feature = differing[0].tolist()
        first = {
            "token_index": int(token),
            "feature_index": int(feature),
            "lhs_value": float(lhs[token, feature].item()),
            "rhs_value": float(rhs[token, feature].item()),
            "abs_diff": float(diff[token, feature].item()),
        }
    max_value = diff.max()
    worst = None
    if float(max_value.item()) > 0.0:
        token, feature = torch.nonzero(diff == max_value, as_tuple=False)[0].tolist()
        worst = {
            "token_index": int(token),
            "feature_index": int(feature),
            "lhs_value": float(lhs[token, feature].item()),
            "rhs_value": float(rhs[token, feature].item()),
            "abs_diff": float(diff[token, feature].item()),
        }
    return first, worst


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with args.rmsnorm_replay.open("r", encoding="utf-8") as handle:
        rmsnorm_replay = json.load(handle)
    if rmsnorm_replay["case_id"] != "developer-message-user-smoke":
        raise ValueError("RMSNorm replay artifact is not the exact smoke case")

    model = load_layer0_replay_model(args.model_root, args.model_root, device)
    q_dim = model.attn.num_attention_heads * model.attn.head_dim
    kv_dim = model.attn.num_key_value_heads * model.attn.head_dim
    qkv_dim = q_dim + 2 * kv_dim
    hidden_size = model.attn.qkv.weight.shape[1]
    token_count = len(EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS)

    policy_outputs = rmsnorm_replay.get("policy_outputs_f32") or rmsnorm_replay.get("policy_outputs")
    if policy_outputs is None:
        raise KeyError("RMSNorm replay artifact does not contain policy_outputs_f32/policy_outputs")
    official_norm = torch.tensor(
        policy_outputs["manual_bf16_input_bf16_weight_f32_reduction_bf16_output"],
        dtype=torch.float32,
        device=device,
    ).reshape(token_count, hidden_size)
    official_norm_bf16 = official_norm.to(torch.bfloat16)

    qkv_weight = model.attn.qkv.weight.detach().contiguous()
    qkv_bias = model.attn.qkv.bias.detach().contiguous() if model.attn.qkv.bias is not None else None
    k_start = q_dim
    k_end = q_dim + kv_dim
    k_weight = qkv_weight[k_start:k_end, :].contiguous()
    k_bias = qkv_bias[k_start:k_end].contiguous() if qkv_bias is not None else None

    with torch.inference_mode():
        module_qkv = model.attn.qkv(official_norm_bf16).contiguous()
        module_k = module_qkv[:, k_start:k_end].contiguous()
        manual_k = torch.nn.functional.linear(official_norm_bf16, k_weight, k_bias).contiguous()
        manual_k_pre_bias = torch.nn.functional.linear(
            official_norm_bf16, k_weight, bias=None
        ).contiguous()

    manual_vs_module = compare_vectors(manual_k, module_k)
    first, worst = first_worst_token_feature(manual_k, module_k)
    output = {
        "schema_version": "runtime_forward_layer0_k_projection_official_weight_arithmetic/v1",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer0_attn_qkv_k_projection_before_grouped_view",
        "provenance": {
            "capture_source": "official_torch",
            "script_path": str(Path(__file__).resolve()),
            "model": str(args.model_root),
            "checkpoint_root": str(resolve_oracle_checkpoint_dir(args.model_root)),
            "device": str(device),
            "torch_version": torch.__version__,
            "torch_matmul_tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_cudnn_tf32_allowed": bool(torch.backends.cudnn.allow_tf32),
            "torch_deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "projection_path": "torch.nn.functional.linear/module attn.qkv on CPU bfloat16 tensors",
            "rmsnorm_replay_artifact": str(args.rmsnorm_replay),
        },
        "input_token_ids": EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        "token_count": token_count,
        "hidden_size": int(hidden_size),
        "q_dim": int(q_dim),
        "kv_dim": int(kv_dim),
        "qkv_dim": int(qkv_dim),
        "official_qkv_weight_metadata": {
            "tensor_name": "block.0.attn.qkv.weight",
            "shape": list(qkv_weight.shape),
            "dtype": str(qkv_weight.dtype).replace("torch.", ""),
            "device": str(qkv_weight.device),
            "layout_orientation": "row-major [qkv_output_feature, hidden]; torch linear uses input @ weight.T",
            "stride": list(qkv_weight.stride()),
        },
        "official_k_weight_metadata": {
            "tensor_name": "block.0.attn.qkv.weight[K slice]",
            "qkv_weight_tensor_name": "block.0.attn.qkv.weight",
            "k_slice_row_range": [int(k_start), int(k_end)],
            "shape": list(k_weight.shape),
            "dtype": str(k_weight.dtype).replace("torch.", ""),
            "device": str(k_weight.device),
            "layout_orientation": "row-major [k_output_feature, hidden]",
            "stride": list(k_weight.stride()),
            "stable_digest": fnv1a64_f32(k_weight),
        },
        "k_bias_metadata": {
            "exists": k_bias is not None,
            "shape": list(k_bias.shape) if k_bias is not None else [],
            "dtype": str(k_bias.dtype).replace("torch.", "") if k_bias is not None else "none",
            "max_abs": float(k_bias.to(torch.float32).abs().max().item()) if k_bias is not None else 0.0,
            "mean_abs": float(k_bias.to(torch.float32).abs().mean().item()) if k_bias is not None else 0.0,
            "stable_digest": fnv1a64_f32(k_bias) if k_bias is not None else "none",
            "all_zero": bool(torch.all(k_bias == 0).item()) if k_bias is not None else True,
        },
        "official_projection_outputs": {
            "official_module_k_output_f32": module_k.to(torch.float32).reshape(-1).cpu().tolist(),
            "official_manual_k_output_f32": manual_k.to(torch.float32).reshape(-1).cpu().tolist(),
            "official_manual_k_pre_bias_f32": manual_k_pre_bias.to(torch.float32).reshape(-1).cpu().tolist(),
        },
        "official_k_weight_f32": k_weight.to(torch.float32).reshape(-1).cpu().tolist(),
        "official_k_bias_f32": k_bias.to(torch.float32).reshape(-1).cpu().tolist() if k_bias is not None else [],
        "official_manual_projection_vs_module": {
            "metrics": manual_vs_module,
            "first_differing_token_feature": first,
            "worst_differing_token_feature": worst,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": output["case_id"],
                "boundary": output["boundary"],
                "output": str(args.output),
                "k_weight_shape": output["official_k_weight_metadata"]["shape"],
                "module_k_shape": [token_count, kv_dim],
                "manual_vs_module": manual_vs_module,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
