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
        description="Capture official PyTorch layer-0 attn.norm output for the exact smoke case."
    )
    parser.add_argument("--model-root", type=Path, required=True)
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
        embedded = model.embedding(input_ids)
        normed = model.attn.norm(embedded).contiguous()

    embedded_cpu = embedded.float().cpu()
    normed_cpu = normed.float().cpu()
    weight_cpu = model.attn.norm.scale.float().cpu()
    output = {
        "schema_version": "runtime_forward_layer0_attn_norm_full_official_capture/v2",
        "case_id": "developer-message-user-smoke",
        "boundary": "layer0_attn_norm_output",
        "provenance": {
            "capture_source": "official_torch",
            "script_path": str(Path(__file__).resolve()),
            "model": str(args.model_root),
            "checkpoint_root": str(resolve_oracle_checkpoint_dir(args.model_root)),
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "input_token_ids": EXACT_DEVELOPER_MESSAGE_USER_SMOKE_PROMPT_TOKEN_IDS,
        "hidden_size": int(normed_cpu.shape[1]),
        "token_count": int(normed_cpu.shape[0]),
        "input_shape": list(embedded_cpu.shape),
        "input_dtype": "bf16_serialized_as_f32",
        "input_layout": "token-major [token, hidden]",
        "output_shape": list(normed_cpu.shape),
        "output_dtype": "bf16_serialized_as_f32",
        "output_layout": "token-major [token, hidden]",
        "weight_shape": list(weight_cpu.shape),
        "weight_dtype": "f32",
        "weight_layout": "[hidden]",
        "rms_norm_epsilon": float(model.attn.norm.eps),
        "layer0_attn_norm_input_f32": embedded_cpu.reshape(-1).tolist(),
        "layer0_attn_norm_output_f32": normed_cpu.reshape(-1).tolist(),
        "layer0_attn_norm_weight_f32": weight_cpu.reshape(-1).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": output["case_id"],
                "boundary": output["boundary"],
                "output": str(args.output),
                "input_shape": output["input_shape"],
                "output_shape": output["output_shape"],
                "output_dtype": output["output_dtype"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
