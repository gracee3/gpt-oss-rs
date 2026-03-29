#!/usr/bin/env python3
import argparse
import json
import math
import os
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
        description="Compare restricted CUDA prefill logits against an independent PyTorch oracle."
    )
    parser.add_argument(
        "--cuda-probe-json",
        type=Path,
        required=True,
        help="Path to restricted-logit-diff.json produced by restricted_logit_diff.",
    )
    parser.add_argument(
        "--original-model",
        type=Path,
        required=True,
        help="Path to the original GPT-OSS checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the oracle comparison JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many top logits to include in the report.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the oracle path. Defaults to cpu.",
    )
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
    my_rank = 0
    world_size = 1
    per_rank_intermediate_size = config.intermediate_size // world_size

    for name, param in model.named_parameters():
        loaded_tensor = checkpoint.get(name)
        if "mlp1" in name:
            loaded_tensor = loaded_tensor[
                :,
                my_rank * 2 * per_rank_intermediate_size : (my_rank + 1)
                * 2
                * per_rank_intermediate_size,
                ...,
            ]
        elif "mlp2_weight" in name:
            loaded_tensor = loaded_tensor[
                ...,
                my_rank * per_rank_intermediate_size : (my_rank + 1)
                * per_rank_intermediate_size,
            ]
        param.data.copy_(loaded_tensor)

    with torch.no_grad():
        for block in model.block:
            block.attn.sinks.zero_()

    return model


def top_logits(logits: list[float], top_k: int) -> list[dict]:
    indexed = sorted(enumerate(logits), key=lambda item: item[1], reverse=True)[:top_k]
    return [{"token_id": token_id, "logit": float(logit)} for token_id, logit in indexed]


def largest_diffs(cuda_logits: list[float], oracle_logits: list[float], top_k: int) -> list[dict]:
    indexed = sorted(
        (
            {
                "token_id": token_id,
                "cuda_logit": float(cuda_logit),
                "oracle_logit": float(oracle_logit),
                "abs_diff": float(abs(cuda_logit - oracle_logit)),
            }
            for token_id, (cuda_logit, oracle_logit) in enumerate(zip(cuda_logits, oracle_logits))
        ),
        key=lambda item: item["abs_diff"],
        reverse=True,
    )
    return indexed[:top_k]


def mean_abs_diff(cuda_logits: list[float], oracle_logits: list[float]) -> float:
    total = 0.0
    for cuda_logit, oracle_logit in zip(cuda_logits, oracle_logits):
        total += abs(cuda_logit - oracle_logit)
    return total / max(len(cuda_logits), 1)


def main() -> int:
    args = parse_args()
    with args.cuda_probe_json.open("r", encoding="utf-8") as handle:
        cuda_probe = json.load(handle)

    prompt_token_ids = cuda_probe["prompt_token_ids"]
    cuda_prefill = next(step for step in cuda_probe["worker_steps"] if step["kind"] == "prefill")
    cuda_logits = cuda_prefill["logits"]

    device = torch.device(args.device)
    model = load_restricted_transformer(args.original_model, device)
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.int64, device=device)
    with torch.inference_mode():
        oracle_logits = model(input_ids)[-1].float().cpu().tolist()

    chosen_cuda = int(max(range(len(cuda_logits)), key=lambda idx: cuda_logits[idx]))
    chosen_oracle = int(max(range(len(oracle_logits)), key=lambda idx: oracle_logits[idx]))
    max_diff = max(abs(cuda - oracle) for cuda, oracle in zip(cuda_logits, oracle_logits))

    report = {
        "prompt": cuda_probe["prompt"],
        "prompt_token_ids": prompt_token_ids,
        "restricted_model_path": cuda_probe["restricted_model_path"],
        "original_model_path": str(args.original_model),
        "oracle_device": str(device),
        "checkpoint": "prefill",
        "cuda_chosen_token_id": chosen_cuda,
        "oracle_chosen_token_id": chosen_oracle,
        "chosen_token_match": chosen_cuda == chosen_oracle,
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_abs_diff(cuda_logits, oracle_logits)),
        "cuda_top_k": top_logits(cuda_logits, args.top_k),
        "oracle_top_k": top_logits(oracle_logits, args.top_k),
        "largest_logit_diffs": largest_diffs(cuda_logits, oracle_logits, args.top_k),
    }

    if max_diff < 1e-2 and chosen_cuda == chosen_oracle:
        report["conclusion"] = (
            "Prefill logits closely match the independent oracle; this favors a semantically invalid "
            "restricted reinterpretation over a lower-level CUDA bug."
        )
    else:
        report["conclusion"] = (
            "Prefill logits diverge from the independent oracle; this favors a shared lower-level "
            "CUDA runner/model semantic bug."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
