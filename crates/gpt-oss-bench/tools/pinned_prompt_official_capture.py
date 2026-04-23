#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

FINAL_CAPTURE_SCHEMA = "pinned-prompt-official-capture-input/v1"
FINAL_CAPTURE_OUTPUT_SCHEMA = "pinned-prompt-official-capture-output/v1"
INTERMEDIATE_CAPTURE_SCHEMA = "pinned-prompt-official-intermediate-capture-input/v2"
INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA = (
    "pinned-prompt-official-intermediate-capture-output/v2"
)
EXPECTED_WORLD_SIZE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one PPP case through the official GPT-OSS PyTorch reference path."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    return parser.parse_args()


def resolve_checkpoint_dir(path: Path) -> Path:
    original_dir = path / "original"
    if original_dir.is_dir():
        return original_dir
    return path


def stable_top_k(logits: list[float], top_k: int) -> list[dict]:
    indexed = list(enumerate(logits))
    indexed.sort(key=lambda item: (-float(item[1]), item[0]))
    return [
        {"token_id": int(token_id), "logit": float(logit)}
        for token_id, logit in indexed[: max(top_k, 1)]
    ]


def capture_final_logits(model, input_token_ids: list[int], top_k: int, torch) -> dict:
    with torch.inference_mode():
        logits = (
            model(
                torch.as_tensor(
                    input_token_ids,
                    dtype=torch.int64,
                    device=model.embedding.weight.device,
                )
            )[-1]
            .float()
            .cpu()
            .tolist()
        )
    return {
        "argmax_token_id": int(max(range(len(logits)), key=lambda idx: logits[idx])),
        "final_position_top_k": stable_top_k(logits, top_k),
    }


def capture_final_token_hidden(model, input_token_ids: list[int], torch) -> dict:
    tokens = torch.as_tensor(
        input_token_ids, dtype=torch.int64, device=model.embedding.weight.device
    )
    with torch.inference_mode():
        hidden = model.embedding(tokens)
        for block in model.block:
            hidden = block(hidden)
        hidden = model.norm(hidden)
        final_token_hidden = hidden[-1].float().cpu().tolist()
    return {
        "hidden_size": len(final_token_hidden),
        "final_token_hidden_f32": final_token_hidden,
    }


def capture_transformer_layer_output(
    model, input_token_ids: list[int], layer_idx: int, torch
) -> dict:
    tokens = torch.as_tensor(
        input_token_ids, dtype=torch.int64, device=model.embedding.weight.device
    )
    if layer_idx < 0 or layer_idx >= len(model.block):
        raise ValueError(
            f"transformer_layer_output layer_idx {layer_idx} is out of range for {len(model.block)} blocks"
        )
    with torch.inference_mode():
        hidden = model.embedding(tokens)
        for block_idx, block in enumerate(model.block):
            hidden = block(hidden)
            if block_idx == layer_idx:
                final_token_hidden = hidden[-1].float().cpu().tolist()
                return {
                    "hidden_size": len(final_token_hidden),
                    "final_token_hidden_f32": final_token_hidden,
                }
    raise ValueError(
        f"failed to capture transformer_layer_output at layer_idx {layer_idx}"
    )


def capture_intermediate(model, capture_input: dict, torch) -> dict:
    boundary = capture_input["boundary"]
    layer_idx = capture_input.get("layer_idx")
    if boundary == "final_token_post_final_norm_pre_unembedding":
        if layer_idx is not None:
            raise ValueError(
                "final_token_post_final_norm_pre_unembedding requires layer_idx=None"
            )
        return capture_final_token_hidden(model, capture_input["input_token_ids"], torch)
    if boundary == "transformer_layer_output":
        if layer_idx is None:
            raise ValueError("transformer_layer_output requires layer_idx")
        return capture_transformer_layer_output(
            model, capture_input["input_token_ids"], int(layer_idx), torch
        )
    raise ValueError(f"unsupported intermediate boundary: {boundary}")


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def should_emit_output(current_rank: int) -> bool:
    return current_rank == 0


def require_expected_world_size() -> None:
    observed_world_size = world_size()
    if observed_world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            "PPP official capture requires WORLD_SIZE=2 via "
            "`python -m torch.distributed.run --standalone --nproc-per-node=2` "
            f"(got WORLD_SIZE={observed_world_size})"
        )


def broadcast_capture_input(input_path: Path, dist) -> dict:
    payload = [None]
    current_rank = rank()
    if should_emit_output(current_rank):
        with input_path.open("r", encoding="utf-8") as handle:
            payload[0] = json.load(handle)
    dist.broadcast_object_list(payload, src=0)
    capture_input = payload[0]
    if not isinstance(capture_input, dict):
        raise RuntimeError("rank 0 failed to broadcast a valid PPP capture input object")
    return capture_input


def build_output(capture_input: dict, capture_body: dict) -> dict:
    schema_version = capture_input["schema_version"]
    if schema_version == FINAL_CAPTURE_SCHEMA:
        return {
            "schema_version": FINAL_CAPTURE_OUTPUT_SCHEMA,
            "suite_id": capture_input["suite_id"],
            "case_id": capture_input["case_id"],
            "backend": "official_torch",
            "official_model": capture_input["official_model"],
            "prompt_renderer": capture_input["prompt_renderer"],
            "input_token_ids": capture_input["input_token_ids"],
            **capture_body,
        }
    if schema_version == INTERMEDIATE_CAPTURE_SCHEMA:
        return {
            "schema_version": INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA,
            "suite_id": capture_input["suite_id"],
            "case_id": capture_input["case_id"],
            "backend": "official_torch",
            "official_model": capture_input["official_model"],
            "prompt_renderer": capture_input["prompt_renderer"],
            "input_token_ids": capture_input["input_token_ids"],
            "boundary": capture_input["boundary"],
            "layer_idx": capture_input.get("layer_idx"),
            **capture_body,
        }
    raise ValueError(f"unsupported PPP official capture schema: {schema_version}")


def write_output(path: Path, output: dict, current_rank: int) -> None:
    if not should_emit_output(current_rank):
        raise RuntimeError(
            f"non-rank-0 helper process attempted output write on rank {current_rank}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


def run_rank_ownership_self_test(args: argparse.Namespace) -> int:
    current_rank = rank()
    if should_emit_output(current_rank):
        output = {
            "rank": current_rank,
            "world_size": world_size(),
            "schema_version": "ppp-helper-rank-ownership-self-test/v1",
        }
        write_output(args.output, output, current_rank)
        print(json.dumps(output, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if os.environ.get("PPP_OFFICIAL_CAPTURE_TEST_MODE") == "rank-ownership":
        return run_rank_ownership_self_test(args)

    require_expected_world_size()
    current_rank = rank()
    sys.path.insert(0, str(args.official_checkout))

    import torch  # noqa: E402
    import torch.distributed as dist  # noqa: E402
    from gpt_oss.torch.model import Transformer  # noqa: E402
    from gpt_oss.torch.utils import init_distributed  # noqa: E402

    initialized = False
    try:
        try:
            device = init_distributed()
            initialized = dist.is_initialized()
        except Exception as exc:
            raise RuntimeError(
                f"official PPP distributed init failed on rank {current_rank}: {exc}"
            ) from exc

        capture_input = (
            broadcast_capture_input(args.input, dist)
            if initialized
            else json.loads(args.input.read_text(encoding="utf-8"))
        )
        checkpoint_dir = resolve_checkpoint_dir(Path(capture_input["official_model"]))
        model = Transformer.from_checkpoint(str(checkpoint_dir), device=device)
        schema_version = capture_input["schema_version"]

        if schema_version == FINAL_CAPTURE_SCHEMA:
            capture_body = capture_final_logits(
                model, capture_input["input_token_ids"], int(capture_input["top_k"]), torch
            )
        elif schema_version == INTERMEDIATE_CAPTURE_SCHEMA:
            capture_body = capture_intermediate(model, capture_input, torch)
        else:
            raise ValueError(
                f"unsupported PPP official capture schema: {schema_version}"
            )

        if initialized:
            dist.barrier()

        if should_emit_output(current_rank):
            output = build_output(capture_input, capture_body)
            write_output(args.output, output, current_rank)
            print(json.dumps(output, indent=2))

        if initialized:
            dist.barrier()
        return 0
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
