#!/usr/bin/env python3
"""Create a restricted GPT-OSS model view for trusted prefill tracing.

This keeps the real checkpoint/tokenizer files in place and derives the smallest
local model directory needed for restricted trusted admission:
- rewrites config.json to full-attention-only with sliding disabled
- adds a late-loading safetensors override that zeroes all self_attn.sinks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def sync_symlink_tree(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_entry in source_dir.iterdir():
        if source_entry.name == "config.json":
            continue
        dest_entry = output_dir / source_entry.name
        if dest_entry.exists() or dest_entry.is_symlink():
            remove_existing(dest_entry)
        os.symlink(source_entry, dest_entry, target_is_directory=source_entry.is_dir())


def build_restricted_config(source_config: dict) -> dict:
    restricted = dict(source_config)
    num_layers = int(restricted["num_hidden_layers"])
    restricted["layer_types"] = ["full_attention"] * num_layers
    restricted["sliding_window"] = 0
    return restricted


def parse_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def dtype_nbytes(dtype: str) -> int:
    if dtype in ("BF16", "F16"):
        return 2
    if dtype in ("F32", "I32"):
        return 4
    if dtype == "U8":
        return 1
    raise ValueError(f"unsupported dtype for override: {dtype}")


def collect_sink_tensors(source_dir: Path) -> list[tuple[str, list[int], str]]:
    sink_tensors: list[tuple[str, list[int], str]] = []
    for shard_path in sorted(source_dir.glob("*.safetensors")):
        header = parse_safetensors_header(shard_path)
        for name, meta in sorted(header.items()):
            if name == "__metadata__" or not name.endswith("self_attn.sinks"):
                continue
            sink_tensors.append((name, meta["shape"], meta["dtype"]))
    if not sink_tensors:
        raise RuntimeError(f"no sink tensors found in {source_dir}")
    return sink_tensors


def build_safetensors_bytes(tensors: list[tuple[str, list[int], str]]) -> bytes:
    header: dict[str, dict[str, object]] = {}
    data_blob = bytearray()

    for name, shape, dtype in tensors:
        elem_count = 1
        for dim in shape:
            elem_count *= int(dim)
        start = len(data_blob)
        data_blob.extend(b"\x00" * (elem_count * dtype_nbytes(dtype)))
        end = len(data_blob)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        }

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + data_blob


def write_metadata(output_dir: Path, source_dir: Path) -> None:
    metadata = {
        "kind": "restricted_probe_model_view",
        "source_model": str(source_dir),
        "notes": [
            "Derived local view for restricted_prefill_trace on trusted runtime",
            "Config narrowed to full_attention only",
            "All self_attn.sinks tensors overridden to zeros in a late-loading shard",
        ],
    }
    (output_dir / "RESTRICTED_MODEL_VIEW.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_model).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"source model directory not found: {source_dir}")

    sync_symlink_tree(source_dir, output_dir)

    source_config = load_json(source_dir / "config.json")
    restricted_config = build_restricted_config(source_config)
    (output_dir / "config.json").write_text(json.dumps(restricted_config, indent=2) + "\n")

    sinks = collect_sink_tensors(source_dir)
    override_bytes = build_safetensors_bytes(sinks)
    (output_dir / "zzzz-sinks-override.safetensors").write_bytes(override_bytes)

    write_metadata(output_dir, source_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
