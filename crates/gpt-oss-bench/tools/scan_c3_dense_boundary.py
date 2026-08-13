#!/usr/bin/env python3
"""Locate the first real layer-0 pre-RoPE K/V BF16 mismatch for C3-X-001."""

import argparse
import json
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def layer_zero(capture: dict) -> tuple[dict, int]:
    if capture.get("scenario") != "harmony_262":
        raise ValueError("C3-X-001 requires harmony_262")
    trace = capture.get("trace")
    if not isinstance(trace, dict) or trace.get("trace_step") != 6:
        raise ValueError("C3-X-001 requires generated-step-6 trace evidence")
    layer = next((item for item in trace.get("layers", []) if item.get("layer_index") == 0), None)
    if layer is None or not isinstance(layer.get("dense_boundary"), dict):
        raise ValueError("C3-X-001 requires layer-0 dense boundary evidence")
    context = trace.get("context_token_ids", trace.get("prompt_token_ids", []))
    return layer["dense_boundary"], len(context) - 1


def first_unequal(left: list[int], right: list[int]) -> int | None:
    if len(left) != len(right):
        raise ValueError("native and official dense boundary lengths differ")
    return next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)


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


def main() -> int:
    args = parse_args()
    native_capture = json.loads(args.native.read_text())
    official_capture = json.loads(args.official.read_text())
    native, native_row = layer_zero(native_capture)
    official, official_row = layer_zero(official_capture)
    if native_row != official_row:
        raise ValueError("native and official absolute trace rows differ")
    if native.get("normalized_input_bf16_bits") != official.get("normalized_input_bf16_bits"):
        result = {
            "schema": "gpt-oss-rs.c3-x-001/v1",
            "status": "invalid",
            "outcome": "producer_mismatch_before_dense_projection",
            "absolute_row": native_row,
        }
        code = 2
    else:
        result = None
        for projection, field in (
            ("k", "key_pre_rope_bf16_bits"),
            ("v", "value_pre_rope_bf16_bits"),
        ):
            output_index = first_unequal(native[field], official[field])
            if output_index is not None:
                result = {
                    "schema": "gpt-oss-rs.c3-x-001/v1",
                    "status": "insufficient_evidence",
                    "outcome": "mismatch_located",
                    "absolute_row": native_row,
                    "projection": projection,
                    "output_index": output_index,
                    "native_bf16_bits": native[field][output_index],
                    "official_bf16_bits": official[field][output_index],
                }
                break
        if result is None:
            result = {
                "schema": "gpt-oss-rs.c3-x-001/v1",
                "status": "pass",
                "outcome": "not_reproduced",
                "absolute_row": native_row,
                "compared_order": ["k", "v"],
            }
        code = 0
    write_new_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
