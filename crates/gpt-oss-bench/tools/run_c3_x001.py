#!/usr/bin/env python3
"""Execute the fresh C3-X-001 pair and exact dense-prefix localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from run_cpu_comparison import attempt_directory, native_command, official_command, run_logged


KERNELS = ("scalar", "avx2", "avx512-vnni")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--oracle-helper", type=Path, required=True)
    parser.add_argument("--oracle-lock", type=Path, required=True)
    parser.add_argument("--oracle-preflight", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repack-cache", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--official-cache", type=Path, required=True)
    return parser.parse_args()


def write_new_atomic(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def initial_scan(repository: Path, attempt: Path, native: Path, official: Path) -> dict:
    output = attempt / "initial-scan.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "crates/gpt-oss-bench/tools/scan_c3_dense_boundary.py"),
            "--native",
            str(native),
            "--official",
            str(official),
            "--output",
            str(output),
        ],
        capture_output=True,
    )
    (attempt / "initial-scan.stdout").write_bytes(completed.stdout)
    (attempt / "initial-scan.stderr").write_bytes(completed.stderr)
    if not output.is_file() or completed.returncode not in (0,):
        sys.stderr.buffer.write(completed.stderr)
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    return json.loads(output.read_text())


def official_isolated_probe(capture: dict) -> dict:
    layer = next(
        layer for layer in capture["trace"]["layers"] if layer["layer_index"] == 0
    )
    return layer["dense_boundary"]["isolated_probe"]


def first_prefix(native_capture: dict, official: dict) -> dict:
    native = native_capture["dense_boundary_probe"]
    for field in (
        "projection",
        "output_index",
        "normalized_input_bf16_bits",
        "weight_row_bf16_bits",
        "bias_fp32_bits",
    ):
        if native.get(field) != official.get(field):
            raise ValueError(f"isolated dense input differs at {field}")
    if (
        native_capture.get("dense_boundary_probe_repetitions") != 5
        or native_capture.get("dense_boundary_probe_repeat_identical") is not True
    ):
        raise ValueError("native isolated probe lacks five repeat-identical results")
    if official.get("repetitions") != 5 or official.get("repeat_identical") is not True:
        raise ValueError("official isolated probe lacks five repeat-identical results")
    native_prefixes = native["prefixes"]
    official_prefixes = official["prefixes"]
    if len(native_prefixes) != len(official_prefixes):
        raise ValueError("isolated prefix lengths differ")
    for native_prefix, official_prefix in zip(native_prefixes, official_prefixes):
        if native_prefix["prefix_len"] != official_prefix["prefix_len"]:
            raise ValueError("isolated prefix coordinates differ")
        differing = [
            field
            for field in ("dot_fp32_bits", "post_bias_fp32_bits", "result_bf16_bits")
            if native_prefix[field] != official_prefix[field]
        ]
        if differing:
            return {
                "outcome": "prefix_mismatch_localized",
                "prefix_len": native_prefix["prefix_len"],
                "differing_fields": differing,
                "native": native_prefix,
                "official": official_prefix,
            }
    return {"outcome": "isolated_prefixes_equal"}


def main() -> int:
    args = parse_args()
    attempt = attempt_directory()
    campaign = Path(os.environ["GPT_OSS_CAMPAIGN_ROOT"]).resolve()
    expected_cache = campaign / "raw" / "official" / "harmony_262.json"
    if args.official_cache.resolve() != expected_cache:
        raise RuntimeError(f"C3 official cache must be {expected_cache}")
    if args.repack_cache.resolve() != campaign / "cache":
        raise RuntimeError("C3 repack cache must be the fresh campaign cache")
    common = SimpleNamespace(
        **vars(args),
        scenario="harmony_262",
        kernel="automatic",
        backend="auto",
        max_new_tokens=8,
    )
    trace_args = ["--trace-layers", "0", "--trace-step", "6"]
    initial_native = attempt / "native-initial.json"
    initial_official = attempt / "official-initial.json"
    run_logged(
        native_command(common, initial_native, trace_args), attempt, "native-initial"
    )
    run_logged(
        official_command(
            common,
            attempt,
            initial_native.name,
            initial_official.name,
            trace_args,
        ),
        attempt,
        "official-initial",
    )
    expected_cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(initial_official, expected_cache)
    except FileExistsError:
        if initial_official.read_bytes() != expected_cache.read_bytes():
            raise RuntimeError("C3 official cache already contains different bytes")

    scan = initial_scan(args.repository.resolve(), attempt, initial_native, initial_official)
    result = {
        "schema": "gpt-oss-rs.c3-x-001/v1",
        "status": scan["status"],
        "initial": scan,
        "initial_capture_sha256": {
            "native": hashlib.sha256(initial_native.read_bytes()).hexdigest(),
            "official": hashlib.sha256(initial_official.read_bytes()).hexdigest(),
        },
        "isolated": {},
    }
    if scan["outcome"] == "not_reproduced":
        result["outcome"] = "not_reproduced"
        code = 0
    else:
        projection = scan["projection"]
        output_index = scan["output_index"]
        dense_args = [
            *trace_args,
            "--dense-boundary-projection",
            projection,
            "--dense-boundary-output",
            str(output_index),
        ]
        official_isolated = attempt / "official-isolated.json"
        run_logged(
            official_command(
                common,
                attempt,
                initial_native.name,
                official_isolated.name,
                dense_args,
            ),
            attempt,
            "official-isolated",
        )
        official_probe = official_isolated_probe(json.loads(official_isolated.read_text()))
        for kernel in KERNELS:
            common.kernel = kernel
            native_isolated = attempt / f"native-isolated-{kernel}.json"
            run_logged(
                native_command(common, native_isolated, dense_args),
                attempt,
                f"native-isolated-{kernel}",
            )
            native_capture = json.loads(native_isolated.read_text())
            result["isolated"][kernel] = first_prefix(native_capture, official_probe)
        if any(
            item["outcome"] == "prefix_mismatch_localized"
            for item in result["isolated"].values()
        ):
            result["outcome"] = "localized"
            result["status"] = "insufficient_evidence"
            code = 0
        else:
            result["outcome"] = "unlocalized"
            result["status"] = "fail"
            code = 1
    output = attempt / "c3-x-001.json"
    write_new_atomic(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
