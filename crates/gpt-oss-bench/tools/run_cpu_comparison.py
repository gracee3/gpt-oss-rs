#!/usr/bin/env python3
"""Run one fresh native/official CPU comparison inside a campaign attempt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


KERNELS = ("automatic", "scalar", "avx2", "avx512-vnni")


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
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--kernel", choices=KERNELS, required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def attempt_directory() -> Path:
    raw = os.environ.get("GPT_OSS_ATTEMPT_DIR")
    if not raw:
        raise RuntimeError("GPT_OSS_ATTEMPT_DIR is required")
    path = Path(raw).resolve()
    if not path.is_dir():
        raise RuntimeError("campaign attempt directory does not exist")
    return path


def run_logged(command: list[str], attempt: Path, role: str) -> None:
    completed = subprocess.run(command, capture_output=True)
    (attempt / f"{role}.stdout").write_bytes(completed.stdout)
    (attempt / f"{role}.stderr").write_bytes(completed.stderr)
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr)
        raise subprocess.CalledProcessError(completed.returncode, command)


def native_command(
    args: argparse.Namespace, output: Path, extra: list[str] | None = None
) -> list[str]:
    kernel = "auto" if args.kernel == "automatic" else args.kernel
    command = [
        str(args.native_binary.resolve()),
        "--model",
        str(args.model.resolve()),
        "--repack-cache",
        str(args.repack_cache.resolve()),
        "--fixtures",
        str(args.fixtures.resolve()),
        "--scenario",
        args.scenario,
        "--kernel",
        kernel,
        "--cpu-matmul-backend",
        args.backend,
        "--threads",
        "4",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output",
        str(output),
    ]
    command.extend(extra or [])
    return command


def official_command(
    args: argparse.Namespace,
    attempt: Path,
    native_name: str,
    official_name: str,
    extra: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(args.oracle_helper.resolve()),
        "exec",
        "--lock",
        str(args.oracle_lock.resolve()),
        "--repository",
        str(args.repository.resolve()),
        "--model",
        str(args.model.resolve()),
        "--attempt-directory",
        str(attempt),
        "--preflight",
        str(args.oracle_preflight.resolve()),
        "--mode",
        "native",
        "--",
        "python",
        "/opt/oracle/official_cpu_oracle.py",
        "--native-capture",
        f"/attempt/{native_name}",
        "--model",
        "/model",
        "--official-source",
        "/opt/gpt-oss",
        "--output",
        f"/attempt/{official_name}",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--threads",
        "4",
    ]
    command.extend(extra or [])
    return command


def publish_official_cache(source: Path, cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, cache)
    except FileExistsError:
        if source.read_bytes() != cache.read_bytes():
            raise RuntimeError("official cache already contains different bytes")


def comparison_command(
    repository: Path, native: Path, official: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(repository.resolve() / "crates/gpt-oss-bench/tools/compare_cpu_parity.py"),
        "--native",
        str(native),
        "--official",
        str(official),
        "--output",
        str(output),
    ]


def main() -> int:
    args = parse_args()
    attempt = attempt_directory()
    campaign = Path(os.environ["GPT_OSS_CAMPAIGN_ROOT"]).resolve()
    official_cache = args.official_cache.resolve()
    expected_cache = campaign / "raw" / "official" / f"{args.scenario}.json"
    if official_cache != expected_cache:
        raise RuntimeError(f"official cache must be {expected_cache}")
    if args.repack_cache.resolve() != campaign / "cache":
        raise RuntimeError("comparison repack cache must be the fresh campaign cache")
    native = attempt / "native.json"
    official = attempt / "official.json"
    comparison = attempt / "comparison.json"
    run_logged(native_command(args, native), attempt, "native-worker")
    if official_cache.is_file():
        os.link(official_cache, official)
    else:
        run_logged(
            official_command(args, attempt, native.name, official.name),
            attempt,
            "official-worker",
        )
        publish_official_cache(official, official_cache)
    completed = subprocess.run(
        comparison_command(args.repository, native, official, comparison),
        capture_output=True,
    )
    (attempt / "comparator.stdout").write_bytes(completed.stdout)
    (attempt / "comparator.stderr").write_bytes(completed.stderr)
    if comparison.is_file():
        print(json.dumps(json.loads(comparison.read_text()), indent=2, sort_keys=True))
    else:
        sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
