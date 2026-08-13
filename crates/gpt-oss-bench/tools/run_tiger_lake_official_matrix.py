#!/usr/bin/env python3
"""Run or resume the frozen six-cell/42-comparison Tiger Lake matrix."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SCENARIOS = (
    "harmony_63",
    "harmony_122",
    "harmony_136",
    "harmony_262",
    "harmony_346",
    "harmony_444",
    "tool_history_180",
)

# (requested kernel, requested matrix, effective operation kernel, effective matrix)
CELLS = (
    ("automatic", "auto", "hybrid", "profiled-tiger-lake"),
    ("scalar", "auto", "scalar", "scalar"),
    ("avx2", "auto", "avx2", "scalar"),
    ("avx512-vnni", "auto", "avx512-vnni", "scalar"),
    ("automatic", "avx2", "hybrid", "avx2"),
    ("automatic", "avx512-vnni", "hybrid", "avx512-vnni"),
)


def matrix_commands(args: argparse.Namespace) -> list[list[str]]:
    repository = args.repository.resolve()
    root = args.root.resolve()
    comparison = repository / "crates/gpt-oss-bench/tools/run_cpu_comparison.py"
    preflight = root / "private/oracle-preflight.json"
    official_cache_root = root / "raw/official"
    commands = []
    for scenario in SCENARIOS:
        for kernel, backend, effective_kernel, effective_backend in CELLS:
            commands.append([
                str(args.validation_binary.resolve()),
                "--root", str(root),
                "run",
                "--phase", "compare",
                "--scenario", scenario,
                "--kernel", kernel,
                "--backend", backend,
                "--effective-kernel", effective_kernel,
                "--effective-backend", effective_backend,
                "--execution-mode", "native",
                "--reserve-gib", str(args.reserve_gib),
                "--",
                "python3", str(comparison),
                "--native-binary", str(args.native_binary.resolve()),
                "--oracle-helper", str(args.oracle_helper.resolve()),
                "--oracle-lock", str(args.oracle_lock.resolve()),
                "--oracle-preflight", str(preflight),
                "--repository", str(repository),
                "--model", str(args.model.resolve()),
                "--repack-cache", str(root / "cache"),
                "--fixtures", str(args.fixtures.resolve()),
                "--official-cache", str(official_cache_root / f"{scenario}.json"),
                "--scenario", scenario,
                "--kernel", kernel,
                "--backend", backend,
                "--max-new-tokens", str(args.max_new_tokens),
            ])
    return commands


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validation-binary", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--oracle-helper", type=Path, default=repository / "oracle/cpu_oracle.py")
    parser.add_argument("--oracle-lock", type=Path, default=repository / "oracle/cpu-oracle.lock.json")
    parser.add_argument("--model", type=Path, default=Path("/data/models/openai/gpt-oss-20b"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=repository / "crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--reserve-gib", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for index, command in enumerate(matrix_commands(args), 1):
        print(f"[{index}/42] {command[command.index('--scenario') + 1]} "
              f"{command[command.index('--kernel') + 1]}/"
              f"{command[command.index('--backend') + 1]}", flush=True)
        subprocess.run(command, cwd=args.repository, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
