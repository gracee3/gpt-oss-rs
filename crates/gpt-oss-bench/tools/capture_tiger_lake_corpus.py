#!/usr/bin/env python3
"""Capture a bounded, hashed Tiger Lake CPU operation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

SCHEMA = "gpt-oss-rs.tiger-lake-corpus/v1"
SCENARIOS = (
    "harmony_63",
    "harmony_122",
    "harmony_136",
    "harmony_262",
    "harmony_346",
    "harmony_444",
    "tool_history_180",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def host_snapshot() -> dict:
    thermal = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        thermal[zone.name] = {
            "type": read_text(zone / "type"),
            "millidegrees_c": read_text(zone / "temp"),
        }
    cpufreq = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        root = cpu / "cpufreq"
        if root.is_dir():
            cpufreq[cpu.name] = {
                key: read_text(root / key)
                for key in (
                    "scaling_cur_freq",
                    "scaling_min_freq",
                    "scaling_max_freq",
                    "scaling_governor",
                )
            }
    power = {}
    for supply in sorted(Path("/sys/class/power_supply").glob("*")):
        power[supply.name] = {
            key: read_text(supply / key)
            for key in ("type", "online", "status", "capacity")
        }
    return {
        "unix_ns": time.time_ns(),
        "thermal": thermal,
        "cpufreq": cpufreq,
        "power": power,
    }


def git(args: list[str], repository: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()


def stable_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_artifact_index(root: Path) -> str:
    index = root / "SHA256SUMS"
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == index:
            continue
        entries.append(f"{sha256(path)}  {path.relative_to(root)}")
    index.write_text("\n".join(entries) + "\n")
    return sha256(index)


def command_for(args: argparse.Namespace, scenario: str, profile: Path, output: Path) -> list[str]:
    command = [
        "taskset", "-c", args.cpus,
        str(args.binary),
        "--model", str(args.model),
        "--repack-cache", str(args.repack_cache),
        "--fixtures", str(args.fixtures),
        "--scenario", scenario,
        "--kernel", args.kernel,
        "--cpu-matmul-backend", args.cpu_matmul_backend,
        "--expert-projection", "residual-q8",
        "--threads", str(args.threads),
        "--max-new-tokens", str(args.max_new_tokens),
        "--layer-major-prefill",
        "--cpu-profile-output", str(profile),
        "--cpu-profile-cap-mib", str(args.profile_cap_mib),
        "--output", str(output),
    ]
    if args.xe:
        command.extend([
            "--xe",
            "--xe-max-resident-mib", str(args.xe_max_resident_mib),
            "--xe-expert-cache-mib", str(args.xe_expert_cache_mib),
        ])
    return command


def capture(args: argparse.Namespace) -> str:
    root = args.output_root.resolve()
    if root.exists():
        raise ValueError(f"output root already exists: {root}")
    root.mkdir(parents=True)
    repository = args.repository.resolve()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.fixtures = args.fixtures.resolve()
    args.repack_cache = args.repack_cache.resolve()
    if not args.binary.is_file():
        raise ValueError(f"missing capture binary: {args.binary}")
    if shutil.which("taskset") is None or not Path("/usr/bin/time").is_file():
        raise ValueError("capture requires taskset and /usr/bin/time")

    source_commit = git(["rev-parse", "HEAD"], repository)
    dirty = bool(git(["status", "--porcelain"], repository))
    if dirty and not args.allow_dirty:
        raise ValueError("refusing a dirty source tree; pass --allow-dirty only for diagnostics")
    manifest = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "source_dirty": dirty,
        "binary": str(args.binary),
        "binary_sha256": sha256(args.binary),
        "model": str(args.model),
        "fixtures": str(args.fixtures),
        "fixtures_sha256": sha256(args.fixtures),
        "repack_cache": str(args.repack_cache),
        "scenarios": list(SCENARIOS),
        "warm_repetitions_per_scenario": args.repetitions,
        "warmup_repetitions_per_scenario": 1,
        "cpus": args.cpus,
        "threads": args.threads,
        "kernel": args.kernel,
        "cpu_matmul_backend": args.cpu_matmul_backend,
        "xe": args.xe,
        "xe_max_resident_mib": args.xe_max_resident_mib,
        "xe_expert_cache_mib": args.xe_expert_cache_mib,
        "max_new_tokens": args.max_new_tokens,
        "profile_cap_mib": args.profile_cap_mib,
        "platform": platform.platform(),
        "start_unix_ns": time.time_ns(),
        "host_start": host_snapshot(),
        "runs": [],
    }
    stable_write(root / "capture-manifest.in-progress.json", manifest)

    warm_profiles = []
    all_profiles = []
    for scenario in SCENARIOS:
        for repetition in range(args.repetitions + 1):
            state = "warmup" if repetition == 0 else "warm"
            name = f"{scenario}-{state}-{repetition:02d}"
            run_root = root / "runs" / name
            run_root.mkdir(parents=True)
            profile = run_root / "execution-profile.json"
            output = run_root / "cpu-parity.json"
            command = command_for(args, scenario, profile, output)
            timed_command = ["/usr/bin/time", "-v", "-o", str(run_root / "time.txt"), *command]
            before = host_snapshot()
            start_ns = time.monotonic_ns()
            with (run_root / "stdout.log").open("wb") as stdout, (run_root / "stderr.log").open("wb") as stderr:
                result = subprocess.run(timed_command, cwd=repository, stdout=stdout, stderr=stderr)
            duration_ns = time.monotonic_ns() - start_ns
            after = host_snapshot()
            run = {
                "scenario": scenario,
                "repetition": repetition,
                "cache_state": state,
                "command": command,
                "returncode": result.returncode,
                "wall_duration_ns": duration_ns,
                "host_before": before,
                "host_after": after,
                "profile": str(profile.relative_to(root)),
                "output": str(output.relative_to(root)),
            }
            stable_write(run_root / "run.json", run)
            manifest["runs"].append(run)
            stable_write(root / "capture-manifest.in-progress.json", manifest)
            if result.returncode != 0:
                write_artifact_index(root)
                raise RuntimeError(f"capture failed for {name}; see {run_root / 'stderr.log'}")
            all_profiles.append(profile)
            if state == "warm":
                warm_profiles.append(profile)

    for label, profiles in (("all", all_profiles), ("warm", warm_profiles)):
        command = [
            str(args.summarizer),
            *map(str, profiles),
            "--output", str(root / f"summary-{label}.json"),
            "--report", str(root / f"summary-{label}.txt"),
        ]
        subprocess.run(command, cwd=repository, check=True)
    manifest["end_unix_ns"] = time.time_ns()
    manifest["host_end"] = host_snapshot()
    (root / "capture-manifest.in-progress.json").unlink()
    stable_write(root / "capture-manifest.json", manifest)
    return write_artifact_index(root)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=repository / "crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json")
    parser.add_argument("--repack-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, default=Path(__file__).with_name("summarize_cpu_profile.py"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--profile-cap-mib", type=int, default=16)
    parser.add_argument("--cpus", default="0-3")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kernel", default="auto")
    parser.add_argument("--cpu-matmul-backend", default="auto")
    parser.add_argument("--xe", action="store_true")
    parser.add_argument("--xe-max-resident-mib", type=int, default=128)
    parser.add_argument("--xe-expert-cache-mib", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.threads < 1 or args.profile_cap_mib < 1:
        parser.error("repetitions, threads, and profile capacity must be positive")
    if args.xe_max_resident_mib < 1 or args.xe_expert_cache_mib < 0:
        parser.error("Xe resident capacity must be positive and expert cache must be non-negative")
    if args.xe_expert_cache_mib and not args.xe:
        parser.error("--xe-expert-cache-mib requires --xe")
    return args


def main() -> int:
    try:
        digest = capture(parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    print(f"artifact_index_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
