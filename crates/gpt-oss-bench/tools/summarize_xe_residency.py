#!/usr/bin/env python3
"""Deterministically summarize explicit-Xe residency captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


SCHEMA = "gpt-oss-rs.xe-residency-summary/v1"
PROJECTION_OPERATIONS = {"gate_up_projection", "down_projection"}


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projection_upload_bytes(record: dict) -> int:
    return int(record["n"]) * (int(record["k"]) // 32) * 17 + int(record["n"]) * 4


def load_capture(path: Path) -> tuple[dict, dict, Path]:
    capture = json.loads(path.read_text())
    profile_path = path.with_name("execution-profile.json")
    if not profile_path.is_file():
        raise ValueError(f"{path}: missing execution-profile.json")
    profile = json.loads(profile_path.read_text())
    if profile.get("schema") != "gpt-oss-rs.execution-profile/v1":
        raise ValueError(f"{profile_path}: unsupported profile schema")
    if profile.get("truncated") or profile.get("records_dropped"):
        raise ValueError(f"{profile_path}: truncated profile")
    if not capture.get("xe") or capture.get("xe_residency") is None:
        raise ValueError(f"{path}: missing Xe descriptor or residency statistics")
    return capture, profile, profile_path


def summarize(paths: list[Path]) -> dict:
    groups: dict[int, list[tuple[Path, dict, dict, Path]]] = defaultdict(list)
    for path in sorted(path.resolve() for path in paths):
        capture, profile, profile_path = load_capture(path)
        capacity = int(capture["xe"]["memory"]["expert_cache_capacity_bytes"])
        groups[capacity].append((path, capture, profile, profile_path))

    capacities = []
    inputs = []
    for capacity, rows in sorted(groups.items()):
        counters = defaultdict(int)
        projection_ns = []
        full_request_seconds = []
        scenarios = []
        estimated_upload_bytes = 0
        for path, capture, profile, profile_path in rows:
            inputs.extend([
                {"path": str(path), "sha256": sha256(path)},
                {"path": str(profile_path), "sha256": sha256(profile_path)},
            ])
            scenarios.append(capture["scenario"])
            full_request_seconds.append(float(capture["full_request_seconds"]))
            for key, value in capture["xe_residency"].items():
                if key != "capacity_bytes":
                    counters[key] += int(value)
            for record in profile["records"]:
                if record["operation"] not in PROJECTION_OPERATIONS:
                    continue
                projection_ns.append(int(record["duration_ns"]))
                if record["residency_state"] != "hit":
                    estimated_upload_bytes += projection_upload_bytes(record)
        lookups = counters["hits"] + counters["misses"]
        capacities.append({
            "capacity_bytes": capacity,
            "captures": len(rows),
            "scenarios": sorted(scenarios),
            "hits": counters["hits"],
            "misses": counters["misses"],
            "hit_rate": counters["hits"] / lookups if lookups else 0.0,
            "bypasses": counters["bypasses"],
            "evictions": counters["evictions"],
            "faults": counters["faults"],
            "resident_high_water_bytes": max(
                int(capture["xe_residency"]["resident_high_water_bytes"])
                for _, capture, _, _ in rows
            ),
            "repacks_avoided": counters["repacks_avoided"],
            "upload_bytes_avoided": counters["upload_bytes_avoided"],
            "cache_insert_uploaded_bytes": counters["uploaded_bytes"],
            "estimated_total_uploaded_bytes": estimated_upload_bytes,
            "projection_median_ns": statistics.median(projection_ns),
            "full_request_median_seconds": statistics.median(full_request_seconds),
        })
    result = {
        "schema": SCHEMA,
        "inputs": sorted(inputs, key=lambda row: row["path"]),
        "capacities": capacities,
    }
    result["summary_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def report(value: dict) -> str:
    lines = ["Xe expert-residency summary"]
    for row in value["capacities"]:
        lines.append(
            f"{row['capacity_bytes'] // (1024 * 1024)} MiB: "
            f"hit-rate={row['hit_rate']:.3f}, evictions={row['evictions']}, "
            f"upload-avoided={row['upload_bytes_avoided']}, "
            f"request-median={row['full_request_median_seconds']:.3f}s"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    value = summarize(args.inputs)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    rendered = report(value)
    if args.report:
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
