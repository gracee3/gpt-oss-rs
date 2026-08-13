#!/usr/bin/env python3
"""Validate and deterministically summarize bounded CPU execution profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "gpt-oss-rs.execution-profile/v1"
SUMMARY_SCHEMA = "gpt-oss-rs.execution-profile-summary/v1"


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def load_profile(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema")
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: records must be a list")
    observed = hashlib.sha256(canonical(records)).hexdigest()
    if observed != value.get("records_sha256"):
        raise ValueError(f"{path}: record hash mismatch")
    if value.get("records_written") != len(records):
        raise ValueError(f"{path}: record count mismatch")
    if bool(value.get("truncated")) != (value.get("records_dropped", 0) != 0):
        raise ValueError(f"{path}: truncation metadata mismatch")
    return value


def summarize(paths: list[Path]) -> dict:
    profiles = [load_profile(path) for path in paths]
    if any(profile["truncated"] for profile in profiles):
        raise ValueError("truncated profiles are invalid for crossover selection")
    operation_ns: dict[str, int] = defaultdict(int)
    operation_count: Counter[str] = Counter()
    shapes: Counter[tuple] = Counter()
    buckets: dict[str, Counter[int]] = defaultdict(Counter)
    phase_count: Counter[str] = Counter()
    failed = 0
    scratch_high_water = 0
    resident_high_water = 0
    total_ns = 0
    for profile in profiles:
        for record in profile["records"]:
            if record["transaction_state"] == "failed":
                failed += 1
                continue
            operation = record["operation"]
            duration = int(record["duration_ns"])
            operation_ns[operation] += duration
            operation_count[operation] += 1
            total_ns += duration
            phase_count[record["phase"]] += 1
            shapes[(operation, record["m"], record["n"], record["k"], record["effective_matrix_backend"])] += 1
            if record["projection_role"] in ("gate_up", "down"):
                buckets[record["projection_role"]][int(record["expert_bucket_m"])] += 1
            scratch_high_water = max(scratch_high_water, int(record["scratch_high_water_bytes"]))
            resident_high_water = max(resident_high_water, int(record["resident_high_water_bytes"]))
    operations = [
        {
            "operation": operation,
            "records": operation_count[operation],
            "duration_ns": operation_ns[operation],
            "time_share": operation_ns[operation] / total_ns if total_ns else 0.0,
        }
        for operation in sorted(operation_ns)
    ]
    shape_rows = [
        {"operation": key[0], "m": key[1], "n": key[2], "k": key[3], "effective_matrix_backend_code": key[4], "count": count}
        for key, count in sorted(shapes.items())
    ]
    bucket_rows = {
        role: [{"m": m, "count": count} for m, count in sorted(counts.items())]
        for role, counts in sorted(buckets.items())
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ],
        "profile_count": len(profiles),
        "record_count": sum(len(profile["records"]) for profile in profiles),
        "failed_transaction_records": failed,
        "phase_records": dict(sorted(phase_count.items())),
        "scratch_high_water_bytes": scratch_high_water,
        "resident_high_water_bytes": resident_high_water,
        "operations": operations,
        "shapes": shape_rows,
        "expert_buckets": bucket_rows,
    }
    summary["summary_sha256"] = hashlib.sha256(canonical(summary)).hexdigest()
    return summary


def report(summary: dict) -> str:
    lines = [
        "CPU execution profile summary",
        f"profiles: {summary['profile_count']}",
        f"records: {summary['record_count']} (failed transaction: {summary['failed_transaction_records']})",
        f"scratch high-water: {summary['scratch_high_water_bytes']} bytes",
        "operation time shares:",
    ]
    for item in sorted(summary["operations"], key=lambda item: (-item["time_share"], item["operation"])):
        lines.append(f"  {item['operation']}: {item['time_share']:.3%} ({item['records']} records)")
    for role, values in summary["expert_buckets"].items():
        lines.append(f"{role} expert buckets: " + ", ".join(f"M={item['m']}:{item['count']}" for item in values))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    summary = summarize(args.inputs)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    rendered = report(summary)
    if args.report:
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
