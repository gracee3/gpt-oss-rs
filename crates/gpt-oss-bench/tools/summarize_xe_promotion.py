#!/usr/bin/env python3
"""Summarize paired CPU/Xe full-model promotion measurements."""

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path


BOOTSTRAP_SEED = 20260812
BOOTSTRAP_SAMPLES = 10_000
SAMPLE_RE = re.compile(r"^(\d{2})-(cpu|xe)\.json$")
EXPECTED_SAMPLE_IDS = [f"{sample:02d}" for sample in range(1, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        action="append",
        required=True,
        help="directory containing NN-cpu.json and NN-xe.json captures",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("ratios must be positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_interval(
    ratios: list[float], samples: int, seed: int
) -> dict[str, float]:
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    rng = random.Random(seed)
    estimates = [
        geometric_mean([ratios[rng.randrange(len(ratios))] for _ in ratios])
        for _ in range(samples)
    ]
    return {
        "estimate": geometric_mean(ratios),
        "lower_95": percentile(estimates, 0.025),
        "upper_95": percentile(estimates, 0.975),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p95(values: list[float]) -> float:
    return percentile(values, 0.95)


def decode_throughput(capture: dict) -> float:
    arrivals = capture["token_arrival_seconds"]
    if len(arrivals) < 2 or arrivals[-1] <= arrivals[0]:
        raise ValueError("capture does not contain a measurable decode interval")
    return (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])


def parse_time(path: Path) -> dict[str, int]:
    fields = {
        "maximum_resident_kib": "Maximum resident set size (kbytes)",
        "major_page_faults": "Major (requiring I/O) page faults",
        "swaps": "Swaps",
    }
    parsed = {}
    lines = path.read_text().splitlines()
    for output_name, label in fields.items():
        match = next((line for line in lines if line.strip().startswith(label + ":")), None)
        if match is None:
            raise ValueError(f"{path} is missing {label!r}")
        parsed[output_name] = int(match.rsplit(":", 1)[1].strip())
    return parsed


def load_pairs(directory: Path) -> list[dict]:
    indexed: dict[str, dict[str, Path]] = {}
    for path in directory.glob("??-*.json"):
        match = SAMPLE_RE.match(path.name)
        if match:
            indexed.setdefault(match.group(1), {})[match.group(2)] = path
    if not indexed:
        raise ValueError(f"{directory} has no paired captures")
    pairs = []
    xe_identity = None
    source_identity = None
    for sample_id in sorted(indexed):
        paths = indexed[sample_id]
        if set(paths) != {"cpu", "xe"}:
            raise ValueError(f"sample {sample_id} is incomplete")
        cpu = json.loads(paths["cpu"].read_text())
        xe = json.loads(paths["xe"].read_text())
        if cpu["scenario"] != xe["scenario"]:
            raise ValueError(f"sample {sample_id} scenario mismatch")
        expected = cpu["expected_official_greedy_tokens"]
        if cpu["generated_token_ids"] != expected or xe["generated_token_ids"] != expected:
            raise ValueError(f"sample {sample_id} does not match the official oracle")
        if cpu["generated_token_ids"] != xe["generated_token_ids"]:
            raise ValueError(f"sample {sample_id} CPU/Xe token mismatch")
        if xe.get("xe", {}).get("effective_backend") != "cpu_xe":
            raise ValueError(f"sample {sample_id} is not an effective Xe capture")
        if cpu.get("xe") is not None:
            raise ValueError(f"sample {sample_id} CPU capture unexpectedly attached Xe")
        current_identity = {
            key: xe["xe"][key]
            for key in (
                "identity",
                "source_sha256",
                "abi_sha256",
                "build_options",
                "gate_up_min_rows",
                "down_min_rows",
                "workgroup_size",
            )
        }
        if xe_identity is None:
            xe_identity = current_identity
        elif current_identity != xe_identity:
            raise ValueError(f"sample {sample_id} Xe identity or policy drifted")
        cpu_time_path = paths["cpu"].with_suffix(".time")
        xe_time_path = paths["xe"].with_suffix(".time")
        cpu_manifest_path = paths["cpu"].with_suffix(".json.manifest.json")
        xe_manifest_path = paths["xe"].with_suffix(".json.manifest.json")
        cpu_manifest = json.loads(cpu_manifest_path.read_text())
        xe_manifest = json.loads(xe_manifest_path.read_text())
        pair_source = None
        for side, capture_path, manifest in (
            ("CPU", paths["cpu"], cpu_manifest),
            ("Xe", paths["xe"], xe_manifest),
        ):
            if manifest.get("status") != "pass":
                raise ValueError(f"sample {sample_id} {side} manifest is not passing")
            artifact_hashes = {
                artifact["sha256"] for artifact in manifest.get("artifacts", [])
            }
            if sha256(capture_path) not in artifact_hashes:
                raise ValueError(f"sample {sample_id} {side} manifest does not hash its capture")
            if pair_source is None:
                pair_source = manifest["source"]
            elif manifest["source"] != pair_source:
                raise ValueError(f"sample {sample_id} CPU/Xe source provenance differs")
            comparable_source = {
                key: value
                for key, value in manifest["source"].items()
                if key != "cargo_lock_sha256"
            }
            if source_identity is None:
                source_identity = comparable_source
            elif comparable_source != source_identity:
                raise ValueError(f"sample {sample_id} executable source identity drifted")
        pairs.append(
            {
                "id": sample_id,
                "cpu": cpu,
                "xe": xe,
                "cpu_time": parse_time(cpu_time_path),
                "xe_time": parse_time(xe_time_path),
                "artifacts": {
                    "cpu_capture_sha256": sha256(paths["cpu"]),
                    "cpu_manifest_sha256": sha256(cpu_manifest_path),
                    "xe_capture_sha256": sha256(paths["xe"]),
                    "xe_manifest_sha256": sha256(xe_manifest_path),
                },
                "source": pair_source,
            }
        )
    actual_sample_ids = [pair["id"] for pair in pairs]
    if actual_sample_ids != EXPECTED_SAMPLE_IDS:
        raise ValueError(
            "paired gate requires exactly samples 01 through 10; "
            f"found {actual_sample_ids}"
        )
    return pairs


def summarize_scenario(directory: Path, samples: int, seed: int) -> dict:
    pairs = load_pairs(directory)
    metrics = {
        "ttft_cpu_over_xe": [
            pair["cpu"]["time_to_first_token_seconds"]
            / pair["xe"]["time_to_first_token_seconds"]
            for pair in pairs
        ],
        "full_request_cpu_over_xe": [
            pair["cpu"]["full_request_seconds"]
            / pair["xe"]["full_request_seconds"]
            for pair in pairs
        ],
        "decode_throughput_xe_over_cpu": [
            decode_throughput(pair["xe"]) / decode_throughput(pair["cpu"])
            for pair in pairs
        ],
        "p95_inter_token_xe_over_cpu": [
            p95(pair["xe"]["inter_token_seconds"])
            / p95(pair["cpu"]["inter_token_seconds"])
            for pair in pairs
        ],
    }
    intervals = {
        name: bootstrap_interval(ratios, samples, seed + index)
        for index, (name, ratios) in enumerate(metrics.items())
    }
    descriptor = pairs[0]["xe"]["xe"]["memory"]
    declared_combined_bound = (
        descriptor["max_resident_bytes"] + descriptor["host_staging_bound_bytes"]
    )
    rss_deltas = [
        (pair["xe_time"]["maximum_resident_kib"] - pair["cpu_time"]["maximum_resident_kib"])
        * 1024
        for pair in pairs
    ]
    all_swaps = [
        pair[side + "_time"]["swaps"] for pair in pairs for side in ("cpu", "xe")
    ]
    post_first_major_faults = [
        pair[side + "_time"]["major_page_faults"]
        for pair in pairs[1:]
        for side in ("cpu", "xe")
    ]
    memory = {
        "declared_device_plus_host_bound_bytes": declared_combined_bound,
        "maximum_paired_xe_minus_cpu_rss_bytes": max(rss_deltas),
        "minimum_paired_xe_minus_cpu_rss_bytes": min(rss_deltas),
        "maximum_swap_count": max(all_swaps),
        "maximum_cpu_major_page_faults": max(
            pair["cpu_time"]["major_page_faults"] for pair in pairs
        ),
        "maximum_xe_major_page_faults": max(
            pair["xe_time"]["major_page_faults"] for pair in pairs
        ),
        "first_pair_cpu_major_page_faults": pairs[0]["cpu_time"]["major_page_faults"],
        "first_pair_xe_major_page_faults": pairs[0]["xe_time"]["major_page_faults"],
        "maximum_post_first_pair_major_page_faults": max(post_first_major_faults),
    }
    gates = {
        "tokens_and_official_oracle": True,
        "ttft_lower_bound_above_one": intervals["ttft_cpu_over_xe"]["lower_95"] > 1.0,
        "full_request_lower_bound_above_one": intervals[
            "full_request_cpu_over_xe"
        ]["lower_95"]
        > 1.0,
        "decode_throughput_lower_bound_at_least_0_98": intervals[
            "decode_throughput_xe_over_cpu"
        ]["lower_95"]
        >= 0.98,
        "p95_inter_token_upper_bound_at_most_1_02": intervals[
            "p95_inter_token_xe_over_cpu"
        ]["upper_95"]
        <= 1.02,
        "rss_delta_within_declared_bound": max(rss_deltas) <= declared_combined_bound,
        "no_swap_growth": max(all_swaps) == 0,
        "no_post_first_pair_major_faults": max(post_first_major_faults) == 0,
    }
    return {
        "scenario": pairs[0]["cpu"]["scenario"],
        "source_variants": [
            json.loads(value)
            for value in sorted(
                {
                    json.dumps(pair["source"], sort_keys=True)
                    for pair in pairs
                }
            )
        ],
        "pair_count": len(pairs),
        "intervals": intervals,
        "memory": memory,
        "gates": gates,
        "passing": all(gates.values()),
        "pairs": [
            {
                "id": pair["id"],
                "ratios": {name: values[index] for name, values in metrics.items()},
                "cpu_time": pair["cpu_time"],
                "xe_time": pair["xe_time"],
                "artifacts": pair["artifacts"],
            }
            for index, pair in enumerate(pairs)
        ],
    }


def main() -> int:
    args = parse_args()
    scenarios = [
        summarize_scenario(directory, args.bootstrap_samples, args.seed + 100 * index)
        for index, directory in enumerate(args.scenario_dir)
    ]
    result = {
        "schema": "gpt-oss-rs.xe-paired-promotion/v1",
        "bootstrap": {
            "statistic": "paired geometric mean ratio",
            "confidence_interval": "percentile 95%",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "scenarios": scenarios,
        "runner": {
            "path": str(args.runner),
            "sha256": sha256(args.runner),
        }
        if args.runner
        else None,
        "automatic_performance_gate": "pass"
        if all(scenario["passing"] for scenario in scenarios)
        else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["automatic_performance_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
