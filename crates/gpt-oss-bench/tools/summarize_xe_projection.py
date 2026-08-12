#!/usr/bin/env python3
"""Summarize transfer-inclusive real-tensor CPU/Xe projection captures."""

import argparse
import json
import math
import random
from pathlib import Path


SEED = 20260812
BOOTSTRAP_SAMPLES = 10_000


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def bootstrap_ratio(
    baseline: list[float], xe: list[float], samples: int, seed: int
) -> dict[str, float]:
    if not baseline or not xe or samples < 1:
        raise ValueError("non-empty samples and a positive bootstrap count are required")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        left = median([baseline[rng.randrange(len(baseline))] for _ in baseline])
        right = median([xe[rng.randrange(len(xe))] for _ in xe])
        estimates.append(left / right)
    return {
        "estimate": median(baseline) / median(xe),
        "lower_95": percentile(estimates, 0.025),
        "upper_95": percentile(estimates, 0.975),
    }


def summarize(capture: dict, samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED) -> dict:
    if capture.get("status") != "pass":
        raise ValueError("projection correctness capture is not passing")
    reports = []
    role_pass: dict[str, bool] = {}
    for index, report in enumerate(capture["reports"]):
        grouped: dict[str, list[float]] = {}
        for sample in report["samples"]:
            grouped.setdefault(sample["method"], []).append(sample["total_ns"])
        expected = {"scalar", "cpu_auto", "avx2", "xe"}
        if set(grouped) != expected or any(len(values) != 90 for values in grouped.values()):
            raise ValueError("each method must contain three trials of thirty samples")
        ratios = {
            "scalar_over_xe": bootstrap_ratio(
                grouped["scalar"], grouped["xe"], samples, seed + index * 10
            ),
            "cpu_auto_over_xe": bootstrap_ratio(
                grouped["cpu_auto"], grouped["xe"], samples, seed + index * 10 + 1
            ),
            "avx2_over_xe": bootstrap_ratio(
                grouped["avx2"], grouped["xe"], samples, seed + index * 10 + 2
            ),
        }
        passing = ratios["cpu_auto_over_xe"]["lower_95"] > 1.0
        role = report["projection"]
        role_pass[role] = role_pass.get(role, True) and passing
        reports.append(
            {
                "projection": role,
                "rows": report["rows"],
                "sample_count_per_method": len(grouped["xe"]),
                "median_ns": {name: median(values) for name, values in grouped.items()},
                "ratios": ratios,
                "automatic_bucket_gate": "pass" if passing else "fail",
            }
        )
    decisions = {
        role: {
            "automatic_projection_gate": "pass" if passing else "fail",
            "selected_min_rows": 4 if passing else None,
        }
        for role, passing in sorted(role_pass.items())
    }
    return {
        "schema": "gpt-oss-rs.xe-transfer-inclusive-projection-summary/v1",
        "bootstrap": {
            "statistic": "independently resampled ratio of medians",
            "confidence_interval": "percentile 95%",
            "samples": samples,
            "seed": seed,
        },
        "source": capture["source"],
        "decisions": decisions,
        "automatic_projection_gate": "pass"
        if all(role_pass.values())
        else "fail",
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()), args.bootstrap_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["automatic_projection_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
