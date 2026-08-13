#!/usr/bin/env python3
"""Analyze paired MXFP4 matrix samples and select proven contiguous regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

SCHEMA = "gpt-oss-rs.mxfp4-matrix-benchmark/v1"
ANALYSIS_SCHEMA = "gpt-oss-rs.mxfp4-matrix-promotion-analysis/v1"
EXPLICIT_METHODS = ("scalar", "avx2", "avx512-vnni")


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported benchmark schema")
    if value.get("repository_dirty"):
        raise ValueError(f"{path}: dirty benchmark source")
    if value.get("trials", 0) < 7:
        raise ValueError(f"{path}: fewer than seven trials")
    if value.get("trials", 0) * value.get("samples_per_trial", 0) < 30:
        raise ValueError(f"{path}: fewer than 30 samples per method/shape")
    correctness = {(row["m"], row["method"]): row for row in value["correctness"]}
    for sample in value["samples"]:
        row = correctness.get((sample["m"], sample["method"]))
        if not row or not row.get("scalar_exact"):
            raise ValueError(f"{path}: sample lacks exact scalar certification")
        if row["output_sha256"] != sample["output_sha256"]:
            raise ValueError(f"{path}: output identity drift")
    return value


def paired_interval(candidate: list[int], comparator: list[int], iterations: int, seed: int) -> dict:
    if len(candidate) != len(comparator) or not candidate:
        raise ValueError("paired samples are missing")
    differences = [left - right for left, right in zip(candidate, comparator)]
    rng = random.Random(seed)
    medians = []
    for _ in range(iterations):
        medians.append(statistics.median(rng.choices(differences, k=len(differences))))
    medians.sort()
    low = medians[int(iterations * 0.025)]
    high = medians[min(iterations - 1, int(iterations * 0.975))]
    candidate_median = statistics.median(candidate)
    comparator_median = statistics.median(comparator)
    return {
        "paired_median_difference_ns": statistics.median(differences),
        "paired_95ci_ns": [low, high],
        "candidate_median_ns": candidate_median,
        "comparator_median_ns": comparator_median,
        "median_ratio": candidate_median / comparator_median,
        "proven_lower_latency": high < 0,
    }


def contiguous(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    regions = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            regions.append([start, previous])
            start = value
        previous = value
    regions.append([start, previous])
    return regions


def analyze(paths: list[Path], iterations: int = 10_000) -> dict:
    documents = [load(path) for path in paths]
    commits = {value["repository_commit"] for value in documents}
    cpus = {canonical(value["cpu_identity"]) for value in documents}
    executables = {value["executable_sha256"] for value in documents}
    if len(commits) != 1 or len(cpus) != 1 or len(executables) != 1:
        raise ValueError("benchmark inputs do not share commit, CPU, and executable identity")

    grouped: dict[tuple, dict[str, dict[tuple[int, int], int]]] = defaultdict(lambda: defaultdict(dict))
    for value in documents:
        for sample in value["samples"]:
            key = (value["activation"], sample["n"], sample["k"], sample["m"])
            sample_key = (sample["trial"], sample["sample"])
            grouped[key][sample["method"]][sample_key] = int(sample["duration_ns"])

    comparisons = []
    winning = defaultdict(list)
    for key in sorted(grouped):
        activation, n, k, m = key
        methods = grouped[key]
        for candidate in ("avx2", "avx512-vnni"):
            if candidate not in methods:
                continue
            outcomes = []
            for comparator in EXPLICIT_METHODS:
                if comparator == candidate or comparator not in methods:
                    continue
                shared = sorted(set(methods[candidate]) & set(methods[comparator]))
                seed_material = canonical([key, candidate, comparator])
                interval = paired_interval(
                    [methods[candidate][item] for item in shared],
                    [methods[comparator][item] for item in shared],
                    iterations,
                    int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"),
                )
                outcomes.append({"comparator": comparator, **interval})
            proven = bool(outcomes) and all(item["proven_lower_latency"] for item in outcomes)
            comparisons.append({
                "activation": activation,
                "m": m,
                "n": n,
                "k": k,
                "candidate": candidate,
                "comparisons": outcomes,
                "qualifies": proven,
            })
            if proven:
                winning[(activation, n, k, candidate)].append(m)

    regions = []
    for (activation, n, k, candidate), values in sorted(winning.items()):
        for start, end in contiguous(sorted(values)):
            regions.append({
                "activation": activation,
                "n": n,
                "k": k,
                "candidate": candidate,
                "m_start": start,
                "m_end": end,
            })
    result = {
        "schema": ANALYSIS_SCHEMA,
        "repository_commit": next(iter(commits)),
        "cpu_identity": documents[0]["cpu_identity"],
        "executable_sha256": next(iter(executables)),
        "bootstrap_iterations": iterations,
        "selection_rule": (
            "paired bootstrap 95% CI upper bound for median latency difference is below zero "
            "against scalar and every other legal explicit candidate"
        ),
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ],
        "comparisons": comparisons,
        "candidate_regions": regions,
        "promotion_status": "positive" if regions else "negative",
        "fallback": "scalar for ties, uncertain intervals, gaps, unobserved shapes, or profile mismatch",
    }
    result["analysis_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def report(value: dict) -> str:
    lines = [
        "MXFP4 matrix promotion analysis",
        f"commit: {value['repository_commit']}",
        f"status: {value['promotion_status']}",
    ]
    if value["candidate_regions"]:
        lines.append("proven candidate regions:")
        for region in value["candidate_regions"]:
            lines.append(
                f"  {region['activation']} N={region['n']} K={region['k']} "
                f"M={region['m_start']}..{region['m_end']}: {region['candidate']}"
            )
    else:
        lines.append("proven candidate regions: none; Auto remains scalar for M>1")
    uncertain = sum(not row["qualifies"] for row in value["comparisons"])
    lines.append(f"non-qualifying candidate/shape rows: {uncertain}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()
    value = analyze(args.inputs, args.bootstrap_iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    rendered = report(value)
    if args.report:
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
