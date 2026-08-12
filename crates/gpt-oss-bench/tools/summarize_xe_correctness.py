#!/usr/bin/env python3
"""Validate and index clean CPU/Xe full-model Harmony correctness captures."""

import argparse
import hashlib
import json
import math
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_fields_are_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(numeric_fields_are_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(numeric_fields_are_finite(item) for item in value)
    return True


def validate_manifest(capture_path: Path) -> dict:
    path = capture_path.with_suffix(".json.manifest.json")
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "pass":
        raise ValueError(f"{path} is not passing")
    hashes = {artifact["sha256"] for artifact in manifest.get("artifacts", [])}
    if sha256(capture_path) not in hashes:
        raise ValueError(f"{path} does not hash {capture_path.name}")
    source = manifest["source"]
    if source.get("dirty") is not False:
        raise ValueError(f"{path} is not a clean-source capture")
    if source.get("features") not in ([], ["xe"]):
        raise ValueError(f"{path} has an unexpected feature set")
    return {"path": path, "sha256": sha256(path), "source": source}


def summarize(root: Path, runner: Path, promotion_record: Path) -> dict:
    runner_hash = sha256(runner)
    record = json.loads(promotion_record.read_text())
    scenarios = []
    source_identity = None
    for scenario in SCENARIOS:
        paths = {
            side: root / f"{scenario}-{side}.json" for side in ("cpu", "xe")
        }
        captures = {side: json.loads(path.read_text()) for side, path in paths.items()}
        cpu, xe = captures["cpu"], captures["xe"]
        if cpu.get("scenario") != scenario or xe.get("scenario") != scenario:
            raise ValueError(f"{scenario} capture name and scenario disagree")
        expected = cpu.get("expected_official_greedy_tokens")
        if not expected or xe.get("expected_official_greedy_tokens") != expected:
            raise ValueError(f"{scenario} official oracle differs between captures")
        if cpu.get("generated_token_ids") != expected or xe.get("generated_token_ids") != expected:
            raise ValueError(f"{scenario} does not match the official oracle")
        if cpu.get("xe") is not None:
            raise ValueError(f"{scenario} CPU capture unexpectedly attached Xe")
        descriptor = xe.get("xe") or {}
        identity = descriptor.get("identity") or {}
        expected_descriptor = {
            "effective_backend": "cpu_xe",
            "validation_class": "validated_explicit",
            "source_sha256": record["kernel_source_sha256"],
            "abi_sha256": record["kernel_abi_sha256"],
            "build_options": record["build_options"],
            "gate_up_min_rows": record["gate_up_min_rows"],
            "down_min_rows": record["down_min_rows"],
            "workgroup_size": record["workgroup_size"],
        }
        for key, expected_value in expected_descriptor.items():
            if descriptor.get(key) != expected_value:
                raise ValueError(f"{scenario} Xe descriptor {key} drifted")
        expected_identity = {
            "pci_vendor_id": record["pci_vendor_id"],
            "pci_device_id": record["pci_device_id"],
            "driver_version": record["driver_version"],
            "opencl_loader_sha256": record["opencl_loader_sha256"],
            "opencl_driver_sha256": record["opencl_driver_sha256"],
            "igc_sha256": record["igc_sha256"],
        }
        for key, expected_value in expected_identity.items():
            if identity.get(key) != expected_value:
                raise ValueError(f"{scenario} Xe identity {key} drifted")
        memory = descriptor.get("memory") or {}
        if memory.get("device_resident_bytes", 1) > memory.get("max_resident_bytes", 0):
            raise ValueError(f"{scenario} Xe device residency exceeds its bound")
        for side, capture in captures.items():
            if capture.get("executable_sha256") != runner_hash:
                raise ValueError(f"{scenario} {side} executable hash drifted")
            if not numeric_fields_are_finite(capture):
                raise ValueError(f"{scenario} {side} has a non-finite numeric field")
        manifests = {side: validate_manifest(path) for side, path in paths.items()}
        pair_source = manifests["cpu"]["source"]
        if manifests["xe"]["source"] != pair_source:
            raise ValueError(f"{scenario} CPU/Xe source provenance differs")
        if source_identity is None:
            source_identity = pair_source
        elif pair_source != source_identity:
            raise ValueError(f"{scenario} clean source provenance drifted")
        scenarios.append(
            {
                "scenario": scenario,
                "official_and_cpu_xe_tokens": expected,
                "artifacts": {
                    side: {
                        "capture_sha256": sha256(paths[side]),
                        "manifest_sha256": manifests[side]["sha256"],
                    }
                    for side in ("cpu", "xe")
                },
            }
        )
    return {
        "schema": "gpt-oss-rs.xe-full-model-correctness/v1",
        "status": "pass",
        "scenario_count": len(scenarios),
        "tokens_match_cpu_xe_and_official_oracle": True,
        "recorded_numeric_fields_finite": True,
        "xe_identity_and_policy_exact": True,
        "source": source_identity,
        "runner": {"path": str(runner), "sha256": runner_hash},
        "validated_promotion_schema": record["schema"],
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--promotion-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.runner, args.promotion_record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
