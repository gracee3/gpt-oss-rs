#!/usr/bin/env python3
"""Materialize one verified scenario from a fresh seven-scenario llama capture."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


LLAMA_REVISION = "030ebb558a5820b444a8f836ed5cdd46c9b4bd7a"
SCENARIOS = {
    "harmony_63",
    "harmony_122",
    "harmony_136",
    "harmony_262",
    "harmony_346",
    "harmony_444",
    "tool_history_180",
}


def write_new_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = Path(os.environ["GPT_OSS_CAMPAIGN_ROOT"]).resolve()
    attempt = Path(os.environ["GPT_OSS_ATTEMPT_DIR"]).resolve()
    try:
        args.combined.resolve().relative_to(campaign)
    except ValueError as error:
        raise ValueError("combined llama capture must belong to the fresh campaign") from error
    output = (
        args.output.resolve()
        if args.output is not None
        else attempt / f"llama-{args.scenario}.json"
    )
    if output.parent != attempt:
        raise ValueError("selected llama output must stay in its campaign attempt")
    combined = json.loads(args.combined.read_text())
    identity = json.loads(os.environ["GPT_OSS_ORACLE_IDENTITY_JSON"])
    captures = combined.get("captures")
    if (
        combined.get("evidence_status") != "insufficient_evidence"
        or combined.get("llama_cpp_revision") != LLAMA_REVISION
        or combined.get("ubatch_size") != 1
        or combined.get("cache_prompt") is not False
        or combined.get("oracle_identity") != identity
        or identity.get("execution_mode") != "native"
        or not isinstance(captures, list)
        or {capture.get("scenario") for capture in captures} != SCENARIOS
        or len(captures) != len(SCENARIOS)
    ):
        raise ValueError("combined llama capture does not match the fresh campaign policy")
    selected = next(capture for capture in captures if capture["scenario"] == args.scenario)
    result = {
        "schema_version": 1,
        "evidence_status": "insufficient_evidence",
        "policy": "advisory",
        "scenario": args.scenario,
        "llama_cpp_revision": LLAMA_REVISION,
        "server_sha256": combined["server_sha256"],
        "ubatch_size": 1,
        "cache_prompt": False,
        "oracle_identity": identity,
        "capture": selected,
    }
    write_new_atomic(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
