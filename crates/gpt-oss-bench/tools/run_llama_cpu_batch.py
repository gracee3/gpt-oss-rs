#!/usr/bin/env python3
"""Create the one fresh seven-scenario pinned llama.cpp capture batch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-tool", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-capture", type=Path, action="append", required=True)
    parser.add_argument("--published-output", type=Path, required=True)
    args = parser.parse_args()
    attempt_raw = os.environ.get("GPT_OSS_ATTEMPT_DIR")
    campaign_raw = os.environ.get("GPT_OSS_CAMPAIGN_ROOT")
    if not attempt_raw or not campaign_raw:
        raise RuntimeError("fresh campaign environment is required")
    attempt = Path(attempt_raw).resolve()
    campaign = Path(campaign_raw).resolve()
    identity = json.loads(os.environ["GPT_OSS_ORACLE_IDENTITY_JSON"])
    expected_output = campaign / "raw" / "llama" / "combined.json"
    if args.published_output.resolve() != expected_output:
        raise RuntimeError(f"published llama output must be {expected_output}")
    if args.build.resolve() != campaign / "build" / "llama.cpp":
        raise RuntimeError("llama build must use the fresh campaign build directory")
    native_values = []
    for capture in args.native_capture:
        resolved = capture.resolve()
        try:
            resolved.relative_to(campaign / "attempts")
        except ValueError as error:
            raise RuntimeError("llama inputs must be fresh campaign attempts") from error
        value = json.loads(resolved.read_text())
        if value.get("oracle_identity") != identity:
            raise RuntimeError("llama input uses a different oracle identity")
        native_values.append(value)
    if len(native_values) != 7 or len({value.get("scenario") for value in native_values}) != 7:
        raise RuntimeError("llama batch requires seven distinct fresh scenario captures")
    output = attempt / "llama-combined.json"
    command = [
        sys.executable,
        str(args.capture_tool.resolve()),
        "--source",
        str(args.source.resolve()),
        "--build",
        str(args.build.resolve()),
        "--model",
        str(args.model.resolve()),
        "--output",
        str(output),
        "--max-new-tokens",
        "8",
        "--threads",
        "4",
    ]
    for capture in args.native_capture:
        command.extend(["--native-capture", str(capture.resolve())])
    completed = subprocess.run(command, capture_output=True)
    (attempt / "llama-worker.stdout").write_bytes(completed.stdout)
    (attempt / "llama-worker.stderr").write_bytes(completed.stderr)
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr)
        return completed.returncode
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    os.link(output, expected_output)
    print(json.dumps(json.loads(output.read_text()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
