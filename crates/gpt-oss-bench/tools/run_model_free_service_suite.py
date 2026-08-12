#!/usr/bin/env python3
"""Run the complete locked model-free workspace lifecycle and HTTP test suite."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    attempt_raw = os.environ.get("GPT_OSS_ATTEMPT_DIR")
    campaign_raw = os.environ.get("GPT_OSS_CAMPAIGN_ROOT")
    if not attempt_raw or not campaign_raw:
        raise RuntimeError("fresh campaign environment is required")
    attempt = Path(attempt_raw).resolve()
    repository = Path.cwd().resolve()
    command = ["cargo", "test", "--workspace", "--locked"]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=repository, capture_output=True)
    elapsed = time.monotonic() - started
    stdout_path = attempt / "workspace-tests.stdout"
    stderr_path = attempt / "workspace-tests.stderr"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    result = {
        "schema": "gpt-oss-rs.model-free-service-suite/v1",
        "status": "pass" if completed.returncode == 0 else "fail",
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "campaign_root": campaign_raw,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "oracle_identity": json.loads(os.environ["GPT_OSS_ORACLE_IDENTITY_JSON"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr[-64 * 1024 :])
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
