#!/usr/bin/env python3
"""Run one bounded local 20B CPU service session and retain its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--service-probe-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repack-cache", type=Path, required=True)
    parser.add_argument("--served-model", default="gpt-oss-20b-cpu-validation")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser.parse_args()


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def request_json(url: str, payload: dict | None, timeout: float) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            body = json.load(error)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {"raw": error.read().decode(errors="replace")}
        return error.code, body


def wait_ready(origin: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with {process.returncode}")
        try:
            status, body = request_json(f"{origin}/ready", None, 5)
            last = {"status": status, "body": body}
            if status == 200:
                return body
        except (OSError, TimeoutError, json.JSONDecodeError) as error:
            last = {"error": f"{type(error).__name__}: {error}"}
        time.sleep(0.25)
    raise TimeoutError(f"bounded server did not become ready: {last}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    attempt_raw = os.environ.get("GPT_OSS_ATTEMPT_DIR")
    if not attempt_raw:
        raise RuntimeError("GPT_OSS_ATTEMPT_DIR is required")
    attempt = Path(attempt_raw).resolve()
    port = unused_port()
    origin = f"http://127.0.0.1:{port}"
    server_log = attempt / "server.log"
    evidence = attempt / "server-evidence"
    evidence.mkdir()
    command = [
        str(args.server_binary.resolve()),
        "serve",
        "--model",
        str(args.model.resolve()),
        "--served-model-name",
        args.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cpu",
        "--profile",
        "gpt-oss-cpu",
        "--cpu-kernel",
        "auto",
        "--cpu-matmul-backend",
        "auto",
        "--cpu-threads",
        "4",
        "--cpu-repack-cache",
        str(args.repack_cache.resolve()),
        "--runtime-mode",
        "experimental",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--max-prefill-chunk",
        "256",
        "--evidence-dir",
        str(evidence),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "RUST_LOG": "gpt_oss_server=info",
        }
    )
    started = time.monotonic()
    with server_log.open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            ready = wait_ready(origin, process, args.startup_timeout)
            ready_seconds = time.monotonic() - started
            probe_output = attempt / "service-probe.json"
            probe = subprocess.run(
                [
                    str(args.service_probe_binary.resolve()),
                    "--base-url",
                    origin,
                    "--served-model",
                    args.served_model,
                    "--output",
                    str(probe_output),
                ],
                capture_output=True,
            )
            (attempt / "service-probe.stdout").write_bytes(probe.stdout)
            (attempt / "service-probe.stderr").write_bytes(probe.stderr)
            if probe.returncode != 0:
                raise subprocess.CalledProcessError(probe.returncode, probe.args)
            request_started = time.monotonic()
            response_status, response = request_json(
                f"{origin}/v1/chat/completions",
                {
                    "model": args.served_model,
                    "messages": [{"role": "user", "content": "Reply with one word."}],
                    "max_tokens": 1,
                    "temperature": 0,
                    "stream": False,
                },
                args.request_timeout,
            )
            request_seconds = time.monotonic() - request_started
            if response_status != 200:
                raise RuntimeError(f"bounded inference returned HTTP {response_status}: {response}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    result = {
        "schema": "gpt-oss-rs.bounded-20b-service/v1",
        "status": "pass",
        "served_model": args.served_model,
        "server_sha256": sha256_file(args.server_binary),
        "model_manifest_sha256": sha256_file(
            args.model / "gpt-oss-rs-fetch-manifest.json"
        ),
        "ready_seconds": ready_seconds,
        "request_seconds": request_seconds,
        "ready": ready,
        "response_status": response_status,
        "response": response,
        "server_log_sha256": sha256_file(server_log),
        "service_probe_sha256": sha256_file(probe_output),
        "oracle_identity": json.loads(os.environ["GPT_OSS_ORACLE_IDENTITY_JSON"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
