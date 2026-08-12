#!/usr/bin/env python3
"""Build pinned llama.cpp out of tree and capture seven CPU-only token fixtures."""

import argparse
import json
import os
import socket
import subprocess
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


LLAMA_REVISION = "030ebb558a5820b444a8f836ed5cdd46c9b4bd7a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=0)
    return parser.parse_args()


def command_text(*command: str) -> str:
    return subprocess.check_output(command, text=True).strip()


def validate(args: argparse.Namespace) -> list[dict]:
    revision = command_text("git", "-C", str(args.source), "rev-parse", "HEAD")
    if revision != LLAMA_REVISION:
        raise RuntimeError(f"llama.cpp revision {revision} does not match {LLAMA_REVISION}")
    if command_text("git", "-C", str(args.source), "status", "--porcelain"):
        raise RuntimeError("llama.cpp checkout is dirty")
    try:
        args.build.resolve().relative_to(args.source.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("llama.cpp build directory must be outside its source tree")
    captures = [json.loads(path.read_text()) for path in args.native_capture]
    if len(captures) != 7:
        raise RuntimeError(f"exactly seven native captures are required, observed {len(captures)}")
    scenarios = [capture.get("scenario") for capture in captures]
    if len(set(scenarios)) != 7:
        raise RuntimeError("native captures must contain seven unique scenarios")
    for capture in captures:
        tokens = capture.get("prompt_token_ids")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(token, int) for token in tokens):
            raise RuntimeError(f"scenario {capture.get('scenario')} has no exact token-ID fixture")
    return captures


def build_server(args: argparse.Namespace) -> Path:
    args.build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(args.source),
            "-B",
            str(args.build),
            "-DGGML_CUDA=OFF",
            "-DGGML_VULKAN=OFF",
            "-DGGML_OPENCL=OFF",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(args.build), "--config", "Release", "--target", "llama-server", "-j2"],
        check=True,
    )
    candidates = [args.build / "bin/llama-server", args.build / "bin/Release/llama-server"]
    server = next((path for path in candidates if path.is_file()), None)
    if server is None:
        raise RuntimeError("pinned build did not produce llama-server")
    return server


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def wait_ready(port: int, process: subprocess.Popen, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with {process.returncode}")
        try:
            request_json(f"http://127.0.0.1:{port}/health")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.25)
    raise TimeoutError("llama-server readiness timeout")


def write_new_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(args: argparse.Namespace) -> int:
    captures = validate(args)
    server = build_server(args)
    port = args.port or unused_port()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["GGML_OPENCL_PLATFORM"] = ""
    command = [
        str(server),
        "--model",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--threads",
        str(args.threads),
        "--ubatch-size",
        "1",
        "--parallel",
        "1",
        "--cache-reuse",
        "0",
    ]
    log_path = args.output.with_suffix(args.output.suffix + ".server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            wait_ready(port, process)
            results = []
            for capture in captures:
                response = request_json(
                    f"http://127.0.0.1:{port}/completion",
                    {
                        "prompt": capture["prompt_token_ids"],
                        "n_predict": args.max_new_tokens,
                        "temperature": 0.0,
                        "seed": 0,
                        "n_probs": 20,
                        "cache_prompt": False,
                    },
                )
                results.append(
                    {
                        "scenario": capture["scenario"],
                        "prompt_token_ids": capture["prompt_token_ids"],
                        "tokens": response.get("tokens", []),
                        "completion_probabilities": response.get("completion_probabilities", []),
                        "raw_response": response,
                    }
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    report = {
        "schema_version": 1,
        "evidence_status": "insufficient_evidence",
        "policy": "advisory",
        "llama_cpp_revision": LLAMA_REVISION,
        "server_sha256": __import__("hashlib").sha256(server.read_bytes()).hexdigest(),
        "ubatch_size": 1,
        "cache_prompt": False,
        "top_logprobs": 20,
        "captures": results,
    }
    write_new_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "evidence_status": "incomplete",
            "worker": "llama_cpp_cpu_capture",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            write_new_atomic(args.output, failure)
        except FileExistsError:
            pass
        print(json.dumps(failure, indent=2), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
