#!/usr/bin/env python3
"""Run and validate the bounded v0.1.0 CPU research benchmark.

This driver intentionally uses only the Python standard library.  It wraps the
repository's Rust evidence binaries and a pinned CPU-only llama.cpp server; it
does not implement inference or silently repair invalid evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


SCHEMA = "gpt-oss-rs.publication-benchmark/v1"
MATRIX_SCHEMA = "gpt-oss-rs.mxfp4-matrix-benchmark/v1"
MATRIX_ANALYSIS_SCHEMA = "gpt-oss-rs.mxfp4-matrix-promotion-analysis/v1"
LLAMA_REVISION = "030ebb558a5820b444a8f836ed5cdd46c9b4bd7a"
MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
SCENARIO = "harmony_63"
LANES = ("gpt-oss-rs-auto", "gpt-oss-rs-scalar", "llama.cpp-normal", "llama.cpp-ubatch-1")
MEASURED_ROUNDS = 5
EXPECTED_MODEL_HASHES = {
    "config.json": "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce",
    "model.safetensors.index.json": "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1",
    "tokenizer.json": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
}
HISTORICAL_GGUF_SHA256 = "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"


class BenchmarkError(RuntimeError):
    """A fail-closed protocol or evidence error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def command_text(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def git(repository: Path, *arguments: str) -> str:
    return command_text(["git", *arguments], repository)


def parse_cpu_set(value: str) -> list[int]:
    cpus: set[int] = set()
    try:
        for segment in value.split(","):
            bounds = segment.split("-", 1)
            start = int(bounds[0])
            end = int(bounds[-1])
            if start < 0 or end < start:
                raise ValueError
            cpus.update(range(start, end + 1))
    except ValueError as error:
        raise BenchmarkError(f"invalid CPU set: {value}") from error
    if not cpus:
        raise BenchmarkError("CPU set is empty")
    return sorted(cpus)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def physical_cpu_identity(cpu: int) -> tuple[str, str]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    package = read_text(root / "physical_package_id")
    core = read_text(root / "core_id")
    if package is None or core is None:
        raise BenchmarkError(f"CPU {cpu} has no readable topology identity")
    return package, core


def validate_physical_cpu_set(cpus: list[int]) -> None:
    identities = []
    for cpu in cpus:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        if not root.is_dir() or read_text(root / "online") == "0":
            raise BenchmarkError(f"CPU {cpu} is absent or offline")
        identities.append(physical_cpu_identity(cpu))
    if len(set(identities)) != len(identities):
        raise BenchmarkError("CPU set includes SMT siblings; physical CPUs are required")


def cpu_info() -> dict:
    records: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = (item.strip() for item in line.split(":", 1))
        if key in {"vendor_id", "cpu family", "model", "model name", "stepping", "microcode"}:
            records.setdefault(key.replace(" ", "_"), value)
    records["logical_cpus"] = str(os.cpu_count())
    return records


def temperatures_c() -> list[float]:
    values = []
    for root in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        if read_text(root / "name") != "coretemp":
            continue
        for path in sorted(root.glob("temp*_input")):
            raw = read_text(path)
            if raw is not None:
                values.append(int(raw) / 1000.0)
    if not values:
        raise BenchmarkError("coretemp exposes no readable temperatures")
    return values


def throttle_snapshot(cpus: list[int]) -> dict[str, int]:
    package_path = Path(
        f"/sys/devices/system/cpu/cpu{cpus[0]}/thermal_throttle/package_throttle_total_time_ms"
    )
    package = read_text(package_path)
    if package is None:
        raise BenchmarkError("package thermal-throttle counter is unavailable")
    result = {"package_total_time_ms": int(package)}
    for cpu in cpus:
        path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/thermal_throttle/core_throttle_total_time_ms"
        )
        raw = read_text(path)
        if raw is None:
            raise BenchmarkError(f"CPU {cpu} thermal-throttle counter is unavailable")
        result[f"cpu{cpu}_core_total_time_ms"] = int(raw)
    return result


def throttle_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    if set(before) != set(after):
        raise BenchmarkError("thermal-throttle counter identity changed")
    result = {key: after[key] - before[key] for key in before}
    if any(value < 0 for value in result.values()):
        raise BenchmarkError("thermal-throttle counter regressed")
    return result


def wait_for_thermal_gate(limit_c: float, timeout_seconds: float) -> dict:
    started = time.monotonic()
    while True:
        current = temperatures_c()
        maximum = max(current)
        if maximum <= limit_c:
            return {
                "maximum_c": maximum,
                "temperatures_c": current,
                "wait_seconds": time.monotonic() - started,
            }
        if time.monotonic() - started >= timeout_seconds:
            raise BenchmarkError(
                f"thermal gate remained at {maximum:.1f} C above {limit_c:.1f} C"
            )
        time.sleep(min(5.0, timeout_seconds))


def fixture_scenario(path: Path) -> tuple[dict, dict]:
    value = json.loads(path.read_text())
    model = value.get("model")
    if not isinstance(model, dict) or model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise BenchmarkError("fixture model identity does not match the publication protocol")
    scenario = next((item for item in value.get("scenarios", []) if item.get("id") == SCENARIO), None)
    if scenario is None:
        raise BenchmarkError(f"fixture has no {SCENARIO} scenario")
    tokens = scenario.get("official_greedy_tokens")
    if scenario.get("expected_prompt_tokens") != 63 or not isinstance(tokens, list) or len(tokens) != 8:
        raise BenchmarkError("harmony_63 fixture does not contain the required 63+8 workload")
    return value, scenario


def validate_model(model: Path, gguf: Path) -> dict:
    hashes = {}
    for name, expected in EXPECTED_MODEL_HASHES.items():
        path = model / name
        if not path.is_file():
            raise BenchmarkError(f"model is missing {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise BenchmarkError(f"model {name} hash does not match the pinned fixture")
        hashes[name] = actual
    if not gguf.is_file():
        raise BenchmarkError("converted GGUF is missing")
    gguf_hash = sha256_file(gguf)
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_hashes": hashes,
        "gguf_sha256": gguf_hash,
        "historical_gguf_sha256": HISTORICAL_GGUF_SHA256,
        "matches_historical_gguf": gguf_hash == HISTORICAL_GGUF_SHA256,
    }


def parse_time_verbose(text: str) -> int:
    prefix = "Maximum resident set size (kbytes):"
    values = [line.split(":", 1)[1].strip() for line in text.splitlines() if prefix in line]
    if len(values) != 1:
        raise BenchmarkError("GNU time output has no unique peak RSS")
    try:
        value = int(values[0])
    except ValueError as error:
        raise BenchmarkError("GNU time peak RSS is not an integer") from error
    if value <= 0:
        raise BenchmarkError("GNU time peak RSS is not positive")
    return value


def process_peak_rss_kib(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        raise BenchmarkError("benchmark process disappeared before RSS capture")
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1])
    raise BenchmarkError("benchmark process exposes no VmHWM")


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict | None = None, timeout: float = 900.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(port: int, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BenchmarkError(f"llama.cpp exited during startup with {process.returncode}")
        try:
            request_json(f"http://127.0.0.1:{port}/health", timeout=5.0)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.25)
    raise BenchmarkError("llama.cpp readiness timeout")


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "GGML_OPENCL_PLATFORM": "",
            "OMP_NUM_THREADS": "8",
            "RAYON_NUM_THREADS": "8",
        }
    )
    return environment


def validate_tokens(tokens: object, label: str) -> list[int]:
    if not isinstance(tokens, list) or not tokens or not all(type(token) is int for token in tokens):
        raise BenchmarkError(f"{label} has no exact generated token IDs")
    return tokens


def normalize_native_capture(value: dict, scenario: dict) -> dict:
    if value.get("scenario") != SCENARIO or value.get("fixture_scenario") != SCENARIO:
        raise BenchmarkError("native capture scenario identity drifted")
    prompt = validate_tokens(value.get("prompt_token_ids"), "native prompt")
    prompt_hash = hashlib.sha256(",".join(map(str, prompt)).encode()).hexdigest()
    if len(prompt) != 63 or prompt_hash != scenario["prompt_token_ids_sha256"]:
        raise BenchmarkError("native prompt token identity drifted")
    tokens = validate_tokens(value.get("generated_token_ids"), "native capture")
    for key in ("startup_seconds", "prompt_seconds", "time_to_first_token_seconds", "full_request_seconds"):
        if not isinstance(value.get(key), (int, float)) or value[key] < 0 or not math.isfinite(value[key]):
            raise BenchmarkError(f"native capture has invalid {key}")
    intervals = value.get("inter_token_seconds")
    if not isinstance(intervals, list) or any(not isinstance(item, (int, float)) or item < 0 for item in intervals):
        raise BenchmarkError("native capture has invalid token-arrival intervals")
    decode_seconds = sum(intervals)
    decode_rate = (len(tokens) - 1) / decode_seconds if len(tokens) > 1 and decode_seconds > 0 else None
    return {
        "prompt_token_ids": prompt,
        "generated_token_ids": tokens,
        "startup_seconds": float(value["startup_seconds"]),
        "prompt_seconds": float(value["prompt_seconds"]),
        "ttft_seconds": float(value["time_to_first_token_seconds"]),
        "full_request_seconds": float(value["full_request_seconds"]),
        "decode_tokens_per_second": decode_rate,
        "executable_sha256": value.get("executable_sha256"),
        "effective_cpu_kernel": value.get("effective_cpu_kernel"),
        "effective_dispatch_plan": value.get("effective_dispatch_plan"),
        "effective_m1_matrix_backend": value.get("effective_m1_matrix_backend"),
        "effective_multirow_matrix_backend": value.get("effective_multirow_matrix_backend"),
        "layer_major_prefill": value.get("layer_major_prefill"),
        "expert_projection": value.get("expert_projection"),
    }


def run_native(args: argparse.Namespace, lane: str, scenario: dict, metadata: dict) -> dict:
    kernel = "auto" if lane == "gpt-oss-rs-auto" else "scalar"
    with tempfile.TemporaryDirectory(prefix="gpt-oss-rs-publication-native-") as directory:
        root = Path(directory)
        capture = root / "capture.json"
        timing = root / "time.txt"
        command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(timing),
            "taskset",
            "-c",
            args.cpus,
            str(args.native_binary),
            "--model",
            str(args.model),
            "--repack-cache",
            str(args.repack_cache),
            "--fixtures",
            str(args.fixtures),
            "--scenario",
            SCENARIO,
            "--kernel",
            kernel,
            "--cpu-matmul-backend",
            kernel,
            "--expert-projection",
            "residual-q8",
            "--threads",
            "8",
            "--max-new-tokens",
            "8",
            "--layer-major-prefill",
            "--output",
            str(capture),
        ]
        completed = subprocess.run(
            command,
            cwd=args.repository,
            env=clean_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            sys.stderr.buffer.write(completed.stderr[-8192:])
            raise BenchmarkError(f"{lane} exited with {completed.returncode}")
        normalized = normalize_native_capture(json.loads(capture.read_text()), scenario)
        normalized["peak_rss_kib"] = parse_time_verbose(timing.read_text())
    return {**metadata, **normalized}


def run_llama(args: argparse.Namespace, lane: str, scenario: dict, metadata: dict) -> dict:
    port = unused_port()
    command = [
        "taskset",
        "-c",
        args.cpus,
        str(args.llama_server),
        "--model",
        str(args.gguf),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--parallel",
        "1",
        "--ctx-size",
        "128",
        "--cache-reuse",
        "0",
        "--n-gpu-layers",
        "0",
    ]
    if lane == "llama.cpp-ubatch-1":
        command.extend(["--ubatch-size", "1"])
    with tempfile.TemporaryDirectory(prefix="gpt-oss-rs-publication-llama-") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("wb") as log:
            startup_start = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=clean_environment())
            try:
                wait_ready(port, process)
                startup_seconds = time.monotonic() - startup_start
                peak_rss_kib = process_peak_rss_kib(process.pid)
                request_start = time.monotonic()
                response = request_json(
                    f"http://127.0.0.1:{port}/completion",
                    {
                        "prompt": metadata["prompt_token_ids"],
                        "n_predict": 8,
                        "temperature": 0.0,
                        "seed": 0,
                        "cache_prompt": False,
                        "return_tokens": True,
                    },
                )
                full_request_seconds = time.monotonic() - request_start
                peak_rss_kib = max(peak_rss_kib, process_peak_rss_kib(process.pid))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if process.returncode not in (0, -15):
            sys.stderr.buffer.write(log_path.read_bytes()[-8192:])
            raise BenchmarkError(f"{lane} exited with {process.returncode}")
    tokens = validate_tokens(response.get("tokens"), lane)
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise BenchmarkError(f"{lane} response has no timings")
    try:
        prompt_seconds = float(timings["prompt_ms"]) / 1000.0
        decode_rate = float(timings["predicted_per_second"])
        prompt_n = int(timings["prompt_n"])
        predicted_n = int(timings["predicted_n"])
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkError(f"{lane} response timings are incomplete") from error
    if prompt_n != 63 or predicted_n != len(tokens) or len(tokens) != 8:
        raise BenchmarkError(f"{lane} token counts do not match the 63+8 protocol")
    return {
        **metadata,
        "generated_token_ids": tokens,
        "startup_seconds": startup_seconds,
        "prompt_seconds": prompt_seconds,
        "ttft_seconds": prompt_seconds,
        "full_request_seconds": full_request_seconds,
        "decode_tokens_per_second": decode_rate,
        "peak_rss_kib": peak_rss_kib,
        "server_timings": {
            key: timings.get(key)
            for key in (
                "cache_n",
                "prompt_n",
                "prompt_ms",
                "prompt_per_second",
                "predicted_n",
                "predicted_ms",
                "predicted_per_second",
            )
        },
    }


def expected_run_order() -> list[dict]:
    order = []
    for round_index in range(MEASURED_ROUNDS + 1):
        phase = "warmup" if round_index == 0 else "measured"
        rotation = round_index % len(LANES)
        lanes = LANES[rotation:] + LANES[:rotation]
        for position, lane in enumerate(lanes):
            order.append(
                {
                    "round": round_index,
                    "phase": phase,
                    "order": position,
                    "lane": lane,
                }
            )
    return order


def validate_run_order(runs: list[dict]) -> None:
    observed = [
        {key: run.get(key) for key in ("round", "phase", "order", "lane")}
        for run in runs
    ]
    if observed != expected_run_order():
        raise BenchmarkError("run order does not match the fixed rotation")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise BenchmarkError("cannot select a percentile from no values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * probability))
    return ordered[index]


def paired_bootstrap_speedup(
    candidate: list[float], comparator: list[float], *, lower_is_better: bool, iterations: int = 10_000
) -> dict:
    if len(candidate) != len(comparator) or not candidate:
        raise BenchmarkError("paired speedup samples are missing")
    rng = random.Random(sha256_json([candidate, comparator, lower_is_better]))
    ratios = []
    for _ in range(iterations):
        indices = [rng.randrange(len(candidate)) for _ in candidate]
        candidate_median = statistics.median(candidate[index] for index in indices)
        comparator_median = statistics.median(comparator[index] for index in indices)
        numerator, denominator = (
            (comparator_median, candidate_median)
            if lower_is_better
            else (candidate_median, comparator_median)
        )
        ratios.append(numerator / denominator)
    estimate = (
        statistics.median(comparator) / statistics.median(candidate)
        if lower_is_better
        else statistics.median(candidate) / statistics.median(comparator)
    )
    interval = [percentile(ratios, 0.025), percentile(ratios, 0.975)]
    supported = interval[0] > 1.0
    return {
        "iterations": iterations,
        "estimate": estimate,
        "95ci": interval,
        "supported": supported,
        "reported_speedup": estimate if supported else None,
    }


def aggregate_runs(runs: list[dict], expected_tokens: list[int]) -> dict:
    validate_run_order(runs)
    measured = [run for run in runs if run["phase"] == "measured"]
    lane_runs = {lane: [run for run in measured if run["lane"] == lane] for lane in LANES}
    if any(len(values) != MEASURED_ROUNDS for values in lane_runs.values()):
        raise BenchmarkError("each lane requires five measured runs")
    metrics = (
        "startup_seconds",
        "prompt_seconds",
        "ttft_seconds",
        "full_request_seconds",
        "decode_tokens_per_second",
        "peak_rss_kib",
    )
    summaries = {}
    token_identity = {}
    for lane, values in lane_runs.items():
        for value in values:
            for metric in metrics:
                if not isinstance(value.get(metric), (int, float)) or value[metric] < 0:
                    raise BenchmarkError(f"{lane} has an invalid {metric}")
        summaries[lane] = {
            "samples": len(values),
            **{f"median_{metric}": statistics.median(value[metric] for value in values) for metric in metrics},
        }
        distinct = {tuple(value["generated_token_ids"]) for value in values}
        token_identity[lane] = {
            "stable": len(distinct) == 1,
            "matches_official": distinct == {tuple(expected_tokens)},
            "sequences": [list(item) for item in sorted(distinct)],
        }
    divergence = any(not value["matches_official"] for value in token_identity.values())
    comparison = None
    if not divergence:
        automatic = sorted(lane_runs["gpt-oss-rs-auto"], key=lambda value: value["round"])
        scalar = sorted(lane_runs["gpt-oss-rs-scalar"], key=lambda value: value["round"])
        comparison = {
            "candidate": "gpt-oss-rs-auto",
            "comparator": "gpt-oss-rs-scalar",
            "prompt_latency": paired_bootstrap_speedup(
                [value["prompt_seconds"] for value in automatic],
                [value["prompt_seconds"] for value in scalar],
                lower_is_better=True,
            ),
            "full_request_latency": paired_bootstrap_speedup(
                [value["full_request_seconds"] for value in automatic],
                [value["full_request_seconds"] for value in scalar],
                lower_is_better=True,
            ),
            "decode_throughput": paired_bootstrap_speedup(
                [value["decode_tokens_per_second"] for value in automatic],
                [value["decode_tokens_per_second"] for value in scalar],
                lower_is_better=False,
            ),
        }
    return {
        "summaries": summaries,
        "token_identity": token_identity,
        "token_divergence": divergence,
        "ratio_policy": (
            "ratios omitted because at least one lane diverged from official tokens"
            if divergence
            else "only paired internal gpt-oss-rs Auto-versus-scalar speedups with 95% CI above 1 are reportable"
        ),
        "internal_comparison": comparison,
        "llama_context_policy": "same-host context only; no universal or cross-project speedup claim",
    }


def run_matrix(args: argparse.Namespace, output: Path, n: int) -> dict:
    wait_for_thermal_gate(args.thermal_gate_c, args.thermal_timeout_seconds)
    command = [
        "taskset",
        "-c",
        args.cpus,
        str(args.matrix_binary),
        "--m-values",
        "1,3,8",
        "--n",
        str(n),
        "--k",
        "2880",
        "--activation",
        "residual-q8",
        "--methods",
        "scalar,avx2,avx512-vnni,auto",
        "--warmups",
        "3",
        "--trials",
        "7",
        "--samples-per-trial",
        "5",
        "--thread-policy",
        "8",
        "--thermal-start-gate-c",
        str(args.thermal_gate_c),
        "--thermal-end-ceiling-c",
        str(args.thermal_ceiling_c),
        "--thermal-max-wait-seconds",
        str(int(args.thermal_timeout_seconds)),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=args.repository,
        env=clean_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr[-8192:])
        raise BenchmarkError(f"MXFP4 N={n} benchmark exited with {completed.returncode}")
    value = json.loads(output.read_text())
    if value.get("schema") != MATRIX_SCHEMA or value.get("repository_dirty"):
        raise BenchmarkError("MXFP4 benchmark did not publish clean v1 evidence")
    if value.get("repository_commit") != args.expected_commit:
        raise BenchmarkError("MXFP4 benchmark source commit drifted")
    if any(not row.get("scalar_exact") for row in value.get("correctness", [])):
        raise BenchmarkError("MXFP4 benchmark lacks exact scalar equivalence")
    return value


def run_matrix_analysis(args: argparse.Namespace, inputs: list[Path], output: Path) -> dict:
    command = [
        sys.executable,
        str(args.matrix_analyzer),
        *map(str, inputs),
        "--output",
        str(output),
        "--bootstrap-iterations",
        "10000",
    ]
    subprocess.run(command, cwd=args.repository, check=True)
    value = json.loads(output.read_text())
    if value.get("schema") != MATRIX_ANALYSIS_SCHEMA or value.get("bootstrap_iterations") != 10_000:
        raise BenchmarkError("MXFP4 paired analysis did not publish the required v1 result")
    return value


def write_checksums(root: Path) -> None:
    checksum = root / "SHA256SUMS"
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path != checksum)
    checksum.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in paths)
    )


def validate_checksums(root: Path) -> int:
    checksum = root / "SHA256SUMS"
    if not checksum.is_file():
        raise BenchmarkError("publication has no SHA256SUMS")
    count = 0
    for line in checksum.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise BenchmarkError(f"published artifact checksum failed: {relative}")
        count += 1
    if count == 0:
        raise BenchmarkError("publication checksum index is empty")
    return count


def validate_preflight(args: argparse.Namespace) -> tuple[dict, dict, list[int]]:
    if not args.repository.is_dir() or not args.native_binary.is_file() or not args.matrix_binary.is_file():
        raise BenchmarkError("repository benchmark binaries are missing")
    if not args.llama_server.is_file() or not args.llama_source.is_dir():
        raise BenchmarkError("pinned llama.cpp source or server is missing")
    if git(args.repository, "rev-parse", "HEAD") != args.expected_commit:
        raise BenchmarkError("repository HEAD does not match --expected-commit")
    if git(args.repository, "status", "--porcelain"):
        raise BenchmarkError("publication benchmark requires a clean repository")
    if git(args.llama_source, "rev-parse", "HEAD") != LLAMA_REVISION:
        raise BenchmarkError("llama.cpp source revision drifted")
    if git(args.llama_source, "status", "--porcelain"):
        raise BenchmarkError("llama.cpp source is dirty")
    try:
        args.output.resolve().relative_to(args.repository.resolve())
    except ValueError:
        pass
    else:
        raise BenchmarkError("benchmark output must be outside the repository")
    if args.output.exists():
        raise BenchmarkError("benchmark output already exists")
    cpus = parse_cpu_set(args.cpus)
    if cpus != list(range(8)):
        raise BenchmarkError("publication protocol requires CPUs 0-7")
    validate_physical_cpu_set(cpus)
    fixtures, scenario = fixture_scenario(args.fixtures)
    if fixtures.get("llama_cpp", {}).get("revision") != LLAMA_REVISION:
        raise BenchmarkError("fixture llama.cpp revision drifted")
    model = validate_model(args.model, args.gguf)
    return model, scenario, cpus


def capture(args: argparse.Namespace) -> dict:
    model, scenario, cpus = validate_preflight(args)
    args.output.mkdir(parents=True)
    in_progress = args.output / "capture.in-progress.json"
    protocol = {
        "scenario": SCENARIO,
        "prompt_tokens": 63,
        "greedy_output_tokens": 8,
        "warmups_per_lane": 1,
        "measured_trials_per_lane": MEASURED_ROUNDS,
        "run_order": expected_run_order(),
        "cpus": args.cpus,
        "threads": 8,
        "gpu_backends": "disabled",
        "prompt_cache_reuse": False,
        "fresh_processes": True,
        "page_cache": "warm; no privileged cache dropping",
        "thermal_start_gate_c": args.thermal_gate_c,
        "thermal_end_ceiling_c": args.thermal_ceiling_c,
        "reject_thermal_throttling": True,
    }
    write_json(in_progress, {"schema": SCHEMA, "status": "in_progress", "protocol": protocol})

    matrix_paths = [args.output / "mxfp4-gate-up.json", args.output / "mxfp4-down.json"]
    run_matrix(args, matrix_paths[0], 5760)
    run_matrix(args, matrix_paths[1], 2880)
    matrix_analysis_path = args.output / "mxfp4-analysis.json"
    matrix_analysis = run_matrix_analysis(args, matrix_paths, matrix_analysis_path)

    prompt_token_ids: list[int] | None = None
    runs = []
    for item in expected_run_order():
        gate = wait_for_thermal_gate(args.thermal_gate_c, args.thermal_timeout_seconds)
        before = throttle_snapshot(cpus)
        metadata = {**item, "thermal_gate": gate, "throttle_before": before}
        lane = item["lane"]
        if lane.startswith("gpt-oss-rs"):
            run = run_native(args, lane, scenario, metadata)
            if prompt_token_ids is None:
                prompt_token_ids = run.pop("prompt_token_ids")
            elif run.pop("prompt_token_ids") != prompt_token_ids:
                raise BenchmarkError("native prompt token identity changed between runs")
        else:
            if prompt_token_ids is None:
                raise BenchmarkError("llama.cpp lane ran before a native prompt capture")
            run = run_llama(args, lane, scenario, {**metadata, "prompt_token_ids": prompt_token_ids})
            run.pop("prompt_token_ids")
        after = throttle_snapshot(cpus)
        delta = throttle_delta(before, after)
        end_temperatures = temperatures_c()
        run["throttle_after"] = after
        run["throttle_delta_ms"] = delta
        run["end_temperatures_c"] = end_temperatures
        if any(delta.values()):
            raise BenchmarkError(f"thermal throttling occurred during {lane}")
        if max(end_temperatures) > args.thermal_ceiling_c:
            raise BenchmarkError(f"thermal ceiling exceeded during {lane}")
        runs.append(run)
        write_json(
            in_progress,
            {"schema": SCHEMA, "status": "in_progress", "protocol": protocol, "completed_runs": runs},
        )

    expected_tokens = validate_tokens(scenario["official_greedy_tokens"], "fixture")
    analysis = aggregate_runs(runs, expected_tokens)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {
            "repository": "https://github.com/gracee3/gpt-oss-rs",
            "commit": args.expected_commit,
            "repository_dirty": False,
            "llama_cpp_repository": "https://github.com/ggml-org/llama.cpp",
            "llama_cpp_commit": LLAMA_REVISION,
            "native_binary_sha256": sha256_file(args.native_binary),
            "matrix_binary_sha256": sha256_file(args.matrix_binary),
            "llama_server_sha256": sha256_file(args.llama_server),
        },
        "model": model,
        "host": {
            "captured_unix_seconds": int(time.time()),
            "operating_system": platform.system(),
            "kernel_release": platform.release(),
            "machine": platform.machine(),
            "cpu": cpu_info(),
            "selected_physical_cpus": cpus,
            "governors": sorted(
                {
                    read_text(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"))
                    for cpu in cpus
                }
                - {None}
            ),
        },
        "protocol": protocol,
        "prompt": {
            "scenario": SCENARIO,
            "token_ids": prompt_token_ids,
            "token_ids_sha256": scenario["prompt_token_ids_sha256"],
            "expected_output_token_ids": expected_tokens,
        },
        "matrix": {
            "raw_artifacts": [path.name for path in matrix_paths],
            "analysis_artifact": matrix_analysis_path.name,
            "bootstrap_iterations": matrix_analysis["bootstrap_iterations"],
            "promotion_status": matrix_analysis["promotion_status"],
        },
        "runs": runs,
        "analysis": analysis,
    }
    report["content_sha256"] = sha256_json(report)
    write_json(args.output / "publication-benchmark.json", report)
    in_progress.unlink()
    write_checksums(args.output)
    report["published_artifact_count"] = validate_checksums(args.output)
    return report


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture", help="run the complete bounded protocol")
    capture_parser.add_argument("--repository", type=Path, default=repository)
    capture_parser.add_argument("--expected-commit", required=True)
    capture_parser.add_argument("--native-binary", type=Path, required=True)
    capture_parser.add_argument("--matrix-binary", type=Path, required=True)
    capture_parser.add_argument(
        "--matrix-analyzer",
        type=Path,
        default=Path(__file__).with_name("analyze_mxfp4_matrix_bench.py"),
    )
    capture_parser.add_argument("--model", type=Path, required=True)
    capture_parser.add_argument("--gguf", type=Path, required=True)
    capture_parser.add_argument("--repack-cache", type=Path, required=True)
    capture_parser.add_argument(
        "--fixtures",
        type=Path,
        default=repository / "crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json",
    )
    capture_parser.add_argument("--llama-source", type=Path, required=True)
    capture_parser.add_argument("--llama-server", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--cpus", default="0-7")
    capture_parser.add_argument("--thermal-gate-c", type=float, default=65.0)
    capture_parser.add_argument("--thermal-ceiling-c", type=float, default=95.0)
    capture_parser.add_argument("--thermal-timeout-seconds", type=float, default=900.0)
    validate_parser = subparsers.add_parser("validate", help="verify a completed artifact directory")
    validate_parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "capture":
            report = capture(args)
            print(json.dumps({"status": report["status"], "content_sha256": report["content_sha256"]}))
        else:
            count = validate_checksums(args.output)
            report = json.loads((args.output / "publication-benchmark.json").read_text())
            if report.get("schema") != SCHEMA or report.get("status") != "complete":
                raise BenchmarkError("publication benchmark report is not complete v1 evidence")
            validate_run_order(report.get("runs", []))
            print(f"validated {count} publication artifacts")
    except (BenchmarkError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
