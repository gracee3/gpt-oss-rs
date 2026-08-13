#!/usr/bin/env python3
"""Deterministic CPU-oracle environment and BF16 operator probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import struct
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn.functional as functional


SCHEMA = "gpt-oss-rs.cpu-oracle-probe/v1"
REQUIRED_PACKAGES = (
    "torch",
    "safetensors",
    "filelock",
    "typing-extensions",
    "setuptools",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
    "mpmath",
    "markupsafe",
)
BANNED_PACKAGES = ("gpt-oss", "triton", "torchvision", "torchaudio", "transformers")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError as error:
        return f"unavailable:{type(error).__name__}"


def cpu_info() -> dict[str, object]:
    first = read_text("/proc/cpuinfo").split("\n\n", 1)[0]
    parsed = {}
    for line in first.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    flags = parsed.get("flags", "").split()
    return {
        key: parsed.get(key, "")
        for key in ("vendor_id", "cpu family", "model", "model name", "stepping", "microcode")
    } | {
        "flags_sha256": sha256_bytes("\n".join(sorted(flags)).encode()),
        "logical_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }


def cgroup_info() -> dict[str, str]:
    paths = (
        "/proc/self/cgroup",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.swap.max",
    )
    return {path: read_text(path) for path in paths}


def host_key(host: dict[str, object]) -> dict[str, object]:
    """Remove only the per-container cgroup leaf from the cross-mode host key."""
    normalized = json.loads(json.dumps(host))
    cgroup = normalized.get("cgroup", {})
    raw = cgroup.get("/proc/self/cgroup", "")
    cgroup["/proc/self/cgroup"] = re.sub(
        r"(?<=/)[0-9a-f]{64}(?=$|\n)", "<container-id>", raw
    )
    return normalized


def tensor_bytes(value: torch.Tensor) -> bytes:
    if value.device.type != "cpu":
        raise RuntimeError("operator fingerprints must remain on CPU")
    contiguous = value.detach().contiguous()
    if contiguous.dtype == torch.bfloat16:
        items = contiguous.view(torch.uint16).reshape(-1).tolist()
        return struct.pack(f"<{len(items)}H", *items)
    if contiguous.dtype == torch.float32:
        items = contiguous.view(torch.int32).reshape(-1).tolist()
        return struct.pack(f"<{len(items)}i", *items)
    raise TypeError(f"unsupported fingerprint dtype {contiguous.dtype}")


def operator_fingerprints() -> dict[str, str]:
    left = torch.tensor(
        [0.5, -1.0, 2.0, 3.5, -0.25, 0.75, -2.5, 1.25,
         4.0, -3.0, 0.125, -0.5, 1.5, 2.5, -1.75, 0.25],
        dtype=torch.bfloat16,
    ).reshape(4, 4)
    right = torch.tensor(
        [1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 3.0, -0.25,
         0.5, 1.25, -2.0, 0.125, 2.5, -1.0, 0.5, 1.5],
        dtype=torch.bfloat16,
    ).reshape(4, 4)
    bias = torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.bfloat16)
    linear = functional.linear(left, right, bias)
    matrix = left @ right
    softmax = functional.softmax(linear.float(), dim=-1).to(torch.bfloat16)
    rms = (
        left.float()
        * torch.rsqrt(torch.mean(left.float() ** 2, dim=-1, keepdim=True) + 1.0e-5)
    ).to(torch.bfloat16)
    ldexp = torch.ldexp(left, torch.tensor([[0, 1, -1, 2]] * 4, dtype=torch.int32))
    return {
        name: sha256_bytes(tensor_bytes(value))
        for name, value in {
            "bf16_linear": linear,
            "bf16_matmul": matrix,
            "bf16_softmax": softmax,
            "bf16_rms_norm": rms,
            "bf16_ldexp": ldexp,
        }.items()
    }


def wheel_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    package = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith(("#", "--", "\\")) and "==" in line:
            package = re.split(r"==", line, maxsplit=1)[0].lower()
        match = re.search(r"--hash=sha256:([0-9a-f]{64})", line)
        if match and package:
            hashes[package] = match.group(1)
            package = None
    return hashes


def package_snapshot() -> dict[str, object]:
    installed = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
    }
    missing = [package for package in REQUIRED_PACKAGES if package not in installed]
    banned = [package for package in BANNED_PACKAGES if package in installed]
    if missing or banned:
        raise RuntimeError(f"package policy failed: missing={missing}, banned={banned}")
    return {
        "versions": {package: installed[package] for package in REQUIRED_PACKAGES},
        "wheel_sha256": wheel_hashes(Path("/opt/oracle/requirements.cpu.lock")),
    }


def probe(mode: str, repetitions: int) -> dict[str, object]:
    expected_capability = os.environ.get("ATEN_CPU_CAPABILITY")
    if mode == "native" and expected_capability:
        raise RuntimeError("native mode must use normal PyTorch CPU dispatch")
    if mode == "generic" and expected_capability != "default":
        raise RuntimeError("generic mode requires ATEN_CPU_CAPABILITY=default")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    if torch.cuda.is_available() or torch.version.cuda is not None:
        raise RuntimeError("CUDA is visible in the CPU oracle")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    fingerprints = [operator_fingerprints() for _ in range(repetitions)]
    digests = [sha256_bytes(canonical(value)) for value in fingerprints]
    if len(set(digests)) != 1:
        raise RuntimeError("BF16 operator fingerprints were not repeat-identical")

    host = {
        "kernel": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu": cpu_info(),
        "cgroup": cgroup_info(),
    }
    capability = torch.backends.cpu.get_cpu_capability()
    if mode == "generic" and capability.upper() != "DEFAULT":
        raise RuntimeError(f"generic dispatch did not select DEFAULT: {capability}")
    requirements = Path("/opt/oracle/requirements.cpu.lock")
    return {
        "schema": SCHEMA,
        "execution_mode": mode,
        "repetitions": repetitions,
        "repeat_identical": True,
        "fingerprint_sha256": digests[0],
        "operator_fingerprints": fingerprints[0],
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": package_snapshot(),
        "software_lock_sha256": sha256_file(requirements),
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "configuration": torch.__config__.show(),
            "cpu_capability": capability,
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "openmp_available": torch.backends.openmp.is_available(),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "host": host,
        "host_fingerprint": sha256_bytes(canonical(host_key(host))),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ATEN_CPU_CAPABILITY",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "generic"), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions != 5:
        raise ValueError("certification probes require exactly five repetitions")
    print(json.dumps(probe(args.mode, args.repetitions), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
