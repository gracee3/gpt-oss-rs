#!/usr/bin/env python3
"""Validate and execute the digest-pinned CPU oracle container."""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


LOCK_SCHEMA = "gpt-oss-rs.cpu-oracle-lock/v1"
PREFLIGHT_SCHEMA = "gpt-oss-rs.cpu-oracle-preflight/v1"
IMAGE_NAME = "ghcr.io/gracee3/gpt-oss-rs-cpu-oracle"
PLATFORM = "linux/amd64"
OFFICIAL_RELEASE = "v0.0.9"
OFFICIAL_REVISION = "599476783c6f88508dab8577808b5ead5cbee8d2"
OFFICIAL_ARCHIVE_SHA256 = "7306d68ae017f461f2ebb82d04628f8dcba7cc7b431ef28e8786c947510c6f6b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
LLAMA_REVISION = "030ebb558a5820b444a8f836ed5cdd46c9b4bd7a"
BASE_REFERENCE = "python:3.12.12-slim-bookworm@sha256:2986c55feb36e6cae00fa1fefb454283e4b33f35e75ff8bdd123b134130be301"
IMAGE_INPUTS = (
    "oracle/Dockerfile.cpu",
    "oracle/requirements.cpu.lock",
    "oracle/environment_probe.py",
    "crates/gpt-oss-bench/tools/official_cpu_oracle.py",
)
WHEEL_SHA256 = {
    "torch": "ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df",
    "safetensors": "fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_REFERENCE = re.compile(
    rf"^{re.escape(IMAGE_NAME)}@sha256:([0-9a-f]{{64}})$"
)
POLICY = {
    "schema": "gpt-oss-rs.cpu-oracle-container-policy/v1",
    "platform": PLATFORM,
    "user": "invoking-uid-and-gid",
    "read_only_root": True,
    "network": "none",
    "capabilities": [],
    "no_new_privileges": True,
    "mounts": {
        "model": "read-only",
        "repository": "read-only",
        "attempt": "read-write",
    },
    "cpuset_cpus": "0-3",
    "cpus": "4",
    "intraop_threads": 4,
    "interop_threads": 1,
    "environment": {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONHASHSEED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
    },
    "memory_bytes": 24 * 1024**3,
    "memory_swap_bytes": 24 * 1024**3,
    "hostname": "gpt-oss-cpu-oracle",
    "pids_limit": 1024,
}


class ValidationError(RuntimeError):
    pass


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


def policy_sha256() -> str:
    return sha256_bytes(canonical(POLICY))


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON document is not an object: {path}")
    return value


def write_new_atomic(path: Path, value: object) -> None:
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


def require_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValidationError(f"{name} must be a lowercase SHA-256")
    return value


def require_revision(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or not all(character in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{name} must be a lowercase 40-hex revision")
    return value


def verify_oci_archive(path: Path, manifest_digest: str) -> None:
    expected = f"sha256:{manifest_digest}"
    try:
        with tarfile.open(path, "r:*") as archive:
            members = {member.name: member for member in archive.getmembers()}
            if "oci-layout" not in members or "index.json" not in members:
                raise ValidationError("OCI archive lacks oci-layout or index.json")
            for name, member in members.items():
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ValidationError(f"unsafe OCI archive member {name}")
                parts = pure.parts
                if len(parts) == 3 and parts[:2] == ("blobs", "sha256"):
                    require_hash(parts[2], f"OCI blob name {name}")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValidationError(f"OCI blob is not a file: {name}")
                    digest = hashlib.sha256()
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                    observed = digest.hexdigest()
                    if observed != parts[2]:
                        raise ValidationError(f"corrupt OCI blob {name}")
            stream = archive.extractfile(members["index.json"])
            if stream is None:
                raise ValidationError("OCI index is not a file")
            index = json.load(stream)
            digests = {descriptor.get("digest") for descriptor in index.get("manifests", [])}
            if expected not in digests:
                raise ValidationError(
                    f"OCI archive does not select pushed manifest {expected}"
                )
    except (tarfile.TarError, OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid OCI archive {path}: {error}") from error


def verify_lock(
    lock_path: Path,
    repository: Path,
    archive: Path | None = None,
) -> dict:
    lock = read_json(lock_path)
    expected_values = {
        "schema": LOCK_SCHEMA,
        "image_name": IMAGE_NAME,
        "platform": PLATFORM,
        "official_release": OFFICIAL_RELEASE,
        "official_source_revision": OFFICIAL_REVISION,
        "official_source_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "model_revision": MODEL_REVISION,
        "llama_cpp_revision": LLAMA_REVISION,
        "base_reference": BASE_REFERENCE,
    }
    for key, expected in expected_values.items():
        if lock.get(key) != expected:
            raise ValidationError(f"{key} mismatch: expected {expected!r}")

    reference = lock.get("image_reference")
    if not isinstance(reference, str):
        raise ValidationError("image_reference is missing")
    match = IMAGE_REFERENCE.fullmatch(reference)
    if not match:
        raise ValidationError("image_reference must use the canonical name@sha256:digest")
    manifest_digest = require_hash(lock.get("image_manifest_digest"), "image_manifest_digest")
    if match.group(1) != manifest_digest:
        raise ValidationError("image reference and manifest digest differ")
    require_hash(lock.get("image_config_digest"), "image_config_digest")
    require_hash(lock.get("software_lock_sha256"), "software_lock_sha256")
    require_hash(lock.get("container_policy_sha256"), "container_policy_sha256")
    require_hash(lock.get("probe_script_sha256"), "probe_script_sha256")
    require_hash(lock.get("oci_archive_sha256"), "oci_archive_sha256")
    require_hash(lock.get("sbom_sha256"), "sbom_sha256")
    require_hash(lock.get("provenance_sha256"), "provenance_sha256")
    require_revision(lock.get("image_input_revision"), "image_input_revision")

    repository = repository.resolve()
    requirements = repository / "oracle/requirements.cpu.lock"
    if sha256_file(requirements) != lock["software_lock_sha256"]:
        raise ValidationError("software dependency lock changed")
    if policy_sha256() != lock["container_policy_sha256"]:
        raise ValidationError("container execution policy changed")
    probe = repository / "oracle/environment_probe.py"
    if sha256_file(probe) != lock["probe_script_sha256"]:
        raise ValidationError("environment probe changed")

    inputs = lock.get("image_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(IMAGE_INPUTS):
        raise ValidationError("image_inputs do not name the exact build-input set")
    for relative, expected in inputs.items():
        require_hash(expected, f"image_inputs[{relative!r}]")
        path = (repository / relative).resolve()
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise ValidationError(f"image input escapes repository: {relative}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise ValidationError(f"image input changed or missing: {relative}")
    if lock.get("wheel_sha256") != WHEEL_SHA256:
        raise ValidationError("direct wheel hashes do not match the fixed CPU oracle")

    if archive is not None:
        if not archive.is_file():
            raise ValidationError(f"OCI archive is missing: {archive}")
        if sha256_file(archive) != lock["oci_archive_sha256"]:
            raise ValidationError("OCI archive SHA-256 mismatch")
        verify_oci_archive(archive, manifest_digest)
    return lock


def git_revision(path: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def verify_model(model: Path) -> dict:
    manifest_path = model / "gpt-oss-rs-fetch-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("resolved_revision") != MODEL_REVISION:
        raise ValidationError("model revision does not match the fixed oracle model revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("model fetch manifest has no files")
    results = []
    for item in files:
        relative = item.get("path")
        expected = require_hash(item.get("sha256"), f"model file {relative}")
        path = model / relative
        if not path.is_file():
            raise ValidationError(f"model file is missing: {relative}")
        if path.stat().st_size != item.get("size"):
            raise ValidationError(f"model file size changed: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValidationError(f"model file hash changed: {relative}")
        results.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed})
    return {
        "revision": MODEL_REVISION,
        "manifest_sha256": sha256_file(manifest_path),
        "files": results,
    }


def docker_group_active() -> bool:
    try:
        docker_gid = grp.getgrnam("docker").gr_gid
    except KeyError:
        return False
    return docker_gid == os.getgid() or docker_gid in os.getgroups()


def docker_inspect(lock: dict) -> dict:
    if not docker_group_active():
        raise ValidationError(
            "Docker group is not active; start a refreshed login or run newgrp docker"
        )
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", lock["image_reference"]],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError(f"Docker daemon/image inspection failed: {error}") from error
    values = json.loads(completed.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise ValidationError("docker image inspect returned an unexpected document")
    image = values[0]
    expected_config = f"sha256:{lock['image_config_digest']}"
    expected_manifest = f"sha256:{lock['image_manifest_digest']}"
    descriptor = image.get("Descriptor") or {}
    image_id = image.get("Id")
    if image_id == expected_config:
        storage_identity = "classic-config"
    elif (
        image_id == expected_manifest
        and descriptor.get("digest") == expected_manifest
        and descriptor.get("mediaType")
        in (
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        )
    ):
        # Docker's containerd image store identifies a pulled multi-platform
        # image by its index descriptor. The platform config digest is still
        # proven by the required, fully hashed OCI archive and image labels.
        storage_identity = "containerd-index"
    else:
        raise ValidationError("local image digest does not match oracle lock")
    if (image.get("Os"), image.get("Architecture")) != ("linux", "amd64"):
        raise ValidationError("oracle image platform is not linux/amd64")
    if lock["image_reference"] not in image.get("RepoDigests", []):
        raise ValidationError("local image is not associated with the locked manifest digest")
    labels = image.get("Config", {}).get("Labels", {}) or {}
    expected_labels = {
        "org.opencontainers.image.revision": lock["image_input_revision"],
        "io.gpt-oss-rs.oracle.platform": PLATFORM,
        "io.gpt-oss-rs.oracle.official-source-revision": OFFICIAL_REVISION,
        "io.gpt-oss-rs.oracle.official-source-sha256": OFFICIAL_ARCHIVE_SHA256,
        "io.gpt-oss-rs.oracle.model-revision": MODEL_REVISION,
    }
    for key, expected in expected_labels.items():
        if labels.get(key) != expected:
            raise ValidationError(f"image label {key} mismatch")
    return {
        "id": image["Id"],
        "repo_digests": image["RepoDigests"],
        "os": image["Os"],
        "architecture": image["Architecture"],
        "labels": expected_labels,
        "storage_identity": storage_identity,
        "descriptor": descriptor,
    }


def docker_policy_args(
    repository: Path,
    model: Path,
    attempt: Path,
    mode: str,
) -> list[str]:
    args = [
        "docker",
        "run",
        "--rm",
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--user={os.getuid()}:{os.getgid()}",
        "--cpuset-cpus=0-3",
        "--cpus=4",
        "--memory=24g",
        "--memory-swap=24g",
        "--pids-limit=1024",
        "--hostname=gpt-oss-cpu-oracle",
        "--workdir=/repo",
        "--env=OMP_NUM_THREADS=4",
        "--env=MKL_NUM_THREADS=4",
        "--env=PYTHONHASHSEED=0",
        "--env=CUDA_VISIBLE_DEVICES=",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        f"--mount=type=bind,src={model.resolve()},dst=/model,readonly",
        f"--mount=type=bind,src={repository.resolve()},dst=/repo,readonly",
        f"--mount=type=bind,src={attempt.resolve()},dst=/attempt",
    ]
    if mode == "generic":
        args.append("--env=ATEN_CPU_CAPABILITY=default")
    return args


def run_probe(
    lock: dict,
    repository: Path,
    model: Path,
    attempt: Path,
    mode: str,
) -> tuple[dict, bytes]:
    command = docker_policy_args(repository, model, attempt, mode)
    command.extend([lock["image_reference"], "--mode", mode, "--repetitions", "5"])
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        raise ValidationError(
            f"{mode} oracle probe failed: {completed.stderr.decode(errors='replace').strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValidationError(f"{mode} oracle probe returned invalid JSON") from error
    if (
        value.get("schema") != "gpt-oss-rs.cpu-oracle-probe/v1"
        or value.get("execution_mode") != mode
        or value.get("repetitions") != 5
        or value.get("repeat_identical") is not True
        or value.get("software_lock_sha256") != lock["software_lock_sha256"]
    ):
        raise ValidationError(f"{mode} oracle probe is incomplete or mismatched")
    if value.get("torch", {}).get("cuda_available") or value.get("torch", {}).get("cuda_version"):
        raise ValidationError("oracle probe exposed CUDA")
    return value, completed.stdout


def validate_probe_pair(probes: dict[str, dict]) -> None:
    for mode in ("native", "generic"):
        probe = probes.get(mode)
        if not isinstance(probe, dict) or not isinstance(probe.get("record"), dict):
            raise ValidationError(f"incomplete {mode} probe artifact")
    if probes["native"]["record"].get("host_fingerprint") != probes["generic"]["record"].get("host_fingerprint"):
        raise ValidationError("native and generic probes produced different host keys")
    if probes["generic"]["record"].get("torch", {}).get("cpu_capability", "").upper() != "DEFAULT":
        raise ValidationError("generic diagnostic probe did not use DEFAULT CPU capability")


def preflight(args: argparse.Namespace) -> int:
    output = Path(args.output)
    record: dict[str, object] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "incomplete",
        "limitations": [],
    }
    try:
        lock_path = Path(args.lock).resolve()
        repository = Path(args.repository).resolve()
        model = Path(args.model).resolve()
        attempt = Path(args.attempt_directory).resolve()
        archive = Path(args.oci_archive).resolve()
        attempt.mkdir(parents=True, exist_ok=False)
        lock = verify_lock(lock_path, repository, archive)
        llama_revision, llama_dirty = git_revision(Path(args.llama_source).resolve())
        if llama_revision != LLAMA_REVISION or llama_dirty:
            raise ValidationError("llama.cpp source is not the clean pinned revision")
        image = docker_inspect(lock)
        model_evidence = verify_model(model)
        probes = {}
        for mode in ("native", "generic"):
            value, raw = run_probe(lock, repository, model, attempt, mode)
            probe_path = attempt / f"probe-{mode}.json"
            probe_path.write_bytes(raw)
            probes[mode] = {
                "path": str(probe_path),
                "sha256": sha256_bytes(raw),
                "record": value,
            }
        validate_probe_pair(probes)
        record.update(
            {
                "status": "pass",
                "lock_path": str(lock_path),
                "lock_sha256": sha256_file(lock_path),
                "container_policy_sha256": policy_sha256(),
                "image": image,
                "model": model_evidence,
                "llama_cpp": {"revision": llama_revision, "dirty": llama_dirty},
                "probes": probes,
            }
        )
        code = 0
    except BaseException as error:
        record["status"] = "unavailable"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        code = 2
    write_new_atomic(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return code


def oracle_identity(preflight_path: Path, lock: dict, mode: str) -> dict[str, str]:
    preflight_record = read_json(preflight_path)
    if preflight_record.get("status") != "pass":
        raise ValidationError("oracle preflight is not passing")
    probe = preflight_record.get("probes", {}).get(mode)
    if not isinstance(probe, dict):
        raise ValidationError(f"preflight lacks a {mode} probe")
    return {
        "image_manifest_digest": lock["image_manifest_digest"],
        "image_config_digest": lock["image_config_digest"],
        "software_lock_sha256": lock["software_lock_sha256"],
        "official_source_revision": lock["official_source_revision"],
        "execution_mode": mode,
        "host_fingerprint": probe["record"]["host_fingerprint"],
        "container_policy_sha256": lock["container_policy_sha256"],
        "probe_artifact_sha256": probe["sha256"],
    }


def execute(args: argparse.Namespace) -> int:
    lock = verify_lock(Path(args.lock), Path(args.repository))
    docker_inspect(lock)
    attempt = Path(args.attempt_directory).resolve()
    if not attempt.is_dir():
        raise ValidationError("attempt directory must already exist")
    preflight_path = Path(args.preflight).resolve()
    identity = oracle_identity(preflight_path, lock, args.mode)
    command = docker_policy_args(
        Path(args.repository), Path(args.model), attempt, args.mode
    )
    command.extend(
        [
            "--entrypoint=",
            f"--env=GPT_OSS_ORACLE_IDENTITY_JSON={json.dumps(identity, sort_keys=True, separators=(',', ':'))}",
            lock["image_reference"],
            *args.command,
        ]
    )
    return subprocess.run(command).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command_name", required=True)

    verify = subcommands.add_parser("verify-lock")
    verify.add_argument("--lock", required=True)
    verify.add_argument("--repository", default=".")
    verify.add_argument("--oci-archive")

    qualify = subcommands.add_parser("preflight")
    qualify.add_argument("--lock", required=True)
    qualify.add_argument("--repository", default=".")
    qualify.add_argument("--model", required=True)
    qualify.add_argument("--llama-source", required=True)
    qualify.add_argument("--oci-archive", required=True)
    qualify.add_argument("--attempt-directory", required=True)
    qualify.add_argument("--output", required=True)

    run = subcommands.add_parser("exec")
    run.add_argument("--lock", required=True)
    run.add_argument("--repository", default=".")
    run.add_argument("--model", required=True)
    run.add_argument("--attempt-directory", required=True)
    run.add_argument("--preflight", required=True)
    run.add_argument("--mode", choices=("native", "generic"), required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command_name == "verify-lock":
        verify_lock(
            Path(args.lock),
            Path(args.repository),
            Path(args.oci_archive) if args.oci_archive else None,
        )
        return 0
    if args.command_name == "preflight":
        return preflight(args)
    if not args.command:
        raise ValidationError("exec requires a command after --")
    if args.command[0] == "--":
        args.command = args.command[1:]
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, subprocess.CalledProcessError) as error:
        print(f"cpu_oracle: {error}", file=sys.stderr)
        raise SystemExit(2)
