#!/usr/bin/env python3
"""Create candidate A's lock from verified publication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cpu_oracle import (
    BASE_REFERENCE,
    IMAGE_INPUTS,
    IMAGE_NAME,
    LLAMA_REVISION,
    LOCK_SCHEMA,
    MODEL_REVISION,
    OFFICIAL_ARCHIVE_SHA256,
    OFFICIAL_RELEASE,
    OFFICIAL_REVISION,
    PLATFORM,
    WHEEL_SHA256,
    policy_sha256,
    require_hash,
    require_revision,
    sha256_file,
    verify_oci_archive,
    write_new_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--oci-archive", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("oracle/cpu-oracle.lock.json"))
    args = parser.parse_args()

    material = json.loads(args.material.read_text())
    manifest_digest = require_hash(
        str(material["image_manifest_digest"]).removeprefix("sha256:"),
        "image_manifest_digest",
    )
    config_digest = require_hash(
        str(material["image_config_digest"]).removeprefix("sha256:"),
        "image_config_digest",
    )
    image_input_revision = require_revision(
        material["image_input_revision"], "image_input_revision"
    )
    image_reference = f"{IMAGE_NAME}@sha256:{manifest_digest}"
    if material.get("image_reference") != image_reference:
        raise ValueError("publication material image reference is inconsistent")

    repository = args.repository.resolve()
    requirements = repository / "oracle/requirements.cpu.lock"
    probe = repository / "oracle/environment_probe.py"
    lock = {
        "schema": LOCK_SCHEMA,
        "image_name": IMAGE_NAME,
        "image_reference": image_reference,
        "image_manifest_digest": manifest_digest,
        "image_config_digest": config_digest,
        "image_input_revision": image_input_revision,
        "platform": PLATFORM,
        "base_reference": BASE_REFERENCE,
        "software_lock_sha256": sha256_file(requirements),
        "probe_script_sha256": sha256_file(probe),
        "container_policy_sha256": policy_sha256(),
        "official_release": OFFICIAL_RELEASE,
        "official_source_revision": OFFICIAL_REVISION,
        "official_source_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "model_revision": MODEL_REVISION,
        "llama_cpp_revision": LLAMA_REVISION,
        "image_inputs": {
            relative: sha256_file(repository / relative) for relative in IMAGE_INPUTS
        },
        "oci_archive_filename": args.oci_archive.name,
        "oci_archive_sha256": sha256_file(args.oci_archive),
        "sbom_filename": args.sbom.name,
        "sbom_sha256": sha256_file(args.sbom),
        "provenance_filename": args.provenance.name,
        "provenance_sha256": sha256_file(args.provenance),
        "wheel_sha256": WHEEL_SHA256,
    }
    if lock["oci_archive_sha256"] != material.get("oci_archive_sha256"):
        raise ValueError("downloaded OCI archive differs from publication material")
    if lock["sbom_sha256"] != material.get("sbom_sha256"):
        raise ValueError("downloaded SBOM differs from publication material")
    if lock["provenance_sha256"] != material.get("provenance_sha256"):
        raise ValueError("downloaded provenance differs from publication material")
    verify_oci_archive(args.oci_archive, manifest_digest)
    write_new_atomic(args.output, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
