import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ORACLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORACLE))

import cpu_oracle


class CpuOracleValidationTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        repository = root / "repository"
        repository.mkdir()
        (repository / "oracle").mkdir()
        requirements = repository / "oracle/requirements.cpu.lock"
        probe = repository / "oracle/environment_probe.py"
        requirements.write_text("torch==2.12.1+cpu\n")
        probe.write_text("probe\n")

        manifest = b'{"schemaVersion":2}'
        manifest_digest = cpu_oracle.sha256_bytes(manifest)
        archive = root / "oracle.oci.tar"
        with tarfile.open(archive, "w") as output:
            values = {
                "oci-layout": b'{"imageLayoutVersion":"1.0.0"}\n',
                "index.json": json.dumps(
                    {
                        "schemaVersion": 2,
                        "manifests": [
                            {
                                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                                "digest": f"sha256:{manifest_digest}",
                                "size": len(manifest),
                            }
                        ],
                    }
                ).encode(),
                f"blobs/sha256/{manifest_digest}": manifest,
            }
            for name, value in values.items():
                info = tarfile.TarInfo(name)
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))

        lock = {
            "schema": cpu_oracle.LOCK_SCHEMA,
            "image_name": cpu_oracle.IMAGE_NAME,
            "image_reference": f"{cpu_oracle.IMAGE_NAME}@sha256:{manifest_digest}",
            "image_manifest_digest": manifest_digest,
            "image_config_digest": "1" * 64,
            "image_input_revision": "2" * 40,
            "platform": cpu_oracle.PLATFORM,
            "base_reference": cpu_oracle.BASE_REFERENCE,
            "software_lock_sha256": cpu_oracle.sha256_file(requirements),
            "probe_script_sha256": cpu_oracle.sha256_file(probe),
            "container_policy_sha256": cpu_oracle.policy_sha256(),
            "official_release": cpu_oracle.OFFICIAL_RELEASE,
            "official_source_revision": cpu_oracle.OFFICIAL_REVISION,
            "official_source_archive_sha256": cpu_oracle.OFFICIAL_ARCHIVE_SHA256,
            "model_revision": cpu_oracle.MODEL_REVISION,
            "llama_cpp_revision": cpu_oracle.LLAMA_REVISION,
            "image_inputs": {
                "oracle/requirements.cpu.lock": cpu_oracle.sha256_file(requirements),
                "oracle/environment_probe.py": cpu_oracle.sha256_file(probe),
            },
            "wheel_sha256": cpu_oracle.WHEEL_SHA256,
            "oci_archive_sha256": cpu_oracle.sha256_file(archive),
            "sbom_sha256": "3" * 64,
            "provenance_sha256": "4" * 64,
        }
        lock_path = repository / "oracle/cpu-oracle.lock.json"
        for relative in cpu_oracle.IMAGE_INPUTS:
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"fixture for {relative}\n")
            lock["image_inputs"][relative] = cpu_oracle.sha256_file(path)
        lock_path.write_text(json.dumps(lock))
        return temporary, repository, archive, lock_path, lock

    def test_exact_lock_and_archive_are_accepted(self):
        temporary, repository, archive, lock_path, _ = self.fixture()
        with temporary:
            cpu_oracle.verify_lock(lock_path, repository, archive)

    def mutate_and_reject(self, key, value):
        temporary, repository, archive, lock_path, lock = self.fixture()
        with temporary:
            lock[key] = value
            lock_path.write_text(json.dumps(lock))
            with self.assertRaises(cpu_oracle.ValidationError):
                cpu_oracle.verify_lock(lock_path, repository, archive)

    def test_mutable_and_wrong_image_references_are_rejected(self):
        self.mutate_and_reject(
            "image_reference", f"{cpu_oracle.IMAGE_NAME}:v0.0.9"
        )
        self.mutate_and_reject(
            "image_reference", f"example.invalid/oracle@sha256:{'a' * 64}"
        )

    def test_wrong_platform_and_source_revision_are_rejected(self):
        self.mutate_and_reject("platform", "linux/arm64")
        self.mutate_and_reject("official_source_revision", "a" * 40)

    def test_changed_software_lock_is_rejected(self):
        temporary, repository, archive, lock_path, _ = self.fixture()
        with temporary:
            (repository / "oracle/requirements.cpu.lock").write_text("changed\n")
            with self.assertRaisesRegex(cpu_oracle.ValidationError, "lock changed"):
                cpu_oracle.verify_lock(lock_path, repository, archive)

    def test_incomplete_image_inputs_and_wrong_wheel_hashes_are_rejected(self):
        temporary, repository, archive, lock_path, lock = self.fixture()
        with temporary:
            lock["image_inputs"].pop("oracle/Dockerfile.cpu")
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(cpu_oracle.ValidationError, "build-input set"):
                cpu_oracle.verify_lock(lock_path, repository, archive)
        self.mutate_and_reject("wheel_sha256", {"torch": "a" * 64})

    def test_corrupt_oci_archive_is_rejected(self):
        temporary, repository, archive, lock_path, lock = self.fixture()
        with temporary:
            archive.write_bytes(archive.read_bytes()[:100])
            lock["oci_archive_sha256"] = cpu_oracle.sha256_file(archive)
            lock_path.write_text(json.dumps(lock))
            with self.assertRaises(cpu_oracle.ValidationError):
                cpu_oracle.verify_lock(lock_path, repository, archive)

    def test_daemon_failure_and_wrong_architecture_are_rejected(self):
        lock = {
            "image_reference": f"{cpu_oracle.IMAGE_NAME}@sha256:{'a' * 64}",
            "image_manifest_digest": "a" * 64,
            "image_config_digest": "b" * 64,
            "image_input_revision": "c" * 40,
        }
        with mock.patch.object(cpu_oracle, "docker_group_active", return_value=True), mock.patch.object(
            cpu_oracle.subprocess, "run", side_effect=OSError("daemon unavailable")
        ):
            with self.assertRaisesRegex(cpu_oracle.ValidationError, "daemon"):
                cpu_oracle.docker_inspect(lock)

        image = [
            {
                "Id": f"sha256:{'b' * 64}",
                "Os": "linux",
                "Architecture": "arm64",
                "RepoDigests": [lock["image_reference"]],
                "Config": {"Labels": {}},
            }
        ]
        completed = mock.Mock(stdout=json.dumps(image))
        with mock.patch.object(cpu_oracle, "docker_group_active", return_value=True), mock.patch.object(
            cpu_oracle.subprocess, "run", return_value=completed
        ):
            with self.assertRaisesRegex(cpu_oracle.ValidationError, "platform"):
                cpu_oracle.docker_inspect(lock)

    def test_containerd_index_identity_requires_the_locked_descriptor(self):
        lock = {
            "image_reference": f"{cpu_oracle.IMAGE_NAME}@sha256:{'a' * 64}",
            "image_manifest_digest": "a" * 64,
            "image_config_digest": "b" * 64,
            "image_input_revision": "c" * 40,
        }
        labels = {
            "org.opencontainers.image.revision": "c" * 40,
            "io.gpt-oss-rs.oracle.platform": cpu_oracle.PLATFORM,
            "io.gpt-oss-rs.oracle.official-source-revision": cpu_oracle.OFFICIAL_REVISION,
            "io.gpt-oss-rs.oracle.official-source-sha256": cpu_oracle.OFFICIAL_ARCHIVE_SHA256,
            "io.gpt-oss-rs.oracle.model-revision": cpu_oracle.MODEL_REVISION,
        }
        image = [
            {
                "Id": f"sha256:{'a' * 64}",
                "Os": "linux",
                "Architecture": "amd64",
                "RepoDigests": [lock["image_reference"]],
                "Descriptor": {
                    "digest": f"sha256:{'a' * 64}",
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                },
                "Config": {"Labels": labels},
            }
        ]
        completed = mock.Mock(stdout=json.dumps(image))
        with mock.patch.object(
            cpu_oracle, "docker_group_active", return_value=True
        ), mock.patch.object(cpu_oracle.subprocess, "run", return_value=completed):
            result = cpu_oracle.docker_inspect(lock)
        self.assertEqual(result["storage_identity"], "containerd-index")
        image[0]["Descriptor"]["digest"] = f"sha256:{'d' * 64}"
        completed = mock.Mock(stdout=json.dumps(image))
        with mock.patch.object(
            cpu_oracle, "docker_group_active", return_value=True
        ), mock.patch.object(cpu_oracle.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(cpu_oracle.ValidationError, "digest"):
                cpu_oracle.docker_inspect(lock)

    def test_cuda_visibility_and_incomplete_probe_are_rejected(self):
        lock = {
            "image_reference": f"{cpu_oracle.IMAGE_NAME}@sha256:{'a' * 64}",
            "software_lock_sha256": "b" * 64,
        }
        cuda_probe = {
            "schema": "gpt-oss-rs.cpu-oracle-probe/v1",
            "execution_mode": "native",
            "repetitions": 5,
            "repeat_identical": True,
            "software_lock_sha256": "b" * 64,
            "torch": {"cuda_available": True, "cuda_version": "13.0"},
        }
        completed = mock.Mock(
            returncode=0, stdout=json.dumps(cuda_probe).encode(), stderr=b""
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            cpu_oracle.subprocess, "run", return_value=completed
        ):
            path = Path(directory)
            with self.assertRaisesRegex(cpu_oracle.ValidationError, "CUDA"):
                cpu_oracle.run_probe(lock, path, path, path, "native")

        with self.assertRaisesRegex(cpu_oracle.ValidationError, "incomplete"):
            cpu_oracle.validate_probe_pair({"native": {}, "generic": {}})

    def test_host_key_mismatch_is_rejected(self):
        probes = {
            "native": {
                "record": {
                    "host_fingerprint": "a" * 64,
                    "torch": {"cpu_capability": "AVX512"},
                }
            },
            "generic": {
                "record": {
                    "host_fingerprint": "b" * 64,
                    "torch": {"cpu_capability": "DEFAULT"},
                }
            },
        }
        with self.assertRaisesRegex(cpu_oracle.ValidationError, "host keys"):
            cpu_oracle.validate_probe_pair(probes)


if __name__ == "__main__":
    unittest.main()
