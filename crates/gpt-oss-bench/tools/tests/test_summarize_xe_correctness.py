import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "summarize_xe_correctness.py"
SPEC = importlib.util.spec_from_file_location("summarize_xe_correctness", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeXeCorrectnessTests(unittest.TestCase):
    def build_fixture(self, root):
        runner = root / "runner"
        runner.write_bytes(b"runner")
        runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
        record = {
            "schema": "gpt-oss-rs.xe-auto-promotion/v1",
            "pci_vendor_id": "8086",
            "pci_device_id": "9a49",
            "driver_version": "test-driver",
            "opencl_loader_sha256": "a" * 64,
            "opencl_driver_sha256": "b" * 64,
            "igc_sha256": "c" * 64,
            "kernel_source_sha256": "d" * 64,
            "kernel_abi_sha256": "e" * 64,
            "build_options": "test-options",
            "gate_up_min_rows": 4,
            "down_min_rows": 4,
            "workgroup_size": 32,
        }
        record_path = root / "record.json"
        record_path.write_text(json.dumps(record))
        source = {
            "repository_commit": "f" * 40,
            "dirty": False,
            "branch_role": "candidate",
            "cargo_lock_sha256": "1" * 64,
            "toolchain": "rustc test",
            "profile": "release",
            "features": ["xe"],
        }
        for scenario in MODULE.SCENARIOS:
            for side in ("cpu", "xe"):
                capture = {
                    "scenario": scenario,
                    "generated_token_ids": [1, 2, 3],
                    "expected_official_greedy_tokens": [1, 2, 3],
                    "executable_sha256": runner_hash,
                    "full_request_seconds": 1.0,
                    "xe": None,
                }
                if side == "xe":
                    capture["xe"] = {
                        "effective_backend": "cpu_xe",
                        "validation_class": "validated_explicit",
                        "source_sha256": record["kernel_source_sha256"],
                        "abi_sha256": record["kernel_abi_sha256"],
                        "build_options": record["build_options"],
                        "gate_up_min_rows": 4,
                        "down_min_rows": 4,
                        "workgroup_size": 32,
                        "identity": {
                            key: record[key]
                            for key in (
                                "pci_vendor_id",
                                "pci_device_id",
                                "driver_version",
                                "opencl_loader_sha256",
                                "opencl_driver_sha256",
                                "igc_sha256",
                            )
                        },
                        "memory": {
                            "device_resident_bytes": 100,
                            "max_resident_bytes": 200,
                        },
                    }
                path = root / f"{scenario}-{side}.json"
                path.write_text(json.dumps(capture))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                path.with_suffix(".json.manifest.json").write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "source": source,
                            "artifacts": [{"sha256": digest}],
                        }
                    )
                )
        return runner, record_path

    def test_clean_exact_matrix_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, record = self.build_fixture(root)
            result = MODULE.summarize(root, runner, record)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["scenario_count"], 7)

    def test_oracle_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, record = self.build_fixture(root)
            path = root / "harmony_122-xe.json"
            capture = json.loads(path.read_text())
            capture["generated_token_ids"] = [9]
            path.write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, "official oracle"):
                MODULE.summarize(root, runner, record)


if __name__ == "__main__":
    unittest.main()
