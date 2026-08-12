import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "summarize_xe_promotion.py"
SPEC = importlib.util.spec_from_file_location("summarize_xe_promotion", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeXePromotionTests(unittest.TestCase):
    def write_capture(self, root, sample, side, ttft, full, interval, rss):
        tokens = [11, 12, 13]
        capture = {
            "scenario": "harmony_test",
            "generated_token_ids": tokens,
            "expected_official_greedy_tokens": tokens,
            "time_to_first_token_seconds": ttft,
            "full_request_seconds": full,
            "token_arrival_seconds": [ttft, ttft + interval, ttft + 2 * interval],
            "inter_token_seconds": [interval, interval],
            "xe": None,
        }
        if side == "xe":
            capture["xe"] = {
                "effective_backend": "cpu_xe",
                "identity": {"pci_vendor_id": "8086", "pci_device_id": "9a49"},
                "source_sha256": "a" * 64,
                "abi_sha256": "b" * 64,
                "build_options": "test",
                "gate_up_min_rows": 4,
                "down_min_rows": 4,
                "workgroup_size": 32,
                "memory": {
                    "max_resident_bytes": 128 * 1024 * 1024,
                    "host_staging_bound_bytes": 128 * 1024 * 1024,
                },
            }
        path = root / f"{sample:02d}-{side}.json"
        path.write_text(json.dumps(capture))
        path.with_suffix(".json.manifest.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "source": {
                        "repository_commit": "test",
                        "dirty": False,
                        "cargo_lock_sha256": "a" * 64 if sample <= 8 else "b" * 64,
                    },
                    "artifacts": [
                        {
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        )
        path.with_suffix(".time").write_text(
            f"Maximum resident set size (kbytes): {rss}\n"
            "Major (requiring I/O) page faults: 0\n"
            "Swaps: 0\n"
        )

    def test_passing_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample in range(1, 11):
                self.write_capture(root, sample, "cpu", 12.0, 14.0, 1.0, 100_000)
                self.write_capture(root, sample, "xe", 10.0, 12.0, 1.0, 120_000)
            summary = MODULE.summarize_scenario(root, 1_000, 7)
            self.assertTrue(summary["passing"])
            self.assertGreater(
                summary["intervals"]["ttft_cpu_over_xe"]["lower_95"], 1.0
            )

    def test_token_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_capture(root, 1, "cpu", 12.0, 14.0, 1.0, 100_000)
            self.write_capture(root, 1, "xe", 10.0, 12.0, 1.0, 120_000)
            path = root / "01-xe.json"
            capture = json.loads(path.read_text())
            capture["generated_token_ids"] = [99]
            path.write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, "official oracle"):
                MODULE.summarize_scenario(root, 10, 7)

    def test_incomplete_sample_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample in range(1, 10):
                self.write_capture(root, sample, "cpu", 12.0, 14.0, 1.0, 100_000)
                self.write_capture(root, sample, "xe", 10.0, 12.0, 1.0, 120_000)
            with self.assertRaisesRegex(ValueError, "exactly samples 01 through 10"):
                MODULE.summarize_scenario(root, 10, 7)

    def test_executable_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample in range(1, 11):
                self.write_capture(root, sample, "cpu", 12.0, 14.0, 1.0, 100_000)
                self.write_capture(root, sample, "xe", 10.0, 12.0, 1.0, 120_000)
            for side in ("cpu", "xe"):
                path = root / f"10-{side}.json.manifest.json"
                manifest = json.loads(path.read_text())
                manifest["source"]["repository_commit"] = "other"
                path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "executable source identity drifted"):
                MODULE.summarize_scenario(root, 10, 7)


if __name__ == "__main__":
    unittest.main()
