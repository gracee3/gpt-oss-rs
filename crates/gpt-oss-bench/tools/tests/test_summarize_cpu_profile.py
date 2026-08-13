import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import summarize_cpu_profile


def profile(records, dropped=0):
    return {
        "schema": summarize_cpu_profile.SCHEMA,
        "records_written": len(records),
        "records_dropped": dropped,
        "truncated": dropped != 0,
        "records_sha256": hashlib.sha256(summarize_cpu_profile.canonical(records)).hexdigest(),
        "records": records,
    }


class SummarizeCpuProfileTests(unittest.TestCase):
    def record(self, operation="gate_up_projection", state="prepared", m=4):
        return {
            "phase": "prefill", "operation": operation, "m": m, "n": 8, "k": 32,
            "effective_matrix_backend": 2, "projection_role": "gate_up",
            "expert_bucket_m": m, "duration_ns": 100,
            "scratch_high_water_bytes": 64, "resident_high_water_bytes": 0,
            "transaction_state": state,
        }

    def test_summary_is_deterministic_and_excludes_failed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile([self.record(), self.record(state="failed")])))
            first = summarize_cpu_profile.summarize([path])
            second = summarize_cpu_profile.summarize([path])
            self.assertEqual(first, second)
            self.assertEqual(first["failed_transaction_records"], 1)
            self.assertEqual(first["operations"][0]["duration_ns"], 100)

    def test_truncated_and_hash_mismatched_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile([self.record()], dropped=1)))
            with self.assertRaisesRegex(ValueError, "truncated"):
                summarize_cpu_profile.summarize([path])
            value = profile([self.record()])
            value["records_sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "hash"):
                summarize_cpu_profile.summarize([path])


if __name__ == "__main__":
    unittest.main()
