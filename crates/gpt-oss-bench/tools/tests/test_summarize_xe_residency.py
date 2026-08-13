import json
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import summarize_xe_residency


class SummarizeXeResidencyTests(unittest.TestCase):
    def test_summary_accounts_hits_uploads_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "cpu-parity.json"
            profile = root / "execution-profile.json"
            capture.write_text(json.dumps({
                "scenario": "harmony_63", "full_request_seconds": 2.0,
                "xe": {"memory": {"expert_cache_capacity_bytes": 1024}},
                "xe_residency": {
                    "capacity_bytes": 1024, "resident_bytes": 544,
                    "resident_high_water_bytes": 544, "hits": 1, "misses": 1,
                    "bypasses": 0, "evictions": 0, "repacks_avoided": 1,
                    "upload_bytes_avoided": 544, "uploaded_bytes": 544, "faults": 0,
                },
            }))
            profile.write_text(json.dumps({
                "schema": "gpt-oss-rs.execution-profile/v1", "truncated": False,
                "records_dropped": 0,
                "records": [
                    {"operation": "gate_up_projection", "n": 32, "k": 32,
                     "duration_ns": 10, "residency_state": "miss"},
                    {"operation": "gate_up_projection", "n": 32, "k": 32,
                     "duration_ns": 4, "residency_state": "hit"},
                ],
            }))
            value = summarize_xe_residency.summarize([capture])
            row = value["capacities"][0]
            self.assertEqual(row["hit_rate"], 0.5)
            self.assertEqual(row["estimated_total_uploaded_bytes"], 672)
            self.assertEqual(row["projection_median_ns"], 7)

    def test_truncated_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpu-parity.json").write_text(json.dumps({
                "scenario": "x", "full_request_seconds": 1,
                "xe": {"memory": {"expert_cache_capacity_bytes": 0}},
                "xe_residency": {},
            }))
            (root / "execution-profile.json").write_text(json.dumps({
                "schema": "gpt-oss-rs.execution-profile/v1", "truncated": True,
                "records_dropped": 1, "records": [],
            }))
            with self.assertRaisesRegex(ValueError, "truncated"):
                summarize_xe_residency.summarize([root / "cpu-parity.json"])


if __name__ == "__main__":
    unittest.main()
