import json
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import publication_benchmark


def complete_runs(divergent_lane=None):
    expected = [10, 11, 12, 13, 14, 15, 16, 17]
    runs = []
    for item in publication_benchmark.expected_run_order():
        lane_index = publication_benchmark.LANES.index(item["lane"])
        scalar = item["lane"] == "gpt-oss-rs-scalar"
        tokens = [99] * 8 if item["lane"] == divergent_lane else expected
        runs.append(
            {
                **item,
                "generated_token_ids": tokens,
                "startup_seconds": 10.0 + lane_index,
                "prompt_seconds": (2.0 if scalar else 1.0) + item["round"] * 0.001,
                "ttft_seconds": (2.0 if scalar else 1.0) + item["round"] * 0.001,
                "full_request_seconds": (4.0 if scalar else 2.0) + item["round"] * 0.001,
                "decode_tokens_per_second": (1.0 if scalar else 2.0) + item["round"] * 0.001,
                "peak_rss_kib": 1024 + lane_index,
            }
        )
    return runs, expected


class PublicationBenchmarkTests(unittest.TestCase):
    def test_parses_gnu_time_peak_rss(self):
        value = "\tMaximum resident set size (kbytes): 123456\n"
        self.assertEqual(publication_benchmark.parse_time_verbose(value), 123456)
        with self.assertRaisesRegex(publication_benchmark.BenchmarkError, "peak RSS"):
            publication_benchmark.parse_time_verbose("missing")

    def test_aggregates_and_reports_only_supported_internal_speedups(self):
        runs, expected = complete_runs()
        result = publication_benchmark.aggregate_runs(runs, expected)
        self.assertFalse(result["token_divergence"])
        self.assertEqual(result["summaries"]["gpt-oss-rs-auto"]["samples"], 5)
        comparison = result["internal_comparison"]
        self.assertTrue(comparison["prompt_latency"]["supported"])
        self.assertIsNotNone(comparison["prompt_latency"]["reported_speedup"])
        self.assertTrue(comparison["decode_throughput"]["supported"])

    def test_divergence_is_retained_and_omits_ratios(self):
        runs, expected = complete_runs("llama.cpp-normal")
        result = publication_benchmark.aggregate_runs(runs, expected)
        self.assertTrue(result["token_divergence"])
        self.assertFalse(result["token_identity"]["llama.cpp-normal"]["matches_official"])
        self.assertIsNone(result["internal_comparison"])
        self.assertIn("omitted", result["ratio_policy"])

    def test_rejects_run_order_and_metric_failures(self):
        runs, expected = complete_runs()
        runs[0], runs[1] = runs[1], runs[0]
        with self.assertRaisesRegex(publication_benchmark.BenchmarkError, "run order"):
            publication_benchmark.aggregate_runs(runs, expected)
        runs, expected = complete_runs()
        runs[-1]["prompt_seconds"] = -1
        with self.assertRaisesRegex(publication_benchmark.BenchmarkError, "prompt_seconds"):
            publication_benchmark.aggregate_runs(runs, expected)

    def test_checksum_publication_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.json").write_text(json.dumps({"ok": True}))
            publication_benchmark.write_checksums(root)
            self.assertEqual(publication_benchmark.validate_checksums(root), 1)
            (root / "artifact.json").write_text(json.dumps({"ok": False}))
            with self.assertRaisesRegex(publication_benchmark.BenchmarkError, "checksum"):
                publication_benchmark.validate_checksums(root)


if __name__ == "__main__":
    unittest.main()
