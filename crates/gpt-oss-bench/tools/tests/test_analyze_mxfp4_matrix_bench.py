import json
import tempfile
import unittest
from pathlib import Path

import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import analyze_mxfp4_matrix_bench


def benchmark(avx2_base=80, avx512_base=70):
    methods = {"scalar": 100, "avx2": avx2_base, "avx512-vnni": avx512_base, "auto": 100}
    samples = []
    correctness = []
    for m in [1, 2, 3, 5]:
        for method in methods:
            correctness.append({
                "m": m, "method": method, "output_sha256": f"hash-{m}", "scalar_exact": True,
            })
        for trial in range(7):
            for sample in range(5):
                for order, (method, base) in enumerate(methods.items()):
                    samples.append({
                        "m": m, "n": 16, "k": 32, "method": method,
                        "trial": trial, "sample": sample, "order": order,
                        "duration_ns": base + ((trial + sample) % 3),
                        "output_sha256": f"hash-{m}",
                    })
    return {
        "schema": analyze_mxfp4_matrix_bench.SCHEMA,
        "repository_commit": "a" * 40,
        "repository_dirty": False,
        "executable_sha256": "b" * 64,
        "cpu_identity": {"family": 6, "model": 140},
        "trials": 7,
        "samples_per_trial": 5,
        "activation": "residual-q8",
        "correctness": correctness,
        "samples": samples,
    }


class AnalyzeMxfp4MatrixBenchTests(unittest.TestCase):
    def test_selects_only_proven_candidate_and_splits_unobserved_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.json"
            path.write_text(json.dumps(benchmark()))
            value = analyze_mxfp4_matrix_bench.analyze([path], iterations=500)
            self.assertEqual(value["promotion_status"], "positive")
            regions = [row for row in value["candidate_regions"] if row["candidate"] == "avx512-vnni"]
            self.assertEqual([(row["m_start"], row["m_end"]) for row in regions], [(1, 3), (5, 5)])
            self.assertFalse(any(row["candidate"] == "avx2" for row in value["candidate_regions"]))

    def test_rejects_dirty_or_under_sampled_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.json"
            value = benchmark()
            value["repository_dirty"] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "dirty"):
                analyze_mxfp4_matrix_bench.analyze([path], iterations=50)
            value["repository_dirty"] = False
            value["trials"] = 6
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "seven"):
                analyze_mxfp4_matrix_bench.analyze([path], iterations=50)


if __name__ == "__main__":
    unittest.main()
