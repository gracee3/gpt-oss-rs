import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCANNER = Path(__file__).resolve().parents[1] / "scan_c3_dense_boundary.py"


def capture(key, value):
    return {
        "scenario": "harmony_262",
        "trace": {
            "trace_step": 6,
            "context_token_ids": [1, 2, 3],
            "layers": [
                {
                    "layer_index": 0,
                    "dense_boundary": {
                        "normalized_input_bf16_bits": [10, 11],
                        "key_pre_rope_bf16_bits": key,
                        "value_pre_rope_bf16_bits": value,
                    },
                }
            ],
        },
    }


class ScanC3DenseBoundaryTests(unittest.TestCase):
    def run_scan(self, native_value, official_value):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.json"
            official = root / "official.json"
            output = root / "result.json"
            native.write_text(json.dumps(native_value))
            official.write_text(json.dumps(official_value))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCANNER),
                    "--native",
                    str(native),
                    "--official",
                    str(official),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            return completed, json.loads(output.read_text())

    def test_complete_equality_is_not_reproduced(self):
        completed, result = self.run_scan(
            capture([1, 2], [3, 4]), capture([1, 2], [3, 4])
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result["outcome"], "not_reproduced")
        self.assertEqual(result["status"], "pass")

    def test_k_is_scanned_before_v_then_lowest_output_index(self):
        completed, result = self.run_scan(
            capture([1, 8], [9, 4]), capture([1, 2], [3, 4])
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result["projection"], "k")
        self.assertEqual(result["output_index"], 1)

    def test_normalized_input_mismatch_stops_before_projection(self):
        native = capture([1], [2])
        official = capture([1], [2])
        official["trace"]["layers"][0]["dense_boundary"][
            "normalized_input_bf16_bits"
        ] = [99, 11]
        completed, result = self.run_scan(native, official)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(result["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
