import importlib.util
import unittest
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "summarize_xe_projection.py"
SPEC = importlib.util.spec_from_file_location("summarize_xe_projection", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeXeProjectionTests(unittest.TestCase):
    def capture(self, xe_ns):
        samples = []
        for method, elapsed in [
            ("scalar", 500),
            ("cpu_auto", 300),
            ("avx2", 200),
            ("xe", xe_ns),
        ]:
            for trial in range(3):
                for sample in range(30):
                    samples.append(
                        {
                            "trial": trial,
                            "method": method,
                            "sample": sample,
                            "total_ns": elapsed,
                        }
                    )
        return {
            "status": "pass",
            "source": {"repository_commit": "test"},
            "reports": [
                {
                    "projection": "gate_up",
                    "rows": 4,
                    "samples": samples,
                }
            ],
        }

    def test_clear_win_passes(self):
        result = MODULE.summarize(self.capture(100), samples=100, seed=7)
        self.assertEqual(result["automatic_projection_gate"], "pass")
        self.assertGreater(
            result["reports"][0]["ratios"]["cpu_auto_over_xe"]["lower_95"],
            1.0,
        )

    def test_regression_fails(self):
        result = MODULE.summarize(self.capture(400), samples=100, seed=7)
        self.assertEqual(result["automatic_projection_gate"], "fail")
        self.assertIsNone(result["decisions"]["gate_up"]["selected_min_rows"])


if __name__ == "__main__":
    unittest.main()
