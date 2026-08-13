import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COMPARATOR = Path(__file__).resolve().parents[1] / "compare_cpu_parity.py"


class CompareCpuParityTests(unittest.TestCase):
    ORACLE_IDENTITY = {
        "image_manifest_digest": "1" * 64,
        "image_config_digest": "2" * 64,
        "software_lock_sha256": "3" * 64,
        "official_source_revision": "4" * 40,
        "execution_mode": "native",
        "host_fingerprint": "5" * 64,
        "container_policy_sha256": "6" * 64,
        "probe_artifact_sha256": "7" * 64,
    }

    def run_comparator(
        self,
        native_tokens,
        official_tokens,
        llama=None,
        native_trace=None,
        official_trace=None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.json"
            official = root / "official.json"
            output = root / "comparison.json"
            native.write_text(
                json.dumps(
                    {
                        "scenario": "test_scenario",
                        "generated_token_ids": native_tokens,
                        "trace": native_trace,
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            official.write_text(
                json.dumps(
                    {
                        "generated_token_ids": official_tokens,
                        "trace": official_trace,
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            command = [
                sys.executable,
                str(COMPARATOR),
                "--native",
                str(native),
                "--official",
                str(official),
                "--output",
                str(output),
            ]
            if llama is not None:
                llama_capture = root / "llama.json"
                llama_capture.write_text(json.dumps(llama))
                command.extend(["--llama", str(llama_capture)])

            completed = subprocess.run(command, capture_output=True, text=True)
            return completed, json.loads(output.read_text())

    def test_llama_divergence_is_advisory_when_official_matches(self):
        completed, result = self.run_comparator(
            [11, 12],
            [11, 12],
            {
                "tokens": [91, 92],
                "completion_probabilities": [
                    {
                        "top_logprobs": [
                            {"id": 91, "logprob": -0.1},
                            {"id": 11, "logprob": -12.1},
                        ]
                    }
                ],
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(result["blocking"])
        self.assertEqual(result["llama_cpp"]["policy"], "advisory")
        self.assertEqual(result["llama_cpp"]["first_divergence"], 0)
        self.assertEqual(result["llama_cpp"]["competing_logit_gap"], 12.0)
        self.assertFalse(result["llama_cpp"]["near_tie"])
        self.assertTrue(result["llama_cpp"]["nonblocking"])

    def test_native_official_divergence_is_blocking(self):
        completed, result = self.run_comparator([11, 12], [11, 13])

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertTrue(result["blocking"])
        self.assertFalse(result["native_matches_official"])
        self.assertEqual(result["native_official_first_divergence"], 1)
        self.assertEqual(result["status"], "fail")

    def test_llama_match_cannot_waive_official_divergence(self):
        completed, result = self.run_comparator(
            [11, 12],
            [11, 13],
            {"tokens": [11, 12], "completion_probabilities": []},
        )

        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertTrue(result["blocking"])
        self.assertFalse(result["native_matches_official"])
        self.assertEqual(result["llama_cpp"]["policy"], "advisory")
        self.assertTrue(result["llama_cpp"]["nonblocking"])

    def test_llama_exact_match_is_reported(self):
        completed, result = self.run_comparator(
            [11, 12],
            [11, 12],
            {"tokens": [11, 12], "completion_probabilities": []},
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIsNone(result["llama_cpp"]["first_divergence"])
        self.assertIsNone(result["llama_cpp"]["competing_logit_gap"])
        self.assertFalse(result["llama_cpp"]["near_tie"])

    def test_llama_near_tie_is_reported_but_not_gating(self):
        completed, result = self.run_comparator(
            [11, 12],
            [11, 12],
            {
                "tokens": [11, 13],
                "completion_probabilities": [
                    {"top_logprobs": []},
                    {
                        "top_logprobs": [
                            {"id": 13, "logprob": -0.1},
                            {"id": 12, "logprob": -0.105},
                        ]
                    },
                ],
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(result["blocking"])
        self.assertEqual(result["llama_cpp"]["first_divergence"], 1)
        self.assertAlmostEqual(
            result["llama_cpp"]["competing_logit_gap"], 0.005
        )
        self.assertEqual(result["llama_cpp"]["near_tie_threshold"], 1e-2)
        self.assertTrue(result["llama_cpp"]["near_tie"])

    def test_llama_capture_remains_optional(self):
        completed, result = self.run_comparator([11, 12], [11, 12])

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(result["blocking"])
        self.assertTrue(result["native_matches_official"])
        self.assertNotIn("llama_cpp", result)
        self.assertEqual(result["status"], "pass")

    def test_negative_input_status_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.json"
            official = root / "official.json"
            output = root / "comparison.json"
            native.write_text(
                json.dumps(
                    {
                        "scenario": "test_scenario",
                        "status": "incomplete",
                        "generated_token_ids": [],
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            official.write_text(
                json.dumps(
                    {
                        "generated_token_ids": [],
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
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
            result = json.loads(output.read_text())
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(result["status"], "incomplete")

    def test_mismatched_prompt_provenance_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.json"
            official = root / "official.json"
            output = root / "comparison.json"
            native.write_text(
                json.dumps(
                    {
                        "scenario": "test_scenario",
                        "prompt_token_ids": [1],
                        "generated_token_ids": [2],
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            official.write_text(
                json.dumps(
                    {
                        "prompt_token_ids": [9],
                        "generated_token_ids": [2],
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
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
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(output.read_text())["status"], "invalid")

    def test_decode_step_expert_trace_comparison_preserves_context_and_order(self):
        def trace(gate_up):
            return {
                "trace_step": 6,
                "context_token_ids": [1, 2, 3, 4, 5, 6, 7],
                "layers": [
                    {
                        "layer_index": 0,
                        "router_logits": [0.5, 0.25],
                        "routing_weights": [0.75, 0.25],
                        "experts": [
                            {
                                "rank": 0,
                                "expert_index": 9,
                                "gate_up_projection": gate_up,
                                "swiglu": [0.25],
                                "down_projection": [0.125],
                                "weighted_output": [0.09375],
                            }
                        ],
                        "moe_output": [0.09375],
                        "layer_output": [1.09375],
                    }
                ],
                "final_norm": [1.0],
            }

        completed, result = self.run_comparator(
            [11],
            [11],
            native_trace=trace([1.0]),
            official_trace=trace([1.25]),
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        comparison = result["trace_comparison"]
        self.assertEqual(comparison["native_trace_step"], 6)
        self.assertEqual(comparison["official_trace_step"], 6)
        self.assertTrue(comparison["context_matches"])
        self.assertEqual(
            comparison["earliest_mismatch"],
            {
                "layer_index": 0,
                "expert_rank": 0,
                "expert_index": 9,
                "stage": "gate_up_projection",
                "elements": 1,
                "mean_abs_diff": 0.25,
                "max_abs_diff": 0.25,
            },
        )

    def test_cross_mode_comparator_mixing_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native.json"
            official = root / "official.json"
            output = root / "comparison.json"
            native.write_text(
                json.dumps(
                    {
                        "scenario": "test_scenario",
                        "generated_token_ids": [1],
                        "oracle_identity": self.ORACLE_IDENTITY,
                    }
                )
            )
            generic = dict(self.ORACLE_IDENTITY, execution_mode="generic")
            official.write_text(
                json.dumps(
                    {
                        "generated_token_ids": [1],
                        "oracle_identity": generic,
                    }
                )
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARATOR),
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
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(
                json.loads(output.read_text())["reason"],
                "cross-mode comparator mixing is forbidden",
            )


if __name__ == "__main__":
    unittest.main()
