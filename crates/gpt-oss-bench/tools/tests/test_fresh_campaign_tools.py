import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from run_c3_x001 import first_prefix
from run_tiger_lake_official_matrix import CELLS, SCENARIOS, matrix_commands


def native_capture(prefixes):
    return {
        "dense_boundary_probe_repetitions": 5,
        "dense_boundary_probe_repeat_identical": True,
        "dense_boundary_probe": {
            "projection": "k",
            "output_index": 7,
            "normalized_input_bf16_bits": [1, 2],
            "weight_row_bf16_bits": [3, 4],
            "bias_fp32_bits": 5,
            "prefixes": prefixes,
        },
    }


def official_probe(prefixes):
    return {
        "projection": "k",
        "output_index": 7,
        "normalized_input_bf16_bits": [1, 2],
        "weight_row_bf16_bits": [3, 4],
        "bias_fp32_bits": 5,
        "repetitions": 5,
        "repeat_identical": True,
        "prefixes": prefixes,
    }


class FreshCampaignToolTests(unittest.TestCase):
    def test_tiger_lake_matrix_is_frozen_unique_and_complete(self):
        from argparse import Namespace

        args = Namespace(
            repository=Path("/repo"), root=Path("/campaign"),
            validation_binary=Path("/bin/cpu_validation"),
            native_binary=Path("/bin/cpu_parity"),
            oracle_helper=Path("/repo/oracle/cpu_oracle.py"),
            oracle_lock=Path("/repo/oracle/cpu-oracle.lock.json"),
            model=Path("/model"), fixtures=Path("/repo/fixtures.json"),
            max_new_tokens=8, reserve_gib=20,
        )
        commands = matrix_commands(args)
        self.assertEqual(len(commands), 42)
        self.assertEqual(len(set(CELLS)), 6)
        self.assertEqual(len(SCENARIOS), 7)
        identities = {
            (command[command.index("--scenario") + 1],
             command[command.index("--kernel") + 1],
             command[command.index("--backend") + 1])
            for command in commands
        }
        self.assertEqual(len(identities), 42)

    def test_c3_prefix_localizes_first_differing_arithmetic_field(self):
        native = [
            {"prefix_len": 1, "dot_fp32_bits": 1, "post_bias_fp32_bits": 2, "result_bf16_bits": 3},
            {"prefix_len": 2, "dot_fp32_bits": 9, "post_bias_fp32_bits": 5, "result_bf16_bits": 6},
        ]
        official = [
            {"prefix_len": 1, "dot_fp32_bits": 1, "post_bias_fp32_bits": 2, "result_bf16_bits": 3},
            {"prefix_len": 2, "dot_fp32_bits": 4, "post_bias_fp32_bits": 5, "result_bf16_bits": 6},
        ]
        result = first_prefix(native_capture(native), official_probe(official))
        self.assertEqual(result["prefix_len"], 2)
        self.assertEqual(result["differing_fields"], ["dot_fp32_bits"])

    def test_c3_prefix_rejects_incomplete_repetition_evidence(self):
        prefixes = [
            {"prefix_len": 1, "dot_fp32_bits": 1, "post_bias_fp32_bits": 2, "result_bf16_bits": 3}
        ]
        native = native_capture(prefixes)
        native["dense_boundary_probe_repetitions"] = 1
        with self.assertRaisesRegex(ValueError, "five repeat-identical"):
            first_prefix(native, official_probe(prefixes))


if __name__ == "__main__":
    unittest.main()
