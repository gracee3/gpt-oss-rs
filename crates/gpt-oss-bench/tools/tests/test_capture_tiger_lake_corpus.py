import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import capture_tiger_lake_corpus


class CaptureTigerLakeCorpusTests(unittest.TestCase):
    def test_command_requests_layer_major_bounded_profile(self):
        args = Namespace(
            cpus="0-3", binary=Path("cpu_parity"), model=Path("model"),
            repack_cache=Path("cache"), fixtures=Path("fixtures.json"),
            kernel="auto", cpu_matmul_backend="auto", threads=4,
            max_new_tokens=8, profile_cap_mib=16,
            xe=False, xe_max_resident_mib=128, xe_expert_cache_mib=0,
            xe_warmup_prefills=0,
        )
        command = capture_tiger_lake_corpus.command_for(
            args, "harmony_63", Path("profile.json"), Path("output.json")
        )
        self.assertEqual(command[:3], ["taskset", "-c", "0-3"])
        self.assertIn("--layer-major-prefill", command)
        self.assertEqual(command[command.index("--cpu-profile-cap-mib") + 1], "16")

    def test_xe_command_is_explicit_and_capacity_bounded(self):
        args = Namespace(
            cpus="0-3", binary=Path("cpu_parity"), model=Path("model"),
            repack_cache=Path("cache"), fixtures=Path("fixtures.json"),
            kernel="auto", cpu_matmul_backend="auto", threads=4,
            max_new_tokens=8, profile_cap_mib=16,
            xe=True, xe_max_resident_mib=128, xe_expert_cache_mib=256,
            xe_warmup_prefills=1,
        )
        command = capture_tiger_lake_corpus.command_for(
            args, "harmony_63", Path("profile.json"), Path("output.json")
        )
        self.assertEqual(command[command.index("--xe-max-resident-mib") + 1], "128")
        self.assertEqual(command[command.index("--xe-expert-cache-mib") + 1], "256")
        self.assertEqual(command[command.index("--xe-warmup-prefills") + 1], "1")
        self.assertEqual(command.count("--xe"), 1)

    def test_artifact_index_is_sorted_and_does_not_hash_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").write_text("z")
            (root / "a").write_text("a")
            digest = capture_tiger_lake_corpus.write_artifact_index(root)
            lines = (root / "SHA256SUMS").read_text().splitlines()
            self.assertEqual([line.split("  ", 1)[1] for line in lines], ["a", "z"])
            self.assertEqual(digest, capture_tiger_lake_corpus.sha256(root / "SHA256SUMS"))


if __name__ == "__main__":
    unittest.main()
