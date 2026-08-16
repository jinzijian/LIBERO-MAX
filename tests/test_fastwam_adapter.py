import unittest
from pathlib import Path


class FastWAMAdapterTest(unittest.TestCase):
    def test_runner_passes_string_path_to_libero_plus(self):
        source = (
            Path(__file__).parents[1] / "scripts" / "run_fastwam_persistent_shard.py"
        ).read_text(encoding="utf-8")
        self.assertIn("bddl_file_name=str(bddl_path)", source)


if __name__ == "__main__":
    unittest.main()
