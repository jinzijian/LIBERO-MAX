import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ModelRunProvenanceTest(unittest.TestCase):
    def test_cross_model_launchers_record_source_and_runtime(self):
        for name in (
            "run_fastwam_persistent_benchmark.py",
            "run_lingbot_persistent_benchmark.py",
            "run_openpi_persistent_benchmark.sh",
        ):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("source_revision", source)
            self.assertIn("runtime_versions", source)
            self.assertIn("checkpoint_bytes", source)
            self.assertIn("control_replay_before_event", source)


if __name__ == "__main__":
    unittest.main()
