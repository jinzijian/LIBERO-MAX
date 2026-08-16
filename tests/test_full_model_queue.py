import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/run_max8000_full_models.sh"


class FullModelQueueTest(unittest.TestCase):
    def test_all_models_use_full_base_plus_pro_denominator(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("libero_max_5600.json", source)
        self.assertIn("libero_max_pro_hard_2400.json", source)
        self.assertIn("libero_max_8000.json", source)
        self.assertIn("build_manifest_complement.py", source)
        self.assertIn("pi05_max8000_q5", source)
        self.assertIn("fastwam_max8000_q16", source)
        self.assertNotIn("lingbot_max8000_q16", source)
        self.assertIn("--expected-runs 15", source)
        self.assertIn('MODELS="${MODELS:-fastwam}"', source)
        self.assertIn("PAPER_QUEUE_FULL_MODELS_DONE", source)
        self.assertIn("BASE runtime smoke passed", source)
        self.assertIn("tables/intent", source)
        self.assertIn("tables/ablation", source)
        self.assertIn('[[ -n "$path" ]] || return 0', source)

    def test_subset_is_reused_not_presented_as_full_result(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("comparison-800", source)
        self.assertIn("pro-remaining-1600", source)
        self.assertIn("compose_paired_run.py", source)
        self.assertIn("model_comparison", source)


if __name__ == "__main__":
    unittest.main()
