import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_readme_paper_results import END, START, build_section


class UpdateReadmePaperResultsTest(unittest.TestCase):
    def test_complete_results_generate_full_denominator_and_media_section(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "tables/main").mkdir(parents=True)
            (root / "tables/tracks").mkdir(parents=True)
            (root / "tables/model_comparison").mkdir(parents=True)
            (root / "human_review").mkdir()
            (root / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": True})
            )
            (root / "tables/main/main_results.md").write_text("| main |\n")
            (root / "tables/tracks/main_results.md").write_text("| tracks |\n")
            (root / "tables/model_comparison/main_results.md").write_text(
                "| compare |\n"
            )
            (root / "tables/model_comparison/results.json").write_text(
                json.dumps(
                    {
                        "main": [
                            {"model": model, "planned_pairs": 8000}
                            for model in ("Cosmos", "pi0.5", "FastWAM")
                        ]
                    }
                )
            )
            (root / "human_review/human_review_queue.json").write_text(
                json.dumps({"candidate_count": 42, "selected_count": 10})
            )
            section = build_section(root)
            self.assertIn(START, section)
            self.assertIn(END, section)
            self.assertIn("full frozen denominator", section)
            self.assertIn("response-query coverage", section)
            self.assertIn("LaTeX", section)
            self.assertIn("MAX8000_RESULTS.md", section)
            self.assertIn("cosmos-rollout-replay.gif", section)
            self.assertIn("intervention-overview.png", section)
            self.assertIn("Base versus PRO-Hard difficulty", section)
            self.assertIn(
                "Full MAX-8000 native-protocol cross-model comparison", section
            )
            self.assertIn("all 8,000 Base + PRO cases", section)
            self.assertIn("Cosmos, pi0.5, and FastWAM", section)
            self.assertIn("qK means one policy query", section)
            self.assertIn("no test-time updates", section)
            self.assertIn("| tracks |", section)
            self.assertIn("42 cases", section)
            self.assertIn("top 10", section)


if __name__ == "__main__":
    unittest.main()
