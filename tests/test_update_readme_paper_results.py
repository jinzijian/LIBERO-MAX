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
            (root / "tables/model_comparison").mkdir(parents=True)
            (root / "human_review").mkdir()
            (root / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": True})
            )
            (root / "tables/main/main_results.md").write_text("| main |\n")
            (root / "tables/model_comparison/main_results.md").write_text(
                "| compare |\n"
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
            self.assertIn("cosmos-rollout-replay.gif", section)
            self.assertIn("42 cases", section)
            self.assertIn("top 10", section)


if __name__ == "__main__":
    unittest.main()
