import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_paper_appendix import TABLES, build_appendix


class BuildPaperAppendixTest(unittest.TestCase):
    def test_requires_complete_runs_and_collects_every_table(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": True})
            )
            review = root / "human_review"
            review.mkdir()
            (review / "human_review_queue.json").write_text(
                json.dumps(
                    {
                        "candidate_count": 42,
                        "selected_count": 10,
                        "human_review_protocol": {
                            "attempts_per_case": 3,
                            "feasible_rule": "one success is feasible",
                            "failure_rule": "require a second reviewer",
                        },
                    }
                )
            )
            analysis = root / "paper/MAX8000_ANALYSIS.md"
            analysis.parent.mkdir()
            analysis.write_text("# analysis\n\nValidated finding.")
            for title, relative, _ in TABLES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("| %s |\n| --- |\n" % title)

            appendix = build_appendix(root)

            self.assertIn("full frozen denominator", appendix)
            self.assertIn("Holm-adjusted", appendix)
            self.assertIn("42 cases", appendix)
            for title, _, _ in TABLES:
                self.assertIn(title, appendix)

    def test_rejects_incomplete_status(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": False})
            )
            with self.assertRaisesRegex(ValueError, "not complete"):
                build_appendix(root)


if __name__ == "__main__":
    unittest.main()
