import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_paper_analysis import build_analysis


def _row(model, change=0.5, delta=-0.25):
    return {
        "model": model,
        "control_accuracy": change - delta,
        "control_accuracy_95ci": [0.6, 0.9],
        "intervention_accuracy": change,
        "intervention_accuracy_95ci": [0.3, 0.7],
        "paired_delta": delta,
        "paired_delta_95ci_full": [delta - 0.1, delta + 0.1],
        "trigger_coverage": 0.8,
        "response_coverage": 0.7,
        "preserved_capability": 4,
        "recoveries": 1,
        "regressions": 3,
        "persistent_failure": 2,
    }


class BuildPaperAnalysisTest(unittest.TestCase):
    def test_generates_evidence_bound_analysis_sections(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": True})
            )
            tables = root / "tables"
            for name in ("main", "tracks", "model_comparison", "intent", "ablation"):
                (tables / name).mkdir(parents=True)
            main = {
                "main": [_row("Cosmos-Policy-q16"), _row("pi0.5-LIBERO-q5", 0.6)],
                "by_change_type": [
                    {
                        "model": "Cosmos-Policy-q16",
                        "change_type": "target_relocation",
                        "intervention_accuracy": 0.2,
                        "paired_delta": -0.5,
                    },
                    {
                        "model": "pi0.5-LIBERO-q5",
                        "change_type": "obstacle_insertion",
                        "intervention_accuracy": 0.3,
                        "paired_delta": -0.4,
                    },
                ],
                "by_type_draw": [
                    {
                        "model": "Cosmos-Policy-q16",
                        "change_type": "target_relocation",
                        "draw_id": 0,
                        "intervention_accuracy": 0.2,
                    },
                    {
                        "model": "Cosmos-Policy-q16",
                        "change_type": "target_relocation",
                        "draw_id": 1,
                        "intervention_accuracy": 0.3,
                    },
                ],
            }
            tracks = {
                "main": [
                    _row("Cosmos-Base-q16", 0.7),
                    _row("Cosmos-PRO-q16", 0.4),
                    _row("pi0.5-Base-q5", 0.8),
                    _row("pi0.5-PRO-q5", 0.5),
                ]
            }
            comparison = {
                "main": [_row("A", 0.4), _row("B", 0.7)],
                "model_comparisons": [
                    {
                        "left": "A",
                        "right": "B",
                        "right_minus_left": 0.3,
                        "right_minus_left_95ci": [0.1, 0.5],
                        "mcnemar_p_holm": 0.02,
                    }
                ],
            }
            for name, payload in (
                ("main", main),
                ("tracks", tracks),
                ("model_comparison", comparison),
                ("intent", {"main": [_row("Intent")]}),
                ("ablation", {"main": [_row("Ablation")]}),
            ):
                (tables / name / "results.json").write_text(json.dumps(payload))
            review = root / "human_review"
            review.mkdir()
            (review / "human_review_queue.json").write_text(
                json.dumps({"candidate_count": 42, "selected_count": 10})
            )

            analysis = build_analysis(root)

            self.assertIn("Headline end-to-end result", analysis)
            self.assertIn("Does PRO-Hard increase difficulty?", analysis)
            self.assertIn("target_relocation", analysis)
            self.assertIn("Holm-adjusted exact McNemar p=0.02", analysis)
            self.assertIn("42 cases", analysis)

    def test_rejects_incomplete_experiment_status(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "experiment_status.json").write_text(
                json.dumps({"paper_experiments_complete": False})
            )
            with self.assertRaisesRegex(ValueError, "not complete"):
                build_analysis(root)


if __name__ == "__main__":
    unittest.main()
