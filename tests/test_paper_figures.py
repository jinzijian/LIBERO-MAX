import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_paper_figures.py"
SPEC = importlib.util.spec_from_file_location("build_paper_figures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PaperFiguresTest(unittest.TestCase):
    def test_summary_uses_full_denominator_and_trigger_coverage(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "benchmark_summary.json").write_text(
                json.dumps(
                    {
                        "coverage": {
                            "planned": 2,
                            "execution_complete": True,
                        }
                    }
                )
            )
            rows = [
                {
                    "control_correct": True,
                    "intervention_correct": False,
                    "trigger_reached": True,
                    "change_type": "camera_shift",
                },
                {
                    "control_correct": False,
                    "intervention_correct": False,
                    "trigger_reached": False,
                    "change_type": "camera_shift",
                },
            ]
            (root / "end_to_end_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            summary = MODULE.summarize_run("model", root)
            self.assertEqual(summary["planned"], 2)
            self.assertEqual(summary["control"], 0.5)
            self.assertEqual(summary["intervention"], 0.0)
            self.assertEqual(summary["trigger_coverage"], 0.5)
            self.assertEqual(summary["by_change_type"]["camera_shift"]["delta"], -0.5)


if __name__ == "__main__":
    unittest.main()
