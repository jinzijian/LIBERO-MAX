import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_paper_tables


class PaperTablesTest(unittest.TestCase):
    def test_distinguishes_trigger_from_response_query_coverage(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run = root / "run"
            output = root / "tables"
            run.mkdir()
            (run / "benchmark_summary.json").write_text(
                json.dumps(
                    {
                        "coverage": {
                            "planned": 2,
                            "completed": 1,
                            "execution_complete": True,
                        },
                        "protocol": {"scoring_track": "physical_completion"},
                        "end_to_end_metrics": {
                            "control": {"accuracy_on_planned": 0.5},
                            "intervention": {"accuracy_on_planned": 0.0},
                            "paired_robustness_delta_on_planned": -0.5,
                        },
                        "metrics": {
                            "overall": {
                                "paired_robustness_delta": -1.0,
                                "paired_robustness_delta_95ci_bootstrap": [
                                    -1.0,
                                    -1.0,
                                ],
                                "safety_measurement_coverage": {
                                    "measured": 0,
                                    "total": 1,
                                },
                                "safety_violation_rate": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            end_to_end = [
                {
                    "pair_id": "responded",
                    "control_correct": True,
                    "intervention_correct": False,
                    "trigger_reached": True,
                    "response_query_reached": True,
                    "change_type": "camera_shift",
                    "severity": "high",
                    "task_suite_name": "libero_spatial",
                },
                {
                    "pair_id": "late-trigger",
                    "control_correct": False,
                    "intervention_correct": False,
                    "trigger_reached": True,
                    "response_query_reached": False,
                    "change_type": "camera_shift",
                    "severity": "high",
                    "task_suite_name": "libero_spatial",
                },
            ]
            (run / "end_to_end_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in end_to_end),
                encoding="utf-8",
            )
            (run / "paired_results.jsonl").write_text(
                json.dumps(
                    {
                        **end_to_end[0],
                        "open_loop_exposure_steps": 2,
                        "post_event_action_chunk_mad": 0.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "build_paper_tables.py",
                    "--run",
                    "Model=%s" % run,
                    "--output-dir",
                    str(output),
                ],
            ):
                build_paper_tables.main()

            table = (output / "main_results.md").read_text(encoding="utf-8")
            self.assertIn("2/2 (100.0)", table)
            self.assertIn("1/2 (50.0)", table)


if __name__ == "__main__":
    unittest.main()
