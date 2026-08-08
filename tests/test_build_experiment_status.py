import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_status import build_status


def write_summary(
    path: Path, complete: bool = True, include_end_to_end: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "coverage": {
                    "planned": 2,
                    "completed": 1,
                    "trigger_reached": 1,
                    "trigger_unreached": 1,
                    "execution_complete": complete,
                },
                **(
                    {
                        "end_to_end_metrics": {
                            "control": {"accuracy_on_planned": 0.5},
                            "intervention": {"accuracy_on_planned": 0.0},
                            "paired_robustness_delta_on_planned": -0.5,
                        }
                    }
                    if include_end_to_end
                    else {}
                ),
            }
        )
    )


class BuildExperimentStatusTest(unittest.TestCase):
    def test_registers_new_and_frozen_runs(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            local = root / "runs/model/benchmark_summary.json"
            external = root / "frozen.json"
            write_summary(local)
            write_summary(external)

            status = build_status(
                root,
                [("frozen/intent/model", external)],
                expected_runs=2,
            )

            self.assertTrue(status["paper_experiments_complete"])
            self.assertEqual(status["registered_runs"], 2)
            self.assertIn("runs/model", status["runs"])
            self.assertIn("frozen/intent/model", status["runs"])

    def test_expected_run_count_is_a_hard_gate(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            write_summary(root / "runs/model/benchmark_summary.json")
            with self.assertRaisesRegex(ValueError, "expected 2"):
                build_status(root, [], expected_runs=2)

    def test_legacy_summary_requires_complete_result_rows(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            summary = root / "runs/legacy/benchmark_summary.json"
            write_summary(summary, include_end_to_end=False)
            (summary.parent / "paired_results.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {"control_correct": True, "intervention_correct": False},
                        {"control_correct": False, "intervention_correct": False},
                    )
                )
            )

            status = build_status(root, [], expected_runs=1)

            row = status["runs"]["runs/legacy"]
            self.assertEqual(row["control_accuracy"], 0.5)
            self.assertEqual(row["intervention_accuracy"], 0.0)
            self.assertEqual(row["paired_delta"], -0.5)


if __name__ == "__main__":
    unittest.main()
