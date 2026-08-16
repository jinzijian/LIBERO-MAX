import copy
import unittest
from pathlib import Path

from libero_max.results import (
    ResultLoadError,
    load_results_jsonl,
    summarize_results,
)
from libero_max.scenario import load_scenarios


ROOT = Path(__file__).resolve().parents[1]


class ResultSummaryTest(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_scenarios([ROOT / "examples/scenarios/pilot.json"])
        self.results = load_results_jsonl(ROOT / "examples/results/pilot_results.jsonl")

    def test_pilot_summary_has_complete_paired_coverage(self):
        summary = summarize_results(self.results, self.scenarios)
        self.assertEqual(summary["coverage"]["planned"], 7)
        self.assertEqual(summary["coverage"]["completed"], 7)
        self.assertTrue(summary["coverage"]["complete"])
        self.assertEqual(
            summary["overall"]["outcome_table"],
            {
                "preserved_capability": 5,
                "intervention_side_gain": 1,
                "regression_under_change": 1,
                "persistent_failure": 0,
            },
        )
        self.assertEqual(summary["overall"]["paired_robustness_delta"], 0.0)
        self.assertEqual(summary["overall"]["regression_rate"], 0.166667)
        self.assertEqual(summary["by_change_type"]["camera_shift"]["episodes"], 1)

    def test_missing_pair_is_reported_without_inflating_metrics(self):
        summary = summarize_results(self.results[:-1], self.scenarios)
        self.assertFalse(summary["coverage"]["complete"])
        self.assertEqual(summary["coverage"]["completed"], 6)
        self.assertEqual(
            summary["coverage"]["missing"], ["feas_remove_receptacle_001:1"]
        )
        self.assertEqual(summary["overall"]["episodes"], 6)

    def test_duplicate_pair_id_is_rejected(self):
        duplicate = copy.deepcopy(self.results[0])
        duplicate["scenario_id"] = "different_scenario"
        with self.assertRaisesRegex(ResultLoadError, "duplicate pair_id"):
            summarize_results(self.results + [duplicate])

    def test_unmeasured_safety_is_not_counted_as_zero_violations(self):
        records = copy.deepcopy(self.results[:2])
        records[0]["safety_violations"] = None
        summary = summarize_results(records)
        self.assertEqual(
            summary["overall"]["safety_measurement_coverage"],
            {"measured": 1, "total": 2},
        )

    def test_exact_paired_statistics_are_reported(self):
        summary = summarize_results(self.results, self.scenarios)
        self.assertEqual(summary["overall"]["mcnemar_exact_two_sided_p"], 1.0)
        self.assertEqual(
            len(summary["overall"]["paired_robustness_delta_95ci_bootstrap"]), 2
        )

    def test_small_mcnemar_p_value_is_not_rounded_to_zero(self):
        records = []
        for index in range(50):
            record = copy.deepcopy(self.results[0])
            record.update(
                {
                    "pair_id": "regression-%d" % index,
                    "scenario_id": "regression-%d" % index,
                    "seed": index,
                    "control_correct": True,
                    "intervention_correct": False,
                }
            )
            records.append(record)
        summary = summarize_results(records)
        self.assertEqual(
            summary["overall"]["mcnemar_exact_two_sided_p"],
            2.0 ** (1 - len(records)),
        )

    def test_response_latency_includes_incorrect_responses(self):
        records = copy.deepcopy(self.results[:2])
        records[0]["intervention_correct"] = False
        records[0]["adaptation_latency_steps"] = 3
        records[1]["intervention_correct"] = True
        records[1]["adaptation_latency_steps"] = 5
        summary = summarize_results(records)
        self.assertEqual(
            summary["overall"]["adaptation_latency_steps"],
            {"count": 2, "mean": 4, "median": 4.0},
        )
        self.assertEqual(
            summary["overall"]["correct_adaptation_latency_steps"],
            {"count": 1, "mean": 5, "median": 5},
        )

    def test_same_randomized_scenario_can_repeat_across_policy_seeds(self):
        records = [copy.deepcopy(self.results[0]), copy.deepcopy(self.results[0])]
        for index, record in enumerate(records):
            record.update(
                {
                    "pair_id": "pair-%d" % index,
                    "task_suite_name": "libero_object",
                    "task_index": 0,
                    "init_state_index": 0,
                    "policy_seed": 195 + index,
                    "intervention_draw_id": 0,
                    "intervention_seed": 123,
                }
            )
        summary = summarize_results(records)
        self.assertEqual(summary["overall"]["episodes"], 2)
        self.assertEqual(summary["by_intervention_draw"]["0"]["episodes"], 2)

    def test_plus_strata_include_null_difficulty_without_sort_errors(self):
        records = [copy.deepcopy(self.results[0]), copy.deepcopy(self.results[1])]
        records[0].update(
            {
                "substrate_category": "Light Conditions",
                "substrate_difficulty": None,
                "dynamic_phase": "pre_grasp_proximity",
            }
        )
        records[1].update(
            {
                "substrate_category": "Sensor Noise",
                "substrate_difficulty": 5,
                "dynamic_phase": "pre_grasp_proximity",
            }
        )
        summary = summarize_results(records)
        self.assertEqual(
            set(summary["by_substrate_difficulty"]), {"5", "None"}
        )


if __name__ == "__main__":
    unittest.main()
