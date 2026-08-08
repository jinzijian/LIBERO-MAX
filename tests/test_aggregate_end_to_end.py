import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_cosmos_benchmark.py"
SPEC = importlib.util.spec_from_file_location("aggregate_cosmos_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregateEndToEndTest(unittest.TestCase):
    def test_response_query_unreached_is_terminal_not_infrastructure(self):
        control = {
            "init_state_sha256": "same",
            "query_interval": 16,
            "intervention_event_count": 0,
            "policy_queries": [
                {"policy_step": 0, "action_chunk_sha256": "a"},
                {"policy_step": 16, "action_chunk_sha256": "b"},
            ],
        }
        intervention = {
            **control,
            "intervention_event_count": 1,
            "intervention_events": [{"cosmos_query_boundary_step": 30}],
        }

        reasons = MODULE._terminal_trace_reasons(
            control, intervention, {"query_interval": 16}
        )

        self.assertEqual(reasons, ["response_query_unreached"])

    def test_post_trigger_query_missing_does_not_hide_prechange_mismatch(self):
        control = {
            "init_state_sha256": "same",
            "query_interval": 16,
            "intervention_event_count": 0,
            "policy_queries": [{"policy_step": 0, "action_chunk_sha256": "a"}],
        }
        intervention = {
            **control,
            "intervention_event_count": 1,
            "intervention_events": [{"cosmos_query_boundary_step": 10}],
            "policy_queries": [{"policy_step": 0, "action_chunk_sha256": "z"}],
        }

        reasons = MODULE._terminal_trace_reasons(
            control, intervention, {"query_interval": 16}
        )

        self.assertEqual(
            reasons, ["pre_change_action_mismatch", "response_query_unreached"]
        )

    def test_trigger_unreached_failures_remain_in_planned_denominator(self):
        summary = MODULE.summarize_end_to_end_outcomes(
            3,
            {"triggered": True, "unreached-a": False, "unreached-b": False},
            {"triggered": True, "unreached-a": False, "unreached-b": False},
        )

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["control"]["measured"], 3)
        self.assertEqual(summary["control"]["accuracy_on_planned"], 1 / 3)
        self.assertEqual(summary["intervention"]["accuracy_on_planned"], 1 / 3)
        self.assertEqual(summary["outcome_table"]["persistent_failure"], 2)

    def test_infrastructure_gaps_are_missing_not_model_failures(self):
        summary = MODULE.summarize_end_to_end_outcomes(
            3,
            {"measured-a": True, "measured-b": False},
            {"measured-a": False, "measured-b": False},
        )

        self.assertFalse(summary["complete"])
        self.assertEqual(summary["control"]["missing"], 1)
        self.assertEqual(summary["control"]["accuracy_on_measured"], 0.5)
        self.assertIsNone(summary["control"]["accuracy_on_planned"])
        self.assertIsNone(summary["paired_robustness_delta_on_planned"])

    def test_breakdown_keeps_untriggered_case_in_its_group(self):
        cases = [
            {
                "case_id": "a",
                "scenario": {"change_type": "lighting"},
            },
            {
                "case_id": "b",
                "scenario": {"change_type": "lighting"},
            },
            {
                "case_id": "c",
                "scenario": {"change_type": "camera"},
            },
        ]
        breakdown = MODULE.summarize_end_to_end_breakdown(
            cases,
            {"a": True, "b": False, "c": True},
            {"a": False, "b": False, "c": True},
            "scenario.change_type",
        )

        self.assertEqual(breakdown["lighting"]["paired_measured"], 2)
        self.assertEqual(breakdown["lighting"]["control"]["accuracy_on_planned"], 0.5)
        self.assertEqual(
            breakdown["lighting"]["intervention"]["accuracy_on_planned"], 0.0
        )
        self.assertEqual(
            breakdown["camera"]["intervention"]["accuracy_on_planned"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
