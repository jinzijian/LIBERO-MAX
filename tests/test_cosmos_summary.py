import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_cosmos_paired_smoke",
    ROOT / "scripts/summarize_cosmos_paired_smoke.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def query(step, digest, value, input_digest=None):
    result = {
        "policy_step": step,
        "action_chunk_sha256": digest,
        "actions": [[value, value + 1.0]],
    }
    if input_digest is not None:
        result.update(
            {
                "policy_image_sha256": {"agentview": input_digest},
                "sim_state_sha256": input_digest,
            }
        )
    return result


def row(arm, success, queries, events):
    return {
        "arm": arm,
        "scenario_id": "s",
        "task_suite_name": "libero_object",
        "original_task_index": 0,
        "init_state_index": 0,
        "task_description": "task",
        "episode_index": 0,
        "policy_seed": 195,
        "init_state_sha256": "state",
        "query_interval": 16,
        "max_policy_steps": 280,
        "success": success,
        "intervention_event_count": len(events),
        "intervention_events": events,
        "policy_queries": queries,
    }


class CosmosSummaryTest(unittest.TestCase):
    def test_rejects_pre_change_policy_input_mismatch_when_measured(self):
        control = row(
            "control",
            False,
            [query(0, "a", 0.0, "input-a")],
            [],
        )
        intervention = row(
            "intervention",
            False,
            [query(0, "a", 0.0, "input-b")],
            [
                {
                    "cosmos_query_boundary_step": 16,
                    "mean_absolute_raw_pixel_delta": 2.0,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "policy inputs differ"):
            MODULE.summarize_pair(control, intervention)

    def test_accepts_measured_bounded_render_variation_with_exact_physics(self):
        control_query = query(0, "a", 0.0, "input-a")
        intervention_query = query(0, "a", 0.0, "input-b")
        intervention_query["sim_state_sha256"] = "input-a"
        intervention_query["paired_policy_input_qa"] = {
            "status": "passed",
            "sim_state_exact": True,
            "images_byte_exact": False,
        }
        summary = MODULE.summarize_pair(
            row("control", False, [control_query], []),
            row(
                "intervention",
                False,
                [intervention_query],
                [
                    {
                        "cosmos_query_boundary_step": 16,
                        "mean_absolute_raw_pixel_delta": 2.0,
                    }
                ],
            ),
        )
        self.assertTrue(summary["pre_change_policy_inputs_match"])

    def test_proximity_event_maps_to_next_policy_query(self):
        control = row(
            "control",
            True,
            [query(0, "a", 0.0), query(16, "b", 1.0), query(32, "c", 2.0)],
            [],
        )
        intervention = row(
            "intervention",
            False,
            [query(0, "a", 0.0), query(16, "b", 1.0), query(32, "d", 4.0)],
            [
                {
                    "cosmos_query_boundary_step": 30,
                    "mean_absolute_raw_pixel_delta": 2.0,
                }
            ],
        )
        summary = MODULE.summarize_pair(control, intervention)
        self.assertEqual(summary["pre_change_query_steps"], [0, 16])
        self.assertEqual(summary["policy_response_query_step"], 32)
        self.assertEqual(summary["open_loop_exposure_steps"], 2)
        self.assertEqual(summary["post_event_action_chunk_mad"], 2.0)
        self.assertEqual(summary["paired_outcome"], "regression_under_change")

    def test_late_trigger_without_next_query_is_a_terminal_model_outcome(self):
        control = row(
            "control",
            False,
            [query(0, "a", 0.0), query(16, "b", 1.0)],
            [],
        )
        intervention = row(
            "intervention",
            False,
            [query(0, "a", 0.0), query(16, "b", 1.0)],
            [
                {
                    "cosmos_query_boundary_step": 30,
                    "mean_absolute_raw_pixel_delta": 2.0,
                }
            ],
        )

        summary = MODULE.summarize_pair(control, intervention)

        self.assertFalse(summary["response_query_reached"])
        self.assertIsNone(summary["policy_response_query_step"])
        self.assertIsNone(summary["open_loop_exposure_steps"])
        self.assertIsNone(summary["post_event_action_chunk_mad"])
        self.assertEqual(summary["paired_outcome"], "persistent_failure")

    def test_trigger_unreached_is_retained_without_response_summary(self):
        control = row(
            "control",
            False,
            [query(0, "a", 0.0), query(16, "b", 1.0)],
            [],
        )
        intervention = row(
            "intervention",
            False,
            [query(0, "a", 0.0), query(16, "b", 1.0)],
            [],
        )

        summary, terminal = MODULE.classify_persistent_pair(control, intervention)

        self.assertIsNone(summary)
        self.assertEqual(terminal["terminal_status"], "trigger_unreached")
        self.assertTrue(terminal["pre_change_action_chunks_match"])
        self.assertFalse(terminal["response_query_reached"])

    def test_trigger_unreached_still_requires_exact_replay(self):
        control = row("control", False, [query(0, "a", 0.0)], [])
        intervention = row("intervention", False, [query(0, "b", 0.0)], [])

        with self.assertRaisesRegex(ValueError, "action chunks differ"):
            MODULE.classify_persistent_pair(control, intervention)


if __name__ == "__main__":
    unittest.main()
