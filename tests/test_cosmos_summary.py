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


def query(step, digest, value):
    return {
        "policy_step": step,
        "action_chunk_sha256": digest,
        "actions": [[value, value + 1.0]],
    }


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


if __name__ == "__main__":
    unittest.main()
