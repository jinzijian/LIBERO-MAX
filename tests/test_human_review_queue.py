import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_human_review_queue.py"
SPEC = importlib.util.spec_from_file_location("human_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HumanReviewQueueTest(unittest.TestCase):
    def test_csv_writer_uses_repository_line_endings(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('lineterminator="\\n"', source)

    def test_all_model_control_failures_raise_review_priority(self):
        case = {
            "case_id": "p0",
            "task_suite_name": "libero_10",
            "task_index": 0,
            "init_state_index": 0,
            "substrate_category": "LIBERO-PRO/view_occlusion",
            "scenario": {
                "change_type": "obstacle_insertion",
                "severity": "high",
                "randomization": {"draw_id": 0},
            },
        }
        row = MODULE._score_case(
            case,
            {"task_description": "put both objects in the basket"},
            {"model-a": False, "model-b": False},
        )
        self.assertGreaterEqual(row["risk_score"], 15.0)
        self.assertIn("all evaluated model controls failed", row["risk_signals"])

    def test_successful_control_is_recorded_as_counterevidence(self):
        case = {
            "case_id": "p0",
            "task_suite_name": "libero_goal",
            "task_index": 0,
            "init_state_index": 0,
            "substrate_category": "base",
            "scenario": {
                "change_type": "illumination_switch",
                "severity": "high",
                "randomization": {"draw_id": 0},
            },
        }
        row = MODULE._score_case(
            case,
            {"task_description": "turn on the stove"},
            {"model-a": True},
        )
        self.assertEqual(row["successful_model_controls"], ["model-a"])
        self.assertIn(
            "at least one model completed the matched control", row["risk_signals"]
        )

    def test_successful_changed_episode_is_stronger_feasibility_evidence(self):
        case = {
            "case_id": "p0",
            "task_suite_name": "libero_goal",
            "task_index": 0,
            "init_state_index": 0,
            "substrate_category": "base",
            "scenario": {
                "change_type": "obstacle_insertion",
                "severity": "high",
                "randomization": {"draw_id": 0},
            },
        }
        row = MODULE._score_case(
            case,
            {"task_description": "pick up the bowl"},
            {
                "model-a": {"control": True, "intervention": True},
                "model-b": {"control": True, "intervention": False},
            },
        )
        self.assertEqual(row["successful_model_interventions"], ["model-a"])
        self.assertIn(
            "at least one model completed the changed episode", row["risk_signals"]
        )

    def test_missing_substrate_runtime_is_not_a_topology_risk(self):
        case = {
            "case_id": "base-0",
            "task_suite_name": "libero_object",
            "task_index": 0,
            "init_state_index": 0,
            "substrate_category": "Objects Layout",
            "scenario": {
                "change_type": "illumination_switch",
                "severity": "high",
                "randomization": {"draw_id": 0},
            },
        }
        row = MODULE._score_case(
            case,
            {"task_description": "pick up the bowl"},
            {},
        )
        self.assertNotIn("topology-extended state adapter", row["risk_signals"])

    def test_failure_evidence_is_normalized_across_model_coverage(self):
        case = {
            "case_id": "p0",
            "task_suite_name": "libero_goal",
            "task_index": 0,
            "init_state_index": 0,
            "substrate_category": "base",
            "scenario": {
                "change_type": "camera_shift",
                "severity": "high",
                "randomization": {"draw_id": 0},
            },
        }
        preflight = {"task_description": "pick up the bowl"}
        two_models = MODULE._score_case(
            case,
            preflight,
            {"a": False, "b": False},
        )
        four_models = MODULE._score_case(
            case,
            preflight,
            {"a": False, "b": False, "c": False, "d": False},
        )
        self.assertEqual(two_models["risk_score"], four_models["risk_score"])
        self.assertEqual(two_models["control_failure_rate"], 1.0)
        self.assertEqual(four_models["control_failure_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
