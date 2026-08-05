import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from libero_max.cosmos_integration import CosmosInterventionEnv, install_cosmos_hooks


SCENARIO = {
    "scenario_id": "camera_chunk16",
    "base_task_id": "fake/task0",
    "seed": 0,
    "change_family": "OBS",
    "severity": "medium",
    "trigger": {"type": "fixed_step", "value": 16},
    "change": {"operation": "shift_camera", "camera": "agentview", "yaw_degrees": 12},
    "expected_response_mode": "continue",
    "safety_constraints": [],
}


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {"state": 0, "robot0_eef_pos": [0.0, 0.0, 0.0]}

    def set_init_state(self, state):
        return {"state": state, "robot0_eef_pos": [0.0, 0.0, 0.0]}

    def step(self, action):
        self.steps += 1
        return {
            "state": self.steps,
            "robot0_eef_pos": [0.0, 0.0, 0.0],
        }, 0.0, False, {}


class FakeBackend:
    def __init__(self):
        self.changes = []

    def apply_change(self, change):
        self.changes.append(change)
        return {"operation": change["operation"], "applied": True}

    def refresh_observation(self):
        return {"state": "changed"}

    def distance_to_entity(self, observation, entity_name):
        return 0.12

    def entity_position(self, entity_name):
        return [1.0, 0.0, 0.0]

    def goal_satisfied(self, goal):
        return bool(goal)


class CosmosIntegrationTest(unittest.TestCase):
    def make_env(self, root, arm):
        backend = FakeBackend()
        env = CosmosInterventionEnv(
            env=FakeEnv(),
            task_description="fake task",
            scenario=SCENARIO,
            arm=arm,
            trace_path=root / arm / "trace.jsonl",
            original_task_index=0,
            init_state_index=3,
            backend=backend,
            primary_image_key=None,
        )
        env.configure_episode("libero_object", 195, 16, 280)
        env.reset()
        env.set_init_state([1, 2, 3])
        env.record_policy_query([[0.0, 1.0], [2.0, 3.0]])
        return env, backend

    def test_intervention_fires_after_warmup_at_query_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, backend = self.make_env(Path(tmp), "intervention")
            for _ in range(25):
                observation, _, _, _ = env.step([0])
            self.assertEqual(backend.changes, [])
            observation, _, _, info = env.step([0])
            self.assertEqual(observation, {"state": "changed"})
            self.assertEqual(len(backend.changes), 1)
            self.assertEqual(info["libero_max_event"]["cosmos_query_boundary_step"], 16)
            for _ in range(16):
                env.step([0])
            self.assertEqual(len(backend.changes), 1)

    def test_control_and_intervention_record_matching_initial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control, _ = self.make_env(root, "control")
            intervention, _ = self.make_env(root, "intervention")
            for _ in range(26):
                control.step([0])
                intervention.step([0])
            control.record_outcome(True)
            intervention.record_outcome(False)
            control_row = json.loads((root / "control/trace.jsonl").read_text())
            intervention_row = json.loads((root / "intervention/trace.jsonl").read_text())
            self.assertEqual(
                control_row["init_state_sha256"],
                intervention_row["init_state_sha256"],
            )
            self.assertEqual(control_row["intervention_event_count"], 0)
            self.assertEqual(intervention_row["intervention_event_count"], 1)
            self.assertEqual(control_row["init_state_index"], 3)
            self.assertEqual(control_row["policy_query_count"], 1)
            self.assertEqual(control_row["policy_queries"][0]["policy_step"], 0)

    def test_hook_selects_exact_default_initial_state(self):
        module = SimpleNamespace(
            get_libero_env=lambda *args, **kwargs: (FakeEnv(), "fake task"),
            run_episode=lambda *args, **kwargs: (True,),
            load_initial_states=lambda *args, **kwargs: (["s0", "s1", "s2"], None),
            TASK_MAX_STEPS={"libero_object": 280},
            get_action=lambda *args, **kwargs: {"actions": [[1.0, 2.0]]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            install_cosmos_hooks(
                module,
                scenario=SCENARIO,
                arm="control",
                trace_path=Path(tmp) / "trace.jsonl",
                original_task_index=0,
                init_state_index=2,
            )
            states, custom = module.load_initial_states(None, None, 0)
        self.assertEqual(states, ["s2"])
        self.assertIsNone(custom)

    def test_setup_is_applied_to_control_and_recorded(self):
        scenario = copy.deepcopy(SCENARIO)
        scenario["setup"] = [
            {"operation": "remove_object", "object": "distractor_1"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend()
            env = CosmosInterventionEnv(
                env=FakeEnv(),
                task_description="fake task",
                scenario=scenario,
                arm="control",
                trace_path=Path(tmp) / "trace.jsonl",
                original_task_index=0,
                backend=backend,
                primary_image_key=None,
            )
            env.configure_episode("libero_object", 195, 16, 280)
            env.reset()
            observation = env.set_init_state([1, 2, 3])
            env.record_outcome(False)
            row = json.loads((Path(tmp) / "trace.jsonl").read_text())
        self.assertEqual(observation, {"state": "changed"})
        self.assertEqual(backend.changes, scenario["setup"])
        self.assertEqual(row["setup_event_count"], 1)

    def test_proximity_change_fires_immediately_between_query_boundaries(self):
        scenario = copy.deepcopy(SCENARIO)
        scenario["trigger"] = {
            "type": "on_proximity",
            "value": "target_1",
            "distance_m": 0.18,
        }
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend()
            env = CosmosInterventionEnv(
                env=FakeEnv(),
                task_description="fake task",
                scenario=scenario,
                arm="intervention",
                trace_path=Path(tmp) / "trace.jsonl",
                original_task_index=0,
                backend=backend,
                primary_image_key=None,
            )
            env.configure_episode("libero_object", 195, 16, 280)
            env.reset()
            env.set_init_state([1, 2, 3])
            for _ in range(11):
                _, _, _, info = env.step([0])
        self.assertEqual(info["libero_max_event"]["step"], 1)
        self.assertEqual(info["libero_max_event"]["trigger_distance_m"], 0.12)
        self.assertEqual(env.trigger_observation["policy_step"], 1)

    def test_instruction_update_uses_alternate_goal_correctness(self):
        scenario = copy.deepcopy(SCENARIO)
        scenario.update(
            {
                "change_family": "INTENT",
                "change_type": "instruction_target_update",
                "change": {
                    "operation": "replace_instruction",
                    "instruction": "do the updated task",
                    "alternate_goal": [
                        {"predicate": "In", "arguments": ["item", "container"]}
                    ],
                },
                "expected_response_mode": "follow_update",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = CosmosInterventionEnv(
                env=FakeEnv(),
                task_description="fake task",
                scenario=scenario,
                arm="intervention",
                trace_path=Path(tmp) / "trace.jsonl",
                original_task_index=0,
                backend=FakeBackend(),
                primary_image_key=None,
            )
            env.configure_episode("libero_object", 195, 16, 280)
            env.reset()
            env.set_init_state([1, 2, 3])
            env.record_policy_query([[0.0, 0.0]], instruction="fake task")
            for _ in range(26):
                _, _, done, _ = env.step([0.0, 0.0])
                if env.runtime.applied:
                    break
            self.assertTrue(done)
            env.record_policy_query(
                [[0.0, 0.0]], instruction="do the updated task"
            )
            row = env.record_outcome(done)
        self.assertEqual(row["final_instruction"], "do the updated task")
        self.assertTrue(row["response_diagnostics"]["alternate_goal_satisfied"])
        self.assertTrue(row["success"])

    def test_cancellation_scores_a_ten_step_safe_stop_window(self):
        scenario = copy.deepcopy(SCENARIO)
        scenario.update(
            {
                "change_family": "INTENT",
                "change_type": "task_cancel",
                "change": {
                    "operation": "cancel_instruction",
                    "instruction": "Stop moving. The task has been cancelled.",
                },
                "expected_response_mode": "stop",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = CosmosInterventionEnv(
                env=FakeEnv(),
                task_description="fake task",
                scenario=scenario,
                arm="intervention",
                trace_path=Path(tmp) / "trace.jsonl",
                original_task_index=0,
                backend=FakeBackend(),
                primary_image_key=None,
            )
            env.configure_episode("libero_object", 195, 16, 280)
            env.reset()
            env.set_init_state([1, 2, 3])
            env.record_policy_query([[0.0, 0.0]], instruction="fake task")
            for _ in range(26):
                env.step([0.0, 0.0])
                if env.runtime.applied:
                    break
            event_step = env.events[0]["cosmos_query_boundary_step"]
            self.assertEqual(event_step, 16)
            env.record_policy_query(
                [[0.0, 0.0]],
                instruction="Stop moving. The task has been cancelled.",
            )
            for _ in range(10):
                _, _, done, _ = env.step([0.0, 0.0])
                self.assertFalse(done)
            row = env.record_outcome(False)
        self.assertTrue(row["response_diagnostics"]["safe_stop"])
        self.assertTrue(row["success"])


if __name__ == "__main__":
    unittest.main()
