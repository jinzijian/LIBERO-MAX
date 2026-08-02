import json
import tempfile
import unittest
from pathlib import Path

from libero_max.cosmos_integration import CosmosInterventionEnv


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
        return {"state": 0}

    def set_init_state(self, state):
        return {"state": state}

    def step(self, action):
        self.steps += 1
        return {"state": self.steps}, 0.0, False, {}


class FakeBackend:
    def __init__(self):
        self.changes = []

    def apply_change(self, change):
        self.changes.append(change)
        return {"operation": change["operation"], "applied": True}

    def refresh_observation(self):
        return {"state": "changed"}


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
            backend=backend,
            primary_image_key=None,
        )
        env.configure_episode("libero_object", 195, 16, 280)
        env.reset()
        env.set_init_state([1, 2, 3])
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


if __name__ == "__main__":
    unittest.main()
