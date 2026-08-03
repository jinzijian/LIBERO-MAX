import copy
import unittest
from pathlib import Path

from libero_max.runtime import InterventionRuntime, TriggerContext, trigger_satisfied
from libero_max.scenario import load_scenarios


ROOT = Path(__file__).resolve().parents[1]


class RecordingBackend:
    def __init__(self):
        self.changes = []

    def apply_change(self, change):
        self.changes.append(change)
        return {"operation": change["operation"], "applied": True}


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        scenarios = load_scenarios([ROOT / "examples/scenarios/pilot.json"])
        self.camera = copy.deepcopy(scenarios[0])
        self.intent = copy.deepcopy(
            next(item for item in scenarios if item["scenario_id"] == "intent_modify_001")
        )

    def test_fixed_step_fires_exactly_once_after_backend_success(self):
        self.camera["trigger"] = {"type": "fixed_step", "value": 3}
        backend = RecordingBackend()
        runtime = InterventionRuntime(self.camera, backend)
        runtime.reset("original instruction")

        self.assertIsNone(runtime.maybe_apply(TriggerContext(step=2, max_steps=10)))
        event = runtime.maybe_apply(TriggerContext(step=3, max_steps=10))
        self.assertEqual(event["step"], 3)
        self.assertEqual(len(backend.changes), 1)
        self.assertIsNone(runtime.maybe_apply(TriggerContext(step=4, max_steps=10)))
        self.assertEqual(len(backend.changes), 1)
        self.assertEqual(runtime.trace, [event])

    def test_progress_fraction_uses_episode_horizon(self):
        trigger = {"type": "progress_fraction", "value": 0.4}
        self.assertFalse(trigger_satisfied(trigger, TriggerContext(step=3, max_steps=10)))
        self.assertTrue(trigger_satisfied(trigger, TriggerContext(step=4, max_steps=10)))

    def test_semantic_trigger_requires_canonical_event(self):
        trigger = {"type": "after_grasp", "value": "mug"}
        self.assertFalse(
            trigger_satisfied(
                trigger,
                TriggerContext(step=5, max_steps=10, events=frozenset({"grasp:bowl"})),
            )
        )
        self.assertTrue(
            trigger_satisfied(
                trigger,
                TriggerContext(step=5, max_steps=10, events=frozenset({"grasp:mug"})),
            )
        )

    def test_instruction_change_updates_next_policy_instruction(self):
        backend = RecordingBackend()
        runtime = InterventionRuntime(self.intent, backend)
        runtime.reset("Put the red mug in the drawer.")
        event = runtime.maybe_apply(
            TriggerContext(
                step=5,
                max_steps=10,
                events=frozenset({"grasp:red_mug"}),
            )
        )
        self.assertEqual(
            runtime.current_instruction, "Put the red mug on the plate instead."
        )
        self.assertEqual(event["instruction_before"], "Put the red mug in the drawer.")
        self.assertEqual(event["instruction_after"], runtime.current_instruction)
        self.assertEqual(backend.changes, [])

    def test_shared_setup_is_applied_once_before_intervention(self):
        scenario = copy.deepcopy(self.camera)
        scenario["setup"] = [
            {"operation": "remove_object", "object": "distractor_1"}
        ]
        backend = RecordingBackend()
        runtime = InterventionRuntime(scenario, backend)
        runtime.reset("original instruction")
        setup_trace = runtime.apply_setup()
        self.assertEqual(len(setup_trace), 1)
        self.assertEqual(backend.changes, scenario["setup"])
        with self.assertRaisesRegex(RuntimeError, "already applied"):
            runtime.apply_setup()


if __name__ == "__main__":
    unittest.main()
