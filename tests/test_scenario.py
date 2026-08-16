import copy
import unittest
from pathlib import Path

from libero_max.scenario import (
    load_scenarios,
    validate_scenario,
    validate_scenario_collection,
)


ROOT = Path(__file__).resolve().parents[1]


class ScenarioValidationTest(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_scenarios([ROOT / "examples/scenarios/pilot.json"])

    def test_pilot_is_valid_and_covers_all_response_modes(self):
        self.assertEqual(validate_scenario_collection(self.scenarios), [])
        self.assertEqual(len(self.scenarios), 7)
        self.assertEqual(
            {scenario["expected_response_mode"] for scenario in self.scenarios},
            {
                "continue",
                "replan",
                "follow_update",
                "clarify",
                "stop",
                "report_infeasible",
            },
        )

    def test_invalid_change_family_is_rejected(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["change_family"] = "ROBOT_ERROR"
        self.assertIn("change_family must be one of", " ".join(validate_scenario(scenario)))

    def test_invalid_change_type_is_rejected(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["change_type"] = "magic_world_change"
        self.assertIn("change_type must be one of", " ".join(validate_scenario(scenario)))

    def test_randomization_provenance_is_strict(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["randomization"] = {
            "sampler": "libero-max-v1.0",
            "draw_id": 2,
            "seed": 123,
        }
        self.assertEqual(validate_scenario(scenario), [])
        scenario["randomization"]["hidden_runtime_rng"] = True
        self.assertIn(
            "randomization has unknown fields",
            " ".join(validate_scenario(scenario)),
        )

    def test_invalid_progress_fraction_is_rejected(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["trigger"] = {"type": "progress_fraction", "value": 1.0}
        self.assertIn("between 0 and 1", " ".join(validate_scenario(scenario)))

    def test_clutter_setup_is_valid(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["change_family"] = "CLUTTER"
        scenario["setup"] = [
            {"operation": "remove_object", "object": "distractor_1"}
        ]
        scenario["change"] = {
            "operation": "insert_distractors",
            "placements": [
                {"object": "distractor_1", "position_m": [0.0, 0.0, 0.0]}
            ],
        }
        self.assertEqual(validate_scenario(scenario), [])

    def test_proximity_trigger_requires_positive_distance(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["trigger"] = {
            "type": "on_proximity",
            "value": "target_1",
            "distance_m": 0.18,
        }
        self.assertEqual(validate_scenario(scenario), [])
        scenario["trigger"]["distance_m"] = 0
        self.assertIn("must be positive", " ".join(validate_scenario(scenario)))

    def test_before_place_requires_target_and_distance(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["trigger"] = {
            "type": "before_place",
            "value": "basket_1",
            "distance_m": 0.2,
        }
        self.assertIn("target_entity", " ".join(validate_scenario(scenario)))
        scenario["trigger"]["target_entity"] = "soup_1"
        self.assertEqual(validate_scenario(scenario), [])

    def test_duplicate_scenario_seed_is_rejected(self):
        scenarios = self.scenarios + [copy.deepcopy(self.scenarios[0])]
        self.assertIn(
            "duplicate (scenario_id, seed)",
            " ".join(validate_scenario_collection(scenarios)),
        )


if __name__ == "__main__":
    unittest.main()
