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

    def test_invalid_progress_fraction_is_rejected(self):
        scenario = copy.deepcopy(self.scenarios[0])
        scenario["trigger"] = {"type": "progress_fraction", "value": 1.0}
        self.assertIn("between 0 and 1", " ".join(validate_scenario(scenario)))

    def test_duplicate_scenario_seed_is_rejected(self):
        scenarios = self.scenarios + [copy.deepcopy(self.scenarios[0])]
        self.assertIn(
            "duplicate (scenario_id, seed)",
            " ".join(validate_scenario_collection(scenarios)),
        )


if __name__ == "__main__":
    unittest.main()
