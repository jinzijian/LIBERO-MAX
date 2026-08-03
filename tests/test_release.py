import copy
import unittest

from libero_max.release import audit_v1_release
from libero_max.v1 import build_v1_manifest


TASK = {
    "task_suite_name": "libero_object",
    "task_index": 0,
    "trigger_entity": "target_1",
    "primary_target": "target_1",
    "primary_receptacle": "basket_1",
    "supports_target_relocation": True,
    "supports_receptacle_relocation": True,
    "available_distractor_count": 5,
    "distractor_objects": ["d%d" % index for index in range(5)],
    "initial_placements": {
        "target_1": {"support_entity": "floor"},
        "basket_1": {"support_entity": "floor"},
        **{
            "d%d" % index: {"support_entity": "floor"}
            for index in range(5)
        },
    },
    "relocation_directions": {
        "target_relocation": [1.0, 0.0],
        "receptacle_relocation": [0.0, -1.0],
    },
}


class ReleaseAuditTest(unittest.TestCase):
    def _artifacts(self):
        core = build_v1_manifest(
            {"tasks": [copy.deepcopy(TASK)]}, profile="core"
        )
        full = build_v1_manifest(
            {"tasks": [copy.deepcopy(TASK)]}, profile="full"
        )
        preflight = {
            "benchmark_id": core["benchmark_id"],
            "complete": True,
            "planned": len(core["cases"]),
            "cases": [
                {
                    "scenario_id": case["scenario"]["scenario_id"],
                    "passed": True,
                }
                for case in core["cases"]
            ],
        }
        catalog = {"relocation_calibration": {"complete": True}}
        return catalog, core, full, preflight

    def test_complete_matched_artifacts_pass(self):
        self.assertEqual(audit_v1_release(*self._artifacts()), [])

    def test_missing_preflight_scenario_fails(self):
        catalog, core, full, preflight = self._artifacts()
        preflight["cases"].pop()
        self.assertIn(
            "preflight scenario coverage does not exactly match Core",
            audit_v1_release(catalog, core, full, preflight),
        )


if __name__ == "__main__":
    unittest.main()
