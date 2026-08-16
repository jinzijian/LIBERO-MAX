import copy
import unittest

from libero_max.pruning import prune_infeasible_configurations
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


class FeasibilityPruningTest(unittest.TestCase):
    def test_one_failed_state_removes_configuration_from_core_and_full(self):
        core = build_v1_manifest({"tasks": [copy.deepcopy(TASK)]}, "core")
        full = build_v1_manifest({"tasks": [copy.deepcopy(TASK)]}, "full")
        failed = next(
            case
            for case in core["cases"]
            if case["scenario"]["change_type"] == "obstacle_insertion"
            and case["scenario"]["randomization"]["draw_id"] == 1
        )
        rows = [
            {
                "case_id": case["case_id"],
                "scenario_id": case["scenario"]["scenario_id"],
                "passed": case["case_id"] != failed["case_id"],
            }
            for case in core["cases"]
        ]
        preflight = {
            "benchmark_id": core["benchmark_id"],
            "planned": len(core["cases"]),
            "passed": len(core["cases"]) - 1,
            "complete": False,
            "failures": {failed["case_id"]: "physics"},
            "cases": rows,
        }
        pruned_core, pruned_full, report = prune_infeasible_configurations(
            core, full, preflight
        )
        self.assertEqual(len(core["cases"]) - len(pruned_core["cases"]), 3)
        self.assertEqual(len(full["cases"]) - len(pruned_full["cases"]), 9)
        self.assertEqual(report["excluded_configurations_by_change_type"], {
            "obstacle_insertion": 1
        })
        self.assertFalse(
            any(
                case["scenario"]["change_type"] == "obstacle_insertion"
                and case["scenario"]["randomization"]["draw_id"] == 1
                for case in pruned_core["cases"]
            )
        )


if __name__ == "__main__":
    unittest.main()
