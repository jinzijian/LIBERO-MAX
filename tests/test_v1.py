import copy
import unittest

from libero_max.manifest import validate_manifest
from libero_max.v1 import (
    build_v1_manifest,
    eligible_change_types,
    manifest_design_summary,
)


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


class V1ManifestTest(unittest.TestCase):
    def test_all_six_physical_change_types_are_eligible(self):
        self.assertEqual(
            eligible_change_types(TASK),
            [
                "illumination_switch",
                "camera_shift",
                "target_relocation",
                "receptacle_relocation",
                "distractor_burst",
                "obstacle_insertion",
            ],
        )

    def test_core_uses_balanced_draws_without_expanding_pair_count(self):
        manifest = build_v1_manifest({"tasks": [TASK]}, profile="core")
        self.assertEqual(validate_manifest(manifest), [])
        self.assertEqual(len(manifest["cases"]), 4 * 3 * 3 + 2 * 3 * 2)
        for change_type in eligible_change_types(TASK):
            draws = [
                case["scenario"]["randomization"]["draw_id"]
                for case in manifest["cases"]
                if case["scenario"]["change_type"] == change_type
            ]
            expected = (
                {0: 3, 1: 3}
                if change_type.endswith("relocation")
                else {0: 3, 1: 3, 2: 3}
            )
            self.assertEqual(
                {draw: draws.count(draw) for draw in set(draws)}, expected
            )

    def test_full_crosses_policy_and_intervention_randomness(self):
        manifest = build_v1_manifest({"tasks": [TASK]}, profile="full")
        summary = manifest_design_summary(manifest)
        expected_pairs = 4 * 3 * 3 * 3 + 2 * 3 * 2 * 3
        self.assertEqual(summary["matched_pairs"], expected_pairs)
        self.assertEqual(summary["episodes"], 2 * expected_pairs)

    def test_generation_is_deterministic_and_changes_across_draws(self):
        left = build_v1_manifest({"tasks": [TASK]}, profile="full")
        right = build_v1_manifest({"tasks": [copy.deepcopy(TASK)]}, profile="full")
        self.assertEqual(left, right)
        relocation = [
            case["scenario"]
            for case in left["cases"]
            if case["scenario"]["change_type"] == "target_relocation"
            and case["init_state_index"] == 0
            and case["policy_seed"] == 195
        ]
        self.assertEqual(len({row["seed"] for row in relocation}), 2)
        self.assertEqual(
            [row["change"]["delta_position_m"] for row in relocation],
            [[0.06, 0.0, 0.0], [0.12, 0.0, 0.0]],
        )
        self.assertEqual(
            {row["change"]["support_entity"] for row in relocation},
            {"floor"},
        )

    def test_relocation_requires_a_calibrated_direction(self):
        task = copy.deepcopy(TASK)
        del task["relocation_directions"]
        with self.assertRaisesRegex(ValueError, "calibrated"):
            build_v1_manifest({"tasks": [task]}, profile="core")


if __name__ == "__main__":
    unittest.main()
