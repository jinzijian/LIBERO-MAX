import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from libero_max.bddl import (
    is_planar_workspace_placement,
    parse_bddl_metadata,
    resolve_libero_bddl_path,
)


SAMPLE = """
(define (problem sample)
  (:fixtures
    floor - floor
  )
  (:objects
    alphabet_soup_1 - alphabet_soup
    basket_1 - basket
    milk_1 butter_1 - food
  )
  (:obj_of_interest
    alphabet_soup_1
    basket_1
  )
  (:init
    (On alphabet_soup_1 floor_target_region)
    (On basket_1 floor_bin_region)
  )
  (:goal
    (And (In alphabet_soup_1 basket_1_contain_region))
  )
)
"""


class BddlParserTest(unittest.TestCase):
    def test_extracts_manipulated_target_and_distractors(self):
        metadata = parse_bddl_metadata(SAMPLE)
        self.assertEqual(metadata["manipulated_objects"], ["alphabet_soup_1"])
        self.assertEqual(
            metadata["goal_entities"], ["alphabet_soup_1", "basket_1"]
        )
        self.assertEqual(metadata["distractor_objects"], ["butter_1", "milk_1"])
        self.assertEqual(metadata["fixtures"], {"floor": "floor"})
        self.assertEqual(
            metadata["goal_relations"],
            [
                {
                    "predicate": "In",
                    "arguments": ["alphabet_soup_1", "basket_1_contain_region"],
                }
            ],
        )
        self.assertEqual(
            metadata["initial_placements"]["alphabet_soup_1"],
            {
                "predicate": "On",
                "region": "floor_target_region",
                "support_entity": "floor",
            },
        )

    def test_preserves_goal_order_for_multistage_targets(self):
        text = SAMPLE.replace(
            "(And (In alphabet_soup_1 basket_1_contain_region))",
            "(And (In milk_1 basket_1_contain_region) "
            "(In alphabet_soup_1 basket_1_contain_region))",
        )
        metadata = parse_bddl_metadata(text)
        self.assertEqual(
            metadata["manipulated_objects"], ["milk_1", "alphabet_soup_1"]
        )

    def test_relocation_requires_a_large_planar_fixture(self):
        metadata = parse_bddl_metadata(SAMPLE)
        self.assertTrue(
            is_planar_workspace_placement(metadata, "alphabet_soup_1")
        )
        metadata["objects"]["tray_1"] = "tray"
        metadata["objects"]["bowl_1"] = "bowl"
        metadata["initial_placements"]["bowl_1"] = {
            "predicate": "On",
            "region": "tray_1_center",
            "support_entity": "tray_1",
        }
        self.assertFalse(is_planar_workspace_placement(metadata, "bowl_1"))

    def test_custom_table_fixture_types_are_planar_workspaces(self):
        metadata = parse_bddl_metadata(SAMPLE)
        metadata["fixtures"] = {"living_room_table": "living_room_table"}
        metadata["initial_placements"]["alphabet_soup_1"][
            "support_entity"
        ] = "living_room_table"
        self.assertTrue(
            is_planar_workspace_placement(metadata, "alphabet_soup_1")
        )

    def test_resolves_libero_plus_virtual_bddl(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "libero_spatial"
            folder.mkdir()
            base = folder / "pick_and_place.bddl"
            base.write_text(SAMPLE, encoding="utf-8")
            path, details = resolve_libero_bddl_path(
                root,
                "libero_spatial",
                "pick_and_place_view_10_-5_120_3_4_initstate_17_noise_2.bddl",
            )
            self.assertEqual(path, base)
            self.assertEqual(details["kind"], "libero_plus_virtual")
            self.assertEqual(details["scale_factor"], 1.2)
            self.assertEqual(details["robot_init_state"], 17)
            self.assertEqual(details["sensor_noise_level"], 2)

    def test_materialized_bddl_takes_precedence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "libero_goal"
            folder.mkdir()
            file_path = folder / "task_view_0_0_100_0_0_initstate_1.bddl"
            file_path.write_text(SAMPLE, encoding="utf-8")
            path, details = resolve_libero_bddl_path(
                root, "libero_goal", file_path.name
            )
            self.assertEqual(path, file_path)
            self.assertEqual(details, {"kind": "materialized"})


if __name__ == "__main__":
    unittest.main()
