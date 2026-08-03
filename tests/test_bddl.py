import unittest

from libero_max.bddl import is_planar_workspace_placement, parse_bddl_metadata


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


if __name__ == "__main__":
    unittest.main()
