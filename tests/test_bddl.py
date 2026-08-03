import unittest

from libero_max.bddl import parse_bddl_metadata


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


if __name__ == "__main__":
    unittest.main()
