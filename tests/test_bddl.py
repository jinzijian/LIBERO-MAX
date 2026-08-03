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


if __name__ == "__main__":
    unittest.main()
