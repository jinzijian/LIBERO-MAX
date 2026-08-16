import unittest

from libero_max.calibration import (
    rank_lateral_directions,
    select_common_direction,
)


class CalibrationTest(unittest.TestCase):
    def test_lateral_direction_is_preferred_over_goal_axis(self):
        ranked = rank_lateral_directions([(0.0, 1.0), (0.1, 1.0)])
        self.assertIn(ranked[0], {(1.0, 0.0), (-1.0, 0.0)})

    def test_ranking_is_deterministic_without_a_task_axis(self):
        self.assertEqual(rank_lateral_directions([])[0], (1.0, 0.0))

    def test_first_ranked_passing_direction_is_selected(self):
        selected = select_common_direction(
            [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)],
            {(0.0, 1.0), (-1.0, 0.0)},
        )
        self.assertEqual(selected, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
