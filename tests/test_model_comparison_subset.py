import unittest

from scripts.build_model_comparison_subset import select_cases


def _case(category, change, draw, index):
    return {
        "case_id": "%s-%s-%d-%d" % (category, change, draw, index),
        "substrate_category": category,
        "scenario": {
            "change_type": change,
            "randomization": {"draw_id": draw},
        },
    }


class ModelComparisonSubsetTest(unittest.TestCase):
    def test_balances_cells_and_is_order_independent(self):
        cases = [
            _case(category, change, draw, index)
            for category in ("a", "b")
            for change in ("light", "move")
            for draw in (0, 1)
            for index in range(4)
        ]
        selected = select_cases(cases, 2)
        reversed_selected = select_cases(list(reversed(cases)), 2)
        self.assertEqual(
            [case["case_id"] for case in selected],
            [case["case_id"] for case in reversed_selected],
        )
        counts = {}
        for case in selected:
            cell = (
                case["substrate_category"],
                case["scenario"]["change_type"],
                case["scenario"]["randomization"]["draw_id"],
            )
            counts[cell] = counts.get(cell, 0) + 1
        self.assertEqual(len(selected), 16)
        self.assertEqual(set(counts.values()), {2})

    def test_rejects_underfilled_cell(self):
        with self.assertRaisesRegex(ValueError, "only 1"):
            select_cases([_case("a", "light", 0, 0)], 2)


if __name__ == "__main__":
    unittest.main()
