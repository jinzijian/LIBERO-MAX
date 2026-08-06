import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_preflight_delta.py"
SPEC = importlib.util.spec_from_file_location("merge_preflight_delta", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(case_id, passed):
    return {
        "case_id": case_id,
        "scenario_id": case_id.removesuffix("-p195"),
        "change_type": "target_relocation",
        "passed": passed,
        "validation_errors": [] if passed else ["invalid placement"],
    }


class MergePreflightDeltaTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "benchmark_id": "benchmark",
            "cases": [
                {"case_id": "a-p195"},
                {"case_id": "b-p195"},
            ],
        }
        self.base = {
            "benchmark_id": "benchmark",
            "shards": 8,
            "cases": [_row("a-p195", True), _row("b-p195", False)],
        }

    def test_replaces_failed_rows_and_restores_complete_coverage(self):
        merged = MODULE.merge_repaired_delta(
            self.manifest,
            self.base,
            {
                "benchmark_id": "benchmark",
                "shards": 2,
                "cases": [_row("b-p195", True)],
            },
        )
        self.assertTrue(merged["complete"])
        self.assertEqual(merged["planned"], 2)
        self.assertEqual(merged["passed"], 2)
        self.assertEqual(merged["failures"], {})

    def test_requires_exact_failed_case_replacement(self):
        with self.assertRaisesRegex(ValueError, "every failed base case"):
            MODULE.merge_repaired_delta(
                self.manifest,
                self.base,
                {"benchmark_id": "benchmark", "cases": []},
            )


if __name__ == "__main__":
    unittest.main()
