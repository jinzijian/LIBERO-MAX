import json
import unittest
from pathlib import Path

from libero_max.manifest import validate_manifest
from scripts.build_manifest_complement import build_complement


ROOT = Path(__file__).parents[1]


class ManifestComplementTest(unittest.TestCase):
    def test_reuses_frozen_800_and_selects_exact_remaining_1600(self):
        full = json.loads(
            (ROOT / "benchmark/max8000/libero_max_pro_hard_2400.json").read_text()
        )
        excluded = json.loads(
            (
                ROOT
                / "benchmark/max8000/libero_max_pro_model_comparison_800.json"
            ).read_text()
        )
        result = build_complement(full, excluded)
        self.assertEqual(len(result["cases"]), 1600)
        self.assertIn(
            "outcome_independent=true",
            result["protocol"]["selection_contract"],
        )
        self.assertEqual(validate_manifest(result), [])
        self.assertFalse(
            {case["case_id"] for case in result["cases"]}
            & {case["case_id"] for case in excluded["cases"]}
        )

    def test_rejects_modified_excluded_case(self):
        full = {
            "benchmark_id": "full",
            "benchmark_version": "1",
            "protocol": {},
            "cases": [{"case_id": "a", "value": 1}],
        }
        excluded = {
            "benchmark_id": "excluded",
            "benchmark_version": "1",
            "protocol": {},
            "cases": [{"case_id": "a", "value": 2}],
        }
        with self.assertRaisesRegex(ValueError, "differ"):
            build_complement(full, excluded)


if __name__ == "__main__":
    unittest.main()
