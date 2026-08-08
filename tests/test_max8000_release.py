import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
RELEASE = ROOT / "benchmark" / "max8000"


class Max8000ReleaseTest(unittest.TestCase):
    def test_release_is_frozen_balanced_and_source_locked(self):
        summary = json.loads((RELEASE / "release_summary.json").read_text())
        combined = json.loads((RELEASE / "libero_max_8000.json").read_text())
        pro = json.loads(
            (RELEASE / "libero_max_pro_hard_2400.json").read_text()
        )
        comparison = json.loads(
            (RELEASE / "libero_max_pro_model_comparison_800.json").read_text()
        )
        source_lock = json.loads((RELEASE / "pro_source_lock.json").read_text())

        self.assertEqual(summary["status"], "released")
        self.assertEqual(summary["benchmark_version"], "3.0.0")
        self.assertEqual(summary["matched_pairs"], 8000)
        self.assertEqual(summary["model_comparison_pairs"], 800)
        self.assertEqual(combined["benchmark_version"], "3.0.0")
        self.assertEqual(pro["benchmark_version"], "3.0.0")
        self.assertEqual(comparison["benchmark_version"], "3.0.0")
        self.assertEqual(len(combined["cases"]), 8000)
        self.assertEqual(len(pro["cases"]), 2400)
        self.assertEqual(len(comparison["cases"]), 800)
        pro_ids = {case["case_id"] for case in pro["cases"]}
        self.assertTrue(
            {case["case_id"] for case in comparison["cases"]}.issubset(pro_ids)
        )
        cells = Counter(
            (
                case["substrate_category"],
                case["scenario"]["change_type"],
                case["scenario"]["randomization"]["draw_id"],
            )
            for case in comparison["cases"]
        )
        self.assertEqual(len(cells), 10 * 8 * 2)
        self.assertEqual(set(cells.values()), {5})
        self.assertEqual(source_lock["status"], "release_gate_satisfied")
        self.assertEqual(
            source_lock["pro_runtime_revision_tested"],
            "2b910b5b5f53016bef9907632f6f840f1ce2229c",
        )

    def test_release_checksums_match(self):
        entries = []
        for line in (RELEASE / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            entries.append((digest, name))
        self.assertEqual(len(entries), 8)
        for expected, name in entries:
            digest = hashlib.sha256((RELEASE / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)


if __name__ == "__main__":
    unittest.main()
