import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from libero_max.manifest import validate_manifest


ROOT = Path(__file__).parents[1]
FULL = ROOT / "benchmark" / "max8000" / "libero_max_8000.json"
LITE = ROOT / "benchmark" / "lite800"


class Lite800ReleaseTest(unittest.TestCase):
    def test_lite_is_valid_balanced_and_a_subset(self):
        full = json.loads(FULL.read_text())
        lite = json.loads((LITE / "libero_max_lite_800.json").read_text())
        summary = json.loads((LITE / "selection_summary.json").read_text())

        self.assertEqual(validate_manifest(lite), [])
        self.assertEqual(lite["benchmark_id"], "libero-max-lite-800")
        self.assertEqual(lite["protocol"]["profile"], "lite")
        self.assertEqual(len(lite["cases"]), 800)
        full_ids = {case["case_id"] for case in full["cases"]}
        lite_ids = {case["case_id"] for case in lite["cases"]}
        self.assertEqual(len(lite_ids), 800)
        self.assertTrue(lite_ids.issubset(full_ids))

        event_counts = Counter(
            case["scenario"]["change_type"] for case in lite["cases"]
        )
        self.assertEqual(len(event_counts), 8)
        self.assertEqual(set(event_counts.values()), {100})
        event_source_counts = Counter(
            (
                case["scenario"]["change_type"],
                "pro" if case["case_id"].startswith("pro-") else "plus",
            )
            for case in lite["cases"]
        )
        for event in event_counts:
            self.assertEqual(event_source_counts[(event, "plus")], 70)
            self.assertEqual(event_source_counts[(event, "pro")], 30)

        self.assertEqual(summary["selection_seed"], 20260830)
        self.assertEqual(summary["matched_pairs"], 800)
        self.assertEqual(summary["source_counts"], {"plus": 560, "pro": 240})

    def test_case_index_matches_manifest(self):
        lite = json.loads((LITE / "libero_max_lite_800.json").read_text())
        with (LITE / "case_index.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["case_id"] for row in rows], [
            case["case_id"] for case in lite["cases"]
        ])

    def test_release_checksums_match(self):
        for line in (LITE / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            actual = hashlib.sha256((LITE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, name)

    def test_builder_reproduces_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "build_lite800_split.py"),
                    "--source",
                    str(FULL),
                    "--output-dir",
                    temp_dir,
                ],
                check=True,
                cwd=ROOT,
            )
            for name in (
                "libero_max_lite_800.json",
                "case_index.csv",
                "selection_summary.json",
                "SHA256SUMS",
            ):
                self.assertEqual((Path(temp_dir) / name).read_bytes(), (LITE / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
