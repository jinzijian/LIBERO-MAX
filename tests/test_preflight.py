import unittest

from libero_max.preflight import (
    PreflightSelectionError,
    changed_entities,
    merge_preflight_reports,
    select_preflight_cases,
    settle_metrics,
)


def _case(scenario_id, policy_seed, change_type="camera_shift"):
    return {
        "case_id": "%s-p%d" % (scenario_id, policy_seed),
        "policy_seed": policy_seed,
        "scenario": {
            "scenario_id": scenario_id,
            "seed": sum(ord(char) for char in scenario_id),
            "change_type": change_type,
        },
    }


class PreflightSelectionTest(unittest.TestCase):
    def test_changed_entities_include_declared_supports(self):
        self.assertEqual(
            changed_entities(
                {
                    "operation": "insert_distractors",
                    "placements": [
                        {"object": "a", "support_entity": "table"},
                        {"object": "b", "support_entity": "table"},
                    ],
                }
            ),
            [("a", "table"), ("b", "table")],
        )

    def test_settle_metrics_measure_drop_and_total_motion(self):
        metrics = settle_metrics(
            {"a": [0.0, 0.0, 0.1]},
            {"a": [0.03, 0.04, 0.08]},
        )
        self.assertAlmostEqual(metrics["max_vertical_drop_m"], 0.02)
        self.assertAlmostEqual(
            metrics["max_settle_displacement_m"],
            (0.03**2 + 0.04**2 + 0.02**2) ** 0.5,
        )

    def test_policy_replicates_are_removed_by_default(self):
        cases = [
            _case("scenario-a", 1),
            _case("scenario-a", 2),
            _case("scenario-b", 1),
        ]
        selected, stats = select_preflight_cases(cases)
        self.assertEqual(
            [case["case_id"] for case in selected],
            ["scenario-a-p1", "scenario-b-p1"],
        )
        self.assertEqual(stats["policy_replicates_removed"], 1)

    def test_shards_are_disjoint_and_cover_filtered_scenarios(self):
        cases = [
            _case("scenario-%d" % index, 1, "target_relocation")
            for index in range(7)
        ] + [_case("camera-only", 1)]
        shards = []
        for shard_index in range(3):
            selected, _ = select_preflight_cases(
                cases,
                num_shards=3,
                shard_index=shard_index,
                change_types=("target_relocation",),
            )
            shards.append({case["case_id"] for case in selected})
        self.assertFalse(shards[0] & shards[1])
        self.assertFalse(shards[0] & shards[2])
        self.assertFalse(shards[1] & shards[2])
        self.assertEqual(len(set().union(*shards)), 7)

    def test_invalid_shard_is_rejected(self):
        with self.assertRaisesRegex(PreflightSelectionError, "shard_index"):
            select_preflight_cases([], num_shards=2, shard_index=2)

    def test_complete_shard_reports_merge_without_duplicates(self):
        reports = []
        for shard_index in range(2):
            case = {
                "case_id": "case-%d" % shard_index,
                "scenario_id": "scenario-%d" % shard_index,
                "change_type": "camera_shift",
                "passed": True,
            }
            reports.append(
                {
                    "benchmark_id": "benchmark",
                    "selection": {
                        "num_shards": 2,
                        "shard_index": shard_index,
                        "unique_scenarios": 2,
                    },
                    "failures": {},
                    "cases": [case],
                }
            )
        merged = merge_preflight_reports(reports)
        self.assertTrue(merged["complete"])
        self.assertEqual(merged["passed"], 2)
        self.assertEqual(
            merged["by_change_type"]["camera_shift"],
            {"planned": 2, "passed": 2},
        )

    def test_merge_rejects_missing_shard(self):
        with self.assertRaisesRegex(PreflightSelectionError, "every shard"):
            merge_preflight_reports(
                [
                    {
                        "benchmark_id": "benchmark",
                        "selection": {
                            "num_shards": 2,
                            "shard_index": 0,
                            "unique_scenarios": 1,
                        },
                        "failures": {},
                        "cases": [],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
