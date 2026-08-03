import unittest

from libero_max.preflight import (
    PreflightSelectionError,
    select_preflight_cases,
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


if __name__ == "__main__":
    unittest.main()
