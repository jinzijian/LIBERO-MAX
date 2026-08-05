import copy
import unittest
from pathlib import Path

from libero_max.manifest import load_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(
            ROOT / "examples/manifests/cosmos_physical_pilot_v0.1.json"
        )

    def test_physical_pilot_manifest_is_valid(self):
        self.assertEqual(validate_manifest(self.manifest), [])
        self.assertEqual(len(self.manifest["cases"]), 5)

    def test_duplicate_case_id_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"][1]["case_id"] = broken["cases"][0]["case_id"]
        self.assertTrue(any("duplicate case_id" in error for error in validate_manifest(broken)))

    def test_physical_track_rejects_intent_scenario(self):
        broken = copy.deepcopy(self.manifest)
        scenario = broken["cases"][0]["scenario"]
        scenario["change_family"] = "INTENT"
        scenario["change"] = {"operation": "cancel_instruction"}
        scenario["expected_response_mode"] = "stop"
        errors = validate_manifest(broken)
        self.assertTrue(any("physical_completion only supports" in error for error in errors))

    def test_trigger_must_align_to_query_interval(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"][0]["scenario"]["trigger"] = {
            "type": "fixed_step",
            "value": 17,
        }
        self.assertTrue(any("aligned" in error for error in validate_manifest(broken)))

    def test_plus_metadata_is_strict_when_present(self):
        enriched = copy.deepcopy(self.manifest)
        enriched["protocol"].update(
            {
                "substrate": "LIBERO-Plus",
                "profile": "core",
                "source_benchmark_commit": "4976dc3",
                "selection_contract": "stratified",
            }
        )
        enriched["cases"][0].update(
            {
                "task_name": "task",
                "substrate_category": "Sensor Noise",
                "substrate_difficulty": 5,
                "dynamic_phase": "pre_grasp_proximity",
            }
        )
        self.assertEqual(validate_manifest(enriched), [])
        enriched["cases"][0]["substrate_difficulty"] = 6
        self.assertTrue(
            any("substrate_difficulty" in error for error in validate_manifest(enriched))
        )


if __name__ == "__main__":
    unittest.main()
