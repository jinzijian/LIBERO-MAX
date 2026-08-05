import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IntentManifestTest(unittest.TestCase):
    def test_frozen_intent_manifest_is_balanced_and_valid(self):
        from libero_max.manifest import load_manifest

        manifest = load_manifest(ROOT / "benchmark/v1/intent_core.json")
        self.assertEqual(len(manifest["cases"]), 96)
        counts = {}
        for case in manifest["cases"]:
            change_type = case["scenario"]["change_type"]
            counts[change_type] = counts.get(change_type, 0) + 1
        self.assertEqual(
            counts,
            {
                "instruction_target_update": 30,
                "instruction_receptacle_update": 18,
                "task_cancel": 48,
            },
        )

    def test_builder_is_deterministic(self):
        import importlib.util

        path = ROOT / "scripts/build_intent_v1_manifest.py"
        spec = importlib.util.spec_from_file_location("build_intent_v1_manifest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        catalog = json.loads((ROOT / "artifacts/libero_task_catalog_v1.json").read_text())
        self.assertEqual(module.build_manifest(catalog), module.build_manifest(catalog))


if __name__ == "__main__":
    unittest.main()
