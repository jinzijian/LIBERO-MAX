import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AblationManifestTest(unittest.TestCase):
    def test_frozen_subset_is_balanced_and_deterministic(self):
        path = ROOT / "scripts/build_ablation_manifest.py"
        spec = importlib.util.spec_from_file_location("build_ablation_manifest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        core = json.loads((ROOT / "benchmark/v1/core.json").read_text())
        selected = module.select_cases(core, 30)
        self.assertEqual(selected, module.select_cases(core, 30))
        self.assertEqual(len(selected), 180)
        counts = {}
        for case in selected:
            change_type = case["scenario"]["change_type"]
            counts[change_type] = counts.get(change_type, 0) + 1
        self.assertEqual(set(counts.values()), {30})


if __name__ == "__main__":
    unittest.main()
