import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/precompute_cosmos_manifest_embeddings.py"
SPEC = importlib.util.spec_from_file_location("manifest_embeddings", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ManifestEmbeddingTest(unittest.TestCase):
    def test_collects_pro_and_intent_prompts(self):
        manifest = {
            "cases": [
                {
                    "substrate_variant": {"language": "perturbed task"},
                    "scenario": {"change": {"instruction": "new goal"}},
                },
                {
                    "substrate_variant": {"language": "perturbed task"},
                    "scenario": {"change": {"operation": "shift_camera"}},
                },
            ]
        }
        self.assertEqual(
            MODULE.manifest_prompts(manifest), {"perturbed task", "new goal"}
        )


if __name__ == "__main__":
    unittest.main()
