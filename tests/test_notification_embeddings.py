import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NotificationEmbeddingTest(unittest.TestCase):
    def test_prompt_matches_runtime_notification_format(self):
        path = ROOT / "scripts/precompute_cosmos_notification_embeddings.py"
        spec = importlib.util.spec_from_file_location("notification_embeddings", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.notified_instruction("pick up the mug.", "Reassess and continue."),
            "pick up the mug Reassess and continue.",
        )


if __name__ == "__main__":
    unittest.main()
