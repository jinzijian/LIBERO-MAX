import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/render_rollout_replay.py"


class RolloutReplayMediaTest(unittest.TestCase):
    def test_renderer_requires_event_and_action_trace_audit(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("executed_actions", source)
        self.assertIn("event step mismatch", source)
        self.assertIn("replayed_action_sequence_sha256", source)
        self.assertIn("hashlib.sha256", source)
        self.assertIn("validate_render", source)


if __name__ == "__main__":
    unittest.main()
