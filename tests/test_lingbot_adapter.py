import unittest
from pathlib import Path

import numpy as np

from libero_max.lingbot_adapter import (
    DUMMY_ACTION,
    flatten_lingbot_actions,
    lingbot_policy_input_digests,
)


class LingBotAdapterTest(unittest.TestCase):
    def test_policy_input_digest_covers_images_and_sim_state(self):
        images = {
            "agentview": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist": np.ones((4, 4, 3), dtype=np.uint8),
        }
        state = np.asarray([1.0, 2.0], dtype=np.float64)
        first = lingbot_policy_input_digests(images, state)
        repeated = lingbot_policy_input_digests(images, state)
        changed = lingbot_policy_input_digests(
            {**images, "wrist": np.zeros((4, 4, 3), dtype=np.uint8)}, state
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertIn("sim_state_sha256", first)

    def test_flattens_native_frame_action_layout(self):
        native = np.arange(7 * 4 * 4, dtype=np.float32).reshape(7, 4, 4)
        flat = flatten_lingbot_actions(native, query_index=1)
        self.assertEqual(flat.shape, (16, 7))
        np.testing.assert_array_equal(flat[0], native[:, 0, 0])
        np.testing.assert_array_equal(flat[5], native[:, 1, 1])

    def test_replaces_conditioned_bootstrap_frame_with_noops(self):
        native = np.ones((7, 4, 4), dtype=np.float32)
        flat = flatten_lingbot_actions(native, query_index=0)
        np.testing.assert_array_equal(flat[:4], np.tile(DUMMY_ACTION, (4, 1)))
        np.testing.assert_array_equal(flat[4], np.ones(7, dtype=np.float32))

    def test_rejects_wrong_native_shape(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            flatten_lingbot_actions(np.zeros((7, 16), dtype=np.float32), query_index=0)

    def test_runner_gives_flash_stub_an_import_spec(self):
        source = (
            Path(__file__).parents[1] / "scripts" / "run_lingbot_persistent_shard.py"
        ).read_text(encoding="utf-8")
        self.assertIn("flash_stub.__spec__", source)
        self.assertIn("ModuleSpec", source)

    def test_runner_restores_exact_state_only_after_the_event(self):
        source = (
            Path(__file__).parents[1] / "scripts" / "run_lingbot_persistent_shard.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_capture_lingbot_runtime_state", source)
        self.assertIn("_restore_lingbot_runtime_state", source)
        self.assertIn('wrapped.runtime.applied', source)
        self.assertIn('restored_before_response_query', source)


if __name__ == "__main__":
    unittest.main()
