import unittest
from pathlib import Path

import numpy as np

from libero_max.lingbot_adapter import DUMMY_ACTION, flatten_lingbot_actions


class LingBotAdapterTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
