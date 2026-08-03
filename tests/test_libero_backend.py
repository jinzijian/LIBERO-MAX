import unittest

import numpy as np

from libero_max.libero_backend import LiberoBackendError, LiberoMujocoBackend


class FakeSim:
    pass


class FakeBaseEnv:
    def __init__(self):
        self.sim = FakeSim()


class FakeWrapper:
    def __init__(self):
        self.env = FakeBaseEnv()


class LiberoBackendTest(unittest.TestCase):
    def test_sim_is_resolved_lazily_after_hard_reset(self):
        wrapper = FakeWrapper()
        backend = LiberoMujocoBackend(wrapper)
        original = backend.sim
        replacement = FakeSim()
        wrapper.env.sim = replacement
        self.assertIsNot(original, replacement)
        self.assertIs(backend.sim, replacement)

    def test_distractor_batch_moves_each_unique_object(self):
        backend = object.__new__(LiberoMujocoBackend)
        moved = []
        backend._requested_position = lambda placement, name: np.asarray(
            placement["position_m"]
        )
        backend._set_entity_position = lambda name, position: moved.append(
            (name, position.tolist())
        ) or {"entity": name, "position_after_m": position.tolist()}
        result = backend._insert_distractors(
            {
                "placements": [
                    {"object": "a", "position_m": [0, 0, 0]},
                    {"object": "b", "position_m": [1, 0, 0]},
                ]
            }
        )
        self.assertEqual(moved, [("a", [0, 0, 0]), ("b", [1, 0, 0])])
        self.assertEqual(len(result["placements"]), 2)

    def test_distractor_batch_rejects_duplicate_object(self):
        backend = object.__new__(LiberoMujocoBackend)
        backend._requested_position = lambda placement, name: np.zeros(3)
        backend._set_entity_position = lambda name, position: {"entity": name}
        with self.assertRaisesRegex(LiberoBackendError, "duplicate distractor"):
            backend._insert_distractors(
                {
                    "placements": [
                        {"object": "a", "position_m": [0, 0, 0]},
                        {"object": "a", "position_m": [1, 0, 0]},
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
