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


class FakePredicateEnv(FakeBaseEnv):
    def __init__(self):
        super().__init__()
        self.predicates = []

    def _eval_predicate(self, state):
        self.predicates.append(state)
        return True


class LiberoBackendTest(unittest.TestCase):
    def test_sim_is_resolved_lazily_after_hard_reset(self):
        wrapper = FakeWrapper()
        backend = LiberoMujocoBackend(wrapper)
        original = backend.sim
        replacement = FakeSim()
        wrapper.env.sim = replacement
        self.assertIsNot(original, replacement)
        self.assertIs(backend.sim, replacement)

    def test_goal_satisfied_normalizes_catalog_predicate_names(self):
        env = FakePredicateEnv()
        backend = LiberoMujocoBackend(env)
        self.assertTrue(
            backend.goal_satisfied(
                [{"predicate": "In", "arguments": ["item", "container"]}]
            )
        )
        self.assertEqual(env.predicates, [["in", "item", "container"]])

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

    def test_relative_placement_preserves_removed_object_height(self):
        backend = object.__new__(LiberoMujocoBackend)
        backend._removed_positions = {"distractor": np.array([0.0, 0.0, 0.12])}
        positions = {
            "distractor": np.array([0.0, 0.0, -5.0]),
            "target": np.array([0.4, -0.2, 0.08]),
        }
        backend._entity = lambda name: (name, False)
        backend._entity_position = lambda entity, fixture: positions[entity].copy()
        result = backend._requested_position(
            {
                "relative_to": "target",
                "offset_m": [0.1, 0.05, 0.0],
                "preserve_initial_z": True,
            },
            "distractor",
        )
        np.testing.assert_allclose(result, [0.5, -0.15, 0.12])

    def test_relative_placement_requires_shared_removal_for_preserved_height(self):
        backend = object.__new__(LiberoMujocoBackend)
        backend._removed_positions = {}
        backend._entity = lambda name: (name, False)
        backend._entity_position = lambda entity, fixture: np.zeros(3)
        with self.assertRaisesRegex(LiberoBackendError, "removed in setup"):
            backend._requested_position(
                {"relative_to": "target", "preserve_initial_z": True},
                "distractor",
            )

    def test_sensor_corruption_is_seeded_and_preserves_non_images(self):
        backend = object.__new__(LiberoMujocoBackend)
        backend._observation_corruption = {}
        backend._observation_counter = 0
        backend._set_sensor_corruption(
            {
                "seed": 17,
                "noise_std": 12.0,
                "occlusion_fraction": 0.1,
            }
        )
        observation = {
            "agentview_image": np.full((20, 20, 3), 127, dtype=np.uint8),
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        }
        first = backend.transform_observation(observation)
        backend._observation_counter = 0
        repeated = backend.transform_observation(observation)
        np.testing.assert_array_equal(
            first["agentview_image"], repeated["agentview_image"]
        )
        self.assertFalse(
            np.array_equal(first["agentview_image"], observation["agentview_image"])
        )
        np.testing.assert_array_equal(
            first["robot0_eef_pos"], observation["robot0_eef_pos"]
        )

    def test_visual_theme_changes_only_rgb_channels(self):
        backend = object.__new__(LiberoMujocoBackend)
        model = type("Model", (), {})()
        model.geom_rgba = np.array([[0.2, 0.4, 0.8, 0.5]], dtype=np.float64)
        backend.env = type(
            "Env", (), {"sim": type("Sim", (), {"model": model})()}
        )()
        result = backend._set_visual_theme(
            {
                "rgb_permutation": [2, 0, 1],
                "rgb_multiplier": [1.0, 1.0, 1.0],
            }
        )
        np.testing.assert_allclose(model.geom_rgba, [[0.8, 0.2, 0.4, 0.5]])
        self.assertEqual(result["changed_arrays"], ["geom_rgba"])


if __name__ == "__main__":
    unittest.main()
