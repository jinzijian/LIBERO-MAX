import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from libero_max.pro_runtime import EXPECTED_PRO_COMPONENTS, _adapt_flattened_state


class _State:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def flatten(self):
        return self.values.copy()


class ProRuntimeStateAdapterTest(unittest.TestCase):
    def test_every_released_pro_category_has_an_explicit_runtime_contract(self):
        self.assertEqual(len(EXPECTED_PRO_COMPONENTS), 10)
        self.assertEqual(
            EXPECTED_PRO_COMPONENTS["view_occlusion"],
            frozenset({"object_placement", "joint_name_v1"}),
        )

    def test_maps_shared_joints_and_preserves_variant_only_joint(self):
        # Source: joint_a free, joint_b hinge. Variant inserts a free occluder
        # between them, exactly the topology change used by view_occlusion.
        source_layout = {
            "nq": 8,
            "nv": 7,
            "na": 0,
            "joints": {
                "joint_a": {"type": 0, "qpos": (0, 7), "qvel": (0, 6)},
                "joint_b": {"type": 3, "qpos": (7, 1), "qvel": (6, 1)},
            },
        }
        target_model = SimpleNamespace(
            nq=15,
            nv=13,
            na=0,
            joint_names=["joint_a", "occluder", "joint_b"],
            jnt_type=np.array([0, 0, 3]),
            jnt_qposadr=np.array([0, 7, 14]),
            jnt_dofadr=np.array([0, 6, 12]),
        )
        target_reset = np.arange(29, dtype=np.float64) + 1000
        env = SimpleNamespace(
            sim=SimpleNamespace(
                model=target_model,
                get_state=lambda: _State(target_reset),
            )
        )
        source = np.arange(16, dtype=np.float64)
        with patch(
            "libero_max.pro_runtime._reference_layout",
            return_value=source_layout,
        ):
            adapted = _adapt_flattened_state(env, source, Path("reference.bddl"))

        # time and joint_a are copied from the source.
        np.testing.assert_array_equal(adapted[:8], source[:8])
        # Variant-only occluder qpos remains at the deterministic reset value.
        np.testing.assert_array_equal(adapted[8:15], target_reset[8:15])
        # joint_b is copied by name despite moving after the inserted joint.
        self.assertEqual(adapted[15], source[8])
        np.testing.assert_array_equal(adapted[16:22], source[9:15])
        np.testing.assert_array_equal(adapted[22:28], target_reset[22:28])
        self.assertEqual(adapted[28], source[15])

    def test_identity_state_does_not_need_reference_model(self):
        values = np.arange(5, dtype=np.float64)
        env = SimpleNamespace(
            sim=SimpleNamespace(get_state=lambda: _State(values))
        )
        with patch("libero_max.pro_runtime._reference_layout") as reference:
            adapted = _adapt_flattened_state(env, values, Path("unused.bddl"))
        reference.assert_not_called()
        np.testing.assert_array_equal(adapted, values)


if __name__ == "__main__":
    unittest.main()
