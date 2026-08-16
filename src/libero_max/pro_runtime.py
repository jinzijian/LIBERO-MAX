"""Apply source-locked LIBERO-PRO substrate perturbations to live envs.

The public PRO robustness BDDLs store several perturbations in a
``:perturbation_config`` block.  Loading the BDDL alone is therefore not
enough: the config must be applied after the frozen LIBERO initial state is
restored, and observation-space perturbations must be applied on every step.
"""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np

from .substrate import variant_path


class ProRuntimeError(RuntimeError):
    """Raised when a PRO task cannot be reproduced faithfully."""


EXPECTED_PRO_COMPONENTS = {
    "semantic": frozenset(),
    "object": frozenset(),
    "position": frozenset(),
    "task": frozenset(),
    "visual_noise_glare": frozenset({"lighting", "observation"}),
    "camera_view_angle": frozenset({"camera"}),
    "object_texture": frozenset({"object_appearance"}),
    "view_occlusion": frozenset({"object_placement", "joint_name_v1"}),
    "object_shape": frozenset({"object_shape"}),
    "initial_pose_position_angle": frozenset({"initial_pose"}),
}


def _joint_widths(joint_type: int) -> Tuple[int, int]:
    # MuJoCo joint types: free=0, ball=1, slide=2, hinge=3.
    if joint_type == 0:
        return 7, 6
    if joint_type == 1:
        return 4, 3
    if joint_type in {2, 3}:
        return 1, 1
    raise ProRuntimeError("unknown MuJoCo joint type: %d" % joint_type)


def _model_layout(model: Any) -> Dict[str, Any]:
    joints = {}
    for index, raw_name in enumerate(model.joint_names):
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        joint_type = int(model.jnt_type[index])
        qpos_width, qvel_width = _joint_widths(joint_type)
        joints[name] = {
            "type": joint_type,
            "qpos": (int(model.jnt_qposadr[index]), qpos_width),
            "qvel": (int(model.jnt_dofadr[index]), qvel_width),
        }
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(model.na),
        "joints": joints,
    }


@lru_cache(maxsize=128)
def _reference_layout(bddl_path: str) -> Dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=64,
        camera_widths=64,
        camera_names=["agentview"],
        use_camera_obs=False,
    )
    try:
        return _model_layout(env.sim.model)
    finally:
        env.close()


def _adapt_flattened_state(
    env: Any,
    initial_state: Any,
    reference_bddl_path: Path,
) -> np.ndarray:
    """Map a frozen source state into a topology-extended PRO model by joint."""

    source = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    target = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
    if source.shape == target.shape:
        return source

    source_layout = _reference_layout(str(reference_bddl_path))
    target_layout = _model_layout(env.sim.model)
    expected_source = (
        1 + source_layout["nq"] + source_layout["nv"] + source_layout["na"]
    )
    expected_target = (
        1 + target_layout["nq"] + target_layout["nv"] + target_layout["na"]
    )
    if len(source) != expected_source or len(target) != expected_target:
        raise ProRuntimeError(
            "unexpected flattened state layout: source=%d/%d target=%d/%d"
            % (len(source), expected_source, len(target), expected_target)
        )

    missing = sorted(set(source_layout["joints"]) - set(target_layout["joints"]))
    if missing:
        raise ProRuntimeError(
            "PRO variant removed source joints: %s" % ", ".join(missing)
        )

    source_qpos = 1
    source_qvel = source_qpos + source_layout["nq"]
    source_act = source_qvel + source_layout["nv"]
    target_qpos = 1
    target_qvel = target_qpos + target_layout["nq"]
    target_act = target_qvel + target_layout["nv"]
    target[0] = source[0]
    for name, source_joint in source_layout["joints"].items():
        target_joint = target_layout["joints"][name]
        if source_joint["type"] != target_joint["type"]:
            raise ProRuntimeError("joint type changed for %s" % name)
        for field, source_offset, target_offset in (
            ("qpos", source_qpos, target_qpos),
            ("qvel", source_qvel, target_qvel),
        ):
            source_start, source_width = source_joint[field]
            target_start, target_width = target_joint[field]
            if source_width != target_width:
                raise ProRuntimeError("joint width changed for %s" % name)
            target[
                target_offset + target_start : target_offset + target_start + target_width
            ] = source[
                source_offset + source_start : source_offset + source_start + source_width
            ]
    if source_layout["na"]:
        if source_layout["na"] != target_layout["na"]:
            raise ProRuntimeError("actuator state width changed in PRO variant")
        target[target_act:] = source[source_act:]
    return target


class ProSubstrateEnv:
    """Environment wrapper that realizes the PRO half of a MAX-PRO case."""

    def __init__(self, env: Any, case: Dict[str, Any]):
        variant = case["substrate_variant"]
        if variant.get("benchmark") != "LIBERO-PRO":
            raise ProRuntimeError("ProSubstrateEnv requires a LIBERO-PRO case")
        from libero.libero import get_libero_path
        from libero.libero.envs.perturbation_config import (
            parse_bddl_perturbation_config,
        )
        from libero.libero.envs.robustness_perturbations import (
            PerturbationRuntimeOptions,
            infer_spec_from_config,
        )

        self._env = env
        self.case = case
        self.variant = variant
        self.bddl_path = variant_path(
            get_libero_path("bddl_files"), variant["bddl_file"]
        )
        self.reference_bddl_path = variant_path(
            get_libero_path("bddl_files"), variant["init_reference_bddl_file"]
        )
        config = parse_bddl_perturbation_config(self.bddl_path)
        self.spec = infer_spec_from_config(
            variant["category"],
            case["task_suite_name"],
            case.get("task_name", self.bddl_path.stem),
            config,
            bddl_path=self.bddl_path,
            init_suite=case["task_suite_name"],
        )
        self.options = PerturbationRuntimeOptions()
        self._vec_env = SimpleNamespace(
            envs=[SimpleNamespace(_env=self._env)]
        )
        self.substrate_info: Dict[str, Any] = {}
        self._static_info: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.substrate_info = {}
        self._static_info = {}
        return self._env.reset(*args, **kwargs)

    def _raw_observation(self) -> Dict[str, Any]:
        base_env = getattr(self._env, "env", self._env)
        base_env.sim.forward()
        if hasattr(base_env, "_post_process"):
            base_env._post_process()
        if hasattr(base_env, "_update_observables"):
            base_env._update_observables(force=True)
        return base_env._get_observations()

    def set_init_state(self, initial_state: Any) -> Dict[str, Any]:
        from libero.libero.envs.robustness_perturbations import (
            apply_camera_pose,
            apply_sim_lighting,
            apply_static_bddl_config_perturbations,
        )

        adapted = _adapt_flattened_state(
            self._env, initial_state, self.reference_bddl_path
        )
        self._env.set_init_state(adapted)
        self._static_info = apply_static_bddl_config_perturbations(
            self._vec_env, self.spec, self.options
        )
        lighting = apply_sim_lighting(self._vec_env, self.spec, self.options)
        camera = apply_camera_pose(self._vec_env, self.spec, self.options)
        state_adapter = (
            "joint_name_v1"
            if len(np.asarray(initial_state).reshape(-1)) != len(adapted)
            else "identity"
        )
        components = {
            name for name, info in self._static_info.items() if info is not None
        }
        if lighting != (None, None):
            components.add("lighting")
        if camera != (None, None):
            components.add("camera")
        if self.spec.observation_perturbation is not None:
            components.add("observation")
        if state_adapter != "identity":
            components.add(state_adapter)
        expected = EXPECTED_PRO_COMPONENTS.get(self.variant["category"])
        if expected is None:
            raise ProRuntimeError(
                "unknown PRO category: %s" % self.variant["category"]
            )
        missing = sorted(expected - components)
        if missing:
            raise ProRuntimeError(
                "PRO category %s did not apply: %s"
                % (self.variant["category"], ", ".join(missing))
            )
        self.substrate_info = {
            "category": self.variant["category"],
            "state_adapter": state_adapter,
            "applied_components": sorted(components),
        }
        return self.transform_observation(self._raw_observation())

    def transform_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        from libero.libero.envs.robustness_perturbations import (
            apply_case_to_observation,
        )

        image_keys = [
            key
            for key, value in observation.items()
            if key.endswith("_image") and isinstance(value, np.ndarray)
        ]
        if not image_keys:
            return observation
        pixels = {key: observation[key][None].copy() for key in image_keys}
        # The substrate draw is frozen per case. Keeping its pixel transform
        # fixed makes control/intervention observations exactly paired before
        # the MAX event and prevents preflight deltas from measuring fresh
        # noise rather than the intervention.
        seed = int(self.case["scenario"]["seed"])
        transformed = apply_case_to_observation(
            {"pixels": pixels}, self.spec, seed, self.options
        )
        result = dict(observation)
        for key in image_keys:
            result[key] = transformed["pixels"][key][0]
        return result

    def step(self, action: Any) -> Any:
        from libero.libero.envs.robustness_perturbations import (
            apply_pinned_object_placements,
        )

        observation, reward, done, info = self._env.step(action)
        if apply_pinned_object_placements(self._vec_env, self._static_info):
            observation = self._raw_observation()
        return self.transform_observation(observation), reward, done, info

    def close(self) -> None:
        self._env.close()


def wrap_case_env(env: Any, case: Dict[str, Any]) -> Any:
    """Return ``env`` unchanged for base cases and a PRO wrapper otherwise."""

    variant = case.get("substrate_variant")
    if variant is None:
        return env
    if variant.get("benchmark") != "LIBERO-PRO":
        raise ProRuntimeError("unsupported substrate benchmark")
    return ProSubstrateEnv(env, case)
