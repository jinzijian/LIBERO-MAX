"""MuJoCo-backed physical interventions for LIBERO environments.

This module is optional: importing the benchmark core does not import NumPy or
LIBERO. Pass either an OffScreenRenderEnv or its underlying robosuite env.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np


class LiberoBackendError(RuntimeError):
    """Raised when a requested physical intervention cannot be applied."""


def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise LiberoBackendError("%s must contain three finite numbers" % name)
    return array


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


class LiberoMujocoBackend:
    """Apply supported LIBERO-MAX changes to a live MuJoCo model."""

    def __init__(self, env: Any):
        base_env = getattr(env, "env", env)
        if not hasattr(base_env, "sim"):
            raise LiberoBackendError("environment does not expose a MuJoCo sim")
        self.env = base_env
        self._removed_positions: Dict[str, np.ndarray] = {}

    @property
    def sim(self) -> Any:
        # LIBERO uses hard resets by default and replaces the MjSim instance.
        # Resolve it lazily so an intervention never targets the stale,
        # invalidated simulator that existed when this backend was created.
        return self.env.sim

    def apply_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        operation = change.get("operation")
        handlers = {
            "shift_camera": self._shift_camera,
            "move_object": self._move_object,
            "insert_obstacle": self._insert_obstacle,
            "insert_distractors": self._insert_distractors,
            "remove_object": self._remove_object,
            "set_lighting": self._set_lighting,
        }
        if operation not in handlers:
            raise LiberoBackendError("unsupported physical operation: %s" % operation)
        result = handlers[operation](change)
        self.sim.forward()
        return result

    def refresh_observation(self) -> Dict[str, Any]:
        self.sim.forward()
        if hasattr(self.env, "_post_process"):
            self.env._post_process()
        if hasattr(self.env, "_update_observables"):
            self.env._update_observables(force=True)
        return self.env._get_observations()

    def distance_to_entity(
        self, observation: Dict[str, Any], entity_name: str
    ) -> float:
        if not isinstance(observation, dict) or "robot0_eef_pos" not in observation:
            raise LiberoBackendError("observation does not contain robot0_eef_pos")
        end_effector = _vector3(observation["robot0_eef_pos"], "robot0_eef_pos")
        entity, fixture = self._entity(entity_name)
        entity_position = self._entity_position(entity, fixture)
        return float(np.linalg.norm(end_effector - entity_position))

    def _shift_camera(self, change: Dict[str, Any]) -> Dict[str, Any]:
        camera = change.get("camera")
        if not isinstance(camera, str) or not camera:
            raise LiberoBackendError("shift_camera requires change.camera")
        try:
            camera_id = self.sim.model.camera_name2id(camera)
        except Exception as exc:
            raise LiberoBackendError("unknown camera: %s" % camera) from exc

        before_position = np.asarray(self.sim.model.cam_pos[camera_id]).copy()
        before_quaternion = np.asarray(self.sim.model.cam_quat[camera_id]).copy()
        position = before_position.copy()
        quaternion = before_quaternion.copy()
        changed = False

        if "delta_position_m" in change:
            position += _vector3(change["delta_position_m"], "delta_position_m")
            changed = True
        if "yaw_degrees" in change:
            yaw = float(change["yaw_degrees"])
            if not math.isfinite(yaw):
                raise LiberoBackendError("yaw_degrees must be finite")
            half_angle = math.radians(yaw) / 2.0
            yaw_quaternion = np.array(
                [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
            )
            quaternion = _quat_multiply_wxyz(yaw_quaternion, quaternion)
            quaternion /= np.linalg.norm(quaternion)
            changed = True
        if not changed:
            raise LiberoBackendError(
                "shift_camera requires delta_position_m and/or yaw_degrees"
            )

        self.sim.model.cam_pos[camera_id] = position
        self.sim.model.cam_quat[camera_id] = quaternion
        return {
            "operation": "shift_camera",
            "camera": camera,
            "position_before_m": before_position.tolist(),
            "position_after_m": position.tolist(),
            "quaternion_before_wxyz": before_quaternion.tolist(),
            "quaternion_after_wxyz": quaternion.tolist(),
        }

    def _entity(self, name: Any) -> Tuple[Any, bool]:
        if not isinstance(name, str) or not name:
            raise LiberoBackendError("change.object must be a non-empty string")
        if name in self.env.objects_dict:
            return self.env.objects_dict[name], False
        if name in self.env.fixtures_dict:
            return self.env.fixtures_dict[name], True
        available = sorted(set(self.env.objects_dict) | set(self.env.fixtures_dict))
        raise LiberoBackendError(
            "unknown entity %s; available: %s" % (name, ", ".join(available))
        )

    def _entity_position(self, entity: Any, fixture: bool) -> np.ndarray:
        if fixture:
            body_id = self.sim.model.body_name2id(entity.root_body)
            return np.asarray(self.sim.model.body_pos[body_id]).copy()
        qpos = np.asarray(self.sim.data.get_joint_qpos(entity.joints[-1])).copy()
        if qpos.size < 3:
            raise LiberoBackendError("entity does not expose a free-position joint")
        return qpos[:3].copy()

    def _set_entity_position(
        self, entity_name: str, position: np.ndarray
    ) -> Dict[str, Any]:
        entity, fixture = self._entity(entity_name)
        before = self._entity_position(entity, fixture)
        if fixture:
            body_id = self.sim.model.body_name2id(entity.root_body)
            self.sim.model.body_pos[body_id] = position
        else:
            joint = entity.joints[-1]
            qpos = np.asarray(self.sim.data.get_joint_qpos(joint)).copy()
            qpos[:3] = position
            self.sim.data.set_joint_qpos(joint, qpos)
            try:
                qvel = np.asarray(self.sim.data.get_joint_qvel(joint)).copy()
                self.sim.data.set_joint_qvel(joint, np.zeros_like(qvel))
            except (AttributeError, ValueError):
                pass
        return {
            "entity": entity_name,
            "entity_kind": "fixture" if fixture else "object",
            "position_before_m": before.tolist(),
            "position_after_m": position.tolist(),
        }

    def _requested_position(self, change: Dict[str, Any], entity_name: str) -> np.ndarray:
        entity, fixture = self._entity(entity_name)
        before = self._entity_position(entity, fixture)
        if "position_m" in change:
            return _vector3(change["position_m"], "position_m")
        if "delta_position_m" in change:
            return before + _vector3(change["delta_position_m"], "delta_position_m")
        if "relative_to" in change:
            anchor_name = change["relative_to"]
            anchor, anchor_fixture = self._entity(anchor_name)
            position = self._entity_position(anchor, anchor_fixture) + _vector3(
                change.get("offset_m", [0.0, 0.0, 0.0]), "offset_m"
            )
            if change.get("preserve_initial_z", False):
                if entity_name not in self._removed_positions:
                    raise LiberoBackendError(
                        "preserve_initial_z requires the object to be removed in setup"
                    )
                position[2] = self._removed_positions[entity_name][2]
            return position
        raise LiberoBackendError(
            "operation requires position_m, delta_position_m, or relative_to"
        )

    def _move_object(self, change: Dict[str, Any]) -> Dict[str, Any]:
        entity_name = change.get("object")
        position = self._requested_position(change, entity_name)
        result = self._set_entity_position(entity_name, position)
        result["operation"] = "move_object"
        return result

    def _insert_obstacle(self, change: Dict[str, Any]) -> Dict[str, Any]:
        entity_name = change.get("object")
        if "path_target" in change:
            position = self._path_obstacle_position(change, entity_name)
        else:
            position = self._requested_position(change, entity_name)
        result = self._set_entity_position(entity_name, position)
        result["operation"] = "insert_obstacle"
        if "path_target" in change:
            result["placement_rule"] = "eef_target_path"
            result["path_target"] = change["path_target"]
            result["path_fraction"] = change.get("path_fraction", 0.5)
            result["lateral_offset_m"] = change.get("lateral_offset_m", 0.0)
        return result

    def _path_obstacle_position(
        self, change: Dict[str, Any], entity_name: str
    ) -> np.ndarray:
        target_name = change.get("path_target")
        target, target_fixture = self._entity(target_name)
        target_position = self._entity_position(target, target_fixture)
        observation = self.env._get_observations()
        if "robot0_eef_pos" not in observation:
            raise LiberoBackendError("environment does not expose robot0_eef_pos")
        end_effector = _vector3(observation["robot0_eef_pos"], "robot0_eef_pos")
        fraction = float(change.get("path_fraction", 0.5))
        lateral = float(change.get("lateral_offset_m", 0.0))
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise LiberoBackendError("path_fraction must be finite and between 0 and 1")
        if not math.isfinite(lateral):
            raise LiberoBackendError("lateral_offset_m must be finite")
        direction_xy = target_position[:2] - end_effector[:2]
        norm = float(np.linalg.norm(direction_xy))
        if norm <= 1e-9:
            raise LiberoBackendError("end effector and target have identical XY position")
        perpendicular = np.array([-direction_xy[1], direction_xy[0]]) / norm
        position = end_effector + fraction * (target_position - end_effector)
        position[:2] += lateral * perpendicular
        if entity_name not in self._removed_positions:
            raise LiberoBackendError(
                "path obstacle must be removed in shared setup before insertion"
            )
        position[2] = self._removed_positions[entity_name][2]
        return position

    def _insert_distractors(self, change: Dict[str, Any]) -> Dict[str, Any]:
        placements = change.get("placements")
        if not isinstance(placements, list) or not placements:
            raise LiberoBackendError(
                "insert_distractors requires a non-empty placements array"
            )
        results = []
        seen = set()
        for index, placement in enumerate(placements):
            if not isinstance(placement, dict):
                raise LiberoBackendError(
                    "placements[%d] must be a JSON object" % index
                )
            entity_name = placement.get("object")
            if entity_name in seen:
                raise LiberoBackendError(
                    "duplicate distractor object: %s" % entity_name
                )
            seen.add(entity_name)
            position = self._requested_position(placement, entity_name)
            results.append(self._set_entity_position(entity_name, position))
        return {"operation": "insert_distractors", "placements": results}

    def _remove_object(self, change: Dict[str, Any]) -> Dict[str, Any]:
        entity_name = change.get("object")
        entity, fixture = self._entity(entity_name)
        self._removed_positions[entity_name] = self._entity_position(
            entity, fixture
        )
        position = _vector3(
            change.get("offworld_position_m", [0.0, 0.0, -5.0]),
            "offworld_position_m",
        )
        result = self._set_entity_position(entity_name, position)
        result["operation"] = "remove_object"
        return result

    def _set_lighting(self, change: Dict[str, Any]) -> Dict[str, Any]:
        scale = float(change.get("scale", 0.0))
        if not math.isfinite(scale) or scale < 0:
            raise LiberoBackendError("set_lighting scale must be finite and non-negative")
        model = self.sim.model
        before = {
            "ambient": np.asarray(model.light_ambient).copy(),
            "diffuse": np.asarray(model.light_diffuse).copy(),
            "specular": np.asarray(model.light_specular).copy(),
        }
        model.light_ambient[:] = before["ambient"] * scale
        model.light_diffuse[:] = before["diffuse"] * scale
        model.light_specular[:] = before["specular"] * scale
        return {
            "operation": "set_lighting",
            "scale": scale,
            "ambient_before": before["ambient"].tolist(),
            "ambient_after": np.asarray(model.light_ambient).tolist(),
        }
