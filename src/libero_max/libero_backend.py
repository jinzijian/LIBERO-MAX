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
        self._observation_corruption: Dict[str, Any] = {}
        self._observation_counter = 0
        self._observation_brightness = 1.0

    def reset_episode_state(self) -> None:
        """Clear wrapper-side state after LIBERO hard-resets an episode."""

        self._removed_positions = {}
        self._observation_corruption = {}
        self._observation_counter = 0
        self._observation_brightness = 1.0

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
            "set_visual_theme": self._set_visual_theme,
            "set_sensor_corruption": self._set_sensor_corruption,
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
        return self.transform_observation(self.env._get_observations())

    def transform_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an enabled sensor event without changing simulator physics."""

        brightness = float(getattr(self, "_observation_brightness", 1.0))
        if not self._observation_corruption and brightness == 1.0:
            return observation
        result = dict(observation)
        config = self._observation_corruption
        seed = int(config.get("seed", 0)) + self._observation_counter
        self._observation_counter += 1
        rng = np.random.RandomState(seed)
        for key, value in observation.items():
            if not key.endswith("_image"):
                continue
            image = np.asarray(value)
            if image.ndim != 3 or image.shape[-1] not in {1, 3, 4}:
                continue
            work = image.astype(np.float32)
            if brightness != 1.0:
                work *= brightness
            noise_std = float(config.get("noise_std", 0.0))
            if noise_std:
                work += rng.normal(0.0, noise_std, size=work.shape)
            fraction = float(config.get("occlusion_fraction", 0.0))
            if fraction:
                height, width = work.shape[:2]
                block_height = max(1, int(round(height * math.sqrt(fraction))))
                block_width = max(1, int(round(width * math.sqrt(fraction))))
                top = int(rng.randint(0, max(1, height - block_height + 1)))
                left = int(rng.randint(0, max(1, width - block_width + 1)))
                work[top : top + block_height, left : left + block_width] = 0
            result[key] = np.clip(work, 0, 255).astype(image.dtype)
        return result

    def distance_to_entity(
        self, observation: Dict[str, Any], entity_name: str
    ) -> float:
        if not isinstance(observation, dict) or "robot0_eef_pos" not in observation:
            raise LiberoBackendError("observation does not contain robot0_eef_pos")
        end_effector = _vector3(observation["robot0_eef_pos"], "robot0_eef_pos")
        entity, fixture = self._entity(entity_name)
        entity_position = self._entity_position(entity, fixture)
        return float(np.linalg.norm(end_effector - entity_position))

    def entity_position(self, entity_name: str) -> List[float]:
        """Return an entity's current world position for response-aware scoring."""

        entity, fixture = self._entity(entity_name)
        return self._entity_position(entity, fixture).tolist()

    def is_grasping(self, entity_name: str) -> bool:
        """Return LIBERO's own grasp predicate for a movable object."""

        entity, fixture = self._entity(entity_name)
        if fixture:
            return False
        try:
            gripper = self.env.robots[0].gripper
            contact_geoms = entity.contact_geoms
            return bool(self.env._check_grasp(gripper, contact_geoms))
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise LiberoBackendError(
                "cannot evaluate grasp state for %s" % entity_name
            ) from exc

    def goal_satisfied(self, goal: List[Dict[str, Any]]) -> bool:
        """Evaluate an explicit LIBERO goal without mutating the task definition."""

        if not isinstance(goal, list) or not goal:
            raise LiberoBackendError("alternate_goal must be a non-empty array")
        results = []
        for index, relation in enumerate(goal):
            if not isinstance(relation, dict):
                raise LiberoBackendError(
                    "alternate_goal[%d] must be a JSON object" % index
                )
            predicate = relation.get("predicate")
            arguments = relation.get("arguments")
            if not isinstance(predicate, str) or not predicate:
                raise LiberoBackendError(
                    "alternate_goal[%d].predicate must be non-empty" % index
                )
            if not isinstance(arguments, list) or not all(
                isinstance(value, str) and value for value in arguments
            ):
                raise LiberoBackendError(
                    "alternate_goal[%d].arguments must be non-empty strings" % index
                )
            results.append(
                # LIBERO's parsed BDDL predicates are lower-case even though
                # task catalogs conventionally preserve the source spelling
                # (for example, ``In`` and ``On``).
                bool(self.env._eval_predicate([predicate.lower(), *arguments]))
            )
        return all(results)

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
        before_fovy = float(self.sim.model.cam_fovy[camera_id])
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
        if "fovy_degrees" in change:
            fovy = float(change["fovy_degrees"])
            if not math.isfinite(fovy) or not 5.0 <= fovy <= 150.0:
                raise LiberoBackendError(
                    "fovy_degrees must be finite and between 5 and 150"
                )
            self.sim.model.cam_fovy[camera_id] = fovy
            changed = True
        elif "delta_fovy_degrees" in change:
            fovy = before_fovy + float(change["delta_fovy_degrees"])
            if not math.isfinite(fovy) or not 5.0 <= fovy <= 150.0:
                raise LiberoBackendError(
                    "resulting camera field of view must be between 5 and 150"
                )
            self.sim.model.cam_fovy[camera_id] = fovy
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
            "fovy_before_degrees": before_fovy,
            "fovy_after_degrees": float(self.sim.model.cam_fovy[camera_id]),
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
        # Some Plus scenes are rendered entirely through emission / headlight
        # paths and ignore the model light arrays. A small deterministic camera
        # exposure response makes the same exogenous light event observable in
        # those scenes without changing geometry or task feasibility.
        self._observation_brightness = 0.75 if scale < 1.0 else 1.25
        return {
            "operation": "set_lighting",
            "scale": scale,
            "ambient_before": before["ambient"].tolist(),
            "ambient_after": np.asarray(model.light_ambient).tolist(),
            "observation_brightness_scale": self._observation_brightness,
        }

    def _set_visual_theme(self, change: Dict[str, Any]) -> Dict[str, Any]:
        permutation = change.get("rgb_permutation", [2, 0, 1])
        multiplier = change.get("rgb_multiplier", [0.8, 1.1, 1.25])
        if (
            not isinstance(permutation, list)
            or sorted(permutation) != [0, 1, 2]
        ):
            raise LiberoBackendError("rgb_permutation must permute [0, 1, 2]")
        scale = np.asarray(multiplier, dtype=np.float64)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale < 0):
            raise LiberoBackendError("rgb_multiplier must contain three non-negative numbers")
        model = self.sim.model
        changed_arrays = []
        evidence = {}
        for name in ("geom_rgba", "mat_rgba"):
            if not hasattr(model, name):
                continue
            rgba = getattr(model, name)
            before = np.asarray(rgba).copy()
            rgba[:, :3] = np.clip(before[:, permutation] * scale, 0.0, 1.0)
            changed_arrays.append(name)
            evidence[name + "_rgb_mean_before"] = before[:, :3].mean(axis=0).tolist()
            evidence[name + "_rgb_mean_after"] = (
                np.asarray(rgba)[:, :3].mean(axis=0).tolist()
            )
        if not changed_arrays:
            raise LiberoBackendError("MuJoCo model exposes no visual RGBA arrays")
        return {
            "operation": "set_visual_theme",
            "rgb_permutation": permutation,
            "rgb_multiplier": scale.tolist(),
            "changed_arrays": changed_arrays,
            **evidence,
        }

    def _set_sensor_corruption(self, change: Dict[str, Any]) -> Dict[str, Any]:
        noise_std = float(change.get("noise_std", 0.0))
        fraction = float(change.get("occlusion_fraction", 0.0))
        seed = change.get("seed")
        if not math.isfinite(noise_std) or not 0.0 <= noise_std <= 100.0:
            raise LiberoBackendError("noise_std must be between 0 and 100")
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 0.5:
            raise LiberoBackendError("occlusion_fraction must be between 0 and 0.5")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise LiberoBackendError("set_sensor_corruption requires a non-negative seed")
        self._observation_corruption = {
            "noise_std": noise_std,
            "occlusion_fraction": fraction,
            "seed": seed,
        }
        self._observation_counter = 0
        return {"operation": "set_sensor_corruption", **self._observation_corruption}
