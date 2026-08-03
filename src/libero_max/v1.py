"""Deterministic construction of randomized LIBERO-MAX v1 physical cases."""

import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .manifest import validate_manifest


SAMPLER_VERSION = "libero-max-v1.1"
CHANGE_TYPE_ORDER = (
    "illumination_switch",
    "camera_shift",
    "target_relocation",
    "receptacle_relocation",
    "distractor_burst",
    "obstacle_insertion",
)
DEFAULT_INITIAL_STATES = (0, 1, 2)
DEFAULT_POLICY_SEEDS = (195, 201, 207)
DEFAULT_DRAW_IDS = (0, 1, 2)
RELOCATION_DRAW_IDS = (0, 1)
CHANGE_DRAW_IDS = {
    "illumination_switch": DEFAULT_DRAW_IDS,
    "camera_shift": DEFAULT_DRAW_IDS,
    "target_relocation": RELOCATION_DRAW_IDS,
    "receptacle_relocation": RELOCATION_DRAW_IDS,
    "distractor_burst": DEFAULT_DRAW_IDS,
    "obstacle_insertion": DEFAULT_DRAW_IDS,
}


class V1BuildError(ValueError):
    """Raised when a task catalog cannot produce a valid v1 manifest."""


def _stable_seed(parts: Iterable[Any]) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _rounded_vector(x: float, y: float, z: float = 0.0) -> List[float]:
    return [round(x, 8), round(y, 8), round(z, 8)]


def _calibrated_relocation_direction(
    task: Dict[str, Any], change_type: str
) -> Tuple[float, float]:
    directions = task.get("relocation_directions")
    value = directions.get(change_type) if isinstance(directions, dict) else None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise V1BuildError(
            "%s requires a calibrated two-number relocation direction"
            % change_type
        )
    x, y = float(value[0]), float(value[1])
    norm = math.hypot(x, y)
    if not math.isfinite(norm) or norm <= 1e-9:
        raise V1BuildError(
            "calibrated relocation direction must be finite and nonzero"
        )
    return x / norm, y / norm


def _initial_support(task: Dict[str, Any], entity: str) -> str:
    placements = task.get("initial_placements")
    placement = placements.get(entity) if isinstance(placements, dict) else None
    support = placement.get("support_entity") if isinstance(placement, dict) else None
    if not isinstance(support, str) or not support:
        raise V1BuildError("missing initial support for %s" % entity)
    return support


def eligible_change_types(task: Dict[str, Any]) -> List[str]:
    required = {
        "trigger_entity",
        "supports_target_relocation",
        "supports_receptacle_relocation",
        "available_distractor_count",
    }
    missing = sorted(required - set(task))
    if missing:
        raise V1BuildError(
            "task catalog row is missing v1 fields: %s" % ", ".join(missing)
        )
    if not task["trigger_entity"]:
        return []
    change_types = ["illumination_switch", "camera_shift"]
    if task["supports_target_relocation"]:
        change_types.append("target_relocation")
    if task["supports_receptacle_relocation"]:
        change_types.append("receptacle_relocation")
    if task["available_distractor_count"] >= 5:
        change_types.append("distractor_burst")
    if task["available_distractor_count"] >= 1:
        change_types.append("obstacle_insertion")
    return [name for name in CHANGE_TYPE_ORDER if name in change_types]


def _sample_illumination(
    draw_id: int,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    if draw_id == 0:
        return "low", [], {
            "operation": "set_lighting",
            "scale": 0.55,
        }
    if draw_id == 1:
        return "high", [], {
            "operation": "set_lighting",
            "scale": 0.30,
        }
    setup_scale = 0.30
    return "medium", [{"operation": "set_lighting", "scale": setup_scale}], {
        "operation": "set_lighting",
        "scale": round(1.0 / setup_scale, 8),
    }


def _sample_camera(
    draw_id: int,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    severity = ("low", "medium", "high")[draw_id]
    delta = ([0.02, 0.0, 0.0], [-0.04, 0.0, 0.0], [0.0, 0.06, 0.0])[
        draw_id
    ]
    yaw = (5.0, -10.0, 15.0)[draw_id]
    change = {
        "operation": "shift_camera",
        "camera": "agentview",
        "delta_position_m": delta,
        "yaw_degrees": yaw,
        "calibration": "fixed_global_viewpoint",
    }
    return severity, [], change


def _sample_relocation(
    draw_id: int,
    entity: str,
    direction_xy: Tuple[float, float],
    support_entity: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    distance = (0.06, 0.12)[draw_id]
    severity = "low" if distance == 0.06 else "high"
    return severity, [], {
        "operation": "move_object",
        "object": entity,
        "delta_position_m": _rounded_vector(
            direction_xy[0] * distance, direction_xy[1] * distance
        ),
        "support_entity": support_entity,
        "calibration": "fixed_task_direction",
    }


def _sample_distractors(
    rng: random.Random, task: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    distractors = list(task["distractor_objects"])
    rng.shuffle(distractors)
    selected = distractors[:5]
    setup = [
        {
            "operation": "remove_object",
            "object": entity,
            "offworld_position_m": [-2.0 - 0.2 * index, 0.0, -5.0],
        }
        for index, entity in enumerate(selected)
    ]
    phase = rng.uniform(0.0, 2.0 * math.pi)
    placements = []
    for index, entity in enumerate(selected):
        angle = phase + index * 2.0 * math.pi / len(selected)
        radius = rng.uniform(0.10, 0.18)
        placements.append(
            {
                "object": entity,
                "relative_to": task["trigger_entity"],
                "offset_m": _rounded_vector(
                    radius * math.cos(angle), radius * math.sin(angle)
                ),
                "preserve_initial_z": True,
                "support_entity": _initial_support(task, entity),
            }
        )
    return "high", setup, {
        "operation": "insert_distractors",
        "placements": placements,
    }


def _sample_obstacle(
    rng: random.Random, task: Dict[str, Any], draw_id: int
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    obstacle = rng.choice(list(task["distractor_objects"]))
    lateral_magnitude = (0.06, 0.03, 0.0)[draw_id]
    lateral = lateral_magnitude if rng.random() < 0.5 else -lateral_magnitude
    setup = [
        {
            "operation": "remove_object",
            "object": obstacle,
            "offworld_position_m": [-2.0, 0.0, -5.0],
        }
    ]
    return ("low", "medium", "high")[draw_id], setup, {
        "operation": "insert_obstacle",
        "object": obstacle,
        "path_target": task["trigger_entity"],
        "path_fraction": round(rng.uniform(0.35, 0.65), 8),
        "lateral_offset_m": round(lateral, 8),
        "support_entity": _initial_support(task, obstacle),
    }


def sample_physical_scenario(
    task: Dict[str, Any],
    init_state_index: int,
    change_type: str,
    draw_id: int,
    proximity_distance_m: float = 0.18,
) -> Dict[str, Any]:
    if change_type not in eligible_change_types(task):
        raise V1BuildError(
            "%s is not eligible for %s task %s"
            % (change_type, task["task_suite_name"], task["task_index"])
        )
    allowed_draw_ids = CHANGE_DRAW_IDS[change_type]
    if draw_id not in allowed_draw_ids:
        raise V1BuildError(
            "%s draw_id must be one of %s"
            % (change_type, ", ".join(str(item) for item in allowed_draw_ids))
        )
    seed = _stable_seed(
        (
            SAMPLER_VERSION,
            task["task_suite_name"],
            task["task_index"],
            init_state_index,
            change_type,
            draw_id,
        )
    )
    rng = random.Random(seed)
    if change_type == "illumination_switch":
        severity, setup, change = _sample_illumination(draw_id)
        family, response = "OBS", "continue"
    elif change_type == "camera_shift":
        severity, setup, change = _sample_camera(draw_id)
        family, response = "OBS", "continue"
    elif change_type == "target_relocation":
        severity, setup, change = _sample_relocation(
            draw_id,
            task["primary_target"],
            _calibrated_relocation_direction(task, change_type),
            _initial_support(task, task["primary_target"]),
        )
        family, response = "GEO", "replan"
    elif change_type == "receptacle_relocation":
        severity, setup, change = _sample_relocation(
            draw_id,
            task["primary_receptacle"],
            _calibrated_relocation_direction(task, change_type),
            _initial_support(task, task["primary_receptacle"]),
        )
        family, response = "GEO", "replan"
    elif change_type == "distractor_burst":
        severity, setup, change = _sample_distractors(rng, task)
        family, response = "CLUTTER", "continue"
    elif change_type == "obstacle_insertion":
        severity, setup, change = _sample_obstacle(rng, task, draw_id)
        family, response = "OBSTACLE", "replan"
    else:
        raise AssertionError("unhandled change_type: %s" % change_type)

    suite = task["task_suite_name"]
    task_index = task["task_index"]
    scenario = {
        "scenario_id": "%s-t%02d-i%d-%s-d%d"
        % (suite, task_index, init_state_index, change_type, draw_id),
        "base_task_id": "%s/task_%d" % (suite, task_index),
        "seed": seed,
        "change_family": family,
        "change_type": change_type,
        "severity": severity,
        "trigger": {
            "type": "on_proximity",
            "value": task["trigger_entity"],
            "distance_m": proximity_distance_m,
        },
        "change": change,
        "expected_response_mode": response,
        "safety_constraints": (
            ["no_obstacle_collision", "no_simulator_error"]
            if change_type == "obstacle_insertion"
            else ["no_simulator_error"]
        ),
        "randomization": {
            "sampler": SAMPLER_VERSION,
            "draw_id": draw_id,
            "seed": seed,
        },
    }
    if setup:
        scenario["setup"] = setup
    return scenario


def _profile_assignments(
    profile: str,
    initial_states: Sequence[int],
    policy_seeds: Sequence[int],
    draw_ids: Sequence[int],
) -> List[Tuple[int, int, int]]:
    if profile == "full":
        return [
            (init_state, policy_seed, draw_id)
            for init_state in initial_states
            for policy_seed in policy_seeds
            for draw_id in draw_ids
        ]
    if profile == "core":
        return [
            (
                init_state,
                policy_seeds[(init_offset + draw_offset) % len(policy_seeds)],
                draw_id,
            )
            for init_offset, init_state in enumerate(initial_states)
            for draw_offset, draw_id in enumerate(draw_ids)
        ]
    raise V1BuildError("profile must be core or full")


def build_v1_manifest(
    catalog: Dict[str, Any],
    profile: str = "core",
    initial_states: Sequence[int] = DEFAULT_INITIAL_STATES,
    policy_seeds: Sequence[int] = DEFAULT_POLICY_SEEDS,
) -> Dict[str, Any]:
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise V1BuildError("catalog.tasks must be a non-empty array")
    if len(policy_seeds) != 3 or len(set(policy_seeds)) != 3:
        raise V1BuildError("v1 requires three distinct policy seeds")
    cases: List[Dict[str, Any]] = []
    for task in sorted(
        tasks, key=lambda row: (row["task_suite_name"], row["task_index"])
    ):
        for change_type in eligible_change_types(task):
            assignments = _profile_assignments(
                profile,
                initial_states,
                policy_seeds,
                CHANGE_DRAW_IDS[change_type],
            )
            for init_state, policy_seed, draw_id in assignments:
                scenario = sample_physical_scenario(
                    task, init_state, change_type, draw_id
                )
                cases.append(
                    {
                        "case_id": "%s-p%d" % (scenario["scenario_id"], policy_seed),
                        "task_suite_name": task["task_suite_name"],
                        "task_index": task["task_index"],
                        "init_state_index": init_state,
                        "policy_seed": policy_seed,
                        "timing_bucket": "middle",
                        "scenario": scenario,
                    }
                )
    manifest = {
        "benchmark_id": "libero-max-v1-%s" % profile,
        "benchmark_version": "1.0.0-candidate",
        "protocol": {
            "arms": ["control", "intervention"],
            "query_interval": 16,
            "scoring_track": "physical_completion",
        },
        "cases": cases,
    }
    errors = validate_manifest(manifest)
    if errors:
        raise V1BuildError("generated manifest is invalid: " + "; ".join(errors))
    return manifest


def manifest_design_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    cases = manifest["cases"]
    by_type: Dict[str, int] = {}
    unique_cells = set()
    draw_ids = set()
    for case in cases:
        scenario = case["scenario"]
        change_type = scenario["change_type"]
        by_type[change_type] = by_type.get(change_type, 0) + 1
        unique_cells.add(
            (case["task_suite_name"], case["task_index"], change_type)
        )
        draw_ids.add(scenario["randomization"]["draw_id"])
    return {
        "benchmark_id": manifest["benchmark_id"],
        "task_type_cells": len(unique_cells),
        "matched_pairs": len(cases),
        "episodes": 2 * len(cases),
        "intervention_draw_ids": sorted(draw_ids),
        "pairs_by_change_type": dict(sorted(by_type.items())),
    }
