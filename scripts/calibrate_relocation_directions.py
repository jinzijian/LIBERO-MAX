#!/usr/bin/env python3
"""Choose one collision-free relocation direction per task and entity role."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from libero.libero import benchmark

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from libero_max.calibration import CANONICAL_DIRECTIONS, rank_lateral_directions
from libero_max.libero_backend import LiberoBackendError, LiberoMujocoBackend
from libero_max.mujoco_geometry import entity_contact_geom_ids


DISTANCES_M = (0.06, 0.12)
INITIAL_STATES = (0, 1, 2)
CALIBRATION_VERSION = "libero-max-relocation-v1.0"
SETTLE_STEPS = 50
MAX_EXCESS_DISPLACEMENT_M = 0.02
MAX_EXCESS_VERTICAL_DROP_M = 0.01


def _entity_contact_geoms(base_env: Any, name: str) -> Set[int]:
    return entity_contact_geom_ids(base_env, name)


def _contact_partners(sim: Any, entity_geoms: Set[int]) -> Set[int]:
    partners: Set[int] = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        left, right = int(contact.geom1), int(contact.geom2)
        if left in entity_geoms and right not in entity_geoms:
            partners.add(right)
        elif right in entity_geoms and left not in entity_geoms:
            partners.add(left)
    return partners


def _position(backend: LiberoMujocoBackend, name: str) -> np.ndarray:
    try:
        entity, fixture = backend._entity(name)
    except LiberoBackendError:
        geom_ids = sorted(entity_contact_geom_ids(backend.env, name))
        return np.asarray(
            [backend.sim.data.geom_xpos[index] for index in geom_ids],
            dtype=np.float64,
        ).mean(axis=0)
    return backend._entity_position(entity, fixture)


def _validate_delta(
    backend: LiberoMujocoBackend,
    entity_name: str,
    support_name: str,
    direction_xy: Sequence[float],
    distance_m: float,
) -> Dict[str, Any]:
    sim = backend.sim
    base_env = backend.env
    entity_geoms = _entity_contact_geoms(base_env, entity_name)
    support_geoms = _entity_contact_geoms(base_env, support_name)
    saved_state = sim.get_state()
    before = _position(backend, entity_name)
    delta = np.asarray(
        [direction_xy[0] * distance_m, direction_xy[1] * distance_m, 0.0],
        dtype=np.float64,
    )
    try:
        for _ in range(SETTLE_STEPS):
            sim.step()
        sim.forward()
        baseline_settled = _position(backend, entity_name)
        baseline_partners = _contact_partners(sim, entity_geoms)
        baseline_displacement = float(
            np.linalg.norm(baseline_settled - before)
        )
        baseline_vertical_drop = max(
            0.0, float(before[2] - baseline_settled[2])
        )
        sim.set_state(saved_state)
        sim.forward()

        backend.apply_change(
            {
                "operation": "move_object",
                "object": entity_name,
                "delta_position_m": delta.tolist(),
            }
        )
        immediate = _position(backend, entity_name)
        for _ in range(SETTLE_STEPS):
            sim.step()
        sim.forward()
        settled = _position(backend, entity_name)
        partners = _contact_partners(sim, entity_geoms)
        new_unrelated = partners - baseline_partners - support_geoms
        support_contact_required = bool(baseline_partners & support_geoms)
        supported = bool(partners & support_geoms)
        support_valid = supported or not support_contact_required
        exact = math.isclose(
            float(np.linalg.norm(immediate[:2] - before[:2])),
            distance_m,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        intervention_displacement = float(np.linalg.norm(settled - immediate))
        intervention_vertical_drop = max(
            0.0, float(immediate[2] - settled[2])
        )
        excess_displacement = max(
            0.0, intervention_displacement - baseline_displacement
        )
        excess_vertical_drop = max(
            0.0, intervention_vertical_drop - baseline_vertical_drop
        )
        stable = (
            excess_displacement <= MAX_EXCESS_DISPLACEMENT_M
            and excess_vertical_drop <= MAX_EXCESS_VERTICAL_DROP_M
        )
        finite = bool(np.all(np.isfinite(sim.data.qpos))) and bool(
            np.all(np.isfinite(sim.data.qvel))
        )
        return {
            "passed": (
                support_valid
                and not new_unrelated
                and exact
                and stable
                and finite
            ),
            "supported": supported,
            "support_contact_required": support_contact_required,
            "support_valid": support_valid,
            "new_unrelated_contact_geom_ids": sorted(new_unrelated),
            "exact_displacement": exact,
            "stable_relative_to_baseline": stable,
            "baseline_settle_displacement_m": baseline_displacement,
            "intervention_settle_displacement_m": intervention_displacement,
            "excess_settle_displacement_m": excess_displacement,
            "baseline_vertical_drop_m": baseline_vertical_drop,
            "intervention_vertical_drop_m": intervention_vertical_drop,
            "excess_vertical_drop_m": excess_vertical_drop,
            "finite_state": finite,
            "position_before_m": before.tolist(),
            "position_immediate_m": immediate.tolist(),
            "position_settled_m": settled.tolist(),
        }
    finally:
        sim.set_state(saved_state)
        sim.forward()


def _support_for(task: Dict[str, Any], entity_name: str) -> str:
    placement = task["initial_placements"].get(entity_name)
    if not placement or not placement.get("support_entity"):
        raise ValueError("missing support for %s" % entity_name)
    return placement["support_entity"]


def _task_axis(
    backend: LiberoMujocoBackend, task: Dict[str, Any]
) -> Tuple[float, float]:
    target = task.get("primary_target")
    receptacle = task.get("primary_receptacle")
    if not target or not receptacle:
        return (0.0, 0.0)
    start = _position(backend, target)
    goal = _position(backend, receptacle)
    return float(goal[0] - start[0]), float(goal[1] - start[1])


def _calibrate_role(
    task: Dict[str, Any],
    role: str,
    state_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ranked = rank_lateral_directions(record["task_axis_xy"] for record in state_records)
    passing = []
    for direction in ranked:
        direction_key = "%.8f,%.8f" % direction
        if all(
            all(result["passed"] for result in record["checks"][role][direction_key])
            for record in state_records
        ):
            passing.append(direction)
    if not passing:
        return {
            "passed": False,
            "reason": "no direction passes all init states at 6 and 12 cm",
        }
    selected = passing[0]
    return {
        "passed": True,
        "direction_xy": [round(selected[0], 8), round(selected[1], 8)],
        "distances_m": list(DISTANCES_M),
        "validated_init_states": list(INITIAL_STATES),
        "passing_direction_count": len(passing),
    }


def _calibrate_task(task: Dict[str, Any]) -> Dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[task["task_suite_name"]](
        task_order_index=0
    )
    initial_states = suite.get_task_init_states(task["task_index"])
    env, description = get_libero_env(
        suite.get_task(task["task_index"]), "cosmos", resolution=128
    )
    roles = {}
    if task["supports_target_relocation"]:
        roles["target_relocation"] = task["primary_target"]
    if task["supports_receptacle_relocation"]:
        roles["receptacle_relocation"] = task["primary_receptacle"]
    state_records = []
    try:
        for init_state_index in INITIAL_STATES:
            if init_state_index >= len(initial_states):
                raise ValueError("task has fewer than three initial states")
            env.reset()
            env.set_init_state(initial_states[init_state_index])
            backend = LiberoMujocoBackend(env)
            record = {
                "init_state_index": init_state_index,
                "task_axis_xy": list(_task_axis(backend, task)),
                "checks": {},
            }
            for role, entity_name in roles.items():
                support_name = _support_for(task, entity_name)
                record["checks"][role] = {}
                for direction in CANONICAL_DIRECTIONS:
                    direction_key = "%.8f,%.8f" % direction
                    record["checks"][role][direction_key] = [
                        _validate_delta(
                            backend,
                            entity_name,
                            support_name,
                            direction,
                            distance,
                        )
                        for distance in DISTANCES_M
                    ]
            state_records.append(record)
    finally:
        env.close()
    return {
        "task_suite_name": task["task_suite_name"],
        "task_index": task["task_index"],
        "task_description": description,
        "roles": {
            role: _calibrate_role(task, role, state_records) for role in roles
        },
        "states": state_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SUITE:TASK_INDEX",
        help="development-only task filter; omit for a complete calibration",
    )
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    calibrated = copy.deepcopy(catalog)
    selected_keys = set(args.only)
    reports = []
    for task in calibrated["tasks"]:
        task_key = "%s:%d" % (
            task["task_suite_name"],
            task["task_index"],
        )
        if selected_keys and task_key not in selected_keys:
            continue
        if not (
            task["supports_target_relocation"]
            or task["supports_receptacle_relocation"]
        ):
            continue
        report = _calibrate_task(task)
        reports.append(report)
        task["relocation_directions"] = {}
        for role, result in report["roles"].items():
            support_field = "supports_%s" % role
            if result["passed"]:
                task["relocation_directions"][role] = result["direction_xy"]
            else:
                task[support_field] = False
        print(
            json.dumps(
                {
                    "task_suite_name": task["task_suite_name"],
                    "task_index": task["task_index"],
                    "roles": report["roles"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "calibration_version": CALIBRATION_VERSION,
        "complete": not selected_keys,
        "selected_tasks": sorted(selected_keys),
        "task_reports": len(reports),
        "target_relocation_tasks": sum(
            result["passed"]
            for report in reports
            for role, result in report["roles"].items()
            if role == "target_relocation"
        ),
        "receptacle_relocation_tasks": sum(
            result["passed"]
            for report in reports
            for role, result in report["roles"].items()
            if role == "receptacle_relocation"
        ),
    }
    if summary["complete"]:
        calibrated["target_relocation_task_count"] = summary[
            "target_relocation_tasks"
        ]
        calibrated["receptacle_relocation_task_count"] = summary[
            "receptacle_relocation_tasks"
        ]
    calibrated["relocation_calibration"] = summary
    args.output_catalog.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_catalog.write_text(
        json.dumps(calibrated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(
            {"summary": summary, "tasks": reports}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
