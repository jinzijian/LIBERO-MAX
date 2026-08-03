#!/usr/bin/env python3
"""Apply every manifest intervention in a real LIBERO env without a policy."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# LIBERO's trusted ``.pruned_init`` files contain NumPy objects. PyTorch 2.6
# changed ``torch.load`` to ``weights_only=True`` by default, so opt back into
# the legacy loader before importing LIBERO / Cosmos Policy.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from libero.libero import benchmark

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from libero_max.libero_backend import LiberoMujocoBackend
from libero_max.manifest import load_manifest
from libero_max.preflight import (
    changed_entities,
    select_preflight_cases,
    settle_metrics,
)
from libero_max.runtime import InterventionRuntime, TriggerContext


def _pixel_mad(before: Any, after: Any) -> float:
    left = np.asarray(before, dtype=np.int16)
    right = np.asarray(after, dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError("pre/post images have different shapes")
    return float(np.mean(np.abs(left - right)))


def _entity_geom_ids(backend: LiberoMujocoBackend, name: str) -> set:
    try:
        entity, _ = backend._entity(name)
    except Exception:
        matching = {
            index
            for index in range(int(backend.sim.model.ngeom))
            if (backend.sim.model.geom_id2name(index) or "") == name
            or (backend.sim.model.geom_id2name(index) or "").startswith(
                name + "_"
            )
        }
        if not matching:
            raise ValueError("unknown entity or support geom: %s" % name)
        return matching
    names = getattr(entity, "contact_geoms", None)
    if not names:
        raise ValueError("entity has no contact geoms: %s" % name)
    return {int(backend.sim.model.geom_name2id(item)) for item in names}


def _contact_partners(backend: LiberoMujocoBackend, name: str) -> set:
    entity_geoms = _entity_geom_ids(backend, name)
    partners = set()
    for index in range(int(backend.sim.data.ncon)):
        contact = backend.sim.data.contact[index]
        left, right = int(contact.geom1), int(contact.geom2)
        if left in entity_geoms and right not in entity_geoms:
            partners.add(right)
        elif right in entity_geoms and left not in entity_geoms:
            partners.add(left)
    return partners


def _entity_position(backend: LiberoMujocoBackend, name: str) -> List[float]:
    entity, fixture = backend._entity(name)
    return backend._entity_position(entity, fixture).tolist()


def _run_case(
    case: Dict[str, Any],
    settle_steps: int,
    max_settle_displacement_m: float,
    max_vertical_drop_m: float,
    min_image_mean: float,
    min_image_std: float,
) -> Dict[str, Any]:
    suite = benchmark.get_benchmark_dict()[case["task_suite_name"]](
        task_order_index=0
    )
    if not 0 <= case["task_index"] < suite.n_tasks:
        raise ValueError("task index is outside suite")
    initial_states = suite.get_task_init_states(case["task_index"])
    if not 0 <= case["init_state_index"] < len(initial_states):
        raise ValueError("initial-state index is outside task")
    env, task_description = get_libero_env(
        suite.get_task(case["task_index"]), "cosmos", resolution=256
    )
    try:
        env.reset()
        observation = env.set_init_state(initial_states[case["init_state_index"]])
        backend = LiberoMujocoBackend(env)
        runtime = InterventionRuntime(case["scenario"], backend)
        runtime.reset(task_description)
        physical_entities = changed_entities(case["scenario"]["change"])
        baseline_immediate_positions = {}
        baseline_settled_positions = {}
        baseline_contacts = {}
        if physical_entities:
            saved_state = backend.sim.get_state()
            baseline_immediate_positions = {
                entity: _entity_position(backend, entity)
                for entity, _ in physical_entities
            }
            for _ in range(settle_steps):
                backend.sim.step()
            backend.sim.forward()
            baseline_settled_positions = {
                entity: _entity_position(backend, entity)
                for entity, _ in physical_entities
            }
            baseline_contacts = {
                entity: _contact_partners(backend, entity)
                for entity, _ in physical_entities
            }
            backend.sim.set_state(saved_state)
            backend.sim.forward()
        setup_events = runtime.apply_setup()
        if setup_events:
            observation = backend.refresh_observation()
        before = observation["agentview_image"].copy()
        trigger = case["scenario"]["trigger"]
        if trigger["type"] == "on_proximity":
            step = 1
            events = frozenset({"proximity:%s" % trigger["value"]})
        else:
            step = trigger["value"]
            events = frozenset()
        event = runtime.maybe_apply(
            TriggerContext(step=step, max_steps=1000, events=events)
        )
        if event is None:
            raise ValueError("intervention did not fire")
        after = backend.refresh_observation()["agentview_image"]
        immediate_positions = {
            entity: _entity_position(backend, entity)
            for entity, _ in physical_entities
        }
        if physical_entities:
            for _ in range(settle_steps):
                backend.sim.step()
        after = backend.refresh_observation()["agentview_image"]
        settled_positions = {
            entity: _entity_position(backend, entity)
            for entity, _ in physical_entities
        }
        settling = settle_metrics(immediate_positions, settled_positions)
        baseline_settling = settle_metrics(
            baseline_immediate_positions, baseline_settled_positions
        )
        excess_settling_by_entity = {}
        for entity, _ in physical_entities:
            baseline_entity = settle_metrics(
                {entity: baseline_immediate_positions[entity]},
                {entity: baseline_settled_positions[entity]},
            )
            intervention_entity = settle_metrics(
                {entity: immediate_positions[entity]},
                {entity: settled_positions[entity]},
            )
            excess_settling_by_entity[entity] = {
                "excess_displacement_m": max(
                    0.0,
                    intervention_entity["max_settle_displacement_m"]
                    - baseline_entity["max_settle_displacement_m"],
                ),
                "excess_vertical_drop_m": max(
                    0.0,
                    intervention_entity["max_vertical_drop_m"]
                    - baseline_entity["max_vertical_drop_m"],
                ),
            }
        max_excess_displacement = max(
            (
                item["excess_displacement_m"]
                for item in excess_settling_by_entity.values()
            ),
            default=0.0,
        )
        max_excess_vertical_drop = max(
            (
                item["excess_vertical_drop_m"]
                for item in excess_settling_by_entity.values()
            ),
            default=0.0,
        )
        contacts = {}
        for entity, support in physical_entities:
            support_geoms = _entity_geom_ids(backend, support)
            partners = _contact_partners(backend, entity)
            unexpected = partners - baseline_contacts[entity] - support_geoms
            support_contact_required = bool(
                baseline_contacts[entity] & support_geoms
            )
            supported = bool(partners & support_geoms)
            contacts[entity] = {
                "support_entity": support,
                "supported": supported,
                "support_contact_required": support_contact_required,
                "support_valid": supported or not support_contact_required,
                "unexpected_contact_geom_ids": sorted(unexpected),
            }
        image_mean = float(np.asarray(after, dtype=np.float32).mean())
        image_std = float(np.asarray(after, dtype=np.float32).std())
        physics_valid = (
            max_excess_displacement <= max_settle_displacement_m
            and max_excess_vertical_drop <= max_vertical_drop_m
            and all(
                contact["support_valid"]
                and not contact["unexpected_contact_geom_ids"]
                for contact in contacts.values()
            )
            and bool(np.all(np.isfinite(backend.sim.data.qpos)))
            and bool(np.all(np.isfinite(backend.sim.data.qvel)))
        )
        return {
            "case_id": case["case_id"],
            "scenario_id": case["scenario"]["scenario_id"],
            "change_type": case["scenario"].get("change_type"),
            "intervention_draw_id": case["scenario"].get(
                "randomization", {}
            ).get("draw_id"),
            "task_description": task_description,
            "setup_event_count": len(setup_events),
            "operation": case["scenario"]["change"]["operation"],
            "mean_absolute_raw_pixel_delta": _pixel_mad(before, after),
            "post_image_mean": image_mean,
            "post_image_std": image_std,
            "image_valid": image_mean >= min_image_mean
            and image_std >= min_image_std,
            "settling": settling,
            "baseline_settling": baseline_settling,
            "excess_settling_by_entity": excess_settling_by_entity,
            "max_excess_settle_displacement_m": max_excess_displacement,
            "max_excess_vertical_drop_m": max_excess_vertical_drop,
            "contacts": contacts,
            "physics_valid": physics_valid,
            "backend_result": event["backend_result"],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--keep-policy-replicates",
        action="store_true",
        help="preflight repeated policy-seed cases instead of unique scenarios",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--max-settle-displacement-m", type=float, default=0.02)
    parser.add_argument("--max-vertical-drop-m", type=float, default=0.01)
    parser.add_argument("--min-image-mean", type=float, default=8.0)
    parser.add_argument("--min-image-std", type=float, default=5.0)
    parser.add_argument(
        "--change-type",
        action="append",
        default=[],
        help="restrict to one change type; may be repeated",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    selected_cases, selection = select_preflight_cases(
        manifest["cases"],
        unique_scenarios=not args.keep_policy_replicates,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        change_types=args.change_type,
    )
    rows: List[Dict[str, Any]] = []
    passed = 0
    failures: Dict[str, str] = {}
    for case in selected_cases:
        try:
            row = _run_case(
                case,
                settle_steps=args.settle_steps,
                max_settle_displacement_m=args.max_settle_displacement_m,
                max_vertical_drop_m=args.max_vertical_drop_m,
                min_image_mean=args.min_image_mean,
                min_image_std=args.min_image_std,
            )
            validation_errors = []
            if row["mean_absolute_raw_pixel_delta"] <= 0:
                validation_errors.append("intervention produced zero pixel change")
            if not row["image_valid"]:
                validation_errors.append(
                    "post-intervention image failed visibility thresholds"
                )
            if not row["physics_valid"]:
                validation_errors.append(
                    "post-intervention physics validation failed"
                )
            row["validation_errors"] = validation_errors
            row["passed"] = not validation_errors
            rows.append(row)
            if validation_errors:
                failures[case["case_id"]] = "; ".join(validation_errors)
            else:
                passed += 1
            print(json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            failures[case["case_id"]] = str(exc)
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "selection": selection,
        "planned": len(selected_cases),
        "passed": passed,
        "complete": passed == len(selected_cases),
        "failures": failures,
        "cases": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
