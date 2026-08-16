#!/usr/bin/env python3
"""Replay logged Cosmos actions to calibrate semantic proximity thresholds."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from libero.libero import benchmark

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from libero_max.libero_backend import LiberoMujocoBackend


ACTION_PATTERN = re.compile(r"^t:\s*(\d+)\s+action:\s*\[(.*)\]\s*$")


def _load_actions(path: Path):
    actions = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ACTION_PATTERN.match(line)
        if not match:
            continue
        action = np.fromstring(match.group(2), sep=" ", dtype=np.float64)
        if action.size == 0:
            raise ValueError("failed to parse action at t=%s" % match.group(1))
        actions.append((int(match.group(1)), action))
    if not actions:
        raise ValueError("no actions found in %s" % path)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("console", type=Path)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--entity", default="alphabet_soup_1")
    parser.add_argument("--thresholds", default="0.12,0.15,0.18,0.21,0.24")
    args = parser.parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",")]
    actions = _load_actions(args.console)
    suite = benchmark.get_benchmark_dict()[args.suite](task_order_index=0)
    env, description = get_libero_env(
        suite.get_task(args.task_index), "cosmos", resolution=256
    )
    try:
        env.reset()
        observation = env.set_init_state(
            suite.get_task_init_states(args.task_index)[args.init_state_index]
        )
        backend = LiberoMujocoBackend(env)
        for _ in range(10):
            observation, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
        distances = []
        success = False
        for logged_t, action in actions:
            observation, _, done, _ = env.step(action.tolist())
            distance = backend.distance_to_entity(observation, args.entity)
            distances.append((logged_t - 9, logged_t, distance))
            if done:
                success = True
                break
        first_crossings = {}
        for threshold in thresholds:
            crossing = next(
                (row for row in distances if row[2] <= threshold), None
            )
            first_crossings[str(threshold)] = None if crossing is None else {
                "policy_step": crossing[0],
                "logged_t": crossing[1],
                "distance_m": crossing[2],
            }
        report = {
            "task_description": description,
            "entity": args.entity,
            "actions_replayed": len(distances),
            "success": success,
            "minimum_distance_m": min(row[2] for row in distances),
            "first_crossings": first_crossings,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
