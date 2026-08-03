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
from libero_max.runtime import InterventionRuntime, TriggerContext


def _pixel_mad(before: Any, after: Any) -> float:
    left = np.asarray(before, dtype=np.int16)
    right = np.asarray(after, dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError("pre/post images have different shapes")
    return float(np.mean(np.abs(left - right)))


def _run_case(case: Dict[str, Any]) -> Dict[str, Any]:
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
        return {
            "case_id": case["case_id"],
            "task_description": task_description,
            "setup_event_count": len(setup_events),
            "operation": case["scenario"]["change"]["operation"],
            "mean_absolute_raw_pixel_delta": _pixel_mad(before, after),
            "backend_result": event["backend_result"],
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    rows: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for case in manifest["cases"]:
        try:
            row = _run_case(case)
            if row["mean_absolute_raw_pixel_delta"] <= 0:
                raise ValueError("intervention produced zero pixel change")
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            failures[case["case_id"]] = str(exc)
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "planned": len(manifest["cases"]),
        "passed": len(rows),
        "complete": len(rows) == len(manifest["cases"]),
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
