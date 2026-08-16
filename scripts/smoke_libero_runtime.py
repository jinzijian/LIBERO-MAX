#!/usr/bin/env python3
"""Apply one camera intervention in a real LIBERO environment."""

import argparse
import json
from pathlib import Path

import numpy as np

from libero_max.libero_backend import LiberoMujocoBackend
from libero_max.runtime import InterventionRuntime, TriggerContext
from libero_max.scenario import load_scenarios, validate_scenario_collection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--scenario-file", type=Path, default=Path("examples/scenarios/pilot.json")
    )
    parser.add_argument("--scenario-id", default="obs_camera_shift_001")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = load_scenarios([args.scenario_file])
    errors = validate_scenario_collection(scenarios)
    if errors:
        raise ValueError("; ".join(errors))
    scenario = next(
        (item for item in scenarios if item["scenario_id"] == args.scenario_id), None
    )
    if scenario is None:
        raise ValueError("unknown scenario-id: %s" % args.scenario_id)
    if scenario["change"]["operation"] != "shift_camera":
        raise ValueError("smoke currently requires a shift_camera scenario")

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    factories = benchmark.get_benchmark_dict()
    if args.suite not in factories:
        raise ValueError("unknown suite: %s" % args.suite)
    suite = factories[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state_id < len(init_states):
        raise ValueError("init-state-id outside available states")

    camera = scenario["change"]["camera"]
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_heights=96,
        camera_widths=96,
        camera_names=[camera, "robot0_eye_in_hand"],
        controller="OSC_POSE",
        control_freq=15,
        horizon=args.max_steps,
        ignore_done=True,
    )
    env.seed(args.seed)
    try:
        env.reset()
        observation = env.set_init_state(init_states[args.init_state_id])
        backend = LiberoMujocoBackend(env)
        runtime = InterventionRuntime(scenario, backend)
        runtime.reset(task.language)
        event = None
        pixel_delta = None

        for step in range(args.max_steps):
            before_image = np.asarray(observation["%s_image" % camera]).copy()
            event = runtime.maybe_apply(
                TriggerContext(step=step, max_steps=args.max_steps)
            )
            if event is not None:
                observation = backend.refresh_observation()
                after_image = np.asarray(observation["%s_image" % camera])
                pixel_delta = float(
                    np.mean(
                        np.abs(
                            after_image.astype(np.int16)
                            - before_image.astype(np.int16)
                        )
                    )
                )
                break
            action = np.zeros(env.env.action_dim, dtype=np.float64)
            observation, _, _, _ = env.step(action)

        if event is None:
            raise RuntimeError("scenario trigger did not fire")
        result = {
            "suite": args.suite,
            "task_id": args.task_id,
            "instruction": task.language,
            "scenario_id": scenario["scenario_id"],
            "event": event,
            "mean_absolute_pixel_delta": pixel_delta,
            "visual_change_detected": bool(pixel_delta and pixel_delta > 0.0),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["visual_change_detected"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
