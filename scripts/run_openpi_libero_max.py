#!/usr/bin/env python3
"""Run one pi0.5-LIBERO episode with LIBERO-MAX intervention tracing."""

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

# LIBERO-Plus's fog augmentation still names the NumPy 1.x scalar alias.
# The pi0.5 simulator client intentionally runs beside a NumPy 2.x Cosmos
# environment, so provide the removed compatibility name before LIBERO is
# imported instead of downgrading the model server runtime.
if not hasattr(np, "float_"):
    np.float_ = np.float64

from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.scenario import load_scenarios, validate_scenario_collection
from libero_max.substrate import load_case_task
from libero_max.pro_runtime import wrap_case_env


MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def _noise_seed(policy_seed: int, query_index: int) -> int:
    digest = hashlib.sha256(
        ("%d:%d" % (policy_seed, query_index)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--arm", choices=("control", "intervention"), required=True)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--init-state-index", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control-trace", type=Path)
    parser.add_argument("--policy-notification")
    args = parser.parse_args()
    if args.replan_steps < 1:
        parser.error("--replan-steps must be positive")

    from cosmos_policy.experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    scenarios = load_scenarios([args.scenario])
    errors = validate_scenario_collection(scenarios)
    if errors or len(scenarios) != 1:
        raise ValueError("scenario must contain exactly one valid record: %s" % errors)
    scenario = scenarios[0]
    control_queries = {}
    if args.control_trace is not None:
        rows = [
            json.loads(line)
            for line in args.control_trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            raise ValueError("control trace must contain exactly one episode")
        control_queries = {
            query["policy_step"]: query["actions"]
            for query in rows[0].get("policy_queries", [])
        }
    case = (
        json.loads(args.case.read_text(encoding="utf-8"))
        if args.case is not None
        else {
            "task_suite_name": args.suite,
            "task_index": args.task_index,
        }
    )
    if case["task_suite_name"] != args.suite or case["task_index"] != args.task_index:
        raise ValueError("--case task identity disagrees with --suite/--task-index")
    task, initial_states = load_case_task(case, benchmark)
    if not 0 <= args.init_state_index < len(initial_states):
        raise IndexError("init-state index is out of range")
    def reseed(seed):
        random.seed(seed)
        np.random.seed(seed)

    env, task_description = create_libero_env_with_retry(
        lambda: get_libero_env(task, "cosmos", resolution=256),
        policy_seed=args.policy_seed,
        reseed=reseed,
    )
    env = wrap_case_env(env, case)
    env.seed(args.policy_seed)
    wrapped = CosmosInterventionEnv(
        env=env,
        task_description=task_description,
        scenario=scenario,
        arm=args.arm,
        trace_path=args.trace,
        original_task_index=args.task_index,
        init_state_index=args.init_state_index,
        policy_notification=args.policy_notification,
    )
    wrapped.configure_episode(
        task_suite_name=args.suite,
        policy_seed=args.policy_seed,
        query_interval=args.replan_steps,
        max_policy_steps=MAX_STEPS[args.suite],
    )
    client = WebsocketClientPolicy(args.host, args.port)
    np.random.seed(args.policy_seed)
    success = False
    try:
        wrapped.reset()
        observation = wrapped.set_init_state(initial_states[args.init_state_index])
        action_plan = collections.deque()
        query_index = 0
        total_limit = MAX_STEPS[args.suite] + wrapped.warmup_steps
        for total_step in range(total_limit):
            if total_step < wrapped.warmup_steps:
                observation, _, _, _ = wrapped.step(get_libero_dummy_action("cosmos"))
                continue
            if not action_plan:
                image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(
                    observation["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(image, 224, 224)
                )
                wrist = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist, 224, 224)
                )
                instruction = wrapped.runtime.current_instruction
                policy_step = max(0, wrapped.total_env_steps - wrapped.warmup_steps)
                replay = (
                    args.arm == "intervention"
                    and not wrapped.runtime.applied
                    and bool(control_queries)
                )
                if replay:
                    if policy_step not in control_queries:
                        raise ValueError(
                            "control trace is missing pre-event query step %d"
                            % policy_step
                        )
                    actions = np.asarray(control_queries[policy_step], dtype=np.float32)
                    source = "control_replay"
                else:
                    request = {
                        "observation/image": image,
                        "observation/wrist_image": wrist,
                        "observation/state": np.concatenate(
                            (
                                observation["robot0_eef_pos"],
                                _quat2axisangle(observation["robot0_eef_quat"]),
                                observation["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": instruction,
                        "libero_max_noise_seed": _noise_seed(
                            args.policy_seed, query_index
                        ),
                    }
                    actions = np.asarray(
                        client.infer(request)["actions"], dtype=np.float32
                    )
                    source = "model"
                if actions.ndim != 2 or len(actions) < args.replan_steps:
                    raise ValueError("pi0.5 returned an invalid action chunk")
                wrapped.record_policy_query(
                    actions, instruction=instruction, source=source
                )
                action_plan.extend(actions[: args.replan_steps])
                query_index += 1
            action = action_plan.popleft()
            observation, _, done, _ = wrapped.step(action.tolist())
            if done:
                success = True
                break
        wrapped.record_outcome(success)
    finally:
        try:
            client._ws.close()
        except Exception:
            pass
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
