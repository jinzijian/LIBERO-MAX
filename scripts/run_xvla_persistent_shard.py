#!/usr/bin/env python3
"""Evaluate one LIBERO-MAX manifest shard with one persistent X-VLA policy."""

import argparse
import collections
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# LIBERO-Plus and its MuJoCo assets still reference NumPy's pre-2.0 scalar
# alias. Keep the compatibility shim local to the evaluator so the model
# environment can retain its official dependency set.
if not hasattr(np, "float_"):
    np.float_ = np.float64

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.manifest import load_manifest
from libero_max.pro_runtime import wrap_case_env
from libero_max.substrate import load_case_task
from summarize_cosmos_paired_smoke import classify_persistent_pair


MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _create_libero_env(
    task: Any,
    resolution: int,
    seed: int,
    *,
    bddl_root: str,
    env_type: Any,
) -> Any:
    bddl_path = Path(bddl_root) / task.problem_folder / task.bddl_file
    env = env_type(
        bddl_file_name=str(bddl_path),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator).astype(
        np.float32
    )


def _format_observation(raw_obs: dict, env: Any) -> dict:
    observation = {}
    image_keys = (
        ("agentview_image", "image"),
        ("robot0_eye_in_hand_image", "image2"),
    )
    for source, target in image_keys:
        image = raw_obs[source]
        observation["observation.images.%s" % target] = (
            torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float()
            / 255.0
        )

    eef_pos = raw_obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = raw_obs.get("robot0_eef_quat", np.zeros(4))
    eef_mat = env.robots[0].controller.ee_ori_mat
    gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
    gripper_qvel = raw_obs.get("robot0_gripper_qvel", np.zeros(2))
    joint_pos = raw_obs.get("robot0_joint_pos", np.zeros(7))
    joint_vel = raw_obs.get("robot0_joint_vel", np.zeros(7))
    observation["observation.robot_state"] = {
        "eef": {
            "pos": torch.from_numpy(eef_pos.copy()).unsqueeze(0).float(),
            "quat": torch.from_numpy(eef_quat.copy()).unsqueeze(0).float(),
            "mat": torch.from_numpy(eef_mat.copy()).unsqueeze(0).float(),
        },
        "gripper": {
            "qpos": torch.from_numpy(gripper_qpos.copy()).unsqueeze(0).float(),
            "qvel": torch.from_numpy(gripper_qvel.copy()).unsqueeze(0).float(),
        },
        "joints": {
            "pos": torch.from_numpy(joint_pos.copy()).unsqueeze(0).float(),
            "vel": torch.from_numpy(joint_vel.copy()).unsqueeze(0).float(),
        },
    }
    return observation


def _predict_action_chunk(
    *,
    raw_obs: dict,
    env: Any,
    instruction: str,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    query_interval: int,
    query_seed: int,
    action_key: str,
) -> np.ndarray:
    """Materialize one native X-VLA chunk under a query-local random seed."""

    torch.manual_seed(query_seed)
    torch.cuda.manual_seed_all(query_seed)
    policy.reset()
    observation = _format_observation(raw_obs, env)
    observation["task"] = [instruction]
    observation = env_preprocessor(observation)
    observation = preprocessor(observation)
    actions = []
    with torch.inference_mode():
        for _ in range(query_interval):
            action = postprocessor(policy.select_action(observation))
            action = env_postprocessor({action_key: action})[action_key]
            actions.append(action.squeeze(0).detach().cpu().numpy())
    return np.asarray(actions, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-interval", type=int, default=30)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")

    sys.path.insert(0, str(args.lerobot_root / "src"))
    from lerobot.envs.libero import get_libero_dummy_action  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: PLC0415
    from lerobot.policies.xvla.processor_xvla import (  # noqa: PLC0415
        make_xvla_libero_pre_post_processors,
    )
    from lerobot.utils.constants import ACTION  # noqa: PLC0415
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    manifest = load_manifest(args.manifest)
    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    policy_seeds = {int(case["policy_seed"]) for case in selected}
    if len(policy_seeds) > 1:
        raise ValueError("one persistent X-VLA shard requires one policy seed")
    policy_seed = next(iter(policy_seeds), 195)

    device = torch.device("cuda")
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    policy = XVLAPolicy.from_pretrained(str(args.checkpoint.resolve())).to(device).eval()
    native_horizon = int(policy.config.n_action_steps)
    if args.query_interval < 1 or args.query_interval > native_horizon:
        parser.error(
            "X-VLA query interval must be in [1, %d]" % native_horizon
        )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(args.checkpoint.resolve()),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    env_preprocessor, env_postprocessor = make_xvla_libero_pre_post_processors()

    args.output_root.mkdir(parents=True, exist_ok=True)
    suite_cache: Dict[str, Any] = {}
    failures = []
    for ordinal, case in enumerate(selected, start=1):
        case_id = case["case_id"]
        case_dir = args.output_root / "cases" / case_id
        summary_path = case_dir / "paired_summary.json"
        done_path = case_dir / "DONE"
        if args.resume and done_path.exists() and summary_path.exists():
            print("[%d/%d] %s skipped-complete" % (ordinal, len(selected), case_id))
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        for stale in (done_path, case_dir / "FAILED", summary_path):
            if stale.exists():
                stale.unlink()
        _write_json(case_dir / "scenario.json", case["scenario"])
        try:
            task, initial_states = load_case_task(case, benchmark, suite_cache)
            initial_state = initial_states[case["init_state_index"]]
            seed = int(case["policy_seed"])
            rows = {}
            for arm in ("control", "intervention"):
                arm_dir = case_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                trace_path = arm_dir / "trace.jsonl"
                trace_path.write_text("", encoding="utf-8")

                def reseed(value: int) -> None:
                    np.random.seed(value)
                    torch.manual_seed(value)
                    torch.cuda.manual_seed_all(value)

                env, task_description = create_libero_env_with_retry(
                    lambda: _create_libero_env(
                        task,
                        360,
                        seed,
                        bddl_root=get_libero_path("bddl_files"),
                        env_type=OffScreenRenderEnv,
                    ),
                    policy_seed=seed,
                    reseed=reseed,
                )
                env = wrap_case_env(env, case)
                env.seed(seed)
                wrapped = CosmosInterventionEnv(
                    env=env,
                    task_description=task_description,
                    scenario=case["scenario"],
                    arm=arm,
                    trace_path=trace_path,
                    original_task_index=case["task_index"],
                    init_state_index=case["init_state_index"],
                )
                wrapped.configure_episode(
                    task_suite_name=case["task_suite_name"],
                    policy_seed=seed,
                    query_interval=args.query_interval,
                    max_policy_steps=MAX_STEPS[case["task_suite_name"]],
                )
                control_queries = {
                    query["policy_step"]: query["actions"]
                    for query in rows.get("control", {}).get("policy_queries", [])
                }
                success = False
                try:
                    wrapped.reset()
                    observation = wrapped.set_init_state(initial_state)
                    action_plan = collections.deque()
                    total_limit = MAX_STEPS[case["task_suite_name"]] + wrapped.warmup_steps
                    for total_step in range(total_limit):
                        if total_step < wrapped.warmup_steps:
                            observation, _, _, _ = wrapped.step(
                                get_libero_dummy_action()
                            )
                            continue
                        # LIBERO's reset-settling actions are relative deltas.
                        # X-VLA predicts absolute end-effector targets, so only
                        # switch the controller after the settling phase.  If
                        # absolute control is enabled before warmup, the dummy
                        # zero action sends the arm toward the world origin.
                        if total_step == wrapped.warmup_steps:
                            for robot in env.robots:
                                robot.controller.use_delta = False
                        if not action_plan:
                            instruction = wrapped.runtime.current_instruction
                            policy_step = max(
                                0, wrapped.total_env_steps - wrapped.warmup_steps
                            )
                            replay = (
                                arm == "intervention"
                                and not wrapped.runtime.applied
                                and bool(control_queries)
                            )
                            if replay:
                                if policy_step not in control_queries:
                                    raise RuntimeError(
                                        "control trace missing query step %d" % policy_step
                                    )
                                actions = np.asarray(
                                    control_queries[policy_step], dtype=np.float32
                                )
                                source = "control_replay"
                            else:
                                actions = _predict_action_chunk(
                                    raw_obs=observation,
                                    env=env,
                                    instruction=instruction,
                                    policy=policy,
                                    preprocessor=preprocessor,
                                    postprocessor=postprocessor,
                                    env_preprocessor=env_preprocessor,
                                    env_postprocessor=env_postprocessor,
                                    query_interval=args.query_interval,
                                    query_seed=seed + policy_step,
                                    action_key=ACTION,
                                )
                                source = "model"
                            if actions.ndim != 2 or len(actions) < args.query_interval:
                                raise ValueError(
                                    "X-VLA returned an invalid action chunk: %s"
                                    % (actions.shape,)
                                )
                            actions = actions[: args.query_interval]
                            wrapped.record_policy_query(
                                actions, instruction=instruction, source=source
                            )
                            action_plan.extend(actions)
                        observation, _, done, _ = wrapped.step(
                            np.asarray(action_plan.popleft()).tolist()
                        )
                        if done:
                            success = True
                            break
                    rows[arm] = wrapped.record_outcome(success)
                finally:
                    env.close()
            summary, terminal = classify_persistent_pair(
                rows["control"], rows["intervention"]
            )
            if summary is not None:
                _write_json(summary_path, summary)
            status = {"case_id": case_id, "shard_index": args.shard_index}
            if terminal is not None:
                status.update(terminal)
            _write_json(case_dir / "status.json", status)
            done_path.touch()
            print("[%d/%d] %s completed" % (ordinal, len(selected), case_id), flush=True)
        except Exception as exc:
            failures.append(case_id)
            _write_json(
                case_dir / "status.json",
                {
                    "case_id": case_id,
                    "shard_index": args.shard_index,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "traceback": traceback.format_exc(),
                },
            )
            (case_dir / "FAILED").touch()
            print("[%d/%d] %s failed: %s" % (ordinal, len(selected), case_id, exc), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
