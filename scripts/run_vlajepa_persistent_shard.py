#!/usr/bin/env python3
"""Evaluate one LIBERO-MAX shard with a persistent in-process VLA-JEPA policy."""

import argparse
import collections
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.manifest import load_manifest
from libero_max.pro_runtime import wrap_case_env
from libero_max.substrate import load_case_task
from summarize_cosmos_paired_smoke import classify_persistent_pair


MAX_STEPS = {
    "libero_spatial": 250,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


def _unnormalize_actions(
    normalized_actions: np.ndarray, action_stats: Dict[str, Any]
) -> np.ndarray:
    normalized = np.asarray(normalized_actions, dtype=np.float32).copy()
    mask = np.asarray(
        action_stats.get("mask", np.ones_like(action_stats["min"], dtype=bool)),
        dtype=bool,
    )
    action_high = np.asarray(action_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_stats["min"], dtype=np.float32)
    normalized = np.clip(normalized, -1.0, 1.0)
    normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)
    return np.where(
        mask,
        0.5 * (normalized + 1.0) * (action_high - action_low) + action_low,
        normalized,
    )


def _to_libero_actions(raw_actions: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 7:
        raise ValueError("VLA-JEPA returned invalid raw actions: %s" % (raw.shape,))
    actions = raw[:, :7].copy()
    actions[:, 6] = 1.0 - 2.0 * (actions[:, 6] > 0.5)
    return actions


def _set_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--vlajepa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-interval", type=int, default=7)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")

    sys.path.insert(0, str(args.vlajepa_root))
    import cv2 as cv  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from deployment.model_server.tools.image_tools import (  # noqa: PLC0415
        to_pil_preserve,
    )
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415
    from starVLA.model.framework.base_framework import baseframework  # noqa: PLC0415
    from starVLA.model.tools import read_mode_config  # noqa: PLC0415

    manifest = load_manifest(args.manifest)
    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    policy_seeds = {int(case["policy_seed"]) for case in selected}
    if len(policy_seeds) > 1:
        raise ValueError("one persistent VLA-JEPA shard requires one policy seed")
    policy_seed = next(iter(policy_seeds), 195)
    _set_seed(policy_seed, torch)

    model_config, norm_stats = read_mode_config(args.checkpoint)
    native_chunk = (
        int(model_config["framework"]["action_model"]["future_action_window_size"])
        + 1
    )
    if args.query_interval != native_chunk:
        parser.error(
            "VLA-JEPA checkpoint native chunk is %d; requested q%d"
            % (native_chunk, args.query_interval)
        )
    if "franka" not in norm_stats:
        raise KeyError("checkpoint is missing the official 'franka' norm key")
    action_stats = norm_stats["franka"]["action"]
    # Keep the derived checkpoint symlink path: VLA-JEPA discovers the adjacent
    # rewritten config.yaml from this parent directory. Resolving the symlink
    # silently reintroduces the authors' machine-local backbone paths.
    model = baseframework.from_pretrained(str(args.checkpoint.absolute()))
    model = model.to(torch.bfloat16).to(torch.device("cuda:0")).eval()

    def predict(observation: Dict[str, Any], instruction: str) -> np.ndarray:
        primary = np.ascontiguousarray(
            observation["agentview_image"][::-1, ::-1]
        )
        wrist = np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        images = [
            cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)
            for image in (primary, wrist)
        ]
        state = np.concatenate(
            (
                observation["robot0_eef_pos"],
                _quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        )
        payload = {
            "batch_images": to_pil_preserve([images]),
            "instructions": [instruction],
            "unnorm_key": "franka",
            "do_sample": False,
            "use_ddim": True,
            "num_ddim_steps": 10,
            # Preserve the extra batch dimension used by the official LIBERO client.
            "state": [np.expand_dims(state, axis=0)],
        }
        with torch.no_grad():
            response = model.predict_action(**payload)
        normalized = np.asarray(response["normalized_actions"])[0]
        return _to_libero_actions(_unnormalize_actions(normalized, action_stats))

    args.output_root.mkdir(parents=True, exist_ok=True)
    suite_cache: Dict[str, Any] = {}
    failures = []
    for ordinal, case in enumerate(selected, start=1):
        case_id = case["case_id"]
        case_dir = args.output_root / "cases" / case_id
        summary_path = case_dir / "paired_summary.json"
        done_path = case_dir / "DONE"
        if args.resume and done_path.exists():
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
                    _set_seed(value, torch)

                env, task_description = create_libero_env_with_retry(
                    lambda: _create_libero_env(
                        task,
                        256,
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
                            observation, _, _, _ = wrapped.step(LIBERO_DUMMY_ACTION)
                            continue
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
                                actions = predict(observation, instruction)
                                source = "model"
                            if actions.ndim != 2 or len(actions) < args.query_interval:
                                raise ValueError(
                                    "VLA-JEPA returned invalid action chunk: %s"
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
            print(
                "[%d/%d] %s completed" % (ordinal, len(selected), case_id),
                flush=True,
            )
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
            print(
                "[%d/%d] %s failed: %s" % (ordinal, len(selected), case_id, exc),
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
