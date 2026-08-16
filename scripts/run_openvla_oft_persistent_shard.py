#!/usr/bin/env python3
"""Evaluate one LIBERO-MAX manifest shard with one persistent OpenVLA-OFT model."""

import argparse
import collections
import json
import os
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


def _set_unnorm_key(cfg: Any, model: Any, task_suite_name: str) -> None:
    """Select the suite-specific statistics embedded in the combined checkpoint."""

    candidates = (task_suite_name, task_suite_name + "_no_noops")
    for candidate in candidates:
        if candidate in model.norm_stats:
            cfg.unnorm_key = candidate
            cfg.task_suite_name = task_suite_name
            return
    raise KeyError(
        "checkpoint has no action normalization statistics for %s; available=%s"
        % (task_suite_name, sorted(model.norm_stats))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--openvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-interval", type=int, default=8)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")
    if args.query_interval < 1 or args.query_interval > 8:
        parser.error("OpenVLA-OFT query interval must be in [1, 8]")

    sys.path.insert(0, str(args.openvla_root))
    from experiments.robot.libero.run_libero_eval import (  # noqa: PLC0415
        GenerateConfig,
        initialize_model,
        prepare_observation,
        process_action,
    )
    from experiments.robot.libero.libero_utils import (  # noqa: PLC0415
        get_libero_dummy_action,
    )
    from experiments.robot.robot_utils import (  # noqa: PLC0415
        get_action,
        get_image_resize_size,
        set_seed_everywhere,
    )
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    manifest = load_manifest(args.manifest)
    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    policy_seeds = {int(case["policy_seed"]) for case in selected}
    if len(policy_seeds) > 1:
        raise ValueError("one persistent OpenVLA-OFT shard requires one policy seed")
    policy_seed = next(iter(policy_seeds), 195)

    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=str(args.checkpoint.resolve()),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=args.query_interval,
        task_suite_name="libero_spatial",
        env_img_res=256,
        seed=policy_seed,
        use_wandb=False,
    )
    set_seed_everywhere(policy_seed)
    (
        model,
        action_head,
        proprio_projector,
        noisy_action_projector,
        processor,
    ) = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

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
            _set_unnorm_key(cfg, model, case["task_suite_name"])
            rows = {}
            for arm in ("control", "intervention"):
                arm_dir = case_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                trace_path = arm_dir / "trace.jsonl"
                trace_path.write_text("", encoding="utf-8")

                def reseed(value: int) -> None:
                    set_seed_everywhere(value)

                env, task_description = create_libero_env_with_retry(
                    lambda: _create_libero_env(
                        task,
                        cfg.env_img_res,
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
                                get_libero_dummy_action(cfg.model_family)
                            )
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
                                policy_observation, _ = prepare_observation(
                                    observation, resize_size
                                )
                                actions = np.asarray(
                                    get_action(
                                        cfg,
                                        model,
                                        policy_observation,
                                        instruction,
                                        processor=processor,
                                        action_head=action_head,
                                        proprio_projector=proprio_projector,
                                        noisy_action_projector=noisy_action_projector,
                                        use_film=cfg.use_film,
                                    ),
                                    dtype=np.float32,
                                )
                                actions = process_action(actions, cfg.model_family)
                                source = "model"
                            if actions.ndim != 2 or len(actions) < args.query_interval:
                                raise ValueError(
                                    "OpenVLA-OFT returned an invalid action chunk: %s"
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
