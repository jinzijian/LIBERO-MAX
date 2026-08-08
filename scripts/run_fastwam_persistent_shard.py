#!/usr/bin/env python3
"""Evaluate one LIBERO-MAX manifest shard with one persistent FastWAM model."""

import argparse
import collections
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.manifest import load_manifest
from libero_max.pro_runtime import wrap_case_env
from libero_max.substrate import load_case_task
from summarize_cosmos_paired_smoke import summarize_pair


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--query-interval", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from libero.libero import benchmark

    libero_eval_dir = args.fastwam_root / "experiments" / "libero"
    sys.path.insert(0, str(libero_eval_dir))
    from eval_libero_single import (  # noqa: PLC0415
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _predict_action_chunk,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: PLC0415
        FastWAMProcessor,
    )
    from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: PLC0415
        load_dataset_stats_from_json,
    )
    from fastwam.utils.pytorch_utils import set_global_seed  # noqa: PLC0415
    from libero_utils import (  # noqa: PLC0415
        LIBERO_ENV_RESOLUTION,
        get_libero_dummy_action,
        get_libero_env,
    )

    manifest = load_manifest(args.manifest)
    query_interval = int(args.query_interval or manifest["protocol"]["query_interval"])
    if query_interval < 1:
        parser.error("query interval must be positive")
    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    policy_seeds = {int(case["policy_seed"]) for case in selected}
    if len(policy_seeds) > 1:
        raise ValueError("one persistent FastWAM shard requires one policy seed")
    policy_seed = next(iter(policy_seeds), 195)

    overrides = [
        "ckpt=%s" % args.checkpoint.resolve(),
        "EVALUATION.dataset_stats_path=%s" % args.dataset_stats.resolve(),
        "EVALUATION.replan_steps=%d" % query_interval,
        "EVALUATION.visualize_future_video=false",
        "EVALUATION.use_action_ensembler=false",
        "model.redirect_common_files=false",
        "seed=%d" % policy_seed,
        "mixed_precision=bf16",
    ]
    with initialize_config_dir(
        config_dir=str((args.fastwam_root / "configs").resolve()), version_base="1.3"
    ):
        cfg = compose(config_name="sim_libero.yaml", overrides=overrides)
    set_global_seed(policy_seed, get_worker_init_fn=False)
    model_device = "cuda"
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(args.checkpoint))
    model = model.to(model_device).eval()
    dataset_stats = load_dataset_stats_from_json(str(args.dataset_stats))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    action_horizon = int(cfg.data.train.num_frames) - 1
    input_h, input_w = [int(value) for value in cfg.data.train.video_size]

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
                    set_global_seed(value, get_worker_init_fn=False)

                env, task_description = create_libero_env_with_retry(
                    lambda: get_libero_env(task, LIBERO_ENV_RESOLUTION, seed),
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
                    query_interval=query_interval,
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
                    total_limit = (
                        MAX_STEPS[case["task_suite_name"]] + wrapped.warmup_steps
                    )
                    for total_step in range(total_limit):
                        if total_step < wrapped.warmup_steps:
                            observation, _, _, _ = wrapped.step(
                                get_libero_dummy_action()
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
                                        "control trace missing query step %d"
                                        % policy_step
                                    )
                                actions = np.asarray(
                                    control_queries[policy_step], dtype=np.float32
                                )
                                source = "control_replay"
                            else:
                                actions, _, _ = _predict_action_chunk(
                                    obs=observation,
                                    task_description=instruction,
                                    model=model,
                                    processor=processor,
                                    cfg=cfg,
                                    action_horizon=action_horizon,
                                    input_w=input_w,
                                    input_h=input_h,
                                    model_device=model_device,
                                )
                                source = "model"
                            if actions.ndim != 2 or len(actions) < query_interval:
                                raise ValueError(
                                    "FastWAM returned an invalid action chunk"
                                )
                            wrapped.record_policy_query(
                                actions, instruction=instruction, source=source
                            )
                            action_plan.extend(actions[:query_interval])
                        observation, _, done, _ = wrapped.step(
                            np.asarray(action_plan.popleft()).tolist()
                        )
                        if done:
                            success = True
                            break
                    rows[arm] = wrapped.record_outcome(success)
                finally:
                    env.close()
            summary = summarize_pair(rows["control"], rows["intervention"])
            _write_json(summary_path, summary)
            _write_json(
                case_dir / "status.json",
                {"case_id": case_id, "shard_index": args.shard_index},
            )
            done_path.touch()
            print(
                "[%d/%d] %s completed" % (ordinal, len(selected), case_id), flush=True
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
