#!/usr/bin/env python3
"""Evaluate one manifest shard while loading Cosmos Policy only once."""

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

# LIBERO-Plus sensor-noise variants use this NumPy 1.x dtype alias.
if not hasattr(np, "float_"):
    np.float_ = np.float64

from libero_max.cosmos_integration import CosmosInterventionEnv, retain_action_prefix
from libero_max.manifest import load_manifest
from libero_max.substrate import load_case_task
from libero_max.pro_runtime import wrap_case_env
from summarize_cosmos_paired_smoke import summarize_pair


DEFAULT_DEPS = Path("/vepfs/zijian/alter-wam-deps")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _model_result_from_replay(actions: Any) -> Dict[str, Any]:
    return {
        "actions": [np.asarray(action, dtype=np.float32) for action in actions],
        "future_image_predictions": {
            "future_image": None,
            "future_wrist_image": None,
        },
        "value_prediction": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--query-interval", type=int)
    parser.add_argument("--policy-notification", default="")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_DEPS / "cosmos-assets/Cosmos-Policy-LIBERO-Predict2-2B",
    )
    parser.add_argument("--t5-embeddings", type=Path)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")
    if args.query_interval is not None and args.query_interval < 1:
        parser.error("--query-interval must be positive")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")

    manifest = load_manifest(args.manifest)
    query_interval = int(args.query_interval or manifest["protocol"]["query_interval"])
    manifest = json.loads(json.dumps(manifest))
    manifest["protocol"]["query_interval"] = query_interval
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "manifest.json", manifest)

    from cosmos_policy.experiments.robot import cosmos_utils
    from cosmos_policy.experiments.robot.libero import run_libero_eval
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
    from cosmos_policy.utils.utils import set_seed_everywhere
    from libero.libero import benchmark

    # Paper runs retain structured paired traces. Upstream run_episode saves
    # two MP4s per arm by default, which would create tens of thousands of
    # redundant videos in Full and can exhaust the shared filesystem.
    run_libero_eval.save_rollout_video = lambda *call_args, **call_kwargs: None
    run_libero_eval.save_rollout_video_with_future_image_predictions = (
        lambda *call_args, **call_kwargs: None
    )

    asset_dir = args.asset_dir.resolve()
    t5_path = (args.t5_embeddings or asset_dir / "libero_t5_embeddings.pkl").resolve()
    cfg = run_libero_eval.PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(asset_dir / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=str(asset_dir / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(t5_path),
        trained_with_image_aug=True,
        chunk_size=16,
        num_open_loop_steps=query_interval,
        task_suite_name="libero_object",
        num_trials_per_task=1,
        local_log_dir=str(args.output_root / "eval"),
        randomize_seed=False,
        data_collection=False,
        available_gpus="0",
        seed=195,
        use_variance_scale=False,
        deterministic=True,
        run_id_note="libero-max-persistent",
        ar_future_prediction=False,
        ar_value_prediction=False,
        ar_qvalue_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )
    os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)
    cosmos_utils.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = cosmos_utils.load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = cosmos_utils.get_model(cfg)
    if cfg.chunk_size != cosmos_config.dataloader_train.dataset.chunk_size:
        raise ValueError("Cosmos checkpoint action chunk does not match evaluator")
    resize_size = get_image_resize_size(cfg.model_family)
    suite_cache: Dict[str, Any] = {}

    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
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
            suite_name = case["task_suite_name"]
            task, initial_states = load_case_task(case, benchmark, suite_cache)
            initial_state = initial_states[case["init_state_index"]]
            cfg.task_suite_name = suite_name
            cfg.seed = int(case["policy_seed"])
            cfg.num_open_loop_steps = query_interval
            set_seed_everywhere(cfg.seed)
            rows = {}

            for arm in ("control", "intervention"):
                arm_dir = case_dir / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                trace_path = arm_dir / "trace.jsonl"
                trace_path.write_text("", encoding="utf-8")
                env, task_description = get_libero_env(
                    task, cfg.model_family, resolution=cfg.env_img_res
                )
                env = wrap_case_env(env, case)
                env.seed(cfg.seed)
                wrapped = CosmosInterventionEnv(
                    env=env,
                    task_description=task_description,
                    scenario=case["scenario"],
                    arm=arm,
                    trace_path=trace_path,
                    original_task_index=case["task_index"],
                    init_state_index=case["init_state_index"],
                    policy_notification=args.policy_notification or None,
                )
                wrapped.configure_episode(
                    task_suite_name=suite_name,
                    policy_seed=cfg.seed,
                    query_interval=query_interval,
                    max_policy_steps=run_libero_eval.TASK_MAX_STEPS[suite_name],
                )
                control_queries = {
                    query["policy_step"]: query["actions"]
                    for query in rows.get("control", {}).get("policy_queries", [])
                }
                original_get_action = run_libero_eval.get_action

                def get_action_with_trace(*call_args: Any, **call_kwargs: Any):
                    instruction = wrapped.runtime.current_instruction
                    if len(call_args) >= 5:
                        mutable = list(call_args)
                        mutable[4] = instruction
                        call_args = tuple(mutable)
                    elif "task_label_or_embedding" in call_kwargs:
                        call_kwargs = dict(call_kwargs)
                        call_kwargs["task_label_or_embedding"] = instruction
                    policy_step = max(0, wrapped.total_env_steps - wrapped.warmup_steps)
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
                        result = _model_result_from_replay(control_queries[policy_step])
                        source = "control_replay"
                    else:
                        result = original_get_action(*call_args, **call_kwargs)
                        source = "model"
                    result = retain_action_prefix(result, query_interval)
                    wrapped.record_policy_query(
                        result["actions"], instruction=instruction, source=source
                    )
                    return result

                run_libero_eval.get_action = get_action_with_trace
                try:
                    result = run_libero_eval.run_episode(
                        cfg,
                        wrapped,
                        task_description,
                        model,
                        None,
                        dataset_stats,
                        None,
                        resize_size,
                        initial_state,
                    )
                    rows[arm] = wrapped.record_outcome(bool(result[0]))
                finally:
                    run_libero_eval.get_action = original_get_action
                    env.close()

            summary = summarize_pair(rows["control"], rows["intervention"])
            _write_json(summary_path, summary)
            _write_json(
                case_dir / "status.json",
                {
                    "case_id": case_id,
                    "shard_index": args.shard_index,
                    "summary_exists": True,
                },
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
                    "summary_exists": summary_path.exists(),
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
