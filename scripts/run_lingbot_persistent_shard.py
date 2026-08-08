#!/usr/bin/env python3
"""Evaluate one LIBERO-MAX shard with one persistent LingBot-VA model."""

import argparse
import collections
import copy
import hashlib
import importlib.machinery
import json
import os
import random
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np

from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.lingbot_adapter import DUMMY_ACTION, flatten_lingbot_actions
from libero_max.manifest import load_manifest
from libero_max.pro_runtime import wrap_case_env
from libero_max.substrate import load_case_task, variant_path
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


def _noise_seed(policy_seed: int, query_index: int) -> int:
    digest = hashlib.sha256(
        ("%d:%d" % (policy_seed, query_index)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _extract_observation(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(
            observation["agentview_image"][::-1]
        ),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")

    import torch
    import torch.distributed as dist
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # LingBot imports FlashAttention unconditionally even when the released
    # server selects PyTorch SDPA. Provide an explicit unused sentinel instead
    # of compiling a backend outside this run's locked configuration.
    flash_stub = types.ModuleType("flash_attn")
    flash_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

    def unavailable_flash_attention(*unused_args: Any, **unused_kwargs: Any) -> None:
        raise RuntimeError("FlashAttention is disabled; LingBot uses torch SDPA")

    flash_stub.flash_attn_func = unavailable_flash_attention
    sys.modules.setdefault("flash_attn", flash_stub)

    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    import wan_va.wan_va_server as server_module

    manifest = load_manifest(args.manifest)
    query_interval = int(manifest["protocol"]["query_interval"])
    if query_interval != 16:
        raise ValueError("LingBot adapter requires its native 16-step chunk")
    selected = manifest["cases"][args.shard_index :: args.num_shards]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    policy_seeds = {int(case["policy_seed"]) for case in selected}
    if len(policy_seeds) > 1:
        raise ValueError("one persistent LingBot shard requires one policy seed")
    policy_seed = next(iter(policy_seeds), 195)
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)

    init_distributed(world_size=1, local_rank=0, rank=0)
    config = copy.deepcopy(VA_CONFIGS["libero"])
    config.wan22_pretrained_model_name_or_path = str(args.checkpoint.resolve())
    config.save_root = str((args.output_root / "runtime_state").resolve())
    config.infer_mode = "server"
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_offload = False
    # Official debug saves would write a latent and action file for every
    # policy query. They are not evaluation evidence and are disabled here.
    server_module.save_async = lambda *unused_args, **unused_kwargs: None
    model = server_module.VA_Server(config)

    args.output_root.mkdir(parents=True, exist_ok=True)
    suite_cache: Dict[str, Any] = {}
    failures = []
    try:
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
                    bddl_path = variant_path(
                        get_libero_path("bddl_files"),
                        str(Path(task.problem_folder) / task.bddl_file),
                    )

                    def reseed(value: int) -> None:
                        random.seed(value)
                        np.random.seed(value)

                    env, task_description = create_libero_env_with_retry(
                        lambda: (
                            OffScreenRenderEnv(
                                bddl_file_name=bddl_path,
                                camera_heights=128,
                                camera_widths=128,
                            ),
                            task.language,
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
                        model.infer({"reset": True, "prompt": task_description})
                        action_plan = collections.deque()
                        native_for_cache = None
                        key_frames = []
                        query_index = 0
                        action_index = 0
                        total_limit = (
                            MAX_STEPS[case["task_suite_name"]] + wrapped.warmup_steps
                        )
                        for total_step in range(total_limit):
                            if total_step < wrapped.warmup_steps:
                                observation, _, _, _ = wrapped.step(
                                    DUMMY_ACTION.tolist()
                                )
                                continue
                            if not action_plan:
                                if native_for_cache is not None:
                                    model.infer(
                                        {
                                            "obs": key_frames,
                                            "compute_kv_cache": True,
                                            "imagine": False,
                                            "state": native_for_cache,
                                        }
                                    )
                                key_frames = []
                                action_index = 0
                                instruction = wrapped.runtime.current_instruction
                                policy_step = max(
                                    0,
                                    wrapped.total_env_steps - wrapped.warmup_steps,
                                )
                                generator_seed = _noise_seed(seed, query_index)
                                torch.manual_seed(generator_seed)
                                np.random.seed(generator_seed)
                                native = np.asarray(
                                    model.infer(
                                        {
                                            "obs": _extract_observation(observation),
                                            "prompt": instruction,
                                        }
                                    )["action"],
                                    dtype=np.float32,
                                )
                                predicted = flatten_lingbot_actions(native, query_index)
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
                                    if not np.allclose(
                                        predicted, actions, rtol=0.0, atol=1e-6
                                    ):
                                        raise RuntimeError(
                                            "LingBot pre-event replay prediction drifted"
                                        )
                                    source = "control_replay"
                                else:
                                    actions = predicted
                                    source = "model"
                                wrapped.record_policy_query(
                                    actions, instruction=instruction, source=source
                                )
                                action_plan.extend(actions)
                                native_for_cache = native
                                query_index += 1
                            observation, _, done, _ = wrapped.step(
                                np.asarray(action_plan.popleft()).tolist()
                            )
                            action_index += 1
                            if action_index % 4 == 0 and not (
                                query_index == 1 and action_index == 4
                            ):
                                key_frames.append(_extract_observation(observation))
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
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
