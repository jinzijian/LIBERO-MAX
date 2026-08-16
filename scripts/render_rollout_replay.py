#!/usr/bin/env python3
"""Render a GIF by replaying a frozen paired-rollout action trace in MuJoCo."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from libero.libero import benchmark
from libero_max.cosmos_integration import CosmosInterventionEnv
from libero_max.env_factory import create_libero_env_with_retry
from libero_max.media_validation import validate_render
from libero_max.pro_runtime import wrap_case_env
from libero_max.substrate import load_case_task


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _frame(image: np.ndarray, step: int, event_step: int) -> Image.Image:
    image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 42), (15, 19, 28))
    canvas.paste(image, (0, 42))
    draw = ImageDraw.Draw(canvas)
    phase = "before event" if step < event_step else "after event"
    draw.text(
        (10, 9),
        "Cosmos trace replay · step %d · %s" % (step, phase),
        fill=(255, 255, 255),
        font=_font(15),
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--before-steps", type=int, default=32)
    parser.add_argument("--after-steps", type=int, default=64)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    if args.case_id not in by_id:
        raise ValueError("case is not present in the manifest")
    case = by_id[args.case_id]
    summary = json.loads(
        (args.run_root / "cases" / args.case_id / "paired_summary.json").read_text(
            encoding="utf-8"
        )
    )
    intervention = summary["intervention"]
    event_step = int(summary["intervention_policy_step"])
    action_rows = intervention["executed_actions"]
    action_digest = hashlib.sha256()
    for row in action_rows:
        action_digest.update(
            np.asarray(row["action"], dtype=np.float32).reshape(-1).tobytes()
        )
    task, initial_states = load_case_task(case, benchmark)
    seed = int(case["policy_seed"])

    def reseed(value: int) -> None:
        random.seed(value)
        np.random.seed(value)

    env, task_description = create_libero_env_with_retry(
        lambda: get_libero_env(task, "cosmos", resolution=args.resolution),
        policy_seed=seed,
        reseed=reseed,
    )
    env = wrap_case_env(env, case)
    env.seed(seed)
    wrapped = CosmosInterventionEnv(
        env=env,
        task_description=task_description,
        scenario=case["scenario"],
        arm="intervention",
        trace_path=args.output.with_suffix(".trace.jsonl"),
        original_task_index=case["task_index"],
        init_state_index=case["init_state_index"],
    )
    query_interval = int(summary["intervention"]["query_interval"])
    wrapped.configure_episode(
        task_suite_name=case["task_suite_name"],
        policy_seed=seed,
        query_interval=query_interval,
        max_policy_steps=int(intervention["max_policy_steps"]),
    )
    images = []
    raw_images = []
    start = max(1, event_step - args.before_steps)
    stop = event_step + args.after_steps
    try:
        wrapped.reset()
        observation = wrapped.set_init_state(initial_states[case["init_state_index"]])
        for _ in range(wrapped.warmup_steps):
            observation, _, _, _ = wrapped.step(get_libero_dummy_action("cosmos"))
        for row in action_rows:
            step = int(row["policy_step"])
            observation, _, done, _ = wrapped.step(row["action"])
            if start <= step <= stop and (step - start) % args.stride == 0:
                pixels = np.ascontiguousarray(observation["agentview_image"][::-1])
                raw_images.append(pixels)
                images.append(_frame(pixels, step, event_step))
            if done or step >= stop:
                break
        if len(wrapped.events) != 1:
            raise RuntimeError("replay did not reproduce exactly one intervention")
        actual_step = int(wrapped.events[0]["cosmos_query_boundary_step"])
        if actual_step != event_step:
            raise RuntimeError(
                "event step mismatch: expected %d, found %d" % (event_step, actual_step)
            )
        if len(images) < 4:
            raise RuntimeError("too few rollout frames were captured")
        change_type = case["scenario"]["change_type"]
        validate_render(change_type, raw_images[0], raw_images[-1])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            args.output,
            save_all=True,
            append_images=images[1:],
            duration=90,
            loop=0,
            optimize=True,
        )
        audit: Dict[str, Any] = {
            "case_id": args.case_id,
            "source_run": str(args.run_root),
            "event_step": event_step,
            "replayed_event_step": actual_step,
            "captured_frames": len(images),
            "stride": args.stride,
            "replayed_action_sequence_sha256": action_digest.hexdigest(),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
