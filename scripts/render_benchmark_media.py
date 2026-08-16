#!/usr/bin/env python3
"""Render reproducible before/after simulator previews for the README."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from libero.libero import benchmark

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from libero_max.libero_backend import LiberoMujocoBackend
from libero_max.media_validation import mean_neighbor_delta, validate_render
from libero_max.pro_runtime import wrap_case_env
from libero_max.runtime import InterventionRuntime, TriggerContext
from libero_max.substrate import load_case_task


CHANGE_LABELS = {
    "illumination_switch": "Lighting switch",
    "camera_shift": "Camera shift",
    "visual_theme_switch": "Visual theme",
    "sensor_noise_onset": "Sensor corruption",
    "target_relocation": "Target relocation",
    "receptacle_relocation": "Receptacle relocation",
    "distractor_burst": "Five distractors",
    "obstacle_insertion": "Path obstacle",
}

# This frozen case has a clean, high-contrast light transition on the locked
# MuJoCo/EGL worker. Ranking illumination previews by the largest pixel delta
# instead selected a view-occlusion frame with a striped off-screen buffer.
README_CASE_OVERRIDES = {
    "illumination_switch": "pro-task-10-t01-i02-illumination_switch-d1-p195",
}


def select_representative_cases(
    cases: Iterable[Dict[str, Any]],
    preflight_rows: Dict[str, Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    by_change: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        by_change.setdefault(case["scenario"]["change_type"], []).append(case)
    selected = []
    used_categories = set()
    for change_type in CHANGE_LABELS:
        candidates = by_change.get(change_type, [])

        override_id = README_CASE_OVERRIDES.get(change_type)
        override = next(
            (case for case in candidates if case["case_id"] == override_id), None
        )
        if override is not None:
            selected.append(override)
            used_categories.add(override.get("substrate_category", "base"))
            if len(selected) == limit:
                break
            continue

        def rank(case: Dict[str, Any]) -> tuple:
            row = preflight_rows.get(case["case_id"], {})
            category = case.get("substrate_category", "base")
            return (
                category in used_categories,
                -float(row.get("mean_absolute_raw_pixel_delta", 0.0)),
                case["case_id"],
            )

        if candidates:
            winner = sorted(candidates, key=rank)[0]
            selected.append(winner)
            used_categories.add(winner.get("substrate_category", "base"))
        if len(selected) == limit:
            break
    return selected


def _reset_with_retry(env: Any, attempts: int = 10) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return env.reset()
        except Exception as exc:
            if "Cannot place all objects" not in str(exc) or attempt == attempts:
                raise
    raise AssertionError("unreachable")


def _render_pair(case: Dict[str, Any], resolution: int) -> Dict[str, Any]:
    task, initial_states = load_case_task(case, benchmark)
    env, task_description = get_libero_env(task, "cosmos", resolution=resolution)
    env = wrap_case_env(env, case)
    try:
        env.seed(case["policy_seed"])
        _reset_with_retry(env)
        observation = env.set_init_state(initial_states[case["init_state_index"]])
        backend = LiberoMujocoBackend(env)
        runtime = InterventionRuntime(case["scenario"], backend)
        runtime.reset(task_description)
        setup_events = runtime.apply_setup()
        if setup_events:
            observation = backend.refresh_observation()
        before = np.ascontiguousarray(observation["agentview_image"][::-1])
        trigger = case["scenario"]["trigger"]
        if trigger["type"] == "on_proximity":
            step = 1
            events = frozenset({"proximity:%s" % trigger["value"]})
        else:
            step = int(trigger["value"])
            events = frozenset()
        event = runtime.maybe_apply(
            TriggerContext(step=step, max_steps=1000, events=events)
        )
        if event is None:
            raise RuntimeError("intervention did not fire")
        for _ in range(20):
            backend.sim.step()
        after_observation = backend.refresh_observation()
        after = np.ascontiguousarray(after_observation["agentview_image"][::-1])
        return {
            "before": before,
            "after": after,
            "task_description": task_description,
            "event": event,
            "substrate_runtime": getattr(env, "substrate_info", None),
        }
    finally:
        env.close()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _captioned(array: np.ndarray, title: str, subtitle: str) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    width, height = image.size
    header = 58
    canvas = Image.new("RGB", (width, height + header), (15, 19, 28))
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 7), title, fill=(255, 255, 255), font=_font(20))
    draw.text((12, 33), subtitle, fill=(174, 189, 211), font=_font(13))
    return canvas


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _short_category(value: str) -> str:
    normalized = value.removeprefix("LIBERO-PRO/").replace("_", " ")
    aliases = {
        "initial pose position angle": "initial pose",
        "camera view angle": "camera view",
        "visual noise glare": "noise and glare",
    }
    return "PRO: " + aliases.get(normalized, normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=8)
    # The locked MuJoCo/EGL runtime produces corrupted off-screen buffers at
    # 384 px on the paper worker. 320 px is the largest resolution validated
    # on that runtime and remains comfortably above the policy input size.
    parser.add_argument("--resolution", type=int, default=320)
    args = parser.parse_args()
    if not 64 <= args.resolution <= 320:
        parser.error("--resolution must be between 64 and the validated 320 px limit")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    preflight_rows = {}
    if args.preflight:
        report = json.loads(args.preflight.read_text(encoding="utf-8"))
        preflight_rows = {row["case_id"]: row for row in report.get("cases", [])}
    if args.case_id:
        missing = sorted(set(args.case_id) - set(by_id))
        if missing:
            raise ValueError("unknown case IDs: %s" % ", ".join(missing))
        selected = [by_id[case_id] for case_id in args.case_id]
    else:
        selected = select_representative_cases(
            manifest["cases"], preflight_rows, args.limit
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[Dict[str, Any]] = []
    overview_tiles = []
    for case in selected:
        rendered = _render_pair(case, args.resolution)
        change_type = case["scenario"]["change_type"]
        validate_render(change_type, rendered["before"], rendered["after"])
        label = CHANGE_LABELS.get(change_type, change_type)
        category = case.get("substrate_category", "Base")
        category_label = _short_category(category)
        before = _captioned(rendered["before"], "Before · %s" % label, category_label)
        after = _captioned(rendered["after"], "After · %s" % label, category_label)
        slug = _slug(change_type)
        gif_path = args.output_dir / (slug + ".gif")
        before.save(
            gif_path,
            save_all=True,
            append_images=[after],
            duration=[1100, 1100],
            loop=0,
            optimize=True,
        )
        after.save(args.output_dir / (slug + "-after.png"), optimize=True)
        overview_tiles.append(after.resize((240, 276), Image.Resampling.LANCZOS))
        metadata.append(
            {
                "case_id": case["case_id"],
                "change_type": change_type,
                "substrate_category": category,
                "task_description": rendered["task_description"],
                "gif": gif_path.name,
                "render_resolution": args.resolution,
                "neighbor_delta_before": mean_neighbor_delta(rendered["before"]),
                "neighbor_delta_after": mean_neighbor_delta(rendered["after"]),
                "substrate_runtime": rendered["substrate_runtime"],
            }
        )

    columns = 4
    rows = (len(overview_tiles) + columns - 1) // columns
    overview = Image.new("RGB", (columns * 240, rows * 276), (15, 19, 28))
    for index, tile in enumerate(overview_tiles):
        overview.paste(tile, ((index % columns) * 240, (index // columns) * 276))
    overview.save(args.output_dir / "intervention-overview.png", optimize=True)
    (args.output_dir / "media_manifest.json").write_text(
        json.dumps({"cases": metadata}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rendered": len(metadata), "cases": metadata}, indent=2))


if __name__ == "__main__":
    main()
