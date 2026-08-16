#!/usr/bin/env python3
"""Build the source-locked 400-task catalog for MAX-PRO-Hard."""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from libero_max.bddl import is_planar_workspace_placement, parse_bddl_metadata
from libero_max.hard import CHANGE_TYPE_ORDER, eligible_change_types
from libero_max.pro_hard import PRO_CATEGORIES, PRO_TASKS_PER_CATEGORY


BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CLASSIC_CATEGORIES = {
    "semantic": "lan",
    "object": "object",
    "position": "swap",
    "task": "task",
}
EXTENSION_CATEGORIES = {
    "visual_noise_glare": "01_visual_noise_glare",
    "camera_view_angle": "02_camera_view_angle",
    "object_texture": "04_object_texture",
    "view_occlusion": "05_view_occlusion",
    "object_shape": "06_object_shape",
    "initial_pose_position_angle": "07_initial_pose_position_angle",
}


def _load_task_map(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return {suite: list(value[suite]) for suite in BASE_SUITES}
    raise ValueError("could not find literal libero_task_map assignment")


def _language(text: str) -> str:
    match = re.search(r"\(:language\s+(.*?)\)", text, flags=re.S | re.I)
    if match is None:
        raise ValueError("BDDL does not contain a language instruction")
    return " ".join(match.group(1).split())


def _primary_receptacle(metadata):
    if not metadata["manipulated_objects"]:
        return None
    target = metadata["manipulated_objects"][0]
    entities = sorted(
        set(metadata["objects"]) | set(metadata["fixtures"]),
        key=len,
        reverse=True,
    )
    for relation in metadata["goal_relations"]:
        arguments = relation["arguments"]
        if len(arguments) < 2 or arguments[0] != target:
            continue
        for entity in entities:
            if arguments[1] == entity or arguments[1].startswith(entity + "_"):
                return entity
    return None


def _task_record(
    dataset_root: Path,
    source_revision: str,
    task_map,
    category: str,
    source_category: str,
    suite: str,
    bddl_path: Path,
    init_path: Path,
    init_reference_bddl_path: Path,
):
    text = bddl_path.read_text(encoding="utf-8")
    metadata = parse_bddl_metadata(text)
    reference_metadata = parse_bddl_metadata(
        init_reference_bddl_path.read_text(encoding="utf-8")
    )
    reserved_objects = sorted(
        set(metadata["objects"]) - set(reference_metadata["objects"])
    )
    metadata["distractor_objects"] = [
        name
        for name in metadata["distractor_objects"]
        if name not in reserved_objects
    ]
    task_name = bddl_path.stem
    try:
        task_index = task_map[suite].index(task_name)
    except ValueError as exc:
        raise ValueError(
            "%s is not an original task in %s" % (task_name, suite)
        ) from exc
    target = (
        metadata["manipulated_objects"][0] if metadata["manipulated_objects"] else None
    )
    trigger = target or (
        metadata["goal_entity_order"][0] if metadata["goal_entity_order"] else None
    )
    receptacle = _primary_receptacle(metadata)
    record = {
        "task_suite_name": suite,
        "task_index": task_index,
        "task_name": task_name,
        "language": _language(text),
        "pro_category": category,
        "source_category": source_category,
        "source_revision": source_revision,
        "bddl_file": str(bddl_path.relative_to(dataset_root / "bddl_files")),
        "init_states_file": str(init_path.relative_to(dataset_root / "init_files")),
        "init_reference_bddl_file": str(
            init_reference_bddl_path.relative_to(dataset_root / "bddl_files")
        ),
        "bddl_sha256": hashlib.sha256(bddl_path.read_bytes()).hexdigest(),
        "init_reference_bddl_sha256": hashlib.sha256(
            init_reference_bddl_path.read_bytes()
        ).hexdigest(),
        "pro_reserved_objects": reserved_objects,
        **metadata,
        "trigger_entity": trigger,
        "primary_target": target,
        "primary_receptacle": receptacle,
        "supports_target_relocation": bool(
            is_planar_workspace_placement(metadata, target)
        ),
        "supports_receptacle_relocation": bool(
            receptacle in metadata["objects"]
            and is_planar_workspace_placement(metadata, receptacle)
        ),
        "available_distractor_count": len(metadata["distractor_objects"]),
    }
    record["eligible_change_types"] = eligible_change_types(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--libero-task-map", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task_map = _load_task_map(args.libero_task_map)
    rows = []
    for category, suffix in CLASSIC_CATEGORIES.items():
        for suite in BASE_SUITES:
            folder = "%s_%s" % (suite, suffix)
            bddl_dir = args.dataset_root / "bddl_files" / folder
            init_dir = args.dataset_root / "init_files" / folder
            for bddl_path in sorted(bddl_dir.glob("*.bddl")):
                rows.append(
                    _task_record(
                        args.dataset_root,
                        args.source_revision,
                        task_map,
                        category,
                        folder,
                        suite,
                        bddl_path,
                        init_dir / (bddl_path.stem + ".pruned_init"),
                        bddl_path,
                    )
                )
    for category, folder in EXTENSION_CATEGORIES.items():
        for suite in BASE_SUITES:
            bddl_dir = args.dataset_root / "bddl_files" / folder / "bddl" / suite
            init_dir = args.dataset_root / "init_files" / suite
            for bddl_path in sorted(bddl_dir.glob("*.bddl")):
                init_reference_bddl_path = (
                    args.dataset_root
                    / "bddl_files"
                    / (suite + "_lan")
                    / bddl_path.name
                )
                if not init_reference_bddl_path.is_file():
                    raise FileNotFoundError(str(init_reference_bddl_path))
                rows.append(
                    _task_record(
                        args.dataset_root,
                        args.source_revision,
                        task_map,
                        category,
                        folder,
                        suite,
                        bddl_path,
                        init_dir / (bddl_path.stem + ".pruned_init"),
                        init_reference_bddl_path,
                    )
                )
    for row in rows:
        init_path = args.dataset_root / "init_files" / row["init_states_file"]
        if not init_path.is_file():
            raise FileNotFoundError(str(init_path))
    category_counts = Counter(row["pro_category"] for row in rows)
    if set(category_counts) != set(PRO_CATEGORIES) or any(
        category_counts[category] != PRO_TASKS_PER_CATEGORY
        for category in PRO_CATEGORIES
    ):
        raise ValueError("catalog is not exactly 10 categories x 40 tasks")
    eligibility = {
        category: {
            event: sum(
                event in row["eligible_change_types"]
                for row in rows
                if row["pro_category"] == category
            )
            for event in CHANGE_TYPE_ORDER
        }
        for category in PRO_CATEGORIES
    }
    payload = {
        "schema_version": 1,
        "source_dataset": "zhouxueyang/LIBERO-Pro",
        "source_revision": args.source_revision,
        "source_license": "CC-BY-4.0",
        "category_count": len(PRO_CATEGORIES),
        "task_variant_count": len(rows),
        "counts_by_category": dict(category_counts),
        "eligible_tasks_by_category_and_change": eligibility,
        "excluded_upstream_categories": {
            "runtime_object_move": "temporal confound with MAX intervention",
            "environment": "complete BDDL/init artifacts absent at pinned revision",
        },
        "tasks": sorted(
            rows,
            key=lambda row: (
                PRO_CATEGORIES.index(row["pro_category"]),
                BASE_SUITES.index(row["task_suite_name"]),
                row["task_index"],
            ),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "tasks"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
