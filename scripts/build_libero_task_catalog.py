#!/usr/bin/env python3
"""Build a task/target/distractor catalog from installed LIBERO BDDL files."""

import argparse
import json
from pathlib import Path

from libero.libero import benchmark

from libero_max.bddl import (
    is_planar_workspace_placement,
    parse_bddl_metadata,
    resolve_libero_bddl_path,
)


DEFAULT_SUITES = "libero_spatial,libero_object,libero_goal,libero_10"


def _primary_receptacle(metadata):
    if not metadata["manipulated_objects"]:
        return None
    target = metadata["manipulated_objects"][0]
    entities = sorted(
        set(metadata["objects"]) | set(metadata["fixtures"]), key=len, reverse=True
    )
    for relation in metadata["goal_relations"]:
        arguments = relation["arguments"]
        if len(arguments) < 2 or arguments[0] != target:
            continue
        reference = arguments[1]
        for entity in entities:
            if reference == entity or reference.startswith(entity + "_"):
                return entity
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument(
        "--classification-json",
        type=Path,
        help="optional LIBERO-Plus task_classification.json",
    )
    parser.add_argument("--suites", default=DEFAULT_SUITES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    classifications = None
    if args.classification_json is not None:
        classifications = json.loads(
            args.classification_json.read_text(encoding="utf-8")
        )
    rows = []
    for suite_name in [item.strip() for item in args.suites.split(",")]:
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        suite_classifications = None
        if classifications is not None:
            suite_classifications = classifications.get(suite_name)
            if not isinstance(suite_classifications, list):
                raise ValueError("classification JSON is missing %s" % suite_name)
            if len(suite_classifications) != suite.n_tasks:
                raise ValueError(
                    "%s classification count %d != benchmark count %d"
                    % (suite_name, len(suite_classifications), suite.n_tasks)
                )
        for task_index in range(suite.n_tasks):
            task = suite.get_task(task_index)
            bddl_path, bddl_resolution = resolve_libero_bddl_path(
                args.bddl_root, task.problem_folder, task.bddl_file
            )
            metadata = parse_bddl_metadata(bddl_path.read_text(encoding="utf-8"))
            classification = None
            if suite_classifications is not None:
                classification = suite_classifications[task_index]
                if classification.get("name") != task.name:
                    raise ValueError(
                        "%s task %d classification name mismatch: %r != %r"
                        % (
                            suite_name,
                            task_index,
                            classification.get("name"),
                            task.name,
                        )
                    )
            primary_target = (
                metadata["manipulated_objects"][0]
                if metadata["manipulated_objects"]
                else None
            )
            trigger_entity = primary_target or (
                metadata["goal_entity_order"][0]
                if metadata["goal_entity_order"]
                else None
            )
            primary_receptacle = _primary_receptacle(metadata)
            rows.append(
                {
                    "task_suite_name": suite_name,
                    "task_index": task_index,
                    "task_name": task.name,
                    "language": task.language,
                    "requested_bddl_file": task.bddl_file,
                    "bddl_file": str(bddl_path),
                    "bddl_resolution": bddl_resolution,
                    "plus_category": (
                        classification.get("category")
                        if classification is not None
                        else None
                    ),
                    "plus_difficulty_level": (
                        classification.get("difficulty_level")
                        if classification is not None
                        else None
                    ),
                    **metadata,
                    "trigger_entity": trigger_entity,
                    "primary_target": primary_target,
                    "primary_receptacle": primary_receptacle,
                    "supports_target_relocation": bool(
                        is_planar_workspace_placement(metadata, primary_target)
                    ),
                    "supports_receptacle_relocation": bool(
                        primary_receptacle in metadata["objects"]
                        and is_planar_workspace_placement(
                            metadata, primary_receptacle
                        )
                    ),
                    "available_distractor_count": len(
                        metadata["distractor_objects"]
                    ),
                }
            )
    report = {
        "suites": sorted({row["task_suite_name"] for row in rows}),
        "task_count": len(rows),
        "virtual_bddl_task_count": sum(
            row["bddl_resolution"]["kind"] == "libero_plus_virtual"
            for row in rows
        ),
        "target_relocation_task_count": sum(
            row["supports_target_relocation"] for row in rows
        ),
        "receptacle_relocation_task_count": sum(
            row["supports_receptacle_relocation"] for row in rows
        ),
        "five_distractor_task_count": sum(
            row["available_distractor_count"] >= 5 for row in rows
        ),
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "tasks"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
