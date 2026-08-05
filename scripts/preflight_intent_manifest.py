#!/usr/bin/env python3
"""Validate intent goals and trigger entities in real LIBERO environments."""

import argparse
import json
from pathlib import Path

from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
from libero.libero import benchmark

from libero_max.libero_backend import LiberoMujocoBackend
from libero_max.manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard")
    manifest = load_manifest(args.manifest)
    rows = []
    suite_cache = {}
    for index, case in enumerate(manifest["cases"]):
        if index % args.num_shards != args.shard_index:
            continue
        suite_name = case["task_suite_name"]
        suite = suite_cache.setdefault(
            suite_name, benchmark.get_benchmark_dict()[suite_name]()
        )
        env = None
        errors = []
        alternate_goal_initially_satisfied = None
        try:
            env, _ = get_libero_env(
                suite.get_task(case["task_index"]), "cosmos", resolution=128
            )
            env.reset()
            observation = env.set_init_state(
                suite.get_task_init_states(case["task_index"])[
                    case["init_state_index"]
                ]
            )
            backend = LiberoMujocoBackend(env)
            trigger = case["scenario"]["trigger"]
            backend.distance_to_entity(observation, trigger["value"])
            change = case["scenario"]["change"]
            if change["operation"] == "replace_instruction":
                alternate_goal_initially_satisfied = backend.goal_satisfied(
                    change["alternate_goal"]
                )
                if alternate_goal_initially_satisfied:
                    errors.append("alternate_goal_initially_satisfied")
        except Exception as exc:
            errors.append("%s: %s" % (type(exc).__name__, exc))
        finally:
            if env is not None:
                env.close()
        rows.append(
            {
                "case_id": case["case_id"],
                "passed": not errors,
                "errors": errors,
                "alternate_goal_initially_satisfied": alternate_goal_initially_satisfied,
            }
        )
    report = {
        "manifest": str(args.manifest),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "checked": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("checked", "passed", "failed")}))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
