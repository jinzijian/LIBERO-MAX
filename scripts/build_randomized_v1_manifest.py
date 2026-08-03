#!/usr/bin/env python3
"""Build a deterministic randomized LIBERO-MAX v1 candidate manifest."""

import argparse
import json
from pathlib import Path

from libero_max.v1 import build_v1_manifest, manifest_design_summary


def _integers(value: str):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--profile", choices=("core", "full"), default="core")
    parser.add_argument("--initial-states", default="0,1,2")
    parser.add_argument("--policy-seeds", default="195,201,207")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SUITE:TASK_INDEX",
        help="restrict generation to selected tasks; may be repeated",
    )
    parser.add_argument(
        "--one-per-change-type",
        action="store_true",
        help="development smoke: retain the first case of each change type",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if args.only:
        requested = set(args.only)
        catalog["tasks"] = [
            task
            for task in catalog["tasks"]
            if "%s:%d" % (task["task_suite_name"], task["task_index"])
            in requested
        ]
        found = {
            "%s:%d" % (task["task_suite_name"], task["task_index"])
            for task in catalog["tasks"]
        }
        missing = sorted(requested - found)
        if missing:
            parser.error("unknown --only task(s): %s" % ", ".join(missing))
    manifest = build_v1_manifest(
        catalog,
        profile=args.profile,
        initial_states=_integers(args.initial_states),
        policy_seeds=_integers(args.policy_seeds),
    )
    if args.one_per_change_type:
        retained = []
        seen = set()
        for case in manifest["cases"]:
            change_type = case["scenario"]["change_type"]
            if change_type not in seen:
                retained.append(case)
                seen.add(change_type)
        manifest["cases"] = retained
        manifest["benchmark_id"] += "-one-per-change-type-smoke"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_design_summary(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
