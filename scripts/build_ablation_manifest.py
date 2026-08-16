#!/usr/bin/env python3
"""Select a deterministic, change-balanced ablation subset from Core."""

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from libero_max.manifest import validate_manifest


def _rank(case):
    return hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest()


def select_cases(manifest, pairs_per_type):
    by_type = defaultdict(list)
    for case in manifest["cases"]:
        by_type[case["scenario"]["change_type"]].append(case)
    selected = []
    for change_type in sorted(by_type):
        strata = defaultdict(list)
        for case in by_type[change_type]:
            key = (case["scenario"]["severity"], case["task_suite_name"])
            strata[key].append(case)
        queues = {
            key: deque(sorted(cases, key=_rank))
            for key, cases in sorted(strata.items())
        }
        chosen = []
        while len(chosen) < pairs_per_type and any(queues.values()):
            for key in sorted(queues):
                if queues[key] and len(chosen) < pairs_per_type:
                    chosen.append(queues[key].popleft())
        if len(chosen) != pairs_per_type:
            raise ValueError(
                "%s has only %d cases, requested %d"
                % (change_type, len(chosen), pairs_per_type)
            )
        selected.extend(chosen)
    return sorted(selected, key=lambda case: case["case_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs-per-type", type=int, default=30)
    args = parser.parse_args()
    if args.pairs_per_type < 1:
        parser.error("--pairs-per-type must be positive")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    manifest = {
        **source,
        "benchmark_id": "libero-max-v1-physical-ablation-core",
        "cases": select_cases(source, args.pairs_per_type),
    }
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid ablation manifest: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts = defaultdict(int)
    for case in manifest["cases"]:
        counts[case["scenario"]["change_type"]] += 1
    print(json.dumps({"cases": len(manifest["cases"]), "by_type": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
