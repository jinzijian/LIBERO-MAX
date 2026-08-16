#!/usr/bin/env python3
"""Build the exact outcome-independent complement of one frozen manifest."""

import argparse
import copy
import json
from pathlib import Path

from libero_max.manifest import load_manifest


def build_complement(full: dict, excluded: dict) -> dict:
    full_by_id = {case["case_id"]: case for case in full["cases"]}
    excluded_by_id = {case["case_id"]: case for case in excluded["cases"]}
    if len(full_by_id) != len(full["cases"]):
        raise ValueError("full manifest contains duplicate case IDs")
    if len(excluded_by_id) != len(excluded["cases"]):
        raise ValueError("excluded manifest contains duplicate case IDs")
    missing = sorted(set(excluded_by_id) - set(full_by_id))
    if missing:
        raise ValueError("excluded cases are absent from full manifest: %s" % missing[:5])
    changed = sorted(
        case_id
        for case_id, case in excluded_by_id.items()
        if full_by_id[case_id] != case
    )
    if changed:
        raise ValueError("excluded cases differ from full manifest: %s" % changed[:5])

    result = copy.deepcopy(full)
    result["benchmark_id"] = "%s-complement-%d" % (
        full["benchmark_id"],
        len(excluded_by_id),
    )
    result["protocol"]["selection_contract"] = (
        "exact_case_id_complement; full=%s (%d); excluded=%s (%d); "
        "selected=%d; outcome_independent=true"
        % (
            full["benchmark_id"],
            len(full["cases"]),
            excluded["benchmark_id"],
            len(excluded_by_id),
            len(full["cases"]) - len(excluded_by_id),
        )
    )
    result["cases"] = [
        case for case in result["cases"] if case["case_id"] not in excluded_by_id
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("full_manifest", type=Path)
    parser.add_argument("excluded_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build_complement(
        load_manifest(args.full_manifest), load_manifest(args.excluded_manifest)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["protocol"]["selection_contract"])


if __name__ == "__main__":
    main()
