#!/usr/bin/env python3
"""Replace failed rows in a full preflight report with a repaired delta run."""

import argparse
import json
from pathlib import Path

from libero_max.manifest import load_manifest


def merge_repaired_delta(manifest, base, delta):
    benchmark_id = manifest.get("benchmark_id")
    if base.get("benchmark_id") != benchmark_id:
        raise ValueError("base preflight benchmark_id does not match manifest")
    if delta.get("benchmark_id") != benchmark_id:
        raise ValueError("delta preflight benchmark_id does not match manifest")

    manifest_ids = {case["case_id"] for case in manifest.get("cases", [])}
    base_by_id = {case["case_id"]: case for case in base.get("cases", [])}
    if set(base_by_id) != manifest_ids:
        raise ValueError("base preflight coverage does not match manifest")

    failed_ids = {
        case_id for case_id, case in base_by_id.items() if not case.get("passed")
    }
    delta_by_id = {case["case_id"]: case for case in delta.get("cases", [])}
    if set(delta_by_id) != failed_ids:
        raise ValueError("delta preflight must replace every failed base case exactly")

    repaired = dict(base_by_id)
    repaired.update(delta_by_id)
    cases = sorted(repaired.values(), key=lambda case: case["scenario_id"])
    failures = {
        case["case_id"]: "; ".join(case.get("validation_errors", []))
        or "post-intervention physics validation failed"
        for case in cases
        if not case.get("passed")
    }
    by_change_type = {}
    for case in cases:
        counts = by_change_type.setdefault(
            case.get("change_type", "unknown"), {"planned": 0, "passed": 0}
        )
        counts["planned"] += 1
        counts["passed"] += int(bool(case.get("passed")))
    passed = sum(bool(case.get("passed")) for case in cases)
    return {
        "benchmark_id": benchmark_id,
        "planned": len(cases),
        "passed": passed,
        "complete": passed == len(cases),
        "failures": dict(sorted(failures.items())),
        "by_change_type": dict(sorted(by_change_type.items())),
        "shards": base.get("shards"),
        "delta_shards": delta.get("shards"),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("base_preflight", type=Path)
    parser.add_argument("delta_preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    base = json.loads(args.base_preflight.read_text(encoding="utf-8"))
    delta = json.loads(args.delta_preflight.read_text(encoding="utf-8"))
    merged = merge_repaired_delta(manifest, base, delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in merged.items() if key != "cases"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if merged["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
