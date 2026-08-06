#!/usr/bin/env python3
"""Compose complete manifest coverage from reusable preflight reports."""

import argparse
import json
from pathlib import Path

from libero_max.manifest import load_manifest


def compose_preflight(manifest, reports):
    manifest_ids = {case["case_id"] for case in manifest.get("cases", [])}
    rows = {}
    for report in reports:
        for case in report.get("cases", []):
            case_id = case.get("case_id")
            if case_id in manifest_ids:
                previous = rows.get(case_id)
                if previous is None or case.get("passed") or not previous.get("passed"):
                    rows[case_id] = case
    selected = sorted(rows.values(), key=lambda case: case["scenario_id"])
    passed = sum(bool(case.get("passed")) for case in selected)
    missing = sorted(manifest_ids - set(rows))
    failures = {
        case["case_id"]: "; ".join(case.get("validation_errors", []))
        or "post-intervention physics validation failed"
        for case in selected
        if not case.get("passed")
    }
    by_change_type = {}
    for case in selected:
        counts = by_change_type.setdefault(
            case.get("change_type", "unknown"), {"planned": 0, "passed": 0}
        )
        counts["planned"] += 1
        counts["passed"] += int(bool(case.get("passed")))
    return {
        "benchmark_id": manifest["benchmark_id"],
        "planned": len(manifest_ids),
        "covered": len(selected),
        "passed": passed,
        "complete": len(selected) == len(manifest_ids) and passed == len(manifest_ids),
        "missing": missing,
        "failures": dict(sorted(failures.items())),
        "by_change_type": dict(sorted(by_change_type.items())),
        "source_reports": len(reports),
        "cases": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    composed = compose_preflight(manifest, reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(composed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in composed.items() if key != "cases"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if composed["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
