#!/usr/bin/env python3
"""Build MAX-Hard Core and Full manifests from an audited Plus catalog."""

import argparse
import json
import re
from pathlib import Path

from libero_max.hard import build_hard_manifests, hard_manifest_summary


CASE_ID_PATTERN = re.compile(
    r"^(libero_(?:spatial|object|goal|10|90))-t(\d+)-"
)


def _rejected_configurations(paths):
    rejected = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for case in report.get("cases", []):
            if case.get("passed"):
                continue
            match = CASE_ID_PATTERN.match(case.get("case_id", ""))
            change_type = case.get("change_type")
            if match is None or not isinstance(change_type, str):
                raise ValueError(
                    "cannot recover task/event key from failed case %r"
                    % case.get("case_id")
                )
            rejected.add((match.group(1), int(match.group(2)), change_type))
    return rejected


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        type=Path,
        help="optional eight-case manifest containing one case per event",
    )
    parser.add_argument(
        "--reject-preflight",
        type=Path,
        action="append",
        default=[],
        help="exclude failed task/event configurations from a preflight report",
    )
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    rejected = _rejected_configurations(args.reject_preflight)
    core, full = build_hard_manifests(catalog, rejected)
    summary = {
        "core": hard_manifest_summary(core),
        "full": hard_manifest_summary(full),
        "core_is_exact_full_subset": {
            case["case_id"] for case in core["cases"]
        }.issubset({case["case_id"] for case in full["cases"]}),
        "rejected_task_event_configurations": len(rejected),
    }
    _write(args.core, core)
    _write(args.full, full)
    _write(args.summary, summary)
    if args.smoke is not None:
        selected = {}
        for case in core["cases"]:
            selected.setdefault(case["scenario"]["change_type"], case)
        smoke = dict(core)
        smoke["benchmark_id"] = "libero-max-hard-smoke"
        smoke["protocol"] = dict(core["protocol"], profile="smoke")
        smoke["cases"] = [selected[event] for event in sorted(selected)]
        _write(args.smoke, smoke)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
