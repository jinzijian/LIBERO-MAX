#!/usr/bin/env python3
"""Build MAX-Hard Core and Full manifests from an audited Plus catalog."""

import argparse
import json
from pathlib import Path

from libero_max.hard import build_hard_manifests, hard_manifest_summary


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
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    core, full = build_hard_manifests(catalog)
    summary = {
        "core": hard_manifest_summary(core),
        "full": hard_manifest_summary(full),
        "core_is_exact_full_subset": {
            case["case_id"] for case in core["cases"]
        }.issubset({case["case_id"] for case in full["cases"]}),
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
