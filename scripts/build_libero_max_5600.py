#!/usr/bin/env python3
"""Build the single official LIBERO-MAX-5600 manifest."""

import argparse
import json
from pathlib import Path

from build_max_hard_manifests import _rejected_configurations
from libero_max.hard import (
    build_max5600_manifest,
    expand_rejected_by_physical_scene,
    hard_manifest_summary,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--frozen-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--reject-preflight", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--expand-rejection-change-type", action="append", default=[]
    )
    parser.add_argument(
        "--reject-frozen-preflight", type=Path, action="append", default=[]
    )
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    frozen_core = json.loads(args.frozen_core.read_text(encoding="utf-8"))
    rejected = _rejected_configurations(args.reject_preflight)
    direct_rejections = len(rejected)
    rejected = expand_rejected_by_physical_scene(
        catalog["tasks"], rejected, args.expand_rejection_change_type
    )
    rejected_frozen_case_ids = {
        case["case_id"]
        for path in args.reject_frozen_preflight
        for case in json.loads(path.read_text(encoding="utf-8")).get("cases", [])
        if not case.get("passed")
    }
    manifest = build_max5600_manifest(
        catalog, frozen_core, rejected, rejected_frozen_case_ids
    )
    summary = {
        **hard_manifest_summary(manifest),
        "directly_rejected_task_event_configurations": direct_rejections,
        "rejected_task_event_configurations": len(rejected),
        "rejected_development_core_cases": len(rejected_frozen_case_ids),
        "frozen_core_is_exact_subset": {
            case["case_id"] for case in frozen_core["cases"]
        }.issubset({case["case_id"] for case in manifest["cases"]}),
    }
    _write(args.output, manifest)
    _write(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
