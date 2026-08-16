#!/usr/bin/env python3
"""Materialize cases in one manifest that are absent from another manifest."""

import argparse
import json
from pathlib import Path

from libero_max.manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("covered_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force-change-draw",
        action="append",
        default=[],
        metavar="CHANGE_TYPE:DRAW_ID",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    covered = load_manifest(args.covered_manifest)
    covered_ids = {case["case_id"] for case in covered["cases"]}
    forced = set()
    for value in args.force_change_draw:
        try:
            change_type, draw_id = value.rsplit(":", 1)
            forced.add((change_type, int(draw_id)))
        except ValueError:
            parser.error("--force-change-draw must use CHANGE_TYPE:DRAW_ID")
    delta = dict(manifest)
    delta["protocol"] = dict(manifest["protocol"], profile="preflight_delta")
    delta["cases"] = [
        case
        for case in manifest["cases"]
        if case["case_id"] not in covered_ids
        or (
            case["scenario"].get("change_type"),
            case["scenario"].get("randomization", {}).get("draw_id"),
        )
        in forced
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"delta_cases": len(delta["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
