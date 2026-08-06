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
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    covered = load_manifest(args.covered_manifest)
    covered_ids = {case["case_id"] for case in covered["cases"]}
    delta = dict(manifest)
    delta["protocol"] = dict(manifest["protocol"], profile="preflight_delta")
    delta["cases"] = [
        case for case in manifest["cases"] if case["case_id"] not in covered_ids
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"delta_cases": len(delta["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
