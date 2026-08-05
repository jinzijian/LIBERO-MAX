#!/usr/bin/env python3
"""Create a manifest containing only cases not passed by an earlier preflight."""

import argparse
import json
from pathlib import Path

from libero_max.manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("passed_preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    report = json.loads(args.passed_preflight.read_text(encoding="utf-8"))
    if report.get("benchmark_id") != manifest["benchmark_id"]:
        raise ValueError("preflight benchmark_id does not match manifest")
    passed = {
        case.get("case_id")
        for case in report.get("cases", [])
        if case.get("passed")
    }
    delta = dict(manifest)
    delta["protocol"] = dict(manifest["protocol"], profile="preflight_delta")
    delta["cases"] = [
        case for case in manifest["cases"] if case["case_id"] not in passed
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"delta_cases": len(delta["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
