#!/usr/bin/env python3
"""Build MAX-PRO-Hard-2400 and the combined MAX-8000 candidate."""

import argparse
import json
from pathlib import Path

from libero_max.pro_hard import (
    build_pro_hard_manifest,
    combine_max8000_manifests,
    pro_hard_summary,
    rejected_configurations_from_reports,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--pro-catalog", type=Path, required=True)
    parser.add_argument("--pro-output", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reject-preflight", type=Path, action="append", default=[])
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.pro_catalog.read_text(encoding="utf-8"))
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.reject_preflight
    ]
    rejected = rejected_configurations_from_reports(catalog, reports)
    pro = build_pro_hard_manifest(catalog, rejected)
    combined = combine_max8000_manifests(base, pro)
    summary = {
        **pro_hard_summary(pro),
        "display_name": "LIBERO-MAX-8000",
        "base_pairs": 5600,
        "pro_hard_pairs": 2400,
        "total_pairs": len(combined["cases"]),
        "total_rollouts_per_model": 2 * len(combined["cases"]),
        "rejected_candidate_configurations": len(rejected),
        "release_gate": "real-MuJoCo preflight required before freeze",
    }
    _write(args.pro_output, pro)
    _write(args.combined_output, combined)
    _write(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
