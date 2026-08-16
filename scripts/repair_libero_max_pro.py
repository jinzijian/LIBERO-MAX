#!/usr/bin/env python3
"""Repair failed PRO-Hard cases without replacing prior passing cases."""

import argparse
import json
from pathlib import Path

from libero_max.pro_hard import (
    _case_candidate_key,
    combine_max8000_manifests,
    pro_hard_summary,
    repair_pro_hard_manifest,
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--pro-catalog", type=Path, required=True)
    parser.add_argument("--current-pro-manifest", type=Path, required=True)
    parser.add_argument("--current-preflight", type=Path, required=True)
    parser.add_argument("--rejection-ledger", type=Path)
    parser.add_argument("--pro-output", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    args = parser.parse_args()
    base = _load(args.base_manifest)
    catalog = _load(args.pro_catalog)
    current = _load(args.current_pro_manifest)
    report = _load(args.current_preflight)
    current_by_id = {case["case_id"]: case for case in current["cases"]}
    if set(current_by_id) != {row["case_id"] for row in report.get("cases", [])}:
        raise ValueError("current preflight must exactly cover current PRO manifest")
    rejected = set()
    if args.rejection_ledger and args.rejection_ledger.is_file():
        rejected = {tuple(item) for item in _load(args.rejection_ledger)["rejected"]}
    failed_ids = {
        row["case_id"] for row in report["cases"] if not row.get("passed")
    }
    rejected.update(_case_candidate_key(current_by_id[case_id]) for case_id in failed_ids)
    repaired = repair_pro_hard_manifest(catalog, current, failed_ids, rejected)
    combined = combine_max8000_manifests(base, repaired)
    summary = {
        **pro_hard_summary(repaired),
        "display_name": "LIBERO-MAX-8000",
        "base_pairs": len(base["cases"]),
        "pro_hard_pairs": len(repaired["cases"]),
        "total_pairs": len(combined["cases"]),
        "total_rollouts_per_model": 2 * len(combined["cases"]),
        "rejected_candidate_configurations": len(rejected),
        "release_gate": "real-MuJoCo preflight required before freeze",
    }
    _write(args.pro_output, repaired)
    _write(args.combined_output, combined)
    _write(args.summary, summary)
    _write(
        args.ledger_output,
        {
            "schema_version": 1,
            "rejected_count": len(rejected),
            "rejected": [list(item) for item in sorted(rejected)],
        },
    )
    print(
        json.dumps(
            {
                "failed_cases_replaced": len(failed_ids),
                "passing_case_ids_preserved": len(current["cases"])
                - len(failed_ids),
                "rejected_candidate_configurations": len(rejected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
