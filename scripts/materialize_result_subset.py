#!/usr/bin/env python3
"""Materialize an exact manifest subset from a complete paired-result run."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from libero_max.manifest import load_manifest
from libero_max.results import load_results_jsonl, summarize_results


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    source_summary = json.loads(
        (args.source_root / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    if not source_summary["coverage"].get("complete"):
        raise ValueError("source run must have complete valid coverage")
    manifest = load_manifest(args.manifest)
    source_records = load_results_jsonl(args.source_root / "paired_results.jsonl")
    by_id = {record["pair_id"]: record for record in source_records}
    expected = [case["case_id"] for case in manifest["cases"]]
    missing = sorted(set(expected) - set(by_id))
    if missing:
        raise ValueError("source run is missing subset pairs: %s" % ", ".join(missing))
    selected: List[Dict[str, Any]] = [by_id[pair_id] for pair_id in expected]
    metrics = summarize_results(selected)

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "manifest.json", manifest)
    (args.output_root / "paired_results.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
        encoding="utf-8",
    )
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "protocol": manifest["protocol"],
        "coverage": {
            "planned": len(expected),
            "completed": len(selected),
            "missing": [],
            "invalid": {},
            "terminal_invalid": {},
            "trigger_unreached": 0,
            "execution_complete": True,
            "complete": True,
        },
        "metrics": {
            "overall": metrics["overall"],
            "by_change_family": metrics["by_change_family"],
            "by_change_type": metrics["by_change_type"],
            "by_intervention_draw": metrics["by_intervention_draw"],
            "by_severity": metrics["by_severity"],
            "by_timing_bucket": metrics["by_timing_bucket"],
            "by_response_mode": metrics["by_response_mode"],
        },
        "measurement_notes": source_summary.get("measurement_notes", {}),
        "derived_from": str(args.source_root),
    }
    _write_json(args.output_root / "benchmark_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
