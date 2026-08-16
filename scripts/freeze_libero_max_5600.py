#!/usr/bin/env python3
"""Freeze the single official, fully preflighted LIBERO-MAX-5600 release."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from libero_max.manifest import validate_manifest
from libero_max.release import audit_max5600_release


RELEASE_VERSION = "2.0.0"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-core", type=Path, required=True)
    parser.add_argument("--physical-preflight", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--rejection-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frozen_core = json.loads(args.frozen_core.read_text(encoding="utf-8"))
    preflight = json.loads(args.physical_preflight.read_text(encoding="utf-8"))
    intent = json.loads(args.intent.read_text(encoding="utf-8"))
    errors = audit_max5600_release(catalog, manifest, frozen_core, preflight)
    errors.extend("Intent manifest: %s" % error for error in validate_manifest(intent))
    if intent.get("protocol", {}).get("scoring_track") != "intent_response":
        errors.append("Intent manifest must use the intent_response track")
    if len(intent.get("cases", [])) != 96:
        errors.append("Intent manifest must contain exactly 96 pairs")
    if errors:
        raise ValueError("LIBERO-MAX-5600 release audit failed: " + "; ".join(errors))

    frozen_manifest = copy.deepcopy(manifest)
    frozen_manifest["benchmark_version"] = RELEASE_VERSION
    selected_keys = {
        (case["task_suite_name"], case["task_index"])
        for case in frozen_manifest["cases"]
    }
    selected_catalog = copy.deepcopy(catalog)
    selected_catalog["tasks"] = [
        task
        for task in selected_catalog["tasks"]
        if (task["task_suite_name"], task["task_index"]) in selected_keys
    ]
    selected_catalog["benchmark_id"] = frozen_manifest["benchmark_id"]
    selected_catalog["benchmark_version"] = RELEASE_VERSION
    frozen_intent = copy.deepcopy(intent)
    frozen_intent["benchmark_id"] = "libero-max-intent-96"
    frozen_intent["benchmark_version"] = RELEASE_VERSION

    rejection_sources = []
    rejected_attempts = 0
    for path in args.rejection_report:
        payload = path.read_bytes()
        report = json.loads(payload)
        failures = sum(not case.get("passed") for case in report.get("cases", []))
        rejected_attempts += failures
        rejection_sources.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "planned": report.get("planned"),
                "passed": report.get("passed"),
                "failures": failures,
            }
        )
    rejection_summary = {
        "sources": rejection_sources,
        "failed_attempt_count": rejected_attempts,
    }
    summary = {
        "benchmark_id": "libero-max-5600",
        "benchmark_version": RELEASE_VERSION,
        "display_name": "LIBERO-MAX-5600",
        "matched_pairs": 5600,
        "rollouts_per_model": 11200,
        "intent_matched_pairs": 96,
        "intent_rollouts_per_model": 192,
        "plus_categories": 7,
        "pairs_per_category": 800,
        "change_types": 8,
        "pairs_per_change_type": 700,
        "physical_preflight_passed": preflight["passed"],
        "development_core_pairs": len(frozen_core["cases"]),
        "retained_development_core_pairs": len(
            {case["case_id"] for case in frozen_core["cases"]}.intersection(
                case["case_id"] for case in frozen_manifest["cases"]
            )
        ),
        "source_benchmark_commit": frozen_manifest["protocol"][
            "source_benchmark_commit"
        ],
        "rejected_attempts": rejected_attempts,
    }
    files = {
        "task_catalog.json": _json_bytes(selected_catalog),
        "libero_max_5600.json": _json_bytes(frozen_manifest),
        "intent_96.json": _json_bytes(frozen_intent),
        "physical_preflight.json": _json_bytes(preflight),
        "rejection_summary.json": _json_bytes(rejection_summary),
        "release_summary.json": _json_bytes(summary),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (args.output_dir / name).write_bytes(payload)
    checksums = "".join(
        "%s  %s\n" % (hashlib.sha256(payload).hexdigest(), name)
        for name, payload in sorted(files.items())
    )
    (args.output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
