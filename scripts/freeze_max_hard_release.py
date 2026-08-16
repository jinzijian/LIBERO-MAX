#!/usr/bin/env python3
"""Freeze fully preflighted MAX-Hard Core and Full artifacts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from libero_max.release import audit_hard_release


RELEASE_VERSION = "2.0.0"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--core-preflight", type=Path, required=True)
    parser.add_argument("--full-preflight", type=Path, required=True)
    parser.add_argument("--rejection-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    core = json.loads(args.core.read_text(encoding="utf-8"))
    full = json.loads(args.full.read_text(encoding="utf-8"))
    core_preflight = json.loads(args.core_preflight.read_text(encoding="utf-8"))
    full_preflight = json.loads(args.full_preflight.read_text(encoding="utf-8"))
    errors = audit_hard_release(
        catalog, core, full, core_preflight, full_preflight
    )
    if errors:
        raise ValueError("MAX-Hard release audit failed: " + "; ".join(errors))

    frozen_catalog = copy.deepcopy(catalog)
    frozen_catalog["benchmark_version"] = RELEASE_VERSION
    frozen_core = copy.deepcopy(core)
    frozen_full = copy.deepcopy(full)
    frozen_core["benchmark_version"] = RELEASE_VERSION
    frozen_full["benchmark_version"] = RELEASE_VERSION

    rejection_sources = []
    failed_configurations = []
    for path in args.rejection_report:
        payload = path.read_bytes()
        report = json.loads(payload)
        failures = [case for case in report.get("cases", []) if not case.get("passed")]
        rejection_sources.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "planned": report.get("planned"),
                "passed": report.get("passed"),
                "failures": len(failures),
            }
        )
        failed_configurations.extend(
            {
                "case_id": case.get("case_id"),
                "scenario_id": case.get("scenario_id"),
                "change_type": case.get("change_type"),
                "validation_errors": case.get("validation_errors", []),
                "source": path.name,
            }
            for case in failures
        )
    rejection_summary = {
        "sources": rejection_sources,
        "failed_attempts": failed_configurations,
        "failed_attempt_count": len(failed_configurations),
    }
    summary = {
        "benchmark_version": RELEASE_VERSION,
        "substrate": "LIBERO-Plus",
        "source_benchmark_commit": frozen_core["protocol"][
            "source_benchmark_commit"
        ],
        "catalog_tasks": len(frozen_catalog["tasks"]),
        "core_matched_pairs": len(frozen_core["cases"]),
        "full_matched_pairs": len(frozen_full["cases"]),
        "core_preflight_passed": core_preflight["passed"],
        "full_preflight_passed": full_preflight["passed"],
        "rejected_attempts": len(failed_configurations),
    }
    files = {
        "task_catalog.json": _json_bytes(frozen_catalog),
        "core.json": _json_bytes(frozen_core),
        "full.json": _json_bytes(frozen_full),
        "physical_preflight_core.json": _json_bytes(core_preflight),
        "physical_preflight_full.json": _json_bytes(full_preflight),
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
