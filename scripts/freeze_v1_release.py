#!/usr/bin/env python3
"""Freeze validated candidate artifacts into the versioned v1 release."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from libero_max.release import audit_v1_release


RELEASE_VERSION = "1.0.0"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    core = json.loads(args.core.read_text(encoding="utf-8"))
    full = json.loads(args.full.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    errors = audit_v1_release(catalog, core, full, preflight)
    if errors:
        raise ValueError("release audit failed: " + "; ".join(errors))

    frozen_catalog = copy.deepcopy(catalog)
    frozen_catalog["benchmark_version"] = RELEASE_VERSION
    frozen_core = copy.deepcopy(core)
    frozen_full = copy.deepcopy(full)
    frozen_core["benchmark_version"] = RELEASE_VERSION
    frozen_full["benchmark_version"] = RELEASE_VERSION
    summary = {
        "benchmark_version": RELEASE_VERSION,
        "suites": frozen_catalog["suites"],
        "tasks": frozen_catalog["task_count"],
        "task_type_cells": len(
            {
                (
                    case["task_suite_name"],
                    case["task_index"],
                    case["scenario"]["change_type"],
                )
                for case in frozen_core["cases"]
            }
        ),
        "unique_physical_scenarios": len(frozen_core["cases"]),
        "core_matched_pairs": len(frozen_core["cases"]),
        "full_matched_pairs": len(frozen_full["cases"]),
        "physical_preflight_passed": preflight["passed"],
    }
    files = {
        "task_catalog.json": _json_bytes(frozen_catalog),
        "core.json": _json_bytes(frozen_core),
        "full.json": _json_bytes(frozen_full),
        "physical_preflight.json": _json_bytes(preflight),
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
