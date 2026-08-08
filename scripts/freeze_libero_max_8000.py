#!/usr/bin/env python3
"""Freeze MAX-8000 only after all 2,400 PRO-Hard cases pass MuJoCo."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from libero_max.manifest import validate_manifest
from libero_max.pro_hard import combine_max8000_manifests
from libero_max.release import audit_max8000_composition, audit_pro_hard_release


RELEASE_VERSION = "3.0.0"


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--pro-catalog", type=Path, required=True)
    parser.add_argument("--pro-manifest", type=Path, required=True)
    parser.add_argument("--pro-preflight", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.pro_catalog.read_text(encoding="utf-8"))
    pro = json.loads(args.pro_manifest.read_text(encoding="utf-8"))
    preflight = json.loads(args.pro_preflight.read_text(encoding="utf-8"))
    intent = json.loads(args.intent.read_text(encoding="utf-8"))
    errors = audit_pro_hard_release(catalog, pro, preflight)
    errors.extend("Intent manifest: %s" % error for error in validate_manifest(intent))
    if len(intent.get("cases", [])) != 96:
        errors.append("Intent manifest must contain exactly 96 pairs")
    combined = combine_max8000_manifests(base, pro)
    errors.extend(audit_max8000_composition(base, pro, combined))
    if errors:
        raise ValueError("LIBERO-MAX-8000 release audit failed: " + "; ".join(errors))
    frozen_pro = copy.deepcopy(pro)
    frozen_pro["benchmark_version"] = RELEASE_VERSION
    frozen_combined = copy.deepcopy(combined)
    frozen_combined["benchmark_version"] = RELEASE_VERSION
    frozen_intent = copy.deepcopy(intent)
    frozen_intent["benchmark_version"] = RELEASE_VERSION
    summary = {
        "benchmark_id": "libero-max-8000",
        "benchmark_version": RELEASE_VERSION,
        "display_name": "LIBERO-MAX-8000",
        "status": "released",
        "base_pairs": 5600,
        "pro_hard_pairs": 2400,
        "matched_pairs": 8000,
        "rollouts_per_model": 16000,
        "intent_matched_pairs": 96,
        "pro_categories": 10,
        "change_types": 8,
        "pro_preflight_passed": preflight["passed"],
        "pro_source_revision": catalog["source_revision"],
    }
    files = {
        "libero_max_8000.json": _json_bytes(frozen_combined),
        "libero_max_pro_hard_2400.json": _json_bytes(frozen_pro),
        "pro_task_catalog.json": _json_bytes(catalog),
        "pro_physical_preflight.json": _json_bytes(preflight),
        "intent_96.json": _json_bytes(frozen_intent),
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
