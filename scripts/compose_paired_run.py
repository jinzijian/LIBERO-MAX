#!/usr/bin/env python3
"""Compose one official paired run from disjoint completed result roots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from libero_max.manifest import load_manifest
from libero_max.provenance import sha256_file, source_run_configs, write_run_config


def _scenario_sha256(scenario) -> str:
    payload = json.dumps(scenario, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _is_terminal_case(case_dir: Path, scenario) -> bool:
    scenario_path = case_dir / "scenario.json"
    if not scenario_path.exists():
        return False
    try:
        recorded = json.loads(scenario_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if _scenario_sha256(recorded) != _scenario_sha256(scenario):
        return False
    if (case_dir / "paired_summary.json").exists():
        return True
    for arm in ("control", "intervention"):
        trace_path = case_dir / arm / "trace.jsonl"
        if not trace_path.exists():
            return False
        try:
            rows = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            return False
        if len(rows) != 1:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source_roots", type=Path, nargs="+")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cases").mkdir(exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    missing = []
    linked = 0
    for case in manifest["cases"]:
        case_id = case["case_id"]
        sources = [root / "cases" / case_id for root in args.source_roots]
        source = next(
            (path for path in sources if _is_terminal_case(path, case["scenario"])),
            None,
        )
        if source is None:
            missing.append(case_id)
            continue
        destination = args.output_root / "cases" / case_id
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                linked += 1
                continue
            destination.unlink()
        elif destination.exists():
            raise ValueError("refusing to replace non-symlink %s" % destination)
        os.symlink(source.resolve(), destination, target_is_directory=True)
        linked += 1
    materialized_manifest = args.output_root / "manifest.json"
    write_run_config(
        args.output_root / "run_config.json",
        {
            "schema_version": 1,
            "run_type": "composed_paired_run",
            "created_by": "scripts/compose_paired_run.py",
            "manifest": {
                "source_path": str(args.manifest.resolve()),
                "source_sha256": sha256_file(args.manifest),
                "materialized_path": str(materialized_manifest.resolve()),
                "materialized_sha256": sha256_file(materialized_manifest),
                "benchmark_id": manifest["benchmark_id"],
                "benchmark_version": manifest["benchmark_version"],
                "planned_cases": len(manifest["cases"]),
            },
            "composition": {
                "planned": len(manifest["cases"]),
                "linked": linked,
                "missing": missing,
                "source_runs": source_run_configs(args.source_roots),
            },
        },
    )
    print(
        json.dumps(
            {"planned": len(manifest["cases"]), "linked": linked, "missing": missing},
            indent=2,
        )
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
