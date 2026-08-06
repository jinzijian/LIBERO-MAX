#!/usr/bin/env python3
"""Compose one official paired run from disjoint completed result roots."""

import argparse
import json
import os
from pathlib import Path

from libero_max.manifest import load_manifest


def _is_terminal_case(case_dir: Path) -> bool:
    if (case_dir / "paired_summary.json").exists():
        return True
    return all(
        (case_dir / arm / "trace.jsonl").exists()
        for arm in ("control", "intervention")
    )


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
        source = next((path for path in sources if _is_terminal_case(path)), None)
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
    print(json.dumps({"planned": len(manifest["cases"]), "linked": linked, "missing": missing}, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
