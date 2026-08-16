#!/usr/bin/env python3
"""Select only infrastructure-invalid cases for a deterministic repair run."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from libero_max.manifest import load_manifest


def repair_case_ids(summary: Dict[str, Any]) -> List[str]:
    coverage = summary["coverage"]
    case_ids = set(coverage.get("missing", []))
    case_ids.update(coverage.get("invalid", {}))
    case_ids.update(coverage.get("blocking_terminal_invalid", {}))
    return sorted(case_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    selected_ids = repair_case_ids(summary)
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError("summary contains cases outside manifest: %s" % unknown)

    repair = dict(manifest)
    repair["benchmark_id"] = "%s-infra-repair" % manifest["benchmark_id"]
    repair["cases"] = [by_id[case_id] for case_id in selected_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(repair, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_benchmark_id": manifest["benchmark_id"],
                "repair_cases": len(selected_ids),
                "case_ids": selected_ids,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
