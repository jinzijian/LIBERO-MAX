#!/usr/bin/env python3
"""Merge and coverage-audit all LIBERO-MAX physical-preflight shards."""

import argparse
import json
from pathlib import Path

from libero_max.preflight import merge_preflight_reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.reports
    ]
    merged = merge_preflight_reports(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in merged.items() if key != "cases"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if merged["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
