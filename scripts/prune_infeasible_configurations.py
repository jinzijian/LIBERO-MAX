#!/usr/bin/env python3
"""Prune failed task/change/draw configurations from Core and Full."""

import argparse
import json
from pathlib import Path

from libero_max.pruning import prune_infeasible_configurations


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-core", type=Path, required=True)
    parser.add_argument("--output-full", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    core, full, report = prune_infeasible_configurations(
        _load(args.core), _load(args.full), _load(args.preflight)
    )
    _write(args.output_core, core)
    _write(args.output_full, full)
    _write(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
