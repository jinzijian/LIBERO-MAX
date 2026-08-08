#!/usr/bin/env python3
"""Build the frozen, outcome-independent MAX-PRO model-comparison subset."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from libero_max.manifest import load_manifest, validate_manifest


SELECTOR = "libero-max-pro-model-comparison-v1"


def _cell(case: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        case["substrate_category"],
        case["scenario"]["change_type"],
        int(case["scenario"]["randomization"]["draw_id"]),
    )


def _rank(case: Dict[str, Any]) -> Tuple[str, str]:
    case_id = case["case_id"]
    digest = hashlib.sha256((SELECTOR + ":" + case_id).encode("utf-8")).hexdigest()
    return digest, case_id


def select_cases(
    cases: List[Dict[str, Any]], per_cell: int
) -> List[Dict[str, Any]]:
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[_cell(case)].append(case)
    selected = []
    for cell, rows in sorted(grouped.items()):
        if len(rows) < per_cell:
            raise ValueError("cell %r contains only %d cases" % (cell, len(rows)))
        selected.extend(sorted(rows, key=_rank)[:per_cell])
    return sorted(selected, key=lambda case: case["case_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-cell", type=int, default=5)
    parser.add_argument("--query-interval", type=int, default=16)
    args = parser.parse_args()
    source = load_manifest(args.source)
    selected = select_cases(source["cases"], args.per_cell)
    manifest = {
        "benchmark_id": "libero-max-pro-model-comparison-%d" % len(selected),
        "benchmark_version": source["benchmark_version"],
        "protocol": {
            **source["protocol"],
            "profile": "pro-model-comparison",
            "query_interval": args.query_interval,
            "selection_contract": (
                "%s; %d cases per substrate-category/change-type/draw cell; "
                "SHA-256 rank independent of model outcomes"
                % (SELECTOR, args.per_cell)
            ),
        },
        "cases": selected,
    }
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("generated invalid manifest: %s" % errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cell_counts = defaultdict(int)
    for case in selected:
        cell_counts[_cell(case)] += 1
    print(
        json.dumps(
            {
                "benchmark_id": manifest["benchmark_id"],
                "cases": len(selected),
                "cells": len(cell_counts),
                "per_cell": sorted(set(cell_counts.values())),
                "query_interval": args.query_interval,
                "selector": SELECTOR,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
