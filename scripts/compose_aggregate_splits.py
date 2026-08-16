#!/usr/bin/env python3
"""Compose disjoint, already aggregated MAX splits into one audited run.

This utility is intentionally trace-free.  It combines the compact aggregate
artifacts produced by ``aggregate_cosmos_benchmark.py`` after each source split
has independently passed its full execution-completeness checks.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from libero_max.provenance import sha256_file
from libero_max.results import summarize_results

from aggregate_cosmos_benchmark import summarize_end_to_end_outcomes


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unique_by_pair_id(rows: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in indexed:
            raise ValueError(f"duplicate {label} pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


def _breakdown(
    records: List[Dict[str, Any]], field: str
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        value = record.get(field)
        if value is not None:
            groups.setdefault(str(value), []).append(record)
    result: Dict[str, Dict[str, Any]] = {}
    for value, rows in sorted(groups.items()):
        result[value] = summarize_end_to_end_outcomes(
            len(rows),
            {row["pair_id"]: bool(row["control_correct"]) for row in rows},
            {row["pair_id"]: bool(row["intervention_correct"]) for row in rows},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    expected = {case["case_id"] for case in manifest["cases"]}
    if len(expected) != len(manifest["cases"]):
        raise ValueError("combined manifest contains duplicate case IDs")

    summaries: List[Dict[str, Any]] = []
    end_to_end_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    for root in args.split:
        summary = _load_json(root / "benchmark_summary.json")
        if not summary["coverage"].get("execution_complete", False):
            raise ValueError(f"split is not execution-complete: {root}")
        summaries.append(summary)
        end_to_end_rows.extend(_load_jsonl(root / "end_to_end_results.jsonl"))
        paired_rows.extend(_load_jsonl(root / "paired_results.jsonl"))

    end_to_end = _unique_by_pair_id(end_to_end_rows, "end-to-end")
    paired = _unique_by_pair_id(paired_rows, "response-evaluable")
    observed = set(end_to_end)
    if observed != expected:
        raise ValueError(
            "combined split coverage differs from manifest: "
            f"missing={len(expected - observed)} extra={len(observed - expected)}"
        )
    if not set(paired).issubset(expected):
        raise ValueError("response-evaluable rows contain unknown case IDs")

    ordered_end_to_end = [end_to_end[case["case_id"]] for case in manifest["cases"]]
    ordered_paired = [paired[pair_id] for pair_id in sorted(paired)]
    planned = len(expected)
    control_outcomes = {
        row["pair_id"]: bool(row["control_correct"]) for row in ordered_end_to_end
    }
    intervention_outcomes = {
        row["pair_id"]: bool(row["intervention_correct"])
        for row in ordered_end_to_end
    }
    end_to_end_metrics = summarize_end_to_end_outcomes(
        planned, control_outcomes, intervention_outcomes
    )
    metrics = summarize_results(ordered_paired) if ordered_paired else None

    trigger_unreached = [
        row["pair_id"] for row in ordered_end_to_end if not row["trigger_reached"]
    ]
    response_unreached = [
        row["pair_id"]
        for row in ordered_end_to_end
        if row["trigger_reached"] and not row["response_query_reached"]
    ]
    protocol = dict(summaries[0]["protocol"])
    query_intervals = {summary["protocol"]["query_interval"] for summary in summaries}
    if len(query_intervals) != 1:
        raise ValueError(f"split query intervals differ: {sorted(query_intervals)}")
    protocol["selection_contract"] = manifest["protocol"]["selection_contract"]
    protocol["source_splits"] = [summary["benchmark_id"] for summary in summaries]

    report = {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "model": args.model,
        "protocol": protocol,
        "coverage": {
            "planned": planned,
            "completed": len(ordered_paired),
            "missing": [],
            "invalid": {},
            "terminal_invalid": {
                **{pair_id: ["trigger_unreached"] for pair_id in trigger_unreached},
                **{
                    pair_id: ["response_query_unreached"]
                    for pair_id in response_unreached
                },
            },
            "trigger_unreached": len(trigger_unreached),
            "trigger_reached": sum(
                bool(row["trigger_reached"]) for row in ordered_end_to_end
            ),
            "response_query_unreached": len(response_unreached),
            "response_evaluable": len(ordered_paired),
            "blocking_terminal_invalid": {},
            "execution_complete": True,
            "complete": True,
            "conditional_complete": len(ordered_paired) == planned,
        },
        "end_to_end_metrics": end_to_end_metrics,
        "end_to_end_breakdowns": {
            "by_change_type": _breakdown(ordered_end_to_end, "change_type"),
            "by_substrate_category": _breakdown(
                ordered_end_to_end, "substrate_category"
            ),
            "by_task_suite": _breakdown(ordered_end_to_end, "task_suite_name"),
        },
        "metrics": None
        if metrics is None
        else {
            "overall": metrics["overall"],
            "by_change_family": metrics["by_change_family"],
            "by_change_type": metrics.get("by_change_type", {}),
            "by_intervention_draw": metrics.get("by_intervention_draw", {}),
            "by_severity": metrics.get("by_severity", {}),
            "by_timing_bucket": metrics.get("by_timing_bucket", {}),
            "by_response_mode": metrics.get("by_response_mode", {}),
            "by_substrate_category": metrics.get("by_substrate_category", {}),
            "by_substrate_difficulty": metrics.get("by_substrate_difficulty", {}),
            "by_dynamic_phase": metrics.get("by_dynamic_phase", {}),
        },
        "composition": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "split_roots": [str(root) for root in args.split],
            "split_planned": [summary["coverage"]["planned"] for summary in summaries],
            "pair_ids_unique": True,
            "pair_ids_equal_manifest": True,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(args.output_dir / "end_to_end_results.jsonl", ordered_end_to_end)
    _write_jsonl(args.output_dir / "paired_results.jsonl", ordered_paired)
    print(
        f"{args.model}: {planned} unique pairs, {2 * planned} episodes, "
        f"response-evaluable={len(ordered_paired)}"
    )


if __name__ == "__main__":
    main()
