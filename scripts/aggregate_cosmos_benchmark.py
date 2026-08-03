#!/usr/bin/env python3
"""Validate full manifest coverage and aggregate paired Cosmos outcomes."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from libero_max.manifest import load_manifest
from libero_max.results import summarize_results


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_case(case: Dict[str, Any], summary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not summary.get("matched"):
        errors.append("paired summary is not matched")
    control = summary.get("control", {})
    intervention = summary.get("intervention", {})
    expected = {
        "scenario_id": case["scenario"]["scenario_id"],
        "scenario_seed": case["scenario"]["seed"],
        "task_suite_name": case["task_suite_name"],
        "original_task_index": case["task_index"],
        "init_state_index": case["init_state_index"],
        "policy_seed": case["policy_seed"],
    }
    for arm_name, arm in (("control", control), ("intervention", intervention)):
        for field, value in expected.items():
            if arm.get(field) != value:
                errors.append(
                    "%s.%s expected %r, found %r"
                    % (arm_name, field, value, arm.get(field))
                )
    if control.get("intervention_event_count") != 0:
        errors.append("control arm contains intervention events")
    if intervention.get("intervention_event_count") != 1:
        errors.append("intervention arm must contain exactly one event")
    if not summary.get("pre_change_action_chunks_match"):
        errors.append("pre-change action chunks are not exactly matched")
    if summary.get("post_event_action_chunk_mad") is None:
        errors.append("post-event action response is missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.root / "manifest.json"
    manifest = load_manifest(manifest_path)
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: Dict[str, List[str]] = {}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        summary_path = args.root / "cases" / case_id / "paired_summary.json"
        if not summary_path.exists():
            missing.append(case_id)
            continue
        try:
            paired = _load_json(summary_path)
            errors = _validate_case(case, paired)
        except (OSError, json.JSONDecodeError) as exc:
            errors = ["failed to load summary: %s" % exc]
            paired = {}
        if errors:
            invalid[case_id] = errors
            continue
        event = paired["intervention"]["intervention_events"][0]
        records.append(
            {
                "pair_id": case_id,
                "scenario_id": case["scenario"]["scenario_id"],
                "seed": case["scenario"]["seed"],
                "change_family": case["scenario"]["change_family"],
                "expected_response_mode": case["scenario"]["expected_response_mode"],
                "control_correct": bool(paired["control_success"]),
                "intervention_correct": bool(paired["intervention_success"]),
                "adaptation_latency_steps": None,
                "safety_violations": None,
                "severity": case["scenario"]["severity"],
                "timing_bucket": case["timing_bucket"],
                "task_suite_name": case["task_suite_name"],
                "task_index": case["task_index"],
                "init_state_index": case["init_state_index"],
                "policy_seed": case["policy_seed"],
                "scoring_mode": "libero_goal_completion",
                "intervention_event_step": event["cosmos_query_boundary_step"],
                "mean_absolute_raw_pixel_delta": event.get(
                    "mean_absolute_raw_pixel_delta"
                ),
                "post_event_action_chunk_mad": paired[
                    "post_event_action_chunk_mad"
                ],
            }
        )

    results_path = args.root / "paired_results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    metrics = summarize_results(records) if records else None
    coverage = {
        "planned": len(manifest["cases"]),
        "completed": len(records),
        "missing": sorted(missing),
        "invalid": invalid,
        "complete": not missing and not invalid and len(records) == len(manifest["cases"]),
    }
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "protocol": manifest["protocol"],
        "coverage": coverage,
        "metrics": None if metrics is None else {
            "overall": metrics["overall"],
            "by_change_family": metrics["by_change_family"],
            "by_severity": metrics.get("by_severity", {}),
            "by_timing_bucket": metrics.get("by_timing_bucket", {}),
            "by_response_mode": metrics.get("by_response_mode", {}),
        },
        "measurement_notes": {
            "adaptation_latency": "not measured by the current Cosmos action-only adapter",
            "safety_violations": "not measured in physical pilot v0.1",
        },
    }
    output = args.root / "benchmark_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if coverage["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
