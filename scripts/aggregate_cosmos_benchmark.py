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


def _load_trace(path: Path) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError("%s must contain exactly one row, found %d" % (path, len(rows)))
    return rows[0]


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
    terminal_invalid: Dict[str, List[str]] = {}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        summary_path = args.root / "cases" / case_id / "paired_summary.json"
        if not summary_path.exists():
            intervention_trace = (
                args.root / "cases" / case_id / "intervention" / "trace.jsonl"
            )
            control_trace = args.root / "cases" / case_id / "control" / "trace.jsonl"
            if intervention_trace.exists() and control_trace.exists():
                try:
                    intervention = _load_trace(intervention_trace)
                    control = _load_trace(control_trace)
                    reasons = []
                    if intervention.get("intervention_event_count") == 0:
                        reasons.append("trigger_unreached")
                    if control.get("init_state_sha256") != intervention.get(
                        "init_state_sha256"
                    ):
                        reasons.append("initial_state_mismatch")
                    terminal_invalid[case_id] = reasons or ["paired_summary_missing"]
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    invalid[case_id] = ["failed to load terminal traces: %s" % exc]
            else:
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
        intent = manifest["protocol"]["scoring_track"] == "intent_response"
        response = paired["intervention"].get("response_diagnostics") or {}
        safety_violations = None
        if intent and case["scenario"]["expected_response_mode"] == "stop":
            safety_violations = 0 if response.get("safe_stop") else 1
        records.append(
            {
                "pair_id": case_id,
                "scenario_id": case["scenario"]["scenario_id"],
                "seed": case["scenario"]["seed"],
                "change_family": case["scenario"]["change_family"],
                **(
                    {"change_type": case["scenario"]["change_type"]}
                    if "change_type" in case["scenario"]
                    else {}
                ),
                **(
                    {
                        "intervention_draw_id": case["scenario"]["randomization"][
                            "draw_id"
                        ],
                        "intervention_seed": case["scenario"]["randomization"][
                            "seed"
                        ],
                    }
                    if "randomization" in case["scenario"]
                    else {}
                ),
                "expected_response_mode": case["scenario"]["expected_response_mode"],
                "control_correct": bool(paired["control_success"]),
                "intervention_correct": bool(paired["intervention_success"]),
                "adaptation_latency_steps": (
                    paired["open_loop_exposure_steps"] if intent else None
                ),
                "safety_violations": safety_violations,
                "severity": case["scenario"]["severity"],
                "timing_bucket": case["timing_bucket"],
                "task_suite_name": case["task_suite_name"],
                "task_index": case["task_index"],
                "init_state_index": case["init_state_index"],
                "policy_seed": case["policy_seed"],
                **(
                    {"substrate_category": case["substrate_category"]}
                    if "substrate_category" in case
                    else {}
                ),
                **(
                    {"substrate_difficulty": case["substrate_difficulty"]}
                    if "substrate_difficulty" in case
                    else {}
                ),
                **(
                    {"dynamic_phase": case["dynamic_phase"]}
                    if "dynamic_phase" in case
                    else {}
                ),
                "scoring_mode": (
                    "intent_response" if intent else "libero_goal_completion"
                ),
                "intervention_event_step": event["cosmos_query_boundary_step"],
                "policy_response_query_step": paired[
                    "policy_response_query_step"
                ],
                "open_loop_exposure_steps": paired["open_loop_exposure_steps"],
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
        "terminal_invalid": terminal_invalid,
        "trigger_unreached": sum(
            "trigger_unreached" in reasons for reasons in terminal_invalid.values()
        ),
        "execution_complete": not missing and not invalid,
        "complete": (
            not missing
            and not invalid
            and not terminal_invalid
            and len(records) == len(manifest["cases"])
        ),
    }
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "protocol": manifest["protocol"],
        "coverage": coverage,
        "metrics": None if metrics is None else {
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
        "measurement_notes": {
            "adaptation_latency": (
                "event-to-first-updated-instruction-query steps"
                if manifest["protocol"]["scoring_track"] == "intent_response"
                else "not measured by the physical-completion adapter"
            ),
            "safety_violations": (
                "task cancellation uses the frozen ten-step safe-stop contract"
                if manifest["protocol"]["scoring_track"] == "intent_response"
                else "not measured in the physical-completion track"
            ),
        },
    }
    output = args.root / "benchmark_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if coverage["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
