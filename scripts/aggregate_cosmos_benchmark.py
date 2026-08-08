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
        raise ValueError(
            "%s must contain exactly one row, found %d" % (path, len(rows))
        )
    return rows[0]


def _capture_outcome(outcomes: Dict[str, bool], case_id: str, value: Any) -> None:
    if isinstance(value, bool):
        outcomes[case_id] = value


def _outcome_rate(outcomes: Dict[str, bool], planned: int) -> Dict[str, Any]:
    measured = len(outcomes)
    successes = sum(outcomes.values())
    return {
        "planned": planned,
        "measured": measured,
        "missing": planned - measured,
        "successes": successes,
        "failures": measured - successes,
        "accuracy_on_measured": successes / measured if measured else None,
        "accuracy_on_planned": successes / planned if measured == planned else None,
    }


def summarize_end_to_end_outcomes(
    planned: int,
    control_outcomes: Dict[str, bool],
    intervention_outcomes: Dict[str, bool],
) -> Dict[str, Any]:
    """Summarize all terminal outcomes, including trigger-unreached cases.

    Trigger-conditioned adaptation metrics remain useful diagnostics, but they
    must not silently remove policies that fail before reaching a trigger. The
    all-case rates are only published when every planned arm has an outcome;
    infrastructure gaps remain missing rather than being charged to a model.
    """

    paired_ids = sorted(set(control_outcomes) & set(intervention_outcomes))
    outcome_table = {
        "preserved_capability": 0,
        "intervention_side_gain": 0,
        "regression_under_change": 0,
        "persistent_failure": 0,
    }
    for case_id in paired_ids:
        control = control_outcomes[case_id]
        intervention = intervention_outcomes[case_id]
        if control and intervention:
            outcome_table["preserved_capability"] += 1
        elif not control and intervention:
            outcome_table["intervention_side_gain"] += 1
        elif control and not intervention:
            outcome_table["regression_under_change"] += 1
        else:
            outcome_table["persistent_failure"] += 1

    control = _outcome_rate(control_outcomes, planned)
    intervention = _outcome_rate(intervention_outcomes, planned)
    paired_complete = len(paired_ids) == planned
    return {
        "control": control,
        "intervention": intervention,
        "paired_measured": len(paired_ids),
        "paired_missing": planned - len(paired_ids),
        "outcome_table": outcome_table,
        "paired_robustness_delta_on_measured": (
            (
                sum(intervention_outcomes[case_id] for case_id in paired_ids)
                - sum(control_outcomes[case_id] for case_id in paired_ids)
            )
            / len(paired_ids)
            if paired_ids
            else None
        ),
        "paired_robustness_delta_on_planned": (
            intervention["accuracy_on_planned"] - control["accuracy_on_planned"]
            if paired_complete
            else None
        ),
        "complete": paired_complete,
    }


def summarize_end_to_end_breakdown(
    cases: List[Dict[str, Any]],
    control_outcomes: Dict[str, bool],
    intervention_outcomes: Dict[str, bool],
    field: str,
) -> Dict[str, Dict[str, Any]]:
    """Return full-denominator paired outcomes for one manifest field."""

    groups: Dict[str, List[str]] = {}
    for case in cases:
        if field.startswith("scenario."):
            value = case["scenario"].get(field.split(".", 1)[1])
        else:
            value = case.get(field)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(case["case_id"])
    result = {}
    for value, case_ids in sorted(groups.items()):
        selected = set(case_ids)
        result[value] = summarize_end_to_end_outcomes(
            len(case_ids),
            {
                case_id: outcome
                for case_id, outcome in control_outcomes.items()
                if case_id in selected
            },
            {
                case_id: outcome
                for case_id, outcome in intervention_outcomes.items()
                if case_id in selected
            },
        )
    return result


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
    parser.add_argument(
        "--case-root",
        action="append",
        type=Path,
        default=[],
        help="additional run root containing case traces",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write derived results here while reading case traces from ROOT",
    )
    args = parser.parse_args()
    manifest_path = args.manifest or args.root / "manifest.json"
    manifest = load_manifest(manifest_path)
    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    case_roots = [args.root, *args.case_root]
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    invalid: Dict[str, List[str]] = {}
    terminal_invalid: Dict[str, List[str]] = {}
    control_outcomes: Dict[str, bool] = {}
    intervention_outcomes: Dict[str, bool] = {}
    for case in manifest["cases"]:
        case_id = case["case_id"]
        matching_case_dirs = [
            root / "cases" / case_id
            for root in case_roots
            if (root / "cases" / case_id).exists()
        ]
        if len(matching_case_dirs) > 1:
            invalid[case_id] = ["case exists in multiple case roots"]
            continue
        if not matching_case_dirs:
            missing.append(case_id)
            continue
        case_dir = matching_case_dirs[0]
        summary_path = case_dir / "paired_summary.json"
        if not summary_path.exists():
            intervention_trace = case_dir / "intervention" / "trace.jsonl"
            control_trace = case_dir / "control" / "trace.jsonl"
            if intervention_trace.exists() and control_trace.exists():
                try:
                    intervention = _load_trace(intervention_trace)
                    control = _load_trace(control_trace)
                    _capture_outcome(control_outcomes, case_id, control.get("success"))
                    _capture_outcome(
                        intervention_outcomes,
                        case_id,
                        intervention.get("success"),
                    )
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
            _capture_outcome(control_outcomes, case_id, paired.get("control_success"))
            _capture_outcome(
                intervention_outcomes,
                case_id,
                paired.get("intervention_success"),
            )
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
                        "intervention_seed": case["scenario"]["randomization"]["seed"],
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
                "policy_response_query_step": paired["policy_response_query_step"],
                "open_loop_exposure_steps": paired["open_loop_exposure_steps"],
                "mean_absolute_raw_pixel_delta": event.get(
                    "mean_absolute_raw_pixel_delta"
                ),
                "post_event_action_chunk_mad": paired["post_event_action_chunk_mad"],
            }
        )

    results_path = output_dir / "paired_results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    end_to_end_records = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        if case_id not in control_outcomes or case_id not in intervention_outcomes:
            continue
        reasons = terminal_invalid.get(case_id, [])
        end_to_end_records.append(
            {
                "pair_id": case_id,
                "control_correct": control_outcomes[case_id],
                "intervention_correct": intervention_outcomes[case_id],
                "trigger_reached": "trigger_unreached" not in reasons,
                "terminal_status": (
                    "trigger_unreached"
                    if "trigger_unreached" in reasons
                    else "triggered"
                ),
                "change_type": case["scenario"].get("change_type"),
                "change_family": case["scenario"]["change_family"],
                "severity": case["scenario"]["severity"],
                "intervention_draw_id": case["scenario"]
                .get("randomization", {})
                .get("draw_id"),
                "task_suite_name": case["task_suite_name"],
                "substrate_category": case.get("substrate_category"),
                "timing_bucket": case.get("timing_bucket"),
            }
        )
    (output_dir / "end_to_end_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in end_to_end_records),
        encoding="utf-8",
    )
    metrics = summarize_results(records) if records else None
    end_to_end_metrics = summarize_end_to_end_outcomes(
        len(manifest["cases"]), control_outcomes, intervention_outcomes
    )
    blocking_terminal_invalid = {
        case_id: reasons
        for case_id, reasons in terminal_invalid.items()
        if set(reasons) != {"trigger_unreached"}
    }
    execution_complete = (
        not missing
        and not invalid
        and not blocking_terminal_invalid
        and end_to_end_metrics["complete"]
    )
    coverage = {
        "planned": len(manifest["cases"]),
        "completed": len(records),
        "missing": sorted(missing),
        "invalid": invalid,
        "terminal_invalid": terminal_invalid,
        "trigger_unreached": sum(
            "trigger_unreached" in reasons for reasons in terminal_invalid.values()
        ),
        "retained_untriggered": sorted(
            case_id
            for case_id, reasons in terminal_invalid.items()
            if set(reasons) == {"trigger_unreached"}
        ),
        "blocking_terminal_invalid": blocking_terminal_invalid,
        "execution_complete": execution_complete,
        "complete": execution_complete,
        "conditional_complete": (
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
        "end_to_end_metrics": end_to_end_metrics,
        "end_to_end_breakdowns": {
            "by_change_type": summarize_end_to_end_breakdown(
                manifest["cases"],
                control_outcomes,
                intervention_outcomes,
                "scenario.change_type",
            ),
            "by_substrate_category": summarize_end_to_end_breakdown(
                manifest["cases"],
                control_outcomes,
                intervention_outcomes,
                "substrate_category",
            ),
            "by_task_suite": summarize_end_to_end_breakdown(
                manifest["cases"],
                control_outcomes,
                intervention_outcomes,
                "task_suite_name",
            ),
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
        "measurement_notes": {
            "end_to_end_scoring": (
                "all %s frozen cases remain in the denominator; "
                "trigger-unreached episodes are retained as pre-intervention "
                "policy failures, while infrastructure errors remain missing"
                % len(manifest["cases"])
            ),
            "conditional_adaptation": (
                "metrics.* is conditioned on a valid intervention event and "
                "must be reported with trigger coverage"
            ),
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
    output = output_dir / "benchmark_summary.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if coverage["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
