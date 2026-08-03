#!/usr/bin/env python3
"""Verify and summarize one matched Cosmos control/intervention smoke."""

import argparse
import json
from pathlib import Path


MATCH_FIELDS = (
    "scenario_id",
    "task_suite_name",
    "original_task_index",
    "init_state_index",
    "task_description",
    "episode_index",
    "policy_seed",
    "init_state_sha256",
    "query_interval",
    "max_policy_steps",
)


def load_one(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("%s must contain exactly one row, found %d" % (path, len(rows)))
    return rows[0]


def _query_by_step(row):
    return {query["policy_step"]: query for query in row.get("policy_queries", [])}


def _action_chunk_mad(left, right):
    left_values = [value for action in left for value in action]
    right_values = [value for action in right for value in action]
    if len(left_values) != len(right_values):
        raise ValueError("paired action chunks have different sizes")
    return sum(abs(a - b) for a, b in zip(left_values, right_values)) / len(
        left_values
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    control = load_one(args.root / "control" / "trace.jsonl")
    intervention = load_one(args.root / "intervention" / "trace.jsonl")
    mismatches = {
        field: {"control": control.get(field), "intervention": intervention.get(field)}
        for field in MATCH_FIELDS
        if control.get(field) != intervention.get(field)
    }
    if control["arm"] != "control" or intervention["arm"] != "intervention":
        raise ValueError("arm labels are invalid")
    if control["intervention_event_count"] != 0:
        raise ValueError("control arm unexpectedly contains an intervention")
    if intervention["intervention_event_count"] != 1:
        raise ValueError("intervention arm must contain exactly one event")

    event = intervention["intervention_events"][0]
    event_step = event["cosmos_query_boundary_step"]
    control_queries = _query_by_step(control)
    intervention_queries = _query_by_step(intervention)
    pre_change_steps = sorted(
        set(control_queries) & set(intervention_queries) & set(range(event_step))
    )
    pre_change_action_chunks_match = bool(pre_change_steps) and all(
        control_queries[step]["action_chunk_sha256"]
        == intervention_queries[step]["action_chunk_sha256"]
        for step in pre_change_steps
    )
    if not pre_change_action_chunks_match:
        raise ValueError("control/intervention action chunks differ before change")
    response_steps = sorted(
        step
        for step in set(control_queries) & set(intervention_queries)
        if step >= event_step
    )
    if not response_steps:
        raise ValueError("missing first post-intervention policy query")
    response_step = response_steps[0]
    post_event_action_chunk_mad = _action_chunk_mad(
        control_queries[response_step]["actions"],
        intervention_queries[response_step]["actions"],
    )
    control_success = bool(control["success"])
    intervention_success = bool(intervention["success"])
    if control_success and intervention_success:
        paired_outcome = "preserved_capability"
    elif not control_success and intervention_success:
        paired_outcome = "intervention_side_gain"
    elif control_success and not intervention_success:
        paired_outcome = "regression_under_change"
    else:
        paired_outcome = "persistent_failure"
    summary = {
        "matched": not mismatches,
        "mismatches": mismatches,
        "control_success": control_success,
        "intervention_success": intervention_success,
        "paired_outcome": paired_outcome,
        "intervention_event_count": intervention["intervention_event_count"],
        "intervention_policy_step": event_step,
        "mean_absolute_raw_pixel_delta": event["mean_absolute_raw_pixel_delta"],
        "pre_change_action_chunks_match": pre_change_action_chunks_match,
        "pre_change_query_steps": pre_change_steps,
        "policy_response_query_step": response_step,
        "open_loop_exposure_steps": response_step - event_step,
        "post_event_action_chunk_mad": post_event_action_chunk_mad,
        "control": control,
        "intervention": intervention,
    }
    output = args.root / "paired_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
