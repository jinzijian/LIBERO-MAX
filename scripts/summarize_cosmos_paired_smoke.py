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


def classify_persistent_pair(control, intervention):
    """Return a paired summary or a terminal trigger-unreached record.

    Full benchmark runs retain trigger-unreached pairs in the end-to-end
    denominator. They intentionally have no paired response summary because no
    intervention event exists; the aggregate script derives their outcomes
    from the two terminal traces.
    """
    event_count = intervention.get("intervention_event_count")
    if event_count != 0:
        return summarize_pair(control, intervention), None

    mismatches = {
        field: {"control": control.get(field), "intervention": intervention.get(field)}
        for field in MATCH_FIELDS
        if control.get(field) != intervention.get(field)
    }
    if control.get("arm") != "control" or intervention.get("arm") != "intervention":
        raise ValueError("arm labels are invalid")
    if control.get("intervention_event_count") != 0:
        raise ValueError("control arm unexpectedly contains an intervention")
    if mismatches:
        raise ValueError("control/intervention metadata differ before trigger")

    control_queries = _query_by_step(control)
    intervention_queries = _query_by_step(intervention)
    common_steps = sorted(set(control_queries) & set(intervention_queries))
    action_chunks_match = bool(common_steps) and all(
        control_queries[step]["action_chunk_sha256"]
        == intervention_queries[step]["action_chunk_sha256"]
        for step in common_steps
    )
    if not action_chunks_match:
        raise ValueError("control/intervention action chunks differ before trigger")

    terminal = {
        "terminal_status": "trigger_unreached",
        "matched": True,
        "mismatches": {},
        "control_success": bool(control.get("success")),
        "intervention_success": bool(intervention.get("success")),
        "intervention_event_count": 0,
        "pre_change_action_chunks_match": True,
        "pre_change_query_steps": common_steps,
        "response_query_reached": False,
    }
    return None, terminal


def summarize_pair(control, intervention):
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
    input_digest_fields = ("policy_image_sha256", "sim_state_sha256")
    pre_change_policy_inputs_measured = bool(pre_change_steps) and all(
        all(field in control_queries[step] and field in intervention_queries[step]
            for field in input_digest_fields)
        for step in pre_change_steps
    )
    def policy_inputs_match(step):
        digests_match = all(
            control_queries[step][field] == intervention_queries[step][field]
            for field in input_digest_fields
        )
        paired_qa = intervention_queries[step].get("paired_policy_input_qa", {})
        return digests_match or (
            paired_qa.get("status") == "passed"
            and paired_qa.get("sim_state_exact") is True
        )

    pre_change_policy_inputs_match = (
        None
        if not pre_change_policy_inputs_measured
        else all(policy_inputs_match(step) for step in pre_change_steps)
    )
    if pre_change_policy_inputs_match is False:
        raise ValueError("control/intervention policy inputs differ before change")
    response_steps = sorted(
        step
        for step in set(control_queries) & set(intervention_queries)
        if step >= event_step
    )
    response_query_reached = bool(response_steps)
    response_step = response_steps[0] if response_steps else None
    post_event_action_chunk_mad = (
        _action_chunk_mad(
            control_queries[response_step]["actions"],
            intervention_queries[response_step]["actions"],
        )
        if response_step is not None
        else None
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
        "pre_change_policy_inputs_measured": pre_change_policy_inputs_measured,
        "pre_change_policy_inputs_match": pre_change_policy_inputs_match,
        "pre_change_query_steps": pre_change_steps,
        "response_query_reached": response_query_reached,
        "policy_response_query_step": response_step,
        "open_loop_exposure_steps": (
            response_step - event_step if response_step is not None else None
        ),
        "post_event_action_chunk_mad": post_event_action_chunk_mad,
        "control": control,
        "intervention": intervention,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    control = load_one(args.root / "control" / "trace.jsonl")
    intervention = load_one(args.root / "intervention" / "trace.jsonl")
    summary = summarize_pair(control, intervention)
    output = args.root / "paired_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
