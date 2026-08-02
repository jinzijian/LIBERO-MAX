#!/usr/bin/env python3
"""Verify and summarize one matched Cosmos control/intervention smoke."""

import argparse
import json
from pathlib import Path


MATCH_FIELDS = (
    "scenario_id",
    "task_suite_name",
    "original_task_index",
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
        "intervention_policy_step": event["cosmos_query_boundary_step"],
        "mean_absolute_raw_pixel_delta": event["mean_absolute_raw_pixel_delta"],
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
