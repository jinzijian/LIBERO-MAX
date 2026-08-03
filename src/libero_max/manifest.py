"""Strict benchmark manifests for reproducible LIBERO-MAX evaluation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .scenario import validate_scenario


SUPPORTED_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
}
TIMING_BUCKETS = {"early", "middle", "late"}
SCORING_TRACKS = {"physical_completion"}
MANIFEST_FIELDS = {"benchmark_id", "benchmark_version", "protocol", "cases"}
PROTOCOL_FIELDS = {"arms", "query_interval", "scoring_track"}
CASE_FIELDS = {
    "case_id",
    "task_suite_name",
    "task_index",
    "init_state_index",
    "policy_seed",
    "timing_bucket",
    "scenario",
}


class ManifestLoadError(ValueError):
    """Raised when a benchmark manifest cannot be loaded or validated."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _field_errors(data: Dict[str, Any], expected: set, label: str) -> List[str]:
    errors: List[str] = []
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing:
        errors.append("%s missing fields: %s" % (label, ", ".join(missing)))
    if unknown:
        errors.append("%s has unknown fields: %s" % (label, ", ".join(unknown)))
    return errors


def validate_manifest(data: Any) -> List[str]:
    """Validate a manifest and return all actionable errors."""

    if not isinstance(data, dict):
        return ["manifest must be a JSON object"]
    errors = _field_errors(data, MANIFEST_FIELDS, "manifest")
    for field in ("benchmark_id", "benchmark_version"):
        if field in data and not _nonempty_string(data[field]):
            errors.append("%s must be a non-empty string" % field)

    protocol = data.get("protocol")
    query_interval = None
    scoring_track = None
    if not isinstance(protocol, dict):
        errors.append("protocol must be a JSON object")
    else:
        errors.extend(_field_errors(protocol, PROTOCOL_FIELDS, "protocol"))
        arms = protocol.get("arms")
        if arms != ["control", "intervention"]:
            errors.append("protocol.arms must be [control, intervention] in that order")
        query_interval = protocol.get("query_interval")
        if not _is_integer(query_interval) or query_interval < 1:
            errors.append("protocol.query_interval must be an integer >= 1")
        scoring_track = protocol.get("scoring_track")
        if scoring_track not in SCORING_TRACKS:
            errors.append(
                "protocol.scoring_track must be one of: %s"
                % ", ".join(sorted(SCORING_TRACKS))
            )

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a non-empty array")
        return errors

    seen_case_ids = set()
    seen_execution_keys = set()
    for index, case in enumerate(cases):
        label = "case[%d]" % index
        if not isinstance(case, dict):
            errors.append("%s must be a JSON object" % label)
            continue
        errors.extend(_field_errors(case, CASE_FIELDS, label))
        case_id = case.get("case_id")
        if not _nonempty_string(case_id):
            errors.append("%s.case_id must be a non-empty string" % label)
        elif case_id in seen_case_ids:
            errors.append("%s has duplicate case_id: %s" % (label, case_id))
        else:
            seen_case_ids.add(case_id)

        suite = case.get("task_suite_name")
        if suite not in SUPPORTED_SUITES:
            errors.append(
                "%s.task_suite_name must be one of: %s"
                % (label, ", ".join(sorted(SUPPORTED_SUITES)))
            )
        for field in ("task_index", "init_state_index", "policy_seed"):
            value = case.get(field)
            if not _is_integer(value) or value < 0:
                errors.append("%s.%s must be a non-negative integer" % (label, field))
        if case.get("timing_bucket") not in TIMING_BUCKETS:
            errors.append(
                "%s.timing_bucket must be one of: %s"
                % (label, ", ".join(sorted(TIMING_BUCKETS)))
            )

        scenario = case.get("scenario")
        scenario_errors = validate_scenario(scenario)
        errors.extend("%s.scenario: %s" % (label, error) for error in scenario_errors)
        if not scenario_errors and scoring_track == "physical_completion":
            if scenario["change_family"] not in {"OBS", "GEO", "CLUTTER", "OBS-NEW"}:
                errors.append(
                    "%s physical_completion only supports OBS, GEO, CLUTTER, and OBS-NEW"
                    % label
                )
            if scenario["expected_response_mode"] not in {"continue", "replan"}:
                errors.append(
                    "%s physical_completion only supports continue or replan" % label
                )
            trigger = scenario["trigger"]
            if trigger["type"] != "fixed_step":
                errors.append(
                    "%s Cosmos physical pilot requires a fixed_step trigger" % label
                )
            elif _is_integer(query_interval) and trigger["value"] % query_interval:
                errors.append(
                    "%s trigger step must be aligned to protocol.query_interval" % label
                )

        if (
            _nonempty_string(case_id)
            and not scenario_errors
            and suite in SUPPORTED_SUITES
            and all(
                _is_integer(case.get(field)) and case[field] >= 0
                for field in ("task_index", "init_state_index", "policy_seed")
            )
        ):
            execution_key: Tuple[Any, ...] = (
                scenario["scenario_id"],
                scenario["seed"],
                suite,
                case["task_index"],
                case["init_state_index"],
                case["policy_seed"],
            )
            if execution_key in seen_execution_keys:
                errors.append("%s duplicates an existing execution key" % label)
            seen_execution_keys.add(execution_key)
    return errors


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestLoadError("failed to load %s: %s" % (path, exc)) from exc
    errors = validate_manifest(data)
    if errors:
        raise ManifestLoadError("; ".join(errors))
    return data
