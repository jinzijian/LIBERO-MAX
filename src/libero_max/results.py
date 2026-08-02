"""Paired outcome loading and reporting for LIBERO-MAX."""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .scenario import CHANGE_FAMILIES, RESPONSE_MODES, scenario_key


RESULT_FIELDS = {
    "pair_id",
    "scenario_id",
    "seed",
    "change_family",
    "expected_response_mode",
    "control_correct",
    "intervention_correct",
    "adaptation_latency_steps",
    "safety_violations",
}


class ResultLoadError(ValueError):
    """Raised when paired result records are malformed."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_result(record: Any) -> List[str]:
    if not isinstance(record, dict):
        return ["result must be a JSON object"]

    errors: List[str] = []
    missing = sorted(RESULT_FIELDS - set(record))
    unknown = sorted(set(record) - RESULT_FIELDS)
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if unknown:
        errors.append("unknown fields: " + ", ".join(unknown))

    for field in ("pair_id", "scenario_id"):
        if field in record and (
            not isinstance(record[field], str) or not record[field].strip()
        ):
            errors.append("%s must be a non-empty string" % field)

    if "seed" in record and (
        not _is_integer(record["seed"]) or record["seed"] < 0
    ):
        errors.append("seed must be a non-negative integer")

    if "change_family" in record and record["change_family"] not in CHANGE_FAMILIES:
        errors.append("invalid change_family")
    if (
        "expected_response_mode" in record
        and record["expected_response_mode"] not in RESPONSE_MODES
    ):
        errors.append("invalid expected_response_mode")

    for field in ("control_correct", "intervention_correct"):
        if field in record and not isinstance(record[field], bool):
            errors.append("%s must be a boolean" % field)

    latency = record.get("adaptation_latency_steps")
    if latency is not None and (not _is_integer(latency) or latency < 0):
        errors.append("adaptation_latency_steps must be null or a non-negative integer")

    violations = record.get("safety_violations")
    if violations is not None and (
        not _is_integer(violations) or violations < 0
    ):
        errors.append("safety_violations must be a non-negative integer")

    return errors


def load_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResultLoadError(
                        "%s:%d: invalid JSON: %s" % (path, line_number, exc)
                    ) from exc
                errors = validate_result(record)
                if errors:
                    raise ResultLoadError(
                        "%s:%d: %s" % (path, line_number, "; ".join(errors))
                    )
                records.append(record)
    except OSError as exc:
        raise ResultLoadError("failed to load %s: %s" % (path, exc)) from exc
    if not records:
        raise ResultLoadError("no result records found in %s" % path)
    return records


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _metric_block(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    preserved = sum(
        record["control_correct"] and record["intervention_correct"]
        for record in records
    )
    gains = sum(
        not record["control_correct"] and record["intervention_correct"]
        for record in records
    )
    regressions = sum(
        record["control_correct"] and not record["intervention_correct"]
        for record in records
    )
    persistent = total - preserved - gains - regressions
    control_correct = preserved + regressions
    intervention_correct = preserved + gains
    latencies = [
        record["adaptation_latency_steps"]
        for record in records
        if record["intervention_correct"]
        and record["adaptation_latency_steps"] is not None
    ]
    safety_failures = sum(record["safety_violations"] > 0 for record in records)

    return {
        "episodes": total,
        "outcome_table": {
            "preserved_capability": preserved,
            "intervention_side_gain": gains,
            "regression_under_change": regressions,
            "persistent_failure": persistent,
        },
        "scenario_aware_outcome_accuracy": _ratio(intervention_correct, total),
        "control_accuracy": _ratio(control_correct, total),
        "paired_robustness_delta": (
            None
            if total == 0
            else round((intervention_correct - control_correct) / total, 6)
        ),
        "regression_rate": _ratio(regressions, control_correct),
        "safety_violation_rate": _ratio(safety_failures, total),
        "adaptation_latency_steps": {
            "count": len(latencies),
            "mean": None if not latencies else round(statistics.mean(latencies), 6),
            "median": None if not latencies else statistics.median(latencies),
        },
    }


def summarize_results(
    records: Sequence[Dict[str, Any]],
    expected_scenarios: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarize unique matched-pair records with explicit coverage."""

    pair_ids = set()
    result_keys = set()
    for index, record in enumerate(records):
        errors = validate_result(record)
        if errors:
            raise ResultLoadError("result[%d]: %s" % (index, "; ".join(errors)))
        if record["pair_id"] in pair_ids:
            raise ResultLoadError("duplicate pair_id: %s" % record["pair_id"])
        key = scenario_key(record)
        if key in result_keys:
            raise ResultLoadError(
                "duplicate result for (scenario_id, seed): %s, %d" % key
            )
        pair_ids.add(record["pair_id"])
        result_keys.add(key)

    expected_keys = (
        None
        if expected_scenarios is None
        else {scenario_key(scenario) for scenario in expected_scenarios}
    )
    if expected_keys is None:
        selected = list(records)
        planned = len(records)
        missing: List[Tuple[str, int]] = []
        unexpected: List[Tuple[str, int]] = []
    else:
        selected = [record for record in records if scenario_key(record) in expected_keys]
        planned = len(expected_keys)
        missing = sorted(expected_keys - result_keys)
        unexpected = sorted(result_keys - expected_keys)

    overall = _metric_block(selected)
    families: Dict[str, Any] = {}
    for family in sorted({record["change_family"] for record in selected}):
        families[family] = _metric_block(
            [record for record in selected if record["change_family"] == family]
        )

    return {
        "coverage": {
            "planned": planned,
            "completed": len(selected),
            "missing": ["%s:%d" % item for item in missing],
            "unexpected": ["%s:%d" % item for item in unexpected],
            "complete": not missing,
        },
        "overall": overall,
        "by_change_family": families,
    }
