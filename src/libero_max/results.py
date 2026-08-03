"""Paired outcome loading and reporting for LIBERO-MAX."""

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .scenario import CHANGE_FAMILIES, CHANGE_TYPES, RESPONSE_MODES, scenario_key


REQUIRED_RESULT_FIELDS = {
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
OPTIONAL_RESULT_FIELDS = {
    "change_type",
    "intervention_draw_id",
    "intervention_seed",
    "severity",
    "timing_bucket",
    "task_suite_name",
    "task_index",
    "init_state_index",
    "policy_seed",
    "scoring_mode",
    "intervention_event_step",
    "policy_response_query_step",
    "open_loop_exposure_steps",
    "mean_absolute_raw_pixel_delta",
    "post_event_action_chunk_mad",
}
RESULT_FIELDS = REQUIRED_RESULT_FIELDS | OPTIONAL_RESULT_FIELDS


class ResultLoadError(ValueError):
    """Raised when paired result records are malformed."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_result(record: Any) -> List[str]:
    if not isinstance(record, dict):
        return ["result must be a JSON object"]

    errors: List[str] = []
    missing = sorted(REQUIRED_RESULT_FIELDS - set(record))
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
    if "change_type" in record and record["change_type"] not in CHANGE_TYPES:
        errors.append("invalid change_type")
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

    if "severity" in record and record["severity"] not in {"low", "medium", "high"}:
        errors.append("invalid severity")
    if "timing_bucket" in record and record["timing_bucket"] not in {
        "early",
        "middle",
        "late",
    }:
        errors.append("invalid timing_bucket")
    for field in (
        "task_index",
        "init_state_index",
        "policy_seed",
        "intervention_draw_id",
        "intervention_seed",
        "intervention_event_step",
        "policy_response_query_step",
        "open_loop_exposure_steps",
    ):
        if field in record and (
            not _is_integer(record[field]) or record[field] < 0
        ):
            errors.append("%s must be a non-negative integer" % field)
    for field in ("task_suite_name", "scoring_mode"):
        if field in record and (
            not isinstance(record[field], str) or not record[field].strip()
        ):
            errors.append("%s must be a non-empty string" % field)
    for field in ("mean_absolute_raw_pixel_delta", "post_event_action_chunk_mad"):
        value = record.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            errors.append("%s must be null or non-negative" % field)

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


def result_execution_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    dimensions = (
        "task_suite_name",
        "task_index",
        "init_state_index",
        "policy_seed",
    )
    if all(field in record for field in dimensions):
        return (
            record["scenario_id"],
            record["seed"],
            *(record[field] for field in dimensions),
        )
    return scenario_key(record)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _wilson_interval(successes: int, total: int) -> Optional[List[float]]:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - radius), 6), round(min(1.0, center + radius), 6)]


def _mcnemar_exact(gains: int, regressions: int) -> Optional[float]:
    discordant = gains + regressions
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, k) for k in range(min(gains, regressions) + 1)
    ) / (2**discordant)
    return round(min(1.0, 2 * tail), 10)


def _paired_bootstrap_delta(
    records: Sequence[Dict[str, Any]], samples: int = 2000
) -> Optional[List[float]]:
    if not records:
        return None
    differences = [
        int(record["intervention_correct"]) - int(record["control_correct"])
        for record in records
    ]
    rng = random.Random(0)
    estimates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    )
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return [round(low, 6), round(high, 6)]


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
    measured_safety = [
        record["safety_violations"]
        for record in records
        if record["safety_violations"] is not None
    ]
    safety_failures = sum(value > 0 for value in measured_safety)

    return {
        "episodes": total,
        "outcome_table": {
            "preserved_capability": preserved,
            "intervention_side_gain": gains,
            "regression_under_change": regressions,
            "persistent_failure": persistent,
        },
        "scenario_aware_outcome_accuracy": _ratio(intervention_correct, total),
        "scenario_aware_outcome_accuracy_95ci_wilson": _wilson_interval(
            intervention_correct, total
        ),
        "control_accuracy": _ratio(control_correct, total),
        "control_accuracy_95ci_wilson": _wilson_interval(control_correct, total),
        "paired_robustness_delta": (
            None
            if total == 0
            else round((intervention_correct - control_correct) / total, 6)
        ),
        "regression_rate": _ratio(regressions, control_correct),
        "paired_robustness_delta_95ci_bootstrap": _paired_bootstrap_delta(records),
        "mcnemar_exact_two_sided_p": _mcnemar_exact(gains, regressions),
        "safety_violation_rate": _ratio(safety_failures, len(measured_safety)),
        "safety_measurement_coverage": {
            "measured": len(measured_safety),
            "total": total,
        },
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
        key = result_execution_key(record)
        if key in result_keys:
            raise ResultLoadError(
                "duplicate result execution key: %r" % (key,)
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

    def grouped(field: str) -> Dict[str, Any]:
        values = sorted({record[field] for record in selected if field in record})
        return {
            str(value): _metric_block(
                [record for record in selected if record.get(field) == value]
            )
            for value in values
        }

    return {
        "coverage": {
            "planned": planned,
            "completed": len(selected),
            "missing": ["%s:%d" % item for item in missing],
            "unexpected": ["%s:%d" % item for item in unexpected],
            "complete": not missing and not unexpected,
        },
        "overall": overall,
        "by_change_family": families,
        "by_change_type": grouped("change_type"),
        "by_intervention_draw": grouped("intervention_draw_id"),
        "by_severity": grouped("severity"),
        "by_timing_bucket": grouped("timing_bucket"),
        "by_response_mode": grouped("expected_response_mode"),
    }
