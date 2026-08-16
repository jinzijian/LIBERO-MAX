"""Dependency-free helpers for deterministic physical-preflight selection."""

from typing import Any, Dict, Iterable, List, Sequence, Tuple


class PreflightSelectionError(ValueError):
    """Raised when a preflight shard or filter is invalid."""


def changed_entities(change: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return ``(entity, support)`` pairs affected by a physical change."""

    operation = change.get("operation")
    if operation in {"move_object", "insert_obstacle"}:
        entity, support = change.get("object"), change.get("support_entity")
        return [(entity, support)] if entity and support else []
    if operation == "insert_distractors":
        result = []
        for placement in change.get("placements", []):
            entity = placement.get("object")
            support = placement.get("support_entity")
            if entity and support:
                result.append((entity, support))
        return result
    return []


def settle_metrics(
    immediate_positions: Dict[str, Sequence[float]],
    settled_positions: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    """Summarize post-insertion drift without importing NumPy."""

    max_displacement = 0.0
    max_vertical_drop = 0.0
    for entity, before in immediate_positions.items():
        after = settled_positions[entity]
        displacement = sum(
            (float(after[index]) - float(before[index])) ** 2
            for index in range(3)
        ) ** 0.5
        vertical_drop = max(0.0, float(before[2]) - float(after[2]))
        max_displacement = max(max_displacement, displacement)
        max_vertical_drop = max(max_vertical_drop, vertical_drop)
    return {
        "max_settle_displacement_m": max_displacement,
        "max_vertical_drop_m": max_vertical_drop,
    }


def merge_preflight_reports(
    reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge a complete deterministic shard set and audit its coverage."""

    if not reports:
        raise PreflightSelectionError("at least one preflight report is required")
    benchmark_ids = {report.get("benchmark_id") for report in reports}
    if len(benchmark_ids) != 1:
        raise PreflightSelectionError("preflight reports have different benchmark IDs")
    num_shards_values = {
        report.get("selection", {}).get("num_shards") for report in reports
    }
    if len(num_shards_values) != 1:
        raise PreflightSelectionError("preflight reports disagree on num_shards")
    num_shards = num_shards_values.pop()
    if not isinstance(num_shards, int) or num_shards < 1:
        raise PreflightSelectionError("invalid num_shards in reports")
    shard_indices = [
        report.get("selection", {}).get("shard_index") for report in reports
    ]
    if sorted(shard_indices) != list(range(num_shards)):
        raise PreflightSelectionError("reports do not cover every shard exactly once")
    unique_counts = {
        report.get("selection", {}).get("unique_scenarios")
        for report in reports
    }
    if len(unique_counts) != 1:
        raise PreflightSelectionError(
            "preflight reports disagree on unique scenario count"
        )
    expected = unique_counts.pop()

    cases = []
    failures = {}
    seen_case_ids = set()
    seen_scenarios = set()
    by_change_type: Dict[str, Dict[str, int]] = {}
    for report in sorted(
        reports, key=lambda item: item["selection"]["shard_index"]
    ):
        failures.update(report.get("failures", {}))
        for case in report.get("cases", []):
            case_id = case.get("case_id")
            scenario_id = case.get("scenario_id")
            if case_id in seen_case_ids:
                raise PreflightSelectionError(
                    "duplicate case across preflight shards: %s" % case_id
                )
            if scenario_id in seen_scenarios:
                raise PreflightSelectionError(
                    "duplicate scenario across preflight shards: %s" % scenario_id
                )
            seen_case_ids.add(case_id)
            seen_scenarios.add(scenario_id)
            cases.append(case)
            change_type = case.get("change_type", "unknown")
            counts = by_change_type.setdefault(
                change_type, {"planned": 0, "passed": 0}
            )
            counts["planned"] += 1
            counts["passed"] += int(bool(case.get("passed")))
    if len(cases) != expected:
        raise PreflightSelectionError(
            "merged coverage %d does not match expected %d"
            % (len(cases), expected)
        )
    passed = sum(bool(case.get("passed")) for case in cases)
    return {
        "benchmark_id": benchmark_ids.pop(),
        "planned": len(cases),
        "passed": passed,
        "complete": passed == len(cases) and not failures,
        "failures": dict(sorted(failures.items())),
        "by_change_type": dict(sorted(by_change_type.items())),
        "shards": num_shards,
        "cases": sorted(cases, key=lambda item: item["scenario_id"]),
    }


def select_preflight_cases(
    cases: Sequence[Dict[str, Any]],
    *,
    unique_scenarios: bool = True,
    num_shards: int = 1,
    shard_index: int = 0,
    change_types: Iterable[str] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter, de-duplicate, and deterministically shard manifest cases.

    A Full manifest repeats each resolved physical scenario across policy
    seeds. Physical preflight needs to execute that state transition once, not
    once per policy replicate.
    """

    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise PreflightSelectionError("num_shards must be an integer")
    if num_shards < 1:
        raise PreflightSelectionError("num_shards must be at least 1")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise PreflightSelectionError("shard_index must be an integer")
    if not 0 <= shard_index < num_shards:
        raise PreflightSelectionError(
            "shard_index must be in [0, num_shards)"
        )

    requested_types = set(change_types)
    filtered = [
        case
        for case in cases
        if not requested_types
        or case.get("scenario", {}).get("change_type") in requested_types
    ]
    deduplicated: List[Dict[str, Any]] = []
    seen = set()
    for case in filtered:
        scenario = case.get("scenario", {})
        key = (scenario.get("scenario_id"), scenario.get("seed"))
        if key[0] is None:
            key = (case.get("case_id"), None)
        if unique_scenarios and key in seen:
            continue
        seen.add(key)
        deduplicated.append(case)

    selected = [
        case
        for index, case in enumerate(deduplicated)
        if index % num_shards == shard_index
    ]
    stats = {
        "manifest_cases": len(cases),
        "filtered_cases": len(filtered),
        "unique_scenarios": len(deduplicated),
        "policy_replicates_removed": len(filtered) - len(deduplicated),
        "selected_cases": len(selected),
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    return selected, stats
