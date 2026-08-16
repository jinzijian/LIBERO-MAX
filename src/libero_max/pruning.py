"""Deterministically remove physical configurations that fail preflight."""

import copy
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .manifest import validate_manifest


FILTER_VERSION = "libero-max-feasibility-filter-v1.0"


class FeasibilityPruningError(ValueError):
    """Raised when preflight evidence cannot safely prune paired manifests."""


def _configuration_key(case: Dict[str, Any]) -> Tuple[str, int, str, int]:
    scenario = case["scenario"]
    return (
        case["task_suite_name"],
        int(case["task_index"]),
        scenario["change_type"],
        int(scenario["randomization"]["draw_id"]),
    )


def _physical_key(case: Dict[str, Any]) -> Tuple[str, int]:
    scenario = case["scenario"]
    return scenario["scenario_id"], int(scenario["seed"])


def _failed_case_ids(preflight: Dict[str, Any]) -> set:
    failed = set(preflight.get("failures", {}))
    failed.update(
        case["case_id"]
        for case in preflight.get("cases", [])
        if case.get("passed") is False and case.get("case_id")
    )
    return failed


def prune_infeasible_configurations(
    core: Dict[str, Any],
    full: Dict[str, Any],
    preflight: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Drop a whole task/change/draw configuration after any state fails.

    Core contains one case per physical scenario and Full repeats the same
    scenario for three policy seeds. Pruning at configuration granularity
    prevents a draw that failed one initial state from surviving in another.
    """

    if preflight.get("benchmark_id") != core.get("benchmark_id"):
        raise FeasibilityPruningError("preflight benchmark_id does not match Core")
    core_cases = core.get("cases", [])
    full_cases = full.get("cases", [])
    if preflight.get("planned") != len(core_cases):
        raise FeasibilityPruningError("preflight does not plan every Core scenario")
    reported_ids = {
        case.get("case_id") for case in preflight.get("cases", [])
    }
    core_by_id = {case["case_id"]: case for case in core_cases}
    if reported_ids != set(core_by_id):
        raise FeasibilityPruningError("preflight does not report every Core case")
    failed_ids = _failed_case_ids(preflight)
    unknown_failed = failed_ids - set(core_by_id)
    if unknown_failed:
        raise FeasibilityPruningError(
            "preflight failures reference unknown Core cases: %s"
            % ", ".join(sorted(unknown_failed))
        )
    blocked = {_configuration_key(core_by_id[case_id]) for case_id in failed_ids}

    def keep(case: Dict[str, Any]) -> bool:
        return _configuration_key(case) not in blocked

    pruned_core = copy.deepcopy(core)
    pruned_full = copy.deepcopy(full)
    pruned_core["cases"] = [copy.deepcopy(case) for case in core_cases if keep(case)]
    pruned_full["cases"] = [copy.deepcopy(case) for case in full_cases if keep(case)]
    policy = "drop_task_change_draw_if_any_init_state_fails_preflight"

    errors: List[str] = []
    for label, manifest in (("core", pruned_core), ("full", pruned_full)):
        errors.extend(
            "%s manifest: %s" % (label, error)
            for error in validate_manifest(manifest)
        )
    core_counts = Counter(_physical_key(case) for case in pruned_core["cases"])
    full_counts = Counter(_physical_key(case) for case in pruned_full["cases"])
    if set(core_counts) != set(full_counts):
        errors.append("pruned Core and Full physical scenarios differ")
    if any(value != 1 for value in core_counts.values()):
        errors.append("pruned Core must retain each physical scenario once")
    if any(value != 3 for value in full_counts.values()):
        errors.append("pruned Full must retain each physical scenario three times")
    retained_types = {
        case["scenario"]["change_type"] for case in pruned_core["cases"]
    }
    original_types = {case["scenario"]["change_type"] for case in core_cases}
    if retained_types != original_types:
        errors.append("feasibility pruning removed an entire change type")
    if errors:
        raise FeasibilityPruningError("; ".join(errors))

    blocked_rows = [
        {
            "task_suite_name": suite,
            "task_index": task_index,
            "change_type": change_type,
            "draw_id": draw_id,
        }
        for suite, task_index, change_type, draw_id in sorted(blocked)
    ]
    removed_by_type = Counter(row["change_type"] for row in blocked_rows)
    report = {
        "filter_version": FILTER_VERSION,
        "policy": policy,
        "source_preflight_complete": bool(preflight.get("complete")),
        "source_preflight_planned": preflight.get("planned"),
        "source_preflight_passed": preflight.get("passed"),
        "failed_case_ids": sorted(failed_ids),
        "excluded_configurations": blocked_rows,
        "excluded_configurations_by_change_type": dict(sorted(removed_by_type.items())),
        "core_cases_before": len(core_cases),
        "core_cases_after": len(pruned_core["cases"]),
        "full_cases_before": len(full_cases),
        "full_cases_after": len(pruned_full["cases"]),
    }
    return pruned_core, pruned_full, report
