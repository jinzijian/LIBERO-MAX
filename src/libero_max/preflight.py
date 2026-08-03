"""Dependency-free helpers for deterministic physical-preflight selection."""

from typing import Any, Dict, Iterable, List, Sequence, Tuple


class PreflightSelectionError(ValueError):
    """Raised when a preflight shard or filter is invalid."""


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
