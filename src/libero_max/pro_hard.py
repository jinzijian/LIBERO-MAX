"""Deterministic LIBERO-PRO substrate for the LIBERO-MAX-8000 extension."""

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .hard import (
    CHANGE_TYPE_ORDER,
    HardBuildError,
    _sample_change,
    _stable_seed,
    eligible_change_types,
)
from .manifest import validate_manifest


PRO_SAMPLER_VERSION = "libero-max-pro-hard-v1.0"
PRO_CATEGORIES = (
    "semantic",
    "object",
    "position",
    "task",
    "visual_noise_glare",
    "camera_view_angle",
    "object_texture",
    "view_occlusion",
    "object_shape",
    "initial_pose_position_angle",
)
EXCLUDED_UPSTREAM_CATEGORIES = {
    "runtime_object_move": (
        "already changes the world during execution and would confound the "
        "MAX intervention-time comparison"
    ),
    "environment": (
        "the pinned public dataset revision does not contain complete "
        "environment BDDL/init artifacts"
    ),
}
PRO_TASKS_PER_CATEGORY = 40
PRO_CASES_PER_CELL = 15
PRO_INIT_STATE_POOL = 10
PRO_HARD_PAIRS = len(PRO_CATEGORIES) * len(CHANGE_TYPE_ORDER) * 2 * PRO_CASES_PER_CELL


def _pro_task_key(task: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        task["pro_category"],
        task["task_suite_name"],
        task["task_index"],
    )


def _candidate_key(
    task: Dict[str, Any], init_state_index: int, event: str, draw_id: int
) -> Tuple[str, str, int, int, str, int]:
    return (*_pro_task_key(task), init_state_index, event, draw_id)


def _select_cell(
    tasks: Sequence[Dict[str, Any]],
    category: str,
    event: str,
    draw_id: int,
    rejected: Set[Tuple[str, str, int, int, str, int]],
    usage: Counter,
) -> List[Tuple[Dict[str, Any], int]]:
    eligible = [task for task in tasks if event in eligible_change_types(task)]
    if not eligible:
        raise HardBuildError("%s has no tasks eligible for %s" % (category, event))
    selected: List[Tuple[Dict[str, Any], int]] = []
    selected_keys = set()
    while len(selected) < PRO_CASES_PER_CELL:
        ranked = []
        for task in eligible:
            task_key = _pro_task_key(task)
            for init_state_index in range(PRO_INIT_STATE_POOL):
                execution_key = _candidate_key(task, init_state_index, event, draw_id)
                if execution_key in rejected or execution_key in selected_keys:
                    continue
                ranked.append(
                    (
                        usage[task_key],
                        sum(selected_task is task for selected_task, _ in selected),
                        _stable_seed(
                            (
                                PRO_SAMPLER_VERSION,
                                "cell",
                                category,
                                event,
                                draw_id,
                                *task_key,
                                init_state_index,
                            )
                        ),
                        task,
                        init_state_index,
                        execution_key,
                    )
                )
        if not ranked:
            raise HardBuildError(
                "%s/%s/d%d cannot supply %d valid cases"
                % (category, event, draw_id, PRO_CASES_PER_CELL)
            )
        _, _, _, task, init_state_index, execution_key = min(
            ranked, key=lambda row: row[:3]
        )
        selected.append((task, init_state_index))
        selected_keys.add(execution_key)
        usage[_pro_task_key(task)] += 1
    return selected


def _build_pro_case(
    task: Dict[str, Any],
    init_state_index: int,
    event: str,
    draw_id: int,
) -> Dict[str, Any]:
    category, suite, task_index = _pro_task_key(task)
    seed = _stable_seed(
        (
            PRO_SAMPLER_VERSION,
            category,
            suite,
            task_index,
            init_state_index,
            event,
            draw_id,
        )
    )
    family, severity, setup, change = _sample_change(task, event, draw_id, seed)
    scenario_id = "pro-%s-%s-t%02d-i%02d-%s-d%d" % (
        category.replace("_", "-"),
        suite.replace("libero_", ""),
        task_index,
        init_state_index,
        event,
        draw_id,
    )
    scenario = {
        "scenario_id": scenario_id,
        "base_task_id": "libero-pro/%s/%s/task_%d" % (category, suite, task_index),
        "seed": seed,
        "change_family": family,
        "change_type": event,
        "severity": severity,
        "trigger": {
            "type": "on_proximity",
            "value": task["trigger_entity"],
            "distance_m": 0.18,
        },
        "change": change,
        "expected_response_mode": (
            "replan" if family in {"GEO", "OBSTACLE"} else "continue"
        ),
        "safety_constraints": [
            "task_remains_feasible",
            "no_simulator_error",
            "no_new_unrelated_collision",
        ],
        "randomization": {
            "sampler": PRO_SAMPLER_VERSION,
            "draw_id": draw_id,
            "seed": seed,
        },
    }
    if setup:
        scenario["setup"] = setup
    return {
        "case_id": scenario_id + "-p195",
        "task_suite_name": suite,
        "task_index": task_index,
        "init_state_index": init_state_index,
        "policy_seed": 195,
        "timing_bucket": "middle",
        "task_name": task["task_name"],
        "substrate_category": "LIBERO-PRO/%s" % category,
        "substrate_difficulty": None,
        "substrate_variant": {
            "benchmark": "LIBERO-PRO",
            "category": category,
            "bddl_file": task["bddl_file"],
            "init_states_file": task["init_states_file"],
            "init_reference_bddl_file": task["init_reference_bddl_file"],
            "language": task["language"],
            "source_revision": task["source_revision"],
        },
        "dynamic_phase": "pre_grasp_proximity",
        "scenario": scenario,
    }


def build_pro_hard_manifest(
    catalog: Dict[str, Any],
    rejected_configurations: Iterable[Tuple[str, str, int, int, str, int]] = (),
) -> Dict[str, Any]:
    """Build the balanced 2,400-pair PRO-initialized MAX subset."""

    tasks = catalog.get("tasks")
    if not isinstance(tasks, list):
        raise HardBuildError("PRO catalog tasks must be an array")
    source_revision = catalog.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise HardBuildError("PRO catalog requires source_revision")
    category_counts = Counter(task.get("pro_category") for task in tasks)
    if set(category_counts) != set(PRO_CATEGORIES) or any(
        category_counts[category] != PRO_TASKS_PER_CATEGORY
        for category in PRO_CATEGORIES
    ):
        raise HardBuildError("PRO catalog must contain 40 tasks in each category")
    if len({_pro_task_key(task) for task in tasks}) != len(tasks):
        raise HardBuildError("PRO catalog contains duplicate category/suite/task keys")
    if any(task.get("source_revision") != source_revision for task in tasks):
        raise HardBuildError("PRO catalog task revisions are inconsistent")

    tasks_by_category = defaultdict(list)
    for task in tasks:
        tasks_by_category[task["pro_category"]].append(task)
    rejected = set(rejected_configurations)
    usage: Counter = Counter()
    cases = []
    for category in PRO_CATEGORIES:
        for event in CHANGE_TYPE_ORDER:
            for draw_id in (0, 1):
                for task, init_state_index in _select_cell(
                    tasks_by_category[category],
                    category,
                    event,
                    draw_id,
                    rejected,
                    usage,
                ):
                    cases.append(
                        _build_pro_case(task, init_state_index, event, draw_id)
                    )
    manifest = {
        "benchmark_id": "libero-max-pro-hard-2400",
        "benchmark_version": "3.0.0-candidate",
        "protocol": {
            "arms": ["control", "intervention"],
            "query_interval": 16,
            "scoring_track": "physical_completion",
            "substrate": "LIBERO-PRO",
            "profile": "pro-hard",
            "source_benchmark_commit": source_revision,
            "selection_contract": (
                "10 PRO categories x 8 MAX changes x 2 deterministic draws x 15 cases"
            ),
        },
        "cases": sorted(cases, key=lambda case: case["case_id"]),
    }
    errors = validate_manifest(manifest)
    if errors:
        raise HardBuildError(
            "generated PRO-Hard manifest is invalid: %s" % "; ".join(errors)
        )
    if len(manifest["cases"]) != PRO_HARD_PAIRS:
        raise AssertionError("PRO-Hard manifest must contain 2,400 pairs")
    return manifest


def rejected_configurations_from_reports(
    catalog: Dict[str, Any], reports: Iterable[Dict[str, Any]]
) -> Set[Tuple[str, str, int, int, str, int]]:
    """Resolve chronological preflight failures, including replacement cases.

    Each report applies to the manifest produced after all earlier reports.
    Rebuilding at every round is necessary because a failed replacement case
    does not exist in the original unfiltered manifest.
    """

    rejected: Set[Tuple[str, str, int, int, str, int]] = set()
    for report in reports:
        current = build_pro_hard_manifest(catalog, rejected)
        cases_by_id = {case["case_id"]: case for case in current["cases"]}
        unknown = []
        for row in report.get("cases", []):
            if row.get("passed"):
                continue
            case = cases_by_id.get(row.get("case_id"))
            if case is None:
                unknown.append(str(row.get("case_id")))
                continue
            rejected.add(
                (
                    case["substrate_variant"]["category"],
                    case["task_suite_name"],
                    case["task_index"],
                    case["init_state_index"],
                    case["scenario"]["change_type"],
                    case["scenario"]["randomization"]["draw_id"],
                )
            )
        if unknown:
            raise HardBuildError(
                "preflight contains failed cases outside its chronological manifest: %s"
                % ", ".join(sorted(unknown)[:10])
            )
    return rejected


def _case_candidate_key(case: Dict[str, Any]) -> Tuple[str, str, int, int, str, int]:
    return (
        case["substrate_variant"]["category"],
        case["task_suite_name"],
        case["task_index"],
        case["init_state_index"],
        case["scenario"]["change_type"],
        case["scenario"]["randomization"]["draw_id"],
    )


def repair_pro_hard_manifest(
    catalog: Dict[str, Any],
    manifest: Dict[str, Any],
    failed_case_ids: Iterable[str],
    rejected_configurations: Iterable[Tuple[str, str, int, int, str, int]],
) -> Dict[str, Any]:
    """Replace failed cases in-cell while preserving every passing case ID."""

    failed = set(failed_case_ids)
    current_ids = {case["case_id"] for case in manifest.get("cases", [])}
    unknown = failed - current_ids
    if unknown:
        raise HardBuildError(
            "repair failures are outside the current manifest: %s"
            % ", ".join(sorted(unknown)[:10])
        )
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list):
        raise HardBuildError("PRO catalog tasks must be an array")
    tasks_by_category = defaultdict(list)
    for task in tasks:
        tasks_by_category[task["pro_category"]].append(task)

    retained = [case for case in manifest["cases"] if case["case_id"] not in failed]
    rejected = set(rejected_configurations)
    rejected.update(
        _case_candidate_key(case)
        for case in manifest["cases"]
        if case["case_id"] in failed
    )
    used_execution_keys = {_case_candidate_key(case) for case in retained}
    usage: Counter = Counter(
        (
            case["substrate_variant"]["category"],
            case["task_suite_name"],
            case["task_index"],
        )
        for case in retained
    )
    by_cell = Counter(
        (
            case["substrate_variant"]["category"],
            case["scenario"]["change_type"],
            case["scenario"]["randomization"]["draw_id"],
        )
        for case in retained
    )
    passing_task_by_cell = Counter(
        (
            case["substrate_variant"]["category"],
            case["scenario"]["change_type"],
            case["scenario"]["randomization"]["draw_id"],
            case["task_suite_name"],
            case["task_index"],
        )
        for case in retained
    )
    additions = []
    for category in PRO_CATEGORIES:
        for event in CHANGE_TYPE_ORDER:
            for draw_id in (0, 1):
                cell = (category, event, draw_id)
                needed = PRO_CASES_PER_CELL - by_cell[cell]
                if needed < 0:
                    raise HardBuildError("repair cell contains more than 15 cases")
                selected_keys = set()
                for _ in range(needed):
                    ranked = []
                    for task in tasks_by_category[category]:
                        if event not in eligible_change_types(task):
                            continue
                        task_key = _pro_task_key(task)
                        for init_state_index in range(PRO_INIT_STATE_POOL):
                            execution_key = _candidate_key(
                                task, init_state_index, event, draw_id
                            )
                            if (
                                execution_key in rejected
                                or execution_key in used_execution_keys
                                or execution_key in selected_keys
                            ):
                                continue
                            ranked.append(
                                (
                                    0
                                    if passing_task_by_cell[
                                        (*cell, task["task_suite_name"], task["task_index"])
                                    ]
                                    else 1,
                                    passing_task_by_cell[
                                        (*cell, task["task_suite_name"], task["task_index"])
                                    ],
                                    usage[task_key],
                                    _stable_seed(
                                        (
                                            PRO_SAMPLER_VERSION,
                                            "repair",
                                            category,
                                            event,
                                            draw_id,
                                            *task_key,
                                            init_state_index,
                                        )
                                    ),
                                    task,
                                    init_state_index,
                                    execution_key,
                                )
                            )
                    if not ranked:
                        raise HardBuildError(
                            "%s/%s/d%d cannot repair its failed cases"
                            % (category, event, draw_id)
                        )
                    _, _, _, _, task, init_state_index, execution_key = min(
                        ranked, key=lambda row: row[:4]
                    )
                    additions.append(
                        _build_pro_case(task, init_state_index, event, draw_id)
                    )
                    selected_keys.add(execution_key)
                    used_execution_keys.add(execution_key)
                    usage[_pro_task_key(task)] += 1
                    passing_task_by_cell[
                        (*cell, task["task_suite_name"], task["task_index"])
                    ] += 1
    repaired = dict(manifest)
    repaired["cases"] = sorted(retained + additions, key=lambda case: case["case_id"])
    errors = validate_manifest(repaired)
    if errors:
        raise HardBuildError("repaired PRO manifest is invalid: %s" % "; ".join(errors))
    counts = Counter(
        (
            case["substrate_variant"]["category"],
            case["scenario"]["change_type"],
            case["scenario"]["randomization"]["draw_id"],
        )
        for case in repaired["cases"]
    )
    if len(repaired["cases"]) != PRO_HARD_PAIRS or set(counts.values()) != {
        PRO_CASES_PER_CELL
    }:
        raise HardBuildError("repaired PRO manifest changed its frozen cell quotas")
    return repaired


def combine_max8000_manifests(
    base_manifest: Dict[str, Any], pro_manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine the frozen Base-5600 and candidate PRO-Hard-2400 cases."""

    for label, manifest in (("base", base_manifest), ("PRO-Hard", pro_manifest)):
        errors = validate_manifest(manifest)
        if errors:
            raise HardBuildError("%s manifest: %s" % (label, "; ".join(errors)))
    if base_manifest.get("benchmark_id") != "libero-max-5600":
        raise HardBuildError("MAX-8000 base must be the official MAX-5600")
    if len(base_manifest["cases"]) != 5600 or len(pro_manifest["cases"]) != 2400:
        raise HardBuildError("MAX-8000 requires exactly 5,600 + 2,400 pairs")
    combined = {
        "benchmark_id": "libero-max-8000",
        "benchmark_version": "3.0.0-candidate",
        "protocol": {
            "arms": ["control", "intervention"],
            "query_interval": 16,
            "scoring_track": "physical_completion",
            "substrate": "LIBERO-Plus + LIBERO-PRO",
            "profile": "official-candidate",
            "source_benchmark_commit": "%s;%s"
            % (
                base_manifest["protocol"]["source_benchmark_commit"],
                pro_manifest["protocol"]["source_benchmark_commit"],
            ),
            "selection_contract": "MAX-Base-5600 + MAX-PRO-Hard-2400",
        },
        "cases": sorted(
            base_manifest["cases"] + pro_manifest["cases"],
            key=lambda case: case["case_id"],
        ),
    }
    errors = validate_manifest(combined)
    if errors:
        raise HardBuildError("combined MAX-8000 manifest: %s" % "; ".join(errors))
    return combined


def pro_hard_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    cases = manifest["cases"]
    cells = Counter(
        (
            case["substrate_variant"]["category"],
            case["scenario"]["change_type"],
            case["scenario"]["randomization"]["draw_id"],
        )
        for case in cases
    )
    return {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_version": manifest["benchmark_version"],
        "status": "candidate",
        "matched_pairs": len(cases),
        "rollouts_per_model": 2 * len(cases),
        "pro_categories": len(PRO_CATEGORIES),
        "change_types": len(CHANGE_TYPE_ORDER),
        "draws_per_change": 2,
        "joint_cells": len(cells),
        "pairs_per_joint_cell": sorted(set(cells.values())),
        "source_revision": manifest["protocol"]["source_benchmark_commit"],
        "excluded_upstream_categories": EXCLUDED_UPSTREAM_CATEGORIES,
    }
