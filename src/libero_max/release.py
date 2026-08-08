"""Release-gate audits for frozen LIBERO-MAX manifests."""

from collections import Counter
from typing import Any, Dict, List, Tuple

from .manifest import validate_manifest


def _scenario_key(case: Dict[str, Any]) -> Tuple[str, int]:
    scenario = case["scenario"]
    return scenario["scenario_id"], scenario["seed"]


def audit_v1_release(
    catalog: Dict[str, Any],
    core: Dict[str, Any],
    full: Dict[str, Any],
    preflight: Dict[str, Any],
) -> List[str]:
    errors = []
    calibration = catalog.get("relocation_calibration")
    if not isinstance(calibration, dict) or not calibration.get("complete"):
        errors.append("relocation calibration is not complete")
    for label, manifest in (("core", core), ("full", full)):
        errors.extend(
            "%s manifest: %s" % (label, error) for error in validate_manifest(manifest)
        )
    if preflight.get("benchmark_id") != core.get("benchmark_id"):
        errors.append("preflight benchmark_id does not match Core")
    if not preflight.get("complete"):
        errors.append("physical preflight is not complete")
    core_counts = Counter(_scenario_key(case) for case in core.get("cases", []))
    full_counts = Counter(_scenario_key(case) for case in full.get("cases", []))
    if any(count != 1 for count in core_counts.values()):
        errors.append("Core must contain each physical scenario exactly once")
    if set(core_counts) != set(full_counts):
        errors.append("Core and Full do not contain the same physical scenarios")
    if any(count != 3 for count in full_counts.values()):
        errors.append("Full must contain each physical scenario for three seeds")
    preflight_scenarios = {
        case.get("scenario_id") for case in preflight.get("cases", [])
    }
    core_scenarios = {key[0] for key in core_counts}
    if preflight_scenarios != core_scenarios:
        errors.append("preflight scenario coverage does not exactly match Core")
    if preflight.get("planned") != len(core_counts):
        errors.append("preflight planned count does not match Core")
    return errors


def _audit_hard_preflight(
    label: str, manifest: Dict[str, Any], preflight: Dict[str, Any]
) -> List[str]:
    errors = []
    if preflight.get("benchmark_id") != manifest.get("benchmark_id"):
        errors.append("%s preflight benchmark_id does not match manifest" % label)
    if not preflight.get("complete"):
        errors.append("%s physical preflight is not complete" % label)
    scenario_ids = {
        case.get("scenario", {}).get("scenario_id")
        for case in manifest.get("cases", [])
    }
    preflight_ids = {case.get("scenario_id") for case in preflight.get("cases", [])}
    if preflight_ids != scenario_ids:
        errors.append("%s preflight coverage does not exactly match manifest" % label)
    if preflight.get("planned") != len(scenario_ids):
        errors.append("%s preflight planned count does not match manifest" % label)
    if any(not case.get("passed") for case in preflight.get("cases", [])):
        errors.append("%s preflight contains a failed case" % label)
    return errors


def audit_hard_release(
    catalog: Dict[str, Any],
    core: Dict[str, Any],
    full: Dict[str, Any],
    core_preflight: Dict[str, Any],
    full_preflight: Dict[str, Any],
    *,
    expected_core_pairs: int = 1400,
    expected_full_pairs: int = 10030,
) -> List[str]:
    """Audit the frozen MAX-Hard release and its full MuJoCo coverage."""

    errors = []
    for label, manifest in (("Core", core), ("Full", full)):
        errors.extend(
            "%s manifest: %s" % (label, error) for error in validate_manifest(manifest)
        )
    core_cases = core.get("cases", [])
    full_cases = full.get("cases", [])
    if len(core_cases) != expected_core_pairs:
        errors.append("Core must contain exactly %d pairs" % expected_core_pairs)
    if len(full_cases) != expected_full_pairs:
        errors.append("Full must contain exactly %d pairs" % expected_full_pairs)
    catalog_tasks = catalog.get("tasks", [])
    catalog_keys = {
        (task.get("task_suite_name"), task.get("task_index")) for task in catalog_tasks
    }
    full_keys = {
        (case.get("task_suite_name"), case.get("task_index")) for case in full_cases
    }
    if len(catalog_tasks) != expected_full_pairs or len(catalog_keys) != len(
        catalog_tasks
    ):
        errors.append("catalog does not contain the exact unique Full task set")
    if full_keys != catalog_keys:
        errors.append("Full task coverage does not exactly match catalog")
    core_ids = {case.get("case_id") for case in core_cases}
    full_ids = {case.get("case_id") for case in full_cases}
    if len(core_ids) != len(core_cases):
        errors.append("Core contains duplicate case IDs")
    if len(full_ids) != len(full_cases):
        errors.append("Full contains duplicate case IDs")
    if not core_ids.issubset(full_ids):
        errors.append("Core is not an exact Full subset")

    event_counts = Counter(
        case.get("scenario", {}).get("change_type") for case in core_cases
    )
    if set(event_counts.values()) != {175} or len(event_counts) != 8:
        errors.append("Core must contain exactly 175 cases for each of 8 events")
    category_counts = Counter(case.get("substrate_category") for case in core_cases)
    if set(category_counts.values()) != {200} or len(category_counts) != 7:
        errors.append("Core must contain exactly 200 cases per Plus category")
    difficulty_counts = Counter(case.get("substrate_difficulty") for case in core_cases)
    if set(difficulty_counts.values()) != {280} or set(difficulty_counts) != {
        1,
        2,
        3,
        4,
        5,
    }:
        errors.append("Core must contain exactly 280 cases per Plus difficulty")
    cell_counts = Counter(
        (case.get("substrate_category"), case.get("substrate_difficulty"))
        for case in core_cases
    )
    if set(cell_counts.values()) != {40} or len(cell_counts) != 35:
        errors.append("Core must contain exactly 40 cases in each 7 x 5 cell")

    errors.extend(_audit_hard_preflight("Core", core, core_preflight))
    errors.extend(_audit_hard_preflight("Full", full, full_preflight))
    return errors


def audit_max5600_release(
    catalog: Dict[str, Any],
    manifest: Dict[str, Any],
    frozen_core: Dict[str, Any],
    preflight: Dict[str, Any],
    *,
    expected_pairs: int = 5600,
) -> List[str]:
    """Audit the single official balanced LIBERO-MAX-5600 release."""

    errors = ["official manifest: %s" % error for error in validate_manifest(manifest)]
    cases = manifest.get("cases", [])
    if manifest.get("benchmark_id") != "libero-max-5600":
        errors.append("official benchmark_id must be libero-max-5600")
    if len(cases) != expected_pairs:
        errors.append(
            "official manifest must contain exactly %d pairs" % expected_pairs
        )
    if manifest.get("protocol", {}).get("query_interval") != 16:
        errors.append("official paired protocol must use query_interval 16")

    catalog_tasks = catalog.get("tasks", [])
    catalog_keys = [
        (task.get("task_suite_name"), task.get("task_index")) for task in catalog_tasks
    ]
    if len(catalog_tasks) != 10030 or len(set(catalog_keys)) != 10030:
        errors.append("source catalog must contain 10,030 unique Plus tasks")
    case_keys = [
        (case.get("task_suite_name"), case.get("task_index")) for case in cases
    ]
    if len(set(case_keys)) != len(cases):
        errors.append("official manifest contains duplicate source tasks")
    if not set(case_keys).issubset(set(catalog_keys)):
        errors.append("official manifest contains tasks outside the source catalog")

    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(cases):
        errors.append("official manifest contains duplicate case IDs")
    core_ids = {case.get("case_id") for case in frozen_core.get("cases", [])}
    if len(core_ids) != 1400:
        errors.append("development Core provenance must contain 1,400 cases")

    category_counts = Counter(case.get("substrate_category") for case in cases)
    if len(category_counts) != 7 or set(category_counts.values()) != {800}:
        errors.append("official manifest must contain 800 cases per Plus category")
    event_counts = Counter(
        case.get("scenario", {}).get("change_type") for case in cases
    )
    if len(event_counts) != 8 or set(event_counts.values()) != {700}:
        errors.append("official manifest must contain 700 cases per change type")
    event_draw_counts = Counter(
        (
            case.get("scenario", {}).get("change_type"),
            case.get("scenario", {}).get("randomization", {}).get("draw_id"),
        )
        for case in cases
    )
    if len(event_draw_counts) != 16 or set(event_draw_counts.values()) != {350}:
        errors.append("official manifest must contain 350 cases per change/draw")
    category_event_draw_counts = Counter(
        (
            case.get("substrate_category"),
            case.get("scenario", {}).get("change_type"),
            case.get("scenario", {}).get("randomization", {}).get("draw_id"),
        )
        for case in cases
    )
    if len(category_event_draw_counts) != 7 * 8 * 2 or set(
        category_event_draw_counts.values()
    ) != {50}:
        errors.append("every category/change/draw cell must contain 50 cases")

    errors.extend(_audit_hard_preflight("Official", manifest, preflight))
    if preflight.get("passed") != expected_pairs:
        errors.append("official preflight must pass all %d cases" % expected_pairs)
    return errors


def audit_pro_hard_release(
    catalog: Dict[str, Any],
    manifest: Dict[str, Any],
    preflight: Dict[str, Any],
    *,
    expected_pairs: int = 2400,
) -> List[str]:
    """Audit the balanced PRO-initialized MAX subset and MuJoCo coverage."""

    errors = ["PRO-Hard manifest: %s" % error for error in validate_manifest(manifest)]
    cases = manifest.get("cases", [])
    if manifest.get("benchmark_id") != "libero-max-pro-hard-2400":
        errors.append("PRO-Hard benchmark_id must be libero-max-pro-hard-2400")
    if len(cases) != expected_pairs:
        errors.append(
            "PRO-Hard manifest must contain exactly %d pairs" % expected_pairs
        )
    if manifest.get("protocol", {}).get("query_interval") != 16:
        errors.append("PRO-Hard paired protocol must use query_interval 16")
    source_revision = catalog.get("source_revision")
    if manifest.get("protocol", {}).get("source_benchmark_commit") != source_revision:
        errors.append("PRO-Hard manifest and catalog revisions do not match")
    catalog_tasks = catalog.get("tasks", [])
    catalog_keys = {
        (
            task.get("pro_category"),
            task.get("task_suite_name"),
            task.get("task_index"),
        )
        for task in catalog_tasks
    }
    if len(catalog_tasks) != 400 or len(catalog_keys) != 400:
        errors.append("PRO catalog must contain 400 unique task variants")
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("PRO-Hard manifest contains duplicate case IDs")
    category_counts = Counter(
        case.get("substrate_variant", {}).get("category") for case in cases
    )
    if len(category_counts) != 10 or set(category_counts.values()) != {240}:
        errors.append("PRO-Hard must contain 240 pairs per PRO category")
    event_counts = Counter(
        case.get("scenario", {}).get("change_type") for case in cases
    )
    if len(event_counts) != 8 or set(event_counts.values()) != {300}:
        errors.append("PRO-Hard must contain 300 pairs per MAX change")
    cell_counts = Counter(
        (
            case.get("substrate_variant", {}).get("category"),
            case.get("scenario", {}).get("change_type"),
            case.get("scenario", {}).get("randomization", {}).get("draw_id"),
        )
        for case in cases
    )
    if len(cell_counts) != 10 * 8 * 2 or set(cell_counts.values()) != {15}:
        errors.append("every PRO category/change/draw cell must contain 15 pairs")
    if any(not 0 <= case.get("init_state_index", -1) < 10 for case in cases):
        errors.append("PRO-Hard init-state indices must use the frozen 0..9 pool")
    errors.extend(_audit_hard_preflight("PRO-Hard", manifest, preflight))
    if preflight.get("passed") != expected_pairs:
        errors.append("PRO-Hard preflight must pass all %d cases" % expected_pairs)
    return errors


def audit_max8000_composition(
    base_manifest: Dict[str, Any],
    pro_manifest: Dict[str, Any],
    combined_manifest: Dict[str, Any],
) -> List[str]:
    """Check that MAX-8000 is exactly Base-5600 plus PRO-Hard-2400."""

    errors = [
        "combined manifest: %s" % error
        for error in validate_manifest(combined_manifest)
    ]
    if combined_manifest.get("benchmark_id") != "libero-max-8000":
        errors.append("combined benchmark_id must be libero-max-8000")
    base_ids = {case.get("case_id") for case in base_manifest.get("cases", [])}
    pro_ids = {case.get("case_id") for case in pro_manifest.get("cases", [])}
    combined_ids = {case.get("case_id") for case in combined_manifest.get("cases", [])}
    if len(base_ids) != 5600 or len(pro_ids) != 2400:
        errors.append("combined provenance must contain 5,600 + 2,400 unique IDs")
    if base_ids & pro_ids:
        errors.append("Base and PRO-Hard case IDs overlap")
    if combined_ids != base_ids | pro_ids or len(combined_ids) != 8000:
        errors.append("MAX-8000 is not the exact Base/PRO-Hard union")
    return errors
