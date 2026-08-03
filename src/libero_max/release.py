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
            "%s manifest: %s" % (label, error)
            for error in validate_manifest(manifest)
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
