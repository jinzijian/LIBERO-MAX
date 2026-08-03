"""Scenario loading and validation for LIBERO-MAX."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


CHANGE_FAMILIES = {
    "OBS",
    "GEO",
    "CLUTTER",
    "OBSTACLE",
    "OBS-NEW",
    "INTENT",
    "FEAS",
}
CHANGE_TYPES = {
    "illumination_switch",
    "camera_shift",
    "target_relocation",
    "receptacle_relocation",
    "distractor_burst",
    "obstacle_insertion",
    "target_removal",
    "receptacle_removal",
    "instruction_target_update",
    "instruction_receptacle_update",
    "task_cancel",
}
SEVERITIES = {"low", "medium", "high"}
TRIGGER_TYPES = {
    "after_grasp",
    "before_grasp",
    "after_subgoal",
    "on_region_entry",
    "on_proximity",
    "progress_fraction",
    "fixed_step",
}
RESPONSE_MODES = {
    "continue",
    "replan",
    "follow_update",
    "clarify",
    "stop",
    "report_infeasible",
}

REQUIRED_FIELDS = {
    "scenario_id",
    "base_task_id",
    "seed",
    "change_family",
    "severity",
    "trigger",
    "change",
    "expected_response_mode",
    "safety_constraints",
}
OPTIONAL_FIELDS = {"setup", "change_type", "randomization"}


class ScenarioLoadError(ValueError):
    """Raised when scenario documents cannot be loaded."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def scenario_key(scenario: Dict[str, Any]) -> Tuple[str, int]:
    return scenario["scenario_id"], scenario["seed"]


def validate_scenario(data: Any) -> List[str]:
    """Return human-readable validation errors for one scenario record."""

    if not isinstance(data, dict):
        return ["scenario must be a JSON object"]

    errors: List[str] = []
    missing = sorted(REQUIRED_FIELDS - set(data))
    unknown = sorted(set(data) - REQUIRED_FIELDS - OPTIONAL_FIELDS)
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if unknown:
        errors.append("unknown fields: " + ", ".join(unknown))

    for field in ("scenario_id", "base_task_id"):
        if field in data and not _nonempty_string(data[field]):
            errors.append("%s must be a non-empty string" % field)

    if "seed" in data and (not _is_integer(data["seed"]) or data["seed"] < 0):
        errors.append("seed must be a non-negative integer")

    if "change_family" in data and data["change_family"] not in CHANGE_FAMILIES:
        errors.append(
            "change_family must be one of: " + ", ".join(sorted(CHANGE_FAMILIES))
        )

    if "change_type" in data and data["change_type"] not in CHANGE_TYPES:
        errors.append("change_type must be one of: " + ", ".join(sorted(CHANGE_TYPES)))

    if "severity" in data and data["severity"] not in SEVERITIES:
        errors.append("severity must be one of: " + ", ".join(sorted(SEVERITIES)))

    if "expected_response_mode" in data and data["expected_response_mode"] not in RESPONSE_MODES:
        errors.append(
            "expected_response_mode must be one of: "
            + ", ".join(sorted(RESPONSE_MODES))
        )

    trigger = data.get("trigger")
    if trigger is not None:
        errors.extend(_validate_trigger(trigger))

    errors.extend(_validate_change(data.get("change"), "change"))

    setup = data.get("setup")
    if setup is not None:
        if not isinstance(setup, list):
            errors.append("setup must be an array")
        else:
            for index, item in enumerate(setup):
                errors.extend(_validate_change(item, "setup[%d]" % index))

    randomization = data.get("randomization")
    if randomization is not None:
        if not isinstance(randomization, dict):
            errors.append("randomization must be a JSON object")
        else:
            expected = {"sampler", "draw_id", "seed"}
            missing = expected - set(randomization)
            unknown = set(randomization) - expected
            if missing:
                errors.append(
                    "randomization missing fields: " + ", ".join(sorted(missing))
                )
            if unknown:
                errors.append(
                    "randomization has unknown fields: " + ", ".join(sorted(unknown))
                )
            if not _nonempty_string(randomization.get("sampler")):
                errors.append("randomization.sampler must be a non-empty string")
            for field in ("draw_id", "seed"):
                value = randomization.get(field)
                if not _is_integer(value) or value < 0:
                    errors.append(
                        "randomization.%s must be a non-negative integer" % field
                    )

    constraints = data.get("safety_constraints")
    if constraints is not None:
        if not isinstance(constraints, list):
            errors.append("safety_constraints must be an array")
        else:
            for index, item in enumerate(constraints):
                if not _nonempty_string(item):
                    errors.append(
                        "safety_constraints[%d] must be a non-empty string" % index
                    )

    return errors


def _validate_change(change: Any, label: str) -> List[str]:
    if not isinstance(change, dict):
        return ["%s must be a JSON object" % label]
    if not _nonempty_string(change.get("operation")):
        return ["%s.operation must be a non-empty string" % label]
    return []


def _validate_trigger(trigger: Any) -> List[str]:
    if not isinstance(trigger, dict):
        return ["trigger must be a JSON object"]

    errors: List[str] = []
    missing = {"type", "value"} - set(trigger)
    unknown = set(trigger) - {"type", "value", "distance_m"}
    if missing:
        errors.append("trigger missing fields: " + ", ".join(sorted(missing)))
    if unknown:
        errors.append("trigger has unknown fields: " + ", ".join(sorted(unknown)))

    trigger_type = trigger.get("type")
    value = trigger.get("value")
    if trigger_type not in TRIGGER_TYPES:
        errors.append("trigger.type must be one of: " + ", ".join(sorted(TRIGGER_TYPES)))
        return errors

    if trigger_type == "on_proximity":
        if not _nonempty_string(value):
            errors.append("on_proximity trigger.value must name an entity")
        distance = trigger.get("distance_m")
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or distance <= 0
        ):
            errors.append("on_proximity trigger.distance_m must be positive")
    elif "distance_m" in trigger:
        errors.append("trigger.distance_m is only valid for on_proximity")
    elif trigger_type == "fixed_step":
        if not _is_integer(value) or value < 1:
            errors.append("fixed_step trigger.value must be an integer >= 1")
    elif trigger_type == "progress_fraction":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            or value >= 1
        ):
            errors.append("progress_fraction trigger.value must be between 0 and 1")
    elif not _nonempty_string(value):
        errors.append("%s trigger.value must be a non-empty string" % trigger_type)

    return errors


def validate_scenario_collection(scenarios: Sequence[Dict[str, Any]]) -> List[str]:
    """Validate collection-level uniqueness after record validation."""

    errors: List[str] = []
    seen = set()
    for index, scenario in enumerate(scenarios):
        record_errors = validate_scenario(scenario)
        for error in record_errors:
            errors.append("scenario[%d]: %s" % (index, error))
        if not record_errors:
            key = scenario_key(scenario)
            if key in seen:
                errors.append(
                    "scenario[%d]: duplicate (scenario_id, seed): %s, %d"
                    % (index, key[0], key[1])
                )
            seen.add(key)
    return errors


def _json_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if not path.exists():
            raise ScenarioLoadError("path does not exist: %s" % path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.suffix.lower() == ".json":
            files.append(path)
        else:
            raise ScenarioLoadError("expected a JSON file or directory: %s" % path)
    if not files:
        raise ScenarioLoadError("no JSON scenario files found")
    return files


def load_scenarios(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Load scenario objects from JSON files or directories."""

    scenarios: List[Dict[str, Any]] = []
    for path in _json_files(paths):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ScenarioLoadError("failed to load %s: %s" % (path, exc)) from exc

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
            records = payload["scenarios"]
        elif isinstance(payload, dict):
            records = [payload]
        else:
            raise ScenarioLoadError(
                "%s must contain a scenario object, array, or {scenarios: [...]}" % path
            )
        scenarios.extend(records)
    return scenarios
