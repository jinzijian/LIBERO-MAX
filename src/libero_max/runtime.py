"""Exactly-once intervention scheduling independent of any simulator."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol

from .scenario import validate_scenario


EVENT_PREFIXES = {
    "before_grasp": "pregrasp",
    "after_grasp": "grasp",
    "after_subgoal": "subgoal",
    "on_region_entry": "region",
}
INTENT_OPERATIONS = {"replace_instruction", "cancel_instruction"}


class ChangeBackend(Protocol):
    def apply_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one physical change and return JSON-serializable evidence."""


@dataclass(frozen=True)
class TriggerContext:
    step: int
    max_steps: int
    events: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


def trigger_satisfied(trigger: Dict[str, Any], context: TriggerContext) -> bool:
    trigger_type = trigger["type"]
    value = trigger["value"]
    if trigger_type == "fixed_step":
        return context.step >= value
    if trigger_type == "progress_fraction":
        return context.step / context.max_steps >= value
    prefix = EVENT_PREFIXES[trigger_type]
    return "%s:%s" % (prefix, value) in context.events


class InterventionRuntime:
    """Schedule and trace one intervention for one scenario episode."""

    def __init__(self, scenario: Dict[str, Any], backend: ChangeBackend):
        errors = validate_scenario(scenario)
        if errors:
            raise ValueError("invalid scenario: " + "; ".join(errors))
        self.scenario = scenario
        self.backend = backend
        self.trace: List[Dict[str, Any]] = []
        self.setup_trace: List[Dict[str, Any]] = []
        self.applied = False
        self.current_instruction = ""

    def reset(self, instruction: str) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        self.trace = []
        self.setup_trace = []
        self.applied = False
        self.current_instruction = instruction

    def apply_setup(self) -> List[Dict[str, Any]]:
        """Apply deterministic pre-episode setup shared by both paired arms."""

        if not self.current_instruction:
            raise RuntimeError("reset() must be called before apply_setup()")
        if self.setup_trace:
            raise RuntimeError("scenario setup was already applied")
        for change in self.scenario.get("setup", []):
            self.setup_trace.append(
                {"change": change, "backend_result": self.backend.apply_change(change)}
            )
        return list(self.setup_trace)

    def maybe_apply(self, context: TriggerContext) -> Optional[Dict[str, Any]]:
        if self.applied or not trigger_satisfied(self.scenario["trigger"], context):
            return None
        if not self.current_instruction:
            raise RuntimeError("reset() must be called before maybe_apply()")

        change = self.scenario["change"]
        instruction_before = self.current_instruction
        operation = change["operation"]
        if operation in INTENT_OPERATIONS:
            backend_result = {"operation": operation, "simulator_changed": False}
            instruction_after = change.get("instruction")
            if not isinstance(instruction_after, str) or not instruction_after.strip():
                if operation == "cancel_instruction":
                    instruction_after = "Stop. The task has been cancelled."
                else:
                    raise ValueError("replace_instruction requires change.instruction")
            self.current_instruction = instruction_after
        else:
            backend_result = self.backend.apply_change(change)

        event = {
            "scenario_id": self.scenario["scenario_id"],
            "seed": self.scenario["seed"],
            "step": context.step,
            "trigger": self.scenario["trigger"],
            "change": change,
            "expected_response_mode": self.scenario["expected_response_mode"],
            "instruction_before": instruction_before,
            "instruction_after": self.current_instruction,
            "backend_result": backend_result,
        }
        self.applied = True
        self.trace.append(event)
        return event
