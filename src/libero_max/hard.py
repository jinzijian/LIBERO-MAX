"""Deterministic LIBERO-MAX Hard construction on the LIBERO-Plus substrate."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .calibration import CANONICAL_DIRECTIONS
from .manifest import validate_manifest


SAMPLER_VERSION = "libero-max-hard-v2.0"
PLUS_CATEGORIES = (
    "Background Textures",
    "Robot Initial States",
    "Camera Viewpoints",
    "Language Instructions",
    "Sensor Noise",
    "Objects Layout",
    "Light Conditions",
)
PLUS_DIFFICULTIES = (1, 2, 3, 4, 5)
CHANGE_TYPE_ORDER = (
    "illumination_switch",
    "camera_shift",
    "visual_theme_switch",
    "sensor_noise_onset",
    "target_relocation",
    "receptacle_relocation",
    "distractor_burst",
    "obstacle_insertion",
)
CORE_PER_STRATUM = 40
CORE_PER_EVENT_PER_STRATUM = CORE_PER_STRATUM // len(CHANGE_TYPE_ORDER)
MAX5600_PER_CATEGORY = 800
MAX5600_PER_EVENT_DRAW_CATEGORY = 50


class HardBuildError(ValueError):
    """Raised when a Plus catalog cannot satisfy the Hard design contract."""


class _FlowEdge:
    def __init__(self, target: int, reverse: int, capacity: int):
        self.target = target
        self.reverse = reverse
        self.capacity = capacity


def _add_flow_edge(graph, source: int, target: int, capacity: int) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity)
    reverse = _FlowEdge(source, len(graph[source]), 0)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _max_flow(graph, source: int, sink: int) -> int:
    """Compute deterministic integral max flow with Dinic's algorithm."""

    total = 0
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue = [source]
        for node in queue:
            for edge in graph[node]:
                if edge.capacity and levels[edge.target] < 0:
                    levels[edge.target] = levels[node] + 1
                    queue.append(edge.target)
        if levels[sink] < 0:
            return total
        offsets = [0] * len(graph)

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while offsets[node] < len(graph[node]):
                edge = graph[node][offsets[node]]
                if edge.capacity and levels[edge.target] == levels[node] + 1:
                    pushed = send(edge.target, min(amount, edge.capacity))
                    if pushed:
                        edge.capacity -= pushed
                        graph[edge.target][edge.reverse].capacity += pushed
                        return pushed
                offsets[node] += 1
            return 0

        while True:
            pushed = send(source, 10**9)
            if not pushed:
                break
            total += pushed


def _stable_seed(parts: Iterable[Any]) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _task_key(task: Dict[str, Any]) -> Tuple[str, int]:
    return task["task_suite_name"], task["task_index"]


def _physical_scene_signature(task: Dict[str, Any]) -> str:
    """Identify Plus variants that share the same physical task layout."""

    payload = {
        "fixtures": task.get("fixtures"),
        "objects": task.get("objects"),
        "initial_placements": task.get("initial_placements"),
        "trigger_entity": task.get("trigger_entity"),
        "primary_target": task.get("primary_target"),
        "primary_receptacle": task.get("primary_receptacle"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def expand_rejected_by_physical_scene(
    tasks: Sequence[Dict[str, Any]],
    rejected: Set[Tuple[str, int, str]],
    change_types: Iterable[str],
) -> Set[Tuple[str, int, str]]:
    """Propagate failed physical placements across equivalent Plus variants."""

    expanded = set(rejected)
    selected_types = set(change_types)
    if not selected_types:
        return expanded
    tasks_by_key = {_task_key(task): task for task in tasks}
    failed_signatures = {
        (_physical_scene_signature(tasks_by_key[(suite, task_index)]), event)
        for suite, task_index, event in rejected
        if event in selected_types and (suite, task_index) in tasks_by_key
    }
    for task in tasks:
        signature = _physical_scene_signature(task)
        for event in selected_types:
            if (signature, event) in failed_signatures:
                expanded.add((*_task_key(task), event))
    return expanded


def _supported_distractors(task: Dict[str, Any]) -> List[str]:
    placements = task.get("initial_placements", {})
    return sorted(
        entity
        for entity in task.get("distractor_objects", [])
        if placements.get(entity, {}).get("support_entity")
    )


def _supported_obstacles(task: Dict[str, Any]) -> List[str]:
    """Return distractors that share the target's planar support."""

    placements = task.get("initial_placements", {})
    trigger = task.get("trigger_entity")
    trigger_support = placements.get(trigger, {}).get("support_entity")
    if not trigger_support:
        return []
    return sorted(
        entity
        for entity in task.get("distractor_objects", [])
        if placements.get(entity, {}).get("support_entity") == trigger_support
    )


def eligible_change_types(task: Dict[str, Any]) -> List[str]:
    if not task.get("trigger_entity"):
        return []
    eligible = list(CHANGE_TYPE_ORDER[:4])
    if task.get("supports_target_relocation"):
        eligible.append("target_relocation")
    if task.get("supports_receptacle_relocation"):
        eligible.append("receptacle_relocation")
    supported_distractors = _supported_distractors(task)
    if len(supported_distractors) >= 5:
        eligible.append("distractor_burst")
    if _supported_obstacles(task):
        eligible.append("obstacle_insertion")
    return [name for name in CHANGE_TYPE_ORDER if name in eligible]


def _support_for(task: Dict[str, Any], entity: str) -> str:
    support = task.get("initial_placements", {}).get(entity, {}).get(
        "support_entity"
    )
    if not isinstance(support, str) or not support:
        raise HardBuildError("missing initial support for %s" % entity)
    return support


def _relocation_direction(
    task: Dict[str, Any], change_type: str
) -> Tuple[Tuple[float, float], str]:
    calibrated = task.get("relocation_directions", {}).get(change_type)
    if isinstance(calibrated, list) and len(calibrated) == 2:
        x, y = float(calibrated[0]), float(calibrated[1])
        norm = math.hypot(x, y)
        if math.isfinite(norm) and norm > 1e-9:
            return (x / norm, y / norm), "catalog_validated_fixed_direction"
    index = _stable_seed(
        (SAMPLER_VERSION, *_task_key(task), change_type, "direction")
    ) % len(CANONICAL_DIRECTIONS)
    return CANONICAL_DIRECTIONS[index], "deterministic_candidate_requires_preflight"


def _round_vector(x: float, y: float, z: float = 0.0) -> List[float]:
    return [round(x, 8), round(y, 8), round(z, 8)]


def _sample_change(
    task: Dict[str, Any], change_type: str, draw_id: int, seed: int
) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    if change_type == "illumination_switch":
        if draw_id == 0:
            return "OBS", "high", [], {"operation": "set_lighting", "scale": 0.20}
        return "OBS", "high", [], {"operation": "set_lighting", "scale": 1.80}
    if change_type == "camera_shift":
        position = ([0.06, 0.0, 0.02], [-0.06, 0.03, 0.0])[draw_id]
        yaw = (18.0, -22.0)[draw_id]
        fovy = (12.0, -12.0)[draw_id]
        return "OBS", "high", [], {
            "operation": "shift_camera",
            "camera": "agentview",
            "delta_position_m": position,
            "yaw_degrees": yaw,
            "delta_fovy_degrees": fovy,
            "calibration": "fixed_global_viewpoint",
        }
    if change_type == "visual_theme_switch":
        permutations = ([2, 0, 1], [1, 2, 0])
        multipliers = ([0.70, 1.15, 1.30], [1.25, 0.70, 1.10])
        return "OBS", "high", [], {
            "operation": "set_visual_theme",
            "rgb_permutation": permutations[draw_id],
            "rgb_multiplier": multipliers[draw_id],
        }
    if change_type == "sensor_noise_onset":
        return "OBS", "high", [], {
            "operation": "set_sensor_corruption",
            "noise_std": (24.0, 36.0)[draw_id],
            "occlusion_fraction": (0.08, 0.16)[draw_id],
            "seed": seed,
        }
    if change_type in {"target_relocation", "receptacle_relocation"}:
        role = "primary_target" if change_type == "target_relocation" else "primary_receptacle"
        entity = task.get(role)
        if not entity:
            raise HardBuildError("%s requires %s" % (change_type, role))
        distance = (0.06, 0.12)[draw_id]
        direction, calibration = _relocation_direction(task, change_type)
        return "GEO", ("medium", "high")[draw_id], [], {
            "operation": "move_object",
            "object": entity,
            "delta_position_m": _round_vector(
                direction[0] * distance, direction[1] * distance
            ),
            "support_entity": _support_for(task, entity),
            "distance_m": distance,
            "direction_xy": [round(direction[0], 8), round(direction[1], 8)],
            "calibration": calibration,
        }
    distractors = _supported_distractors(task)
    rng.shuffle(distractors)
    if change_type == "distractor_burst":
        count = min(len(distractors), (5, 8)[draw_id])
        selected = distractors[:count]
        setup = [
            {
                "operation": "remove_object",
                "object": entity,
                "offworld_position_m": [-2.0 - 0.2 * index, 0.0, -5.0],
            }
            for index, entity in enumerate(selected)
        ]
        phase = rng.uniform(0.0, 2.0 * math.pi)
        placements = []
        for index, entity in enumerate(selected):
            angle = phase + index * 2.0 * math.pi / count
            radius = rng.uniform(0.09, 0.17)
            placements.append(
                {
                    "object": entity,
                    "relative_to": task["trigger_entity"],
                    "offset_m": _round_vector(
                        radius * math.cos(angle), radius * math.sin(angle)
                    ),
                    "preserve_initial_z": True,
                    "support_entity": _support_for(task, entity),
                }
            )
        return "CLUTTER", "high", setup, {
            "operation": "insert_distractors",
            "placements": placements,
            "distractor_count": count,
        }
    if change_type == "obstacle_insertion":
        obstacles = _supported_obstacles(task)
        obstacle = obstacles[draw_id % len(obstacles)]
        setup = [
            {
                "operation": "remove_object",
                "object": obstacle,
                "offworld_position_m": [-2.0, 0.0, -5.0],
            }
        ]
        return "OBSTACLE", "high", setup, {
            "operation": "insert_obstacle",
            "object": obstacle,
            "relative_to": task["trigger_entity"],
            "offset_m": ([0.14, 0.0, 0.0], [0.0, -0.14, 0.0])[draw_id],
            "preserve_initial_z": True,
            "placement_rule": "target_support_approach_ring",
            "support_entity": _support_for(task, obstacle),
        }
    raise AssertionError("unhandled change type: %s" % change_type)


def _build_case(
    task: Dict[str, Any], change_type: str, draw_id: Optional[int] = None
) -> Dict[str, Any]:
    if change_type not in eligible_change_types(task):
        raise HardBuildError("task is not eligible for %s" % change_type)
    if draw_id is None:
        draw_id = _stable_seed(
            (SAMPLER_VERSION, *_task_key(task), change_type, "draw")
        ) % 2
    seed = _stable_seed(
        (SAMPLER_VERSION, *_task_key(task), change_type, draw_id)
    )
    family, severity, setup, change = _sample_change(
        task, change_type, draw_id, seed
    )
    suite, task_index = _task_key(task)
    scenario_id = "%s-t%04d-%s-d%d" % (
        suite,
        task_index,
        change_type,
        draw_id,
    )
    scenario = {
        "scenario_id": scenario_id,
        "base_task_id": "%s/task_%d" % (suite, task_index),
        "seed": seed,
        "change_family": family,
        "change_type": change_type,
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
            "sampler": SAMPLER_VERSION,
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
        "init_state_index": 0,
        "policy_seed": 195,
        "timing_bucket": "middle",
        "task_name": task["task_name"],
        "substrate_category": task["plus_category"],
        "substrate_difficulty": task["plus_difficulty_level"],
        "dynamic_phase": "pre_grasp_proximity",
        "scenario": scenario,
    }


def _stratum_candidates(
    tasks: Sequence[Dict[str, Any]],
    stratum: Tuple[str, int],
    event: str,
    rejected: Set[Tuple[str, int, str]],
    sampler_label: str,
) -> List[Dict[str, Any]]:
    candidates = [
        task
        for task in tasks
        if event in eligible_change_types(task)
        and (*_task_key(task), event) not in rejected
    ]
    candidates.sort(
        key=lambda task: _stable_seed(
            (SAMPLER_VERSION, sampler_label, *stratum, event, *_task_key(task))
        )
    )
    return candidates


def _match_stratum(
    tasks: Sequence[Dict[str, Any]],
    stratum: Tuple[str, int],
    quotas: Dict[str, int],
    rejected: Set[Tuple[str, int, str]],
) -> Optional[Dict[Tuple[str, int], Tuple[str, int]]]:
    """Find a deterministic task-to-event matching for one 40-task cell."""

    if sum(quotas.values()) != CORE_PER_STRATUM:
        raise AssertionError("Core stratum quotas must sum to 40")
    candidates_by_event = {
        event: _stratum_candidates(
            tasks, stratum, event, rejected, "core"
        )
        for event in CHANGE_TYPE_ORDER
    }
    if any(
        len(candidates_by_event[event]) < quotas[event]
        for event in CHANGE_TYPE_ORDER
    ):
        return None
    slots = [
        (event, rank)
        for event in CHANGE_TYPE_ORDER
        for rank in range(quotas[event])
    ]
    slots.sort(
        key=lambda slot: (
            len(candidates_by_event[slot[0]]) - quotas[slot[0]],
            CHANGE_TYPE_ORDER.index(slot[0]),
            slot[1],
        )
    )
    task_to_slot = {}
    slot_to_task = {}

    def assign(slot, seen_tasks):
        event, _ = slot
        for task in candidates_by_event[event]:
            key = _task_key(task)
            if key in seen_tasks:
                continue
            seen_tasks.add(key)
            previous = task_to_slot.get(key)
            if previous is None or assign(previous, seen_tasks):
                task_to_slot[key] = slot
                slot_to_task[slot] = task
                return True
        return False

    for slot in slots:
        if not assign(slot, set()):
            return None
    return {
        _task_key(task): (event, rank % 2)
        for (event, rank), task in slot_to_task.items()
    }


def _core_assignments(
    tasks: Sequence[Dict[str, Any]],
    rejected: Set[Tuple[str, int, str]] = frozenset(),
) -> Dict[Tuple[str, int], Tuple[str, int]]:
    strata: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        key = (task.get("plus_category"), task.get("plus_difficulty_level"))
        if key[0] in PLUS_CATEGORIES and key[1] in PLUS_DIFFICULTIES:
            strata[key].append(task)
    expected = {
        (category, difficulty)
        for category in PLUS_CATEGORIES
        for difficulty in PLUS_DIFFICULTIES
    }
    if set(strata) != expected:
        raise HardBuildError("catalog does not cover all 7 x 5 Plus strata")

    quotas = {
        stratum: {
            event: CORE_PER_EVENT_PER_STRATUM for event in CHANGE_TYPE_ORDER
        }
        for stratum in strata
    }
    # Feasibility filtering can leave a particular Plus cell with only four
    # safe configurations for a scarce event. Preserve the exact global 175
    # cases per event by swapping one quota with a donor cell (4/6 and 6/4),
    # rather than relaxing the Core's event balance or accepting a bad scene.
    seen_quota_states = set()
    while True:
        matches = {
            stratum: _match_stratum(
                strata[stratum], stratum, quotas[stratum], rejected
            )
            for stratum in sorted(strata)
        }
        failed = [stratum for stratum in sorted(strata) if matches[stratum] is None]
        if not failed:
            break
        state = tuple(
            (stratum, tuple(quotas[stratum][event] for event in CHANGE_TYPE_ORDER))
            for stratum in sorted(strata)
        )
        if state in seen_quota_states:
            raise HardBuildError("Core quota repair entered a cycle")
        seen_quota_states.add(state)
        stratum = failed[0]
        availability = {
            event: len(
                _stratum_candidates(
                    strata[stratum], stratum, event, rejected, "core"
                )
            )
            for event in CHANGE_TYPE_ORDER
        }
        scarce_events = sorted(
            CHANGE_TYPE_ORDER,
            key=lambda event: (
                availability[event] - quotas[stratum][event],
                CHANGE_TYPE_ORDER.index(event),
            ),
        )
        repaired = False
        donors = sorted(
            (candidate for candidate in strata if candidate != stratum),
            key=lambda candidate: _stable_seed(
                (SAMPLER_VERSION, "core-quota-donor", *stratum, *candidate)
            ),
        )
        for scarce in scarce_events:
            if quotas[stratum][scarce] <= 0:
                continue
            minimum_shift = max(
                1, quotas[stratum][scarce] - availability[scarce]
            )
            for replacement in CHANGE_TYPE_ORDER[:4]:
                if replacement == scarce:
                    continue
                for shift in range(
                    minimum_shift, quotas[stratum][scarce] + 1
                ):
                    local_trial = dict(quotas[stratum])
                    local_trial[scarce] -= shift
                    local_trial[replacement] += shift
                    if _match_stratum(
                        strata[stratum], stratum, local_trial, rejected
                    ) is None:
                        continue
                    for donor in donors:
                        if quotas[donor][replacement] < shift:
                            continue
                        donor_trial = dict(quotas[donor])
                        donor_trial[scarce] += shift
                        donor_trial[replacement] -= shift
                        if _match_stratum(
                            strata[donor], donor, donor_trial, rejected
                        ) is None:
                            continue
                        quotas[stratum] = local_trial
                        quotas[donor] = donor_trial
                        repaired = True
                        break
                    if repaired:
                        break
                if repaired:
                    break
            if repaired:
                break
        if not repaired:
            raise HardBuildError(
                "%s cannot be rebalanced after feasibility rejection"
                % (stratum,)
            )

    assignments: Dict[Tuple[str, int], Tuple[str, int]] = {}
    for stratum in sorted(strata):
        stratum_assignments = matches[stratum]
        if stratum_assignments is None or len(stratum_assignments) != CORE_PER_STRATUM:
            raise AssertionError("core stratum does not contain exactly 40 tasks")
        assignments.update(stratum_assignments)
    counts = Counter(event for event, _ in assignments.values())
    if any(counts[event] != 175 for event in CHANGE_TYPE_ORDER):
        raise AssertionError("Core event quotas are not globally balanced")
    return assignments


def _full_assignments(
    tasks: Sequence[Dict[str, Any]],
    core: Dict[Tuple[str, int], Tuple[str, int]],
    rejected: Set[Tuple[str, int, str]] = frozenset(),
) -> Dict[Tuple[str, int], Tuple[str, Optional[int]]]:
    assignments: Dict[Tuple[str, int], Tuple[str, Optional[int]]] = dict(core)
    counts = Counter(change_type for change_type, _ in assignments.values())
    remaining = [task for task in tasks if _task_key(task) not in assignments]
    remaining.sort(
        key=lambda task: _stable_seed((SAMPLER_VERSION, "full", *_task_key(task)))
    )
    for task in remaining:
        task_had_failure = any(
            suite == task["task_suite_name"] and task_index == task["task_index"]
            for suite, task_index, _ in rejected
        )
        eligible = [
            event
            for event in eligible_change_types(task)
            if (*_task_key(task), event) not in rejected
            and (not task_had_failure or event in CHANGE_TYPE_ORDER[:4])
        ]
        if not eligible:
            raise HardBuildError("task %s has no dynamic event" % (_task_key(task),))
        minimum = min(counts[event] for event in eligible)
        tied = [event for event in eligible if counts[event] == minimum]
        tied.sort(
            key=lambda event: _stable_seed(
                (SAMPLER_VERSION, "tie", *_task_key(task), event)
            )
        )
        event = tied[0]
        assignments[_task_key(task)] = (event, None)
        counts[event] += 1
    return assignments


def _balanced_difficulty_quotas(
    tasks: Sequence[Dict[str, Any]], total: int
) -> Dict[int, int]:
    """Allocate a near-uniform difficulty quota without exceeding the catalog."""

    capacity = Counter(
        task.get("plus_difficulty_level")
        for task in tasks
        if task.get("plus_difficulty_level") in PLUS_DIFFICULTIES
    )
    base, remainder = divmod(total, len(PLUS_DIFFICULTIES))
    quotas = {
        difficulty: min(
            capacity[difficulty],
            base + int(index < remainder),
        )
        for index, difficulty in enumerate(PLUS_DIFFICULTIES)
    }
    while sum(quotas.values()) < total:
        candidates = [
            difficulty
            for difficulty in PLUS_DIFFICULTIES
            if quotas[difficulty] < capacity[difficulty]
        ]
        if not candidates:
            raise HardBuildError("category cannot supply its MAX-5600 quota")
        difficulty = min(
            candidates,
            key=lambda value: (quotas[value], PLUS_DIFFICULTIES.index(value)),
        )
        quotas[difficulty] += 1
    return quotas


def _balanced_category_assignments(
    tasks: Sequence[Dict[str, Any]],
    category: str,
    frozen: Dict[Tuple[str, int], Tuple[str, int]],
    rejected: Set[Tuple[str, int, str]],
    per_event_draw: int = MAX5600_PER_EVENT_DRAW_CATEGORY,
) -> Tuple[Dict[Tuple[str, int], Tuple[str, int]], Dict[int, int]]:
    """Fill one category using exact event/draw and bounded difficulty quotas."""

    category_tasks = [
        task
        for task in tasks
        if task.get("plus_category") == category
        and task.get("plus_difficulty_level") in PLUS_DIFFICULTIES
    ]
    category_total = len(CHANGE_TYPE_ORDER) * 2 * per_event_draw
    difficulty_quotas = _balanced_difficulty_quotas(category_tasks, category_total)
    tasks_by_key = {_task_key(task): task for task in category_tasks}
    frozen_here = {
        key: assignment
        for key, assignment in frozen.items()
        if key in tasks_by_key
    }
    if len(frozen_here) > category_total:
        raise HardBuildError("frozen Core exceeds a MAX-5600 category quota")

    frozen_by_difficulty = Counter(
        tasks_by_key[key]["plus_difficulty_level"] for key in frozen_here
    )
    remaining_by_difficulty = {
        difficulty: difficulty_quotas[difficulty]
        - frozen_by_difficulty[difficulty]
        for difficulty in PLUS_DIFFICULTIES
    }
    if any(value < 0 for value in remaining_by_difficulty.values()):
        raise HardBuildError("frozen Core exceeds a MAX-5600 difficulty quota")

    frozen_by_bucket = Counter(frozen_here.values())
    buckets = [
        (event, draw)
        for event in CHANGE_TYPE_ORDER
        for draw in (0, 1)
    ]
    remaining_by_bucket = {
        bucket: per_event_draw - frozen_by_bucket[bucket]
        for bucket in buckets
    }
    if any(value < 0 for value in remaining_by_bucket.values()):
        raise HardBuildError("frozen Core exceeds a MAX-5600 event/draw quota")
    required = category_total - len(frozen_here)
    if sum(remaining_by_difficulty.values()) != required or sum(
        remaining_by_bucket.values()
    ) != required:
        raise AssertionError("MAX-5600 residual quotas disagree")

    candidates = [
        task for task in category_tasks if _task_key(task) not in frozen_here
    ]
    candidates.sort(
        key=lambda task: _stable_seed(
            (SAMPLER_VERSION, "max5600", category, *_task_key(task))
        )
    )
    graph = []

    def node() -> int:
        graph.append([])
        return len(graph) - 1

    source = node()
    difficulty_nodes = {difficulty: node() for difficulty in PLUS_DIFFICULTIES}
    task_nodes = {_task_key(task): node() for task in candidates}
    bucket_nodes = {bucket: node() for bucket in buckets}
    sink = node()
    for difficulty in PLUS_DIFFICULTIES:
        _add_flow_edge(
            graph,
            source,
            difficulty_nodes[difficulty],
            remaining_by_difficulty[difficulty],
        )
    for task in candidates:
        key = _task_key(task)
        _add_flow_edge(
            graph,
            difficulty_nodes[task["plus_difficulty_level"]],
            task_nodes[key],
            1,
        )
    for bucket in buckets:
        _add_flow_edge(
            graph, bucket_nodes[bucket], sink, remaining_by_bucket[bucket]
        )

    assignment_edges = {}
    for task in candidates:
        key = _task_key(task)
        eligible = [
            event
            for event in eligible_change_types(task)
            if (*key, event) not in rejected
        ]
        task_buckets = [
            (event, draw) for event in eligible for draw in (0, 1)
        ]
        task_buckets.sort(
            key=lambda bucket: _stable_seed(
                (SAMPLER_VERSION, "max5600-edge", category, *key, *bucket)
            )
        )
        for bucket in task_buckets:
            edge = _add_flow_edge(
                graph, task_nodes[key], bucket_nodes[bucket], 1
            )
            assignment_edges[(key, bucket)] = edge

    achieved = _max_flow(graph, source, sink)
    if achieved != required:
        raise HardBuildError(
            "%s MAX-5600 matching reached %d/%d" % (category, achieved, required)
        )
    selected = dict(frozen_here)
    for (key, bucket), edge in assignment_edges.items():
        if edge.capacity == 0:
            if key in selected:
                raise AssertionError("MAX-5600 selected a task twice")
            selected[key] = bucket
    if len(selected) != category_total:
        raise AssertionError("MAX-5600 category cardinality is incorrect")
    return selected, difficulty_quotas


def _max5600_assignments(
    tasks: Sequence[Dict[str, Any]],
    frozen_core: Dict[str, Any],
    rejected: Set[Tuple[str, int, str]],
) -> Tuple[Dict[Tuple[str, int], Tuple[str, int]], Dict[str, Dict[int, int]]]:
    frozen = _frozen_core_assignments(frozen_core, tasks)
    assignments = {}
    difficulty_quotas = {}
    for category in PLUS_CATEGORIES:
        selected, quotas = _balanced_category_assignments(
            tasks, category, frozen, rejected
        )
        assignments.update(selected)
        difficulty_quotas[category] = quotas
    if len(assignments) != 5600:
        raise AssertionError("MAX-5600 must contain exactly 5,600 assignments")
    category_counts = Counter(
        next(task for task in tasks if _task_key(task) == key)["plus_category"]
        for key in assignments
    )
    event_draw_counts = Counter(assignments.values())
    if any(category_counts[category] != 800 for category in PLUS_CATEGORIES):
        raise AssertionError("MAX-5600 category quotas are not balanced")
    if any(
        event_draw_counts[(event, draw)] != 350
        for event in CHANGE_TYPE_ORDER
        for draw in (0, 1)
    ):
        raise AssertionError("MAX-5600 event/draw quotas are not balanced")
    return assignments, difficulty_quotas


def _manifest(
    profile: str,
    cases: List[Dict[str, Any]],
    *,
    benchmark_id: Optional[str] = None,
    selection_contract: Optional[str] = None,
    protocol_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "benchmark_id": benchmark_id or "libero-max-hard-%s" % profile,
        "benchmark_version": "2.0.0-candidate",
        "protocol": {
            "arms": ["control", "intervention"],
            "query_interval": 16,
            "scoring_track": "physical_completion",
            "substrate": "LIBERO-Plus",
            "profile": profile,
            "source_benchmark_commit": "4976dc3",
            "selection_contract": selection_contract
            or (
                "7 categories x 5 difficulty levels x 40 tasks"
                if profile == "core"
                else "all 10030 LIBERO-Plus tasks"
            ),
        },
        "cases": sorted(
            cases, key=lambda case: (case["task_suite_name"], case["task_index"])
        ),
    }
    if protocol_metadata:
        manifest["protocol"].update(protocol_metadata)
    errors = validate_manifest(manifest)
    if errors:
        raise HardBuildError("generated %s manifest is invalid: %s" % (profile, "; ".join(errors)))
    return manifest


def build_max5600_manifest(
    catalog: Dict[str, Any],
    frozen_core: Dict[str, Any],
    rejected_configurations: Iterable[Tuple[str, int, str]] = (),
) -> Dict[str, Any]:
    """Build the single official 5,600-pair LIBERO-MAX benchmark."""

    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10030:
        raise HardBuildError("MAX-5600 requires the exact 10030-task Plus catalog")
    if len({_task_key(task) for task in tasks}) != len(tasks):
        raise HardBuildError("catalog contains duplicate suite/task keys")
    rejected = set(rejected_configurations)
    assignments, difficulty_quotas = _max5600_assignments(
        tasks, frozen_core, rejected
    )
    tasks_by_key = {_task_key(task): task for task in tasks}
    cases = [
        _build_case(tasks_by_key[key], event, draw)
        for key, (event, draw) in assignments.items()
    ]
    return _manifest(
        "official",
        cases,
        benchmark_id="libero-max-5600",
        selection_contract=(
            "7 Plus categories x 800; 8 changes x 700; "
            "2 deterministic draws x 350 per change"
        ),
        protocol_metadata={
            "official_name": "LIBERO-MAX",
            "official_pairs": 5600,
            "frozen_core_pairs": 1400,
            "difficulty_quotas_by_category": difficulty_quotas,
        },
    )


def _frozen_core_assignments(
    core: Dict[str, Any], tasks: Sequence[Dict[str, Any]]
) -> Dict[Tuple[str, int], Tuple[str, int]]:
    errors = validate_manifest(core)
    if errors:
        raise HardBuildError("frozen Core is invalid: %s" % "; ".join(errors))
    if len(core.get("cases", [])) != 1400:
        raise HardBuildError("frozen Core must contain exactly 1400 cases")
    tasks_by_key = {_task_key(task): task for task in tasks}
    assignments = {}
    for case in core["cases"]:
        key = (case["task_suite_name"], case["task_index"])
        if key in assignments or key not in tasks_by_key:
            raise HardBuildError("frozen Core has an unknown or duplicate task")
        event = case["scenario"]["change_type"]
        draw = case["scenario"].get("randomization", {}).get("draw_id")
        if event not in eligible_change_types(tasks_by_key[key]) or draw not in {
            0,
            1,
        }:
            raise HardBuildError("frozen Core contains an ineligible assignment")
        rebuilt = _build_case(tasks_by_key[key], event, draw)
        if rebuilt["case_id"] != case["case_id"]:
            raise HardBuildError("frozen Core assignment is not reproducible")
        assignments[key] = (event, draw)
    event_counts = Counter(event for event, _ in assignments.values())
    if any(event_counts[event] != 175 for event in CHANGE_TYPE_ORDER):
        raise HardBuildError("frozen Core is not globally event-balanced")
    return assignments


def build_hard_manifests(
    catalog: Dict[str, Any],
    rejected_configurations: Iterable[Tuple[str, int, str]] = (),
    frozen_core: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10030:
        raise HardBuildError("Hard Full requires the exact 10030-task Plus catalog")
    if len({_task_key(task) for task in tasks}) != len(tasks):
        raise HardBuildError("catalog contains duplicate suite/task keys")
    rejected = set(rejected_configurations)
    core_assignments = (
        _core_assignments(tasks, rejected)
        if frozen_core is None
        else _frozen_core_assignments(frozen_core, tasks)
    )
    full_assignments = _full_assignments(tasks, core_assignments, rejected)
    full_cases = []
    core_cases = []
    for task in tasks:
        key = _task_key(task)
        event, draw = full_assignments[key]
        case = _build_case(task, event, draw)
        full_cases.append(case)
        if key in core_assignments:
            core_cases.append(case)
    if len(core_cases) != 1400 or len(full_cases) != 10030:
        raise AssertionError("Hard manifest cardinality contract failed")
    return _manifest("core", core_cases), _manifest("full", full_cases)


def hard_manifest_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    cases = manifest["cases"]
    by_event = Counter(case["scenario"]["change_type"] for case in cases)
    by_category = Counter(case["substrate_category"] for case in cases)
    by_difficulty = Counter(str(case["substrate_difficulty"]) for case in cases)
    calibration = Counter(
        case["scenario"]["change"].get("calibration", "not_required")
        for case in cases
    )
    return {
        "benchmark_id": manifest["benchmark_id"],
        "matched_pairs": len(cases),
        "episodes": 2 * len(cases),
        "pairs_by_change_type": dict(sorted(by_event.items())),
        "pairs_by_plus_category": dict(sorted(by_category.items())),
        "pairs_by_plus_difficulty": dict(sorted(by_difficulty.items())),
        "relocation_calibration_status": dict(sorted(calibration.items())),
    }
