"""Small dependency-free BDDL catalog parser for benchmark construction."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def resolve_libero_bddl_path(
    bddl_root: Path, problem_folder: str, bddl_file: str
) -> Tuple[Path, Dict[str, Any]]:
    """Resolve both materialized LIBERO tasks and LIBERO-Plus virtual tasks.

    LIBERO-Plus represents camera / robot-initial-state variants as virtual
    filenames of the form ``<base>_view_<five values>_initstate_<id>.bddl``.
    Its environment wrapper strips that suffix before loading the BDDL.  The
    benchmark catalog must mirror that behavior instead of incorrectly
    reporting those tasks as missing files.
    """

    requested = bddl_root / problem_folder / bddl_file
    if requested.is_file():
        return requested, {"kind": "materialized"}

    marker = "_view_"
    init_marker = "_initstate_"
    stem = Path(bddl_file).stem
    if marker not in stem or init_marker not in stem:
        raise FileNotFoundError(str(requested))
    base_name, encoded = stem.rsplit(marker, 1)
    view_text, init_text = encoded.rsplit(init_marker, 1)
    view_parts = view_text.split("_")
    if len(view_parts) != 5 or not all(
        re.fullmatch(r"-?\d+(?:\.\d+)?", part) for part in view_parts
    ):
        raise ValueError("invalid LIBERO-Plus virtual view suffix: %s" % bddl_file)
    init_parts = init_text.split("_noise_", 1)
    if not re.fullmatch(r"\d+", init_parts[0]):
        raise ValueError("invalid LIBERO-Plus init-state suffix: %s" % bddl_file)
    noise = 0
    if len(init_parts) == 2:
        if not re.fullmatch(r"\d+", init_parts[1]):
            raise ValueError("invalid LIBERO-Plus noise suffix: %s" % bddl_file)
        noise = int(init_parts[1])

    resolved = bddl_root / problem_folder / (base_name + ".bddl")
    if not resolved.is_file():
        raise FileNotFoundError(
            "%s (virtual LIBERO-Plus task resolved to missing %s)"
            % (requested, resolved)
        )
    horizon, vertical, scale_percent, endpoint_rot, endpoint_vertical = (
        float(part) for part in view_parts
    )
    return resolved, {
        "kind": "libero_plus_virtual",
        "horizon_view_degrees": horizon,
        "vertical_view_degrees": vertical,
        "scale_factor": scale_percent / 100.0,
        "endpoint_rotation_degrees": endpoint_rot,
        "endpoint_vertical_degrees": endpoint_vertical,
        "robot_init_state": int(init_parts[0]),
        "sensor_noise_level": noise,
    }


def extract_section(text: str, name: str) -> str:
    marker = "(:%s" % name.lower()
    start = text.lower().find(marker)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unbalanced BDDL section: %s" % name)


def _typed_entities(section: str, header: str) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    for raw_line in section.splitlines():
        line = raw_line.split(";", 1)[0].strip().strip("()")
        if not line or line.startswith(":%s" % header) or "-" not in line:
            continue
        names_text, entity_type = line.split("-", 1)
        entity_type = entity_type.strip().split()[0]
        for name in names_text.split():
            entities[name] = entity_type
    return entities


def _plain_names(section: str, header: str) -> List[str]:
    if not section:
        return []
    cleaned = section.replace("(:%s" % header, "", 1).replace(")", " ")
    return sorted(set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", cleaned)))


def _atomic_predicates(section: str) -> List[Dict[str, Any]]:
    predicates: List[Dict[str, Any]] = []
    for match in re.finditer(
        r"\(\s*([A-Za-z_][A-Za-z0-9_-]*)\s+([^()]*)\)", section
    ):
        arguments = match.group(2).split()
        if arguments:
            predicates.append(
                {"predicate": match.group(1), "arguments": arguments}
            )
    return predicates


def _resolve_entity(reference: str, entities: List[str]) -> Optional[str]:
    matches = [
        entity
        for entity in entities
        if reference == entity or reference.startswith(entity + "_")
    ]
    return max(matches, key=len) if matches else None


def is_planar_workspace_placement(
    metadata: Dict[str, Any], entity_name: Optional[str]
) -> bool:
    """Return whether an object starts on a large floor/table workspace.

    A generic ``On`` predicate is not enough for relocation: LIBERO also uses
    it for small supports such as a ramekin, stove, cookie box, or cabinet top.
    Translating those objects in XY would create an unsupported pose.
    """

    if not entity_name or entity_name not in metadata.get("objects", {}):
        return False
    placement = metadata.get("initial_placements", {}).get(entity_name)
    if not placement or placement.get("predicate", "").lower() != "on":
        return False
    support = placement.get("support_entity")
    support_type = metadata.get("fixtures", {}).get(support, "").lower()
    return support_type == "floor" or "table" in support_type


def parse_bddl_metadata(text: str) -> Dict[str, Any]:
    objects = _typed_entities(extract_section(text, "objects"), "objects")
    fixtures = _typed_entities(extract_section(text, "fixtures"), "fixtures")
    interests = _plain_names(
        extract_section(text, "obj_of_interest"), "obj_of_interest"
    )
    goal = extract_section(text, "goal")
    init = extract_section(text, "init")
    all_entities = sorted(set(objects) | set(fixtures))
    goal_relations = _atomic_predicates(goal)
    goal_entity_order: List[str] = []
    manipulated: List[str] = []
    for relation in goal_relations:
        for argument in relation["arguments"]:
            entity = _resolve_entity(argument, all_entities)
            if entity is not None and entity not in goal_entity_order:
                goal_entity_order.append(entity)
        first_argument = relation["arguments"][0]
        if first_argument in objects and first_argument not in manipulated:
            manipulated.append(first_argument)

    initial_placements: Dict[str, Dict[str, Any]] = {}
    for relation in _atomic_predicates(init):
        if relation["predicate"].lower() not in {"on", "in"}:
            continue
        if len(relation["arguments"]) < 2:
            continue
        entity_name, region = relation["arguments"][:2]
        if entity_name not in objects:
            continue
        initial_placements[entity_name] = {
            "predicate": relation["predicate"],
            "region": region,
            "support_entity": _resolve_entity(region, all_entities),
        }

    goal_entities = sorted(goal_entity_order)
    distractors = sorted(set(objects) - set(goal_entities) - set(interests))
    return {
        "objects": objects,
        "fixtures": fixtures,
        "objects_of_interest": interests,
        "goal_entities": goal_entities,
        "goal_entity_order": goal_entity_order,
        "goal_relations": goal_relations,
        "manipulated_objects": manipulated,
        "initial_placements": initial_placements,
        "distractor_objects": distractors,
    }
