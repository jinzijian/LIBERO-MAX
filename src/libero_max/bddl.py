"""Small dependency-free BDDL catalog parser for benchmark construction."""

import re
from typing import Any, Dict, List, Optional


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
