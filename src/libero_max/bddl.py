"""Small dependency-free BDDL catalog parser for benchmark construction."""

import re
from typing import Any, Dict, List


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


def parse_bddl_metadata(text: str) -> Dict[str, Any]:
    objects = _typed_entities(extract_section(text, "objects"), "objects")
    fixtures = _typed_entities(extract_section(text, "fixtures"), "fixtures")
    interests = _plain_names(
        extract_section(text, "obj_of_interest"), "obj_of_interest"
    )
    goal = extract_section(text, "goal")
    all_entities = sorted(set(objects) | set(fixtures))
    goal_entities = sorted(
        entity
        for entity in all_entities
        if re.search(r"\b%s(?:\b|_)" % re.escape(entity), goal)
    )
    manipulated = set()
    for match in re.finditer(
        r"\(\s*[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z][A-Za-z0-9_]*)",
        goal,
    ):
        first_argument = match.group(1)
        if first_argument in objects:
            manipulated.add(first_argument)
    distractors = sorted(set(objects) - set(goal_entities) - set(interests))
    return {
        "objects": objects,
        "fixtures": fixtures,
        "objects_of_interest": interests,
        "goal_entities": goal_entities,
        "manipulated_objects": sorted(manipulated),
        "distractor_objects": distractors,
    }
