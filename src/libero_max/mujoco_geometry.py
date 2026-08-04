"""Resolve LIBERO entities and BDDL support aliases to collision geoms."""

from typing import Any, Dict, Set, Tuple


# LIBERO's tabletop BDDL files name the logical support by scene, while the
# corresponding robosuite model uses the generic ``table`` body / geom prefix.
_SUPPORT_GEOM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "main_table": ("table",),
    "kitchen_table": ("table",),
}


def _matches_prefix(value: str, candidate: str) -> bool:
    return value == candidate or value.startswith(candidate + "_")


def entity_contact_geom_ids(base_env: Any, name: str) -> Set[int]:
    """Return collision geoms for a LIBERO object, fixture, or scene support.

    Movable objects and fixtures expose ``contact_geoms`` directly. Static
    scene supports such as ``living_room_table`` are sometimes represented by
    unnamed geoms under a named MuJoCo body, so the fallback checks both geom
    and body names. Visual-only geoms are excluded.
    """

    entity = base_env.objects_dict.get(name) or base_env.fixtures_dict.get(name)
    if entity is not None:
        geom_names = getattr(entity, "contact_geoms", None)
        if not geom_names:
            raise ValueError("entity has no contact geoms: %s" % name)
        return {
            int(base_env.sim.model.geom_name2id(geom_name))
            for geom_name in geom_names
        }

    model = base_env.sim.model
    candidates = (name,) + _SUPPORT_GEOM_ALIASES.get(name, ())
    matching: Set[int] = set()
    for index in range(int(model.ngeom)):
        geom_name = model.geom_id2name(index) or ""
        body_id = int(model.geom_bodyid[index])
        body_name = model.body_id2name(body_id) or ""
        if not any(
            _matches_prefix(geom_name, candidate)
            or _matches_prefix(body_name, candidate)
            for candidate in candidates
        ):
            continue
        if int(model.geom_contype[index]) == 0 and int(
            model.geom_conaffinity[index]
        ) == 0:
            continue
        matching.add(index)
    if not matching:
        raise ValueError("unknown entity or support geom: %s" % name)
    return matching
