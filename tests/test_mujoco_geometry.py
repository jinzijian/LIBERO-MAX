import unittest

from libero_max.mujoco_geometry import entity_contact_geom_ids


class _Model:
    ngeom = 8
    _geom_names = (
        "floor",
        "table_collision",
        "table_visual",
        None,
        None,
        None,
        "object_collision",
        "object_visual",
    )
    _body_names = (
        "floor",
        "table",
        "table",
        "living_room_table_col",
        "study_table",
        "other",
        "object",
        "object",
    )
    geom_bodyid = tuple(range(ngeom))
    geom_contype = (1, 1, 0, 1, 1, 1, 1, 0)
    geom_conaffinity = (1, 1, 0, 1, 1, 1, 1, 0)

    def geom_id2name(self, index):
        return self._geom_names[index]

    def body_id2name(self, index):
        return self._body_names[index]

    def geom_name2id(self, name):
        return self._geom_names.index(name)


class _Sim:
    model = _Model()


class _Entity:
    contact_geoms = ("object_collision",)


class _Env:
    sim = _Sim()
    objects_dict = {"object": _Entity()}
    fixtures_dict = {}


class MujocoGeometryTest(unittest.TestCase):
    def test_object_uses_declared_contact_geoms(self):
        self.assertEqual(entity_contact_geom_ids(_Env(), "object"), {6})

    def test_tabletop_aliases_resolve_generic_collision_geom(self):
        self.assertEqual(entity_contact_geom_ids(_Env(), "main_table"), {1})
        self.assertEqual(entity_contact_geom_ids(_Env(), "kitchen_table"), {1})

    def test_unnamed_geoms_resolve_through_body_prefix(self):
        self.assertEqual(entity_contact_geom_ids(_Env(), "living_room_table"), {3})
        self.assertEqual(entity_contact_geom_ids(_Env(), "study_table"), {4})

    def test_unknown_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown entity or support geom"):
            entity_contact_geom_ids(_Env(), "missing_table")


if __name__ == "__main__":
    unittest.main()
