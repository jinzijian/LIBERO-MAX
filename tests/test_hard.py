import unittest

from libero_max.hard import (
    CHANGE_TYPE_ORDER,
    PLUS_CATEGORIES,
    PLUS_DIFFICULTIES,
    _build_case,
    _core_assignments,
    eligible_change_types,
)


def _task(category, difficulty, index):
    return {
        "task_suite_name": "libero_spatial",
        "task_index": index,
        "task_name": "task_%d" % index,
        "plus_category": category,
        "plus_difficulty_level": difficulty,
        "trigger_entity": "target",
        "primary_target": "target",
        "primary_receptacle": "basket",
        "supports_target_relocation": True,
        "supports_receptacle_relocation": True,
        "distractor_objects": ["d%d" % value for value in range(8)],
        "initial_placements": {
            **{
                "d%d" % value: {"support_entity": "table"}
                for value in range(8)
            },
            "target": {"support_entity": "table"},
            "basket": {"support_entity": "table"},
        },
    }


class HardManifestTest(unittest.TestCase):
    def test_all_eight_events_are_eligible_for_a_rich_task(self):
        self.assertEqual(
            eligible_change_types(_task("Sensor Noise", 5, 0)),
            list(CHANGE_TYPE_ORDER),
        )

    def test_core_is_exactly_balanced_by_stratum_and_event(self):
        tasks = []
        index = 0
        for category in PLUS_CATEGORIES:
            for difficulty in PLUS_DIFFICULTIES:
                for _ in range(48):
                    tasks.append(_task(category, difficulty, index))
                    index += 1
        assignments = _core_assignments(tasks)
        self.assertEqual(len(assignments), 1400)
        for category in PLUS_CATEGORIES:
            for difficulty in PLUS_DIFFICULTIES:
                selected = [
                    event
                    for task in tasks
                    if task["plus_category"] == category
                    and task["plus_difficulty_level"] == difficulty
                    and (event_draw := assignments.get(
                        (task["task_suite_name"], task["task_index"])
                    ))
                    for event in [event_draw[0]]
                ]
                self.assertEqual(len(selected), 40)
                self.assertEqual(
                    {event: selected.count(event) for event in CHANGE_TYPE_ORDER},
                    {event: 5 for event in CHANGE_TYPE_ORDER},
                )

    def test_rejected_task_event_is_not_selected(self):
        tasks = []
        index = 0
        for category in PLUS_CATEGORIES:
            for difficulty in PLUS_DIFFICULTIES:
                for _ in range(48):
                    tasks.append(_task(category, difficulty, index))
                    index += 1
        baseline = _core_assignments(tasks)
        task_key, event_draw = next(iter(baseline.items()))
        repaired = _core_assignments(
            tasks, {(task_key[0], task_key[1], event_draw[0])}
        )
        self.assertNotEqual(repaired.get(task_key), event_draw)
        self.assertEqual(len(repaired), 1400)

    def test_obstacle_uses_a_fixed_target_relative_placement(self):
        case = _build_case(
            _task("Objects Layout", 4, 9), "obstacle_insertion", draw_id=0
        )
        change = case["scenario"]["change"]
        self.assertEqual(change["placement_rule"], "target_approach_ring")
        self.assertEqual(change["relative_to"], "target")
        self.assertTrue(change["preserve_initial_z"])
        self.assertNotIn("path_target", change)


if __name__ == "__main__":
    unittest.main()
