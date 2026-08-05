import unittest

from libero_max.hard import (
    CHANGE_TYPE_ORDER,
    PLUS_CATEGORIES,
    PLUS_DIFFICULTIES,
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


if __name__ == "__main__":
    unittest.main()
