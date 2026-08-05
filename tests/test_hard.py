import unittest

from libero_max.hard import (
    CHANGE_TYPE_ORDER,
    PLUS_CATEGORIES,
    PLUS_DIFFICULTIES,
    _build_case,
    _core_assignments,
    _full_assignments,
    _frozen_core_assignments,
    _manifest,
    eligible_change_types,
    expand_rejected_by_physical_scene,
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

    def test_scarce_cell_is_rebalanced_without_changing_global_event_quota(self):
        tasks = []
        index = 0
        scarce_keys = []
        for category in PLUS_CATEGORIES:
            for difficulty in PLUS_DIFFICULTIES:
                for _ in range(48):
                    task = _task(category, difficulty, index)
                    tasks.append(task)
                    if category == PLUS_CATEGORIES[0] and difficulty == 1:
                        scarce_keys.append((task["task_suite_name"], index))
                    index += 1
        rejected = {
            (suite, task_index, "distractor_burst")
            for suite, task_index in scarce_keys[2:]
        }
        assignments = _core_assignments(tasks, rejected)
        counts = {
            event: sum(draw[0] == event for draw in assignments.values())
            for event in CHANGE_TYPE_ORDER
        }
        scarce_distractors = sum(
            assignments.get(key, (None, None))[0] == "distractor_burst"
            for key in scarce_keys
        )
        self.assertEqual(len(assignments), 1400)
        self.assertEqual(counts, {event: 175 for event in CHANGE_TYPE_ORDER})
        self.assertEqual(scarce_distractors, 2)

    def test_obstacle_uses_a_fixed_target_support_placement(self):
        case = _build_case(
            _task("Objects Layout", 4, 9), "obstacle_insertion", draw_id=0
        )
        change = case["scenario"]["change"]
        self.assertEqual(
            change["placement_rule"], "target_support_approach_ring"
        )
        self.assertEqual(change["relative_to"], "target")
        self.assertTrue(change["preserve_initial_z"])
        self.assertNotIn("path_target", change)

    def test_obstacle_requires_a_distractor_on_the_target_support(self):
        task = _task("Objects Layout", 4, 9)
        for entity in task["distractor_objects"]:
            task["initial_placements"][entity]["support_entity"] = "shelf"
        self.assertNotIn("obstacle_insertion", eligible_change_types(task))

    def test_physical_rejection_expands_across_equivalent_plus_variants(self):
        left = _task("Light Conditions", 1, 1)
        right = _task("Camera Viewpoints", 5, 2)
        expanded = expand_rejected_by_physical_scene(
            [left, right],
            {("libero_spatial", 1, "obstacle_insertion")},
            ["obstacle_insertion"],
        )
        self.assertIn(
            ("libero_spatial", 2, "obstacle_insertion"), expanded
        )

    def test_full_falls_back_to_observation_after_a_physical_failure(self):
        tasks = [
            _task("Objects Layout", 4, 1),
            _task("Objects Layout", 4, 2),
        ]
        assignments = _full_assignments(
            tasks,
            {},
            {("libero_spatial", 1, "obstacle_insertion")},
        )
        self.assertIn(
            assignments[("libero_spatial", 1)][0], CHANGE_TYPE_ORDER[:4]
        )

    def test_frozen_core_assignments_are_reproduced_exactly(self):
        tasks = []
        index = 0
        for category in PLUS_CATEGORIES:
            for difficulty in PLUS_DIFFICULTIES:
                for _ in range(48):
                    tasks.append(_task(category, difficulty, index))
                    index += 1
        assignments = _core_assignments(tasks)
        tasks_by_key = {
            (task["task_suite_name"], task["task_index"]): task
            for task in tasks
        }
        cases = [
            _build_case(tasks_by_key[key], event, draw)
            for key, (event, draw) in assignments.items()
        ]
        frozen = _manifest("core", cases)
        self.assertEqual(
            _frozen_core_assignments(frozen, tasks), assignments
        )


if __name__ == "__main__":
    unittest.main()
