import unittest
from collections import Counter

from libero_max.manifest import validate_manifest
from libero_max.pro_hard import (
    PRO_CATEGORIES,
    PRO_HARD_PAIRS,
    build_pro_hard_manifest,
    pro_hard_summary,
)
from libero_max.release import audit_pro_hard_release


REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"


def _task(category, suite, index, rich_distractors=True):
    distractor_count = 6 if rich_distractors else 4
    distractors = ["d%d" % value for value in range(distractor_count)]
    return {
        "task_suite_name": suite,
        "task_index": index,
        "task_name": "task_%d" % index,
        "language": "move the target into the basket",
        "pro_category": category,
        "source_category": category,
        "source_revision": REVISION,
        "bddl_file": "%s/%s/task_%d.bddl" % (category, suite, index),
        "init_states_file": "%s/task_%d.pruned_init" % (suite, index),
        "init_reference_bddl_file": "semantic/%s/task_%d.bddl" % (
            suite,
            index,
        ),
        "trigger_entity": "target",
        "primary_target": "target",
        "primary_receptacle": "basket",
        "supports_target_relocation": True,
        "supports_receptacle_relocation": True,
        "distractor_objects": distractors,
        "initial_placements": {
            "target": {"support_entity": "table"},
            "basket": {"support_entity": "table"},
            **{entity: {"support_entity": "table"} for entity in distractors},
        },
    }


def _catalog():
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    tasks = []
    for category in PRO_CATEGORIES:
        for suite in suites:
            for index in range(10):
                # Match the upstream capacity: only twelve of forty tasks can
                # support a five-object burst. The builder must use multiple
                # frozen init states rather than silently shrinking the cell.
                rich = suite == "libero_object" or (suite == "libero_10" and index < 2)
                tasks.append(_task(category, suite, index, rich))
    return {"source_revision": REVISION, "tasks": tasks}


class ProHardManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_pro_hard_manifest(_catalog())

    def test_builds_exact_max8000_extension_size(self):
        self.assertEqual(len(self.manifest["cases"]), PRO_HARD_PAIRS)
        self.assertEqual(PRO_HARD_PAIRS, 2400)
        self.assertEqual(validate_manifest(self.manifest), [])

    def test_every_category_event_draw_cell_has_fifteen_pairs(self):
        counts = Counter(
            (
                case["substrate_variant"]["category"],
                case["scenario"]["change_type"],
                case["scenario"]["randomization"]["draw_id"],
            )
            for case in self.manifest["cases"]
        )
        self.assertEqual(len(counts), len(PRO_CATEGORIES) * 8 * 2)
        self.assertEqual(set(counts.values()), {15})

    def test_scarce_distractor_cells_use_distinct_task_init_pairs(self):
        cases = [
            case
            for case in self.manifest["cases"]
            if case["substrate_variant"]["category"] == "semantic"
            and case["scenario"]["change_type"] == "distractor_burst"
            and case["scenario"]["randomization"]["draw_id"] == 0
        ]
        task_init = {
            (
                case["task_suite_name"],
                case["task_index"],
                case["init_state_index"],
            )
            for case in cases
        }
        self.assertEqual(len(task_init), 15)
        self.assertEqual(
            len({(case["task_suite_name"], case["task_index"]) for case in cases}),
            12,
        )

    def test_case_identity_and_source_paths_are_explicit(self):
        case_ids = [case["case_id"] for case in self.manifest["cases"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        case = self.manifest["cases"][0]
        self.assertEqual(case["substrate_variant"]["benchmark"], "LIBERO-PRO")
        self.assertFalse(case["substrate_variant"]["bddl_file"].startswith("/"))
        self.assertFalse(
            case["substrate_variant"]["init_reference_bddl_file"].startswith("/")
        )
        self.assertEqual(case["substrate_variant"]["source_revision"], REVISION)

    def test_summary_marks_candidate_until_mujoco_preflight(self):
        summary = pro_hard_summary(self.manifest)
        self.assertEqual(summary["status"], "candidate")
        self.assertEqual(summary["pairs_per_joint_cell"], [15])

    def test_failed_candidate_can_be_replaced_without_changing_quotas(self):
        failed = self.manifest["cases"][0]
        rejection = {
            (
                failed["substrate_variant"]["category"],
                failed["task_suite_name"],
                failed["task_index"],
                failed["init_state_index"],
                failed["scenario"]["change_type"],
                failed["scenario"]["randomization"]["draw_id"],
            )
        }
        repaired = build_pro_hard_manifest(_catalog(), rejection)
        self.assertEqual(len(repaired["cases"]), 2400)
        self.assertNotIn(
            failed["case_id"], {case["case_id"] for case in repaired["cases"]}
        )

    def test_release_audit_requires_complete_exact_preflight(self):
        preflight = {
            "benchmark_id": self.manifest["benchmark_id"],
            "complete": True,
            "planned": 2400,
            "passed": 2400,
            "cases": [
                {
                    "case_id": case["case_id"],
                    "scenario_id": case["scenario"]["scenario_id"],
                    "passed": True,
                }
                for case in self.manifest["cases"]
            ],
        }
        self.assertEqual(
            audit_pro_hard_release(_catalog(), self.manifest, preflight), []
        )
        preflight["cases"].pop()
        self.assertTrue(
            any(
                "coverage does not exactly match" in error
                for error in audit_pro_hard_release(
                    _catalog(), self.manifest, preflight
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
