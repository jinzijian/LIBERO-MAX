import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_dynamic_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_dynamic_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DynamicSchedulerTest(unittest.TestCase):
    def test_auto_uses_every_visible_gpu(self):
        self.assertEqual(
            MODULE.visible_gpus("auto", {"CUDA_VISIBLE_DEVICES": "2,5,7"}),
            ["2", "5", "7"],
        )

    @mock.patch.object(MODULE.subprocess, "run")
    def test_auto_discovers_host_gpus_when_not_constrained(self, run):
        run.return_value = mock.Mock(stdout="0\n1\n2\n")
        self.assertEqual(MODULE.visible_gpus("auto", {}), ["0", "1", "2"])
        run.assert_called_once()

    def test_explicit_gpu_list_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.visible_gpus("0,1,0")

    def test_suite_units_are_disjoint_and_cover_arbitrary_suites(self):
        manifest = {
            "cases": [
                {"case_id": "a0", "task_suite_name": "libero_10"},
                {"case_id": "a1", "task_suite_name": "libero_10"},
                {"case_id": "b0", "task_suite_name": "custom_suite"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            units = MODULE.materialize_work_units(
                manifest, Path(directory), shards_per_suite=4
            )
            self.assertEqual(len(units), 3)
            self.assertEqual(units[0].suite, "libero_10")
            self.assertEqual(
                {unit.suite for unit in units}, {"libero_10", "custom_suite"}
            )
            self.assertEqual(
                {(unit.suite, unit.shard_index, unit.num_shards) for unit in units},
                {
                    ("libero_10", 0, 2),
                    ("libero_10", 1, 2),
                    ("custom_suite", 0, 1),
                },
            )

    def test_scheduler_owned_arguments_cannot_be_overridden(self):
        unit = MODULE.WorkUnit("suite", Path("suite.json"), 0, 1)
        with self.assertRaisesRegex(ValueError, "must not override"):
            MODULE.build_runner_command(
                "python", Path("runner.py"), unit, Path("output"), ["--resume"]
            )

    def test_model_arguments_are_forwarded(self):
        unit = MODULE.WorkUnit("suite", Path("suite.json"), 1, 3)
        command = MODULE.build_runner_command(
            "python",
            Path("runner.py"),
            unit,
            Path("output"),
            ["--checkpoint", "checkpoint"],
        )
        self.assertEqual(command[-2:], ["--checkpoint", "checkpoint"])
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--shard-index") + 1], "1")
        self.assertEqual(command[command.index("--num-shards") + 1], "3")


if __name__ == "__main__":
    unittest.main()
