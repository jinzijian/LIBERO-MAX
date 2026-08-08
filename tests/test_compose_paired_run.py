import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compose_paired_run.py"
SPEC = importlib.util.spec_from_file_location("compose_paired_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComposePairedRunTest(unittest.TestCase):
    def test_empty_terminal_trace_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            scenario = {"scenario_id": "s", "seed": 1}
            (root / "scenario.json").write_text(json.dumps(scenario))
            for arm in ("control", "intervention"):
                (root / arm).mkdir()
                (root / arm / "trace.jsonl").write_text("")
            self.assertFalse(MODULE._is_terminal_case(root, scenario))

    def test_one_row_per_arm_is_terminal(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            scenario = {"scenario_id": "s", "seed": 1}
            (root / "scenario.json").write_text(json.dumps(scenario))
            for arm in ("control", "intervention"):
                (root / arm).mkdir()
                (root / arm / "trace.jsonl").write_text('{"success": false}\n')
            self.assertTrue(MODULE._is_terminal_case(root, scenario))


if __name__ == "__main__":
    unittest.main()
