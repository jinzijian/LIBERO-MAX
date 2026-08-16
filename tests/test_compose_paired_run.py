import importlib.util
import json
import subprocess
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

    def test_composition_writes_recursive_credential_safe_provenance(self):
        source_manifest = (
            Path(__file__).parents[1]
            / "benchmark/max8000/libero_max_pro_model_comparison_800.json"
        )
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        manifest["cases"] = manifest["cases"][:1]
        case = manifest["cases"][0]
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            manifest_path = temp / "manifest.json"
            source = temp / "source"
            output = temp / "output"
            case_root = source / "cases" / case["case_id"]
            manifest_path.write_text(json.dumps(manifest))
            case_root.mkdir(parents=True)
            (case_root / "scenario.json").write_text(json.dumps(case["scenario"]))
            for arm in ("control", "intervention"):
                (case_root / arm).mkdir()
                (case_root / arm / "trace.jsonl").write_text('{"success": false}\n')
            (source / "run_config.json").write_text(
                json.dumps(
                    {
                        "model": "example",
                        "password": "must-not-ship",
                        "parent": {"schema_version": 1},
                    }
                )
            )

            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(manifest_path),
                    str(source),
                    "--output-root",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["run_type"], "composed_paired_run")
            self.assertEqual(config["composition"]["linked"], 1)
            nested = config["composition"]["source_runs"][0]["run_config"]
            self.assertEqual(nested["model"], "example")
            self.assertEqual(nested["password"], "<redacted>")
            self.assertNotIn("must-not-ship", (output / "run_config.json").read_text())


if __name__ == "__main__":
    unittest.main()
