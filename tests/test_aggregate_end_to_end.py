import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_cosmos_benchmark.py"
SPEC = importlib.util.spec_from_file_location("aggregate_cosmos_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregateEndToEndTest(unittest.TestCase):
    def test_render_qa_is_an_infrastructure_gate_when_required(self):
        control = {
            "init_state_sha256": "same",
            "query_interval": 16,
            "intervention_event_count": 0,
            "policy_queries": [],
        }
        intervention = dict(control)

        reasons = MODULE._terminal_trace_reasons(
            control,
            intervention,
            {"query_interval": 16},
            require_render_qa=True,
        )

        self.assertEqual(
            reasons,
            [
                "control_render_initialization_qa_missing",
                "intervention_render_initialization_qa_missing",
                "trigger_unreached",
            ],
        )

        passed = {"status": "passed"}
        control["render_initialization_qa"] = passed
        intervention["render_initialization_qa"] = passed
        self.assertEqual(
            MODULE._terminal_trace_reasons(
                control,
                intervention,
                {"query_interval": 16},
                require_render_qa=True,
            ),
            ["trigger_unreached"],
        )

    def test_response_query_unreached_is_terminal_not_infrastructure(self):
        control = {
            "init_state_sha256": "same",
            "query_interval": 16,
            "intervention_event_count": 0,
            "policy_queries": [
                {"policy_step": 0, "action_chunk_sha256": "a"},
                {"policy_step": 16, "action_chunk_sha256": "b"},
            ],
        }
        intervention = {
            **control,
            "intervention_event_count": 1,
            "intervention_events": [{"cosmos_query_boundary_step": 30}],
        }

        reasons = MODULE._terminal_trace_reasons(
            control, intervention, {"query_interval": 16}
        )

        self.assertEqual(reasons, ["response_query_unreached"])

    def test_post_trigger_query_missing_does_not_hide_prechange_mismatch(self):
        control = {
            "init_state_sha256": "same",
            "query_interval": 16,
            "intervention_event_count": 0,
            "policy_queries": [{"policy_step": 0, "action_chunk_sha256": "a"}],
        }
        intervention = {
            **control,
            "intervention_event_count": 1,
            "intervention_events": [{"cosmos_query_boundary_step": 10}],
            "policy_queries": [{"policy_step": 0, "action_chunk_sha256": "z"}],
        }

        reasons = MODULE._terminal_trace_reasons(
            control, intervention, {"query_interval": 16}
        )

        self.assertEqual(
            reasons, ["pre_change_action_mismatch", "response_query_unreached"]
        )

    def test_trigger_unreached_failures_remain_in_planned_denominator(self):
        summary = MODULE.summarize_end_to_end_outcomes(
            3,
            {"triggered": True, "unreached-a": False, "unreached-b": False},
            {"triggered": True, "unreached-a": False, "unreached-b": False},
        )

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["control"]["measured"], 3)
        self.assertEqual(summary["control"]["accuracy_on_planned"], 1 / 3)
        self.assertEqual(summary["intervention"]["accuracy_on_planned"], 1 / 3)
        self.assertEqual(summary["outcome_table"]["persistent_failure"], 2)

    def test_infrastructure_gaps_are_missing_not_model_failures(self):
        summary = MODULE.summarize_end_to_end_outcomes(
            3,
            {"measured-a": True, "measured-b": False},
            {"measured-a": False, "measured-b": False},
        )

        self.assertFalse(summary["complete"])
        self.assertEqual(summary["control"]["missing"], 1)
        self.assertEqual(summary["control"]["accuracy_on_measured"], 0.5)
        self.assertIsNone(summary["control"]["accuracy_on_planned"])
        self.assertIsNone(summary["paired_robustness_delta_on_planned"])

    def test_breakdown_keeps_untriggered_case_in_its_group(self):
        cases = [
            {
                "case_id": "a",
                "scenario": {"change_type": "lighting"},
            },
            {
                "case_id": "b",
                "scenario": {"change_type": "lighting"},
            },
            {
                "case_id": "c",
                "scenario": {"change_type": "camera"},
            },
        ]
        breakdown = MODULE.summarize_end_to_end_breakdown(
            cases,
            {"a": True, "b": False, "c": True},
            {"a": False, "b": False, "c": True},
            "scenario.change_type",
        )

        self.assertEqual(breakdown["lighting"]["paired_measured"], 2)
        self.assertEqual(breakdown["lighting"]["control"]["accuracy_on_planned"], 0.5)
        self.assertEqual(
            breakdown["lighting"]["intervention"]["accuracy_on_planned"], 0.0
        )
        self.assertEqual(
            breakdown["camera"]["intervention"]["accuracy_on_planned"], 1.0
        )

    def test_derived_aggregation_materializes_manifest_and_provenance(self):
        source_manifest = (
            Path(__file__).parents[1]
            / "benchmark/max8000/libero_max_pro_model_comparison_800.json"
        )
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        manifest["cases"] = manifest["cases"][:1]
        case = manifest["cases"][0]
        query_interval = manifest["protocol"]["query_interval"]
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            manifest_path = temp / "manifest.json"
            root = temp / "source"
            output = temp / "derived"
            case_root = root / "cases" / case["case_id"]
            manifest_path.write_text(json.dumps(manifest))
            for arm in ("control", "intervention"):
                (case_root / arm).mkdir(parents=True)
                (case_root / arm / "trace.jsonl").write_text(
                    json.dumps(
                        {
                            "success": False,
                            "init_state_sha256": "same",
                            "query_interval": query_interval,
                            "intervention_event_count": 0,
                            "policy_queries": [],
                        }
                    )
                    + "\n"
                )

            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "manifest.json").is_file())
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["run_type"], "derived_aggregation")
            self.assertEqual(config["manifest"]["planned_cases"], 1)
            self.assertFalse(
                config["aggregation"]["source_runs"][0]["run_config_present"]
            )


if __name__ == "__main__":
    unittest.main()
