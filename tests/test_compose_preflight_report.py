import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compose_preflight_report.py"
SPEC = importlib.util.spec_from_file_location("compose_preflight_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _case(case_id, passed=True):
    return {
        "case_id": case_id,
        "scenario_id": case_id,
        "change_type": "camera_shift",
        "passed": passed,
        "validation_errors": [] if passed else ["bad placement"],
    }


class ComposePreflightReportTest(unittest.TestCase):
    def test_composes_cross_profile_reports_by_case_id(self):
        manifest = {
            "benchmark_id": "libero-max-5600",
            "cases": [{"case_id": "core"}, {"case_id": "new"}],
        }
        result = MODULE.compose_preflight(
            manifest,
            [
                {"benchmark_id": "old-core", "cases": [_case("core")]},
                {"benchmark_id": "delta", "cases": [_case("new")]},
            ],
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["planned"], 2)
        self.assertEqual(result["passed"], 2)

    def test_later_pass_replaces_an_earlier_failure(self):
        manifest = {
            "benchmark_id": "libero-max-5600",
            "cases": [{"case_id": "repaired"}],
        }
        result = MODULE.compose_preflight(
            manifest,
            [
                {"cases": [_case("repaired", False)]},
                {"cases": [_case("repaired", True)]},
            ],
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["failures"], {})


if __name__ == "__main__":
    unittest.main()
