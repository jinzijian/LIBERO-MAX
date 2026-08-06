import importlib.util
import hashlib
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "compose_preflight_report.py"
SPEC = importlib.util.spec_from_file_location("compose_preflight_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest_case(case_id, variant="current"):
    return {"case_id": case_id, "scenario": {"variant": variant}}


def _case(case_id, passed=True, variant=None):
    row = {
        "case_id": case_id,
        "scenario_id": case_id,
        "change_type": "camera_shift",
        "passed": passed,
        "validation_errors": [] if passed else ["bad placement"],
    }
    if variant is not None:
        payload = json.dumps(
            {"variant": variant}, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        row["scenario_sha256"] = hashlib.sha256(payload).hexdigest()
    return row


class ComposePreflightReportTest(unittest.TestCase):
    def test_composes_cross_profile_reports_by_case_id(self):
        manifest = {
            "benchmark_id": "libero-max-5600",
            "cases": [_manifest_case("core"), _manifest_case("new")],
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
            "cases": [_manifest_case("repaired")],
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

    def test_current_scenario_hash_overrides_legacy_pass(self):
        manifest = {
            "benchmark_id": "libero-max-5600",
            "cases": [_manifest_case("changed")],
        }
        result = MODULE.compose_preflight(
            manifest,
            [
                {"cases": [_case("changed", True)]},
                {"cases": [_case("changed", False, variant="current")]},
            ],
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["passed"], 0)

    def test_stale_scenario_hash_is_not_reused(self):
        manifest = {
            "benchmark_id": "libero-max-5600",
            "cases": [_manifest_case("changed")],
        }
        result = MODULE.compose_preflight(
            manifest,
            [{"cases": [_case("changed", True, variant="stale")]}],
        )
        self.assertEqual(result["missing"], ["changed"])


if __name__ == "__main__":
    unittest.main()
