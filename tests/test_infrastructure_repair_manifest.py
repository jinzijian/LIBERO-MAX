import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "build_infrastructure_repair_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_infrastructure_repair_manifest", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InfrastructureRepairManifestTest(unittest.TestCase):
    def test_only_blocking_cases_are_selected(self):
        summary = {
            "coverage": {
                "missing": ["missing"],
                "invalid": {"invalid": ["bad trace"]},
                "blocking_terminal_invalid": {"blocked": ["state mismatch"]},
                "terminal_invalid": {
                    "unreached": ["trigger_unreached"],
                    "blocked": ["state mismatch"],
                },
            }
        }
        self.assertEqual(
            MODULE.repair_case_ids(summary), ["blocked", "invalid", "missing"]
        )


if __name__ == "__main__":
    unittest.main()
