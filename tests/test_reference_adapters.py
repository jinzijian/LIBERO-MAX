import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReferenceAdapterContractTest(unittest.TestCase):
    def read(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_xvla_records_provenance_and_native_cadence(self):
        launcher = self.read("scripts/run_xvla_persistent_benchmark.py")
        self.assertIn('default=30', launcher)
        for field in (
            '"source_revision"',
            '"checkpoint_bytes"',
            '"runtime_versions"',
            '"control_replay_before_event"',
        ):
            self.assertIn(field, launcher)

    def test_xvla_shard_uses_absolute_control_after_warmup(self):
        shard = self.read("scripts/run_xvla_persistent_shard.py")
        self.assertIn('default=30', shard)
        self.assertIn("robot.controller.use_delta = False", shard)
        self.assertIn("total_step == wrapped.warmup_steps", shard)

    def test_aggregator_can_validate_native_query_cadence(self):
        aggregator = self.read("scripts/aggregate_cosmos_benchmark.py")
        self.assertIn('"--query-interval"', aggregator)
        self.assertIn("manifest[\"protocol\"][\"query_interval\"]", aggregator)
        self.assertIn("--query-interval must be positive", aggregator)

    def test_source_overlay_requires_an_explicit_checkout(self):
        overlay = self.read("scripts/libero_source_overlay/libero/__init__.py")
        self.assertIn("LIBERO_SOURCE_PACKAGE_ROOT", overlay)
        self.assertNotIn("/" + "vepfs/", overlay)
        self.assertNotIn("/" + "Users/", overlay)


class PublicRepositoryPrivacyTest(unittest.TestCase):
    def test_public_files_contain_no_machine_or_secret_markers(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT
        ).decode("utf-8").split("\0")
        candidates = {
            path for path in tracked if path and (ROOT / path).is_file()
        }
        candidates.update(
            {
                "assets/figures/main_results.json",
                "scripts/libero_source_overlay/libero/__init__.py",
                "scripts/run_dynamic_benchmark.py",
                "scripts/run_xvla_persistent_benchmark.py",
                "scripts/run_xvla_persistent_shard.py",
                "tests/test_dynamic_scheduler.py",
                "tests/test_reference_adapters.py",
            }
        )
        literals = (
            "/" + "Users/",
            "/" + "vepfs/",
            "/opt/" + "dlami/",
            "Identity" + "File",
            "BEGIN PRIVATE " + "KEY",
            "HF_" + "TOKEN",
            "WANDB_API_" + "KEY",
        )
        forbidden = re.compile(
            "|".join(re.escape(value) for value in literals)
            + r"|ec2-[0-9]|115\.190\."
        )
        violations = []
        for relative_path in sorted(candidates):
            path = ROOT / relative_path
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if forbidden.search(content):
                violations.append(relative_path)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
