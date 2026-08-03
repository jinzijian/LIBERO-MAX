import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CosmosLauncherTest(unittest.TestCase):
    def test_config_file_is_an_importable_module_path(self) -> None:
        launcher = (ROOT / "scripts/run_cosmos_paired_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "--config_file cosmos_policy/config/config.py",
            launcher,
        )
        self.assertNotIn(
            '--config_file "$COSMOS_POLICY_DIR/cosmos_policy/config/config.py"',
            launcher,
        )

    def test_preflight_enables_trusted_libero_state_loading_before_imports(self) -> None:
        preflight = (
            ROOT / "scripts/preflight_manifest_interventions.py"
        ).read_text(encoding="utf-8")
        opt_out = (
            'os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")'
        )
        self.assertIn(opt_out, preflight)
        self.assertLess(preflight.index(opt_out), preflight.index("from libero.libero"))
        self.assertLess(
            preflight.index(opt_out),
            preflight.index("from cosmos_policy.experiments.robot.libero"),
        )


if __name__ == "__main__":
    unittest.main()
