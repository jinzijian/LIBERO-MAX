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


if __name__ == "__main__":
    unittest.main()
