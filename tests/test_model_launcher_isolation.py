import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ModelLauncherIsolationTest(unittest.TestCase):
    def test_model_venv_precedes_shared_libero_site_packages(self):
        for name in ("run_max_pro_fastwam.sh", "run_max_pro_lingbot.sh"):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("ACTIVE_SITE_PACKAGES", source)
            active = source.index('export PYTHONPATH="$ACTIVE_SITE_PACKAGES:')
            shared = source.index("$DEPS_DIR/.venv-libero/lib/python3.10/site-packages")
            self.assertLess(active, shared)

    def test_openpi_pro_launcher_pins_pro_runtime_paths(self):
        source = (ROOT / "scripts" / "run_max_pro_openpi.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("LIBERO-PRO", source)
        self.assertIn("libero-pro-python-overlay", source)
        self.assertIn("robosuite-1.4.0", source)


if __name__ == "__main__":
    unittest.main()
